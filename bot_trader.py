"""
Modul de executie si conectare la Alpaca Paper Trading API.
Functionalitati Institutionale:
  - Executie Split-Order (Scale-Out): Imparte pozitia in 2 loturi Bracket:
      * Lot 1 (50% actiuni): Take-Profit la TP1 (1.5R - 2.0R) + Stop-Loss initial.
      * Lot 2 (50% actiuni - 'Runner'): Take-Profit la TP2 (3.0R) + Stop-Loss urmarit dinamic (Trailing Stop).
  - Verificare Pre-Trade de Portfolio Heat: Numara doar pozitiile cu RISC ACTIV
    (cele ajunse la Breakeven sunt 'Free-Roll' cu risc 0%).
  - Inregistrare automata a metadatelor pozitiei (setup_type, initial_risk, tp1, tp2) in positions_state.json.
"""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import config

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
POSITIONS_STATE_FILE = DATA_DIR / "positions_state.json"
ACCOUNT_META_FILE = DATA_DIR / "account_meta.json"

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        TakeProfitRequest,
        StopLossRequest,
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


class AlpacaTrader:
    def __init__(self):
        self.client: Optional[TradingClient] = None
        self.is_connected = False

        if config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY and ALPACA_AVAILABLE:
            try:
                self.client = TradingClient(
                    api_key=config.ALPACA_API_KEY,
                    secret_key=config.ALPACA_SECRET_KEY,
                    paper=config.ALPACA_PAPER
                )
                account = self.client.get_account()
                self.is_connected = True
                summary = self.get_portfolio_summary()
                print(" ✅ Conectat cu succes la Alpaca Paper Trading!")
                print(
                    f"  Cont: {account.account_number} | Buget Activ Bot: ${summary['portfolio_value']:,.2f} "
                    f"(Cash: ${summary['cash']:,.2f})"
                )
            except Exception as e:
                print(f"⚠️ Nu s-a putut conecta la Alpaca: {e}")
                self.is_connected = False
        else:
            print("ℹ️ Cheile Alpaca API lipsesc sau sunt incomplete. Modulul ruleaza in mod SIMULAT.")

    def get_portfolio_summary(self) -> Dict:
        """
        Returneaza rezumatul capitalului si al puterii de cumparare.
        Daca ACCOUNT_BUDGET_CAP > 0 (ex: $1,000), calibreaza automat capitalul si cash-ul
        la bugetul de $1,000 + PnL-ul real acumulat pe contul Alpaca.
        """
        if not self.is_connected:
            return {
                "connected": False,
                "portfolio_value": config.DEFAULT_SIMULATED_CAPITAL,
                "cash": config.DEFAULT_SIMULATED_CAPITAL,
                "buying_power": config.DEFAULT_SIMULATED_CAPITAL
            }

        account = self.client.get_account()
        raw_equity = float(account.portfolio_value)
        raw_cash = float(account.cash)
        long_mv = abs(float(getattr(account, "long_market_value", 0.0)))
        budget_cap = float(getattr(config, "ACCOUNT_BUDGET_CAP", 0.0))

        if budget_cap > 0 and raw_equity > budget_cap * 2.5:
            meta = {}
            if ACCOUNT_META_FILE.exists():
                try:
                    meta = json.loads(ACCOUNT_META_FILE.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}

            if meta.get("account_number") != account.account_number or "baseline_equity" not in meta:
                meta = {
                    "account_number": account.account_number,
                    "baseline_equity": raw_equity,
                    "starting_budget": budget_cap,
                    "initialized_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                try:
                    ACCOUNT_META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                except Exception:
                    pass

            net_pnl = raw_equity - float(meta.get("baseline_equity", raw_equity))
            eff_equity = round(max(100.0, budget_cap + net_pnl), 2)
            eff_cash = round(max(0.0, eff_equity - long_mv), 2)
            return {
                "connected": True,
                "portfolio_value": eff_equity,
                "cash": eff_cash,
                "buying_power": eff_cash,
                "raw_alpaca_equity": raw_equity,
                "net_pnl": round(net_pnl, 2),
                "status": account.status
            }

        return {
            "connected": True,
            "portfolio_value": raw_equity,
            "cash": raw_cash,
            "buying_power": float(account.buying_power),
            "raw_alpaca_equity": raw_equity,
            "status": account.status
        }

    def get_open_positions(self) -> List[Dict]:
        """Returneaza pozitiile deschise in prezent."""
        if not self.is_connected:
            return []

        positions = self.client.get_all_positions()
        result = []
        for p in positions:
            result.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc) * 100
            })
        return result

    def get_active_risk_positions_count(self) -> int:
        """
        Returneaza numarul de pozitii deschise care au inca RISC ACTIV
        (nu au ajuns inca la Breakeven / Free-Roll).
        """
        positions = self.get_open_positions()
        if not positions:
            return 0

        state = {}
        if POSITIONS_STATE_FILE.exists():
            try:
                with open(POSITIONS_STATE_FILE, "r", encoding="utf-8") as f:
                    state = json.load(f)
            except Exception:
                state = {}

        active_risk = 0
        for p in positions:
            sym = p["symbol"]
            st = state.get(sym, {})
            if not st.get("breakeven_active", False):
                active_risk += 1
        return active_risk

    def _register_position_state(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        tp1_price: float,
        tp2_price: float,
        setup_type: str,
        qty: float,
        fractional_managed: bool = False
    ):
        """Salveaza metadatele pozitiei in positions_state.json pentru PositionManager si RiskLearner."""
        state = {}
        if POSITIONS_STATE_FILE.exists():
            try:
                with open(POSITIONS_STATE_FILE, "r", encoding="utf-8") as f:
                    state = json.load(f)
            except Exception:
                state = {}

        state[symbol] = {
            "setup_type": setup_type,
            "entry_price": round(entry_price, 2),
            "initial_stop": round(stop_loss_price, 2),
            "initial_risk": round(max(0.01, entry_price - stop_loss_price), 2),
            "tp1": round(tp1_price, 2),
            "tp2": round(tp2_price, 2),
            "initial_qty": round(float(qty), 4),
            "peak_price": round(entry_price, 2),
            "tp1_hit": False,
            "breakeven_active": False,
            "fractional_managed": fractional_managed,
            "last_stop_price": round(stop_loss_price, 2),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            with open(POSITIONS_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4, ensure_ascii=False)
        except Exception:
            pass

    def place_bracket_buy_order(
        self,
        symbol: str,
        qty: float,
        limit_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        tp2_price: Optional[float] = None,
        setup_type: str = "PULLBACK_EMA20"
    ):
        """
        Plaseaza ordin(e) pe Alpaca cu suport complet pentru ACTIUNI FRACTIONARE (ex: 0.28x LLY, 1.75x NVDA)
        cat si pentru actiuni intregi:
        - Daca cantitatea este fractionara (sau < 2 actiuni), trimite ordinul Fractional DAY pe Alpaca si
          activeaza gestionarea automata de catre PositionManager pentru SL, TP1 (50%), TP2 (50%),
          Auto-Breakeven si Trailing Stop.
        - Daca cantitatea este intreaga (>= 2 actiuni), foloseste Split-Bracket nativ pe server.
        """
        exec_qty = round(float(qty), 4)
        if exec_qty <= 0:
            print(f"⚠️ {symbol} omis: Cantitate calculată invalidă ({exec_qty}).")
            return None

        risk_per_share = max(0.01, limit_price - stop_loss_price)
        if tp2_price is None or tp2_price <= take_profit_price:
            tp2_rr = getattr(config, "TP2_RISK_REWARD_RATIO", 3.0)
            tp2_price = round(limit_price + (tp2_rr * risk_per_share), 2)

        if not self.is_connected:
            q1 = round(exec_qty * 0.5, 4)
            q2 = round(exec_qty - q1, 4)
            print(
                f"[SIMULARE SPLIT-ORDER FRACȚIONAR] {symbol} ({exec_qty} acțiuni @ ${limit_price:.2f} | Valoare: ${exec_qty*limit_price:.2f} | SL: ${stop_loss_price:.2f}):\n"
                f"   ├─ Lot 1 ({q1}x): Take-Profit 1 @ ${take_profit_price:.2f}\n"
                f"   └─ Lot 2 ({q2}x): Runner TP2 @ ${tp2_price:.2f} + Trailing Stop"
            )
            self._register_position_state(symbol, limit_price, stop_loss_price, take_profit_price, tp2_price, setup_type, exec_qty, fractional_managed=True)
            return None

        # Verificare Cash Disponibil in bugetul de $1,000
        summary = self.get_portfolio_summary()
        avail_cash = summary.get("cash", 0.0)
        order_cost = exec_qty * limit_price
        if order_cost > avail_cash:
            if avail_cash >= 10.0:
                exec_qty = round((avail_cash * 0.99) / limit_price, 4)
                order_cost = exec_qty * limit_price
            else:
                print(
                    f"🛡️ [BUGET CASH LIMIT] {symbol} refuzat: Cash disponibil insuficient (${avail_cash:.2f})."
                )
                return None

        # Verificare Portfolio Heat inainte de trimiterea ordinului
        active_risk_count = self.get_active_risk_positions_count()
        if active_risk_count >= config.MAX_OPEN_POSITIONS:
            print(
                f"🛡️ [PORTFOLIO HEAT LIMIT] {symbol} refuzat: Exista deja {active_risk_count}/{config.MAX_OPEN_POSITIONS} "
                f"pozitii cu risc activ. Asteptam ca o pozitie existenta sa treaca la Breakeven (Free-Roll)."
            )
            return None

        is_fractional = (exec_qty != int(exec_qty)) or (exec_qty < 2)

        try:
            if not is_fractional and getattr(config, "USE_SPLIT_ORDERS", True) and int(exec_qty) >= 2:
                whole_qty = int(exec_qty)
                qty_tp1 = whole_qty // 2
                qty_tp2 = whole_qty - qty_tp1

                req_lot1 = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty_tp1,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                    limit_price=round(limit_price, 2),
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 2)),
                    stop_loss=StopLossRequest(stop_price=round(stop_loss_price, 2))
                )
                order_lot1 = self.client.submit_order(order_data=req_lot1)

                req_lot2 = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty_tp2,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                    limit_price=round(limit_price, 2),
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=round(tp2_price, 2)),
                    stop_loss=StopLossRequest(stop_price=round(stop_loss_price, 2))
                )
                order_lot2 = self.client.submit_order(order_data=req_lot2)

                print(
                    f" ✅ Ordine Split-Bracket plasate pe Alpaca pentru {symbol} ({whole_qty} acțiuni):\n"
                    f"    • Lot 1 ({qty_tp1}x TP1 @ ${take_profit_price:.2f}) -> ID: {order_lot1.id}\n"
                    f"    • Lot 2 ({qty_tp2}x TP2 @ ${tp2_price:.2f}) -> ID: {order_lot2.id}"
                )
                primary_order = order_lot1
                self._register_position_state(
                    symbol, limit_price, stop_loss_price, take_profit_price, tp2_price, setup_type, whole_qty, fractional_managed=False
                )
            else:
                # Execuție Fracționară (sau < 2 acțiuni): Alpaca cere ordin simplu DAY pentru fracții,
                # iar SL, TP1 (50%), TP2 (50%), Breakeven și Trailing Stop sunt gestionate automat de PositionManager!
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=exec_qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                primary_order = self.client.submit_order(order_data=req)
                q1 = round(exec_qty * 0.5, 4)
                q2 = round(exec_qty - q1, 4)
                print(
                    f" ✅ Ordin Fracționar plasat pe Alpaca pentru {symbol}: {exec_qty} acțiuni (~${order_cost:.2f}) -> ID: {primary_order.id}\n"
                    f"    🛡️ Protecție Activă Bot: SL @ ${stop_loss_price:.2f} | Lot 1 ({q1}x) TP1 @ ${take_profit_price:.2f} | Lot 2 ({q2}x) TP2 @ ${tp2_price:.2f}"
                )
                self._register_position_state(
                    symbol, limit_price, stop_loss_price, take_profit_price, tp2_price, setup_type, exec_qty, fractional_managed=True
                )

            try:
                from notifier import send_order_placed_alert
                send_order_placed_alert(
                    symbol=symbol,
                    qty=exec_qty,
                    limit_price=limit_price,
                    sl=stop_loss_price,
                    tp=take_profit_price,
                    order_id=str(primary_order.id),
                    tp2=tp2_price
                )
            except Exception:
                pass

            return primary_order
        except Exception as e:
            print(f"❌ Eroare la plasarea ordinului pe Alpaca: {e}")
            return None
