"""
Modul de execuție rezilientă și conectare la Alpaca Paper / Live Trading API (Institutional V2).
Respectă toate regulile de siguranță la execuție și control al riscului:
  - Validare Fail-Fast la pornire a variabilelor de mediu (`paper` vs `live` + chei API).
  - Reziliență API: Timeout explicit + Retry cu Exponential Backoff pe toate apelurile.
  - Idempotență la plasarea ordinelor: Verifică pozițiile deschise ȘI ordinele `open`/`pending`,
    folosește `client_order_id` determinist și verifică starea ordinului înainte de retry la POST.
  - Kill-Switch Automat: Blochează intrările noi dacă pierderea zilnică (`MAX_DAILY_LOSS_PCT`) sau
    drawdown-ul față de High-Water Mark (`MAX_PORTFOLIO_DRAWDOWN_PCT`) depășește pragul configurat.
  - Protecție Anti-Fragmentare, Sector Exposure Cap și ordine `Limit` (fără slippage de ordin Market).
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Callable, Any
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
        GetOrdersRequest,
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


def _configure_client_timeout(client: Any, timeout_sec: float):
    """Setează un timeout explicit pe sesiunea HTTP internă a clientului Alpaca."""
    session = getattr(client, "_session", None)
    if session is not None and not getattr(session, "_timeout_patched", False):
        orig_request = session.request

        def _request_with_timeout(method, url, **kwargs):
            kwargs.setdefault("timeout", timeout_sec)
            return orig_request(method, url, **kwargs)

        session.request = _request_with_timeout
        session._timeout_patched = True


class AlpacaTrader:
    def __init__(self):
        # 1. Validare Fail-Fast a modului de rulare ('paper' vs 'live') și a cheilor API
        env_info = config.validate_runtime_environment(require_broker=False)
        self.execution_mode = env_info["mode"]
        self.client: Optional[TradingClient] = None
        self.is_connected = False

        if env_info["has_keys"] and ALPACA_AVAILABLE:
            try:
                self.client = TradingClient(
                    api_key=config.ALPACA_API_KEY,
                    secret_key=config.ALPACA_SECRET_KEY,
                    paper=config.ALPACA_PAPER
                )
                _configure_client_timeout(
                    self.client, float(getattr(config, "API_TIMEOUT_SECONDS", 10.0))
                )
                account = self._api_call(self.client.get_account)
                self.is_connected = True
                summary = self.get_portfolio_summary()
                mode_lbl = "Paper Trading" if config.ALPACA_PAPER else "LIVE TRADING"
                print(f" ✅ Conectat cu succes la Alpaca ({mode_lbl})!")
                print(
                    f"  Cont: {account.account_number} | Buget Activ Bot: ${summary['portfolio_value']:,.2f} "
                    f"(Cash: ${summary['cash']:,.2f} | HWM: ${summary.get('high_water_mark', summary['portfolio_value']):,.2f})"
                )
            except Exception as e:
                if not config.ALPACA_PAPER:
                    raise RuntimeError(f"Eșec critic la conectarea în modul LIVE Alpaca: {e}") from e
                print(f"⚠️ Nu s-a putut conecta la Alpaca: {e}")
                self.is_connected = False
        else:
            print("ℹ️ Cheile Alpaca API lipsesc sau sunt incomplete. Modulul rulează în mod SIMULAT.")

    def _api_call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Împachetează apelurile de citire/administrare către API-ul brokerului în retry
        cu exponential backoff pentru erori tranzitorii de rețea, timeout, 429 sau 5xx.
        """
        max_retries = int(getattr(config, "API_MAX_RETRIES", 3))
        base_delay = float(getattr(config, "API_BACKOFF_BASE_SECONDS", 1.0))
        last_exc = None

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                transient = any(
                    tok in msg for tok in ("timeout", "timed out", "connection", "429", "rate limit", "500", "502", "503", "504", "reset")
                )
                if transient and attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                raise
        raise last_exc

    def _submit_order_idempotent(self, order_req: Any, client_order_id: str) -> Any:
        """
        Trimite un ordin POST în mod idempotent:
          - Atașează `client_order_id` determinist.
          - În caz de timeout/eroare de rețea incertă, interoghează starea ordinului pe server
            înainte de a reîncerca, prevenind dublarea ordinelor.
        """
        order_req.client_order_id = client_order_id
        max_retries = int(getattr(config, "API_MAX_RETRIES", 3))
        base_delay = float(getattr(config, "API_BACKOFF_BASE_SECONDS", 1.0))
        last_exc = None

        # Verificăm preventiv dacă acest client_order_id a fost deja trimis anterior
        try:
            existing = self.client.get_order_by_client_id(client_order_id)
            if existing is not None:
                st = getattr(existing.status, "name", str(existing.status)).lower()
                if st not in ("canceled", "expired", "rejected"):
                    print(f"ℹ️ [IDEMPOTENȚĂ] Ordinul {client_order_id} există deja pe Alpaca (status: {st}).")
                    return existing
        except Exception:
            pass

        for attempt in range(max_retries):
            try:
                return self.client.submit_order(order_data=order_req)
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                if "client_order_id" in msg and ("unique" in msg or "already" in msg or "exists" in msg):
                    return self.client.get_order_by_client_id(client_order_id)

                transient = any(
                    tok in msg for tok in ("timeout", "timed out", "connection", "429", "rate limit", "500", "502", "503", "504")
                )
                if transient:
                    # Verificăm dacă ordinul a ajuns totuși la broker înainte de retry!
                    try:
                        existing = self.client.get_order_by_client_id(client_order_id)
                        if existing is not None:
                            return existing
                    except Exception:
                        pass
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (2 ** attempt))
                        continue
                raise
        raise last_exc

    def _load_account_meta(self) -> Dict:
        if ACCOUNT_META_FILE.exists():
            try:
                return json.loads(ACCOUNT_META_FILE.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_account_meta(self, meta: Dict):
        try:
            ACCOUNT_META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass

    def get_portfolio_summary(self) -> Dict:
        """
        Returnează rezumatul capitalului, P&L-ul zilei și High-Water Mark-ul pentru Kill-Switch.
        """
        if not self.is_connected or not self.client:
            cap = float(config.DEFAULT_SIMULATED_CAPITAL)
            return {
                "connected": False,
                "portfolio_value": cap,
                "cash": cap,
                "buying_power": cap,
                "high_water_mark": cap,
                "daily_loss_pct": 0.0,
                "drawdown_pct": 0.0,
            }

        account = self._api_call(self.client.get_account)
        raw_equity = float(account.portfolio_value)
        last_equity = float(getattr(account, "last_equity", raw_equity) or raw_equity)
        raw_cash = float(account.cash)
        long_mv = abs(float(getattr(account, "long_market_value", 0.0)))
        budget_cap = float(getattr(config, "ACCOUNT_BUDGET_CAP", 0.0))

        meta = self._load_account_meta()

        if budget_cap > 0 and raw_equity > budget_cap * 2.5:
            if meta.get("account_number") != account.account_number or "baseline_equity" not in meta:
                meta = {
                    "account_number": account.account_number,
                    "baseline_equity": raw_equity,
                    "starting_budget": budget_cap,
                    "high_water_mark": budget_cap,
                    "initialized_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            net_pnl = raw_equity - float(meta.get("baseline_equity", raw_equity))
            eff_equity = round(max(100.0, budget_cap + net_pnl), 2)
            eff_cash = round(max(0.0, eff_equity - long_mv), 2)
            eff_buying_power = eff_cash
            daily_pnl = raw_equity - last_equity
            base_for_daily = max(100.0, eff_equity - daily_pnl)
            daily_change_pct = (daily_pnl / base_for_daily) * 100.0
        else:
            eff_equity = round(raw_equity, 2)
            eff_cash = round(raw_cash, 2)
            eff_buying_power = round(float(account.buying_power), 2)
            daily_change_pct = ((raw_equity - last_equity) / last_equity * 100.0) if last_equity > 0 else 0.0

        prev_hwm = float(meta.get("high_water_mark", eff_equity))
        if meta.get("account_number") != account.account_number:
            prev_hwm = eff_equity
            meta["account_number"] = account.account_number

        hwm = round(max(prev_hwm, eff_equity), 2)
        if hwm != prev_hwm or "high_water_mark" not in meta:
            meta["high_water_mark"] = hwm
            meta["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._save_account_meta(meta)

        drawdown_pct = round(((hwm - eff_equity) / hwm) * 100.0, 2) if hwm > 0 else 0.0

        return {
            "connected": True,
            "portfolio_value": eff_equity,
            "cash": eff_cash,
            "buying_power": eff_buying_power,
            "raw_alpaca_equity": raw_equity,
            "high_water_mark": hwm,
            "daily_change_pct": round(daily_change_pct, 2),
            "drawdown_pct": max(0.0, drawdown_pct),
            "status": account.status,
        }

    def check_kill_switch(self) -> Tuple[bool, str]:
        """
        Evaluează Kill-Switch-ul automat înainte de deschiderea oricărei poziții noi:
          - Blochează intrările dacă pierderea zilnică depășește `MAX_DAILY_LOSS_PCT`.
          - Blochează intrările dacă drawdown-ul față de High-Water Mark depășește `MAX_PORTFOLIO_DRAWDOWN_PCT`.
        """
        if not getattr(config, "KILL_SWITCH_ENABLED", True) or not self.is_connected:
            return False, ""

        summary = self.get_portfolio_summary()
        daily_chg = float(summary.get("daily_change_pct", 0.0))
        dd_pct = float(summary.get("drawdown_pct", 0.0))

        max_daily_loss = float(getattr(config, "MAX_DAILY_LOSS_PCT", 0.03)) * 100.0
        max_dd = float(getattr(config, "MAX_PORTFOLIO_DRAWDOWN_PCT", 0.10)) * 100.0

        if daily_chg <= -max_daily_loss:
            return True, (
                f"🚨 KILL-SWITCH ZILNIC ACTIV: Pierderea zilei ({daily_chg:+.2f}%) a atins pragul maxim "
                f"admis (-{max_daily_loss:.1f}%). Intrările noi sunt oprite pentru protejarea capitalului!"
            )

        if dd_pct >= max_dd:
            return True, (
                f"🚨 KILL-SWITCH DRAWDOWN ACTIV: Scăderea față de High-Water Mark (-{dd_pct:.2f}% de la "
                f"${summary['high_water_mark']:,.2f}) a atins pragul maxim (-{max_dd:.1f}%). Intrările noi sunt blocate!"
            )

        return False, ""

    def get_open_positions(self) -> List[Dict]:
        """Returnează pozițiile deschise în prezent pe Alpaca."""
        if not self.is_connected or not self.client:
            return []

        positions = self._api_call(self.client.get_all_positions)
        result = []
        for p in positions:
            result.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc) * 100.0
            })
        return result

    def get_open_buy_orders(self) -> List[Any]:
        """Returnează toate ordinele de CUMPĂRARE active / în așteptare (`open` / `pending`)."""
        if not self.is_connected or not self.client:
            return []
        try:
            orders = self._api_call(
                self.client.get_orders,
                GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100)
            )
            buy_orders = []
            for o in orders:
                side = getattr(o.side, "name", str(o.side)).lower()
                if side == "buy":
                    buy_orders.append(o)
            return buy_orders
        except Exception:
            return []

    def get_active_risk_positions_count(self) -> int:
        """
        Returnează numărul total de poziții cu RISC ACTIV (poziții deschise care nu au trecut
        încă de TP1 la Breakeven + simboluri unice cu ordine BUY în așteptare).
        """
        positions = self.get_open_positions()
        pending_buys = self.get_open_buy_orders()
        pending_syms = {o.symbol for o in pending_buys}

        state = {}
        if POSITIONS_STATE_FILE.exists():
            try:
                state = json.loads(POSITIONS_STATE_FILE.read_text(encoding="utf-8"))
            except Exception:
                state = {}

        active_risk = 0
        open_syms = set()
        for p in positions:
            sym = p["symbol"]
            open_syms.add(sym)
            st = state.get(sym, {})
            if not (st.get("tp1_hit", False) and st.get("breakeven_active", False)):
                active_risk += 1

        for sym in pending_syms:
            if sym not in open_syms:
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
        """Salvează metadatele poziției în `positions_state.json`."""
        state = {}
        if POSITIONS_STATE_FILE.exists():
            try:
                state = json.loads(POSITIONS_STATE_FILE.read_text(encoding="utf-8"))
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
            POSITIONS_STATE_FILE.write_text(json.dumps(state, indent=4, ensure_ascii=False), encoding="utf-8")
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
        setup_type: str = "PULLBACK_EMA20",
        signal_date: str = ""
    ):
        """
        Plasează ordin(e) pe Alpaca respectând idempotența, Kill-Switch-ul, limitele de expunere
        și folosind ordine `Limit` (fără urmărirea oarbă a prețului cu ordine Market).
        """
        symbol = symbol.upper().strip()
        exec_qty = round(float(qty), 4)
        if exec_qty <= 0 or limit_price <= 0 or stop_loss_price <= 0 or limit_price <= stop_loss_price:
            print(f"⚠️ {symbol} omis: Parametri de ordin/risc invalizi (qty={exec_qty}, entry={limit_price}, sl={stop_loss_price}).")
            return None

        risk_per_share = limit_price - stop_loss_price
        if tp2_price is None or tp2_price <= take_profit_price:
            tp2_rr = float(getattr(config, "TP2_RISK_REWARD_RATIO", 2.5))
            tp2_price = round(limit_price + (tp2_rr * risk_per_share), 2)

        # Validare programatică obligatorie a pragului minim Risk/Reward
        min_rr_req = float(getattr(config, "MIN_RISK_REWARD_RATIO", 2.0))
        calc_rr = (tp2_price - limit_price) / risk_per_share
        if calc_rr < min_rr_req - 1e-6:
            print(f"🛡️ [R:R GUARD] {symbol} refuzat: R:R calculat ({calc_rr:.2f}) este sub pragul minim ({min_rr_req:.2f}).")
            return None

        if not self.is_connected or not self.client:
            q1 = round(exec_qty * 0.5, 4)
            q2 = round(exec_qty - q1, 4)
            print(
                f"[SIMULARE LIMIT SPLIT-ORDER] {symbol} ({exec_qty} acțiuni @ Limit ${limit_price:.2f} | SL: ${stop_loss_price:.2f}):\n"
                f"   ├─ Lot 1 ({q1}x): TP1 @ ${take_profit_price:.2f} -> Mutare SL la Breakeven\n"
                f"   └─ Lot 2 ({q2}x): Runner TP2 @ ${tp2_price:.2f} + Trailing pe EMA 20"
            )
            self._register_position_state(
                symbol, limit_price, stop_loss_price, take_profit_price, tp2_price, setup_type, exec_qty, fractional_managed=True
            )
            return None

        # 1. Evaluare Kill-Switch Automat (pierdere zilnică / drawdown)
        ks_triggered, ks_msg = self.check_kill_switch()
        if ks_triggered:
            print(ks_msg)
            try:
                from notifier import send_telegram_message
                send_telegram_message(ks_msg)
            except Exception:
                pass
            return None

        # 2. Verificare Idempotență: poziții deja deschise ȘI ordine active/în așteptare
        open_positions = self.get_open_positions()
        open_syms = {p["symbol"] for p in open_positions}
        if symbol in open_syms:
            print(f"ℹ️ [IDEMPOTENȚĂ] Există deja poziție deschisă pe {symbol}. Ordin nou anulat.")
            return None

        open_buy_orders = self.get_open_buy_orders()
        pending_buy_syms = {o.symbol for o in open_buy_orders}
        if symbol in pending_buy_syms:
            print(f"ℹ️ [IDEMPOTENȚĂ] Există deja un ordin BUY activ/în așteptare pe {symbol}. Ordin nou anulat.")
            return None

        # 3. Verificare Limite de Expunere Simultană (Portfolio Heat, Total Positions & Sector Cap)
        total_active_and_pending = len(open_syms | pending_buy_syms)
        max_total_pos = int(getattr(config, "MAX_TOTAL_POSITIONS", config.MAX_OPEN_POSITIONS + 1))
        if total_active_and_pending >= max_total_pos:
            print(f"🛡️ [MAX TOTAL POSITIONS] {symbol} refuzat: {total_active_and_pending}/{max_total_pos} poziții/ordine active.")
            return None

        active_risk_count = self.get_active_risk_positions_count()
        if active_risk_count >= config.MAX_OPEN_POSITIONS:
            print(
                f"🛡️ [PORTFOLIO HEAT LIMIT] {symbol} refuzat: Există deja {active_risk_count}/{config.MAX_OPEN_POSITIONS} "
                f"poziții/ordine cu risc activ."
            )
            return None

        sec = config.get_ticker_sector(symbol)
        max_sec = config.get_sector_max_positions(sec)
        sec_count = sum(1 for sym in (open_syms | pending_buy_syms) if config.get_ticker_sector(sym) == sec)
        if sec_count >= max_sec:
            print(f"🛡️ [SECTOR EXPOSURE CAP] {symbol} refuzat: Sectorul {sec} are deja {sec_count}/{max_sec} poziții.")
            return None

        # 4. Verificare Preț Live (Gap Guard: nu cumpărăm în gap-down spre SL sau gap-up excesiv)
        try:
            from data_loader import get_realtime_price
            live_p = get_realtime_price(symbol)
            if live_p is not None and live_p > 0:
                if live_p <= stop_loss_price * 1.005:
                    print(f"🛡️ [GAP GUARD] {symbol} refuzat: Prețul live (${live_p:.2f}) a spart zona Stop-Loss (${stop_loss_price:.2f}).")
                    return None
                max_gap = float(getattr(config, "MAX_ENTRY_GAP_PCT", 0.012))
                if setup_type == "VOLUME_BREAKOUT" and live_p > limit_price * (1.0 + max_gap):
                    print(f"🛡️ [GAP GUARD] {symbol} refuzat: Gap-up prea mare (${live_p:.2f} > + {max_gap*100:.1f}% față de ${limit_price:.2f}).")
                    return None
        except Exception:
            pass

        # 5. Verificare Cash & Protecție Anti-Fragmentare (fără poziții miniaturale de $45!)
        summary = self.get_portfolio_summary()
        avail_cash = float(summary.get("cash", 0.0))
        eff_equity = float(summary.get("portfolio_value", config.DEFAULT_SIMULATED_CAPITAL))
        min_cost = max(
            float(getattr(config, "MIN_POSITION_COST_USD", 50.0)),
            eff_equity * float(getattr(config, "MIN_POSITION_ALLOC_PCT", 0.12))
        )
        order_cost = exec_qty * limit_price

        if avail_cash < min_cost:
            print(
                f"🛡️ [ANTI-FRAGMENTARE CASH] {symbol} refuzat: Cash disponibil (${avail_cash:.2f}) "
                f"sub alocarea minimă eficientă (${min_cost:.2f})."
            )
            return None

        if order_cost > avail_cash * 0.99:
            exec_qty = round((avail_cash * 0.99) / limit_price, 4)
            order_cost = exec_qty * limit_price
            if order_cost < min_cost:
                print(f"🛡️ [ANTI-FRAGMENTARE CASH] {symbol} refuzat: Valoare ordin (${order_cost:.2f}) sub pragul minim (${min_cost:.2f}).")
                return None

        sig_tag = (signal_date or datetime.now().strftime("%Y-%m-%d")).replace("-", "")
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
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(limit_price, 2),
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 2)),
                    stop_loss=StopLossRequest(stop_price=round(stop_loss_price, 2))
                )
                order_lot1 = self._submit_order_idempotent(req_lot1, f"SWING_{symbol}_{sig_tag}_L1")

                req_lot2 = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty_tp2,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(limit_price, 2),
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=round(tp2_price, 2)),
                    stop_loss=StopLossRequest(stop_price=round(stop_loss_price, 2))
                )
                order_lot2 = self._submit_order_idempotent(req_lot2, f"SWING_{symbol}_{sig_tag}_L2")

                print(
                    f" ✅ Ordine Split-Bracket LIMIT plasate pe Alpaca pentru {symbol} ({whole_qty} acțiuni @ ${limit_price:.2f}):\n"
                    f"    • Lot 1 ({qty_tp1}x TP1 @ ${take_profit_price:.2f}) -> ID: {order_lot1.id}\n"
                    f"    • Lot 2 ({qty_tp2}x TP2 @ ${tp2_price:.2f}) -> ID: {order_lot2.id}"
                )
                primary_order = order_lot1
                self._register_position_state(
                    symbol, limit_price, stop_loss_price, take_profit_price, tp2_price, setup_type, whole_qty, fractional_managed=False
                )
            else:
                # Execuție Fracționară prin ordin LIMIT DAY (fără slippage de ordin Market!)
                req_lim = LimitOrderRequest(
                    symbol=symbol,
                    qty=exec_qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(limit_price, 2),
                )
                primary_order = self._submit_order_idempotent(req_lim, f"SWING_{symbol}_{sig_tag}_FRAC")
                q1 = round(exec_qty * 0.5, 4)
                q2 = round(exec_qty - q1, 4)
                print(
                    f" ✅ Ordin Fracționar LIMIT plasat pe Alpaca pentru {symbol}: {exec_qty} acțiuni @ Limit ${limit_price:.2f} (~${order_cost:.2f}) -> ID: {primary_order.id}\n"
                    f"    🛡️ Protecție Activă: SL @ ${stop_loss_price:.2f} | Lot 1 ({q1}x) TP1 @ ${take_profit_price:.2f} | Lot 2 ({q2}x) TP2 @ ${tp2_price:.2f}"
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
            print(f"❌ Eroare la plasarea ordinului pe Alpaca pentru {symbol}: {e}")
            return None
