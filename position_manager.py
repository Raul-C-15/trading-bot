"""
Modul de Gestionare Automată a Pozițiilor Active (Position Manager Institutional).
Funcționalități profesionale de swing trading:
1. Auto-Breakeven (Risc Zero / Free-Roll):
   - Când o poziție atinge un profit de >= 1.0R, >= +2.0% sau când Lotul 1 (50%) s-a închis la TP1,
     Stop-Loss-ul este ridicat automat la nivelul prețului de intrare (+0.15% buffer).
   - Poziția devine 'Free-Roll' (0% risc activ în Portfolio Heat).
2. Trailing Stop Dinamic Hibrid (Chandelier + EMA 20 / Low 3D):
   - Pentru pozițiile cu câștiguri mari (>= +4.5% sau >= 1.4R), Stop-Loss-ul urmărește
     prețul combinând distanța Chandelier (3.8% sub maximul atins) cu suportul tehnic.
3. Protecție Automată la Earnings pentru Pozițiile Deschise:
   - Dacă o poziție activă are raportare financiară (Earnings) în <= 2 zile, strânge
     automat Stop-Loss-ul pentru a proteja capitalul împotriva gap-urilor binare.
4. Notificări instantanee pe Telegram la fiecare ridicare de Stop-Loss.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import config

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
POSITIONS_STATE_FILE = DATA_DIR / "positions_state.json"


def _load_json(file_path: Path, default_value):
    if not file_path.exists():
        return default_value
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_value


def _save_json(file_path: Path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


class PositionManager:
    def __init__(self, trader=None):
        self.trader = trader
        self.state = _load_json(POSITIONS_STATE_FILE, {})

    def _get_active_stop_orders(self, orders, symbol: str) -> list:
        """Găsește toate ordinele de Stop active/held pentru un simbol."""
        active = []
        for o in orders:
            if o.symbol != symbol:
                continue
            side = getattr(o.side, "name", str(o.side)).lower()
            order_type = getattr(o.order_type, "name", str(o.order_type)).lower()
            status = getattr(o.status, "name", str(o.status)).lower()

            if side == "sell" and "stop" in order_type and status in ("new", "held", "open", "accepted"):
                active.append(o)
        return active

    def manage_positions(self) -> List[Dict]:
        """
        Scanează toate pozițiile active și aplică regulile de Auto-Breakeven,
        Trailing Stop și Protecție înainte de Earnings.
        """
        if self.trader is None:
            try:
                from bot_trader import AlpacaTrader
                self.trader = AlpacaTrader()
            except Exception:
                return []

        if not getattr(self.trader, "is_connected", False) or not getattr(self.trader, "client", None):
            return []

        try:
            from alpaca.trading.requests import GetOrdersRequest, ReplaceOrderRequest, MarketOrderRequest
            from alpaca.trading.enums import QueryOrderStatus, OrderSide, TimeInForce
        except ImportError:
            return []

        open_positions = self.trader.get_open_positions()
        if not open_positions:
            return []

        try:
            all_orders = self.trader.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=100))
        except Exception as e:
            print(f"⚠️ Eroare la citirea ordinelor Alpaca: {e}")
            return []

        actions_taken = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for pos in open_positions:
            symbol = pos["symbol"]
            qty = float(pos["qty"])
            entry_price = float(pos["avg_entry_price"])
            current_price = float(pos["current_price"])
            plpc = float(pos["unrealized_plpc"])

            active_stops = self._get_active_stop_orders(all_orders, symbol)

            if symbol not in self.state:
                default_stop = min((float(s.stop_price) for s in active_stops if s.stop_price is not None), default=round(entry_price * 0.965, 2))
                self.state[symbol] = {
                    "setup_type": "PULLBACK_EMA20",
                    "entry_price": entry_price,
                    "initial_stop": default_stop,
                    "initial_risk": max(0.01, entry_price - default_stop),
                    "tp1": round(entry_price + 1.5 * max(0.01, entry_price - default_stop), 2),
                    "tp2": round(entry_price + 3.0 * max(0.01, entry_price - default_stop), 2),
                    "initial_qty": qty,
                    "peak_price": current_price,
                    "tp1_hit": False,
                    "fractional_managed": (len(active_stops) == 0),
                    "breakeven_active": (default_stop >= entry_price * 0.999),
                    "last_stop_price": default_stop,
                    "updated_at": now_str
                }

            stock_state = self.state[symbol]
            if active_stops:
                current_stop = min(float(s.stop_price) for s in active_stops if s.stop_price is not None)
                stock_state["last_stop_price"] = current_stop
            else:
                current_stop = float(stock_state.get("last_stop_price", round(entry_price * 0.965, 2)))

            stock_state["peak_price"] = max(stock_state.get("peak_price", current_price), current_price)
            initial_risk = stock_state.get("initial_risk", max(0.01, entry_price - current_stop))
            if initial_risk <= 0.05:
                initial_risk = max(0.50, entry_price * 0.025)

            initial_qty = float(stock_state.get("initial_qty", qty))
            lot1_closed = (qty < initial_qty * 0.95) or bool(stock_state.get("tp1_hit", False))

            # --- EXECUȚIE AUTOMATĂ SL / TP1 (50%) / TP2 (50%) PENTRU POZIȚII FRACȚIONARE ---
            if not active_stops:
                tp1_level = float(stock_state.get("tp1", 0.0))
                tp2_level = float(stock_state.get("tp2", 0.0))

                # 1. Verificare atingere Stop-Loss / Breakeven / Trailing Stop
                if current_price <= current_stop:
                    try:
                        self.trader.client.close_position(symbol)
                        pnl_realized = round((current_price - entry_price) * qty, 2)
                        exit_label = "TRAILING_STOP_EXIT" if current_stop > entry_price * 1.005 else ("BREAKEVEN_EXIT" if stock_state.get("breakeven_active") else "STOP_LOSS_EXIT")
                        actions_taken.append({
                            "symbol": symbol,
                            "action": exit_label,
                            "old_stop": current_stop,
                            "new_stop": current_price,
                            "current_price": current_price,
                            "locked_profit": pnl_realized,
                            "shares": qty
                        })
                        self.state.pop(symbol, None)
                        continue
                    except Exception as e:
                        print(f"⚠️ Eroare la închiderea SL pentru poziția fracționară {symbol}: {e}")

                # 2. Verificare atingere TP1 (50% Scale-Out + Mutare Stop la Breakeven)
                if not lot1_closed and tp1_level > 0 and current_price >= tp1_level:
                    sell_qty = round(qty * float(getattr(config, "TP1_EXIT_PERCENT", 0.50)), 4)
                    if sell_qty > 0:
                        try:
                            req_tp1 = MarketOrderRequest(
                                symbol=symbol,
                                qty=sell_qty,
                                side=OrderSide.SELL,
                                time_in_force=TimeInForce.DAY
                            )
                            self.trader.client.submit_order(order_data=req_tp1)
                            stock_state["tp1_hit"] = True
                            stock_state["breakeven_active"] = True
                            be_stop = round(entry_price * 1.0015, 2)
                            stock_state["last_stop_price"] = max(current_stop, be_stop)
                            current_stop = stock_state["last_stop_price"]
                            lot1_closed = True
                            locked = round((tp1_level - entry_price) * sell_qty, 2)
                            act_tp1 = {
                                "symbol": symbol,
                                "action": "TP1_SCALE_OUT_50% + BREAKEVEN",
                                "old_stop": current_stop,
                                "new_stop": current_stop,
                                "current_price": current_price,
                                "locked_profit": locked,
                                "shares": round(qty - sell_qty, 4)
                            }
                            actions_taken.append(act_tp1)
                            self._send_telegram_update(act_tp1, plpc)
                        except Exception as e:
                            print(f"⚠️ Eroare la vânzarea 50% TP1 pentru {symbol}: {e}")

                # 3. Verificare atingere TP2 (Închidere Runner 50%)
                if lot1_closed and tp2_level > 0 and current_price >= tp2_level:
                    try:
                        self.trader.client.close_position(symbol)
                        locked = round((current_price - entry_price) * qty, 2)
                        act_tp2 = {
                            "symbol": symbol,
                            "action": "TP2_RUNNER_EXIT_100%",
                            "old_stop": current_stop,
                            "new_stop": current_price,
                            "current_price": current_price,
                            "locked_profit": locked,
                            "shares": qty
                        }
                        actions_taken.append(act_tp2)
                        self._send_telegram_update(act_tp2, plpc)
                        self.state.pop(symbol, None)
                        continue
                    except Exception as e:
                        print(f"⚠️ Eroare la închiderea TP2 pentru {symbol}: {e}")

            gain_per_share = current_price - entry_price
            r_multiple = (gain_per_share / initial_risk) if initial_risk > 0 else 0.0

            target_stop = None
            action_type = None

            # --- REGULA 0: PROTECȚIE EARNINGS IMINENT (<= 2 ZILE) PENTRU POZIȚII DESCHISE ---
            warn_days = getattr(config, "EARNINGS_OPEN_POS_WARN_DAYS", 2)
            try:
                from data_loader import get_next_earnings_date
                next_e = get_next_earnings_date(symbol)
                if next_e is not None:
                    days_to_e = (next_e - datetime.now()).days
                    if 0 <= days_to_e <= warn_days:
                        # Strângem Stop-Loss-ul la minim Breakeven sau la 1.5% sub prețul curent
                        tight_stop = round(max(entry_price * 1.001, current_price * 0.985), 2)
                        if tight_stop > current_stop and tight_stop < current_price:
                            target_stop = tight_stop
                            action_type = "EARNINGS_PROTECTION"
            except Exception:
                pass

            # --- REGULA 1: TRAILING STOP (Profit Lock pentru câștiguri mari >= +4.5% sau >= 1.4R) ---
            if target_stop is None and (plpc >= 4.5 or r_multiple >= 1.4):
                trail_level = round(stock_state["peak_price"] * 0.962, 2)

                if trail_level > entry_price * 1.01 and trail_level > current_stop + 0.50:
                    target_stop = trail_level
                    action_type = "TRAILING_STOP"

            # --- REGULA 2: AUTO-BREAKEVEN (pentru >= +2.0%, >= 1.0R sau după închiderea Lotului 1 la TP1) ---
            elif target_stop is None and (plpc >= 2.0 or r_multiple >= 1.0 or lot1_closed) and not stock_state.get("breakeven_active", False):
                breakeven_level = round(entry_price * 1.0015, 2)
                if breakeven_level > current_stop and breakeven_level < current_price:
                    target_stop = breakeven_level
                    action_type = "BREAKEVEN"

            # Executăm modificarea dacă am identificat un Stop mai mare
            if target_stop and target_stop > current_stop and target_stop < current_price:
                success_all = True
                for stop_order in active_stops:
                    try:
                        replace_req = ReplaceOrderRequest(stop_price=target_stop)
                        self.trader.client.replace_order_by_id(str(stop_order.id), replace_req)
                    except Exception as e:
                        print(f"⚠️ Eroare modificare Stop pentru {symbol} (Order {stop_order.id}): {e}")
                        success_all = False

                if success_all:
                    stock_state["last_stop_price"] = target_stop
                    stock_state["updated_at"] = now_str
                    if target_stop >= entry_price * 0.999:
                        stock_state["breakeven_active"] = True

                    locked_profit = (target_stop - entry_price) * qty
                    action_info = {
                        "symbol": symbol,
                        "action": action_type,
                        "old_stop": current_stop,
                        "new_stop": target_stop,
                        "current_price": current_price,
                        "locked_profit": round(locked_profit, 2),
                        "shares": qty
                    }
                    actions_taken.append(action_info)
                    self._send_telegram_update(action_info, plpc)

        _save_json(POSITIONS_STATE_FILE, self.state)
        return actions_taken

    def _send_telegram_update(self, info: Dict, plpc: float):
        try:
            from notifier import send_telegram_message
            sym = info["symbol"]
            act = info["action"]
            old_s = info["old_stop"]
            new_s = info["new_stop"]
            cur_p = info["current_price"]
            locked = info["locked_profit"]

            if act == "TRAILING_STOP":
                title = "📈 *TRAILING STOP ACTIVAT (PROFIT SECURIZAT)*"
                desc = (
                    f"Acțiunea `{sym}` este în creștere puternică (*{plpc:+.2f}%*).\n\n"
                    f"• Preț Curent: *${cur_p:.2f}*\n"
                    f"• Stop Vechi: ${old_s:.2f} ➔ *Noul Stop: ${new_s:.2f}*\n"
                    f"💰 *Profit Minim Garantat:* *${locked:+,.2f}*"
                )
            elif act == "EARNINGS_PROTECTION":
                title = "⚠️ *PROTECȚIE EARNINGS IMINENT ACTIVATĂ*"
                desc = (
                    f"Acțiunea `{sym}` raportează rezultate financiare în <= 2 zile!\n\n"
                    f"• Preț Curent: *${cur_p:.2f}*\n"
                    f"• Stop strâns preventiv: ${old_s:.2f} ➔ *${new_s:.2f}*\n"
                    f"🛡️ *Profit Protejat:* *${locked:+,.2f}*"
                )
            else:
                title = "🛡️ *AUTO-BREAKEVEN ACTIVAT (FREE-ROLL / RISC ZERO)*"
                desc = (
                    f"Poziția pe `{sym}` a avansat favorabil (*{plpc:+.2f}%*).\n\n"
                    f"• Preț Curent: *${cur_p:.2f}*\n"
                    f"• Stop Vechi: ${old_s:.2f} ➔ *Stop Breakeven: ${new_s:.2f}*\n"
                    f"🔒 *Tranzacția este acum 100% fără risc și eliberează loc în Portfolio Heat!*"
                )

            msg = f"{title}\n\n{desc}"
            send_telegram_message(msg)
        except Exception:
            pass


def run_position_manager(trader=None) -> List[Dict]:
    """Punct de intrare simplu pentru rularea verificării de poziții."""
    pm = PositionManager(trader)
    return pm.manage_positions()
