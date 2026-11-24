"""
Modul de Gestionare Automată a Pozițiilor Active (Position Manager Institutional V2).
100% aliniat cu `backtester.py` și regulile instituționale de Swing Trading:
  1. FĂRĂ Breakeven prematur la +2% (care sufoca tranzacțiile pe zgomotul intraday):
     - La atingerea TP1 (1.5R), închide 50% din poziție cu ordin Limit și mută Stop-Loss-ul
       pentru restul de 50% (Runner) la Breakeven (+0.20% buffer).
  2. Trailing Stop pe EMA 20 Zilnic (EXCLUSIV după TP1):
     - Urmărește Runner-ul pe media mobilă exponențială EMA 20 zilnică închisă până la TP2 (2.5R).
  3. Protecție la Deschiderea Bursei (09:30 - 09:40 ET):
     - Evită execuțiile Market în spread-ul larg din primele minute ale deschiderii (cu excepția
       unui gap-down sever de urgență).
  4. Curățare automată a pozițiilor fantomă din `positions_state.json` și protecție Earnings (<= 2z).
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
        """Găsește toate ordinele de Stop active/held pe server pentru un simbol."""
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

    def _is_opening_chaos_window(self) -> bool:
        """Returnează True dacă suntem în primele N minute de la deschiderea SUA (09:30 - 09:40 ET)."""
        guard_min = int(getattr(config, "POSITION_MGR_OPEN_GUARD_MINUTES", 10))
        if guard_min <= 0 or not self.trader or not getattr(self.trader, "client", None):
            return False
        try:
            clock = self.trader._api_call(self.trader.client.get_clock)
            if clock.is_open and clock.timestamp.hour == 9 and (30 <= clock.timestamp.minute < 30 + guard_min):
                return True
        except Exception:
            pass
        return False

    def _get_closed_ema20(self, symbol: str) -> float:
        """Returnează valoarea EMA 20 de pe ultima lumânare zilnică complet închisă."""
        try:
            from data_loader import fetch_stock_data
            df = fetch_stock_data(symbol, period="6mo", closed_only=True)
            if not df.empty and "EMA_20" in df.columns:
                val = float(df["EMA_20"].iloc[-1])
                if val > 0:
                    return val
        except Exception:
            pass
        return 0.0

    def manage_positions(self) -> List[Dict]:
        """
        Sincronizează și gestionează pozițiile active:
          - Elimină stările fantomă pentru simbolurile închise.
          - Execută SL, TP1 (50% + Breakeven), Trailing pe EMA 20 și TP2 (50%).
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
            from alpaca.trading.requests import GetOrdersRequest, ReplaceOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import QueryOrderStatus, OrderSide, TimeInForce
        except ImportError:
            return []

        open_positions = self.trader.get_open_positions()
        open_buy_orders = self.trader.get_open_buy_orders()
        valid_active_syms = {p["symbol"] for p in open_positions} | {o.symbol for o in open_buy_orders}

        # Curățăm automat pozițiile fantomă închise anterior din positions_state.json
        stale_keys = [k for k in list(self.state.keys()) if k not in valid_active_syms]
        if stale_keys:
            for k in stale_keys:
                self.state.pop(k, None)
            _save_json(POSITIONS_STATE_FILE, self.state)

        if not open_positions:
            return []

        try:
            all_orders = self.trader._api_call(
                self.trader.client.get_orders,
                GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100)
            )
        except Exception as e:
            print(f"⚠️ Eroare la citirea ordinelor Alpaca: {e}")
            return []

        in_open_guard = self._is_opening_chaos_window()
        actions_taken = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        today_tag = datetime.now().strftime("%Y%m%d")

        tp1_rr = float(getattr(config, "TP1_RISK_REWARD_RATIO", 1.5))
        tp2_rr = float(getattr(config, "TP2_RISK_REWARD_RATIO", 2.5))
        tp1_exit_pct = float(getattr(config, "TP1_EXIT_PERCENT", 0.50))
        be_buf = float(getattr(config, "BREAKEVEN_BUFFER_PCT", 0.0020))
        ema_trail_buf = float(getattr(config, "EMA20_TRAIL_BUFFER_PCT", 0.015))

        for pos in open_positions:
            symbol = pos["symbol"]
            qty = float(pos["qty"])
            entry_price = float(pos["avg_entry_price"])
            current_price = float(pos["current_price"])
            plpc = float(pos["unrealized_plpc"])

            active_stops = self._get_active_stop_orders(all_orders, symbol)

            if symbol not in self.state:
                default_stop = min(
                    (float(s.stop_price) for s in active_stops if s.stop_price is not None),
                    default=round(entry_price * 0.955, 2)
                )
                init_r = max(0.01, entry_price - default_stop)
                self.state[symbol] = {
                    "setup_type": "PULLBACK_EMA20",
                    "entry_price": round(entry_price, 2),
                    "initial_stop": default_stop,
                    "initial_risk": round(init_r, 2),
                    "tp1": round(entry_price + tp1_rr * init_r, 2),
                    "tp2": round(entry_price + tp2_rr * init_r, 2),
                    "initial_qty": qty,
                    "peak_price": max(entry_price, current_price),
                    "tp1_hit": False,
                    "fractional_managed": (len(active_stops) == 0),
                    "breakeven_active": (default_stop >= entry_price * 0.999),
                    "last_stop_price": default_stop,
                    "updated_at": now_str
                }

            stock_state = self.state[symbol]
            # Recalibrăm țintele la prețul real de umplere dacă tocmai s-a executat intrarea
            if abs(float(stock_state.get("entry_price", entry_price)) - entry_price) > 0.05 and not stock_state.get("tp1_hit", False):
                init_stop = float(stock_state.get("initial_stop", round(entry_price * 0.955, 2)))
                real_risk = max(0.01, entry_price - init_stop)
                stock_state["entry_price"] = round(entry_price, 2)
                stock_state["initial_risk"] = round(real_risk, 2)
                stock_state["tp1"] = round(entry_price + tp1_rr * real_risk, 2)
                stock_state["tp2"] = round(entry_price + tp2_rr * real_risk, 2)

            if active_stops:
                current_stop = min(float(s.stop_price) for s in active_stops if s.stop_price is not None)
                stock_state["last_stop_price"] = current_stop
            else:
                current_stop = float(stock_state.get("last_stop_price", round(entry_price * 0.955, 2)))

            stock_state["peak_price"] = max(float(stock_state.get("peak_price", current_price)), current_price)
            initial_qty = float(stock_state.get("initial_qty", qty))
            lot1_closed = (qty < initial_qty * 0.85) or bool(stock_state.get("tp1_hit", False))
            if lot1_closed and not stock_state.get("tp1_hit", False):
                stock_state["tp1_hit"] = True

            # --- EXECUȚIE AUTOMATĂ PENTRU POZIȚII FRACȚIONARE (FĂRĂ BRACKET NATIV PE SERVER) ---
            if not active_stops:
                tp1_level = float(stock_state.get("tp1", 0.0))
                tp2_level = float(stock_state.get("tp2", 0.0))

                # 1. Verificare Stop-Loss (în primele 10 min de la deschidere ieșim doar la urgență <= stop * 0.98)
                sl_triggered = (current_price <= current_stop * 0.98) if in_open_guard else (current_price <= current_stop)
                if sl_triggered:
                    try:
                        self.trader._api_call(self.trader.client.close_position, symbol)
                        pnl_realized = round((current_price - entry_price) * qty, 2)
                        exit_label = (
                            "TRAILING_STOP_EXIT"
                            if current_stop > entry_price * 1.005
                            else ("BREAKEVEN_EXIT" if stock_state.get("breakeven_active") else "STOP_LOSS_EXIT")
                        )
                        act_sl = {
                            "symbol": symbol,
                            "action": exit_label,
                            "old_stop": current_stop,
                            "new_stop": current_price,
                            "current_price": current_price,
                            "locked_profit": pnl_realized,
                            "shares": qty
                        }
                        actions_taken.append(act_sl)
                        self._send_telegram_update(act_sl, plpc)
                        self.state.pop(symbol, None)
                        continue
                    except Exception as e:
                        print(f"⚠️ Eroare la închiderea SL pentru {symbol}: {e}")

                if in_open_guard:
                    continue

                # 2. Verificare TP1 (50% Scale-Out prin ordin Limit + Mutare Stop la Breakeven)
                if not lot1_closed and tp1_level > 0 and current_price >= tp1_level:
                    sell_qty = round(qty * tp1_exit_pct, 4)
                    if sell_qty > 0:
                        try:
                            lim_sell_p = round(max(tp1_level * 0.998, current_price * 0.998), 2)
                            req_tp1 = LimitOrderRequest(
                                symbol=symbol,
                                qty=sell_qty,
                                side=OrderSide.SELL,
                                time_in_force=TimeInForce.DAY,
                                limit_price=lim_sell_p,
                            )
                            self.trader._submit_order_idempotent(req_tp1, f"SWING_{symbol}_{today_tag}_TP1")
                            stock_state["tp1_hit"] = True
                            stock_state["breakeven_active"] = True
                            be_stop = round(entry_price * (1.0 + be_buf), 2)
                            old_s = current_stop
                            stock_state["last_stop_price"] = max(current_stop, be_stop)
                            current_stop = stock_state["last_stop_price"]
                            stock_state["updated_at"] = now_str
                            lot1_closed = True
                            locked = round((lim_sell_p - entry_price) * sell_qty, 2)
                            act_tp1 = {
                                "symbol": symbol,
                                "action": "TP1_SCALE_OUT_50% + BREAKEVEN",
                                "old_stop": old_s,
                                "new_stop": current_stop,
                                "current_price": current_price,
                                "locked_profit": locked,
                                "shares": round(qty - sell_qty, 4)
                            }
                            actions_taken.append(act_tp1)
                            self._send_telegram_update(act_tp1, plpc)
                            continue
                        except Exception as e:
                            print(f"⚠️ Eroare la vânzarea 50% TP1 pentru {symbol}: {e}")

                # 3. Verificare TP2 (Închidere Runner 50% la ținta finală)
                elif lot1_closed and tp2_level > 0 and current_price >= tp2_level:
                    try:
                        lim_tp2_p = round(max(tp2_level * 0.998, current_price * 0.998), 2)
                        req_tp2 = LimitOrderRequest(
                            symbol=symbol,
                            qty=qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                            limit_price=lim_tp2_p,
                        )
                        self.trader._submit_order_idempotent(req_tp2, f"SWING_{symbol}_{today_tag}_TP2")
                        locked = round((lim_tp2_p - entry_price) * qty, 2)
                        act_tp2 = {
                            "symbol": symbol,
                            "action": "TP2_RUNNER_EXIT_100%",
                            "old_stop": current_stop,
                            "new_stop": lim_tp2_p,
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

            if in_open_guard:
                continue

            target_stop = None
            action_type = None

            # --- REGULA A: PROTECȚIE EARNINGS IMINENT (<= 2 ZILE) ---
            warn_days = int(getattr(config, "EARNINGS_OPEN_POS_WARN_DAYS", 2))
            try:
                from data_loader import get_next_earnings_date
                next_e = get_next_earnings_date(symbol)
                if next_e is not None:
                    days_to_e = (next_e - datetime.now()).days
                    if 0 <= days_to_e <= warn_days and current_price > entry_price * 1.005:
                        tight_stop = round(max(entry_price * (1.0 + be_buf), current_price * 0.985), 2)
                        if tight_stop > current_stop and tight_stop < current_price * 0.995:
                            target_stop = tight_stop
                            action_type = "EARNINGS_PROTECTION"
            except Exception:
                pass

            # --- REGULA B: BREAKEVEN PENTRU BRACKET-URI ÎNTREGI DUPĂ ÎNCHIDEREA LOTULUI 1 LA TP1 ---
            if target_stop is None and lot1_closed and not stock_state.get("breakeven_active", False):
                be_level = round(entry_price * (1.0 + be_buf), 2)
                if be_level > current_stop and be_level < current_price * 0.995:
                    target_stop = be_level
                    action_type = "BREAKEVEN"

            # --- REGULA C: TRAILING STOP PE EMA 20 ZILNIC (EXCLUSIV DUPĂ ATINGEREA TP1!) ---
            if target_stop is None and lot1_closed:
                closed_ema20 = self._get_closed_ema20(symbol)
                if closed_ema20 > 0:
                    trail_ema = round(closed_ema20 * (1.0 - ema_trail_buf), 2)
                    if trail_ema > entry_price * (1.0 + be_buf) and trail_ema > current_stop + 0.25 and trail_ema < current_price * 0.995:
                        target_stop = trail_ema
                        action_type = "TRAILING_STOP"

            if target_stop and target_stop > current_stop and target_stop < current_price:
                success_all = True
                for stop_order in active_stops:
                    try:
                        replace_req = ReplaceOrderRequest(stop_price=target_stop)
                        self.trader._api_call(self.trader.client.replace_order_by_id, str(stop_order.id), replace_req)
                    except Exception as e:
                        print(f"⚠️ Eroare modificare Stop pe server pentru {symbol} (Order {stop_order.id}): {e}")
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

            if act == "STOP_LOSS_EXIT":
                title = "🛑 *POZIȚIE ÎNCHISĂ LA STOP-LOSS*"
                desc = (
                    f"Acțiunea `{sym}` a atins Stop-Loss-ul structural (`${old_s:.2f}`).\n\n"
                    f"• Preț Execuție: *${cur_p:.2f}* (*{plpc:+.2f}%*)\n"
                    f"• Rezultat Realizat: *${locked:+,.2f}*"
                )
            elif act in ("BREAKEVEN_EXIT", "TRAILING_STOP_EXIT"):
                title = "🔒 *RUNNER ÎNCHIS LA BREAKEVEN / TRAILING STOP*"
                desc = (
                    f"Acțiunea `{sym}` a atins Stop-ul dinamic (`${old_s:.2f}`).\n\n"
                    f"• Preț Execuție: *${cur_p:.2f}* (*{plpc:+.2f}%*)\n"
                    f"💰 *Rezultat Runner:* *${locked:+,.2f}*"
                )
            elif "TP1_SCALE_OUT" in act:
                title = "🎯 *TP1 ATINS (50% PROFIT MARCAT + BREAKEVEN ACTIV)*"
                desc = (
                    f"Acțiunea `{sym}` a atins ținta TP1 (`1.5R`)!\n\n"
                    f"• Preț Curent: *${cur_p:.2f}* (*{plpc:+.2f}%*)\n"
                    f"• Profit Marcat pe Lotul 1 (50%): *${locked:+,.2f}*\n"
                    f"🛡️ *Stop-Loss-ul pentru Runner (50%) a fost mutat la Breakeven (${new_s:.2f})!*"
                )
            elif "TP2_RUNNER_EXIT" in act:
                title = "🚀 *TP2 ATINS (RUNNER ÎNCHIS LA ȚINTA FINALĂ)*"
                desc = (
                    f"Runner-ul pe `{sym}` a atins ținta TP2 (`2.5R`)!\n\n"
                    f"• Preț Execuție: *${cur_p:.2f}* (*{plpc:+.2f}%*)\n"
                    f"💰 *Profit Realizat Lot 2:* *${locked:+,.2f}*"
                )
            elif act == "TRAILING_STOP":
                title = "📈 *TRAILING STOP PE EMA 20 ACTUALIZAT*"
                desc = (
                    f"Runner-ul pe `{sym}` își continuă trendul (*{plpc:+.2f}%*).\n\n"
                    f"• Preț Curent: *${cur_p:.2f}*\n"
                    f"• Stop Vechi: ${old_s:.2f} ➔ *Noul Stop EMA 20: ${new_s:.2f}*\n"
                    f"💰 *Profit Minim Garantat:* *${locked:+,.2f}*"
                )
            elif act == "EARNINGS_PROTECTION":
                title = "⚠️ *PROTECȚIE EARNINGS IMINENT ACTIVATĂ*"
                desc = (
                    f"Acțiunea `{sym}` raportează rezultate financiare în <= 2 zile!\n\n"
                    f"• Preț Curent: *${cur_p:.2f}*\n"
                    f"• Stop strâns preventiv: ${old_s:.2f} ➔ *${new_s:.2f}*"
                )
            else:
                title = "🛡️ *BREAKEVEN DUPĂ TP1 ACTIVAT*"
                desc = (
                    f"Lotul 1 pe `{sym}` a fost marcat la TP1.\n\n"
                    f"• Stop Vechi: ${old_s:.2f} ➔ *Stop Breakeven: ${new_s:.2f}*"
                )

            send_telegram_message(f"{title}\n\n{desc}")
        except Exception:
            pass


def run_position_manager(trader=None) -> List[Dict]:
    """Punct de intrare pentru gestionarea pozițiilor active."""
    pm = PositionManager(trader)
    return pm.manage_positions()
