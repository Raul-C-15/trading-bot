"""
Listener pentru butoanele interactive si comenzile de chat de pe Telegram.
Asculta in fundal:
1. Evenimentele callback_query ([✅ CUMPARA], [❌ IGNORA] si butoanele rapide de meniu).
2. Comenzile scrise direct in chat (/status, /positions, /scan, /watchlist, /help).
"""

import warnings
warnings.filterwarnings("ignore")

import json
import threading
import time
from pathlib import Path
import requests
import config
from notifier import (
    PENDING_SIGNALS,
    send_telegram_message,
    edit_telegram_message,
    answer_callback_query,
    load_telegram_state,
    save_telegram_state,
)

POSITIONS_STATE_FILE = Path(__file__).parent / "data" / "positions_state.json"

QUICK_MENU_MARKUP = {
    "inline_keyboard": [
        [
            {"text": "📊 Status & Piață", "callback_data": "CMD:STATUS"},
            {"text": "📂 Poziții Active", "callback_data": "CMD:POSITIONS"},
        ],
        [
            {"text": "🌟 Watchlist Azi", "callback_data": "CMD:WATCHLIST"},
            {"text": "🔎 Scanează Acum", "callback_data": "CMD:SCAN"},
        ],
        [
            {"text": "🤖 Comută Auto-Trade (ON / OFF)", "callback_data": "CMD:AUTOTRADE"},
        ],
    ]
}


class TelegramListener:
    def __init__(self, trader):
        self.trader = trader
        self.running = False
        state = load_telegram_state()
        self.last_update_id = int(state.get("last_update_id", 0))
        self.thread = None
        self._scan_in_progress = False

    def _register_bot_commands(self):
        """Inregistreaza meniul oficial de comenzi in aplicatia Telegram."""
        if not config.TELEGRAM_BOT_TOKEN:
            return
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/setMyCommands"
        commands = [
            {"command": "status", "description": "📊 Portofoliu Alpaca, Cash & Regimul Pieței"},
            {"command": "positions", "description": "📂 Poziții deschise, P&L și niveluri Stop-Loss/TP"},
            {"command": "scan", "description": "🔎 Declanșează o scanare live a pieței acum"},
            {"command": "watchlist", "description": "🌟 Lista de lideri monitorizați azi"},
            {"command": "autotrade", "description": "🤖 Activează / Dezactivează Auto-Trade-ul (ON/OFF)"},
            {"command": "help", "description": "ℹ️ Meniu rapid și lista de comenzi"},
        ]
        try:
            requests.post(url, json={"commands": commands}, timeout=5)
        except Exception:
            pass

    def start(self):
        """Porneste listenerul intr-un thread separat in fundal."""
        if not config.TELEGRAM_BOT_TOKEN:
            print("ℹ️ Telegram Token lipseste; ascultatorul de butoane nu va porni.")
            return

        self._register_bot_commands()
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()
        print("📱 Ascultator Telegram (Butoane + Comenzi /status, /positions, /scan, /watchlist) ACTIVAT!")

    def stop(self):
        """Opreste bucla de ascultare."""
        self.running = False

    def poll_once(self, wait_seconds: int = 2) -> int:
        """
        Verifica si proceseaza toate apasarile de butoane si comenzile Telegram acumulate in coada.
        Returneaza numarul de actiuni procesate.
        """
        if not config.TELEGRAM_BOT_TOKEN:
            return 0

        load_telegram_state()
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/getUpdates"
        processed = 0
        try:
            params = {"timeout": wait_seconds}
            if self.last_update_id > 0:
                params["offset"] = self.last_update_id + 1

            res = requests.get(url, params=params, timeout=wait_seconds + 8)
            if not res.ok:
                return 0

            data = res.json()
            if not data.get("ok"):
                return 0

            for update in data.get("result", []):
                self.last_update_id = update["update_id"]
                if "callback_query" in update:
                    self._handle_callback(update["callback_query"])
                    processed += 1
                elif "message" in update:
                    self._handle_message(update["message"])
                    processed += 1

            save_telegram_state(last_update_id=self.last_update_id)
        except Exception:
            pass
        return processed

    def _poll_loop(self):
        """Bucla de long polling pentru actualizari Telegram."""
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/getUpdates"

        # Sincronizare initiala doar daca nu avem un last_update_id salvat
        if self.last_update_id == 0:
            try:
                init_res = requests.get(url, params={"limit": 10}, timeout=5).json()
                if init_res.get("ok") and init_res.get("result"):
                    self.last_update_id = init_res["result"][-1]["update_id"]
                    save_telegram_state(last_update_id=self.last_update_id)
            except Exception:
                pass

        while self.running:
            try:
                params = {
                    "offset": self.last_update_id + 1,
                    "timeout": 5
                }
                res = requests.get(url, params=params, timeout=10)
                if not res.ok:
                    time.sleep(2)
                    continue

                data = res.json()
                if not data.get("ok"):
                    time.sleep(2)
                    continue

                for update in data.get("result", []):
                    self.last_update_id = update["update_id"]

                    if "callback_query" in update:
                        self._handle_callback(update["callback_query"])
                    elif "message" in update:
                        self._handle_message(update["message"])
                    save_telegram_state(last_update_id=self.last_update_id)

            except Exception:
                time.sleep(2)

    def _handle_message(self, msg: dict):
        """Proceseaza comenzile text trimise pe Telegram (/status, /positions, /scan, /watchlist)."""
        chat_id = str(msg.get("chat", {}).get("id", ""))
        if config.TELEGRAM_CHAT_ID and chat_id != str(config.TELEGRAM_CHAT_ID):
            return

        text = (msg.get("text") or "").strip()
        if not text:
            return

        parts = text.split()
        cmd = parts[0].lower().split("@")[0]
        arg = parts[1].lower() if len(parts) > 1 else None

        if cmd in ("/status", "status"):
            self._cmd_status()
        elif cmd in ("/positions", "/pozitii", "positions", "pozitii"):
            self._cmd_positions()
        elif cmd in ("/watchlist", "watchlist"):
            self._cmd_watchlist()
        elif cmd in ("/scan", "scan"):
            self._cmd_scan()
        elif cmd in ("/autotrade", "autotrade"):
            self._cmd_autotrade(arg)
        elif cmd in ("/help", "/start", "help", "meniu"):
            self._cmd_help()

    def _cmd_help(self):
        mode_str = "🟢 ACTIVAT (Intrare Automată)" if config.AUTO_TRADE else "🟡 DEZACTIVAT (Confirmare 1-Click)"
        msg = (
            "🤖 *MENIU DE CONTROL — SWING TRADING BOT*\n\n"
            "Poți folosi oricând următoarele comenzi sau butoanele de mai jos:\n"
            "• `/status` — Valoare portofoliu, Cash, Ceas bursă & Regim Piață\n"
            "• `/positions` — Poziții deschise, P&L și niveluri Stop-Loss / TP\n"
            "• `/watchlist` — Liderii monitorizați azi (Smart Hybrid)\n"
            "• `/scan` — Pornește o scanare completă a pieței acum\n"
            f"• `/autotrade` — Comută modul de execuție (Acum: *{mode_str}*)"
        )
        send_telegram_message(msg, reply_markup=QUICK_MENU_MARKUP)

    def _cmd_autotrade(self, arg: str = None):
        """Comuta modul AUTO_TRADE direct din Telegram si salveaza starea pe disc/Git."""
        load_telegram_state()
        if arg in ("on", "true", "1", "da", "activ"):
            new_state = True
        elif arg in ("off", "false", "0", "nu", "manual"):
            new_state = False
        else:
            new_state = not bool(config.AUTO_TRADE)

        config.AUTO_TRADE = new_state
        save_telegram_state(auto_trade=new_state)
        try:
            from scheduler import sync_state_to_git
            sync_state_to_git()
        except Exception:
            pass

        if new_state:
            msg = (
                "🤖 *MOD AUTO-TRADE: ACTIVAT (ON)* 🟢\n\n"
                "De acum, când botul găsește un semnal valid (1W Stage 2 + ML + Știri OK + Sector Cap OK), "
                "*va plasa automat ordinul pe Alpaca* la prețul optim și îți va trimite confirmarea pe Telegram!"
            )
        else:
            msg = (
                "📱 *MOD AUTO-TRADE: DEZACTIVAT (OFF — Confirmare 1-Click)* 🟡\n\n"
                "De acum, botul *nu mai cumpără singur*. Îți va trimite alerta cu butonul "
                "*`[✅ CUMPĂRĂ]`* și va intra doar când apeși tu pe buton!"
            )
        send_telegram_message(msg, reply_markup=QUICK_MENU_MARKUP)

    def _cmd_status(self):
        try:
            summary = self.trader.get_portfolio_summary()
            open_pos = self.trader.get_open_positions()
            active_risk = self.trader.get_active_risk_positions_count()

            clock_str = "⚪ Mod Simulat"
            if self.trader.is_connected and self.trader.client:
                try:
                    c = self.trader.client.get_clock()
                    if c.is_open:
                        clock_str = "🟢 DESCHISĂ ACUM"
                    else:
                        next_open = c.next_open.strftime("%Y-%m-%d %H:%M %Z")
                        clock_str = f"🌙 ÎNCHISĂ (Deschide: `{next_open}`)"
                except Exception:
                    pass

            from data_loader import fetch_market_regime
            m = fetch_market_regime(config.BENCHMARK_TICKER)
            regime = m.get("regime", "N/A")
            spy_price = m.get("price", 0.0)
            spy_sma200 = m.get("sma_200", 0.0)
            vix_val = m.get("vix", 0.0)
            risk_mult = m.get("risk_multiplier", 1.0) * 100

            msg = (
                f"📊 *STATUS PORTOFOLIU & PIAȚĂ*\n\n"
                f"🏛️ *Bursa SUA:* {clock_str}\n"
                f"💰 *Valoare Portofoliu (Buget):* `${summary['portfolio_value']:,.2f}`\n"
                f"💵 *Cash Disponibil:* `${summary['cash']:,.2f}`\n"
                f"📂 *Poziții Deschise:* `{len(open_pos)}` (Risc activ: `{active_risk}/{config.MAX_OPEN_POSITIONS}`)\n\n"
                f"📈 *Regim Piață:* `{regime}` (Risc: `{risk_mult:.0f}%`)\n"
                f"🔹 *SPY:* `${spy_price:.2f}` (SMA 200: `${spy_sma200:.2f}`)\n"
                f"🔹 *Volatilitate VIX:* `{vix_val:.1f}`\n"
                f"⚙️ *Auto-Trade:* `{'ACTIVAT' if config.AUTO_TRADE else 'DEZACTIVAT (Confirmare 1-Click)'}`"
            )
            send_telegram_message(msg, reply_markup=QUICK_MENU_MARKUP)
        except Exception as e:
            send_telegram_message(f"⚠️ Eroare la citirea statusului: `{e}`")

    def _cmd_positions(self):
        try:
            positions = self.trader.get_open_positions()
            if not positions:
                send_telegram_message(
                    "📂 *Nu ai nicio poziție deschisă în acest moment.*",
                    reply_markup=QUICK_MENU_MARKUP,
                )
                return

            pos_state = {}
            if POSITIONS_STATE_FILE.exists():
                try:
                    with open(POSITIONS_STATE_FILE, "r", encoding="utf-8") as f:
                        pos_state = json.load(f)
                except Exception:
                    pass

            lines = [f"📂 *POZIȚII ACTIVE PE ALPACA ({len(positions)})*\n"]
            total_pl = 0.0
            for p in positions:
                sym = p["symbol"]
                qty = p["qty"]
                entry = p["avg_entry_price"]
                curr = p["current_price"]
                pl = p["unrealized_pl"]
                pl_pct = p["unrealized_plpc"]
                total_pl += pl

                st = pos_state.get(sym, {})
                sl = st.get("last_stop_price", st.get("current_stop", st.get("initial_stop", 0.0)))
                tp1 = st.get("tp1", st.get("tp1_price", 0.0))
                tp2 = st.get("tp2", st.get("tp2_price", 0.0))
                be = "🛡️ Free-Roll (BE)" if (st.get("breakeven_active") or st.get("breakeven_moved")) else "⚡ Risc Activ"

                qty_str = f"{int(qty)}" if float(qty).is_integer() else f"{qty:.4f}".rstrip("0").rstrip(".")
                icon = "🟢" if pl >= 0 else "🔴"
                sl_info = f"SL: `${sl:.2f}`" if sl else "SL: N/A"
                if tp1 and tp2:
                    sl_info += f" | TP1: `${tp1:.2f}` | TP2: `${tp2:.2f}`"

                lines.append(
                    f"{icon} *{sym}* (`{qty_str}` acțiuni) — {be}\n"
                    f"   • Intrare: `${entry:.2f}` -> Curent: `${curr:.2f}`\n"
                    f"   • P&L: `${pl:+,.2f}` (`{pl_pct:+.2f}%`)\n"
                    f"   • {sl_info}"
                )

            lines.append(f"\n💰 *P&L Nerealizat Total:* `${total_pl:+,.2f}`")
            send_telegram_message("\n".join(lines), reply_markup=QUICK_MENU_MARKUP)
        except Exception as e:
            send_telegram_message(f"⚠️ Eroare la citirea pozițiilor: `{e}`")

    def _cmd_watchlist(self):
        try:
            from data_loader import get_smart_hybrid_watchlist
            wl, meta = get_smart_hybrid_watchlist(force_refresh=False)
            date_str = meta.get("date", "Azi")
            msg = (
                f"🌟 *WATCHLIST SMART HYBRID ({len(wl)} LIDERI)*\n"
                f"📅 *Actualizat:* `{date_str}`\n\n"
                f"`{', '.join(wl)}`"
            )
            send_telegram_message(msg, reply_markup=QUICK_MENU_MARKUP)
        except Exception as e:
            send_telegram_message(f"⚠️ Eroare la încărcarea Watchlist-ului: `{e}`")

    def _cmd_scan(self):
        if self._scan_in_progress:
            send_telegram_message("⏳ *O scanare este deja în curs de desfășurare...*")
            return

        def _run_bg_scan():
            self._scan_in_progress = True
            try:
                send_telegram_message("🔎 *Scanare live pornită la cerere...* Te anunț imediat ce se finalizează!")
                from scheduler import run_automated_scan
                found = run_automated_scan(self.trader)
                if not found:
                    send_telegram_message(
                        "✅ *Scanare finalizată:* Niciun semnal nou de cumpărare în acest moment (filtrele de risc și trend sunt active).",
                        reply_markup=QUICK_MENU_MARKUP,
                    )
            except Exception as e:
                send_telegram_message(f"⚠️ Eroare în timpul scanării: `{e}`")
            finally:
                self._scan_in_progress = False

        threading.Thread(target=_run_bg_scan, daemon=True).start()

    def _handle_callback(self, cq):
        """Proceseaza apasarea unui buton."""
        cq_id = cq.get("id")
        data = cq.get("data", "")
        message = cq.get("message", {})
        message_id = message.get("message_id")

        if data.startswith("CMD:"):
            cmd_type = data.split(":", 1)[1]
            answer_callback_query(cq_id, f"Se execută {cmd_type}...")
            if cmd_type == "STATUS":
                self._cmd_status()
            elif cmd_type == "POSITIONS":
                self._cmd_positions()
            elif cmd_type == "WATCHLIST":
                self._cmd_watchlist()
            elif cmd_type == "SCAN":
                self._cmd_scan()
            elif cmd_type == "AUTOTRADE":
                self._cmd_autotrade()
            return

        if data.startswith("BUY:"):
            sig_id = data.split(":", 1)[1]
            signal = PENDING_SIGNALS.get(sig_id)

            if signal:
                answer_callback_query(cq_id, f"Se plaseaza ordinul pentru {signal.ticker}...")

                # Executie ordin pe Alpaca (cu suport Split-Order TP1 + TP2 + Fractii)
                order_qty = float(signal.shares)
                order = self.trader.place_bracket_buy_order(
                    symbol=signal.ticker,
                    qty=order_qty,
                    limit_price=signal.entry_price,
                    stop_loss_price=signal.stop_loss,
                    take_profit_price=signal.tp1,
                    tp2_price=getattr(signal, "tp2", None),
                    setup_type=getattr(signal, "setup_type", "PULLBACK_EMA20")
                )

                if order:
                    tp2_val = getattr(signal, "tp2", None)
                    tp_text = f"`${signal.tp1:.2f}` (Lot 1 50%) / `${tp2_val:.2f}` (Runner 50%)" if tp2_val else f"`${signal.tp1:.2f}`"
                    qty_str = f"{int(order_qty)}" if float(order_qty).is_integer() else f"{order_qty:.4f}".rstrip("0").rstrip(".")
                    status_text = (
                        f"✅ *ORDIN EXECUTAT PE ALPACA!*\n\n"
                        f"🔹 Simbol: `{signal.ticker}`\n"
                        f"🔹 Cantitate: `{qty_str}` actiuni\n"
                        f"🔹 Pret Intrare: `${signal.entry_price:.2f}`\n"
                        f"🔹 Stop-Loss: `${signal.stop_loss:.2f}`\n"
                        f"🔹 Take-Profit: {tp_text}\n"
                        f"🔹 ID Ordin: `{order.id}`"
                    )
                else:
                    status_text = f"⚠️ Ordinul pentru {signal.ticker} a esuat la plasare pe broker."

                edit_telegram_message(message_id, status_text)
                PENDING_SIGNALS.pop(sig_id, None)
                save_telegram_state()

            else:
                answer_callback_query(cq_id, "Semnalul a expirat sau a fost deja procesat.")
                edit_telegram_message(message_id, "⚠️ *Semnal expirat sau deja procesat.*")

        elif data.startswith("IGN:"):
            sig_id = data.split(":", 1)[1]
            signal = PENDING_SIGNALS.pop(sig_id, None)
            save_telegram_state()
            ticker_name = signal.ticker if signal else ""

            answer_callback_query(cq_id, "Semnal ignorat.")
            edit_telegram_message(message_id, f"❌ *Semnal ignorat de tine ({ticker_name}). Nu s-a plasat niciun ordin.*")
