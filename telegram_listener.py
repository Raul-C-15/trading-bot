"""
Listener pentru butoanele interactive de pe Telegram.
Asculta in fundal evenimentele de tip callback_query cand utilizatorul
apasa pe butonul [✅ CUMPARA] sau [❌ IGNORA] pe telefon.
"""

import warnings
warnings.filterwarnings("ignore")

import threading
import time
import requests
import config
from notifier import (
    PENDING_SIGNALS,
    edit_telegram_message,
    answer_callback_query,
    load_telegram_state,
    save_telegram_state,
)


class TelegramListener:
    def __init__(self, trader):
        self.trader = trader
        self.running = False
        state = load_telegram_state()
        self.last_update_id = int(state.get("last_update_id", 0))
        self.thread = None

    def start(self):
        """Porneste listenerul intr-un thread separat in fundal."""
        if not config.TELEGRAM_BOT_TOKEN:
            print("ℹ️ Telegram Token lipseste; ascultatorul de butoane nu va porni.")
            return

        self.running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()
        print("📱 Ascultator de butoane Telegram ACTIVAT in fundal!")

    def stop(self):
        """Opreste bucla de ascultare."""
        self.running = False

    def poll_once(self, wait_seconds: int = 2) -> int:
        """
        Verifica si proceseaza toate apasarile de butoane Telegram acumulate in coada.
        Ideal pentru rularea periodica in Cloud (GitHub Actions --once).
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
                    save_telegram_state(last_update_id=self.last_update_id)

            except Exception:
                time.sleep(2)

    def _handle_callback(self, cq):
        """Proceseaza apasarea unui buton."""
        cq_id = cq.get("id")
        data = cq.get("data", "")
        message = cq.get("message", {})
        message_id = message.get("message_id")

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

            else:
                answer_callback_query(cq_id, "Semnalul a expirat sau a fost deja procesat.")
                edit_telegram_message(message_id, "⚠️ *Semnal expirat sau deja procesat.*")

        elif data.startswith("IGN:"):
            sig_id = data.split(":", 1)[1]
            signal = PENDING_SIGNALS.pop(sig_id, None)
            ticker_name = signal.ticker if signal else ""
            
            answer_callback_query(cq_id, f"Semnal ignorat.")
            edit_telegram_message(message_id, f"❌ *Semnal ignorat de tine ({ticker_name}). Nu s-a plasat niciun ordin.*")
