"""
Modul de Notificari Telegram pentru Trading Bot.
Trimite alerte instant pe telefon cand se identifica semnale sau se plaseaza ordine.
Suporta Butoane Interactive (Inline Keyboard) pentru confirmare cu 1-click de pe telefon.
"""

import warnings
warnings.filterwarnings("ignore")

import time
import json
from pathlib import Path
from types import SimpleNamespace
import requests
import config

STATE_FILE = Path(__file__).parent / "data" / "telegram_state.json"

# Cache pentru semnalele in asteptare de confirmare
PENDING_SIGNALS = {}


def load_telegram_state() -> dict:
    """Incarca starea persistenta Telegram (semnale in asteptare, alerte zilnice, last_update_id)."""
    if not STATE_FILE.exists():
        return {"pending_signals": {}, "alerted_today": {}, "last_update_id": 0}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        # Reconstruim PENDING_SIGNALS in memorie daca nu sunt deja incarcate
        for sig_id, sig_dict in state.get("pending_signals", {}).items():
            if sig_id not in PENDING_SIGNALS and isinstance(sig_dict, dict):
                PENDING_SIGNALS[sig_id] = SimpleNamespace(**sig_dict)
        return state
    except Exception:
        return {"pending_signals": {}, "alerted_today": {}, "last_update_id": 0}


def save_telegram_state(alerted_today: dict = None, last_update_id: int = None) -> None:
    """Salveaza starea Telegram pe disc pentru persistenta intre rularile Cloud / GitHub Actions."""
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        current = load_telegram_state() if STATE_FILE.exists() else {
            "pending_signals": {},
            "alerted_today": {},
            "last_update_id": 0
        }
        serialized_signals = {}
        # Pastram doar semnalele din ultimele 24 ore
        cutoff = time.time() - 86400
        for sig_id, sig_obj in list(PENDING_SIGNALS.items()):
            try:
                ts = int(sig_id.rsplit("_", 1)[-1])
                if ts < cutoff:
                    PENDING_SIGNALS.pop(sig_id, None)
                    continue
            except Exception:
                pass
            if hasattr(sig_obj, "__dict__"):
                serialized_signals[sig_id] = dict(sig_obj.__dict__)
            elif isinstance(sig_obj, dict):
                serialized_signals[sig_id] = sig_obj

        current["pending_signals"] = serialized_signals
        if alerted_today is not None:
            current["alerted_today"] = alerted_today
        if last_update_id is not None:
            current["last_update_id"] = last_update_id

        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Eroare salvare telegram_state.json: {e}")


# Incarcam starea la import
load_telegram_state()


def send_telegram_message(text: str, reply_markup: dict = None) -> dict:
    """Trimite un mesaj text prin Telegram Bot API si returneaza raspunsul."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return {}

    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json().get("result", {})
        else:
            print(f"⚠️ Eroare Telegram API ({response.status_code}): {response.text}")
            return {}
    except Exception as e:
        print(f"⚠️ Eroare trimitere notificare Telegram: {e}")
        return {}


def edit_telegram_message(message_id: int, new_text: str) -> bool:
    """Modifica textul unui mesaj existent pe Telegram (ex: dupa apasarea butonului)."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return False

    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/editMessageText"
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "message_id": message_id,
        "text": new_text,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"⚠️ Eroare editare mesaj Telegram: {e}")
        return False


def answer_callback_query(callback_query_id: str, text: str = "") -> bool:
    """Trimite raspuns la apasarea unui buton inline pe Telegram."""
    if not config.TELEGRAM_BOT_TOKEN:
        return False

    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
    payload = {
        "callback_query_id": callback_query_id,
        "text": text
    }
    try:
        response = requests.post(url, json=payload, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def send_trade_alert(signal, with_buttons: bool = True, is_power_hour: bool = False) -> bool:
    """Trimite o alerta detaliata cu butoane interactive [CUMPARA] si [IGNORA]."""
    if getattr(signal, "setup_type", "") == "VOLUME_BREAKOUT":
        header = "🚀 *SEMNAL NOU: BREAKOUT DE VOLUM*"
    elif getattr(signal, "setup_type", "") == "PULLBACK_EMA20":
        header = "🎯 *SEMNAL NOU: PULLBACK EMA 20*"
    else:
        header = "🎯 *SEMNAL NOU DE CUMPARARE*"

    if is_power_hour:
        header += "\n🔥 *[POWER HOUR 22:00-23:00 — CONFIRMAT APROAPE DE CLOSE]*"

    qty_val = float(signal.shares)
    qty_str = f"{int(qty_val)}" if qty_val.is_integer() else f"{qty_val:.4f}".rstrip("0").rstrip(".")
    ml_conf = getattr(signal, "ml_confidence", 0.0)
    rs_score = getattr(signal, "rs_score", 0.0)
    comp_score = getattr(signal, "composite_score", 0.0)
    sector = getattr(signal, "sector", config.get_ticker_sector(signal.ticker))
    news_sent = getattr(signal, "news_sentiment", "⚪ NEUTRU")
    news_head = getattr(signal, "news_headline", "")

    ml_str = f"🧠 *Încredere ML:* `{ml_conf * 100:.1f}%` | *Scor Global:* `{comp_score:.0f}/100`\n" if ml_conf > 0 else ""
    rs_str = f"💪 *Putere Relativă (RS vs SPY):* `{rs_score:+.1f}%`\n"
    news_str = f"📰 *Știri (48h):* {news_sent}" + (f" — _{news_head[:85]}_\n" if news_head else "\n")

    msg = (
        f"{header}\n\n"
        f"📈 *Simbol:* `{signal.ticker}`  |  🏢 *Sector:* `{sector}`\n"
        f"💵 *Pret Intrare:* `${signal.entry_price:.2f}`\n"
        f"🛑 *Stop-Loss:* `${signal.stop_loss:.2f}`\n"
        f"🎯 *TP1 (50% Lot 1 -> Free-Roll):* `${signal.tp1:.2f}`\n"
        f"🎯 *TP2 (50% Runner + Trailing):* `${signal.tp2:.2f}`\n\n"
        f"📊 *Marime Pozitie:* `{qty_str}` actiuni (~${qty_val * signal.entry_price:,.2f})\n"
        f"{rs_str}"
        f"{ml_str}"
        f"{news_str}"
        f"🛡️ *Risc Asumat:* Max 1.5% din capital\n"
        f"📝 *Detalii:* {signal.reason}"
    )

    reply_markup = None
    if with_buttons:
        sig_id = f"{signal.ticker}_{int(time.time())}"
        PENDING_SIGNALS[sig_id] = signal
        save_telegram_state()

        reply_markup = {
            "inline_keyboard": [
                [
                    {
                        "text": f"✅ CUMPĂRĂ {qty_str}x {signal.ticker}",
                        "callback_data": f"BUY:{sig_id}"
                    },
                    {
                        "text": "❌ IGNORĂ",
                        "callback_data": f"IGN:{sig_id}"
                    }
                ]
            ]
        }

    res = send_telegram_message(msg, reply_markup=reply_markup)
    return bool(res)


def send_order_placed_alert(
    symbol: str,
    qty: int,
    limit_price: float,
    sl: float,
    tp: float,
    order_id: str,
    tp2: float = None
) -> bool:
    """Trimite confirmare dupa plasarea ordinului pe Alpaca."""
    tp_str = f"`${tp:.2f}` (Lot 1) / `${tp2:.2f}` (Lot 2 Runner)" if tp2 and qty >= 2 else f"`${tp:.2f}`"
    msg = (
        f"🚀 *ORDIN BRACKET PLASAT PE ALPACA!*\n\n"
        f"🔹 *Broker:* Alpaca Paper Trading\n"
        f"🔹 *Simbol:* `{symbol}`\n"
        f"🔹 *Cantitate:* `{qty}` actiuni\n"
        f"🔹 *Pret Limita:* `${limit_price:.2f}`\n"
        f"🔹 *Stop-Loss atasat:* `${sl:.2f}`\n"
        f"🔹 *Take-Profit atasat:* {tp_str}\n"
        f"🔹 *ID Ordin:* `{order_id}`"
    )
    res = send_telegram_message(msg)
    return bool(res)
