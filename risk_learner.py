"""
Modulul de Invatare din Erori si Management Adaptiv al Riscului.
Functii principale:
1. Jurnal automat de tranzactii (Trade Journal).
2. Sistem de Cooldown (Carantina temporara): blocheaza automat orice ticker
   care inregistreaza 2 pierderi (Stop-Loss) consecutive.
3. Analiza statistica a performantei pe fiecare tip de setup.
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from tabulate import tabulate
import config

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

JOURNAL_FILE = DATA_DIR / "trade_journal.json"
COOLDOWN_FILE = DATA_DIR / "cooldown.json"
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


def is_in_cooldown(ticker: str) -> Tuple[bool, str]:
    """
    Verifica daca un ticker se afla in perioada de Cooldown (carantina).
    Daca perioada de pauza a expirat, il elibereaza automat.
    """
    cooldowns = _load_json(COOLDOWN_FILE, {})
    if ticker not in cooldowns:
        return False, ""

    until_str = cooldowns[ticker].get("until", "")
    try:
        until_dt = datetime.strptime(until_str, "%Y-%m-%d %H:%M:%S")
        if datetime.now() < until_dt:
            days_left = (until_dt - datetime.now()).days
            hours_left = int((until_dt - datetime.now()).seconds / 3600)
            return True, f"In Cooldown pana la {until_str} ({days_left}z {hours_left}h ramase dupa 2 Stop-Loss-uri)"
        else:
            # Perioada a expirat, curatam lista
            cooldowns.pop(ticker, None)
            _save_json(COOLDOWN_FILE, cooldowns)
            return False, ""
    except Exception:
        return False, ""


def add_to_cooldown(ticker: str, days: int = 7, reason: str = "2 Stop-Loss-uri consecutive"):
    """Pune un ticker in carantina pentru un numar de zile."""
    cooldowns = _load_json(COOLDOWN_FILE, {})
    until_dt = datetime.now() + timedelta(days=days)
    cooldowns[ticker] = {
        "ticker": ticker,
        "since": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "until": until_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": reason
    }
    _save_json(COOLDOWN_FILE, cooldowns)

    # Trimite alerta pe Telegram
    try:
        from notifier import send_telegram_message
        send_telegram_message(
            f"🛡️ *AUTO-PROTECȚIE (LEARNING SYSTEM)*\n\n"
            f"Simbolul `{ticker}` a inregistrat 2 pierderi consecutive la Stop-Loss.\n\n"
            f"⚠️ *Decizie:* Tickerul a fost plasat in *COOLDOWN (Carantina)* timp de {days} zile (pana la {until_dt.strftime('%d.%m.%Y')}).\n"
            f"Botul nu va mai cumpara {ticker} pana cand piata nu se stabilizeaza."
        )
    except Exception:
        pass


def record_closed_trade(
    ticker: str,
    setup_type: str,
    entry_price: float,
    exit_price: float,
    shares: float,
    pnl: float,
    exit_type: str  # "STOP_LOSS", "TP1", "TP2"
):
    """Inregistreaza o tranzactie finalizata in jurnalul de invatare."""
    journal = _load_json(JOURNAL_FILE, [])

    trade_entry = {
        "id": f"TR_{len(journal)+1:04d}",
        "ticker": ticker,
        "setup_type": setup_type,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "shares": shares,
        "pnl": round(pnl, 2),
        "pnl_pct": round(((exit_price - entry_price) / entry_price) * 100, 2),
        "result": "WIN" if pnl > 0 else "LOSS",
        "exit_type": exit_type,
        "closed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    journal.append(trade_entry)
    _save_json(JOURNAL_FILE, journal)

    # Dupa fiecare trade inregistrat, verificam daca trebuie activat Cooldown
    _evaluate_ticker_mistakes(ticker, journal)


def _evaluate_ticker_mistakes(ticker: str, journal: List[Dict]):
    """
    Analizeaza ultimele tranzactii pe ticker.
    Daca ultimele 2 pozitii distincte au fost ambele LOSS (Stop-Loss), il baga in Cooldown.
    Grupeaza loturile Split-Order dupa data inchiderii/pretul de intrare pentru a nu numara
    cele 2 jumatati ale aceleiasi pozitii ca 2 pierderi separate.
    """
    ticker_trades = [t for t in journal if t["ticker"] == ticker]
    if len(ticker_trades) < 2:
        return

    # Deduplicam loturile din aceeasi pozitie (acelasi entry_price si aceeasi zi de inchidere)
    distinct_positions = []
    for t in ticker_trades:
        closed_day = str(t.get("closed_at", ""))[:10]
        pos_key = (t.get("entry_price"), closed_day)
        if distinct_positions and distinct_positions[-1]["key"] == pos_key:
            distinct_positions[-1]["pnl"] += t.get("pnl", 0.0)
            distinct_positions[-1]["result"] = "WIN" if distinct_positions[-1]["pnl"] > 0 else "LOSS"
        else:
            distinct_positions.append({
                "key": pos_key,
                "pnl": t.get("pnl", 0.0),
                "result": t.get("result", "LOSS")
            })

    if len(distinct_positions) >= 2:
        last_two = distinct_positions[-2:]
        if last_two[0]["result"] == "LOSS" and last_two[1]["result"] == "LOSS":
            add_to_cooldown(ticker, days=7, reason="2 Stop-Loss-uri consecutive identificate de Jurnal")


def get_learning_insights() -> Dict:
    """
    Genereaza concluzii statistice si lectii invatate din istoricul de tranzactii.
    """
    journal = _load_json(JOURNAL_FILE, [])
    cooldowns = _load_json(COOLDOWN_FILE, {})

    if not journal:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "setups": {},
            "cooldown_list": list(cooldowns.keys()),
            "insight": "Jurnalul este gol. Pe masura ce se inchid pozitii pe Alpaca, botul va colecta date."
        }

    total_trades = len(journal)
    wins = [t for t in journal if t["result"] == "WIN"]
    losses = [t for t in journal if t["result"] == "LOSS"]
    win_rate = round((len(wins) / total_trades) * 100, 1)
    total_pnl = round(sum(t["pnl"] for t in journal), 2)

    # Performanta pe fiecare setup in parte
    setups_perf = {}
    for t in journal:
        s = t.get("setup_type", "GENERAL")
        if s not in setups_perf:
            setups_perf[s] = {"wins": 0, "losses": 0, "pnl": 0.0}
        if t["result"] == "WIN":
            setups_perf[s]["wins"] += 1
        else:
            setups_perf[s]["losses"] += 1
        setups_perf[s]["pnl"] += t["pnl"]

    # Lectii invatate
    best_setup = max(setups_perf.items(), key=lambda x: x[1]["pnl"])[0] if setups_perf else "N/A"

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "setups": setups_perf,
        "cooldown_list": list(cooldowns.keys()),
        "insight": f"Setup-ul cu cel mai mare randament net este: {best_setup}"
    }


def sync_closed_trades_from_alpaca(trader=None) -> List[Dict]:
    """
    Sincronizeaza automat tranzactiile inchise de pe contul Alpaca Paper
    cu Jurnalul local (trade_journal.json).
    Detecteaza Stop-Loss-urile si Take-Profit-urile executate, calculeaza
    PnL-ul realizat si evalueaza aplicarea Cooldown-ului (carantina).
    """
    if trader is None:
        try:
            from bot_trader import AlpacaTrader
            trader = AlpacaTrader()
        except Exception:
            return []

    if not getattr(trader, "is_connected", False) or not getattr(trader, "client", None):
        return []

    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
    except ImportError:
        return []

    try:
        req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=100)
        orders = trader.client.get_orders(req)
    except Exception as e:
        print(f"⚠️ Eroare la interogarea ordinelor Alpaca: {e}")
        return []

    journal = _load_json(JOURNAL_FILE, [])
    pos_state = _load_json(POSITIONS_STATE_FILE, {})
    synced_order_ids = {str(t.get("alpaca_order_id")) for t in journal if "alpaca_order_id" in t}

    filled_sells = [o for o in orders if getattr(o.side, "name", str(o.side)).lower() == "sell" and getattr(o.status, "name", str(o.status)).lower() == "filled"]
    filled_buys = [o for o in orders if getattr(o.side, "name", str(o.side)).lower() == "buy" and getattr(o.status, "name", str(o.status)).lower() == "filled"]

    new_synced = []

    for s in filled_sells:
        order_id = str(s.id)
        if order_id in synced_order_ids:
            continue

        # Gasim ordinul de cumparare corespondent (cel mai recent inainte de vanzare)
        matching_buys = [b for b in filled_buys if b.symbol == s.symbol and b.filled_at < s.filled_at]
        if not matching_buys:
            continue

        matching_buys.sort(key=lambda x: x.filled_at, reverse=True)
        b = matching_buys[0]

        entry_price = float(b.filled_avg_price)
        exit_price = float(s.filled_avg_price)
        shares = float(s.qty)
        pnl = round((exit_price - entry_price) * shares, 2)
        pnl_pct = round(((exit_price - entry_price) / entry_price) * 100, 2)

        order_type_str = getattr(s.order_type, "value", str(s.order_type)).lower()
        if "stop" in order_type_str:
            exit_type = "STOP_LOSS"
        elif "limit" in order_type_str:
            exit_type = "TAKE_PROFIT"
        else:
            exit_type = "TAKE_PROFIT" if pnl > 0 else "STOP_LOSS"

        closed_at_str = s.filled_at.strftime("%Y-%m-%d %H:%M:%S") if hasattr(s.filled_at, "strftime") else str(s.filled_at)
        saved_setup = pos_state.get(s.symbol, {}).get("setup_type", "PULLBACK_EMA20")

        trade_entry = {
            "id": f"TR_{len(journal)+1:04d}",
            "alpaca_order_id": order_id,
            "ticker": s.symbol,
            "setup_type": saved_setup,
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "shares": shares,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "result": "WIN" if pnl > 0 else "LOSS",
            "exit_type": exit_type,
            "closed_at": closed_at_str
        }

        journal.append(trade_entry)
        synced_order_ids.add(order_id)
        new_synced.append(trade_entry)

        # Evaluare pentru Cooldown (daca a inregistrat 2 Stop-Loss-uri)
        _evaluate_ticker_mistakes(s.symbol, journal)

        # Notificare Telegram pentru tranzactie finalizata
        try:
            from notifier import send_telegram_message
            res_icon = "🟢 CÂȘTIG" if trade_entry["result"] == "WIN" else "🔴 PIERDERE"
            send_telegram_message(
                f"📥 *TRANZACȚIE FINALIZATĂ PE ALPACA*\n\n"
                f"Simbol: `{trade_entry['ticker']}` | {res_icon}\n"
                f"• Ieșire: *{trade_entry['exit_type']}* la ${trade_entry['exit_price']:.2f}\n"
                f"• Intrare: ${trade_entry['entry_price']:.2f} ({trade_entry['shares']} acțiuni)\n"
                f"• PnL Realizat: *${trade_entry['pnl']:+,.2f}* ({trade_entry['pnl_pct']:+.2f}%)\n"
                f"• Data: {trade_entry['closed_at']}"
            )
        except Exception:
            pass

    if new_synced:
        _save_json(JOURNAL_FILE, journal)
        print(f"📥 S-au sincronizat {len(new_synced)} tranzactii inchise din contul Alpaca.")

    return new_synced


def print_learning_report(trader=None):
    """Afiseaza un raport frumos formatat in consola si sincronizeaza datele."""
    # Sincronizare live cu contul Alpaca
    sync_closed_trades_from_alpaca(trader)

    stats = get_learning_insights()
    journal = _load_json(JOURNAL_FILE, [])

    print("\n" + "=" * 65)
    print("      📊 JURNAL DE TRANZACTII & SISTEM DE INVATARE      ")
    print("=" * 65)
    print(f"  • Total Tranzactii Inregistrate: {stats['total_trades']}")
    print(f"  • Win Rate Global: {stats['win_rate']}%")
    print(f"  • PnL Total Realizat: ${stats['total_pnl']:+,.2f}")
    print(f"  • Simboluri in Cooldown (Carantina): {stats['cooldown_list'] if stats['cooldown_list'] else 'Niciunul'}")
    print(f"  • Concluzie / Lectie: {stats['insight']}")

    if journal:
        print("\n📜 Istoricul Ultimelor Tranzactii Sincronizate:")
        table = []
        for t in journal[-8:]:
            res = "🟢 WIN" if t["result"] == "WIN" else "🔴 LOSS"
            table.append([
                t["id"], t["ticker"], f"${t['entry_price']:.2f}",
                f"${t['exit_price']:.2f}", t["exit_type"], f"${t['pnl']:+,.2f}",
                f"{t['pnl_pct']:+.2f}%", res, t.get("closed_at", "N/A")
            ])
        headers = ["ID", "Simbol", "Intrare", "Iesire", "Tip Iesire", "PnL ($)", "PnL (%)", "Rezultat", "Data"]
        print(tabulate(table, headers=headers, tablefmt="grid"))
    print("=" * 65)
