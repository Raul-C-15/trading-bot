"""
Modulul de Jurnalizare, Statistici de Risc și Management Adaptiv al Erorilor (Risk Learner V2).
Funcții principale:
  1. Jurnal automat de tranzacții sincronizat rezilient cu Alpaca.
  2. Clasificare fidelă a ieșirilor (`TAKE_PROFIT`, `TRAILING_STOP`, `BREAKEVEN_EXIT`, `STOP_LOSS`),
     fără a distorsiona Win Rate-ul sau Cooldown-ul cu ieșiri neutre la Breakeven (±0.5%).
  3. Sistem de Cooldown (Carantină 7 zile) după 2 pierderi reale consecutive pe același simbol.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from tabulate import tabulate

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
    """Verifică dacă un simbol se află în perioada de Cooldown (carantină)."""
    cooldowns = _load_json(COOLDOWN_FILE, {})
    if ticker not in cooldowns:
        return False, ""

    until_str = cooldowns[ticker].get("until", "")
    try:
        until_dt = datetime.strptime(until_str, "%Y-%m-%d %H:%M:%S")
        if datetime.now() < until_dt:
            days_left = (until_dt - datetime.now()).days
            hours_left = int((until_dt - datetime.now()).seconds / 3600)
            return True, f"În Cooldown până la {until_str} ({days_left}z {hours_left}h rămase)"
        cooldowns.pop(ticker, None)
        _save_json(COOLDOWN_FILE, cooldowns)
        return False, ""
    except Exception:
        return False, ""


def add_to_cooldown(ticker: str, days: int = 7, reason: str = "2 Stop-Loss-uri reale consecutive"):
    """Plasează un simbol în carantină temporară."""
    cooldowns = _load_json(COOLDOWN_FILE, {})
    until_dt = datetime.now() + timedelta(days=days)
    cooldowns[ticker] = {
        "ticker": ticker,
        "since": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "until": until_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": reason
    }
    _save_json(COOLDOWN_FILE, cooldowns)

    try:
        from notifier import send_telegram_message
        send_telegram_message(
            f"🛡️ *AUTO-PROTECȚIE (COOLDOWN ACTIVAT)*\n\n"
            f"Simbolul `{ticker}` a înregistrat 2 pierderi reale consecutive la Stop-Loss.\n"
            f"⚠️ Blocat temporar timp de {days} zile (până la {until_dt.strftime('%d.%m.%Y')})."
        )
    except Exception:
        pass


def _classify_exit(order_obj, pnl_pct: float) -> Tuple[str, str]:
    """
    Clasifică fidel tipul ieșirii și rezultatul (`WIN`, `BREAKEVEN`, `LOSS`):
      - Ieșirile cu amplitudine mică (-0.50% .. +0.55%) sunt `BREAKEVEN_EXIT`,
        nu false `TAKE_PROFIT` și nici `STOP_LOSS` penalizatoare.
    """
    cid = str(getattr(order_obj, "client_order_id", "") or "").upper()
    order_type_str = getattr(order_obj.order_type, "value", str(order_obj.order_type)).lower()

    if "_TP1" in cid or "_TP2" in cid:
        return "TAKE_PROFIT", ("WIN" if pnl_pct > 0 else "LOSS")
    if -0.50 <= pnl_pct <= 0.55:
        return "BREAKEVEN_EXIT", ("WIN" if pnl_pct >= 0 else "LOSS")
    if pnl_pct > 0.55:
        exit_lbl = "TRAILING_STOP" if "stop" in order_type_str else "TAKE_PROFIT"
        return exit_lbl, "WIN"
    return "STOP_LOSS", "LOSS"


def _evaluate_ticker_mistakes(ticker: str, journal: List[Dict]):
    """
    Activează Cooldown doar dacă ultimele 2 poziții distincte au fost pierderi REALE
    (pnl_pct <= -1.0%, ignorând ieșirile neutre la Breakeven).
    """
    ticker_trades = [t for t in journal if t["ticker"] == ticker]
    if len(ticker_trades) < 2:
        return

    distinct_positions = []
    for t in ticker_trades:
        closed_day = str(t.get("closed_at", ""))[:10]
        pos_key = (t.get("entry_price"), closed_day)
        if distinct_positions and distinct_positions[-1]["key"] == pos_key:
            distinct_positions[-1]["pnl"] += float(t.get("pnl", 0.0))
            distinct_positions[-1]["pnl_pct"] = (
                distinct_positions[-1]["pnl_pct"] + float(t.get("pnl_pct", 0.0))
            ) / 2.0
        else:
            distinct_positions.append({
                "key": pos_key,
                "pnl": float(t.get("pnl", 0.0)),
                "pnl_pct": float(t.get("pnl_pct", 0.0)),
            })

    meaningful = [p for p in distinct_positions if abs(p["pnl_pct"]) > 0.55]
    if len(meaningful) >= 2:
        last_two = meaningful[-2:]
        if last_two[0]["pnl_pct"] <= -1.0 and last_two[1]["pnl_pct"] <= -1.0:
            add_to_cooldown(ticker, days=7, reason="2 Stop-Loss-uri reale consecutive")


def record_closed_trade(
    ticker: str,
    setup_type: str,
    entry_price: float,
    exit_price: float,
    shares: float,
    pnl: float,
    exit_type: str
):
    """Înregistrează manual/local o tranzacție închisă în jurnal."""
    journal = _load_json(JOURNAL_FILE, [])
    pnl_pct = round(((exit_price - entry_price) / entry_price) * 100.0, 2) if entry_price > 0 else 0.0

    trade_entry = {
        "id": f"TR_{len(journal)+1:04d}",
        "ticker": ticker,
        "setup_type": setup_type,
        "entry_price": round(entry_price, 2),
        "exit_price": round(exit_price, 2),
        "shares": shares,
        "pnl": round(pnl, 2),
        "pnl_pct": pnl_pct,
        "result": "WIN" if pnl > 0 else "LOSS",
        "exit_type": exit_type,
        "closed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    journal.append(trade_entry)
    _save_json(JOURNAL_FILE, journal)
    _evaluate_ticker_mistakes(ticker, journal)


def get_learning_insights() -> Dict:
    """Generează statistici complete de risc și performanță din jurnal."""
    journal = _load_json(JOURNAL_FILE, [])
    cooldowns = _load_json(COOLDOWN_FILE, {})

    if not journal:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_pnl": 0.0,
            "setups": {},
            "cooldown_list": list(cooldowns.keys()),
            "insight": "Jurnalul este gol."
        }

    total_trades = len(journal)
    wins = [t for t in journal if t["pnl"] > 0]
    losses = [t for t in journal if t["pnl"] < 0]
    win_rate = round((len(wins) / total_trades) * 100.0, 1)
    total_pnl = round(sum(t["pnl"] for t in journal), 2)
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = round(gp / gl, 2) if gl > 0 else (99.0 if gp > 0 else 0.0)

    setups_perf = {}
    for t in journal:
        s = t.get("setup_type", "GENERAL")
        if s not in setups_perf:
            setups_perf[s] = {"wins": 0, "losses": 0, "pnl": 0.0}
        if t["pnl"] > 0:
            setups_perf[s]["wins"] += 1
        else:
            setups_perf[s]["losses"] += 1
        setups_perf[s]["pnl"] += t["pnl"]

    best_setup = max(setups_perf.items(), key=lambda x: x[1]["pnl"])[0] if setups_perf else "N/A"
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": pf,
        "total_pnl": total_pnl,
        "setups": setups_perf,
        "cooldown_list": list(cooldowns.keys()),
        "insight": f"Profit Factor global: {pf:.2f} | Cel mai performant setup: {best_setup}"
    }


def sync_closed_trades_from_alpaca(trader=None) -> List[Dict]:
    """Sincronizează tranzacțiile închise de pe Alpaca cu `trade_journal.json`."""
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
        orders = trader._api_call(trader.client.get_orders, req)
    except Exception as e:
        print(f"⚠️ Eroare la interogarea ordinelor închise Alpaca: {e}")
        return []

    journal = _load_json(JOURNAL_FILE, [])
    pos_state = _load_json(POSITIONS_STATE_FILE, {})
    synced_order_ids = {str(t.get("alpaca_order_id")) for t in journal if "alpaca_order_id" in t}

    filled_sells = [
        o for o in orders
        if getattr(o.side, "name", str(o.side)).lower() == "sell"
        and getattr(o.status, "name", str(o.status)).lower() == "filled"
    ]
    filled_buys = [
        o for o in orders
        if getattr(o.side, "name", str(o.side)).lower() == "buy"
        and getattr(o.status, "name", str(o.status)).lower() == "filled"
    ]

    new_synced = []

    for s in sorted(filled_sells, key=lambda x: x.filled_at or datetime.min):
        order_id = str(s.id)
        if order_id in synced_order_ids:
            continue

        matching_buys = [b for b in filled_buys if b.symbol == s.symbol and b.filled_at and s.filled_at and b.filled_at < s.filled_at]
        if not matching_buys:
            continue

        matching_buys.sort(key=lambda x: x.filled_at, reverse=True)
        b = matching_buys[0]

        entry_price = float(b.filled_avg_price)
        exit_price = float(s.filled_avg_price)
        shares = float(s.filled_qty or s.qty)
        pnl = round((exit_price - entry_price) * shares, 2)
        pnl_pct = round(((exit_price - entry_price) / entry_price) * 100.0, 2) if entry_price > 0 else 0.0

        exit_type, result_lbl = _classify_exit(s, pnl_pct)
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
            "result": result_lbl,
            "exit_type": exit_type,
            "closed_at": closed_at_str
        }

        journal.append(trade_entry)
        synced_order_ids.add(order_id)
        new_synced.append(trade_entry)

        _evaluate_ticker_mistakes(s.symbol, journal)

        try:
            from notifier import send_telegram_message
            res_icon = "🟢 CÂȘTIG" if pnl > 0 else "🔴 PIERDERE"
            send_telegram_message(
                f"📥 *TRANZACȚIE FINALIZATĂ PE ALPACA*\n\n"
                f"Simbol: `{trade_entry['ticker']}` | {res_icon}\n"
                f"• Ieșire: *{trade_entry['exit_type']}* la `${trade_entry['exit_price']:.2f}`\n"
                f"• Intrare: `${trade_entry['entry_price']:.2f}` (`{trade_entry['shares']}` acțiuni)\n"
                f"• PnL Realizat: *${trade_entry['pnl']:+,.2f}* (`{trade_entry['pnl_pct']:+.2f}%`)\n"
                f"• Data: {trade_entry['closed_at']}"
            )
        except Exception:
            pass

    if new_synced:
        _save_json(JOURNAL_FILE, journal)
        print(f"📥 S-au sincronizat {len(new_synced)} tranzacții închise din contul Alpaca.")

    return new_synced


def print_learning_report(trader=None):
    """Afișează raportul jurnalului de tranzacții cu metrici complete de risc."""
    sync_closed_trades_from_alpaca(trader)
    stats = get_learning_insights()
    journal = _load_json(JOURNAL_FILE, [])

    print("\n" + "=" * 70)
    print("      📊 JURNAL DE TRANZACȚII & METRICI DE PERFORMANȚĂ LIVE      ")
    print("=" * 70)
    print(f"  • Total Execuții Înregistrate : {stats['total_trades']}")
    print(f"  • Win Rate Global             : {stats['win_rate']}%")
    print(f"  • Profit Factor Global        : {stats['profit_factor']:.2f}")
    print(f"  • PnL Total Realizat          : ${stats['total_pnl']:+,.2f}")
    print(f"  • Simboluri în Cooldown       : {stats['cooldown_list'] if stats['cooldown_list'] else 'Niciunul'}")
    print(f"  • Sinteză                     : {stats['insight']}")

    if journal:
        print("\n📜 Ultimele Tranzacții Sincronizate:")
        table = []
        for t in journal[-10:]:
            res = "🟢 WIN" if t["pnl"] > 0 else "🔴 LOSS"
            table.append([
                t["id"], t["ticker"], f"${t['entry_price']:.2f}",
                f"${t['exit_price']:.2f}", t["exit_type"], f"${t['pnl']:+,.2f}",
                f"{t['pnl_pct']:+.2f}%", res, t.get("closed_at", "N/A")
            ])
        headers = ["ID", "Simbol", "Intrare", "Ieșire", "Tip Ieșire", "PnL ($)", "PnL (%)", "Rezultat", "Data"]
        print(tabulate(table, headers=headers, tablefmt="grid"))
    print("=" * 70)
