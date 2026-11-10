"""
Scheduler Autonom pentru Trading Bot (Full-Session Cloud & Local Edition).
Monitorizeaza bursa SUA si ruleaza automat in timpul orelor de tranzactionare:
- Scanare la fiecare 15 minute + Indicator Power Hour (22:00 - 23:00 RO)
- Verificare SL / TP1 / TP2 / Breakeven / Trailing Stop la fiecare 60 secunde
- Comenzi interactive pe Telegram (/status, /positions, /scan, /watchlist)
- Briefing Pre-Market (16:15 RO) & Raport de Inchidere Bursa (23:05 RO)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import subprocess
import time
import sys
import argparse
from datetime import datetime
import config
from data_loader import fetch_market_regime, fetch_stock_data, get_next_earnings_date
from strategy import analyze_ticker
from bot_trader import AlpacaTrader
from notifier import (
    send_telegram_message,
    send_trade_alert,
    load_telegram_state,
    save_telegram_state,
)
from telegram_listener import TelegramListener, QUICK_MENU_MARKUP

# Deduplicare zilnica a alertelor (ticker -> "YYYY-MM-DD")
ALERTED_TODAY = load_telegram_state().get("alerted_today", {})


def sync_state_to_git():
    """
    Salveaza si urca pe GitHub in timp real fisierele din data/ cand ruleaza in GitHub Actions.
    Astfel, niciun ordin sau mutare de Stop-Loss nu se pierde intre sesiuni.
    """
    if not os.getenv("GITHUB_ACTIONS"):
        return
    try:
        subprocess.run(["git", "config", "user.name", "Trading Bot Cloud"], check=False)
        subprocess.run(["git", "config", "user.email", "bot@users.noreply.github.com"], check=False)
        subprocess.run(["git", "add", "data/*.json"], check=False)
        diff = subprocess.run(["git", "diff", "--staged", "--quiet"], check=False)
        if diff.returncode != 0:
            now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            subprocess.run(["git", "commit", "-m", f"🤖 Live sync state ({now_utc}) [skip ci]"], check=False)
            subprocess.run(["git", "pull", "--rebase", "origin", "main"], check=False)
            subprocess.run(["git", "push"], check=False)
    except Exception:
        pass


def is_us_power_hour(trader: AlpacaTrader) -> bool:
    """
    Returneaza True daca ne aflam in ultima ora a bursei SUA (15:00 - 16:00 ET / 22:00 - 23:00 RO),
    cand lumanarea zilnica 1D si volumul sunt aproape finalizate.
    """
    if trader.is_connected and trader.client:
        try:
            clock = trader.client.get_clock()
            if clock.is_open and clock.timestamp.hour == 15:
                return True
        except Exception:
            pass
    return False


def run_automated_scan(trader: AlpacaTrader):
    """Executa o sesiune de scanare autonoma."""
    load_telegram_state()
    now_str = datetime.now().strftime("%H:%M:%S")
    today_str = datetime.now().strftime("%Y-%m-%d")
    power_hour = is_us_power_hour(trader)
    ph_tag = " 🔥 [POWER HOUR 15:00-16:00 ET]" if power_hour else ""
    print(f"\n[{now_str}] 🔎 Rulare scanare automata...{ph_tag}")

    # 0. Verificare automata Model ML (< 7 zile) & Watchlist Dinamic Zilnic
    try:
        from ml_predictor import ensure_ml_ready
        ensure_ml_ready(max_age_days=7)
    except Exception as e:
        print(f"[{now_str}] ⚠️ Eroare Auto-ML: {e}")

    try:
        from data_loader import get_smart_hybrid_watchlist
        wl, wl_meta = get_smart_hybrid_watchlist(force_refresh=False)
        if not wl_meta.get("from_cache"):
            print(f"[{now_str}] 🌟 Watchlist Smart Hybrid actualizat pentru azi ({len(wl)} lideri): {', '.join(wl[:8])}...")
    except Exception as e:
        print(f"[{now_str}] ⚠️ Eroare Auto-Watchlist: {e}")

    # 0.2. Sincronizare automata cu pozitiile inchise pe Alpaca (Jurnal & Cooldown)
    try:
        from risk_learner import sync_closed_trades_from_alpaca
        sync_closed_trades_from_alpaca(trader)
    except Exception as e:
        print(f"[{now_str}] ⚠️ Eroare la sincronizarea tranzactiilor Alpaca: {e}")

    # 0.5. Gestionare automata pozitii active: Auto-Breakeven, Trailing Stop & Protectie Earnings
    try:
        from position_manager import run_position_manager
        pm_actions = run_position_manager(trader)
        for act in pm_actions:
            print(f"[{now_str}] 🛡️ {act['symbol']}: {act['action']} ACTIVAT -> Stop ridicat la ${act['new_stop']:.2f} (Profit securizat: ${act['locked_profit']:+,.2f})")
    except Exception as e:
        print(f"[{now_str}] ⚠️ Eroare Position Manager: {e}")

    # 1. Verificare Regim de Piata Multi-Tier (SPY + QQQ + VIX)
    try:
        market = fetch_market_regime(config.BENCHMARK_TICKER)
        if not market["is_bullish"]:
            print(f"[{now_str}] ⚠️ Piata este BEARISH (SPY=${market['price']} sub SMA 200=${market['sma_200']}). Nu se deschid pozitii.")
            return 0
        regime = market.get("regime", "POWER_TREND")
        risk_mult = market.get("risk_multiplier", 1.0)
        max_pos_allowed = market.get("max_positions_allowed", config.MAX_OPEN_POSITIONS)
        print(f"[{now_str}] 📊 Regim Piata: {regime} | VIX: {market.get('vix', 18.0)} | Multiplicator Risc: {risk_mult*100:.0f}%")
    except Exception as e:
        print(f"[{now_str}] Eroare Regim Piata: {e}")
        return 0

    # 2. Scanare actiuni
    summary = trader.get_portfolio_summary()
    capital = summary["portfolio_value"]

    # Verificare pozitii deja deschise pentru a nu dubla acelasi ticker si pentru diversificare pe sectoare
    open_positions = trader.get_open_positions()
    open_symbols = {p["symbol"] for p in open_positions}
    sector_exposure = {}
    for p in open_positions:
        sec = config.get_ticker_sector(p["symbol"])
        sector_exposure[sec] = sector_exposure.get(sec, 0) + 1
    max_per_sector = getattr(config, "MAX_POSITIONS_PER_SECTOR", 2)

    buy_signals = []

    for ticker in config.WATCHLIST:
        if ticker in open_symbols:
            continue  # Deja avem pozitie deschisa pe acest simbol

        try:
            df = fetch_stock_data(ticker)
            signal = analyze_ticker(
                ticker=ticker,
                df=df,
                is_market_bullish=market["is_bullish"],
                total_capital=capital,
                risk_multiplier=risk_mult,
                market_regime=regime
            )

            if signal.action == "BUY":
                buy_signals.append(signal)
            elif "ML FILTER" in signal.reason or "NEWS RISK FILTER" in signal.reason:
                print(f"[{now_str}] 🛡️ {ticker}: Semnal filtrat ({signal.reason})")
        except Exception as e:
            print(f"[{now_str}] Eroare evaluare {ticker}: {e}")

    # 3. Sortam semnalele descrescator dupa Scorul Compozit (ML Confidence + RS vs SPY + News)
    if buy_signals:
        buy_signals.sort(key=lambda s: getattr(s, "composite_score", 0.0), reverse=True)

        for signal in buy_signals:
            ticker = signal.ticker
            sec = getattr(signal, "sector", config.get_ticker_sector(ticker))
            ml_conf = getattr(signal, "ml_confidence", 0.0)
            rs_val = getattr(signal, "rs_score", 0.0)
            comp_val = getattr(signal, "composite_score", 0.0)
            news_lbl = getattr(signal, "news_sentiment", "⚪ NEUTRU")
            ml_log = f" | Sector: {sec} | ML: {ml_conf*100:.1f}% | RS: {rs_val:+.1f}% | News: {news_lbl} | Scor: {comp_val:.0f}/100"
            print(f"[{now_str}] 🎯 Semnal BUY gasit: {ticker} @ ${signal.entry_price}{ml_log}")

            # Verificare Limita de Concentrare pe Sector (ex: max 2 pozitii pe SEMICONDUCTORS)
            if sector_exposure.get(sec, 0) >= max_per_sector:
                print(
                    f"[{now_str}] 🏢 SECTOR CAP: Sectorul {sec} are deja {sector_exposure[sec]}/{max_per_sector} "
                    f"pozitii deschise. Se omite {ticker} pentru protectia portofoliului."
                )
                continue

            # Verificare Portfolio Heat daca AUTO_TRADE este activ
            active_risk_count = trader.get_active_risk_positions_count()
            can_auto_execute = bool(config.AUTO_TRADE and active_risk_count < max_pos_allowed)

            # Deduplicare alerta: trimitem cel mult o alerta pe zi pentru acelasi simbol
            if ALERTED_TODAY.get(ticker) != today_str:
                send_trade_alert(signal, with_buttons=not can_auto_execute, is_power_hour=power_hour)
                ALERTED_TODAY[ticker] = today_str
                save_telegram_state(alerted_today=ALERTED_TODAY)
            else:
                print(f"[{now_str}] ℹ️ Alerta Telegram pentru {ticker} a fost deja trimisa astazi ({today_str}).")

            # Daca este activat AUTO_TRADE, plaseaza ordinul direct (in ordinea scorului si in limita Portfolio Heat + Sector Cap)
            if config.AUTO_TRADE:
                if active_risk_count >= max_pos_allowed:
                    print(
                        f"[{now_str}] 🛡️ Limita de risc pentru regimul {regime} atinsa "
                        f"({active_risk_count}/{max_pos_allowed} pozitii cu risc). Se omite {ticker}."
                    )
                    continue

                print(f"[{now_str}] 🤖 AUTO-TRADE ACTIVAT: Se plaseaza ordinul pentru {ticker} ({sec})...")
                res = trader.place_bracket_buy_order(
                    symbol=signal.ticker,
                    qty=signal.shares,
                    limit_price=signal.entry_price,
                    stop_loss_price=signal.stop_loss,
                    take_profit_price=signal.tp1,
                    tp2_price=signal.tp2,
                    setup_type=signal.setup_type
                )
                if res:
                    sector_exposure[sec] = sector_exposure.get(sec, 0) + 1
                    active_risk_count += 1
        sync_state_to_git()
        return len(buy_signals)
    else:
        print(f"[{now_str}] ℹ️ Niciun semnal nou in aceasta runda.")
        sync_state_to_git()
        return 0


def run_premarket_briefing(trader: AlpacaTrader):
    """
    Briefing Pre-Market / Deschidere Bursa SUA:
    Pregateste Watchlist-ul zilei, verifica modelul ML, verifica apropierea de Earnings
    pentru pozitiile active si trimite planul zilei pe Telegram.
    """
    global ALERTED_TODAY
    state = load_telegram_state()
    ALERTED_TODAY = state.get("alerted_today", ALERTED_TODAY)
    today_str = datetime.now().strftime("%Y-%m-%d")
    if ALERTED_TODAY.get("_PREMARKET_SENT_") == today_str:
        return

    print("\n🌅 Pregătire Briefing Pre-Market...")
    try:
        from ml_predictor import ensure_ml_ready
        ensure_ml_ready(max_age_days=7)
    except Exception:
        pass

    wl_preview = "N/A"
    try:
        from data_loader import get_smart_hybrid_watchlist
        wl, _ = get_smart_hybrid_watchlist(force_refresh=True)
        wl_preview = ", ".join(wl[:12])
    except Exception:
        pass

    regime_str = "N/A"
    try:
        m = fetch_market_regime(config.BENCHMARK_TICKER)
        regime_str = (
            f"`{m.get('regime', 'N/A')}` (SPY `${m.get('price', 0):.2f}` | "
            f"VIX `{m.get('vix', 0):.1f}` | Risc `{m.get('risk_multiplier', 1)*100:.0f}%`)"
        )
    except Exception:
        pass

    summary = trader.get_portfolio_summary()
    open_pos = trader.get_open_positions()
    earnings_warnings = []
    for p in open_pos:
        sym = p["symbol"]
        ed = get_next_earnings_date(sym)
        if ed is not None:
            days_e = (ed.date() - datetime.now().date()).days
            if 0 <= days_e <= 5:
                earnings_warnings.append(f"⚠️ `{sym}` raportează Earnings în `{days_e}` zile!")

    earn_block = ("\n" + "\n".join(earnings_warnings) + "\n") if earnings_warnings else ""

    send_telegram_message(
        f"🌅 *BRIEFING PRE-MARKET — DESCHIDEREA BURSEI SUA*\n\n"
        f"💰 *Portofoliu:* `${summary['portfolio_value']:,.2f}` (Cash: `${summary['cash']:,.2f}`)\n"
        f"📂 *Poziții Deschise:* `{len(open_pos)}` active\n"
        f"📊 *Regim Piață:* {regime_str}\n"
        f"{earn_block}"
        f"🌟 *Top Lideri Monitorizați Azi:* `{wl_preview}...`\n\n"
        f"⏱️ _Scanarea începe după Buffer-ul de Deschidere (16:45 RO), la fiecare {config.SCAN_INTERVAL_MINUTES} minute!_",
        reply_markup=QUICK_MENU_MARKUP,
    )
    ALERTED_TODAY["_PREMARKET_SENT_"] = today_str
    save_telegram_state(alerted_today=ALERTED_TODAY)
    sync_state_to_git()


def run_eod_report(trader: AlpacaTrader):
    """
    Raport de Inchidere Bursa (End-of-Day, ~23:05 RO / 16:05 ET):
    Sincronizeaza tranzactiile inchise, afiseaza P&L-ul zilei si starea pozitiilor peste noapte.
    """
    global ALERTED_TODAY
    state = load_telegram_state()
    ALERTED_TODAY = state.get("alerted_today", ALERTED_TODAY)
    today_str = datetime.now().strftime("%Y-%m-%d")
    if ALERTED_TODAY.get("_EOD_SENT_") == today_str:
        return

    print("\n🔔 Generare Raport Închidere Bursă (End-of-Day)...")
    try:
        from risk_learner import sync_closed_trades_from_alpaca
        sync_closed_trades_from_alpaca(trader)
    except Exception:
        pass

    summary = trader.get_portfolio_summary()
    open_pos = trader.get_open_positions()
    active_risk = trader.get_active_risk_positions_count()

    pos_lines = []
    total_unrealized = 0.0
    for p in open_pos:
        pl = p["unrealized_pl"]
        pl_pct = p["unrealized_plpc"]
        total_unrealized += pl
        icon = "🟢" if pl >= 0 else "🔴"
        pos_lines.append(f"  {icon} `{p['symbol']}`: `${pl:+,.2f}` (`{pl_pct:+.2f}%`)")

    pos_summary = "\n".join(pos_lines) if pos_lines else "  • _Nicio poziție deschisă peste noapte (100% Cash)_"

    send_telegram_message(
        f"🔔 *RAPORT ÎNCHIDERE BURSĂ SUA (END-OF-DAY)*\n\n"
        f"💰 *Valoare Portofoliu:* `${summary['portfolio_value']:,.2f}`\n"
        f"💵 *Cash Disponibil:* `${summary['cash']:,.2f}`\n"
        f"📈 *P&L Nerealizat Poziții:* `${total_unrealized:+,.2f}`\n"
        f"🛡️ *Poziții Peste Noapte ({len(open_pos)} | Risc Activ: {active_risk}/{config.MAX_OPEN_POSITIONS}):*\n"
        f"{pos_summary}\n\n"
        f"🌙 _Bursa s-a închis. Ne revedem mâine la 16:15 pentru Briefing-ul Pre-Market!_",
        reply_markup=QUICK_MENU_MARKUP,
    )
    ALERTED_TODAY["_EOD_SENT_"] = today_str
    save_telegram_state(alerted_today=ALERTED_TODAY)
    sync_state_to_git()


def run_weekend_maintenance(trader: AlpacaTrader):
    """
    Rutina de Weekend (Ruleaza Sâmbătă/Duminică):
    1. Sincronizeaza jurnalul de tranzactii cu Alpaca.
    2. Reantreneaza modelul ML XGBoost pe cele mai recente date.
    3. Regenereaza Watchlist-ul Smart Hybrid de lideri pentru saptamana urmatoare.
    4. Trimite raportul saptamanal pe Telegram.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("=" * 65)
    print(f"   🛠️  MENTENANȚĂ DE WEEKEND & PREGĂTIRE SĂPTĂMÂNĂ NOUĂ ({now_str})")
    print("=" * 65)

    # 1. Sincronizare pozitii inchise
    try:
        from risk_learner import sync_closed_trades_from_alpaca
        sync_closed_trades_from_alpaca(trader)
    except Exception as e:
        print(f"⚠️ Eroare sincronizare jurnal: {e}")

    # 2. Reantrenare Model ML XGBoost
    ml_info = "N/A"
    try:
        from ml_predictor import ensure_ml_ready
        pred = ensure_ml_ready(max_age_days=0)
        if pred and pred.is_ready:
            roc = pred.meta.get("cv_roc_auc_mean", 0.0)
            prec = pred.meta.get("cv_precision_mean", 0.0)
            ml_info = f"ROC-AUC `{roc:.3f}` | Precision `{prec:.3f}`"
    except Exception as e:
        print(f"⚠️ Eroare reantrenare ML: {e}")

    # 3. Actualizare Watchlist Lideri
    wl_preview = "N/A"
    try:
        from data_loader import get_smart_hybrid_watchlist
        wl, _ = get_smart_hybrid_watchlist(force_refresh=True)
        wl_preview = ", ".join(wl[:12])
    except Exception as e:
        print(f"⚠️ Eroare actualizare Watchlist: {e}")

    # 4. Evaluare Regim Piata & Portofoliu
    regime_str = "N/A"
    try:
        market = fetch_market_regime(config.BENCHMARK_TICKER)
        regime_str = f"`{market.get('regime', 'N/A')}` (SPY `${market.get('price', 0):.2f}` | VIX `{market.get('vix', 0):.1f}`)"
    except Exception:
        pass

    summary = trader.get_portfolio_summary()
    open_pos = trader.get_open_positions()

    send_telegram_message(
        f"📅 *RAPORT DE WEEKEND & PREGĂTIRE LUNI*\n\n"
        f"💰 *Valoare Portofoliu:* `${summary['portfolio_value']:,.2f}` (Cash: `${summary['cash']:,.2f}`)\n"
        f"📂 *Poziții Active:* `{len(open_pos)}` deschise\n"
        f"📊 *Regim Piață:* {regime_str}\n"
        f"🧠 *Model ML Reantrenat:* {ml_info}\n"
        f"🌟 *Top Lideri Săptămâna Viitoare:* `{wl_preview}`\n\n"
        f"✅ _Botul este optimizat și pregătit pentru deschiderea de Luni, 16:15!_",
        reply_markup=QUICK_MENU_MARKUP,
    )
    sync_state_to_git()
    print("✅ Mentenanța de weekend s-a încheiat cu succes!")


def _set_github_continue_output(should_continue: bool):
    """
    Scrie in GITHUB_OUTPUT daca sesiunea curenta trebuie continuata automat
    de urmatorul job din lant (session-1 -> session-2 -> session-3) fara a depinde de cron.
    """
    gh_out = os.getenv("GITHUB_OUTPUT")
    if gh_out:
        try:
            with open(gh_out, "a", encoding="utf-8") as f:
                val = "true" if should_continue else "false"
                f.write(f"continue_day={val}\n")
        except Exception:
            pass


def run_cloud_session(trader: AlpacaTrader, force: bool = False, max_minutes: int = 330):
    """
    Arhitectura Full-Session Chained (acopera toata ziua de tranzactionare fara goluri):
    - Rulare continua pana la 330 min (5.5h) per job, inlantuita automat (session-1 -> session-2 -> session-3)
    - Daca porneste inainte de deschidere in zi de bursa, ramane in Standby -> Pre-Market (15:45 RO) -> Live Open (16:30 RO)
    - Scanare la 15 min + SL/TP la 60s + Power Hour (22:00-23:00 RO) -> Raport EOD la 23:00 RO
    - Raspunde INSTANT la comenzile Telegram (/status, /positions, /scan, /watchlist) si butoane!
    """
    listener = TelegramListener(trader)
    listener.poll_once(wait_seconds=2)

    if force:
        listener.start()
        run_automated_scan(trader)
        print("📱 Sesiune test (--force): Ascultare comenzi si butoane Telegram timp de 90s...")
        time.sleep(90)
        listener.stop()
        sync_state_to_git()
        _set_github_continue_output(False)
        return

    # Pornim ascultatorul Telegram in timp real pe toata durata sesiunii
    listener.start()
    session_start = time.time()
    max_seconds = max_minutes * 60
    was_open_during_session = False
    standby_logged = False

    print(f"🚀 Sesiune Cloud Continua pornita (durata maxima job: {max_minutes} min | Scanare: {config.SCAN_INTERVAL_MINUTES} min | SL/TP check: 60s)")

    # Daca sesiunea a fost pornita manual ("Run workflow") sau prin push si este prima veriga din lant, trimitem confirmare pe Telegram!
    if os.getenv("GITHUB_EVENT_NAME") in ("workflow_dispatch", "push") and os.getenv("IS_CONTINUATION_SESSION") != "true":
        try:
            summary = trader.get_portfolio_summary()
            clock_info = "🟢 DESCHISĂ" if (trader.is_connected and trader.client and trader.client.get_clock().is_open) else "🌙 ÎNCHISĂ (Deschide la 16:30 RO — Mod Standby Activ)"
            send_telegram_message(
                f"🟢 *Botul a pornit în Cloud (Sesiune Activă)!*\n\n"
                f"🏛️ *Bursa SUA:* {clock_info}\n"
                f"💰 *Portofoliu:* `${summary['portfolio_value']:,.2f}` (Cash: `${summary['cash']:,.2f}`)\n"
                f"⚙️ *Auto-Trade:* `{'ACTIVAT (ON)' if config.AUTO_TRADE else 'DEZACTIVAT (Confirmare 1-Click)'}`\n\n"
                f"💬 _Ascultătorul Telegram este LIVE! Poți folosi oricând comenzile `/status`, `/positions`, `/watchlist`, `/scan` sau butoanele de mai jos:_",
                reply_markup=QUICK_MENU_MARKUP,
            )
        except Exception:
            pass

    while (time.time() - session_start) < max_seconds:
        if trader.is_connected and trader.client:
            try:
                clock = trader.client.get_clock()
                market_time = clock.timestamp
            except Exception as e:
                print(f"⚠️ Eroare citire ceas Alpaca: {e}")
                time.sleep(30)
                continue

            if not clock.is_open:
                seconds_to_open = (clock.next_open - clock.timestamp).total_seconds()
                opens_today_et = (clock.next_open.date() == market_time.date())

                # Cazul 1: Piata tocmai s-a inchis in timpul acestei sesiuni sau suntem dupa 16:00 ET (23:00 RO) -> Trimitem Raport EOD si oprim!
                if was_open_during_session or (not opens_today_et and market_time.hour in (16, 17, 18, 19, 20)):
                    run_eod_report(trader)
                    if os.getenv("GITHUB_ACTIONS"):
                        print("🔔 Sesiunea Cloud de azi s-a incheiat dupa Raportul EOD. Oprire container.")
                        time.sleep(15)
                        listener.stop()
                        sync_state_to_git()
                        _set_github_continue_output(False)
                        return

                # Cazul 2: Suntem in Pre-Market apropiat (cu pana la 60 min inainte de deschidere, de la 15:30 RO / 08:30 ET)
                if 0 < seconds_to_open <= 3600:
                    if seconds_to_open <= 2700:
                        run_premarket_briefing(trader)
                    wait_sec = min(60, int(seconds_to_open))
                    print(f"⏳ Pre-Market activ: Bursa se deschide in {int(seconds_to_open // 60)} min (Telegram Live activ)...")
                    time.sleep(wait_sec)
                    continue

                # Cazul 3: Este zi de tranzactionare SUA (bursa se deschide AZI, in mai putin de 10 ore)
                # NU inchidem sesiunea! Ramanem in Standby cu Telegram Live si intram automat in piata la 16:30 RO!
                if opens_today_et or (0 < seconds_to_open <= 36000):
                    if not standby_logged:
                        hrs = int(seconds_to_open // 3600)
                        mins = int((seconds_to_open % 3600) // 60)
                        print(
                            f"☀️ Zi de tranzactionare SUA: Bursa se deschide azi ({clock.next_open.strftime('%H:%M %Z')}, "
                            f"peste {hrs}h {mins}m). Rulare in mod STANDBY + Telegram Live pana la deschidere..."
                        )
                        standby_logged = True
                    time.sleep(60)
                    continue

                # Cazul 4: Bursa este inchisa pentru tot restul zilei (noaptea dupa 23:00 RO sau in Weekend/Sarbatori)
                if os.getenv("GITHUB_ACTIONS") and os.getenv("GITHUB_EVENT_NAME") != "workflow_dispatch":
                    print(
                        f"🌙 Bursa SUA este INCHISA (Urmatoarea deschidere: {clock.next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}). "
                        f"Ascultare rapida Telegram 15s, apoi inchidere sesiune..."
                    )
                    time.sleep(15)
                    listener.stop()
                    sync_state_to_git()
                    _set_github_continue_output(False)
                    return

                if not standby_logged:
                    print(
                        f"🌙 Bursa SUA este INCHISA (Urmatoarea deschidere: {clock.next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}). "
                        f"Rulare in mod STANDBY + Ascultare Telegram Live..."
                    )
                    standby_logged = True
                time.sleep(60)
                continue

            # Piata este DESCHISA!
            standby_logged = False
            was_open_during_session = True

            # Daca nu s-a trimis inca briefing-ul azi (chiar daca GitHub Actions a pornit cu intarziere), il trimitem imediat!
            today_str = datetime.now().strftime("%Y-%m-%d")
            if ALERTED_TODAY.get("_PREMARKET_SENT_") != today_str:
                run_premarket_briefing(trader)

            # Buffer de protectie primele 15 minute (9:30 - 9:45 ET / 16:30 - 16:45 RO)
            if market_time.hour == 9 and market_time.minute < 45:
                wait_min = 45 - market_time.minute
                print(f"⏳ Buffer de Deschidere (9:30 - 9:45 ET): Asteptare {wait_min} min (Telegram si Position Manager active)...")
                try:
                    from position_manager import run_position_manager
                    run_position_manager(trader)
                except Exception:
                    pass
                time.sleep(60)
                continue

        # Executam scanarea la fiecare 15 minute
        run_automated_scan(trader)

        # Intre scanari (15 min), verificam la fiecare 60 secunde pozitiile (SL/TP1/TP2/Breakeven/Trailing)
        sleep_seconds = config.SCAN_INTERVAL_MINUTES * 60
        check_step = 60
        elapsed = 0
        while elapsed < sleep_seconds and (time.time() - session_start) < max_seconds:
            time.sleep(check_step)
            elapsed += check_step

            if trader.is_connected and trader.client:
                try:
                    clock = trader.client.get_clock()
                    if not clock.is_open:
                        run_eod_report(trader)
                        listener.stop()
                        sync_state_to_git()
                        _set_github_continue_output(False)
                        return

                    from position_manager import run_position_manager
                    pm_actions = run_position_manager(trader)
                    if pm_actions:
                        for act in pm_actions:
                            now_pm = datetime.now().strftime("%H:%M:%S")
                            print(f"[{now_pm}] 🛡️ {act['symbol']}: {act['action']} ACTIVAT -> Stop/Exit la ${act['new_stop']:.2f}")
                        sync_state_to_git()
                except Exception:
                    pass

    # Durata maxima a jobului curent (max_minutes) a expirat -> verificam daca predam stafeta jobului urmator din lant!
    should_continue = False
    if trader.is_connected and trader.client:
        try:
            clock = trader.client.get_clock()
            sec_to_open = (clock.next_open - clock.timestamp).total_seconds()
            if clock.is_open or clock.next_open.date() == clock.timestamp.date() or (0 < sec_to_open <= 36000):
                should_continue = True
        except Exception:
            should_continue = True

    listener.stop()
    sync_state_to_git()
    _set_github_continue_output(should_continue)
    if should_continue:
        print("🔄 Timpul jobului curent s-a incheiat cu piata activa/in asteptare. Predare stafeta automata catre urmatoarea sesiune...")


def main():
    parser = argparse.ArgumentParser(description="Scheduler Autonom Trading Bot")
    parser.add_argument("--session", action="store_true", help="Ruleaza o sesiune Cloud continua (~5.5h per veriga, scan la 15m, SL/TP la 60s)")
    parser.add_argument("--once", action="store_true", help="Alias pentru sesiunea Cloud")
    parser.add_argument("--weekend", action="store_true", help="Ruleaza mentenanta si raportul de weekend")
    parser.add_argument("--premarket", action="store_true", help="Trimite Briefing-ul Pre-Market")
    parser.add_argument("--eod", action="store_true", help="Trimite Raportul de Inchidere Bursa (End-of-Day)")
    parser.add_argument("--force", action="store_true", help="Ruleaza o scanare de test chiar daca bursa este inchisa")
    parser.add_argument("--max-minutes", type=int, default=330, help="Durata maxima a sesiunii Cloud in minute (default: 330)")
    args = parser.parse_args()

    trader = AlpacaTrader()

    if args.weekend:
        run_weekend_maintenance(trader)
        return

    if args.premarket:
        run_premarket_briefing(trader)
        return

    if args.eod:
        run_eod_report(trader)
        return

    if args.session or args.once:
        run_cloud_session(trader, force=args.force, max_minutes=args.max_minutes)
        return

    print("=" * 65)
    print("      SCHEDULER AUTONOM - SWING TRADING BOT           ")
    print(f"   Auto-Trade pe Alpaca: {'🟢 ACTIVAT' if config.AUTO_TRADE else '⚪ DEZACTIVAT (Doar Alerte)'}")
    print(f"   Interval scanare: {config.SCAN_INTERVAL_MINUTES} minute")
    print("=" * 65)

    mode_text = "Tranzacționare automată activă" if config.AUTO_TRADE else "Mod Alerte + Confirmare 1-Click"
    send_telegram_message(
        f"🟢 *Maur's Trading Bot a pornit în mod AUTONOM!*\n\n"
        f"⚙️ *Regim:* {mode_text}\n"
        f"⏱️ *Interval scanare:* La fiecare {config.SCAN_INTERVAL_MINUTES} min (verificare SL/TP la 60s).\n"
        f"💬 *Comenzi active:* `/status`, `/positions`, `/scan`, `/watchlist`\n"
        f"💰 *Capital monitorizat:* `${trader.get_portfolio_summary()['portfolio_value']:,.2f}`",
        reply_markup=QUICK_MENU_MARKUP,
    )

    try:
        while True:
            run_cloud_session(trader, force=False, max_minutes=240)
            time.sleep(30)
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Scheduler oprit manual de utilizator.")
        send_telegram_message("🛑 *Maur's Trading Bot Scheduler a fost oprit.*")
        sys.exit(0)


if __name__ == "__main__":
    main()
