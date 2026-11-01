"""
Scheduler Autonom pentru Trading Bot.
Monitorizeaza bursa SUA si ruleaza automat in timpul orelor de tranzactionare.
Trimite alerte Telegram si (optional) executa ordinele automat pe Alpaca.
"""

import warnings
warnings.filterwarnings("ignore")

import time
import sys
import argparse
from datetime import datetime
import config
from data_loader import fetch_market_regime, fetch_stock_data
from strategy import analyze_ticker
from bot_trader import AlpacaTrader
from notifier import (
    send_telegram_message,
    send_trade_alert,
    load_telegram_state,
    save_telegram_state,
)
from telegram_listener import TelegramListener

# Deduplicare zilnica a alertelor (ticker -> "YYYY-MM-DD") pentru a evita spam-ul la fiecare 30 min
ALERTED_TODAY = load_telegram_state().get("alerted_today", {})


def run_automated_scan(trader: AlpacaTrader):
    """Executa o sesiune de scanare autonoma."""
    now_str = datetime.now().strftime("%H:%M:%S")
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"\n[{now_str}] 🔎 Rulare scanare automata...")

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
            return
        regime = market.get("regime", "POWER_TREND")
        risk_mult = market.get("risk_multiplier", 1.0)
        max_pos_allowed = market.get("max_positions_allowed", config.MAX_OPEN_POSITIONS)
        print(f"[{now_str}] 📊 Regim Piata: {regime} | VIX: {market.get('vix', 18.0)} | Multiplicator Risc: {risk_mult*100:.0f}%")
    except Exception as e:
        print(f"[{now_str}] Eroare Regim Piata: {e}")
        return

    # 2. Scanare actiuni
    summary = trader.get_portfolio_summary()
    capital = summary["portfolio_value"]

    # Verificare pozitii deja deschise pentru a nu dubla acelasi ticker
    open_positions = trader.get_open_positions()
    open_symbols = {p["symbol"] for p in open_positions}

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
            elif "ML FILTER" in signal.reason:
                print(f"[{now_str}] 🤖 {ticker}: Filtrul ML a respins semnalul ({signal.reason})")
        except Exception as e:
            print(f"[{now_str}] Eroare evaluare {ticker}: {e}")

    # 3. Sortam semnalele descrescator dupa Scorul Compozit (ML Confidence + RS vs SPY)
    if buy_signals:
        buy_signals.sort(key=lambda s: getattr(s, "composite_score", 0.0), reverse=True)

        for signal in buy_signals:
            ticker = signal.ticker
            ml_conf = getattr(signal, "ml_confidence", 0.0)
            rs_val = getattr(signal, "rs_score", 0.0)
            comp_val = getattr(signal, "composite_score", 0.0)
            ml_log = f" | ML: {ml_conf*100:.1f}% | RS: {rs_val:+.1f}% | Scor: {comp_val:.0f}/100"
            print(f"[{now_str}] 🎯 Semnal BUY gasit: {ticker} @ ${signal.entry_price}{ml_log}")

            # Deduplicare alerta: trimitem cel mult o alerta pe zi pentru acelasi simbol
            if ALERTED_TODAY.get(ticker) != today_str:
                send_trade_alert(signal)
                ALERTED_TODAY[ticker] = today_str
                save_telegram_state(alerted_today=ALERTED_TODAY)
            else:
                print(f"[{now_str}] ℹ️ Alerta Telegram pentru {ticker} a fost deja trimisa astazi ({today_str}).")

            # Daca este activat AUTO_TRADE, plaseaza ordinul direct (in ordinea scorului si in limita Portfolio Heat)
            if config.AUTO_TRADE:
                active_risk_count = trader.get_active_risk_positions_count()
                if active_risk_count >= max_pos_allowed:
                    print(
                        f"[{now_str}] 🛡️ Limita de risc pentru regimul {regime} atinsa "
                        f"({active_risk_count}/{max_pos_allowed} pozitii cu risc). Se omite {ticker}."
                    )
                    continue

                print(f"[{now_str}] 🤖 AUTO-TRADE ACTIVAT: Se plaseaza ordinul pentru {ticker}...")
                trader.place_bracket_buy_order(
                    symbol=signal.ticker,
                    qty=signal.shares,
                    limit_price=signal.entry_price,
                    stop_loss_price=signal.stop_loss,
                    take_profit_price=signal.tp1,
                    tp2_price=signal.tp2,
                    setup_type=signal.setup_type
                )
        return len(buy_signals)
    else:
        print(f"[{now_str}] ℹ️ Niciun semnal nou in aceasta runda.")
        return 0


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
        f"✅ _Botul este optimizat și pregătit pentru deschiderea de Luni, 16:30!_"
    )
    print("✅ Mentenanța de weekend s-a încheiat cu succes!")


def run_once_cloud(trader: AlpacaTrader, force: bool = False, active_minutes: int = 25):
    """
    Executa o sesiune Cloud de 25 de minute (ideal pentru GitHub Actions la fiecare 30 min Luni-Vineri):
    - Nu doarme intre scanari: tine ascultatorul de butoane Telegram pornit LIVE si
      verifica la fiecare 2 minute Stop-Loss / TP1 / TP2 / Breakeven / Trailing Stop!
    """
    listener = TelegramListener(trader)
    # 1. Procesam orice apasare de buton [CUMPARA]/[IGNORA] facuta pe telefon de la ultima rulare
    handled = listener.poll_once(wait_seconds=2)
    if handled > 0:
        print(f"📱 S-au procesat {handled} comenzi de pe butoanele Telegram.")

    # 2. Verificam daca bursa SUA este deschisa (daca nu este fortat manual)
    if trader.is_connected and not force:
        clock = trader.client.get_clock()
        if not clock.is_open:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] 🌙 Bursa SUA este INCHISA. Urmatoarea deschidere: {clock.next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            return

        market_time = clock.timestamp
        if market_time.hour == 9 and market_time.minute < 45:
            wait_seconds = (45 - market_time.minute) * 60
            print(f"⏳ Buffer de Deschidere activ (9:30 - 9:45 ET): Se asteapta {45 - market_time.minute} min...")
            time.sleep(min(wait_seconds, 900))

    # 3. Pornim ascultatorul LIVE de butoane Telegram in fundal pe toata durata sesiunii
    listener.start()

    # 4. Rulam scanarea automata completa + Position Manager (Breakeven / Trailing Stop / Earnings)
    run_automated_scan(trader)

    # 5. Daca a fost un test manual (--force) cu bursa inchisa, asteptam 60s pt butoane si iesim
    if force and trader.is_connected and not trader.client.get_clock().is_open:
        print("📱 Sesiune test (--force): Asteptare 60s pentru butoanele Telegram...")
        time.sleep(60)
        listener.stop()
        return

    # 6. Ramanem ACTIVI timp de ~25 minute pana la urmatorul ciclu de 30 min:
    #    - Butoanele Telegram [CUMPARA]/[IGNORA] raspund INSTANT (in 1-2 secunde)
    #    - La fiecare 2 minute verificam pozitiile active (SL / TP1 50% / TP2 50% / Trailing Stop)
    total_wait = max(0, active_minutes * 60)
    check_step = 120
    elapsed = 0
    print(f"🛡️ Monitorizare ACTIVA timp de {active_minutes} min (Butoane Telegram LIVE + verificare SL/TP la fiecare 2 min)...")

    while elapsed < total_wait:
        step = min(check_step, total_wait - elapsed)
        time.sleep(step)
        elapsed += step

        if trader.is_connected:
            try:
                clock = trader.client.get_clock()
                if not clock.is_open:
                    print("🔔 Bursa SUA s-a inchis. Se finalizeaza sesiunea curenta...")
                    break

                from position_manager import run_position_manager
                pm_actions = run_position_manager(trader)
                for act in pm_actions:
                    now_pm = datetime.now().strftime("%H:%M:%S")
                    print(f"[{now_pm}] 🛡️ {act['symbol']}: {act['action']} ACTIVAT -> Stop/Exit la ${act['new_stop']:.2f} (Profit: ${act['locked_profit']:+,.2f})")
            except Exception:
                pass

    listener.stop()


def main():
    parser = argparse.ArgumentParser(description="Scheduler Autonom Trading Bot")
    parser.add_argument("--once", action="store_true", help="Ruleaza o singura scanare (pentru Cloud / GitHub Actions)")
    parser.add_argument("--weekend", action="store_true", help="Ruleaza mentenanta si raportul de weekend")
    parser.add_argument("--force", action="store_true", help="Ruleaza scanarea chiar daca bursa este inchisa")
    args = parser.parse_args()

    trader = AlpacaTrader()

    if args.weekend:
        run_weekend_maintenance(trader)
        return

    if args.once:
        run_once_cloud(trader, force=args.force)
        return

    print("=" * 65)
    print("      SCHEDULER AUTONOM - SWING TRADING BOT           ")
    print(f"   Auto-Trade pe Alpaca: {'🟢 ACTIVAT' if config.AUTO_TRADE else '⚪ DEZACTIVAT (Doar Alerte)'}")
    print(f"   Interval scanare: {config.SCAN_INTERVAL_MINUTES} minute")
    print("=" * 65)

    # Pornire listener butoane interactive Telegram in fundal
    listener = TelegramListener(trader)
    listener.start()

    # Notificare pornire pe Telegram
    mode_text = "Tranzacționare automată activă" if config.AUTO_TRADE else "Mod Alerte pe telefon"
    send_telegram_message(
        f"🟢 *Maur's Trading Bot a pornit în mod AUTONOM!*\n\n"
        f"⚙️ *Regim:* {mode_text}\n"
        f"⏱️ *Interval:* La fiecare {config.SCAN_INTERVAL_MINUTES} min în timpul orelor de piață.\n"
        f"💰 *Capital monitorizat:* ${trader.get_portfolio_summary()['portfolio_value']:,.2f}"
    )

    try:
        while True:
            # Verificare daca piata este deschisa folosind Alpaca Clock
            is_market_open = True
            if trader.is_connected:
                clock = trader.client.get_clock()
                is_market_open = clock.is_open

                if not is_market_open:
                    now = datetime.now().strftime("%H:%M:%S")
                    print(f"[{now}] 🌙 Bursa SUA este INCHISA. Urmatoarea deschidere: {clock.next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    time.sleep(60)
                    continue

                # FILTRU DE PROTECTIE: Primele 15 minute de la deschidere (16:30 - 16:45 ora RO / 9:30 - 9:45 ET)
                # Evita gap-urile violente si caderile false de la sunetul clopotelului
                market_time = clock.timestamp
                if market_time.hour == 9 and market_time.minute < 45:
                    now = datetime.now().strftime("%H:%M:%S")
                    wait_minutes = 45 - market_time.minute
                    print(f"[{now}] ⏳ Buffer de Deschidere activ (9:30 - 9:45 ET): Se asteapta {wait_minutes} min pentru stabilizarea pietei...")
                    time.sleep(45)
                    continue

            # Piata este deschisa, rulam scanarea
            run_automated_scan(trader)

            # Pauza pana la urmatoarea scanare completa, dar cu verificare rapida la fiecare 2 min
            # a pozitiilor deschise (SL / TP1 50% / TP2 50% / Breakeven / Trailing Stop pt fractii)
            sleep_seconds = config.SCAN_INTERVAL_MINUTES * 60
            print(f"⏳ Urmatoarea scanare completa peste {config.SCAN_INTERVAL_MINUTES} minute (monitorizare SL/TP activa la fiecare 2 min)...")
            check_step = 120
            elapsed = 0
            while elapsed < sleep_seconds:
                step = min(check_step, sleep_seconds - elapsed)
                time.sleep(step)
                elapsed += step
                if elapsed < sleep_seconds and trader.is_connected:
                    try:
                        from position_manager import run_position_manager
                        pm_actions = run_position_manager(trader)
                        for act in pm_actions:
                            now_pm = datetime.now().strftime("%H:%M:%S")
                            print(f"[{now_pm}] 🛡️ {act['symbol']}: {act['action']} ACTIVAT -> Stop/Exit la ${act['new_stop']:.2f} (Profit: ${act['locked_profit']:+,.2f})")
                    except Exception:
                        pass

    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Scheduler oprit manual de utilizator.")
        send_telegram_message("🛑 *Maur's Trading Bot Scheduler a fost oprit.*")
        sys.exit(0)


if __name__ == "__main__":
    main()
