"""
Punctul principal de intrare pentru Trading Bot (Smart Cockpit Edition).
Include un Mini-Dashboard automat la pornire și 4 comenzi principale automatizate:
  1. 🚀 Scanare Smart & Execuție (All-in-One: Auto-ML + Protecție Poziții + Lideri + Semnale)
  2. 🤖 Autopilot 24/7 (Scheduler Autonom în Fundal + Telegram)
  3. 📊 Backtest & Laborator Vizual în Browser (TradingView Local + Simulare Portofoliu)
  4. 🛠️ Unelte Avansate & Mentenanță (Jurnal, Optimizare Walk-Forward, Screener, ML)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from datetime import datetime
import pandas as pd
from tabulate import tabulate

import config
from data_loader import (
    fetch_market_regime,
    fetch_stock_data,
    screen_top_rs_leaders,
    get_next_earnings_date,
    get_smart_hybrid_watchlist,
    DAILY_WATCHLIST_CACHE_FILE,
)
from strategy import analyze_ticker
from backtester import Backtester
from bot_trader import AlpacaTrader


def print_cockpit_dashboard(trader: AlpacaTrader):
    """
    Afișează un Mini-Dashboard instantaneu cu starea contului, pozițiile active,
    modelul ML și Watchlist-ul Hibrid fără ca utilizatorul să apese vreun buton.
    """
    summary = trader.get_portfolio_summary()
    mode_str = "🟢 Alpaca Paper Trading" if summary.get("connected") else "🟡 Simulare Locală"
    port_val = summary.get("portfolio_value", config.DEFAULT_SIMULATED_CAPITAL)
    cash_val = summary.get("cash", port_val)

    # Status ML
    from ml_predictor import get_predictor
    pred = get_predictor()
    if pred.is_ready:
        trained_at = pred.meta.get("trained_at", "N/A")[:10]
        prec = pred.meta.get("cv_precision_mean", 0.0) * 100
        ml_status = f"🟢 Activ (Precision: {prec:.1f}%, antrenat {trained_at})"
    else:
        ml_status = "🟡 Neantrenat (se va antrena automat la scanare)"

    # Status Watchlist Zilnic
    today_str = datetime.now().strftime("%Y-%m-%d")
    wl_status = f"⚪ Standard ({len(config.WATCHLIST)} simboluri — se actualizează automat la scanare)"
    if DAILY_WATCHLIST_CACHE_FILE.exists():
        try:
            import json
            wl_meta = json.loads(DAILY_WATCHLIST_CACHE_FILE.read_text(encoding="utf-8"))
            if wl_meta.get("date") == today_str:
                wl_list = wl_meta.get("watchlist", config.WATCHLIST)
                config.WATCHLIST = wl_list
                top_str = ", ".join(wl_meta.get("top_5_rs", [])[:3])
                wl_status = f"🟢 Smart Hybrid Activ azi ({len(wl_list)} lideri | Top RS: {top_str})"
        except Exception:
            pass

    print("\n" + "=" * 78)
    print("             🚀 SWING TRADING BOT — SMART INSTITUTIONAL COCKPIT             ")
    print("=" * 78)
    print(f"  💼 Conexiune   : {mode_str} | Capital: ${port_val:,.2f} (Cash: ${cash_val:,.2f})")
    print(f"  🧠 Model ML    : {ml_status}")
    print(f"  🌟 Watchlist   : {wl_status}")

    if summary.get("connected"):
        try:
            positions = trader.get_open_positions()
            active_risk = trader.get_active_risk_positions_count()
            print(f"  📈 Poziții     : {len(positions)} deschise ({active_risk}/{config.MAX_OPEN_POSITIONS} cu risc activ)")
        except Exception:
            pass
    print("=" * 78)


def run_smart_scan(trader: AlpacaTrader):
    """
    OPȚIUNEA 1 (ALL-IN-ONE SMART SCAN):
      Pas 1: Verifică/antrenează automat modelul ML dacă e mai vechi de 7 zile.
      Pas 2: Sincronizează jurnalul Alpaca și protejează pozițiile deschise (Auto-Breakeven / Trail).
      Pas 3: Actualizează automat (o dată pe zi) Watchlist-ul Smart Hybrid (Core + Top RS Stage 2).
      Pas 4: Evaluează Regimul Pieței (SPY + QQQ + VIX).
      Pas 5: Scanează liderii, filtrează pozițiile deja deținute, ordonează semnalele după Scorul
             Compozit (ML + RS) și propune spre execuție exact câte poziții mai încap în portofoliu.
    """
    print("\n⚡ [PAS 1/5] Verificare automată Model ML (XGBoost)...")
    from ml_predictor import ensure_ml_ready
    ensure_ml_ready(max_age_days=7)

    print("\n🛡️ [PAS 2/5] Sincronizare Jurnal & Protecție Poziții Deschise (Auto-Breakeven / Trail)...")
    open_symbols = set()
    active_risk_count = 0
    if trader.is_connected:
        try:
            from risk_learner import sync_closed_trades_from_alpaca
            sync_closed_trades_from_alpaca(trader)
        except Exception as e:
            print(f"   ⚠️ Avertisment sincronizare jurnal: {e}")

        try:
            from position_manager import run_position_manager
            actions = run_position_manager(trader)
            for act in actions:
                print(
                    f"   🔒 {act['symbol']}: {act['action']} -> Stop mutat la ${act['new_stop']:.2f} "
                    f"(Profit securizat: ${act['locked_profit']:+,.2f})"
                )
            open_positions = trader.get_open_positions()
            open_symbols = {p["symbol"] for p in open_positions}
            active_risk_count = trader.get_active_risk_positions_count()
            if open_positions:
                print(f"   ✅ Poziții active în portofoliu ({len(open_positions)}): {', '.join(sorted(open_symbols))}")
            else:
                print("   ℹ️ Nicio poziție deschisă momentan.")
        except Exception as e:
            print(f"   ⚠️ Avertisment Position Manager: {e}")
    else:
        print("   ℹ️ Mod Simulare Locală activ.")

    print("\n🌟 [PAS 3/5] Pregătire Watchlist Dinamic 'Sector Rotation & AI/Tech' (Focus pe sectoarele puternice)...")
    watchlist, wl_meta = get_smart_hybrid_watchlist(force_refresh=False)
    cache_tag = "din cache-ul de azi (instant)" if wl_meta.get("from_cache") else "proaspăt calculat"
    print(f"   ✅ {len(watchlist)} lideri selectați ({cache_tag}).")
    if wl_meta.get("hot_sectors"):
        print(f"   🔥 Sectoare Fierbinți (Money Flow): {', '.join(wl_meta['hot_sectors'])}")
    if wl_meta.get("top_5_rs"):
        print(f"   🚀 Top Lideri de Atac azi: {', '.join(wl_meta['top_5_rs'])}")

    print("\n📊 [PAS 4/5] Analiză Regim de Piață Multi-Tier (SPY + QQQ + VIX)...")
    try:
        market = fetch_market_regime(config.BENCHMARK_TICKER)
        regime = market.get("regime", "POWER_TREND")
        risk_mult = market.get("risk_multiplier", 1.0)
        max_pos_allowed = market.get("max_positions_allowed", config.MAX_OPEN_POSITIONS)
        if regime == "POWER_TREND":
            status_text = f"🟢 POWER TREND (Risc 100% -> {config.RISK_PER_TRADE_PERCENT*100:.2f}% / trade)"
        elif regime == "CAUTIOUS_UPTREND":
            status_text = f"🟡 CAUTIOUS UPTREND (Risc redus {risk_mult*100:.0f}% -> {config.RISK_PER_TRADE_PERCENT*risk_mult*100:.2f}% / trade)"
        else:
            status_text = "🔴 BEARISH (Protecție Capital — Nu deschidem poziții Long)"

        print(f"   SPY: ${market['price']:.2f} | SMA 50: ${market.get('sma_50', 0):.2f} | SMA 200: ${market['sma_200']:.2f} ({market['pct_from_sma']:+.2f}%)")
        print(f"   QQQ Bullish: {'DA' if market.get('qqq_bullish', True) else 'NU'} | VIX: {market.get('vix', 18.0):.2f} | Regim: {status_text}")
    except Exception as e:
        print(f"   ⚠️ Eroare preluare Regim Piață ({e}), se folosește regimul implicit.")
        market = {"is_bullish": True, "price": 0, "regime": "POWER_TREND", "risk_multiplier": 1.0, "max_positions_allowed": config.MAX_OPEN_POSITIONS}
        max_pos_allowed = config.MAX_OPEN_POSITIONS

    print(f"\n🔍 [PAS 5/5] Scanare {len(watchlist)} Lideri de Piață (1W Stage 2 + 1D Pullback/Breakout + ML)...")
    summary = trader.get_portfolio_summary()
    capital = summary["portfolio_value"]

    table_data = []
    buy_signals = []

    for ticker in watchlist:
        try:
            df = fetch_stock_data(ticker)
            if ticker in open_symbols:
                last_close = round(float(df["Close"].iloc[-1]), 2)
                rsi = round(float(df["RSI_14"].iloc[-1]), 1)
                ema_20 = round(float(df["EMA_20"].iloc[-1]), 2)
                rs_val = float(df.get("RS_vs_SPY", pd.Series([0.0])).iloc[-1])
                table_data.append([
                    ticker, f"${last_close}", f"${ema_20}", f"{rsi}", f"{rs_val:+.1f}%",
                    "🟢 Deținut", "-", "-", "💼 ÎN PORTOFOLIU", "Poziție deja deschisă (monitorizată automat)"
                ])
                continue

            signal = analyze_ticker(
                ticker=ticker,
                df=df,
                is_market_bullish=market["is_bullish"],
                total_capital=capital,
                risk_multiplier=market.get("risk_multiplier", 1.0),
                market_regime=market.get("regime", "POWER_TREND")
            )

            last_close = round(float(df["Close"].iloc[-1]), 2)
            rsi = round(float(df["RSI_14"].iloc[-1]), 1)
            ema_20 = round(float(df["EMA_20"].iloc[-1]), 2)
            rs_val = float(df.get("RS_vs_SPY", pd.Series([0.0])).iloc[-1])
            rs_str = f"{rs_val:+.1f}%"

            w_ema = float(df.get("Weekly_EMA_10", pd.Series([0.0])).iloc[-1])
            w_sma = float(df.get("Weekly_SMA_30", pd.Series([0.0])).iloc[-1])
            if w_ema > 0 and w_sma > 0:
                if last_close >= w_ema and last_close >= w_sma:
                    w_trend_str = "🟢 Bullish"
                elif last_close >= w_sma:
                    w_trend_str = "🟡 Neutru"
                else:
                    w_trend_str = "🔴 Bearish"
            else:
                w_trend_str = "-"

            try:
                next_e = get_next_earnings_date(ticker)
                if next_e:
                    days_to_e = (next_e - datetime.now()).days
                    earnings_str = f"{next_e.strftime('%d %b')} ({days_to_e}z)"
                    if days_to_e <= config.EARNINGS_BUFFER_DAYS:
                        earnings_str = f"⚠️ {earnings_str}"
                else:
                    earnings_str = "N/A"
            except Exception:
                earnings_str = "N/A"

            if signal.action == "BUY":
                buy_signals.append(signal)
                tag = "BREAKOUT" if signal.setup_type == "VOLUME_BREAKOUT" else "PULLBACK"
                status_display = f"🟢 BUY ({tag})"
                details = f"SL:${signal.stop_loss} | TP1:${signal.tp1} | TP2:${signal.tp2} ({signal.shares}x)"
            else:
                status_display = "⚪ AȘTEAPTĂ"
                details = signal.reason

            ml_val = getattr(signal, "ml_confidence", 0.0)
            if ml_val > 0:
                ml_str = f"{ml_val * 100:.1f}%"
                if ml_val >= 0.55:
                    ml_str = f"🟢 {ml_str}"
                elif ml_val >= getattr(config, "ML_MIN_CONFIDENCE", 0.52):
                    ml_str = f"🟡 {ml_str}"
                else:
                    ml_str = f"🔴 {ml_str}"
            else:
                ml_str = "-"

            table_data.append([
                ticker, f"${last_close}", f"${ema_20}", f"{rsi}", rs_str,
                w_trend_str, earnings_str, ml_str, status_display, details
            ])
        except Exception as e:
            table_data.append([ticker, "N/A", "N/A", "N/A", "-", "-", "N/A", "-", "EROARE", str(e)])

    headers = ["Simbol", "Preț", "EMA 20", "RSI", "RS vs SPY", "1W Trend", "Earnings", "ML Conf.", "Semnal", "Detalii / Motiv"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))

    if buy_signals:
        buy_signals.sort(key=lambda s: getattr(s, "composite_score", 0.0), reverse=True)
        available_slots = max(0, max_pos_allowed - active_risk_count)
        recommended_signals = buy_signals[:available_slots] if available_slots > 0 else []

        print(f"\n🚀 S-au identificat {len(buy_signals)} semnale valide de CUMPĂRARE (ordonate după Scorul de Atac AI/Tech + RS + ML):")
        for idx, sig in enumerate(buy_signals, start=1):
            sec_str = getattr(sig, "sector", config.get_ticker_sector(sig.ticker))
            ml_tag = f" | ML: {sig.ml_confidence*100:.1f}%" if getattr(sig, "ml_confidence", 0) > 0 else ""
            rs_tag = f" | Sector: {sec_str} | RS: {sig.rs_score:+.1f}% | Scor Atac: {sig.composite_score:.0f}/100"
            slot_tag = "✅ SELECTAT PT. EXECUȚIE" if sig in recommended_signals else "⏸️ Peste limita de poziții"
            print(
                f"   {idx}. [{slot_tag}] {sig.ticker} ({sig.setup_type}): Intrare ~${sig.entry_price} | "
                f"SL: ${sig.stop_loss} | TP1: ${sig.tp1} | TP2: ${sig.tp2}{ml_tag}{rs_tag} | "
                f"Acțiuni: {sig.shares} (${sig.position_cost})"
            )

        try:
            from notifier import send_trade_alert
            for sig in buy_signals:
                send_trade_alert(sig)
        except Exception:
            pass

        if not recommended_signals:
            print(
                f"\n🛡️ Ai deja {active_risk_count}/{max_pos_allowed} poziții cu risc activ în portofoliu. "
                f"Nu se recomandă deschiderea de poziții noi până când o poziție existentă nu ajunge la Breakeven."
            )
            return

        print(
            f"\nApasă ENTER (sau scrie 'd') pentru a plasa automat cele {len(recommended_signals)} "
            f"ordine recomandate (Split-Order TP1 + TP2), sau 'n' pentru a anula:"
        )
        choice = input("Confirmare [ENTER = Da / n = Nu]: ").strip().lower()
        if choice in ("", "d", "da", "y", "yes"):
            for sig in recommended_signals:
                trader.place_bracket_buy_order(
                    symbol=sig.ticker,
                    qty=sig.shares,
                    limit_price=sig.entry_price,
                    stop_loss_price=sig.stop_loss,
                    take_profit_price=sig.tp1,
                    tp2_price=sig.tp2,
                    setup_type=sig.setup_type
                )
        else:
            print("ℹ️ Plasarea ordinelor a fost anulată.")
    else:
        print("\nℹ️ Niciun semnal BUY îndeplinit astăzi. Botul așteaptă condițiile optime de risc/recompensă.")


def run_backtest_all():
    """Rulează backtesting individual + simulare realistă de portofoliu cu capital partajat."""
    from ml_predictor import ensure_ml_ready
    pred = ensure_ml_ready(max_age_days=7)
    use_ml = pred.is_ready
    mode_str = "cu FILTRU ML (Out-of-Sample)" if use_ml else "Standard (fără filtru ML)"
    print(f"\n📊 Rulare Backtest ({mode_str}) pe ultimii 2 ani pentru {len(config.WATCHLIST)} companii...")
    print(f"   Capital Inițial: ${config.DEFAULT_SIMULATED_CAPITAL:,.2f} | Risc per trade: {config.RISK_PER_TRADE_PERCENT*100}% | Slippage: {config.BACKTEST_SLIPPAGE_PCT*100:.2f}%")

    try:
        spy_df = fetch_stock_data(config.BENCHMARK_TICKER)
    except Exception as e:
        print(f"⚠️ Nu s-a putut descărca SPY pentru backtest ({e}), rulare fără filtru SPY.")
        spy_df = None

    results_table = []
    stock_dfs = {}
    total_trades_all = 0
    total_pnl_all = 0.0

    for ticker in config.WATCHLIST:
        try:
            df = fetch_stock_data(ticker, spy_df=spy_df)
            stock_dfs[ticker] = df
            tester = Backtester(
                initial_capital=config.DEFAULT_SIMULATED_CAPITAL,
                risk_per_trade=config.RISK_PER_TRADE_PERCENT,
                use_ml=use_ml
            )
            res = tester.run(ticker, df, spy_df=spy_df)

            if "error" not in res:
                total_trades_all += res["total_trades"]
                total_pnl_all += res["total_pnl"]
                pf_str = f"{res['profit_factor']:.2f}" if res["profit_factor"] < 100 else "Max"
                results_table.append([
                    res["ticker"],
                    f"{res.get('total_positions', res['total_trades'])} ({res['total_trades']} lot)",
                    f"{res.get('position_win_rate', res['win_rate'])}%",
                    f"{res.get('expectancy_r', 0.0):+.2f}R",
                    f"${res['total_pnl']:+,.2f}",
                    f"{res['total_return_pct']:+,.2f}%",
                    pf_str,
                    f"{res.get('sharpe_ratio', 0.0):.2f}",
                    f"{res['max_drawdown_pct']:.1f}%"
                ])
        except Exception as e:
            print(f"Eroare backtest {ticker}: {e}")

    headers = [
        "Simbol", "Poziții (Loturi)", "Win Rate Poz.", "Expectancy", "Profit ($)",
        "Randament (%)", "Profit Factor", "Sharpe", "Max DD"
    ]
    print(f"\n📋 1. PERFORMANȚĂ INDIVIDUALĂ PE SIMBOL (${config.DEFAULT_SIMULATED_CAPITAL:,.0f} per simbol):")
    print(tabulate(results_table, headers=headers, tablefmt="grid"))
    print(f"  • Total Loturi Simulate Individual: {total_trades_all} | PnL Cumulat: ${total_pnl_all:+,.2f}")

    if stock_dfs:
        print(f"\n🏦 2. SIMULARE REALISTĂ DE PORTOFOLIU (Cont Unic ${config.DEFAULT_SIMULATED_CAPITAL:,.2f}, Max {config.MAX_OPEN_POSITIONS} poziții simultane cu risc):")
        port_tester = Backtester(
            initial_capital=config.DEFAULT_SIMULATED_CAPITAL,
            risk_per_trade=config.RISK_PER_TRADE_PERCENT,
            use_ml=use_ml
        )
        port_res = port_tester.run_portfolio_backtest(stock_dfs, spy_df=spy_df)
        if "error" not in port_res:
            pf_p = f"{port_res['profit_factor']:.2f}" if port_res["profit_factor"] < 100 else "Max"
            port_table = [[
                f"${port_res['initial_capital']:,.2f}",
                f"${port_res['final_capital']:,.2f}",
                f"${port_res['total_pnl']:+,.2f} ({port_res['total_return_pct']:+.2f}%)",
                f"{port_res['total_positions']} ({port_res['total_trades']} loturi)",
                f"{port_res['position_win_rate']}%",
                f"{port_res['expectancy_r']:+.2f}R",
                pf_p,
                f"{port_res['sharpe_ratio']:.2f} / {port_res['sortino_ratio']:.2f}",
                f"{port_res['max_drawdown_pct']:.2f}%"
            ]]
            port_headers = [
                "Capital Start", "Capital Final", "Profit Net (%)", "Poziții",
                "Win Rate", "Expectancy", "Profit Factor", "Sharpe / Sortino", "Max Drawdown"
            ]
            print(tabulate(port_table, headers=port_headers, tablefmt="grid"))


def run_backtest_and_lab_menu():
    """Opțiunea 3: Unifică Laboratorul Vizual în Browser (TradingView Local) și Backtestul de Portofoliu."""
    print("\n--- 📊 BACKTEST & LABORATOR VIZUAL (TRADINGVIEW LOCAL) ---")
    print("1. 🌐 Deschide Laboratorul Vizual în Browser (Grafic TradingView + Comparație 5 Strategii)")
    print("2. 🏦 Rulează Backtest Complet pe Portofoliu (Simulare Cont Unic + Statistici Detaliate)")
    print("3. 🚀 Ambele (Backtest Portofoliu + Grafic Interactiv în Browser)")
    print("0. Înapoi")

    sub = input("\nAlege (0-3) [implicit: 1]: ").strip() or "1"
    if sub == "1":
        from strategy_lab import run_strategy_lab
        run_strategy_lab()
    elif sub == "2":
        run_backtest_all()
    elif sub == "3":
        run_backtest_all()
        from strategy_lab import run_strategy_lab
        run_strategy_lab()


def run_dynamic_screener():
    """Scanează universul extins (UNIVERSE_POOL) și afișează tabelul liderilor de piață Top RS Stage 2 (Focus AI & Tech)."""
    print(f"\n🌟 SCREENER SECTOR ROTATION & LIDERI AI/TECH (Univers: {len(config.UNIVERSE_POOL)} companii High-Alpha)")
    print("   Se calculează Puterea Relativă (IBD Weighted ROC vs SPY), Rotația pe Sectoare și filtrele Stage 2...")
    ranked = screen_top_rs_leaders(top_n=config.TOP_LEADERS_COUNT)
    if not ranked:
        print("⚠️ Nu s-au putut clasifica simbolurile.")
        return

    table = []
    for idx, item in enumerate(ranked[:20], start=1):
        is_st2 = item.get("stage2", True)
        st2 = "🟢 Stage 2 Lider" if is_st2 else "⚪ Consolidare"
        atr_c = item.get("atr_contraction", 1.0)
        vcp = "🔥 VCP Strâns" if atr_c <= 0.85 else f"{atr_c:.2f}"
        sec_str = item.get("sector", config.get_ticker_sector(item["ticker"]))
        atk_score = item.get("attack_score", item.get("rs_vs_spy", 0.0))
        d_vol = item.get("dollar_vol_m", 0.0)
        table.append([
            idx,
            item["ticker"],
            sec_str,
            f"${item['close']:.2f}",
            f"{item['rs_vs_spy']:+.2f}%",
            f"{atk_score:+.1f}",
            st2,
            vcp,
            f"${d_vol:.0f}M"
        ])

    headers = ["#", "Simbol", "Sector", "Preț", "RS vs SPY", "Scor Atac", "Trend Stage 2", "Contracție ATR", "Volum Zilnic ($)"]
    print("\n" + tabulate(table, headers=headers, tablefmt="grid"))

    wl, wl_meta = get_smart_hybrid_watchlist(force_refresh=True)
    if wl_meta.get("hot_sectors"):
        print(f"\n🔥 Sectoare Fierbinți (Money Flow): {', '.join(wl_meta['hot_sectors'])}")
    print(f"✅ Watchlist-ul AI/Tech & Sector Rotation a fost actualizat ({len(wl)} lideri): {', '.join(wl[:15])}...")


def show_account_status(trader: AlpacaTrader):
    """Afișează statusul contului Alpaca și pozițiile active."""
    summary = trader.get_portfolio_summary()
    print("\n💼 STATUS CONT:")
    if summary.get("connected"):
        print("  - Broker: Alpaca (Paper Trading)")
        print(f"  - Valoare Portofoliu: ${summary['portfolio_value']:,.2f}")
        print(f"  - Cash Disponibil: ${summary['cash']:,.2f}")
        print(f"  - Putere de Cumpărare: ${summary['buying_power']:,.2f}")

        positions = trader.get_open_positions()
        if positions:
            from position_manager import run_position_manager, POSITIONS_STATE_FILE, _load_json
            run_position_manager(trader)
            pos_state = _load_json(POSITIONS_STATE_FILE, {})

            print("\n📈 Poziții Deschise:")
            pos_table = []
            for p in positions:
                sym = p["symbol"]
                st = pos_state.get(sym, {})
                stop_str = f"${st['last_stop_price']:.2f}" if "last_stop_price" in st else "N/A"
                if st.get("breakeven_active") and p["current_price"] >= p["avg_entry_price"] * 1.045:
                    prot_status = "🔒 TRAILING STOP (Profit blocat)"
                elif st.get("breakeven_active"):
                    prot_status = "🛡️ BREAKEVEN (Risc 0)"
                else:
                    prot_status = "⚪ Stop Standard"

                pos_table.append([
                    sym, p["qty"], f"${p['avg_entry_price']:.2f}",
                    f"${p['current_price']:.2f}", stop_str,
                    f"${p['unrealized_pl']:+,.2f} ({p['unrealized_plpc']:+.2f}%)",
                    prot_status
                ])
            headers = ["Simbol", "Acțiuni", "Preț Intrare", "Preț Actual", "Stop Activ", "PnL Nerealizat", "Status Protecție"]
            print(tabulate(pos_table, headers=headers, tablefmt="grid"))
        else:
            print("  - Nu există poziții deschise în acest moment.")
    else:
        print("  - Mod: Simulare Locală (Fără chei Alpaca API)")
        print(f"  - Capital Virtual de Lucru: ${summary['portfolio_value']:,.2f}")


def run_advanced_tools_menu(trader: AlpacaTrader):
    """Opțiunea 4: Sub-meniu pentru unelte avansate și mentenanță manuală."""
    while True:
        print("\n--- 🛠️ UNELTE AVANSATE & MENTENANȚĂ ---")
        print("1. 💼 Status Detaliat Cont & Poziții Deschise (Alpaca)")
        print("2. 📓 Jurnal Tranzacții, Statistici & Carantină (Cooldown)")
        print("3. 🌟 Forțează Reîmprospătarea Screener-ului de Lideri (Top RS Stage 2)")
        print("4. 🧠 Status & Reantrenare Manuală Model ML (XGBoost)")
        print("5. 🧪 Optimizare Robustă Walk-Forward (Anti-Overfitting pe 4 ani)")
        print("0. 🔙 Înapoi la Meniul Principal")

        sub = input("\nAlege o unealtă (0-5): ").strip()
        if sub == "1":
            show_account_status(trader)
        elif sub == "2":
            from risk_learner import print_learning_report
            print_learning_report(trader)
        elif sub == "3":
            run_dynamic_screener()
        elif sub == "4":
            from ml_predictor import get_predictor
            get_predictor().print_status()
            retrain = input("\nDorești reantrenarea forțată a modelului ML acum? (d/n) [implicit: n]: ").strip().lower()
            if retrain in ("d", "da", "y", "yes"):
                from ml_model import run_ml_training
                run_ml_training()
                import ml_predictor
                ml_predictor._predictor_instance = None
                get_predictor()
        elif sub == "5":
            import optimizer
            optimizer.run_walk_forward_optimization()
        elif sub == "0" or sub == "":
            break
        else:
            print("Opțiune invalidă! Alege între 0 și 5.")


def main():
    trader = AlpacaTrader()
    print_cockpit_dashboard(trader)

    while True:
        print("\n--- MENIU PRINCIPAL (SMART COCKPIT) ---")
        print("1. 🚀 Scanare Smart & Execuție (All-in-One: Auto-ML + Protecție Poziții + Lideri + Semnale)")
        print("2. 🤖 Autopilot 24/7 (Rulare Autonomă în Fundal + Alerte/Execuție Telegram)")
        print("3. 📊 Backtest & Laborator Vizual în Browser (TradingView Local + Simulare Portofoliu)")
        print("4. 🛠️ Unelte Avansate & Mentenanță (Cont, Jurnal, Screener, ML, Optimizare)")
        print("0. Ieșire")

        try:
            choice = input("\nAlege o opțiune (0-4) [ENTER = 1 Scanare Smart]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nLa revedere!")
            sys.exit(0)

        if choice in ("1", ""):
            run_smart_scan(trader)
        elif choice == "2":
            import scheduler
            scheduler.main()
        elif choice == "3":
            run_backtest_and_lab_menu()
        elif choice == "4":
            run_advanced_tools_menu(trader)
        elif choice == "0":
            print("La revedere!")
            break
        else:
            print("Opțiune invalidă! Alege între 0 și 4.")


if __name__ == "__main__":
    main()
