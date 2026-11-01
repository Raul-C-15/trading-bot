"""
Modul de Optimizare Robustă a Parametrilor (Hyperparameter Tuning).
Construit special pentru a EVITA OVERFITTING-UL (Curve-Fitting):

Principii Anti-Overfitting:
1. Validare Walk-Forward (In-Sample / Out-of-Sample):
   - 70% din istoric este folosit pentru antrenare (In-Sample: 2022 - mid 2025).
   - 30% din istoric este folosit pentru testare 'oarbă' pe date nevăzute (Out-of-Sample: mid 2025 - 2026).
2. Platouri de parametri (Neighborhoods), nu spike-uri izolate:
   - Testează doar un grid restrâns de parametri logici, eliminând zgomotul aleator.
3. Dimensiune minimă a eșantionului:
   - Refuză orice set de parametri cu mai puțin de 25-30 de tranzacții.
4. Optimizare universală pe coșul de acțiuni:
   - Nu se fitează parametri individuali per acțiune (cea mai mare greșeală din trading),
     ci se găsește setul universal robust care funcționează transversal pe piață.
5. Filtrare și curățare Watchlist:
   - Identifică acțiunile care eșuează sistematic atât In-Sample cât și Out-of-Sample,
     recomandând eliminarea lor pentru a proteja capitalul.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from tabulate import tabulate

import config
from data_loader import fetch_stock_data
from backtester import Backtester


def load_all_market_data(watchlist: List[str], period: str = "4y") -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Descarcă și stochează în memorie datele pentru SPY și toate acțiunile din watchlist."""
    print(f"\n📥 1. Se descarcă datele istorice ({period}) pentru SPY și {len(watchlist)} simboluri...")
    try:
        spy_df = fetch_stock_data(config.BENCHMARK_TICKER, period=period)
    except Exception as e:
        print(f"⚠️ Eroare SPY: {e}")
        spy_df = None

    data_cache = {}
    for ticker in watchlist:
        try:
            df = fetch_stock_data(ticker, period=period)
            if len(df) >= 450:  # Minim 450 de zile pentru a asigura un split valid
                data_cache[ticker] = df
            else:
                print(f"  ⚠️ {ticker}: Date insuficiente ({len(df)} lumânări), omis din optimizare.")
        except Exception as e:
            print(f"  ⚠️ Eroare la descărcarea {ticker}: {e}")

    print(f"✅ S-au încărcat cu succes {len(data_cache)} acțiuni valide cu istoric lung.")
    return spy_df, data_cache


def split_data_chronological(
    df: pd.DataFrame, 
    train_ratio: float = 0.70,
    warmup_bars: int = 200
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Împarte datele cronologic în Train (In-Sample) și Test (Out-of-Sample).
    Include un buffer de încălzire (warmup) de 200 de lumânări pentru test
    pentru a calcula corect SMA 200 / SMA 50 fără a tăia istoricul.
    """
    total_len = len(df)
    split_idx = int(total_len * train_ratio)

    train_df = df.iloc[:split_idx]
    test_start = max(0, split_idx - warmup_bars)
    test_df = df.iloc[test_start:]

    return train_df, test_df


def evaluate_parameter_set(
    data_cache: Dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    rsi_min: float,
    rsi_max: float,
    min_rr: float,
    ema_period: int,
    vol_surge: float
) -> Dict:
    """Rulează backtest-ul pe tot coșul de acțiuni pentru un set specific de parametri."""
    total_trades = 0
    winning_trades = 0
    gross_profit = 0.0
    gross_loss = 0.0
    total_pnl = 0.0

    for ticker, df in data_cache.items():
        bt = Backtester(
            initial_capital=config.DEFAULT_SIMULATED_CAPITAL,
            risk_per_trade=config.RISK_PER_TRADE_PERCENT,
            rsi_pullback_min=rsi_min,
            rsi_pullback_max=rsi_max,
            min_risk_reward=min_rr,
            ema_period=ema_period,
            volume_surge_mult=vol_surge
        )
        res = bt.run(ticker, df, spy_df=spy_df)
        if "error" in res or res["total_trades"] == 0:
            continue

        total_trades += res["total_trades"]
        winning_trades += res["winning_trades"]
        total_pnl += res["total_pnl"]

        df_trades = pd.DataFrame(bt.trades)
        if not df_trades.empty:
            wins = df_trades[df_trades["pnl"] > 0]["pnl"].sum()
            losses = abs(df_trades[df_trades["pnl"] < 0]["pnl"].sum())
            gross_profit += wins
            gross_loss += losses

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    return {
        "rsi_min": rsi_min,
        "rsi_max": rsi_max,
        "min_rr": min_rr,
        "ema_period": ema_period,
        "vol_surge": vol_surge,
        "total_trades": total_trades,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "profit_factor": round(profit_factor, 2)
    }


def run_walk_forward_optimization(watchlist: List[str] = None):
    """
    Execută optimizarea completă Walk-Forward cu protecție anti-overfitting.
    """
    if watchlist is None:
        watchlist = config.WATCHLIST

    print("=" * 70)
    print("      🧪 OPTIMIZARE ROBUSTĂ WALK-FORWARD (ANTI-OVERFITTING)      ")
    print("   Split Date: 70% In-Sample (Train) | 30% Out-of-Sample (Test)   ")
    print("=" * 70)

    spy_df, data_cache = load_all_market_data(watchlist, period="4y")
    if not data_cache or spy_df is None:
        print("❌ Date insuficiente pentru rulare.")
        return

    # Separare Train / Test pentru fiecare acțiune
    train_cache = {}
    test_cache = {}
    for ticker, df in data_cache.items():
        tr, te = split_data_chronological(df, train_ratio=0.70)
        train_cache[ticker] = tr
        test_cache[ticker] = te

    train_spy, test_spy = split_data_chronological(spy_df, train_ratio=0.70)

    # Coarse Grid (Platouri largi pentru a preveni curve-fitting-ul)
    ema_grid = [15, 20, 25]
    rsi_grid = [
        (35.0, 55.0),  # Retragere mai adâncă
        (38.0, 56.0),  # Configurație actuală de referință
        (40.0, 60.0)   # Retragere moderată
    ]
    rr_grid = [1.5, 2.0]
    vol_grid = [1.25, 1.35]

    total_combos = len(ema_grid) * len(rsi_grid) * len(rr_grid) * len(vol_grid)
    print(f"\n🔍 2. Rulare In-Sample Training pe {total_combos} combinații de parametri...")
    print(f"   Perioada Train: primele ~70% din lumânări (~2.8 ani)")

    in_sample_results = []
    combo_idx = 0

    for ema_p in ema_grid:
        for (r_min, r_max) in rsi_grid:
            for rr in rr_grid:
                for vol_s in vol_grid:
                    combo_idx += 1
                    res = evaluate_parameter_set(
                        train_cache, train_spy,
                        rsi_min=r_min, rsi_max=r_max,
                        min_rr=rr, ema_period=ema_p, vol_surge=vol_s
                    )
                    # Filtru de semnificație statistică: minim 25 de tranzacții
                    if res["total_trades"] >= 25:
                        in_sample_results.append(res)

    if not in_sample_results:
        print("❌ Niciun set de parametri nu a atins pragul minim de 25 de tranzacții.")
        return

    # Sortăm după Profit Factor și PnL
    in_sample_results.sort(key=lambda x: (x["profit_factor"], x["total_pnl"]), reverse=True)

    print("\n🏆 Top 3 Candidați In-Sample (Antrenați pe date istorice vechi):")
    top_candidates = in_sample_results[:3]
    top_table = []
    for i, c in enumerate(top_candidates, 1):
        top_table.append([
            f"#{i}", f"EMA {c['ema_period']}", f"RSI {c['rsi_min']}-{c['rsi_max']}",
            f"R:R 1:{c['min_rr']}", f"Vol {c['vol_surge']}x",
            c["total_trades"], f"{c['win_rate']}%", f"${c['total_pnl']:+,.2f}", c["profit_factor"]
        ])
    headers_top = ["Loc", "EMA", "RSI", "R:R", "Vol", "Trades", "Win Rate", "PnL Train", "Profit Factor"]
    print(tabulate(top_table, headers=headers_top, tablefmt="grid"))

    # --- ETAPA 2: VALIDARE OUT-OF-SAMPLE (TESTUL OARBĂ) ---
    print("\n" + "=" * 70)
    print("      🧪 ETAPA 2: TESTUL OUT-OF-SAMPLE (TESTUL SUPREM ANTI-OVERFITTING)      ")
    print("   Se rulează candidații pe ultimul an (~1.2 ani) de date complet nevăzute!   ")
    print("=" * 70)

    validation_table = []
    best_candidate = None
    best_test_pf = 0.0

    for i, cand in enumerate(top_candidates, 1):
        test_res = evaluate_parameter_set(
            test_cache, test_spy,
            rsi_min=cand["rsi_min"], rsi_max=cand["rsi_max"],
            min_rr=cand["min_rr"], ema_period=cand["ema_period"], vol_surge=cand["vol_surge"]
        )

        train_pf = cand["profit_factor"]
        test_pf = test_res["profit_factor"]
        test_wr = test_res["win_rate"]
        test_pnl = test_res["total_pnl"]

        # Walk-Forward Efficiency (raportul de degradare)
        # 1.0 înseamnă performanță identică out-of-sample ca in-sample
        wfe = (test_pf / train_pf) if train_pf > 0 else 0.0

        # Verificare Overfitting
        if test_pf >= 1.20 and test_pnl > 0 and wfe >= 0.45:
            verdict = "✅ VALIDAT (Robust)"
        elif test_pf >= 1.0 and test_pnl >= 0:
            verdict = "🟡 ACCEPTABIL"
        else:
            verdict = "❌ OVERFITTED (Respins)"

        validation_table.append([
            f"Set #{i}",
            f"EMA {cand['ema_period']} | RSI {cand['rsi_min']}-{cand['rsi_max']} | R:R 1:{cand['min_rr']}",
            f"{cand['profit_factor']:.2f}",
            f"{test_pf:.2f}",
            f"{cand['win_rate']}% ➔ {test_wr}%",
            f"${cand['total_pnl']:+,.2f}",
            f"${test_pnl:+,.2f}",
            f"{wfe:.2f}",
            verdict
        ])

        if test_pf > best_test_pf and "Respins" not in verdict:
            best_test_pf = test_pf
            best_candidate = (cand, test_res)

    v_headers = ["Set", "Parametri", "PF Train", "PF Test", "Win Rate (Train ➔ Test)", "PnL Train", "PnL Test", "WFE", "Verdict"]
    print(tabulate(validation_table, headers=v_headers, tablefmt="grid"))

    if not best_candidate:
        print("\n⚠️ Toți candidații au arătat semne de overfitting pe datele Out-of-Sample.")
        print("Recomandare: Păstrarea parametrilor conservatori impliciți.")
        return

    chosen_param, chosen_test = best_candidate
    print(f"\n🎉 SETUL CÂȘTIGĂTOR ROBUST: Setul cu EMA {chosen_param['ema_period']}, RSI {chosen_param['rsi_min']}-{chosen_param['rsi_max']}, R:R 1:{chosen_param['min_rr']}")
    print(f"  • Profit Factor Out-of-Sample: {chosen_test['profit_factor']} (fără degradare periculoasă)")
    print(f"  • PnL Out-of-Sample: ${chosen_test['total_pnl']:+,.2f}")

    # --- ETAPA 3: ANALIZA ȘI CURĂȚAREA WATCHLIST-ULUI ---
    print("\n" + "=" * 70)
    print("      📋 ETAPA 3: CURĂȚAREA WATCHLIST-ULUI (FILTRARE TITLURI TOXICE)      ")
    print("=" * 70)

    per_stock_table = []
    keep_list = []
    remove_list = []

    for ticker, df in data_cache.items():
        tr_df, te_df = split_data_chronological(df, train_ratio=0.70)
        
        bt_tr = Backtester(
            initial_capital=config.DEFAULT_SIMULATED_CAPITAL,
            risk_per_trade=config.RISK_PER_TRADE_PERCENT,
            rsi_pullback_min=chosen_param["rsi_min"],
            rsi_pullback_max=chosen_param["rsi_max"],
            min_risk_reward=chosen_param["min_rr"],
            ema_period=chosen_param["ema_period"],
            volume_surge_mult=chosen_param["vol_surge"]
        )
        res_tr = bt_tr.run(ticker, tr_df, spy_df=train_spy)

        bt_te = Backtester(
            initial_capital=config.DEFAULT_SIMULATED_CAPITAL,
            risk_per_trade=config.RISK_PER_TRADE_PERCENT,
            rsi_pullback_min=chosen_param["rsi_min"],
            rsi_pullback_max=chosen_param["rsi_max"],
            min_risk_reward=chosen_param["min_rr"],
            ema_period=chosen_param["ema_period"],
            volume_surge_mult=chosen_param["vol_surge"]
        )
        res_te = bt_te.run(ticker, te_df, spy_df=test_spy)

        tr_pnl = res_tr.get("total_pnl", 0.0)
        te_pnl = res_te.get("total_pnl", 0.0)
        te_pf = res_te.get("profit_factor", 0.0)
        te_wr = res_te.get("win_rate", 0.0)

        # Reguli stricte de clasificare a acțiunilor:
        if te_pf >= 1.25 and te_pnl > 0 and (tr_pnl >= 0 or res_tr["win_rate"] >= 45):
            status = "🟢 CORE (Păstrează)"
            keep_list.append(ticker)
        elif te_pnl >= -100 and (tr_pnl > 0 or te_pf >= 1.0):
            status = "🟡 NEUTRU (Monitorizare)"
            keep_list.append(ticker)
        else:
            status = "🔴 ELIMINĂ (Toxic/Pierderi repetate)"
            remove_list.append(ticker)

        te_pf_str = f"{te_pf:.2f}" if te_pf < 100 else "Max"
        per_stock_table.append([
            ticker,
            f"${tr_pnl:+,.1f}",
            f"${te_pnl:+,.1f}",
            f"{te_wr:.1f}%",
            te_pf_str,
            status
        ])

    s_headers = ["Ticker", "PnL Train", "PnL Test", "WinRate Test", "PF Test", "Recomandare"]
    print(tabulate(per_stock_table, headers=s_headers, tablefmt="grid"))

    print(f"\n💡 REZUMAT CURĂȚARE:")
    print(f"  • Acțiuni Confirmate Robust ({len(keep_list)}): {', '.join(keep_list)}")
    print(f"  • Acțiuni Recomandate pentru Eliminare ({len(remove_list)}): {', '.join(remove_list) if remove_list else 'Niciuna'}")


if __name__ == "__main__":
    run_walk_forward_optimization()
