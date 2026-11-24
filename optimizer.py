"""
Modul de Optimizare Robustă Walk-Forward V2 (Strict In-Sample / Out-of-Sample & Parameter Plateaus).
Respectă toate regulile anti-overfitting:
  1. Separare cronologică strictă: 70% In-Sample (Train) vs. 30% Out-of-Sample (Validare oarbă).
  2. Selecție din Platouri de Parametri (Parameter Plateaus), niciodată din vârfuri izolate:
     scorul fiecărui candidat este media vecinătății sale în fereastra In-Sample.
  3. Raportează simultan toate metricile obligatorii de risc:
     Max Drawdown (%), Profit Factor, Sharpe Ratio, Sortino Ratio, Total Tranzacții & Win Rate.
"""

import warnings
warnings.filterwarnings("ignore")

from typing import Dict, List, Tuple, Optional
import pandas as pd
from tabulate import tabulate

import config
from data_loader import fetch_stock_data
from backtester import Backtester, run_portfolio_backtest


def load_all_market_data(watchlist: List[str], period: str = "5y") -> Tuple[Optional[pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Descarcă datele istorice pe lumânări închise pentru SPY și simbolurile analizate."""
    print(f"\n📥 1. Se descarcă datele istorice ({period}, lumânări închise) pentru SPY și {len(watchlist)} simboluri...")
    try:
        spy_df = fetch_stock_data(config.BENCHMARK_TICKER, period=period, closed_only=True)
    except Exception as e:
        print(f"⚠️ Eroare SPY: {e}")
        spy_df = None

    data_cache = {}
    for ticker in watchlist:
        try:
            df = fetch_stock_data(ticker, period=period, spy_df=spy_df, closed_only=True)
            if len(df) >= 500:
                data_cache[ticker] = df
        except Exception:
            continue

    print(f"✅ S-au încărcat {len(data_cache)} simboluri valide cu istoric complet.")
    return spy_df, data_cache


def split_data_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    warmup_bars: int = 205
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Împarte cronologic datele în In-Sample (70%) și Out-of-Sample (30%, cu warmup pt SMA 200)."""
    total_len = len(df)
    split_idx = int(total_len * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_start = max(0, split_idx - warmup_bars)
    test_df = df.iloc[test_start:].copy()
    return train_df, test_df


def evaluate_parameter_set(
    data_cache: Dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    rsi4_max: float,
    rsi_min: float,
    rsi_max: float,
    min_rr: float,
    ema_period: int = 20,
    vol_surge: float = 1.50
) -> Dict:
    """Evaluează un set de parametri pe portofoliul comun (Zero Look-Ahead + Costuri Reale)."""
    # Rulăm simularea realistă pe portofoliu actualizând temporar pragurile testate
    orig_vals = (
        config.RSI4_PULLBACK_MAX,
        config.RSI_PULLBACK_MIN,
        config.RSI_PULLBACK_MAX,
        config.MIN_RISK_REWARD_RATIO,
        config.EMA_PULLBACK_PERIOD,
        config.VOLUME_SURGE_MULTIPLIER,
    )
    try:
        config.RSI4_PULLBACK_MAX = rsi4_max
        config.RSI_PULLBACK_MIN = rsi_min
        config.RSI_PULLBACK_MAX = rsi_max
        config.MIN_RISK_REWARD_RATIO = min_rr
        config.EMA_PULLBACK_PERIOD = ema_period
        config.VOLUME_SURGE_MULTIPLIER = vol_surge

        port_res = run_portfolio_backtest(
            data_cache=data_cache,
            spy_df=spy_df,
            initial_capital=config.DEFAULT_SIMULATED_CAPITAL,
            risk_per_trade=config.RISK_PER_TRADE_PERCENT,
            use_ml=False,
        )
    finally:
        (
            config.RSI4_PULLBACK_MAX,
            config.RSI_PULLBACK_MIN,
            config.RSI_PULLBACK_MAX,
            config.MIN_RISK_REWARD_RATIO,
            config.EMA_PULLBACK_PERIOD,
            config.VOLUME_SURGE_MULTIPLIER,
        ) = orig_vals

    if "error" in port_res:
        return {
            "rsi4_max": rsi4_max, "rsi_min": rsi_min, "rsi_max": rsi_max, "min_rr": min_rr,
            "total_positions": 0, "total_trades": 0, "position_win_rate": 0.0,
            "total_pnl": 0.0, "total_return_pct": 0.0, "profit_factor": 0.0,
            "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "max_drawdown_pct": 0.0,
        }

    port_res.update({"rsi4_max": rsi4_max, "rsi_min": rsi_min, "rsi_max": rsi_max, "min_rr": min_rr})
    return port_res


def run_walk_forward_optimization(watchlist: Optional[List[str]] = None):
    """Execută optimizarea Walk-Forward pe Platouri de Parametri (In-Sample 70% -> Out-of-Sample 30%)."""
    if watchlist is None:
        watchlist = list(dict.fromkeys(list(config.WATCHLIST) + list(config.UNIVERSE_POOL)))

    print("=" * 78)
    print("   🧪 OPTIMIZARE WALK-FORWARD PE PLATOURI DE PARAMETRI (ANTI-OVERFITTING)   ")
    print("      70% In-Sample (Antrenare)  |  30% Out-of-Sample (Validare Oarbă)      ")
    print("=" * 78)

    spy_df, data_cache = load_all_market_data(watchlist, period="5y")
    if not data_cache or spy_df is None:
        print("❌ Date insuficiente pentru rularea optimizării.")
        return

    train_cache, test_cache = {}, {}
    for ticker, df in data_cache.items():
        tr, te = split_data_chronological(df, train_ratio=0.70)
        train_cache[ticker] = tr
        test_cache[ticker] = te

    train_spy, test_spy = split_data_chronological(spy_df, train_ratio=0.70)

    rsi4_grid = [28.0, 30.0, 32.0]
    rsi14_grid = [(38.0, 54.0), (40.0, 56.0)]
    rr_grid = [2.0, 2.5]

    in_sample_results = []
    for r4 in rsi4_grid:
        for (r_min, r_max) in rsi14_grid:
            for rr in rr_grid:
                orig_tp2 = config.TP2_RISK_REWARD_RATIO
                try:
                    config.TP2_RISK_REWARD_RATIO = max(rr, 2.5)
                    res = evaluate_parameter_set(train_cache, train_spy, r4, r_min, r_max, min_rr=2.0)
                    res["tp2_rr"] = config.TP2_RISK_REWARD_RATIO
                finally:
                    config.TP2_RISK_REWARD_RATIO = orig_tp2

                if res["total_positions"] >= 25:
                    in_sample_results.append(res)

    if not in_sample_results:
        print("❌ Niciun set de parametri nu a generat minim 25 de poziții In-Sample.")
        return

    # Scor de Platou: penalizează vârfurile izolate calculând media cu vecinii de grid
    for cand in in_sample_results:
        neighbors = [
            o for o in in_sample_results
            if abs(o["rsi4_max"] - cand["rsi4_max"]) <= 2.0
        ]
        avg_neighbor_pf = sum(n["profit_factor"] for n in neighbors) / len(neighbors)
        cand["plateau_score"] = round(0.5 * cand["profit_factor"] + 0.5 * avg_neighbor_pf, 3)

    in_sample_results.sort(key=lambda x: (x["plateau_score"], x["sharpe_ratio"]), reverse=True)
    top_candidates = in_sample_results[:3]

    print("\n🏆 ETAPA 1 — TOP 3 PLATOURI STABILE IN-SAMPLE:")
    is_rows = []
    for i, c in enumerate(top_candidates, 1):
        is_rows.append([
            f"#{i}",
            f"RSI4<={c['rsi4_max']:.0f} | RSI14 {c['rsi_min']:.0f}-{c['rsi_max']:.0f} | TP2 {c['tp2_rr']}R",
            f"{c['total_positions']} ({c['total_trades']})",
            f"{c['position_win_rate']}%",
            f"{c['total_return_pct']:+.2f}%",
            f"{c['profit_factor']:.2f} (Platou: {c['plateau_score']:.2f})",
            f"{c['sharpe_ratio']:.2f} / {c['sortino_ratio']:.2f}",
            f"-{c['max_drawdown_pct']:.2f}%",
        ])
    print(tabulate(
        is_rows,
        headers=["#", "Parametri", "Poziții (Loturi)", "Win Rate", "Randament", "Profit Factor", "Sharpe / Sortino", "Max DD"],
        tablefmt="grid"
    ))

    print("\n🧪 ETAPA 2 — VALIDARE OARBĂ OUT-OF-SAMPLE (DATE NEVĂZUTE):")
    oos_rows = []
    for i, cand in enumerate(top_candidates, 1):
        orig_tp2 = config.TP2_RISK_REWARD_RATIO
        try:
            config.TP2_RISK_REWARD_RATIO = cand["tp2_rr"]
            te_res = evaluate_parameter_set(
                test_cache, test_spy, cand["rsi4_max"], cand["rsi_min"], cand["rsi_max"], min_rr=2.0
            )
        finally:
            config.TP2_RISK_REWARD_RATIO = orig_tp2

        verdict = "✅ ROBUST OOS" if (te_res["profit_factor"] >= 1.05 and te_res["total_return_pct"] > 0) else "🟡 NEUTRU/DEFENSIV"
        oos_rows.append([
            f"#{i}",
            f"RSI4<={cand['rsi4_max']:.0f} | RSI14 {cand['rsi_min']:.0f}-{cand['rsi_max']:.0f}",
            f"{te_res['total_positions']} ({te_res['total_trades']})",
            f"{te_res['position_win_rate']}%",
            f"{te_res['total_return_pct']:+.2f}%",
            f"{te_res['profit_factor']:.2f}",
            f"{te_res['sharpe_ratio']:.2f} / {te_res['sortino_ratio']:.2f}",
            f"-{te_res['max_drawdown_pct']:.2f}%",
            verdict,
        ])
    print(tabulate(
        oos_rows,
        headers=["#", "Parametri", "Poziții OOS", "Win Rate OOS", "Randament OOS", "PF OOS", "Sharpe / Sortino", "Max DD OOS", "Verdict"],
        tablefmt="grid"
    ))


if __name__ == "__main__":
    run_walk_forward_optimization()
