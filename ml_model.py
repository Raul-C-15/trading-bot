"""
Modul de Machine Learning Institutional pentru Trading Bot.
Antrenează un clasificator XGBoost (XGBClassifier) pe datele istorice generate
de backtester pentru a filtra semnalele de calitate slabă.

Arhitectură și Protecții Anti-Overfitting:
  - Eliminare Data Leakage & Lookahead Bias: Eșantioanele sunt ordonate strict cronologic.
  - Cross-validare TimeSeriesSplit: Nu se amestecă viitorul cu trecutul.
  - Cache de predicții Out-of-Sample (OOS) salvat în metadate pentru backtesting onest.
  - Regularizare dură XGBoost (max_depth=3, learning_rate=0.02, reg_alpha=0.2, reg_lambda=1.5).
  - 11 Caracteristici Tehnice, de Volatilitate (VCP), Multi-Timeframe (1W) și Forță Relativă (RS vs SPY).
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional

import config
from data_loader import fetch_stock_data
from backtester import Backtester

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

MODEL_PATH = DATA_DIR / "ml_model.pkl"
MODEL_META_PATH = DATA_DIR / "ml_model_meta.json"

FEATURE_NAMES = [
    "rsi",
    "close_to_ema_pct",
    "close_to_sma50_pct",
    "volume_ratio",
    "body_pct",
    "lower_wick_pct",
    "upper_wick_pct",
    "atr_pct",
    "rs_vs_spy",
    "atr_contraction",
    "weekly_rsi",
]


def extract_bar_features(bar: pd.Series, vol_sma20: float) -> Optional[List[float]]:
    """
    Extrage vectorul de 11 caracteristici dintr-o lumânare OHLCV + indicatori.
    Returnează None dacă datele sunt invalide.
    """
    try:
        open_p  = float(bar["Open"])
        high_p  = float(bar["High"])
        low_p   = float(bar["Low"])
        close_p = float(bar["Close"])
        rsi     = float(bar["RSI_14"])
        ema_20  = float(bar["EMA_20"])
        sma_50  = float(bar["SMA_50"])
        atr     = float(bar.get("ATR_14", close_p * 0.02))
        volume  = float(bar["Volume"])

        rs_vs_spy = float(bar.get("RS_vs_SPY", 0.0))
        if np.isnan(rs_vs_spy):
            rs_vs_spy = 0.0

        atr_contraction = float(bar.get("ATR_Contraction", 1.0))
        if np.isnan(atr_contraction) or atr_contraction <= 0:
            atr_contraction = 1.0

        weekly_rsi = float(bar.get("Weekly_RSI_14", 55.0))
        if np.isnan(weekly_rsi) or weekly_rsi <= 0:
            weekly_rsi = 55.0

        candle_range = high_p - low_p
        if candle_range <= 0 or close_p <= 0 or ema_20 <= 0 or sma_50 <= 0:
            return None

        body        = abs(close_p - open_p)
        lower_wick  = min(open_p, close_p) - low_p
        upper_wick  = high_p - max(open_p, close_p)

        return [
            rsi,
            (close_p - ema_20) / ema_20 * 100,
            (close_p - sma_50) / sma_50 * 100,
            volume / vol_sma20 if vol_sma20 > 0 else 1.0,
            body        / candle_range,
            lower_wick  / candle_range,
            upper_wick  / candle_range,
            atr / close_p * 100,
            rs_vs_spy,
            atr_contraction,
            weekly_rsi,
        ]
    except Exception:
        return None


def generate_training_data(
    watchlist: List[str] = None,
    period: str = "4y"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Rulează backtestul pe universul de acțiuni și extrage caracteristicile
    pentru fiecare semnal de intrare, cu label WIN (1) sau LOSS (0).
    Eșantioanele sunt obligatoriu sortate strict cronologic (fără data leakage).
    """
    if watchlist is None:
        # Folosim watchlist-ul + lideri din UNIVERSE_POOL daca avem nevoie de esantioane bogate
        watchlist = list(config.WATCHLIST)

    print(f"\n📥 Se descarcă date istorice ({period}) pentru {len(watchlist)} acțiuni...")

    try:
        spy_df = fetch_stock_data("SPY", period=period)
    except Exception:
        spy_df = None
        print("  ⚠️ SPY indisponibil, se continuă fără filtrul de regim.")

    raw_samples: List[Tuple[pd.Timestamp, str, List[float], int]] = []

    for ticker in watchlist:
        try:
            df = fetch_stock_data(ticker, period=period, spy_df=spy_df)
            if len(df) < 250:
                print(f"  ⚠️ {ticker}: date insuficiente ({len(df)} lumânări), omis.")
                continue

            bt = Backtester(
                initial_capital=config.DEFAULT_SIMULATED_CAPITAL,
                risk_per_trade=config.RISK_PER_TRADE_PERCENT,
                rsi_pullback_min=config.RSI_PULLBACK_MIN,
                rsi_pullback_max=config.RSI_PULLBACK_MAX,
                min_risk_reward=config.MIN_RISK_REWARD_RATIO,
                ema_period=20,
                volume_surge_mult=config.VOLUME_SURGE_MULTIPLIER,
                use_ml=False
            )
            bt.run(ticker, df, spy_df=spy_df)

            if not bt.trades:
                print(f"  ℹ️ {ticker}: nicio tranzacție simulată.")
                continue

            entry_pnl: Dict[str, float] = {}
            for trade in bt.trades:
                ed = str(trade.get("entry_date", ""))
                entry_pnl[ed] = entry_pnl.get(ed, 0.0) + trade["pnl"]

            df_indexed = df.copy()
            df_indexed.index = pd.to_datetime(df_indexed.index)
            if df_indexed.index.tz is not None:
                df_indexed.index = df_indexed.index.tz_localize(None)
            extracted = 0

            for entry_date_str, total_pnl in entry_pnl.items():
                try:
                    entry_ts = pd.to_datetime(entry_date_str)
                    if getattr(entry_ts, "tz", None) is not None:
                        entry_ts = entry_ts.tz_localize(None)

                    if entry_ts not in df_indexed.index:
                        continue
                    bar = df_indexed.loc[entry_ts]
                    if isinstance(bar, pd.DataFrame):
                        bar = bar.iloc[0]

                    vol_sma20 = float(bar.get("Volume_SMA_20", bar["Volume"]))
                    features = extract_bar_features(bar, vol_sma20)
                    if features is None:
                        continue

                    label = 1 if total_pnl > 0 else 0
                    sample_key = f"{entry_ts.strftime('%Y-%m-%d')}_{float(bar['Close']):.2f}"
                    raw_samples.append((entry_ts, sample_key, features, label))
                    extracted += 1
                except Exception:
                    continue

            wins_t = sum(1 for ed, pnl in entry_pnl.items() if pnl > 0)
            print(f"  ✅ {ticker}: {extracted} eșantioane extrase "
                  f"({wins_t} WIN / {len(entry_pnl)-wins_t} LOSS)")

        except Exception as e:
            print(f"  ❌ Eroare {ticker}: {e}")

    if not raw_samples:
        print("\n📊 Total eșantioane pentru antrenare: 0")
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32), []

    # Sortare strict cronologica dupa entry_ts
    raw_samples.sort(key=lambda item: item[0])

    sample_keys  = [s[1] for s in raw_samples]
    all_features = [s[2] for s in raw_samples]
    all_labels   = [s[3] for s in raw_samples]

    total = len(all_features)
    first_dt = raw_samples[0][0].strftime("%Y-%m-%d")
    last_dt  = raw_samples[-1][0].strftime("%Y-%m-%d")
    print(f"\n📊 Total eșantioane pentru antrenare: {total} (ordonate strict cronologic)")
    print(f"   Interval temporal date: {first_dt} -> {last_dt}")

    return np.array(all_features, dtype=np.float32), np.array(all_labels, dtype=np.int32), sample_keys


def _build_pipeline(scale_pos_weight: float = 1.0):
    """
    Construiește pipeline-ul StandardScaler + XGBClassifier cu hiperparametri
    de regularizare duri anti-overfitting pentru date financiare zgomotoase.
    """
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            max_depth=3,
            learning_rate=0.02,
            n_estimators=140,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=1.5,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )),
    ])


def train_and_save_model(X: np.ndarray, y: np.ndarray, sample_keys: Optional[List[str]] = None) -> Dict:
    """
    Antrenează un Pipeline (StandardScaler + XGBClassifier) și îl salvează.
    Evaluează modelul prin TimeSeriesSplit (5-fold) și salvează predicțiile
    Out-of-Sample (OOS) pentru a garanta backtesting fără Data Leakage!
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score, precision_score
    import joblib

    min_samples = getattr(config, "ML_MIN_TRAINING_SAMPLES", 50)
    if len(X) < min_samples:
        print(f"❌ Date insuficiente ({len(X)} < {min_samples} minim). Adaugă mai multe acțiuni sau mărește perioada.")
        return {}

    wins   = int(np.sum(y == 1))
    losses = int(np.sum(y == 0))
    wr     = round(wins / len(y) * 100, 1) if len(y) > 0 else 0.0
    print(f"\n📊 Distribuție: {wins} WIN / {losses} LOSS  (Win Rate istoric de bază: {wr}%)")

    # Folosim scale_pos_weight=1.0 cand WIN >= LOSS pentru a nu sub-calibra probabilitatile
    final_scale_pos_weight = float(losses / wins) if (wins > 0 and losses > wins) else 1.0
    print(f"⚖️  XGBoost scale_pos_weight: {final_scale_pos_weight:.3f}")

    print("\n🔁 Cross-validare TimeSeriesSplit (5-fold) [ROC-AUC & Precision]...")
    tscv = TimeSeriesSplit(n_splits=5)
    auc_scores: List[float] = []
    precision_scores: List[float] = []
    oos_predictions: Dict[str, float] = {}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_wins = int(np.sum(y_train == 1))
        fold_losses = int(np.sum(y_train == 0))
        fold_scale = float(fold_losses / fold_wins) if (fold_wins > 0 and fold_losses > fold_wins) else 1.0

        fold_pipeline = _build_pipeline(scale_pos_weight=fold_scale)
        fold_pipeline.fit(X_train, y_train)

        y_prob = fold_pipeline.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= getattr(config, "ML_MIN_CONFIDENCE", 0.52)).astype(int)

        if sample_keys is not None:
            for idx_pos, prob_val in zip(val_idx, y_prob):
                if idx_pos < len(sample_keys):
                    oos_predictions[sample_keys[idx_pos]] = round(float(prob_val), 4)

        if len(np.unique(y_val)) >= 2:
            fold_auc = roc_auc_score(y_val, y_prob)
            auc_scores.append(fold_auc)
            auc_str = f"{fold_auc:.3f}"
        else:
            auc_str = "N/A"

        fold_prec = precision_score(y_val, y_pred, zero_division=0)
        precision_scores.append(fold_prec)

        print(f"   Fold {fold}: ROC-AUC = {auc_str} | Precision = {fold_prec:.3f} "
              f"(Train={len(train_idx)}, Val={len(val_idx)})")

    mean_auc  = float(np.nanmean(auc_scores)) if auc_scores else 0.5
    std_auc   = float(np.nanstd(auc_scores)) if auc_scores else 0.0
    mean_prec = float(np.mean(precision_scores)) if precision_scores else 0.0
    std_prec  = float(np.std(precision_scores)) if precision_scores else 0.0

    print(f"\n   📈 Rezultate TimeSeriesSplit CV:")
    print(f"      • ROC-AUC mean   : {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"      • Precision mean : {mean_prec:.3f} ± {std_prec:.3f}")

    # Antrenare finala pe tot setul (pentru predictiile LIVE din prezent)
    pipeline = _build_pipeline(scale_pos_weight=final_scale_pos_weight)
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)

    meta = {
        "trained_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type":        "XGBClassifier",
        "n_samples":         len(X),
        "n_wins":            wins,
        "n_losses":          losses,
        "win_rate_pct":      wr,
        "scale_pos_weight":  round(final_scale_pos_weight, 4),
        "cv_roc_auc_mean":   round(mean_auc, 4),
        "cv_roc_auc_std":    round(std_auc, 4),
        "cv_precision_mean": round(mean_prec, 4),
        "cv_precision_std":  round(std_prec, 4),
        "features":          FEATURE_NAMES,
        "watchlist":         config.WATCHLIST,
        "oos_predictions":   oos_predictions,
    }
    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    print(f"✅ Model salvat → {MODEL_PATH} (cu {len(oos_predictions)} predicții Out-of-Sample salvate)")
    return meta


def run_ml_training(watchlist=None):
    """Punct de intrare principal pentru antrenarea modelului ML."""
    print("=" * 65)
    print("   🧠 ANTRENARE MODEL ML — XGBOOST CLASSIFIER (11 FEATURES PRO)")
    print("   Include: RS vs SPY, VCP Contraction, Weekly RSI & Out-of-Sample")
    print("=" * 65)

    try:
        import sklearn  # noqa: F401
        import xgboost  # noqa: F401
    except ImportError as e:
        print(f"\n❌ Dependență lipsă ({getattr(e, 'name', 'xgboost/sklearn')})!")
        print("   Rulează: pip install scikit-learn joblib xgboost")
        return

    X, y, sample_keys = generate_training_data(watchlist)
    if len(X) == 0:
        print("❌ Nu s-au putut genera date de antrenare.")
        return

    meta = train_and_save_model(X, y, sample_keys=sample_keys)
    if not meta:
        return

    # Reincarcam singleton-ul MLPredictor in memorie
    try:
        import ml_predictor
        ml_predictor._predictor_instance = None
    except Exception:
        pass

    print("\n" + "=" * 65)
    print("   🏆 REZULTATE ANTRENARE ML (XGBOOST)")
    print("=" * 65)
    print(f"   • Eșantioane totale : {meta['n_samples']}")
    print(f"   • WIN / LOSS        : {meta['n_wins']} / {meta['n_losses']}")
    print(f"   • Win Rate de bază  : {meta['win_rate_pct']}%")
    print(f"   • ROC-AUC (TS-CV)   : {meta['cv_roc_auc_mean']:.3f} ± {meta['cv_roc_auc_std']:.3f}")
    print(f"   • Precision (TS-CV) : {meta['cv_precision_mean']:.3f} ± {meta['cv_precision_std']:.3f}")
    print(f"   • Antrenat la       : {meta['trained_at']}")
    print("=" * 65)

    try:
        from notifier import send_telegram_message
        send_telegram_message(
            f"🧠 *MODEL ML (XGBOOST) ANTRENAT CU SUCCES!*\n\n"
            f"• Eșantioane: *{meta['n_samples']}* tranzacții istorice\n"
            f"• Win Rate de bază: *{meta['win_rate_pct']}%*\n"
            f"• ROC-AUC (TS-CV): *{meta['cv_roc_auc_mean']:.3f}*\n"
            f"• Precision (TS-CV): *{meta['cv_precision_mean']:.3f}*\n"
            f"• Antrenat la: {meta['trained_at']}"
        )
    except Exception:
        pass


if __name__ == "__main__":
    run_ml_training()
