"""
Modul de Machine Learning Instituțional V2 (Zero Data Leakage & Poartă Automată Out-of-Sample).
  - Caracteristicile sunt extrase EXCLUSIV de pe bara `signal_date` (lumânarea închisă `t`),
    niciodată de pe bara de execuție `t + 1`.
  - Normalizarea (`StandardScaler`) este inclusă în `Pipeline` și fitată strict pe fiecare fereastră `In-Sample`.
  - Validare `TimeSeriesSplit` (5-fold) și activare condiționată: modelul filtrează semnalele DOAR dacă
    demonstrează avantaj statistic real Out-of-Sample (`ROC-AUC >= ML_MIN_OOS_ROC_AUC`).
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
    "rsi_14",
    "rsi_4",
    "close_to_ema20_pct",
    "close_to_sma50_pct",
    "volume_ratio_3d",
    "atr_pct",
    "rs_vs_spy",
    "atr_contraction",
    "sma50_slope",
    "sma200_slope",
    "weekly_rsi",
]


def extract_bar_features(bar: pd.Series, vol_sma20: float = 0.0) -> Optional[List[float]]:
    """Extrage vectorul cauzal de 11 caracteristici de pe o lumânare zilnică închisă."""
    try:
        close_p = float(bar["Close"])
        rsi_14 = float(bar["RSI_14"])
        rsi_4 = float(bar.get("RSI_4", 50.0))
        ema_20 = float(bar["EMA_20"])
        sma_50 = float(bar["SMA_50"])
        atr_pct = float(bar.get("ATR_Pct", float(bar.get("ATR_14", close_p * 0.02)) / close_p * 100.0))
        vol_r3 = float(bar.get("Volume_Ratio_3D", 1.0))
        rs_vs_spy = float(bar.get("RS_vs_SPY", 0.0))
        atr_c = float(bar.get("ATR_Contraction", 1.0))
        slope_50 = float(bar.get("SMA_50_Slope", 0.0))
        slope_200 = float(bar.get("SMA_200_Slope", 0.0))
        weekly_rsi = float(bar.get("Weekly_RSI_14", 55.0))

        if close_p <= 0 or ema_20 <= 0 or sma_50 <= 0:
            return None

        vals = [
            rsi_14,
            50.0 if np.isnan(rsi_4) else rsi_4,
            ((close_p - ema_20) / ema_20) * 100.0,
            ((close_p - sma_50) / sma_50) * 100.0,
            1.0 if np.isnan(vol_r3) else vol_r3,
            2.5 if np.isnan(atr_pct) else atr_pct,
            0.0 if np.isnan(rs_vs_spy) else rs_vs_spy,
            1.0 if (np.isnan(atr_c) or atr_c <= 0) else atr_c,
            0.0 if np.isnan(slope_50) else slope_50,
            0.0 if np.isnan(slope_200) else slope_200,
            55.0 if (np.isnan(weekly_rsi) or weekly_rsi <= 0) else weekly_rsi,
        ]
        if any(np.isnan(v) or np.isinf(v) for v in vals):
            return None
        return vals
    except Exception:
        return None


def generate_training_data(
    watchlist: Optional[List[str]] = None,
    period: str = "5y",
    data_map: Optional[Dict[str, pd.DataFrame]] = None,
    spy_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Rulează Backtester-ul fără look-ahead pe universul extins și extrage caracteristicile
    de pe bara `signal_date` (lumânarea închisă `t`), sortate strict cronologic.
    """
    if watchlist is None:
        watchlist = list(dict.fromkeys(list(config.WATCHLIST) + list(config.UNIVERSE_POOL)))

    if spy_df is None:
        try:
            spy_df = fetch_stock_data("SPY", period=period, closed_only=True)
        except Exception:
            spy_df = None

    raw_samples: List[Tuple[pd.Timestamp, str, List[float], int]] = []

    for ticker in watchlist:
        try:
            if data_map is not None and ticker in data_map:
                df = data_map[ticker]
            else:
                df = fetch_stock_data(ticker, period=period, spy_df=spy_df, closed_only=True)

            if len(df) < 250:
                continue

            bt = Backtester(
                initial_capital=config.DEFAULT_SIMULATED_CAPITAL,
                risk_per_trade=config.RISK_PER_TRADE_PERCENT,
                use_ml=False,
            )
            bt.run(ticker, df, spy_df=spy_df)
            if not bt.trades:
                continue

            # Grupăm loturile aceleiași poziții după bara semnalului (`signal_date`)
            sig_pnl: Dict[pd.Timestamp, float] = {}
            for trade in bt.trades:
                sd = pd.Timestamp(trade.get("signal_date", trade["entry_date"]))
                if getattr(sd, "tz", None) is not None:
                    sd = sd.tz_localize(None)
                sig_pnl[sd] = sig_pnl.get(sd, 0.0) + float(trade["pnl"])

            df_idx = df.copy()
            df_idx.index = pd.to_datetime(df_idx.index)
            if df_idx.index.tz is not None:
                df_idx.index = df_idx.index.tz_localize(None)

            for sig_ts, total_pnl in sig_pnl.items():
                if sig_ts not in df_idx.index:
                    continue
                bar = df_idx.loc[sig_ts]
                if isinstance(bar, pd.DataFrame):
                    bar = bar.iloc[0]

                features = extract_bar_features(bar)
                if features is None:
                    continue

                label = 1 if total_pnl > 0 else 0
                sample_key = f"{sig_ts.strftime('%Y-%m-%d')}_{float(bar['Close']):.2f}"
                raw_samples.append((sig_ts, sample_key, features, label))
        except Exception:
            continue

    if not raw_samples:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32), []

    raw_samples.sort(key=lambda item: item[0])
    sample_keys = [s[1] for s in raw_samples]
    all_features = [s[2] for s in raw_samples]
    all_labels = [s[3] for s in raw_samples]

    return np.array(all_features, dtype=np.float32), np.array(all_labels, dtype=np.int32), sample_keys


def _build_pipeline(scale_pos_weight: float = 1.0):
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            max_depth=2,
            learning_rate=0.02,
            n_estimators=100,
            min_child_weight=5,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )),
    ])


def train_and_save_model(X: np.ndarray, y: np.ndarray, sample_keys: Optional[List[str]] = None) -> Dict:
    """
    Evaluează modelul strict Out-of-Sample prin TimeSeriesSplit (5-fold) și activează
    filtrul doar dacă `cv_roc_auc_mean >= ML_MIN_OOS_ROC_AUC`.
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score, precision_score
    import joblib

    min_samples = int(getattr(config, "ML_MIN_TRAINING_SAMPLES", 80))
    if len(X) < min_samples:
        print(f"❌ Eșantioane insuficiente pentru antrenare ML ({len(X)} < {min_samples}).")
        return {}

    wins = int(np.sum(y == 1))
    losses = int(np.sum(y == 0))
    wr = round(wins / len(y) * 100.0, 1) if len(y) > 0 else 0.0
    final_scale = float(losses / wins) if (wins > 0 and losses > wins) else 1.0

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

        fold_pipe = _build_pipeline(scale_pos_weight=fold_scale)
        fold_pipe.fit(X_train, y_train)

        y_prob = fold_pipe.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= float(getattr(config, "ML_MIN_CONFIDENCE", 0.53))).astype(int)

        if sample_keys is not None:
            for idx_pos, prob_val in zip(val_idx, y_prob):
                if idx_pos < len(sample_keys):
                    oos_predictions[sample_keys[idx_pos]] = round(float(prob_val), 4)

        if len(np.unique(y_val)) >= 2:
            auc_scores.append(float(roc_auc_score(y_val, y_prob)))
        precision_scores.append(float(precision_score(y_val, y_pred, zero_division=0)))

    mean_auc = float(np.nanmean(auc_scores)) if auc_scores else 0.50
    std_auc = float(np.nanstd(auc_scores)) if auc_scores else 0.0
    mean_prec = float(np.mean(precision_scores)) if precision_scores else 0.0
    std_prec = float(np.std(precision_scores)) if precision_scores else 0.0

    min_req_auc = float(getattr(config, "ML_MIN_OOS_ROC_AUC", 0.55))
    has_edge = bool(mean_auc >= min_req_auc and (mean_prec * 100.0) >= wr)

    pipeline = _build_pipeline(scale_pos_weight=final_scale)
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)

    meta = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "XGBClassifier_V2",
        "n_samples": len(X),
        "n_wins": wins,
        "n_losses": losses,
        "win_rate_pct": wr,
        "scale_pos_weight": round(final_scale, 4),
        "cv_roc_auc_mean": round(mean_auc, 4),
        "cv_roc_auc_std": round(std_auc, 4),
        "cv_precision_mean": round(mean_prec, 4),
        "cv_precision_std": round(std_prec, 4),
        "has_predictive_edge": has_edge,
        "min_required_oos_auc": min_req_auc,
        "features": FEATURE_NAMES,
        "oos_predictions": oos_predictions,
    }
    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    edge_str = "🟢 ACTIV (Avantaj OOS confirmat)" if has_edge else f"⚪ STANDBY NEUTRU (ROC-AUC {mean_auc:.3f} < {min_req_auc:.2f} — nu blochează semnalele)"
    print(f"✅ Model ML V2 salvat | WinRate bază: {wr}% | OOS ROC-AUC: {mean_auc:.3f} | Status Filtru: {edge_str}")
    return meta


def run_ml_training(watchlist=None, data_map=None, spy_df=None):
    """Punct de intrare pentru antrenarea modelului ML V2 fără Data Leakage."""
    print("=" * 68)
    print("   🧠 ANTRENARE MODEL ML V2 — ZERO DATA LEAKAGE & POARTĂ OOS")
    print("=" * 68)

    X, y, sample_keys = generate_training_data(watchlist=watchlist, data_map=data_map, spy_df=spy_df)
    if len(X) == 0:
        print("❌ Nu s-au putut genera eșantioane de antrenare.")
        return

    meta = train_and_save_model(X, y, sample_keys=sample_keys)
    if not meta:
        return

    try:
        import ml_predictor
        ml_predictor._predictor_instance = None
    except Exception:
        pass


if __name__ == "__main__":
    run_ml_training()
