"""
Modul de predicție ML V2 cu Poartă Automată de Calitate Out-of-Sample (Zero Data Leakage).
  - Activează respingerea semnalelor DOAR dacă `cv_roc_auc_mean >= ML_MIN_OOS_ROC_AUC`.
  - În Backtest folosește EXCLUSIV predicțiile Out-of-Sample din `TimeSeriesSplit`
    (nu evaluează niciodată Fold 0 pe modelul antrenat pe întregul istoric).
"""

import warnings
warnings.filterwarnings("ignore")

import json
from typing import Tuple, Optional
import numpy as np
import pandas as pd

import config
from ml_model import extract_bar_features, MODEL_PATH, MODEL_META_PATH, FEATURE_NAMES


class MLPredictor:
    def __init__(self):
        self._model = None
        self._meta: dict = {}
        self._ready: bool = False
        self._load()

    def _load(self):
        try:
            import joblib
            if not MODEL_PATH.exists() or not MODEL_META_PATH.exists():
                return

            with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
                self._meta = json.load(f)

            if self._meta.get("features") != FEATURE_NAMES:
                self._ready = False
                return

            self._model = joblib.load(MODEL_PATH)
            self._ready = True
        except Exception:
            self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready and self._model is not None

    @property
    def has_predictive_edge(self) -> bool:
        """Returnează True DOAR dacă modelul a depășit pragul statistic Out-of-Sample."""
        if not self.is_ready:
            return False
        min_auc = float(getattr(config, "ML_MIN_OOS_ROC_AUC", 0.55))
        min_samples = int(getattr(config, "ML_MIN_TRAINING_SAMPLES", 80))
        auc = float(self._meta.get("cv_roc_auc_mean", 0.0))
        n_samp = int(self._meta.get("n_samples", 0))
        base_wr = float(self._meta.get("win_rate_pct", 50.0)) / 100.0
        prec = float(self._meta.get("cv_precision_mean", 0.0))
        return (n_samp >= min_samples) and (auc >= min_auc) and (prec >= base_wr)

    @property
    def meta(self) -> dict:
        return self._meta

    def _predict_from_features(self, features: list) -> float:
        X = np.array(features, dtype=np.float32).reshape(1, -1)
        proba = self._model.predict_proba(X)[0]
        return float(proba[1])

    def predict(self, df: pd.DataFrame) -> Tuple[float, bool]:
        if not self.has_predictive_edge:
            return 0.0, True

        try:
            bar = df.iloc[-1]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[0]

            features = extract_bar_features(bar)
            if features is None:
                return 0.0, True

            confidence = self._predict_from_features(features)
            threshold = float(getattr(config, "ML_MIN_CONFIDENCE", 0.53))
            return round(confidence, 3), (confidence >= threshold)
        except Exception:
            return 0.0, True

    def predict_for_backtest(self, bar: pd.Series, vol_sma20: float, bar_date_str: str) -> Tuple[float, bool]:
        """
        În Backtest folosim STRICT predicțiile Out-of-Sample (`oos_predictions`).
        Dacă o bară timpurie nu se află în `oos_predictions`, returnăm neutru (fără In-Sample Leakage!).
        """
        if not self.has_predictive_edge:
            return 0.55, True

        try:
            threshold = float(getattr(config, "ML_MIN_CONFIDENCE", 0.53))
            oos_map = self._meta.get("oos_predictions", {})
            sample_key = f"{bar_date_str}_{float(bar['Close']):.2f}"
            if sample_key in oos_map:
                conf = float(oos_map[sample_key])
                return round(conf, 3), (conf >= threshold)
            return 0.55, True
        except Exception:
            return 0.55, True

    def feature_importances(self) -> Optional[dict]:
        if not self.is_ready:
            return None
        try:
            clf = self._model.named_steps["clf"] if hasattr(self._model, "named_steps") else self._model
            importances = getattr(clf, "feature_importances_", None)
            if importances is None:
                return None
            feat_names = self._meta.get("features", FEATURE_NAMES)[:len(importances)]
            return dict(sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True))
        except Exception:
            return None

    def print_status(self):
        if not self.is_ready:
            print("🔴 Modelul ML nu este antrenat pe arhitectura V2.")
            return
        edge_lbl = "🟢 ACTIV (Avantaj OOS validat)" if self.has_predictive_edge else "⚪ NEUTRU (Sub pragul ROC-AUC OOS — nu blochează semnalele)"
        print("\n" + "=" * 64)
        print("   🧠 STATUS MODEL ML (XGBOOST V2 — ANTI-LEAKAGE)")
        print("=" * 64)
        print(f"   • Status Filtru    : {edge_lbl}")
        print(f"   • Antrenat la      : {self._meta.get('trained_at', 'N/A')}")
        print(f"   • Eșantioane       : {self._meta.get('n_samples', 'N/A')} ({len(self._meta.get('oos_predictions', {}))} OOS)")
        print(f"   • Win Rate de bază : {self._meta.get('win_rate_pct', '?')}%")
        print(f"   • ROC-AUC (TS-CV)  : {self._meta.get('cv_roc_auc_mean', 0.0):.3f} (prag activare: {getattr(config, 'ML_MIN_OOS_ROC_AUC', 0.55):.2f})")
        print(f"   • Precision (TS-CV): {self._meta.get('cv_precision_mean', 0.0):.3f}")
        print("=" * 64)


_predictor_instance: Optional[MLPredictor] = None


def get_predictor() -> MLPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = MLPredictor()
    return _predictor_instance


def ensure_ml_ready(max_age_days: int = 7) -> MLPredictor:
    global _predictor_instance
    pred = get_predictor()
    needs_training = not pred.is_ready

    if pred.is_ready:
        trained_str = pred.meta.get("trained_at", "")
        if trained_str:
            try:
                from datetime import datetime
                trained_dt = datetime.strptime(trained_str[:19], "%Y-%m-%d %H:%M:%S")
                if (datetime.now() - trained_dt).days > max_age_days:
                    needs_training = True
            except Exception:
                pass

    if needs_training:
        try:
            from ml_model import run_ml_training
            run_ml_training()
            _predictor_instance = MLPredictor()
            pred = _predictor_instance
        except Exception as e:
            print(f"⚠️ Auto-ML indisponibil ({e}), se continuă pe scorul cantitativ pur.")

    return pred
