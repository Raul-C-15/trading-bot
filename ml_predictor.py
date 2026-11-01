"""
Modul de predicție ML în timp real și în Backtest (fără Data Leakage).
Încarcă modelul antrenat XGBoost (ml_model.pkl) și evaluează calitatea
unui semnal live înainte ca botul să decidă intrarea.
"""

import warnings
warnings.filterwarnings("ignore")

import json
from typing import Tuple, Optional
import numpy as np
import pandas as pd

import config
from ml_model import extract_bar_features, MODEL_PATH, MODEL_META_PATH, FEATURE_NAMES

ML_MIN_CONFIDENCE = getattr(config, "ML_MIN_CONFIDENCE", 0.52)


class MLPredictor:
    """
    Evaluator live și istoric al calității semnalelor de trading.
    Folosește modelul XGBoost antrenat pe date istorice și cache-ul Out-of-Sample.
    """

    def __init__(self):
        self._model  = None
        self._meta:  dict = {}
        self._ready: bool = False
        self._load()

    def _load(self):
        """Încearcă să încarce modelul și metadatele salvate."""
        try:
            import joblib
            if not MODEL_PATH.exists():
                print("ℹ️  Model ML negăsit. Alege opțiunea 7 din meniu pentru antrenare.")
                return

            self._model = joblib.load(MODEL_PATH)

            if MODEL_META_PATH.exists():
                with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
                    self._meta = json.load(f)

            self._ready = True
            trained_at = self._meta.get("trained_at", "necunoscut")
            roc        = self._meta.get("cv_roc_auc_mean", 0.0)
            prec       = self._meta.get("cv_precision_mean", 0.0)
            n_samples  = self._meta.get("n_samples", "?")
            model_type = self._meta.get("model_type", "XGBoost")
            print(
                f"🧠 Model ML activ ({model_type})  |  antrenat: {trained_at}  |  "
                f"ROC-AUC: {roc:.3f}  |  Precision: {prec:.3f}  |  {n_samples} eșantioane"
            )
        except Exception as e:
            print(f"⚠️ Nu s-a putut încărca modelul ML: {e}")
            self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready and self._model is not None

    @property
    def meta(self) -> dict:
        return self._meta

    def _predict_from_features(self, features: list) -> float:
        """Evaluează vectorul de features asigurând compatibilitatea numărului de coloane."""
        expected_n = getattr(self._model, "n_features_in_", len(features))
        feats_to_use = features[:expected_n]
        X = np.array(feats_to_use, dtype=np.float32).reshape(1, -1)
        proba = self._model.predict_proba(X)[0]
        return float(proba[1])

    def predict(self, df: pd.DataFrame) -> Tuple[float, bool]:
        """
        Analizează ultimul bar din DataFrame și returnează:
          confidence    : probabilitatea estimată de WIN (0.0 – 1.0)
          is_confident  : True dacă depășește ML_MIN_CONFIDENCE din config
        """
        if not self.is_ready:
            return 0.0, True

        try:
            bar = df.iloc[-1]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[0]

            vol_sma20 = float(bar.get("Volume_SMA_20", bar["Volume"]))
            features  = extract_bar_features(bar, vol_sma20)

            if features is None:
                return 0.0, True

            confidence   = self._predict_from_features(features)
            threshold    = getattr(config, "ML_MIN_CONFIDENCE", ML_MIN_CONFIDENCE)
            is_confident = confidence >= threshold

            return round(confidence, 3), is_confident

        except Exception:
            return 0.0, True

    def predict_for_backtest(self, bar: pd.Series, vol_sma20: float, bar_date_str: str) -> Tuple[float, bool]:
        """
        Evaluează o lumânare în timpul Backtestului:
          - Dacă există o predicție Out-of-Sample (OOS) salvată din TimeSeriesSplit pentru acea bară,
            o folosește direct pentru a elimina 100% In-Sample Data Leakage!
          - Altfel, evaluează cu modelul curent.
        """
        if not self.is_ready:
            return 0.55, True

        try:
            threshold = getattr(config, "ML_MIN_CONFIDENCE", ML_MIN_CONFIDENCE)
            oos_map = self._meta.get("oos_predictions", {})
            sample_key = f"{bar_date_str}_{float(bar['Close']):.2f}"

            if sample_key in oos_map:
                conf = float(oos_map[sample_key])
                return round(conf, 3), (conf >= threshold)

            features = extract_bar_features(bar, vol_sma20)
            if features is None:
                return 0.55, True

            conf = self._predict_from_features(features)
            return round(conf, 3), (conf >= threshold)
        except Exception:
            return 0.55, True

    def feature_importances(self) -> Optional[dict]:
        """Returnează importanța fiecărei caracteristici din modelul XGBoost."""
        if not self.is_ready:
            return None
        try:
            if hasattr(self._model, "named_steps") and "clf" in self._model.named_steps:
                clf = self._model.named_steps["clf"]
            else:
                clf = self._model

            importances = getattr(clf, "feature_importances_", None)
            if importances is None:
                return None

            feat_names = self._meta.get("features", FEATURE_NAMES)[:len(importances)]
            return dict(sorted(
                zip(feat_names, importances),
                key=lambda x: x[1], reverse=True
            ))
        except Exception:
            return None

    def print_status(self):
        """Afișează un rezumat al stării modelului ML."""
        if not self.is_ready:
            print("🔴 Modelul ML nu este activ.")
            return
        print("\n" + "=" * 60)
        print("   🧠 STATUS MODEL ML (XGBOOST PRO)")
        print("=" * 60)
        print(f"   • Tip model        : {self._meta.get('model_type', 'XGBoost')}")
        print(f"   • Antrenat la      : {self._meta.get('trained_at', 'N/A')}")
        print(f"   • Eșantioane       : {self._meta.get('n_samples', 'N/A')} ({len(self._meta.get('oos_predictions', {}))} predicții OOS)")
        print(f"   • WIN / LOSS       : {self._meta.get('n_wins', '?')} / {self._meta.get('n_losses', '?')}")
        print(f"   • Win Rate istoric : {self._meta.get('win_rate_pct', '?')}%")
        print(f"   • ROC-AUC (TS-CV)  : {self._meta.get('cv_roc_auc_mean', 0.0):.3f}")
        print(f"   • Precision (TS-CV): {self._meta.get('cv_precision_mean', 0.0):.3f}")
        print(f"   • Prag minim conf. : {int(getattr(config, 'ML_MIN_CONFIDENCE', 0.52)*100)}%")

        imps = self.feature_importances()
        if imps:
            print("\n   📊 Importanța caracteristicilor (Top 8):")
            for i, (feat, imp) in enumerate(list(imps.items())[:8], 1):
                bar_len = int(imp * 40)
                bar_vis = "█" * bar_len
                print(f"   {i}. {feat:<22} {bar_vis} {imp:.3f}")
        print("=" * 60)


_predictor_instance: Optional[MLPredictor] = None


def get_predictor() -> MLPredictor:
    """Returnează instanța singleton a predictor-ului ML (lazy init)."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = MLPredictor()
    return _predictor_instance


def ensure_ml_ready(max_age_days: int = 7) -> MLPredictor:
    """
    Verifică automat dacă modelul ML există și dacă a fost antrenat în ultimele `max_age_days` zile.
    Dacă lipsește sau este învechit, declanșează automat antrenarea în fundal și reîncarcă predictorul.
    """
    global _predictor_instance
    pred = get_predictor()
    needs_training = not pred.is_ready

    if pred.is_ready:
        trained_str = pred.meta.get("trained_at", "")
        if trained_str:
            try:
                from datetime import datetime
                trained_dt = datetime.strptime(trained_str[:19], "%Y-%m-%d %H:%M:%S")
                age_days = (datetime.now() - trained_dt).days
                if age_days > max_age_days:
                    print(f"🔄 Modelul ML are {age_days} zile vechime (> {max_age_days}z). Se actualizează automat...")
                    needs_training = True
            except Exception:
                pass

    if needs_training:
        try:
            print("🧠 Auto-ML: Se antrenează automat modelul XGBoost...")
            from ml_model import run_ml_training
            run_ml_training()
            _predictor_instance = MLPredictor()
            pred = _predictor_instance
        except Exception as e:
            print(f"⚠️ Auto-ML nu a putut reantrena modelul ({e}), se continuă fără filtru ML.")

    return pred
