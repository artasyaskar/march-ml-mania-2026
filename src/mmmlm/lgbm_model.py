from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class LGBMConfig:
    num_leaves: int = 64
    min_data_in_leaf: int = 80
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    learning_rate: float = 0.03
    n_estimators: int = 2000
    reg_lambda: float = 2.0


def _fit_isotonic(y_true: np.ndarray, y_pred: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_pred, y_true)
    return iso


def train_lgbm_oof(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cfg: LGBMConfig | None = None,
    seed: int = 42,
) -> dict:
    cfg = cfg or LGBMConfig()
    gkf = GroupKFold(n_splits=min(5, int(groups.nunique())))

    oof = np.zeros(len(X), dtype=float)
    models: list[lgb.LGBMClassifier] = []

    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            min_data_in_leaf=cfg.min_data_in_leaf,
            feature_fraction=cfg.feature_fraction,
            bagging_fraction=cfg.bagging_fraction,
            bagging_freq=cfg.bagging_freq,
            reg_lambda=cfg.reg_lambda,
            random_state=seed,
            n_jobs=-1,
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False)],
        )

        p = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = p
        models.append(model)

    iso = _fit_isotonic(y.to_numpy(dtype=float), oof)
    oof_cal = iso.predict(oof)

    return {
        "models": models,
        "iso": iso,
        "brier_raw": float(mean_squared_error(y, oof)),
        "brier_cal": float(mean_squared_error(y, oof_cal)),
        "feature_names": list(X.columns),
        "seed": seed,
    }


def predict_lgbm_ensemble(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    feats = bundle["feature_names"]
    X2 = X[feats]
    ps = []
    for m in bundle["models"]:
        ps.append(m.predict_proba(X2)[:, 1])
    p = np.mean(ps, axis=0)
    iso: IsotonicRegression = bundle["iso"]
    return iso.predict(p)
