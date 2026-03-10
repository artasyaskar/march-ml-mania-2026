from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mmmlm.data_paths import DataPaths
from mmmlm.efficiency import compute_team_season_efficiency, compute_team_season_efficiency_recency
from mmmlm.features import EloConfig, compute_elo_table, make_seed_features, season_team_skeleton
from mmmlm.lgbm_model import LGBMConfig, train_lgbm_oof
from mmmlm.massey import compute_massey_features


def _load_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def _build_team_table(data_root: Path, prefix: str, start_season: int, end_season: int) -> pd.DataFrame:
    paths = DataPaths(data_root)

    rs_compact = _load_csv(paths.file(f"{prefix}RegularSeasonCompactResults.csv"))
    rs_detailed = _load_csv(paths.file(f"{prefix}RegularSeasonDetailedResults.csv"))
    seeds = _load_csv(paths.file(f"{prefix}NCAATourneySeeds.csv"))

    rs_compact = rs_compact[(rs_compact.Season >= start_season) & (rs_compact.Season <= end_season)].copy()
    rs_detailed = rs_detailed[(rs_detailed.Season >= start_season) & (rs_detailed.Season <= end_season)].copy()
    seeds = seeds[(seeds.Season >= start_season) & (seeds.Season <= end_season)].copy()

    elo = compute_elo_table(
        rs_compact,
        cfg=EloConfig(k=18.0, home_adv=55.0, margin_mult=0.0, reg_to_mean=0.12),
    )
    eff = compute_team_season_efficiency(rs_detailed)
    rec = compute_team_season_efficiency_recency(rs_detailed, last_n_games=10, last_day_window=30, exp_halflife_days=15.0)
    seed_feat = make_seed_features(seeds)

    teams = _load_csv(paths.file(f"{prefix}Teams.csv"))[["TeamID"]].copy()
    seasons = np.arange(start_season, end_season + 1)
    skel = season_team_skeleton(teams, seasons)

    team = skel.merge(elo, on=["Season", "TeamID"], how="left")
    team = team.merge(eff, on=["Season", "TeamID"], how="left")
    team = team.merge(rec, on=["Season", "TeamID"], how="left")
    team = team.merge(seed_feat, on=["Season", "TeamID"], how="left")

    # Fill seed missing
    team["SeedNum"] = team["SeedNum"].fillna(20).astype(float)
    team["SeedPlayIn"] = team["SeedPlayIn"].fillna(0).astype(float)

    # Fill missing Elo with prior season (then global mean)
    team = team.sort_values(["TeamID", "Season"]).reset_index(drop=True)
    team["Elo"] = team.groupby("TeamID")["Elo"].ffill()
    team["Elo"] = team["Elo"].fillna(team["Elo"].mean())

    # Fill efficiency/recency with prior season then global mean (keeps coverage)
    num_cols = [c for c in team.columns if c not in ["Season", "TeamID"]]
    for c in num_cols:
        if c in ["SeedNum", "SeedPlayIn", "Elo"]:
            continue
        team[c] = team.groupby("TeamID")[c].ffill()
        team[c] = team[c].fillna(team[c].mean())

    team = team.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return team


def _prepare_training(
    data_root: Path,
    prefix: str,
    start_season: int,
    end_season: int,
    use_detailed: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    paths = DataPaths(data_root)

    # Kept for backwards-compat; the upgraded pipeline always uses compact for Elo
    # and detailed regular season stats for efficiency features.
    tr = _load_csv(paths.file(f"{prefix}NCAATourneyCompactResults.csv"))

    tr = tr[(tr.Season >= start_season) & (tr.Season <= end_season)].copy()

    team = _build_team_table(data_root, prefix, start_season, end_season)

    # Men only: add Massey features, aligned to the final pre-tournament ranking day (133)
    if prefix == "M":
        mm = _load_csv(paths.file("MMasseyOrdinals.csv"))
        massey_rows = []
        for s in range(start_season, end_season + 1):
            massey_rows.append(compute_massey_features(mm, s, max_ranking_day=133))
        massey = pd.concat(massey_rows, axis=0, ignore_index=True)
        team = team.merge(massey, on=["Season", "TeamID"], how="left")
        for c in ["MasseyMean", "MasseyStd", "MasseyBest", "MasseyN"]:
            if c in team.columns:
                team[c] = team[c].fillna(0.0).astype(float)

    # Build training examples from tourney games: predict if TeamID_A (lower) wins
    g = tr[["Season", "DayNum", "WTeamID", "LTeamID"]].copy()
    g["TeamID_A"] = g[["WTeamID", "LTeamID"]].min(axis=1).astype(int)
    g["TeamID_B"] = g[["WTeamID", "LTeamID"]].max(axis=1).astype(int)
    g["y"] = (g["WTeamID"] == g["TeamID_A"]).astype(int)

    g = g.merge(team.add_prefix("A_").rename(columns={"A_Season": "Season", "A_TeamID": "TeamID_A"}), on=["Season", "TeamID_A"], how="left")
    g = g.merge(team.add_prefix("B_").rename(columns={"B_Season": "Season", "B_TeamID": "TeamID_B"}), on=["Season", "TeamID_B"], how="left")

    g["EloDiff"] = g["A_Elo"] - g["B_Elo"]
    g["SeedDiff"] = g["B_SeedNum"] - g["A_SeedNum"]
    g["PlayInDiff"] = g["A_SeedPlayIn"] - g["B_SeedPlayIn"]

    g["NetRtgDiff"] = g["A_NetRtg"] - g["B_NetRtg"]
    g["OffRtgDiff"] = g["A_OffRtg"] - g["B_OffRtg"]
    g["DefRtgDiff"] = g["A_DefRtg"] - g["B_DefRtg"]
    g["TempoDiff"] = g["A_Tempo"] - g["B_Tempo"]

    g["eFGDiff"] = g["A_eFG"] - g["B_eFG"]
    g["TOVPctDiff"] = g["A_TOVPct"] - g["B_TOVPct"]
    g["ORBPctDiff"] = g["A_ORBPct"] - g["B_ORBPct"]
    g["FTRDiff"] = g["A_FTR"] - g["B_FTR"]

    # Recency features (may be filled via priors in team table)
    rec_cols = [c for c in g.columns if c.startswith("A_L10_") or c.startswith("B_L10_") or c.startswith("A_D30_") or c.startswith("B_D30_") or c.startswith("A_EXP_") or c.startswith("B_EXP_")]
    for c in rec_cols:
        pass

    # Explicit diffs for recency stats
    for stat in [
        "WinPct",
        "Tempo",
        "OffRtg",
        "DefRtg",
        "NetRtg",
        "eFG",
        "TOVPct",
        "ORBPct",
        "FTR",
    ]:
        for pfx in ["L10_", "D30_", "EXP_"]:
            a = f"A_{pfx}{stat}"
            b = f"B_{pfx}{stat}"
            if a in g.columns and b in g.columns:
                g[f"{pfx}{stat}Diff"] = g[a] - g[b]

    if prefix == "M":
        g["MasseyMeanDiff"] = g["A_MasseyMean"] - g["B_MasseyMean"]
        g["MasseyBestDiff"] = g["A_MasseyBest"] - g["B_MasseyBest"]
        g["MasseyNDiff"] = g["A_MasseyN"] - g["B_MasseyN"]

    base_cols = ["EloDiff", "SeedDiff", "PlayInDiff"]
    eff_cols = [
        "NetRtgDiff",
        "OffRtgDiff",
        "DefRtgDiff",
        "TempoDiff",
        "eFGDiff",
        "TOVPctDiff",
        "ORBPctDiff",
        "FTRDiff",
    ]
    rec_diff_cols = [c for c in g.columns if c.endswith("Diff") and (c.startswith("L10_") or c.startswith("D30_") or c.startswith("EXP_"))]
    massey_cols = [c for c in ["MasseyMeanDiff", "MasseyBestDiff", "MasseyNDiff"] if c in g.columns]

    X = g[base_cols + eff_cols + rec_diff_cols + massey_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = g["y"].copy()
    groups = g["Season"].copy()
    return X, y, groups


def _train_model(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> Pipeline:
    base = LogisticRegression(max_iter=2000, C=0.6, solver="lbfgs")
    # group-aware calibration CV to reduce overconfidence; small n => use fewer folds
    n_splits = min(5, int(groups.nunique()))
    # IMPORTANT: do not pass a generator (e.g., gkf.split(...)) into cv; it breaks pickling.
    # CalibratedClassifierCV can accept an int (StratifiedKFold), which keeps the estimator serializable.
    clf = CalibratedClassifierCV(base, method="isotonic", cv=n_splits)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", clf),
        ]
    )
    pipe.fit(X, y)
    return pipe


def _cv_brier_model_fn(model_fn, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> float:
    gkf = GroupKFold(n_splits=min(5, int(groups.nunique())))
    preds = np.zeros(len(X), dtype=float)

    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        model = model_fn(X.iloc[tr_idx], y.iloc[tr_idx], groups.iloc[tr_idx])
        preds[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]

    return float(mean_squared_error(y, preds))


def _oof_preds(model_fn, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> np.ndarray:
    gkf = GroupKFold(n_splits=min(5, int(groups.nunique())))
    preds = np.zeros(len(X), dtype=float)
    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        model = model_fn(X.iloc[tr_idx], y.iloc[tr_idx], groups.iloc[tr_idx])
        preds[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
    return preds


def _fit_meta_blender(oof_stack: pd.DataFrame, y: pd.Series) -> Pipeline:
    # Simple, strong blender for Brier: logistic regression on component probabilities.
    base = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs")
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", base)])
    pipe.fit(oof_stack, y)
    return pipe


def _oof_lgbm_preds(X: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int) -> tuple[np.ndarray, dict]:
    gkf = GroupKFold(n_splits=min(5, int(groups.nunique())))
    oof = np.zeros(len(X), dtype=float)
    fold_models = []

    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
        fold_bundle = train_lgbm_oof(X_tr, y_tr, groups.iloc[tr_idx], cfg=LGBMConfig(), seed=seed)
        p_va = np.mean([m.predict_proba(X_va[fold_bundle["feature_names"]])[:, 1] for m in fold_bundle["models"]], axis=0)
        p_va = fold_bundle["iso"].predict(p_va)
        oof[va_idx] = p_va
        fold_models.append(fold_bundle)

    # Return oof preds and per-fold bundles (for potential diagnostics)
    return oof, {"fold_bundles": fold_models}


def _cv_brier(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> float:
    gkf = GroupKFold(n_splits=min(5, groups.nunique()))
    preds = np.zeros(len(X), dtype=float)

    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        model = _train_model(X.iloc[tr_idx], y.iloc[tr_idx], groups.iloc[tr_idx])
        preds[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]

    return mean_squared_error(y, preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--outdir", type=str, default=str(Path(__file__).resolve().parents[2] / "artifacts"))
    ap.add_argument("--start-season", type=int, default=2003)
    ap.add_argument("--end-season", type=int, default=2025)
    ap.add_argument("--use-detailed", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    artifacts = {}
    metrics = {}
    for prefix in ["M", "W"]:
        X, y, groups = _prepare_training(
            data_root=data_root,
            prefix=prefix,
            start_season=args.start_season,
            end_season=args.end_season,
            use_detailed=args.use_detailed,
        )

        # Baseline Elo+Seed logit
        X_elo = X[["EloDiff", "SeedDiff", "PlayInDiff"]].copy()
        elo_model = _train_model(X_elo, y, groups)
        brier_elo = _cv_brier_model_fn(_train_model, X_elo, y, groups)

        # Efficiency + (optional) Massey logit
        eff_cols = [c for c in X.columns if c not in ["EloDiff", "SeedDiff", "PlayInDiff"]]
        X_eff = X[eff_cols].copy()
        eff_model = _train_model(X_eff, y, groups)
        brier_eff = _cv_brier_model_fn(_train_model, X_eff, y, groups)

        # LightGBM on full features with season-based OOF + isotonic calibration
        lgbm = train_lgbm_oof(X, y, groups, cfg=LGBMConfig(), seed=args.seed)

        # OOF stacking for blender
        oof_elo = _oof_preds(_train_model, X_elo, y, groups)
        oof_eff = _oof_preds(_train_model, X_eff, y, groups)
        oof_lgbm, lgbm_oof_meta = _oof_lgbm_preds(X, y, groups, seed=args.seed)

        stack = pd.DataFrame({"p_elo": oof_elo, "p_eff": oof_eff, "p_lgbm": oof_lgbm})
        meta = _fit_meta_blender(stack, y)

        artifacts[prefix] = {
            "elo_model": elo_model,
            "eff_model": eff_model,
            "lgbm": lgbm,
            "meta": meta,
            "meta_cols": ["p_elo", "p_eff", "p_lgbm"],
            "lgbm_oof_meta": lgbm_oof_meta,
            "feature_names": list(X.columns),
            "eff_cols": eff_cols,
        }
        metrics[prefix] = {
            "brier_elo": float(brier_elo),
            "brier_eff": float(brier_eff),
            "brier_lgbm_raw": float(lgbm["brier_raw"]),
            "brier_lgbm_cal": float(lgbm["brier_cal"]),
            "brier_meta": float(mean_squared_error(y, meta.predict_proba(stack)[:, 1])),
            "n": int(len(X)),
            "seasons": int(groups.nunique()),
        }

    joblib.dump({"artifacts": artifacts, "metrics": metrics}, outdir / "models.joblib")
    pd.DataFrame.from_dict(metrics, orient="index").to_csv(outdir / "cv_metrics.csv")


if __name__ == "__main__":
    main()
