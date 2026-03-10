from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mmmlm.data_paths import DataPaths
from mmmlm.efficiency import compute_team_season_efficiency, compute_team_season_efficiency_recency
from mmmlm.features import EloConfig, build_matchup_frame, compute_elo_table, make_seed_features
from mmmlm.lgbm_model import predict_lgbm_ensemble
from mmmlm.massey import compute_massey_features


def _load_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def _build_team_table_for_season(data_root: Path, prefix: str, season: int, cfg: EloConfig) -> pd.DataFrame:
    paths = DataPaths(data_root)

    rs = _load_csv(paths.file(f"{prefix}RegularSeasonCompactResults.csv"))
    rs_det = _load_csv(paths.file(f"{prefix}RegularSeasonDetailedResults.csv"))
    seeds = _load_csv(paths.file(f"{prefix}NCAATourneySeeds.csv"))

    rs = rs[rs.Season == season].copy()
    rs_det = rs_det[rs_det.Season == season].copy()
    seeds = seeds[seeds.Season == season].copy()

    elo = compute_elo_table(rs, cfg=cfg)
    elo = elo[elo.Season == season].copy()
    eff = compute_team_season_efficiency(rs_det)
    rec = compute_team_season_efficiency_recency(rs_det, last_n_games=10, last_day_window=30, exp_halflife_days=15.0)
    seed_feat = make_seed_features(seeds)

    team = elo.merge(eff, on=["Season", "TeamID"], how="left")
    team = team.merge(rec, on=["Season", "TeamID"], how="left")
    team = team.merge(seed_feat, on=["Season", "TeamID"], how="left")

    team["SeedNum"] = team["SeedNum"].fillna(20).astype(float)
    team["SeedPlayIn"] = team["SeedPlayIn"].fillna(0).astype(float)
    team = team.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if prefix == "M":
        mm = _load_csv(paths.file("MMasseyOrdinals.csv"))
        massey = compute_massey_features(mm, season, max_ranking_day=133)
        team = team.merge(massey, on=["Season", "TeamID"], how="left")
        for c in ["MasseyMean", "MasseyStd", "MasseyBest", "MasseyN"]:
            if c in team.columns:
                team[c] = team[c].fillna(0.0).astype(float)

    return team


def _blend(p_elo: np.ndarray, p_eff: np.ndarray, p_lgbm: np.ndarray) -> np.ndarray:
    return 0.20 * p_elo + 0.15 * p_eff + 0.65 * p_lgbm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--artifacts", type=str, default=str(Path(__file__).resolve().parents[2] / "artifacts" / "models.joblib"))
    ap.add_argument("--sample", type=str, default=str(Path(__file__).resolve().parents[2] / "Competition_data" / "SampleSubmissionStage2.csv"))
    ap.add_argument("--out", type=str, default=str(Path(__file__).resolve().parents[2] / "submission.csv"))
    ap.add_argument("--season", type=int, default=2026)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    bundle = joblib.load(args.artifacts)
    artifacts = bundle.get("artifacts")
    if artifacts is None:
        raise ValueError("Expected upgraded artifacts bundle. Re-run: py -m mmmlm.train")

    sample = pd.read_csv(args.sample)
    matchups = build_matchup_frame(sample)
    matchups = matchups[matchups.Season == args.season].copy()

    preds = np.full(len(matchups), 0.5, dtype=float)

    # Men
    men = matchups[(matchups.TeamID_A < 2000) & (matchups.TeamID_B < 2000)].copy()
    if len(men):
        team = _build_team_table_for_season(
            data_root=data_root,
            prefix="M",
            season=args.season,
            cfg=EloConfig(k=18.0, home_adv=55.0, margin_mult=0.0, reg_to_mean=0.12),
        )
        a = team.add_prefix("A_").rename(columns={"A_Season": "Season", "A_TeamID": "TeamID_A"})
        b = team.add_prefix("B_").rename(columns={"B_Season": "Season", "B_TeamID": "TeamID_B"})
        men = men.merge(a, on=["Season", "TeamID_A"], how="left").merge(b, on=["Season", "TeamID_B"], how="left")

        men["EloDiff"] = men["A_Elo"] - men["B_Elo"]
        men["SeedDiff"] = men["B_SeedNum"] - men["A_SeedNum"]
        men["PlayInDiff"] = men["A_SeedPlayIn"] - men["B_SeedPlayIn"]
        men["NetRtgDiff"] = men["A_NetRtg"] - men["B_NetRtg"]
        men["OffRtgDiff"] = men["A_OffRtg"] - men["B_OffRtg"]
        men["DefRtgDiff"] = men["A_DefRtg"] - men["B_DefRtg"]
        men["TempoDiff"] = men["A_Tempo"] - men["B_Tempo"]
        men["eFGDiff"] = men["A_eFG"] - men["B_eFG"]
        men["TOVPctDiff"] = men["A_TOVPct"] - men["B_TOVPct"]
        men["ORBPctDiff"] = men["A_ORBPct"] - men["B_ORBPct"]
        men["FTRDiff"] = men["A_FTR"] - men["B_FTR"]
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
                if a in men.columns and b in men.columns:
                    men[f"{pfx}{stat}Diff"] = men[a] - men[b]
        if "A_MasseyMean" in men.columns:
            men["MasseyMeanDiff"] = men["A_MasseyMean"] - men["B_MasseyMean"]
            men["MasseyBestDiff"] = men["A_MasseyBest"] - men["B_MasseyBest"]
            men["MasseyNDiff"] = men["A_MasseyN"] - men["B_MasseyN"]

        art = artifacts["M"]
        feat_names = art["feature_names"]
        X_full = men[feat_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        p_elo = art["elo_model"].predict_proba(X_full[["EloDiff", "SeedDiff", "PlayInDiff"]])[:, 1]
        p_eff = art["eff_model"].predict_proba(X_full[art["eff_cols"]])[:, 1]
        p_lgbm = predict_lgbm_ensemble(art["lgbm"], X_full)

        stack = pd.DataFrame({"p_elo": p_elo, "p_eff": p_eff, "p_lgbm": p_lgbm})
        preds[men.index.values] = art["meta"].predict_proba(stack[art["meta_cols"]])[:, 1]

    # Women
    wom = matchups[(matchups.TeamID_A >= 3000) & (matchups.TeamID_B >= 3000)].copy()
    if len(wom):
        team = _build_team_table_for_season(
            data_root=data_root,
            prefix="W",
            season=args.season,
            cfg=EloConfig(k=18.0, home_adv=55.0, margin_mult=0.0, reg_to_mean=0.12),
        )
        a = team.add_prefix("A_").rename(columns={"A_Season": "Season", "A_TeamID": "TeamID_A"})
        b = team.add_prefix("B_").rename(columns={"B_Season": "Season", "B_TeamID": "TeamID_B"})
        wom = wom.merge(a, on=["Season", "TeamID_A"], how="left").merge(b, on=["Season", "TeamID_B"], how="left")

        wom["EloDiff"] = wom["A_Elo"] - wom["B_Elo"]
        wom["SeedDiff"] = wom["B_SeedNum"] - wom["A_SeedNum"]
        wom["PlayInDiff"] = wom["A_SeedPlayIn"] - wom["B_SeedPlayIn"]
        wom["NetRtgDiff"] = wom["A_NetRtg"] - wom["B_NetRtg"]
        wom["OffRtgDiff"] = wom["A_OffRtg"] - wom["B_OffRtg"]
        wom["DefRtgDiff"] = wom["A_DefRtg"] - wom["B_DefRtg"]
        wom["TempoDiff"] = wom["A_Tempo"] - wom["B_Tempo"]
        wom["eFGDiff"] = wom["A_eFG"] - wom["B_eFG"]
        wom["TOVPctDiff"] = wom["A_TOVPct"] - wom["B_TOVPct"]
        wom["ORBPctDiff"] = wom["A_ORBPct"] - wom["B_ORBPct"]
        wom["FTRDiff"] = wom["A_FTR"] - wom["B_FTR"]
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
                if a in wom.columns and b in wom.columns:
                    wom[f"{pfx}{stat}Diff"] = wom[a] - wom[b]

        art = artifacts["W"]
        feat_names = art["feature_names"]
        X_full = wom[feat_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        p_elo = art["elo_model"].predict_proba(X_full[["EloDiff", "SeedDiff", "PlayInDiff"]])[:, 1]
        p_eff = art["eff_model"].predict_proba(X_full[art["eff_cols"]])[:, 1]
        p_lgbm = predict_lgbm_ensemble(art["lgbm"], X_full)

        stack = pd.DataFrame({"p_elo": p_elo, "p_eff": p_eff, "p_lgbm": p_lgbm})
        preds[wom.index.values] = art["meta"].predict_proba(stack[art["meta_cols"]])[:, 1]

    # Clamp to avoid exact 0/1
    preds = np.clip(preds, 0.001, 0.999)

    out = pd.DataFrame({"ID": sample["ID"].values, "Pred": preds})
    out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
