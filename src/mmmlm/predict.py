from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mmmlm.data_paths import DataPaths
from mmmlm.efficiency import compute_team_season_efficiency, compute_team_season_efficiency_recency
from mmmlm.features import EloConfig, build_matchup_frame, compute_elo_table, make_seed_features, season_team_skeleton
from mmmlm.lgbm_model import predict_lgbm_ensemble
from mmmlm.massey import compute_massey_features


def _load_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def _build_team_table_for_season(data_root: Path, prefix: str, season: int, cfg: EloConfig) -> pd.DataFrame:
    paths = DataPaths(data_root)

    rs = _load_csv(paths.file(f"{prefix}RegularSeasonCompactResults.csv"))
    rs_det = _load_csv(paths.file(f"{prefix}RegularSeasonDetailedResults.csv"))
    seeds = _load_csv(paths.file(f"{prefix}NCAATourneySeeds.csv"))
    teams = _load_csv(paths.file(f"{prefix}Teams.csv"))[["TeamID"]].copy()

    # Use all seasons up to target to compute Elo with proper carry-over.
    rs = rs[rs.Season <= season].copy()
    rs_det_season = rs_det[rs_det.Season == season].copy()
    seeds = seeds[seeds.Season == season].copy()

    elo = compute_elo_table(rs, cfg=cfg)
    elo = elo[elo.Season == season].copy()
    eff = compute_team_season_efficiency(rs_det_season)
    rec = compute_team_season_efficiency_recency(rs_det_season, last_n_games=10, last_day_window=30, exp_halflife_days=15.0)
    seed_feat = make_seed_features(seeds)

    skel = season_team_skeleton(teams, [season])
    team = skel.merge(elo, on=["Season", "TeamID"], how="left")
    team = team.merge(eff, on=["Season", "TeamID"], how="left")
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


def _apply_prediction_priors(
    team_target: pd.DataFrame,
    team_prior: pd.DataFrame | None,
) -> pd.DataFrame:
    # Backfill missing target-season features from prior season (same TeamID), then global mean.
    t = team_target.copy()
    t = t.sort_values(["TeamID"]).reset_index(drop=True)

    if team_prior is not None and len(team_prior):
        p = team_prior.copy()
        p = p.drop(columns=[c for c in ["Season"] if c in p.columns])
        p = p.add_prefix("P_")
        p = p.rename(columns={"P_TeamID": "TeamID"})
        t = t.merge(p, on="TeamID", how="left")

        # Fill missing columns in target from prior
        for c in list(t.columns):
            if c in ["Season", "TeamID"]:
                continue
            pc = f"P_{c}"
            if pc in t.columns:
                t[c] = t[c].where(~t[c].isna(), t[pc])

        t = t.drop(columns=[c for c in t.columns if c.startswith("P_")])

    # If a team has no Elo (never played historically), use global mean.
    if "Elo" in t.columns:
        t["Elo"] = t["Elo"].fillna(t["Elo"].mean())

    # Global mean fallback + safety
    num_cols = [c for c in t.columns if c not in ["Season", "TeamID"]]
    for c in num_cols:
        if t[c].dtype.kind in "biufc":
            t[c] = t[c].fillna(t[c].mean())
        else:
            t[c] = t[c].fillna(0)
    t = t.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return t


def _blend(p_elo: np.ndarray, p_eff: np.ndarray, p_lgbm: np.ndarray) -> np.ndarray:
    return 0.20 * p_elo + 0.15 * p_eff + 0.65 * p_lgbm


def _score_with_fallback(art: dict, X_full: pd.DataFrame) -> np.ndarray:
    base_X = X_full[["EloDiff", "SeedDiff", "PlayInDiff"]].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Optional shrinkage to avoid extreme Elo-only probabilities when inputs are out-of-scale.
    # This is especially useful for women's where seed coverage/scale and Elo carry can be noisy.
    shrink = float(art.get("elo_shrink", 1.0))
    if shrink != 1.0:
        base_X = base_X.copy()
        base_X["EloDiff"] = base_X["EloDiff"] * shrink
        base_X["SeedDiff"] = base_X["SeedDiff"] * shrink

    p_elo = art["elo_model"].predict_proba(base_X)[:, 1]
    p_elo = np.clip(p_elo, 0.001, 0.999)

    # Hard safety: if requested (e.g., women's model is degenerate), force Elo-only.
    if art.get("force_elo_only", False):
        # Mix with neutral prior to avoid huge clamp-floor blocks when Elo is unreliable/out-of-scale.
        prior_w = float(art.get("prior_weight", 0.25))
        p = (1.0 - prior_w) * p_elo + prior_w * 0.5
        lo = float(art.get("clamp_lo", 0.02))
        hi = float(art.get("clamp_hi", 0.98))
        return np.clip(p, lo, hi)

    # Coverage gate: only use higher-capacity models when the row looks in-distribution.
    # Using only an "eff non-zero" test is too weak for women where many features can be
    # structurally zero/constant after fills.
    eff_cols = art.get("eff_cols", [])
    req_cols = list(dict.fromkeys(["EloDiff", "SeedDiff", "PlayInDiff"] + eff_cols))
    req_cols = [c for c in req_cols if c in X_full.columns]

    req = X_full[req_cols].replace([np.inf, -np.inf], np.nan)
    has_req = (~req.isna()).all(axis=1).to_numpy()
    # Require at least one meaningful efficiency signal (non-trivial magnitude and global variance).
    if len(eff_cols):
        eff_X = X_full[eff_cols].replace([np.inf, -np.inf], np.nan)
        eff_mag = eff_X.abs().sum(axis=1)
        # Raise magnitude threshold to avoid treating tiny fills as signal
        mag_ok = (eff_mag > 1e-3).to_numpy()
        # Global variance filter: if all efficiency columns have near-zero variance, treat as no signal
        global_var_ok = (eff_X.var(axis=0) > 1e-6).any()
        has_sig = mag_ok if global_var_ok else np.zeros(len(X_full), dtype=bool)
    else:
        eff_X = pd.DataFrame(index=X_full.index)
        has_sig = np.zeros(len(X_full), dtype=bool)

    use_full = has_req & has_sig

    p = p_elo.copy()
    if use_full.any():
        p_eff = art["eff_model"].predict_proba(eff_X.loc[use_full].replace([np.inf, -np.inf], np.nan).fillna(0.0))[:, 1]
        p_lgbm = predict_lgbm_ensemble(art["lgbm"], X_full.loc[use_full].replace([np.inf, -np.inf], np.nan).fillna(0.0))
        stack = pd.DataFrame({"p_elo": p_elo[use_full], "p_eff": p_eff, "p_lgbm": p_lgbm})
        p[use_full] = art["meta"].predict_proba(stack[art["meta_cols"]])[:, 1]

    return np.clip(p, 0.001, 0.999)


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

    preds = np.zeros(len(matchups), dtype=float)

    # Men
    men = matchups[(matchups.TeamID_A < 2000) & (matchups.TeamID_B < 2000)].copy()
    if len(men):
        team_2026 = _build_team_table_for_season(
            data_root=data_root,
            prefix="M",
            season=args.season,
            cfg=EloConfig(k=18.0, home_adv=55.0, margin_mult=0.0, reg_to_mean=0.12),
        )
        team_2025 = _build_team_table_for_season(
            data_root=data_root,
            prefix="M",
            season=args.season - 1,
            cfg=EloConfig(k=18.0, home_adv=55.0, margin_mult=0.0, reg_to_mean=0.12),
        )
        team = _apply_prediction_priors(team_2026, team_2025)
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

        preds[men.index.values] = _score_with_fallback(art, X_full)

    # Women
    wom = matchups[(matchups.TeamID_A >= 3000) & (matchups.TeamID_B >= 3000)].copy()
    if len(wom):
        team_2026 = _build_team_table_for_season(
            data_root=data_root,
            prefix="W",
            season=args.season,
            cfg=EloConfig(k=18.0, home_adv=55.0, margin_mult=0.0, reg_to_mean=0.12),
        )
        team_2025 = _build_team_table_for_season(
            data_root=data_root,
            prefix="W",
            season=args.season - 1,
            cfg=EloConfig(k=18.0, home_adv=55.0, margin_mult=0.0, reg_to_mean=0.12),
        )
        team = _apply_prediction_priors(team_2026, team_2025)
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

        art = dict(artifacts["W"])
        # Women's predictions are currently collapsing to the clamp floor for many matchups.
        # Force the robust Elo+seed model until women's feature distributions are fixed.
        art["force_elo_only"] = True
        # Additionally shrink Elo/seed diffs toward 0 to avoid near-deterministic blocks.
        art["elo_shrink"] = 0.35
        # And blend with a neutral prior + softer clamp to eliminate 0.001/0.999 blocks.
        art["prior_weight"] = 0.30
        art["clamp_lo"] = 0.02
        art["clamp_hi"] = 0.98
        feat_names = art["feature_names"]
        X_full = wom[feat_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        p_w = _score_with_fallback(art, X_full)
        # Hard safety: ensure no women rows are exactly at the global clamp limits (0.001/0.999)
        # Blend with neutral prior only for rows that are still at the clamp after scoring.
        lo_thr = 0.001 + 1e-12
        hi_thr = 0.999 - 1e-12
        stuck_lo = (p_w <= lo_thr)
        stuck_hi = (p_w >= hi_thr)
        n_stuck_lo = int(stuck_lo.sum())
        n_stuck_hi = int(stuck_hi.sum())
        if n_stuck_lo or n_stuck_hi:
            # Blend with 0.5 to lift off the clamp floor/ceiling
            p_w = np.where(stuck_lo, 0.7 * p_w + 0.3 * 0.5, p_w)
            p_w = np.where(stuck_hi, 0.7 * p_w + 0.3 * 0.5, p_w)
            p_w = np.clip(p_w, 0.001, 0.999)
        # Debug: print coverage and extreme probability counts for women
        try:
            eff_cols = art.get("eff_cols", [])
            if len(eff_cols):
                eff_mag = X_full[eff_cols].abs().sum(axis=1).to_numpy()
                n_sig = int((eff_mag > 1e-3).sum())
            else:
                n_sig = 0
            n_lo = int((p_w <= lo_thr).sum())
            n_hi = int((p_w >= hi_thr).sum())
            forced = bool(art.get("force_elo_only", False))
            print(f"[predict] Women rows={len(wom)} eff_signal_rows={n_sig} clamp_lo={n_lo} clamp_hi={n_hi} forced_elo_only={forced} stuck_blended_lo={n_stuck_lo} stuck_blended_hi={n_stuck_hi}")
        except Exception:
            pass

        preds[wom.index.values] = p_w

    # Any remaining rows (shouldn't happen) fallback to men's model based on TeamID space
    # This prevents leaving blocks at an unscored default.
    rem = (preds == 0.0)
    if rem.any():
        # Use Elo-only probability as a safe non-0.5 fallback
        # (If these rows exist, features were missing due to unexpected TeamID ranges.)
        sub = matchups.loc[rem].copy()
        # Determine which artifact to use: if TeamIDs look like women's use W else M
        use_w = (sub["TeamID_A"] >= 3000) & (sub["TeamID_B"] >= 3000)
        if use_w.any():
            art = artifacts["W"]
            team = _apply_prediction_priors(
                _build_team_table_for_season(data_root, "W", args.season, EloConfig(k=18.0, home_adv=55.0, margin_mult=0.0, reg_to_mean=0.12)),
                _build_team_table_for_season(data_root, "W", args.season - 1, EloConfig(k=18.0, home_adv=55.0, margin_mult=0.0, reg_to_mean=0.12)),
            )
            a = team.add_prefix("A_").rename(columns={"A_Season": "Season", "A_TeamID": "TeamID_A"})
            b = team.add_prefix("B_").rename(columns={"B_Season": "Season", "B_TeamID": "TeamID_B"})
            subw = sub[use_w].merge(a, on=["Season", "TeamID_A"], how="left").merge(b, on=["Season", "TeamID_B"], how="left")
            subw["EloDiff"] = subw["A_Elo"] - subw["B_Elo"]
            subw["SeedDiff"] = subw["B_SeedNum"] - subw["A_SeedNum"]
            subw["PlayInDiff"] = subw["A_SeedPlayIn"] - subw["B_SeedPlayIn"]
            pe = art["elo_model"].predict_proba(subw[["EloDiff", "SeedDiff", "PlayInDiff"]].replace([np.inf, -np.inf], np.nan).fillna(0.0))[:, 1]
            preds[subw.index.values] = pe
        use_m = ~use_w
        if use_m.any():
            art = artifacts["M"]
            team = _apply_prediction_priors(
                _build_team_table_for_season(data_root, "M", args.season, EloConfig(k=18.0, home_adv=55.0, margin_mult=0.0, reg_to_mean=0.12)),
                _build_team_table_for_season(data_root, "M", args.season - 1, EloConfig(k=18.0, home_adv=55.0, margin_mult=0.0, reg_to_mean=0.12)),
            )
            a = team.add_prefix("A_").rename(columns={"A_Season": "Season", "A_TeamID": "TeamID_A"})
            b = team.add_prefix("B_").rename(columns={"B_Season": "Season", "B_TeamID": "TeamID_B"})
            subm = sub[use_m].merge(a, on=["Season", "TeamID_A"], how="left").merge(b, on=["Season", "TeamID_B"], how="left")
            subm["EloDiff"] = subm["A_Elo"] - subm["B_Elo"]
            subm["SeedDiff"] = subm["B_SeedNum"] - subm["A_SeedNum"]
            subm["PlayInDiff"] = subm["A_SeedPlayIn"] - subm["B_SeedPlayIn"]
            pe = art["elo_model"].predict_proba(subm[["EloDiff", "SeedDiff", "PlayInDiff"]].replace([np.inf, -np.inf], np.nan).fillna(0.0))[:, 1]
            preds[subm.index.values] = pe

    # Final clamp to avoid exact 0/1; for women we already clamped to [0.02,0.98] in the block above
    # Therefore, apply a softer final clamp for women only to avoid re-clamping to 0.001.
    is_women = (matchups.TeamID_A >= 3000) & (matchups.TeamID_B >= 3000)
    preds = np.where(is_women, np.clip(preds, 0.02, 0.98), np.clip(preds, 0.001, 0.999))

    out = pd.DataFrame({"ID": sample["ID"].values, "Pred": preds})
    out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
