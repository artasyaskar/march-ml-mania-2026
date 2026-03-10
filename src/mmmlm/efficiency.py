from __future__ import annotations

import numpy as np
import pandas as pd


def _prep_team_games_from_detailed(d: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "Season",
        "DayNum",
        "WTeamID",
        "LTeamID",
        "WScore",
        "LScore",
        "WLoc",
        "NumOT",
    ]

    wcols = [
        "WFGM",
        "WFGA",
        "WFGM3",
        "WFGA3",
        "WFTM",
        "WFTA",
        "WOR",
        "WDR",
        "WAst",
        "WTO",
        "WStl",
        "WBlk",
        "WPF",
    ]
    lcols = [c.replace("W", "L", 1) for c in wcols]

    use_cols = [c for c in base_cols + wcols + lcols if c in d.columns]
    df = d[use_cols].copy()

    w = df[["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"] + wcols + lcols].copy()
    w = w.rename(columns={"WTeamID": "TeamID", "LTeamID": "OppTeamID", "WScore": "Score", "LScore": "OppScore"})
    w["IsWin"] = 1

    # Swap L* columns into team perspective
    rename_map = {c: c[1:] for c in wcols}  # WFGM->FGM etc
    w = w.rename(columns=rename_map)
    w = w.rename(columns={c.replace("L", "Opp", 1)[1:]: c.replace("L", "Opp", 1) for c in lcols})

    l = df[["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"] + wcols + lcols].copy()
    l = l.rename(columns={"LTeamID": "TeamID", "WTeamID": "OppTeamID", "LScore": "Score", "WScore": "OppScore"})
    l["IsWin"] = 0

    # For losing team, their stats are in L*, opponent in W*
    rename_map_l = {c: c[1:] for c in lcols}  # LFGM->FGM
    l = l.rename(columns=rename_map_l)
    l = l.rename(columns={c.replace("W", "Opp", 1)[1:]: c.replace("W", "Opp", 1) for c in wcols})

    out = pd.concat([w, l], axis=0, ignore_index=True)

    # Ensure required columns exist
    for c in ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]:
        if c not in out.columns:
            out[c] = np.nan
        if f"Opp{c}" not in out.columns:
            out[f"Opp{c}"] = np.nan

    return out


def _possessions(fga: pd.Series, fta: pd.Series, orb: pd.Series, tov: pd.Series) -> pd.Series:
    # Common NCAA approximation
    return fga - orb + tov + 0.475 * fta


def compute_team_season_efficiency(detailed_results: pd.DataFrame) -> pd.DataFrame:
    tg = _prep_team_games_from_detailed(detailed_results)

    tg["Poss"] = _possessions(tg["FGA"], tg["FTA"], tg["OR"], tg["TO"])
    tg["OppPoss"] = _possessions(tg["OppFGA"], tg["OppFTA"], tg["OppOR"], tg["OppTO"])
    tg["Poss"] = tg[["Poss", "OppPoss"]].mean(axis=1)

    agg = (
        tg.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            Games=("IsWin", "size"),
            Wins=("IsWin", "sum"),
            Pts=("Score", "sum"),
            OppPts=("OppScore", "sum"),
            Poss=("Poss", "sum"),
            FGA=("FGA", "sum"),
            FGM=("FGM", "sum"),
            FGA3=("FGA3", "sum"),
            FGM3=("FGM3", "sum"),
            FTA=("FTA", "sum"),
            FTM=("FTM", "sum"),
            OR=("OR", "sum"),
            DR=("DR", "sum"),
            Ast=("Ast", "sum"),
            TO=("TO", "sum"),
            Stl=("Stl", "sum"),
            Blk=("Blk", "sum"),
            PF=("PF", "sum"),
        )
    )

    eps = 1e-9
    agg["WinPct"] = agg["Wins"] / (agg["Games"] + eps)
    agg["Tempo"] = agg["Poss"] / (agg["Games"] + eps)
    agg["OffRtg"] = 100.0 * agg["Pts"] / (agg["Poss"] + eps)
    agg["DefRtg"] = 100.0 * agg["OppPts"] / (agg["Poss"] + eps)
    agg["NetRtg"] = agg["OffRtg"] - agg["DefRtg"]

    agg["eFG"] = (agg["FGM"] + 0.5 * agg["FGM3"]) / (agg["FGA"] + eps)
    agg["TOVPct"] = agg["TO"] / (agg["FGA"] + 0.475 * agg["FTA"] + agg["TO"] + eps)
    agg["ORBPct"] = agg["OR"] / (agg["OR"] + agg["DR"] + eps)
    agg["FTR"] = agg["FTA"] / (agg["FGA"] + eps)

    keep = [
        "Season",
        "TeamID",
        "Games",
        "WinPct",
        "Tempo",
        "OffRtg",
        "DefRtg",
        "NetRtg",
        "eFG",
        "TOVPct",
        "ORBPct",
        "FTR",
    ]
    return agg[keep]


def _weighted_agg(tg: pd.DataFrame, weight_col: str) -> pd.DataFrame:
    eps = 1e-9
    w = tg[weight_col].astype(float)

    def wsum(x: pd.Series) -> float:
        return float(np.sum(x.to_numpy(dtype=float) * w.loc[x.index].to_numpy(dtype=float)))

    def wcount(_: pd.Series) -> float:
        return float(np.sum(w.loc[_.index].to_numpy(dtype=float)))

    g = (
        tg.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            WGames=("IsWin", "size"),
            w_sum=("IsWin", wcount),
            Wins=("IsWin", wsum),
            Pts=("Score", wsum),
            OppPts=("OppScore", wsum),
            Poss=("Poss", wsum),
            FGA=("FGA", wsum),
            FGM=("FGM", wsum),
            FGA3=("FGA3", wsum),
            FGM3=("FGM3", wsum),
            FTA=("FTA", wsum),
            FTM=("FTM", wsum),
            OR=("OR", wsum),
            DR=("DR", wsum),
            TO=("TO", wsum),
        )
    )

    g["WinPct"] = g["Wins"] / (g["w_sum"] + eps)
    g["Tempo"] = g["Poss"] / (g["w_sum"] + eps)
    g["OffRtg"] = 100.0 * g["Pts"] / (g["Poss"] + eps)
    g["DefRtg"] = 100.0 * g["OppPts"] / (g["Poss"] + eps)
    g["NetRtg"] = g["OffRtg"] - g["DefRtg"]
    g["eFG"] = (g["FGM"] + 0.5 * g["FGM3"]) / (g["FGA"] + eps)
    g["TOVPct"] = g["TO"] / (g["FGA"] + 0.475 * g["FTA"] + g["TO"] + eps)
    g["ORBPct"] = g["OR"] / (g["OR"] + g["DR"] + eps)
    g["FTR"] = g["FTA"] / (g["FGA"] + eps)

    return g[[
        "Season",
        "TeamID",
        "WinPct",
        "Tempo",
        "OffRtg",
        "DefRtg",
        "NetRtg",
        "eFG",
        "TOVPct",
        "ORBPct",
        "FTR",
    ]]


def compute_team_season_efficiency_recency(
    detailed_results: pd.DataFrame,
    last_n_games: int = 10,
    last_day_window: int = 30,
    exp_halflife_days: float = 15.0,
) -> pd.DataFrame:
    tg = _prep_team_games_from_detailed(detailed_results)

    tg["Poss"] = _possessions(tg["FGA"], tg["FTA"], tg["OR"], tg["TO"])
    tg["OppPoss"] = _possessions(tg["OppFGA"], tg["OppFTA"], tg["OppOR"], tg["OppTO"])
    tg["Poss"] = tg[["Poss", "OppPoss"]].mean(axis=1)

    tg = tg.sort_values(["Season", "TeamID", "DayNum"]).reset_index(drop=True)
    tg["GameIdx"] = tg.groupby(["Season", "TeamID"]).cumcount()
    tg["GamesInSeason"] = tg.groupby(["Season", "TeamID"])["GameIdx"].transform("max") + 1
    tg["IsLastN"] = (tg["GameIdx"] >= (tg["GamesInSeason"] - last_n_games)).astype(int)

    max_day = tg.groupby(["Season", "TeamID"])["DayNum"].transform("max")
    tg["IsLast30d"] = (tg["DayNum"] >= (max_day - last_day_window)).astype(int)

    # exponential decay by recency (higher weight = more recent)
    decay = np.log(2.0) / max(exp_halflife_days, 1e-6)
    tg["ExpW"] = np.exp(-decay * (max_day - tg["DayNum"]))

    # weights for subsets
    tg["W_lastN"] = tg["IsLastN"].astype(float)
    tg["W_last30"] = tg["IsLast30d"].astype(float)

    lastn = _weighted_agg(tg[tg["IsLastN"] == 1].copy(), weight_col="W_lastN")
    last30 = _weighted_agg(tg[tg["IsLast30d"] == 1].copy(), weight_col="W_last30")
    expw = _weighted_agg(tg.copy(), weight_col="ExpW")

    def add_prefix(df: pd.DataFrame, p: str) -> pd.DataFrame:
        ren = {c: f"{p}{c}" for c in df.columns if c not in ["Season", "TeamID"]}
        return df.rename(columns=ren)

    out = add_prefix(lastn, "L10_")
    out = out.merge(add_prefix(last30, "D30_"), on=["Season", "TeamID"], how="outer")
    out = out.merge(add_prefix(expw, "EXP_"), on=["Season", "TeamID"], how="outer")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out
