from __future__ import annotations

import pandas as pd


def compute_massey_features(mm: pd.DataFrame, season: int, max_ranking_day: int = 133) -> pd.DataFrame:
    # Use last available ordinal per (SystemName, TeamID) with RankingDayNum <= max_ranking_day.
    # Convert ordinal to a strength score where higher is better.
    df = mm[mm["Season"] == season].copy()
    df = df[df["RankingDayNum"] <= max_ranking_day].copy()
    if df.empty:
        return pd.DataFrame(columns=["Season", "TeamID", "MasseyMean", "MasseyStd", "MasseyBest", "MasseyN"]) 

    df = df.sort_values(["SystemName", "TeamID", "RankingDayNum"]).groupby(["SystemName", "TeamID"], as_index=False).tail(1)

    # Some systems rank fewer teams; keep as-is, aggregate across systems
    agg = (
        df.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            MasseyMean=("OrdinalRank", "mean"),
            MasseyStd=("OrdinalRank", "std"),
            MasseyBest=("OrdinalRank", "min"),
            MasseyN=("OrdinalRank", "size"),
        )
    )
    agg["MasseyStd"] = agg["MasseyStd"].fillna(0.0)

    # Strength transform: lower ordinal means stronger
    agg["MasseyMean"] = -agg["MasseyMean"]
    agg["MasseyBest"] = -agg["MasseyBest"]
    return agg
