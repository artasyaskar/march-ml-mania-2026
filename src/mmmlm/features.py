from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class EloConfig:
    k: float = 20.0
    home_adv: float = 60.0  # Elo points
    margin_mult: float = 0.0  # 0 disables MOV scaling
    reg_to_mean: float = 0.10  # seasonal regression to mean
    init_elo: float = 1500.0


def compute_elo_table(
    games: pd.DataFrame,
    season_col: str = "Season",
    day_col: str = "DayNum",
    w_col: str = "WTeamID",
    l_col: str = "LTeamID",
    wloc_col: str = "WLoc",
    wscore_col: str = "WScore",
    lscore_col: str = "LScore",
    cfg: EloConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or EloConfig()

    g = games[[season_col, day_col, w_col, l_col, wloc_col, wscore_col, lscore_col]].copy()
    g = g.sort_values([season_col, day_col]).reset_index(drop=True)

    seasons = g[season_col].unique()
    rows: list[dict] = []

    prev_season_elos: dict[int, float] = {}

    for season in seasons:
        sg = g[g[season_col] == season]

        # init elos for teams in this season
        teams = pd.unique(pd.concat([sg[w_col], sg[l_col]], axis=0))
        elos: dict[int, float] = {}
        for t in teams:
            base = prev_season_elos.get(int(t), cfg.init_elo)
            elos[int(t)] = cfg.init_elo + (base - cfg.init_elo) * (1.0 - cfg.reg_to_mean)

        for _, r in sg.iterrows():
            wt = int(r[w_col])
            lt = int(r[l_col])
            wloc = r[wloc_col]

            elo_w = elos.get(wt, cfg.init_elo)
            elo_l = elos.get(lt, cfg.init_elo)

            # Convert WLoc into advantage for the winning team
            adv_w = 0.0
            if wloc == "H":
                adv_w = cfg.home_adv
            elif wloc == "A":
                adv_w = -cfg.home_adv

            p_w = 1.0 / (1.0 + 10.0 ** (-(elo_w + adv_w - elo_l) / 400.0))

            mult = 1.0
            if cfg.margin_mult and wscore_col in r and lscore_col in r:
                mov = float(r[wscore_col]) - float(r[lscore_col])
                mult = np.log1p(max(0.0, mov))
                mult = 1.0 + cfg.margin_mult * mult

            k = cfg.k * mult
            elos[wt] = elo_w + k * (1.0 - p_w)
            elos[lt] = elo_l + k * (0.0 - (1.0 - p_w))

        # snapshot end-of-regular-season elos
        for t, e in elos.items():
            rows.append({"Season": season, "TeamID": t, "Elo": float(e)})

        prev_season_elos = elos

    out = pd.DataFrame(rows)
    return out


def make_seed_features(seeds: pd.DataFrame) -> pd.DataFrame:
    s = seeds[["Season", "Seed", "TeamID"]].copy()
    s["SeedNum"] = s["Seed"].str[1:3].astype(int)
    s["SeedRegion"] = s["Seed"].str[0]
    s["SeedPlayIn"] = s["Seed"].str.len().gt(3).astype(int)
    return s[["Season", "TeamID", "SeedNum", "SeedPlayIn"]]


def season_team_skeleton(teams: pd.DataFrame, seasons: np.ndarray | list[int]) -> pd.DataFrame:
    # Cartesian skeleton to guarantee every TeamID has a row for every season.
    # Used to avoid missing-feature matchups collapsing to neutral predictions.
    s = pd.DataFrame({"Season": list(seasons)})
    t = teams[["TeamID"]].drop_duplicates().copy()
    s["_k"] = 1
    t["_k"] = 1
    out = s.merge(t, on="_k", how="outer").drop(columns=["_k"])
    return out


def build_matchup_frame(sample_submission: pd.DataFrame) -> pd.DataFrame:
    ids = sample_submission["ID"].str.split("_", expand=True)
    df = pd.DataFrame({
        "Season": ids[0].astype(int),
        "TeamID_A": ids[1].astype(int),
        "TeamID_B": ids[2].astype(int),
        "ID": sample_submission["ID"].values,
    })
    return df
