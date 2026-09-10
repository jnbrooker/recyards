"""
nflsim/data.py — shared NFL data layer for the whole model suite.

Every per-stat page and the game simulator import from here so they all speak
the same data. Responsibilities:

  * pull & cache nflverse weekly player stats (offense: passing/rushing/receiving)
  * pull & cache depth charts and injury reports (for the game simulator)
  * season weighting (recent seasons weighted more heavily)
  * small shared stats helpers (weighted mean/std, Beta matching, shrinkage)

Design note: this generalises the loader from the original receiving model so
it keeps *all* offensive columns (not just receiving), because rushing, TDs,
sacks and INTs all read from the same weekly release.
"""

from __future__ import annotations

import functools
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OFFENSE_POSITIONS = ["QB", "RB", "WR", "TE", "FB"]

# How hard to lean on a single defense's (noisy) season splits.
DEFAULT_DEF_SHRINK = 0.6


def season_weight(season: int, latest: int) -> float:
    """More recent seasons carry more weight when building priors."""
    gap = latest - season
    return {0: 1.0, 1: 0.7, 2: 0.45}.get(gap, 0.3)


# ---------------------------------------------------------------------------
# Weekly player stats (offense)
# ---------------------------------------------------------------------------

# Current nflverse weekly player-stats release, read directly (older installed
# nfl_data_py versions point at a retired path that 404s on 2025+).
_WEEKLY_URLS = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{year}.parquet",
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "player_stats/player_stats_{year}.parquet",
)

# Columns we keep from the (150-wide) weekly release.
_KEEP = [
    "player_id", "player_display_name", "position", "recent_team", "team",
    "opponent_team", "season", "week", "season_type",
    # receiving
    "targets", "receptions", "receiving_yards", "receiving_air_yards",
    "receiving_yards_after_catch", "receiving_tds", "target_share",
    # rushing
    "carries", "rushing_yards", "rushing_tds", "rushing_first_downs",
    "rushing_fumbles_lost",
    # passing
    "attempts", "completions", "passing_yards", "passing_tds",
    "interceptions", "sacks", "sack_yards", "passing_air_yards",
    "dropbacks",
]

_NUMERIC = [
    "targets", "receptions", "receiving_yards", "receiving_air_yards",
    "receiving_yards_after_catch", "receiving_tds",
    "carries", "rushing_yards", "rushing_tds", "rushing_first_downs",
    "rushing_fumbles_lost",
    "attempts", "completions", "passing_yards", "passing_tds",
    "interceptions", "sacks", "sack_yards", "passing_air_yards", "dropbacks",
]


def _load_one_season(year: int) -> pd.DataFrame | None:
    for tmpl in _WEEKLY_URLS:
        try:
            return pd.read_parquet(tmpl.format(year=year), engine="auto")
        except Exception:
            continue
    return None


@functools.lru_cache(maxsize=8)
def load_weekly(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Regular-season weekly offensive lines for the given seasons (cached).

    `seasons` is a tuple so it is hashable for lru_cache. Seasons not yet
    available are skipped; if none load, a clear error is raised.
    """
    frames, loaded, missing = [], [], []
    for yr in seasons:
        d = _load_one_season(int(yr))
        if d is None or len(d) == 0:
            missing.append(int(yr))
            continue
        if "recent_team" not in d.columns and "team" in d.columns:
            d = d.rename(columns={"team": "recent_team"})
        frames.append(d)
        loaded.append(int(yr))

    if not frames:
        raise ValueError(
            f"No NFL data available for seasons {list(seasons)}. "
            "That season may not have been played/published yet.")

    df = pd.concat(frames, ignore_index=True)
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"].copy()
    df = df[df["position"].isin(OFFENSE_POSITIONS)].copy()

    keep = [c for c in _KEEP if c in df.columns]
    df = df[keep].copy()
    for c in _NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    latest = int(df["season"].max())
    df["season_w"] = df["season"].map(lambda s: season_weight(int(s), latest))
    df.attrs["loaded_seasons"] = sorted(loaded)
    df.attrs["missing_seasons"] = sorted(missing)
    return df


def list_defenses(wk: pd.DataFrame) -> list[str]:
    return sorted(wk["opponent_team"].dropna().unique().tolist())


def list_players(wk: pd.DataFrame, stat: str = "targets",
                 min_vol: int = 20) -> pd.DataFrame:
    """Players with enough volume in `stat` (e.g. 'targets' or 'carries'),
    most-used first, with a display label."""
    vol = stat if stat in wk.columns else "targets"
    g = (wk.groupby(["player_id", "player_display_name", "position"],
                    as_index=False)
           .agg(vol=(vol, "sum"),
                team=("recent_team", "last"),
                games=("week", "count")))
    g = g[g["vol"] >= min_vol].sort_values("vol", ascending=False)
    g["label"] = (g["player_display_name"] + " (" + g["position"] + ", "
                  + g["team"] + ")")
    return g.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Depth charts & injuries (for the game simulator)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def load_depth_charts(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Weekly depth charts via nfl_data_py. `depth_team` 1/2/3 = starter/backup."""
    import nfl_data_py as nfl
    dc = nfl.import_depth_charts(list(seasons))
    if "club_code" in dc.columns and "team" not in dc.columns:
        dc = dc.rename(columns={"club_code": "team"})
    dc["depth_team"] = pd.to_numeric(dc["depth_team"], errors="coerce")
    return dc


@functools.lru_cache(maxsize=8)
def load_injuries(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Weekly injury reports. `report_status` in {Out, Doubtful, Questionable}."""
    import nfl_data_py as nfl
    return nfl.import_injuries(list(seasons))


@functools.lru_cache(maxsize=4)
def _pfr_id_crosswalk() -> pd.DataFrame:
    """gsis_id <-> pfr_id map, so PFR advanced stats can be joined to players."""
    import nfl_data_py as nfl
    ids = nfl.import_ids()
    return (ids[["gsis_id", "pfr_id"]].dropna(subset=["gsis_id", "pfr_id"])
              .drop_duplicates("gsis_id"))


@functools.lru_cache(maxsize=8)
def load_pfr_rush(seasons: tuple[int, ...]) -> pd.DataFrame:
    """PFR advanced weekly rushing: yards before/after contact and broken tackles,
    per game, keyed to gsis_id. Returns per-game rows with derived per-attempt
    rates so callers can compute means AND game-to-game variance."""
    import nfl_data_py as nfl
    d = nfl.import_weekly_pfr("rush", list(seasons))
    d = d[d["carries"] > 0].copy()
    d["ybc_att"] = d["rushing_yards_before_contact"] / d["carries"]
    d["yac_att"] = d["rushing_yards_after_contact"] / d["carries"]
    d["brk_rate"] = d["rushing_broken_tackles"] / d["carries"]
    xw = _pfr_id_crosswalk().rename(columns={"pfr_id": "pfr_player_id"})
    d = d.merge(xw, on="pfr_player_id", how="left")
    return d


# ---------------------------------------------------------------------------
# Shared stats helpers
# ---------------------------------------------------------------------------

def wmean(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[m], weights=w[m])) if m.any() else np.nan


def wstd(x, w, fallback):
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if m.sum() < 2:
        return fallback
    mu = np.average(x[m], weights=w[m])
    var = np.average((x[m] - mu) ** 2, weights=w[m]) * (m.sum() / (m.sum() - 1))
    sd = float(np.sqrt(var))
    return sd if sd > 1e-6 else fallback


def beta_params(mean, sd):
    """Beta(a,b) matched to a mean and std, guarded to stay valid."""
    mean = float(np.clip(mean, 1e-3, 1 - 1e-3))
    max_sd = np.sqrt(mean * (1 - mean)) * 0.99
    sd = float(np.clip(sd, 1e-4, max_sd))
    k = mean * (1 - mean) / (sd ** 2) - 1
    return max(mean * k, 0.1), max((1 - mean) * k, 0.1)


def shrink(ratio, strength):
    """Pull a defense ratio toward 1 (no effect) by (1 - strength)."""
    return 1.0 + strength * (ratio - 1.0)


def american(p: float) -> str:
    p = min(max(p, 1e-4), 1 - 1e-4)
    dec = 1.0 / p
    return f"+{round((dec - 1) * 100)}" if dec >= 2.0 else f"-{round(100 / (dec - 1))}"
