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


# The current weekly release renamed several passing columns; the retired
# `player_stats` release (still the second URL above) uses the older names. We
# normalise both to one canonical schema so every model reads the same columns.
_ALIASES = {
    "recent_team": ("team",),
    "sacks": ("sacks_suffered",),
    "sack_yards": ("sack_yards_lost",),
    "interceptions": ("passing_interceptions",),
}


def _normalize_columns(d: pd.DataFrame) -> pd.DataFrame:
    """Copy any aliased column onto its canonical name (originals are kept)."""
    for canon, alts in _ALIASES.items():
        if canon in d.columns:
            continue
        for alt in alts:
            if alt in d.columns:
                d[canon] = d[alt]
                break
    return d


def _load_one_season(year: int) -> pd.DataFrame | None:
    for tmpl in _WEEKLY_URLS:
        try:
            return _normalize_columns(pd.read_parquet(tmpl.format(year=year),
                                                      engine="auto"))
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

    # Dropbacks are not published in the weekly feed. Attempts + sacks taken is
    # the conventional sack-rate denominator (it excludes scrambles), so derive
    # it once here and let every passing model read one consistent column.
    if ("dropbacks" not in df.columns or df["dropbacks"].sum() <= 0)             and {"attempts", "sacks"} <= set(df.columns):
        df["dropbacks"] = df["attempts"] + df["sacks"]

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

# nflverse release URLs, read directly (no nfl_data_py dependency — matches the
# original receiving model's approach, so the app runs with just pandas/pyarrow).
_DEPTH_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
              "depth_charts/depth_charts_{year}.parquet")
_INJURY_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
               "injuries/injuries_{year}.parquet")
_PFR_RUSH_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
                 "pfr_advstats/advstats_week_rush_{year}.parquet")
_IDS_URL = ("https://raw.githubusercontent.com/dynastyprocess/data/master/"
            "files/db_playerids.csv")


def _read_seasons(url_tmpl, seasons):
    """Read one nflverse parquet per season, skipping any that aren't published."""
    frames = []
    for yr in seasons:
        try:
            frames.append(pd.read_parquet(url_tmpl.format(year=int(yr))))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# Skill positions the game engine allocates volume to. Linemen are on the depth
# chart too, but they never touch the ball, so they are dropped at load time
# (the raw feed is ~550k rows a season).
SKILL_POSITIONS = ["QB", "RB", "FB", "WR", "TE"]

# The offensive personnel group in the current depth-chart feed. Older releases
# used `pos_grp == "OFF"`; both are accepted.
_OFFENSE_GROUPS = ("3WR 1TE", "OFF", "Offense")


@functools.lru_cache(maxsize=8)
def load_depth_charts(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Offensive skill-position depth charts, normalised across feed versions.

    The current release is a stream of dated snapshots (`dt`) with `pos_abb` and
    `pos_rank`; older ones were weekly with `position` / `depth_team`. Both are
    mapped onto: team, player_id, player_name, position, depth (1 = starter),
    and `dt` as a timestamp. Empty frame if the feed is unavailable.
    """
    dc = _read_seasons(_DEPTH_URL, seasons)
    if dc.empty:
        return dc

    if "club_code" in dc.columns and "team" not in dc.columns:
        dc = dc.rename(columns={"club_code": "team"})
    if "pos_grp" in dc.columns:
        grp = dc["pos_grp"].astype(str)
        if grp.isin(_OFFENSE_GROUPS).any():
            dc = dc[grp.isin(_OFFENSE_GROUPS)]

    # position / depth / id, whichever generation of the feed this is
    pos = "pos_abb" if "pos_abb" in dc.columns else "position"
    depth = "pos_rank" if "pos_rank" in dc.columns else "depth_team"
    pid = ("gsis_id" if "gsis_id" in dc.columns else
           "player_id" if "player_id" in dc.columns else None)
    if pos not in dc.columns or depth not in dc.columns or pid is None:
        return pd.DataFrame()

    out = pd.DataFrame({
        "team": dc["team"],
        "player_id": dc[pid],
        "player_name": dc.get("player_name", dc.get("full_name")),
        "position": dc[pos].astype(str),
        "depth": pd.to_numeric(dc[depth], errors="coerce"),
    })
    if "dt" in dc.columns:
        out["dt"] = pd.to_datetime(dc["dt"], errors="coerce", utc=True)
    elif {"season", "week"} <= set(dc.columns):
        # older weekly feed: synthesise an orderable stamp
        out["dt"] = pd.to_datetime(dc["season"].astype(str), errors="coerce", utc=True)             + pd.to_timedelta(pd.to_numeric(dc["week"], errors="coerce") * 7, unit="D")
    else:
        out["dt"] = pd.NaT

    out = out[out["position"].isin(SKILL_POSITIONS)]
    return out.dropna(subset=["player_id", "depth"]).reset_index(drop=True)


def depth_chart_snapshot(dc: pd.DataFrame, team: str,
                         as_of=None) -> pd.DataFrame:
    """One team's most recent depth chart at or before `as_of` (latest if None).

    Returns one row per player, best depth first. A player listed at more than
    one spot keeps his best (lowest) depth.
    """
    if dc is None or dc.empty:
        return pd.DataFrame(columns=["team", "player_id", "player_name",
                                     "position", "depth", "dt"])
    t = dc[dc["team"] == team]
    if as_of is not None and "dt" in t.columns:
        stamp = pd.to_datetime(as_of, utc=True)
        t = t[t["dt"] <= stamp]
    if t.empty:
        return t
    latest = t["dt"].max()
    snap = t[t["dt"] == latest] if pd.notna(latest) else t
    snap = (snap.sort_values("depth")
                .drop_duplicates("player_id", keep="first")
                .sort_values(["position", "depth"]))
    return snap.reset_index(drop=True)


def players_ruled_out(inj: pd.DataFrame, team: str, season: int | None = None,
                      week: int | None = None,
                      statuses: tuple[str, ...] = ("Out", "Doubtful")) -> set:
    """gsis_ids a team has ruled out on its latest (or a given) injury report."""
    if inj is None or inj.empty or "report_status" not in inj.columns:
        return set()
    t = inj[inj["team"] == team]
    # Resolve season FIRST, then the week within it: taking the max week across
    # several seasons mixes a stale week 18 into a current week 1 and silently
    # benches healthy players.
    if "season" in t.columns and not t.empty:
        t = t[t["season"] == (int(season) if season is not None else t["season"].max())]
    if "week" in t.columns and not t.empty:
        t = t[t["week"] == (int(week) if week is not None else t["week"].max())]
    t = t[t["report_status"].isin(statuses)]
    idcol = "gsis_id" if "gsis_id" in t.columns else "player_id"
    return set(t[idcol].dropna().astype(str))


@functools.lru_cache(maxsize=8)
def load_injuries(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Weekly injury reports (direct parquet). report_status in {Out, Doubtful, Questionable}."""
    return _read_seasons(_INJURY_URL, seasons)


@functools.lru_cache(maxsize=4)
def _pfr_id_crosswalk() -> pd.DataFrame:
    """gsis_id <-> pfr_id map (dynastyprocess), so PFR stats can join to players."""
    try:
        ids = pd.read_csv(_IDS_URL, low_memory=False)
        return (ids[["gsis_id", "pfr_id"]].dropna(subset=["gsis_id", "pfr_id"])
                  .drop_duplicates("gsis_id"))
    except Exception:
        return pd.DataFrame(columns=["gsis_id", "pfr_id"])


@functools.lru_cache(maxsize=8)
def load_pfr_rush(seasons: tuple[int, ...]) -> pd.DataFrame:
    """PFR advanced weekly rushing (direct parquet): yards before/after contact and
    broken tackles per game, keyed to gsis_id. Empty frame if unavailable — callers
    fall back to a yards-per-carry split."""
    d = _read_seasons(_PFR_RUSH_URL, seasons)
    if d.empty:
        return d
    d = d[d["carries"] > 0].copy()
    d["ybc_att"] = d["rushing_yards_before_contact"] / d["carries"]
    d["yac_att"] = d["rushing_yards_after_contact"] / d["carries"]
    d["brk_rate"] = d["rushing_broken_tackles"] / d["carries"]
    xw = _pfr_id_crosswalk().rename(columns={"pfr_id": "pfr_player_id"})
    if not xw.empty:
        d = d.merge(xw, on="pfr_player_id", how="left")
    else:
        d["gsis_id"] = np.nan
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


# ---------------------------------------------------------------------------
# Play-by-play (Phase 2 spike): richer run-defense factors
# ---------------------------------------------------------------------------
# The PFR advanced-rush feed only carries YBC / YAC / broken tackles, which the
# three existing defense factors already use up. Genuinely new run-defense
# dimensions — how often a front STUFFS a run, how often it surrenders an
# EXPLOSIVE run, and its overall down-adjusted EFFICIENCY (success rate / EPA) —
# live in the play-by-play feed. This block loads pbp and turns it into those
# per-defense factors, expressed as ratios vs league so the simulator can shrink
# them like the others.

_PBP_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
            "pbp/play_by_play_{year}.parquet")

# Columns we need from the (370-wide) pbp release; only these are read. One
# cached read serves both consumers: the run-defense factors and the drive
# table the game engine is built on.
_PBP_KEEP = [
    "season", "week", "season_type", "play_type", "game_id",
    "posteam", "defteam", "home_team", "away_team",
    "yards_gained", "rushing_yards", "rush_attempt",
    "epa", "success", "qb_scramble", "qb_kneel", "qb_spike",
    "two_point_attempt", "down", "ydstogo",
    # drive level (game engine)
    "fixed_drive", "fixed_drive_result", "drive_inside20", "yardline_100",
    "touchdown", "field_goal_attempt", "home_score", "away_score",
]

# Thresholds defining a stuffed / explosive run.
STUFF_MAX = 0        # yards_gained <= 0  → stuffed (TFL / no gain)
EXPL_MIN = 10        # yards_gained >= 10 → explosive
EXPL15_MIN = 15      # yards_gained >= 15 → chunk/breakaway

MIN_DEF_RUNS = 50    # designed runs a defense must have faced to get a profile


@functools.lru_cache(maxsize=4)
def load_pbp_raw(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Regular-season plays from the nflverse pbp feed, trimmed to `_PBP_KEEP`
    (cached). Shared by `load_pbp` (runs) and `load_drives` (game engine) so a
    session downloads each season's pbp once. Empty frame if unavailable."""
    frames = []
    for yr in seasons:
        try:
            d = pd.read_parquet(_PBP_URL.format(year=int(yr)))
        except Exception:
            continue
        cols = [c for c in _PBP_KEEP if c in d.columns]
        frames.append(d[cols].copy())
    if not frames:
        return pd.DataFrame()
    pbp = pd.concat(frames, ignore_index=True)
    if "season_type" in pbp.columns:
        pbp = pbp[pbp["season_type"] == "REG"]
    return pbp


def load_pbp(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Regular-season *designed-run* plays (QB scrambles, kneels and two-point
    plays removed) with the columns needed to score stuff / explosive /
    efficiency. Empty frame if pbp is unavailable — callers then skip them."""
    pbp = load_pbp_raw(seasons)
    if pbp.empty:
        return pbp
    pbp = pbp.copy()

    # designed runs only
    is_run = (pbp.get("play_type") == "run")
    if "rush_attempt" in pbp.columns:
        is_run = is_run | (pd.to_numeric(pbp["rush_attempt"], errors="coerce") == 1)
    pbp = pbp[is_run].copy()
    for flag in ("qb_scramble", "qb_kneel", "qb_spike", "two_point_attempt"):
        if flag in pbp.columns:
            pbp = pbp[pd.to_numeric(pbp[flag], errors="coerce").fillna(0) != 1]

    # gain: prefer rushing_yards, fall back to yards_gained
    g = pd.to_numeric(pbp.get("rushing_yards"), errors="coerce")
    if "yards_gained" in pbp.columns:
        g = g.fillna(pd.to_numeric(pbp["yards_gained"], errors="coerce"))
    pbp["gain"] = g
    pbp = pbp.dropna(subset=["gain", "defteam"])

    # success: use provided flag, else EPA > 0
    if "success" in pbp.columns:
        pbp["succ"] = pd.to_numeric(pbp["success"], errors="coerce")
    if "success" not in pbp.columns or pbp["succ"].isna().all():
        pbp["succ"] = (pd.to_numeric(pbp.get("epa"), errors="coerce") > 0).astype(float)
    pbp["epa"] = pd.to_numeric(pbp.get("epa"), errors="coerce")
    return pbp


def rush_defense_pbp(pbp: pd.DataFrame) -> dict:
    """Per-defense run factors from pbp, as ratios vs league.

    For each defense (keyed by `defteam`):
      r_stuff  — stuff rate allowed / league   (>1 = stuffs runs more = tougher)
      r_expl   — 10+ yd run rate allowed / league (>1 = gives up big runs = softer)
      r_eff    — success rate allowed / league  (>1 = softer, easier yards)
    plus the raw allowed rates and league baselines for display. Empty dict if
    pbp is missing.
    """
    if pbp is None or pbp.empty or "gain" not in pbp.columns:
        return {}
    g = pbp.copy()
    g["is_stuff"] = (g["gain"] <= STUFF_MAX).astype(float)
    g["is_expl"] = (g["gain"] >= EXPL_MIN).astype(float)
    g["is_expl15"] = (g["gain"] >= EXPL15_MIN).astype(float)

    d = (g.groupby("defteam")
           .agg(runs=("gain", "size"), stuff=("is_stuff", "mean"),
                expl=("is_expl", "mean"), expl15=("is_expl15", "mean"),
                succ=("succ", "mean"), epa=("epa", "mean"))
           .reset_index())
    d = d[d["runs"] >= MIN_DEF_RUNS]
    if d.empty:
        return {}

    lg_stuff = float((g["is_stuff"].sum()) / len(g))
    lg_expl = float((g["is_expl"].sum()) / len(g))
    lg_expl15 = float((g["is_expl15"].sum()) / len(g))
    lg_succ = float(g["succ"].mean())
    lg_epa = float(g["epa"].mean(skipna=True))

    prof = {}
    for _, r in d.iterrows():
        prof[r["defteam"]] = dict(
            r_stuff=float(r["stuff"] / lg_stuff) if lg_stuff > 0 else 1.0,
            r_expl=float(r["expl"] / lg_expl) if lg_expl > 0 else 1.0,
            r_eff=float(r["succ"] / lg_succ) if lg_succ > 0 else 1.0,
            stuff_allowed=float(r["stuff"]), expl_allowed=float(r["expl"]),
            expl15_allowed=float(r["expl15"]), succ_allowed=float(r["succ"]),
            epa_allowed=float(r["epa"]),
            lg_stuff=lg_stuff, lg_expl=lg_expl, lg_expl15=lg_expl15,
            lg_succ=lg_succ, lg_epa=lg_epa, def_runs=int(r["runs"]))
    return prof


# ---------------------------------------------------------------------------
# Drive table (Phase 5): the unit the game engine simulates
# ---------------------------------------------------------------------------
# The scoring engine is drive-based, so the atom of the team-strength layer is
# ONE DRIVE: who had the ball, against whom, and what it produced. nflverse's
# `fixed_drive` numbers possessions consistently within a game, and
# `fixed_drive_result` labels how each one ended.

# Points credited to the offense for how its drive ended. A touchdown is worth
# slightly less than 7 because extra points are missed and two-pointers fail;
# "Opp touchdown" is a defensive return score against this offense.
DRIVE_POINTS = {
    "Touchdown": 6.96,
    "Field goal": 3.0,
    "Safety": -2.0,
    "Opp touchdown": -6.96,
}

# Drive endings that are clock artefacts rather than football outcomes; they are
# flagged so efficiency ratings can exclude kneel-downs and half-enders.
_DEAD_DRIVE_RESULTS = {"End of half", "End of game"}

TURNOVER_RESULTS = {"Turnover", "Opp touchdown"}


def load_drives(seasons: tuple[int, ...]) -> pd.DataFrame:
    """One row per drive: posteam / defteam / result / points / red-zone flag.

    Columns: season, week, game_id, drive, posteam, defteam, result, points,
    plays, inside20, start_yl (yards from the opponent's end zone at the start),
    is_td, is_fg, is_turnover, live (False for clock-artefact drives).
    Empty frame if the pbp feed or its drive columns are unavailable.
    """
    raw = load_pbp_raw(seasons)
    need = {"game_id", "fixed_drive", "fixed_drive_result", "posteam", "defteam"}
    if raw.empty or not need <= set(raw.columns):
        return pd.DataFrame()

    d = raw.dropna(subset=["posteam", "defteam", "fixed_drive"]).copy()
    agg = {"result": ("fixed_drive_result", "first"), "plays": ("play_type", "size")}
    if "drive_inside20" in d.columns:
        agg["inside20"] = ("drive_inside20", "max")
    if "yardline_100" in d.columns:
        agg["start_yl"] = ("yardline_100", "first")

    g = (d.groupby(["season", "week", "game_id", "fixed_drive", "posteam", "defteam"],
                   as_index=False)
           .agg(**agg)
           .rename(columns={"fixed_drive": "drive"}))

    g["points"] = g["result"].map(DRIVE_POINTS).fillna(0.0).astype(float)
    g["is_td"] = (g["result"] == "Touchdown").astype(int)
    g["is_fg"] = (g["result"] == "Field goal").astype(int)
    g["is_turnover"] = g["result"].isin(TURNOVER_RESULTS).astype(int)
    g["live"] = ~g["result"].isin(_DEAD_DRIVE_RESULTS)
    if "inside20" in g.columns:
        g["inside20"] = pd.to_numeric(g["inside20"], errors="coerce").fillna(0).astype(int)

    latest = int(g["season"].max())
    g["season_w"] = g["season"].map(lambda x: season_weight(int(x), latest))
    return g


def load_games(seasons: tuple[int, ...]) -> pd.DataFrame:
    """One row per game with the FINAL score (season, week, game_id, home/away
    team and score). The drive table only knows about points scored on drives,
    so the game engine needs the real finals to calibrate the scoring level:
    kick/punt return touchdowns and safeties never belong to any drive.
    Empty frame if pbp is unavailable."""
    raw = load_pbp_raw(seasons)
    need = {"game_id", "home_team", "away_team", "home_score", "away_score"}
    if raw.empty or not need <= set(raw.columns):
        return pd.DataFrame()
    cols = ["season", "week", "game_id", "home_team", "away_team",
            "home_score", "away_score"]
    g = raw[[c for c in cols if c in raw.columns]].drop_duplicates("game_id").copy()
    for c in ("home_score", "away_score"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    return g.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)


def team_game_points(games: pd.DataFrame) -> pd.DataFrame:
    """`games` reshaped to one row per team-game: season, game_id, team, points."""
    if games is None or games.empty:
        return pd.DataFrame(columns=["season", "game_id", "team", "points"])
    home = games.rename(columns={"home_team": "team", "home_score": "points"})
    away = games.rename(columns={"away_team": "team", "away_score": "points"})
    keep = ["season", "game_id", "team", "points"]
    return pd.concat([home[keep], away[keep]], ignore_index=True)


# ---------------------------------------------------------------------------
# QB disruption: sacks (taken) & INTs (thrown) — Phase 4
# ---------------------------------------------------------------------------
# Both are OFFENSE-side rates read from the weekly feed (sacks taken per
# dropback, INTs thrown per attempt), nudged by the opponent's defensive rate.
# Rates are combined in LOG-ODDS so "how much this defense deviates from league"
# transfers onto any QB's own rate without pushing a probability past 0/1.

LG_SACK_RATE = 0.065     # league sacks taken per dropback (fallback)
LG_INT_RATE = 0.023      # league INTs thrown per attempt (fallback)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def combine_rate_logodds(base_rate, def_ratio, lg_rate, strength):
    """Adjust an offense's own `base_rate` by how far a defense sits from league.

    `def_ratio` = defense rate / league rate. We move `base_rate` in log-odds by
    the defense's league-relative log-odds shift, scaled by `strength` (shrinkage
    toward no-adjustment). Returns a probability. Vectorises over base_rate.
    """
    def_ratio = float(max(def_ratio, 1e-6))
    lg_rate = float(np.clip(lg_rate, 1e-6, 1 - 1e-6))
    def_rate = float(np.clip(lg_rate * def_ratio, 1e-6, 1 - 1e-6))
    shift = (_logit(def_rate) - _logit(lg_rate)) * float(strength)
    return _expit(_logit(base_rate) + shift)


def def_pass_rates(wk: pd.DataFrame) -> dict:
    """Per defense: sack rate (per dropback) and INT rate (per attempt) generated,
    as ratios vs league. r>1 = a more disruptive pass defense. Uses the offense
    feed grouped by opponent, so it needs no defensive box scores."""
    neutral = dict(r_sack=1.0, r_int=1.0, sack_rate_allowed=LG_SACK_RATE,
                   int_rate_allowed=LG_INT_RATE, lg_sack=LG_SACK_RATE,
                   lg_int=LG_INT_RATE, pass_plays=0)
    if not {"sacks", "interceptions", "attempts", "dropbacks"} <= set(wk.columns):
        return {"_LEAGUE_": neutral}

    g = (wk.groupby("opponent_team", as_index=False)
           .agg(sacks=("sacks", "sum"), dropbacks=("dropbacks", "sum"),
                ints=("interceptions", "sum"), atts=("attempts", "sum")))
    lg_sack = float(g["sacks"].sum() / max(g["dropbacks"].sum(), 1))
    lg_int = float(g["ints"].sum() / max(g["atts"].sum(), 1))
    lg_sack = lg_sack if lg_sack > 0 else LG_SACK_RATE
    lg_int = lg_int if lg_int > 0 else LG_INT_RATE
    prof = {}
    for _, r in g.iterrows():
        db, at = r["dropbacks"], r["atts"]
        sack_rate = r["sacks"] / db if db >= 150 else lg_sack
        int_rate = r["ints"] / at if at >= 150 else lg_int
        prof[r["opponent_team"]] = dict(
            r_sack=float(sack_rate / lg_sack) if lg_sack > 0 else 1.0,
            r_int=float(int_rate / lg_int) if lg_int > 0 else 1.0,
            sack_rate_allowed=float(sack_rate), int_rate_allowed=float(int_rate),
            lg_sack=lg_sack, lg_int=lg_int, pass_plays=int(db))
    prof["_LEAGUE_"] = dict(r_sack=1.0, r_int=1.0, sack_rate_allowed=lg_sack,
                            int_rate_allowed=lg_int, lg_sack=lg_sack, lg_int=lg_int,
                            pass_plays=0)
    return prof


# --- Next Gen Stats passing (optional): avg_time_to_throw for the sack model ---
# NGS ships as ONE all-seasons file per stat group (not one per season), with
# `week == 0` rows carrying the season-level aggregate per player.
_NGS_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
            "nextgen_stats/ngs_passing.parquet")

LG_TIME_TO_THROW = 2.75   # league-ish avg seconds to throw (fallback anchor)


@functools.lru_cache(maxsize=4)
def load_ngs_pass(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Season-level NGS passing rows (week 0) for the requested seasons.
    Empty frame if unavailable — the sack model then skips the time-to-throw
    scaling. Kept optional so nothing hard-depends on NGS loading."""
    try:
        d = pd.read_parquet(_NGS_URL)
    except Exception:
        return pd.DataFrame()
    if "season" in d.columns:
        d = d[d["season"].astype(int).isin([int(s) for s in seasons])]
    # season-level ('week' 0) rows when present, else all
    if "week" in d.columns and (d["week"] == 0).any():
        d = d[d["week"] == 0]
    return d.copy()


def ngs_time_to_throw(ngs: pd.DataFrame) -> dict:
    """gsis_id -> avg_time_to_throw (seconds), plus league mean under key
    '_LEAGUE_'. Empty dict (league only) if the feed lacks the column.

    A QB with several seasons of NGS is collapsed to one number, weighting
    recent seasons more heavily (same curve as the weekly priors).
    """
    if ngs is None or ngs.empty or "avg_time_to_throw" not in ngs.columns:
        return {"_LEAGUE_": LG_TIME_TO_THROW}
    idcol = "player_gsis_id" if "player_gsis_id" in ngs.columns else (
        "gsis_id" if "gsis_id" in ngs.columns else None)

    d = ngs.copy()
    d["ttt"] = pd.to_numeric(d["avg_time_to_throw"], errors="coerce")
    d = d[d["ttt"].notna()]
    if d.empty:
        return {"_LEAGUE_": LG_TIME_TO_THROW}

    lg_mean = float(d["ttt"].mean())
    out = {"_LEAGUE_": lg_mean if np.isfinite(lg_mean) and lg_mean > 0
           else LG_TIME_TO_THROW}
    if idcol is None:
        return out

    if "season" in d.columns:
        latest = int(d["season"].max())
        d["w"] = d["season"].map(lambda s: season_weight(int(s), latest))
    else:
        d["w"] = 1.0
    d = d[d[idcol].notna()]
    for pid, grp in d.groupby(d[idcol].astype(str)):
        out[pid] = float(np.average(grp["ttt"], weights=grp["w"]))
    return out
