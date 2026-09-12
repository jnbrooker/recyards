"""
nflsim/data.py — shared NFL data layer for the whole model suite.

Every per-stat page and the game simulator import from here so they all speak
the same data. Responsibilities:

  * pull & cache nflverse weekly player stats (offense: passing/rushing/receiving)
  * pull & cache depth charts and injury reports (for the game simulator)
  * recency weighting (recent seasons AND recent games weighted more heavily)
  * small shared stats helpers (weighted mean/std, Beta matching, shrinkage)

Design note: this generalises the loader from the original receiving model so
it keeps *all* offensive columns (not just receiving), because rushing, TDs,
sacks and INTs all read from the same weekly release.
"""

from __future__ import annotations

import datetime as _dt
import functools
import time as _time
import urllib.request
from typing import NamedTuple
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Caching: nflverse republishes the weekly feeds within hours of games, the
# depth charts and injury reports daily. A plain lru_cache would pin the first
# download for the life of the process, so every loader uses a cache keyed on a
# time bucket instead — after REFRESH_HOURS the next call re-downloads. This is
# what lets a long-running app pick up a new week without a restart.
# ---------------------------------------------------------------------------

REFRESH_HOURS = 6.0


def ttl_cache(maxsize: int = 8):
    """lru_cache that forgets everything every REFRESH_HOURS."""
    def deco(fn):
        @functools.lru_cache(maxsize=maxsize)
        def inner(_bucket, *args, **kwargs):
            return fn(*args, **kwargs)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bucket = int(_time.time() // (REFRESH_HOURS * 3600))
            return inner(bucket, *args, **kwargs)

        wrapper.cache_clear = inner.cache_clear
        return wrapper
    return deco

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OFFENSE_POSITIONS = ["QB", "RB", "WR", "TE", "FB"]

# How hard to lean on a single defense's (noisy) season splits.
DEFAULT_DEF_SHRINK = 0.6


# ---------------------------------------------------------------------------
# Recency: how much a game counts, by how long ago it was played
# ---------------------------------------------------------------------------
# Every rate, share and volume in the suite is a WEIGHTED estimate, and every
# feed carries the same two columns so the weighting is consistent everywhere:
#
#   season_w  — the season curve: `season_decay ** (seasons ago)`
#   w         — season_w × 0.5 ** (games_ago / half_life), where games_ago is
#               counted PER TEAM over the games it has actually played, running
#               continuously back through earlier seasons (bye-aware: a bye is
#               not a game). So last season's finale is a few games staler than
#               this season's opener, and its week 1 is a whole season staler.
#
# `Recency` is the single control the pages expose ("how much to trust this
# season"); the presets below map it to the two constants. The current season's
# share of the weight, with two prior seasons in the window:
#
#                  after week:   1     4     8    12    17
#   Long memory  (0.70, inf)     5%   17%   28%   37%   46%   ← the pre-2026-09
#   Balanced     (0.85, 12)      8%   27%   46%   59%   70%     flat curve
#   Recent form  (0.70, 6)      16%   47%   70%   82%   90%
#
# Weighted totals shrink toward the regression pseudo-counts (SACK_PRIOR_N,
# CATCH_PRIOR_N, …) faster than raw counts do, which is the intended behaviour:
# a rate built on old games deserves more regression than one built on this
# month's.

class Recency(NamedTuple):
    season_decay: float = 0.85   # weight multiplier per season of age
    half_life: float = 12.0      # games; a game this many back counts half.
                                 # inf = seasons as flat blocks (no game decay)


RECENCY_DEFAULT = Recency()

RECENCY_PRESETS = {
    "Long memory": Recency(season_decay=0.7, half_life=float("inf")),
    "Balanced": RECENCY_DEFAULT,
    "Recent form": Recency(season_decay=0.7, half_life=6.0),
}


def season_weight(season: int, latest: int,
                  season_decay: float = RECENCY_DEFAULT.season_decay) -> float:
    """More recent seasons carry more weight when building priors."""
    gap = max(int(latest) - int(season), 0)
    return float(season_decay) ** gap


def game_weights(df: pd.DataFrame, team_col: str,
                 recency: Recency = RECENCY_DEFAULT,
                 season_col: str = "season", week_col: str = "week") -> pd.DataFrame:
    """Return `df` with `season_w` and `w` set (see the block comment above).

    Works on any per-game frame with a season, a week and a team column: the
    weekly player feed (`recent_team`), the drive table (`posteam`), pbp
    (`defteam`) and PFR (`team`). `games_ago` is a dense rank of the distinct
    (season, week) pairs each team has played, newest first, so a bye week does
    not count as a game and the count runs straight through the off-season.
    """
    r = Recency(*recency)
    out = df.copy()
    if season_col not in out.columns or out.empty:
        out["season_w"] = 1.0
        out["w"] = 1.0
        return out
    season = pd.to_numeric(out[season_col], errors="coerce")
    latest = int(season.max())
    sw = np.power(float(r.season_decay), (latest - season).clip(lower=0).astype(float))
    w = sw.to_numpy(dtype=float).copy()
    if (np.isfinite(r.half_life) and r.half_life > 0
            and week_col in out.columns and team_col in out.columns):
        stamp = season * 100 + pd.to_numeric(out[week_col], errors="coerce")
        ago = stamp.groupby(out[team_col]).rank(method="dense", ascending=False) - 1.0
        w *= np.power(0.5, ago.to_numpy(dtype=float) / float(r.half_life))
    out["season_w"] = sw.to_numpy(dtype=float)
    out["w"] = w
    return out


def weight_shares(df: pd.DataFrame, team_col: str = "recent_team") -> dict:
    """Share of total weight each season carries, over team-games. For the
    sidebar caption: 'this season is 38% of the model'."""
    if df is None or df.empty or "w" not in df.columns:
        return {}
    keys = [c for c in (team_col, "season", "week") if c in df.columns]
    tg = df.drop_duplicates(keys)
    tot = float(tg["w"].sum())
    if tot <= 0:
        return {}
    return {int(s): float(v) for s, v in (tg.groupby("season")["w"].sum() / tot).items()}


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


@ttl_cache(maxsize=8)
def _load_weekly_raw(seasons: tuple[int, ...]) -> pd.DataFrame:
    """The download half of `load_weekly` (cached); weights are applied on top."""
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

    df.attrs["loaded_seasons"] = sorted(loaded)
    df.attrs["missing_seasons"] = sorted(missing)
    return df


def load_weekly(seasons: tuple[int, ...],
                recency: Recency = RECENCY_DEFAULT) -> pd.DataFrame:
    """Regular-season weekly offensive lines for the given seasons, with the
    recency weights `season_w` / `w` on every row (see `game_weights`).

    `seasons` is a tuple so it is hashable for the cache. Seasons not yet
    available are skipped; if none load, a clear error is raised. The download
    is cached; re-weighting for a different `recency` is a cheap copy.
    """
    raw = _load_weekly_raw(tuple(int(s) for s in seasons))
    df = game_weights(raw, "recent_team", recency)
    df.attrs.update(raw.attrs)
    df.attrs["recency"] = tuple(Recency(*recency))
    return df


def _url_exists(url: str, timeout: float = 6.0) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "nflsim"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


@ttl_cache(maxsize=4)
def available_seasons(candidates: tuple[int, ...]) -> tuple[int, ...]:
    """Which of `candidates` have a published weekly stats file right now.

    A HEAD request per season, cached for REFRESH_HOURS, so the app can decide
    its default season window from what nflverse has actually released — the
    new season appears in the defaults the week its first file lands.
    """
    out = []
    for yr in candidates:
        if any(_url_exists(t.format(year=int(yr))) for t in _WEEKLY_URLS):
            out.append(int(yr))
    return tuple(out)


def season_choices(n_options: int = 5, n_default: int = 3,
                   today: _dt.date | None = None) -> tuple[list[int], list[int]]:
    """(options, defaults) for a season picker.

    Options run back from the current calendar year. Defaults are the most
    recent `n_default` seasons that have data — so in-season the current year
    is included automatically (and weighted most heavily by `season_weight`),
    and in the off-season the picker falls back to the last completed ones.
    """
    today = today or _dt.date.today()
    options = list(range(today.year, today.year - n_options, -1))
    have = available_seasons(tuple(options))
    defaults = [s for s in options if s in have][:n_default]
    return options, (defaults or options[1:1 + n_default])


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

# The personnel groups in the current depth-chart feed. Older releases used
# `pos_grp == "OFF"` / plain positions; both are accepted.
_OFFENSE_GROUPS = ("3WR 1TE", "OFF", "Offense")
_DEFENSE_GROUPS = ("Base 3-4 D", "Base 4-3 D", "DEF", "Defense")

# Defensive depth-chart positions (both feed generations), for the
# availability layer. Side-specific abbreviations are kept as they come.
DEF_POSITIONS = {"DE", "DT", "NT", "LDE", "RDE", "LDT", "RDT", "EDGE",
                 "LB", "ILB", "MLB", "OLB", "WLB", "SLB", "LILB", "RILB", "LOLB", "ROLB",
                 "CB", "LCB", "RCB", "NB", "NCB", "DB", "S", "FS", "SS"}
OL_POSITIONS = {"LT", "LG", "C", "RG", "RT"}


@ttl_cache(maxsize=8)
def load_depth_charts(seasons: tuple[int, ...], side: str = "offense") -> pd.DataFrame:
    """Depth charts, normalised across feed versions.

    `side` is "offense" (skill positions only — what the game engine allocates
    to), "oline" (the five line slots), "defense" (every defensive slot), or
    "all". The current release is a stream of dated snapshots (`dt`) with
    `pos_abb` and `pos_rank`; older ones were weekly with `position` /
    `depth_team`. Both are mapped onto: team, player_id, player_name,
    position, depth (1 = starter), `dt` as a timestamp, and season / week
    when the feed carries them. Empty frame if the feed is unavailable.
    """
    dc = _read_seasons(_DEPTH_URL, seasons)
    if dc.empty:
        return dc

    # The two feed generations name things differently; when several seasons
    # of mixed generations are concatenated, coalesce each pair.
    def _coalesce(*names):
        have = [dc[n] for n in names if n in dc.columns]
        if not have:
            return None
        col = have[0]
        for other in have[1:]:
            col = col.fillna(other)
        return col

    team = _coalesce("team", "club_code")
    if "game_type" in dc.columns:           # older weekly feed includes playoffs
        dc = dc[dc["game_type"].isna() | (dc["game_type"] == "REG")]
        team = team.loc[dc.index]
    if "pos_grp" in dc.columns and side != "all":
        # group filter only where the row carries a group (the newer feed);
        # older rows fall through to the position filter below
        want = _OFFENSE_GROUPS if side in ("offense", "oline") else _DEFENSE_GROUPS
        keep = dc["pos_grp"].isna() | dc["pos_grp"].astype(str).isin(want)
        dc, team = dc[keep], team[keep]

    # slot label: the newer feed's pos_abb; the older feed's depth_position
    # (LT/LG/C/RG/RT for linemen, where `position` is only T/G/C)
    pos = _coalesce("pos_abb", "depth_position", "position")
    if "position" in dc.columns:
        blank = pos.isna() | (pos.astype(str).str.strip() == "")
        pos = pos.where(~blank, dc["position"])
    depth = _coalesce("pos_rank", "depth_team")
    pid = _coalesce("gsis_id", "player_id")
    if pos is None or depth is None or pid is None or team is None:
        return pd.DataFrame()

    out = pd.DataFrame({
        "team": team,
        "player_id": pid,
        "player_name": _coalesce("player_name", "full_name"),
        "position": pos.astype(str),
        "depth": pd.to_numeric(depth, errors="coerce"),
    })
    for c in ("season", "week"):
        if c in dc.columns:
            out[c] = pd.to_numeric(dc[c], errors="coerce")
    out["dt"] = (pd.to_datetime(dc["dt"], errors="coerce", utc=True)
                 if "dt" in dc.columns else pd.NaT)
    if {"season", "week"} <= set(dc.columns):
        # older weekly feed: synthesise an orderable stamp — the Tuesday of
        # that week in a September-start season, so `as_of` a kickoff picks
        # the chart published for that week and not a later one
        season_start = pd.to_datetime(dc["season"].astype("Int64").astype(str) + "-09-01",
                                      errors="coerce", utc=True)
        synth = season_start + pd.to_timedelta((pd.to_numeric(dc["week"], errors="coerce") - 1) * 7, unit="D")
        out["dt"] = out["dt"].fillna(synth)

    if side == "offense":
        out = out[out["position"].isin(SKILL_POSITIONS)]
    elif side == "oline":
        out = out[out["position"].isin(OL_POSITIONS)]
    elif side == "defense":
        out = out[out["position"].isin(DEF_POSITIONS)]
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


@ttl_cache(maxsize=8)
def load_injuries(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Weekly injury reports (direct parquet). report_status in {Out, Doubtful, Questionable}."""
    return _read_seasons(_INJURY_URL, seasons)


@ttl_cache(maxsize=4)
def _pfr_id_crosswalk() -> pd.DataFrame:
    """gsis_id <-> pfr_id map (dynastyprocess), so PFR stats can join to players."""
    try:
        ids = pd.read_csv(_IDS_URL, low_memory=False)
        return (ids[["gsis_id", "pfr_id"]].dropna(subset=["gsis_id", "pfr_id"])
                  .drop_duplicates("gsis_id"))
    except Exception:
        return pd.DataFrame(columns=["gsis_id", "pfr_id"])


@ttl_cache(maxsize=8)
def _load_pfr_rush_raw(seasons: tuple[int, ...]) -> pd.DataFrame:
    d = _read_seasons(_PFR_RUSH_URL, seasons)
    if d.empty:
        return d
    if "game_type" in d.columns:
        d = d[d["game_type"] == "REG"]
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


def load_pfr_rush(seasons: tuple[int, ...],
                  recency: Recency = RECENCY_DEFAULT) -> pd.DataFrame:
    """PFR advanced weekly rushing (direct parquet): yards before/after contact and
    broken tackles per game, keyed to gsis_id, with recency weights `w`. Empty
    frame if unavailable — callers fall back to a yards-per-carry split."""
    d = _load_pfr_rush_raw(tuple(int(s) for s in seasons))
    if d.empty:
        return d
    return game_weights(d, "team", recency)


_SNAP_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
             "snap_counts/snap_counts_{year}.parquet")


@ttl_cache(maxsize=8)
def load_snap_counts(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Per player-game snap shares (regular season), keyed to gsis_id via the
    PFR crosswalk: season, week, team, opponent, player_id, position,
    offense_pct, defense_pct. Empty frame if unavailable."""
    d = _read_seasons(_SNAP_URL, seasons)
    if d.empty:
        return d
    if "game_type" in d.columns:
        d = d[d["game_type"] == "REG"]
    xw = _pfr_id_crosswalk().rename(columns={"pfr_id": "pfr_player_id", "gsis_id": "player_id"})
    d = d.merge(xw, on="pfr_player_id", how="left") if not xw.empty else d.assign(player_id=np.nan)
    keep = ["season", "week", "team", "opponent", "player_id", "player", "position",
            "offense_pct", "defense_pct", "offense_snaps", "defense_snaps"]
    d = d[[c for c in keep if c in d.columns]].dropna(subset=["player_id"]).copy()
    for c in ("offense_pct", "defense_pct", "offense_snaps", "defense_snaps"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d["player_id"] = d["player_id"].astype(str)
    return d.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Schedule (for picking a real game to simulate)
# ---------------------------------------------------------------------------
# One all-seasons CSV. Besides the fixture list it carries closing market lines
# (`spread_line`, positive = home favoured; `total_line`), roof and weather —
# useful as COMPARATORS beside the model. Nothing here is ever fed into a
# rating; the model stays self-contained.

_SCHEDULE_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
                 "schedules/games.csv")

_SCHED_KEEP = ["game_id", "season", "game_type", "week", "gameday", "weekday",
               "gametime", "away_team", "home_team", "away_score", "home_score",
               "spread_line", "total_line", "away_moneyline", "home_moneyline",
               "roof", "surface", "temp", "wind",
               "away_qb_name", "home_qb_name", "div_game", "stadium"]


@ttl_cache(maxsize=4)
def load_schedule(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Regular-season fixtures for the given seasons, one row per game.

    `played` is True once a final score exists. Empty frame if unavailable.
    """
    try:
        d = pd.read_csv(_SCHEDULE_URL, low_memory=False)
    except Exception:
        return pd.DataFrame()
    d = d[d["season"].isin([int(s) for s in seasons])]
    if "game_type" in d.columns:
        d = d[d["game_type"] == "REG"]
    d = d[[c for c in _SCHED_KEEP if c in d.columns]].copy()
    for c in ("away_score", "home_score", "spread_line", "total_line",
              "away_moneyline", "home_moneyline", "temp", "wind"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["played"] = d["home_score"].notna() & d["away_score"].notna()
    d["kickoff"] = pd.to_datetime(d["gameday"].astype(str) + " " + d["gametime"].astype(str),
                                  errors="coerce")
    return d.sort_values(["season", "week", "kickoff"]).reset_index(drop=True)


def current_week(sched: pd.DataFrame, season: int | None = None) -> int:
    """The week to default to: the first week of the (latest) season that still
    has an unplayed game, or the final week once the season is done."""
    if sched is None or sched.empty:
        return 1
    s = sched[sched["season"] == (int(season) if season is not None else sched["season"].max())]
    if s.empty:
        return 1
    open_weeks = s.loc[~s["played"], "week"]
    return int(open_weeks.min()) if len(open_weeks) else int(s["week"].max())


def game_label(row) -> str:
    """'CHI @ CAR — Sun 13 Sep 13:00' with the final score once played."""
    when = row["kickoff"]
    when_txt = when.strftime("%a %d %b %H:%M") if pd.notna(when) else str(row.get("gameday", ""))
    s = f"{row['away_team']} @ {row['home_team']} — {when_txt}"
    if bool(row.get("played")):
        s += f"  (final {int(row['away_score'])}-{int(row['home_score'])})"
    return s


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


def between_sd(x, w, sampling_var, floor: float, fallback: float | None = None) -> float:
    """The game-to-game spread of a per-game rate, net of its sampling noise.

    A per-game catch rate on six targets, or yards per carry on twelve carries,
    varies mostly because the sample is tiny — and the simulator already draws
    that noise (Binomial on the targets, Gamma per carry). Feeding the raw
    per-game SD in as a game-level wobble counts it twice, which is why the
    first backtest found 80% bands covering 93% of outcomes. Method of moments:
    true var = observed var - mean sampling var, floored at `floor`.
    `sampling_var` is per game (same length as `x`).
    """
    x = np.asarray(x, float); w = np.asarray(w, float); sv = np.asarray(sampling_var, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0) & np.isfinite(sv)
    if m.sum() < 3:
        return float(fallback if fallback is not None else floor)
    obs = wstd(x[m], w[m], 0.0) ** 2
    noise = float(np.average(sv[m], weights=w[m]))
    return float(max(np.sqrt(max(obs - noise, 0.0)), floor))


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
    # who touched the ball (goal-line role for touchdowns)
    "rusher_player_id", "receiver_player_id", "pass_attempt", "complete_pass",
    "rush_touchdown", "pass_touchdown",
]

# Thresholds defining a stuffed / explosive run.
STUFF_MAX = 0        # yards_gained <= 0  → stuffed (TFL / no gain)
EXPL_MIN = 10        # yards_gained >= 10 → explosive
EXPL15_MIN = 15      # yards_gained >= 15 → chunk/breakaway

MIN_DEF_RUNS = 50    # designed runs a defense must have faced to get a profile


@ttl_cache(maxsize=4)
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


def load_pbp(seasons: tuple[int, ...],
             recency: Recency = RECENCY_DEFAULT) -> pd.DataFrame:
    """Regular-season *designed-run* plays (QB scrambles, kneels and two-point
    plays removed) with the columns needed to score stuff / explosive /
    efficiency, weighted by recency (`w`, keyed on the defense's schedule since
    the run-defense profile is the consumer). Empty frame if pbp is unavailable
    — callers then skip them."""
    pbp = load_pbp_raw(seasons)
    if pbp.empty:
        return pbp
    pbp = game_weights(pbp, "defteam", recency)

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


def _wmean_by(df: pd.DataFrame, key, cols: list[str], w: str = "w") -> pd.DataFrame:
    """Weighted mean of each column in `cols`, grouped by `key`."""
    ww = df[w].astype(float)
    num = df[cols].multiply(ww, axis=0).groupby(df[key] if isinstance(key, str) else key).sum()
    den = ww.groupby(df[key] if isinstance(key, str) else key).sum()
    return num.div(den, axis=0)


GOAL_LINE_YL = 10      # yards from the end zone that define a goal-line touch


def load_touches(seasons: tuple[int, ...],
                 recency: Recency = RECENCY_DEFAULT) -> pd.DataFrame:
    """One row per player-game of ball touches from pbp: carries and targets,
    how many came inside the GOAL_LINE_YL, and the touchdowns — the direct
    measure of goal-line ROLE (roadmap §3.3 / §8.2). Weighted by `w`. Empty
    frame if the pbp feed lacks the player-id columns."""
    raw = load_pbp_raw(seasons)
    need = {"rusher_player_id", "receiver_player_id", "yardline_100", "posteam"}
    if raw.empty or not need <= set(raw.columns):
        return pd.DataFrame()
    x = raw[pd.to_numeric(raw.get("two_point_attempt"), errors="coerce").fillna(0) != 1]
    yl = pd.to_numeric(x["yardline_100"], errors="coerce")
    gl = (yl <= GOAL_LINE_YL).astype(int)
    rush = x[(pd.to_numeric(x["rush_attempt"], errors="coerce") == 1) & x["rusher_player_id"].notna()]
    pas = x[(pd.to_numeric(x["pass_attempt"], errors="coerce") == 1) & x["receiver_player_id"].notna()]
    keys = ["season", "week", "posteam"]
    r = (rush.assign(player_id=rush["rusher_player_id"].astype(str), gl=gl.loc[rush.index],
                     td=pd.to_numeric(rush.get("rush_touchdown"), errors="coerce").fillna(0))
             .groupby(keys + ["player_id"], as_index=False)
             .agg(car=("gl", "size"), gl_car=("gl", "sum"), rush_td=("td", "sum")))
    p = (pas.assign(player_id=pas["receiver_player_id"].astype(str), gl=gl.loc[pas.index],
                    td=pd.to_numeric(pas.get("pass_touchdown"), errors="coerce").fillna(0))
            .groupby(keys + ["player_id"], as_index=False)
            .agg(tgt=("gl", "size"), gl_tgt=("gl", "sum"), rec_td=("td", "sum")))
    t = r.merge(p, on=keys + ["player_id"], how="outer").fillna(0)
    for c in ("car", "gl_car", "rush_td", "tgt", "gl_tgt", "rec_td"):
        t[c] = t[c].astype(int)
    return game_weights(t, "posteam", recency)


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
    if "w" not in g.columns:
        g["w"] = 1.0
    g["is_stuff"] = (g["gain"] <= STUFF_MAX).astype(float)
    g["is_expl"] = (g["gain"] >= EXPL_MIN).astype(float)
    g["is_expl15"] = (g["gain"] >= EXPL15_MIN).astype(float)

    # Rates are recency-weighted; the sample-size guard stays on raw run counts.
    cols = ["is_stuff", "is_expl", "is_expl15", "succ", "epa"]
    d = _wmean_by(g, "defteam", cols).rename(columns=dict(
        is_stuff="stuff", is_expl="expl", is_expl15="expl15"))
    d["runs"] = g.groupby("defteam")["gain"].size()
    d = d[d["runs"] >= MIN_DEF_RUNS].reset_index()
    if d.empty:
        return {}

    lg_stuff = float(wmean(g["is_stuff"], g["w"]))
    lg_expl = float(wmean(g["is_expl"], g["w"]))
    lg_expl15 = float(wmean(g["is_expl15"], g["w"]))
    lg_succ = float(wmean(g["succ"], g["w"]))
    lg_epa = float(wmean(g["epa"], g["w"]))

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


def load_drives(seasons: tuple[int, ...],
                recency: Recency = RECENCY_DEFAULT) -> pd.DataFrame:
    """One row per drive: posteam / defteam / result / points / red-zone flag.

    Columns: season, week, game_id, drive, posteam, defteam, result, points,
    plays, inside20, start_yl (yards from the opponent's end zone at the start),
    is_td, is_fg, is_turnover, live (False for clock-artefact drives), and the
    recency weights season_w / w (keyed on the offense's schedule).
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

    return game_weights(g, "posteam", recency)


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

    # Weighted totals give the rates; raw dropbacks/attempts guard sample size.
    x = wk.assign(_w=wk["w"] if "w" in wk.columns else 1.0)
    for c in ("sacks", "dropbacks", "interceptions", "attempts"):
        x[f"w_{c}"] = x[c] * x["_w"]
    g = (x.groupby("opponent_team", as_index=False)
          .agg(sacks=("w_sacks", "sum"), dropbacks=("w_dropbacks", "sum"),
               ints=("w_interceptions", "sum"), atts=("w_attempts", "sum"),
               raw_db=("dropbacks", "sum"), raw_att=("attempts", "sum")))
    lg_sack = float(g["sacks"].sum() / max(g["dropbacks"].sum(), 1e-9))
    lg_int = float(g["ints"].sum() / max(g["atts"].sum(), 1e-9))
    lg_sack = lg_sack if lg_sack > 0 else LG_SACK_RATE
    lg_int = lg_int if lg_int > 0 else LG_INT_RATE
    prof = {}
    for _, r in g.iterrows():
        db, at = r["dropbacks"], r["atts"]
        sack_rate = r["sacks"] / db if r["raw_db"] >= 150 and db > 0 else lg_sack
        int_rate = r["ints"] / at if r["raw_att"] >= 150 and at > 0 else lg_int
        prof[r["opponent_team"]] = dict(
            r_sack=float(sack_rate / lg_sack) if lg_sack > 0 else 1.0,
            r_int=float(int_rate / lg_int) if lg_int > 0 else 1.0,
            sack_rate_allowed=float(sack_rate), int_rate_allowed=float(int_rate),
            lg_sack=lg_sack, lg_int=lg_int, pass_plays=int(r["raw_db"]))
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


@ttl_cache(maxsize=4)
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


def ngs_time_to_throw(ngs: pd.DataFrame,
                      recency: Recency = RECENCY_DEFAULT) -> dict:
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
        d["w"] = d["season"].map(
            lambda s: season_weight(int(s), latest, Recency(*recency).season_decay))
    else:
        d["w"] = 1.0
    d = d[d[idcol].notna()]
    for pid, grp in d.groupby(d[idcol].astype(str)):
        out[pid] = float(np.average(grp["ttt"], weights=grp["w"]))
    return out
