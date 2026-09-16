"""
nflsim/playengine.py — a PLAY-LEVEL game engine, built from ten seasons of
play-by-play. Lives beside the drive engine (`game.py`); nothing uses it unless
asked to (page 12, and the Engine toggle).

The drive engine resolves a possession in one draw. This one plays the game:

  state     down, distance, yard line, half, clock, score, who has the ball
  decision  pass / run / punt / field goal / go for it / kneel / spike, at the
            rates teams actually chose in that state
  outcome   an actual play drawn from the plays that happened in that state
            (yards, completion, sack, turnover, penalty, clock stop), shifted
            by the matchup's pass / rush efficiency ratings
  clock     seconds consumed by that kind of play in that part of the game
  kicks     field goals by distance, punts by field position, kickoffs under
            the current rules, extra points and two-point tries

Why: the score distribution then comes from the MECHANICS — games land on 3
and 7 because of how ends of games work (a trailing team's late touchdown, a
field goal to tie, a kneel-down), not because a curve was fitted. Play counts
are honest (a 10-drive game and a 13-drive game consume the same clock), which
is what player volume should hang off. And any state can be priced, which is
what live and alternate markets need.

Stage 1 (this file, first cut): the empirical tables, with a report that checks
them against league facts. Stage 2: the simulation loop. Stage 3: team
adjustment and the harness. Stage 4: players on top.

Simplifications, stated: timeouts are not tracked as a resource (their effect
is inside the empirical clock use in the last two minutes); penalties are
sampled as outcomes of the play they replaced; overtime is a single period,
first score wins, ties allowed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D

BASE = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

TABLE_SEASONS = tuple(range(2016, 2026))     # decisions, outcomes, clock
KICKOFF_SEASONS = (2025,)                    # the current kickoff rules only
MIN_CELL = 40                                # plays a cell needs before we trust it
CALL_CODE = {"pass": 0, "run": 1, "kneel": 2, "spike": 3}   # integer keys for the samplers

COLS = ["game_id", "play_id", "season", "week", "season_type", "posteam", "defteam",
        "home_team", "away_team", "qtr", "game_half", "half_seconds_remaining",
        "game_seconds_remaining", "down", "ydstogo", "yardline_100", "goal_to_go",
        "score_differential", "posteam_timeouts_remaining", "play_type", "yards_gained",
        "sack", "interception", "fumble_lost", "complete_pass", "incomplete_pass",
        "touchdown", "td_team", "return_touchdown", "safety", "field_goal_result",
        "kick_distance", "penalty", "penalty_yards", "penalty_team", "first_down",
        "out_of_bounds", "timeout", "two_point_attempt", "two_point_conv_result",
        "extra_point_result", "touchback", "qb_kneel", "qb_spike", "aborted_play",
        "punt_blocked", "epa", "pass", "rush"]


# ---------------------------------------------------------------------------
# Loading, with a disk cache
# ---------------------------------------------------------------------------

def _season_file(year: int) -> pd.DataFrame:
    import datetime as _dt
    CACHE_DIR.mkdir(exist_ok=True)
    f = CACHE_DIR / f"pbp_engine_{year}.parquet"
    current = int(year) >= _dt.date.today().year - (1 if _dt.date.today().month < 3 else 0)
    if f.exists() and not current:
        return pd.read_parquet(f)
    d = pd.read_parquet(BASE.format(year=year), columns=COLS)
    d = d[d["season_type"] == "REG"].reset_index(drop=True)
    if not current:                       # a finished season is written once
        d.to_parquet(f)
    return d


@D.ttl_cache(maxsize=2)
def load_plays(seasons: tuple[int, ...] = TABLE_SEASONS) -> pd.DataFrame:
    """Regular-season plays for the engine tables, ordered within each game,
    with the NEXT play's clock and field position attached (for clock use,
    punt / kickoff / turnover field position)."""
    frames = []
    for yr in seasons:
        try:
            frames.append(_season_file(int(yr)))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d = d.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    for c in ("down", "ydstogo", "yardline_100", "yards_gained", "score_differential",
              "half_seconds_remaining", "game_seconds_remaining", "kick_distance",
              "penalty_yards", "epa"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ("sack", "interception", "fumble_lost", "complete_pass", "incomplete_pass",
              "touchdown", "return_touchdown", "safety", "penalty", "first_down",
              "out_of_bounds", "timeout", "two_point_attempt", "touchback", "qb_kneel",
              "qb_spike", "aborted_play", "punt_blocked", "goal_to_go"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)
    # the next real play in the same game: its clock and its yard line / possession
    real = d["play_type"].notna() & (d["play_type"] != "no_play") | (d["penalty"] == 1)
    nxt = d[real].groupby("game_id")[["half_seconds_remaining", "game_half", "yardline_100", "posteam"]].shift(-1)
    d.loc[real, "next_half_secs"] = nxt["half_seconds_remaining"]
    d.loc[real, "next_half"] = nxt["game_half"]
    d.loc[real, "next_yl"] = nxt["yardline_100"]
    d.loc[real, "next_pos"] = nxt["posteam"]
    return d


# ---------------------------------------------------------------------------
# State bins
# ---------------------------------------------------------------------------

def ytg_bin(y):
    y = np.asarray(y, float)
    return np.select([y <= 1, y == 2, y <= 5, y <= 9, y == 10], [0, 1, 2, 3, 4], 5)


def yl_bin(yl):
    return np.clip((np.asarray(yl, float) - 1) // 10, 0, 9).astype(int)        # 0 = inside the 10


def sd_bin(sd):
    """Score differential bins. Exact within a field goal either way — being
    down 3 (kick to tie) is a different game from being down 1 (kick to win),
    and that difference is where the mass at 3 comes from."""
    sd = np.asarray(sd, float)
    return np.select([sd <= -14, sd <= -8, sd <= -4, sd == -3, sd == -2, sd == -1, sd == 0,
                      sd == 1, sd == 2, sd == 3, sd <= 7, sd <= 13],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 12)


def lead_class(sd):
    """-1 trailing by 9+, 0 within 8, +1 leading by 9+. Pace only changes once
    a game is out of one-score range: a trailing team snaps every ~28 s in the
    second half against ~37 s for a team up big; inside 8 points neither moves."""
    sd = np.asarray(sd, float)
    return np.where(sd <= -9, -1, np.where(sd >= 9, 1, 0))


def time_bin(half, secs):
    """0 H1 normal · 1 H1 last 2:00 · 2 H2 >10:00 · 3 H2 5-10 · 4 H2 2-5 · 5 H2 last 2:00 · 6 OT"""
    half = np.asarray(half); secs = np.asarray(secs, float)
    h2 = (half == "Half2") | (half == 2)
    ot = (half == "Overtime") | (half == 3)
    return np.select([ot, ~h2 & (secs > 120), ~h2, h2 & (secs > 600), h2 & (secs > 300), h2 & (secs > 120)],
                     [6, 0, 1, 2, 3, 4], 5)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

class Table:
    """A conditional distribution over categories keyed by a state tuple,
    with hierarchical fallback: if a cell is thin, drop the last key."""

    def __init__(self, keys: list[str], cats: list[str]):
        self.keys, self.cats = keys, cats
        self.cells: dict[tuple, np.ndarray] = {}

    def fit(self, df: pd.DataFrame, cat_col: str):
        df = df[df[cat_col].isin(self.cats)]
        c = pd.Categorical(df[cat_col], categories=self.cats)
        codes = c.codes
        for depth in range(len(self.keys), 0, -1):
            ks = self.keys[:depth]
            g = pd.DataFrame({k: df[k].to_numpy() for k in ks}).assign(_c=codes)
            g = g[g["_c"] >= 0]
            cnt = g.groupby(ks + ["_c"]).size().unstack("_c", fill_value=0)
            cnt = cnt.reindex(columns=range(len(self.cats)), fill_value=0)
            for key, row in cnt.iterrows():
                key = key if isinstance(key, tuple) else (key,)
                n = row.sum()
                if n >= MIN_CELL and key not in self.cells:
                    self.cells[key] = (row.to_numpy(float) / n)
        # the unconditional distribution is the last resort
        n = (codes >= 0).sum()
        self.cells[()] = np.bincount(codes[codes >= 0], minlength=len(self.cats)) / max(n, 1)
        return self

    def probs(self, key: tuple) -> np.ndarray:
        for depth in range(len(key), -1, -1):
            k = tuple(key[:depth])
            if k in self.cells:
                return self.cells[k]
        return self.cells[()]


class Sampler:
    """Empirical outcome rows keyed by a state tuple, with the same fallback.
    Holds row indices into an outcomes array; `draw` returns row indices."""

    def __init__(self, keys: list[str]):
        self.keys = keys
        self.cells: dict[tuple, np.ndarray] = {}

    def fit(self, df: pd.DataFrame):
        idx = np.arange(len(df))
        for depth in range(len(self.keys), 0, -1):
            ks = self.keys[:depth]
            g = pd.DataFrame({k: df[k].to_numpy() for k in ks}).assign(_i=idx)
            for key, rows in g.groupby(ks)["_i"]:
                key = key if isinstance(key, tuple) else (key,)
                if len(rows) >= MIN_CELL and key not in self.cells:
                    self.cells[key] = rows.to_numpy()
        self.cells[()] = idx
        return self

    def pool(self, key: tuple) -> np.ndarray:
        for depth in range(len(key), -1, -1):
            k = tuple(key[:depth])
            if k in self.cells:
                return self.cells[k]
        return self.cells[()]


def build_tables(plays: pd.DataFrame) -> dict:
    """Every empirical table the simulator needs, from the play feed."""
    d = plays.copy()
    d["ytg_b"] = ytg_bin(d["ydstogo"].fillna(10))
    d["yl_b"] = yl_bin(d["yardline_100"].fillna(50))
    d["sd_b"] = sd_bin(d["score_differential"].fillna(0))
    d["t_b"] = time_bin(d["game_half"], d["half_seconds_remaining"].fillna(900))
    d["down_i"] = d["down"].fillna(0).astype(int)

    scrim = d[d["down_i"].between(1, 4) & d["play_type"].isin(
        ["pass", "run", "punt", "field_goal", "qb_kneel", "qb_spike", "no_play"])].copy()
    # a penalty that wiped the play counts as its own outcome; other no_plays (timeouts) drop
    scrim = scrim[(scrim["play_type"] != "no_play") | (scrim["penalty"] == 1)]

    # --- decisions --------------------------------------------------------
    def call(r):
        pt = r["play_type"]
        if pt == "no_play":
            return "pass" if r["pass"] == 1 else "run" if r["rush"] == 1 else None
        return {"pass": "pass", "run": "run", "punt": "punt", "field_goal": "fg",
                "qb_kneel": "kneel", "qb_spike": "spike"}.get(pt)
    scrim["call"] = [call(r) for _, r in scrim[["play_type", "pass", "rush"]].iterrows()]
    scrim = scrim[scrim["call"].notna()]
    early = scrim[scrim["down_i"] <= 3]
    fourth = scrim[scrim["down_i"] == 4]
    tables = dict(
        # field goals (and the rare quick punt) happen on early downs when the
        # clock is about to run out; the cells make them vanish otherwise
        call_early=Table(["down_i", "ytg_b", "yl_b", "t_b", "sd_b"],
                         ["pass", "run", "kneel", "spike", "punt", "fg"]).fit(early, "call"),
        call_fourth=Table(["ytg_b", "yl_b", "t_b", "sd_b"],
                          ["pass", "run", "punt", "fg"]).fit(fourth[fourth["call"] != "kneel"], "call"),
    )

    # --- outcomes of pass and run plays (incl. penalties on them) -----------
    out = scrim[scrim["call"].isin(["pass", "run"])].copy()
    out["is_pen"] = ((out["play_type"] == "no_play") & (out["penalty"] == 1)).astype(int)
    # offense's gain from a penalty: positive if on the defense
    pen_gain = np.where(out["penalty_team"] == out["defteam"], out["penalty_yards"].fillna(0),
                        -out["penalty_yards"].fillna(0))
    out["gain"] = np.where(out["is_pen"] == 1, pen_gain, out["yards_gained"].fillna(0))
    # yards relative to the sticks: a play drawn from 3rd-and-3 applied to a
    # simulated 3rd-and-5 would fall short of sticks it actually made, so on
    # downs 2-4 the engine samples (gain - distance) and adds the simulated
    # distance back. Keeps every conversion rate exactly as the data has it.
    out["rel"] = out["gain"] - out["ydstogo"].fillna(10)
    out["to_int"] = out["interception"]
    out["to_fum"] = out["fumble_lost"]
    out["ret_td"] = ((out["touchdown"] == 1) & (out["td_team"] == out["defteam"])).astype(int)
    out["clock_stop"] = ((out["incomplete_pass"] == 1) | (out["out_of_bounds"] == 1)
                         | (out["to_int"] == 1) | (out["to_fum"] == 1) | (out["is_pen"] == 1)).astype(int)
    out["auto_fd"] = ((out["is_pen"] == 1) & (out["first_down"] == 1)).astype(int)
    # field position after a turnover (the return is inside it); NaN if none
    out["to_yl"] = np.where((out["to_int"] == 1) | (out["to_fum"] == 1), out["next_yl"], np.nan)
    out["call"] = out["call"].map(CALL_CODE)
    out = out.reset_index(drop=True)
    tables["outcomes"] = out[["call", "down_i", "ytg_b", "yl_b", "gain", "rel", "is_pen", "to_int", "to_fum",
                              "ret_td", "clock_stop", "auto_fd", "safety", "sack", "complete_pass",
                              "to_yl", "epa"]].copy()
    tables["outcome_sampler"] = Sampler(["call", "down_i", "ytg_b", "yl_b"]).fit(tables["outcomes"])

    # --- clock: seconds to the next play, by kind of play and part of game --
    ck = scrim[scrim["call"].isin(["pass", "run", "kneel", "spike"])].copy()
    ck["dt"] = ck["half_seconds_remaining"] - ck["next_half_secs"]
    ck = ck[(ck["next_half"] == ck["game_half"]) & ck["dt"].between(0, 60)]
    ck["stop"] = ((ck["incomplete_pass"] == 1) | (ck["out_of_bounds"] == 1)
                  | (ck["play_type"] == "no_play") | (ck["qb_spike"] == 1)).astype(int)
    ck["lead"] = lead_class(ck["score_differential"].fillna(0))
    ck["call"] = ck["call"].map(CALL_CODE)
    ck = ck.reset_index(drop=True)
    tables["clock"] = ck[["call", "stop", "t_b", "lead", "dt"]].copy()
    tables["clock_sampler"] = Sampler(["call", "stop", "t_b", "lead"]).fit(tables["clock"])

    # --- kicking ----------------------------------------------------------
    fg = d[(d["play_type"] == "field_goal") & d["field_goal_result"].notna()].copy()
    fg["dist"] = (fg["yardline_100"] + 17).clip(18, 70)
    fg["made"] = (fg["field_goal_result"] == "made").astype(int)
    # logistic fit of make probability on distance
    x, y = fg["dist"].to_numpy(float), fg["made"].to_numpy(float)
    tables["fg_coef"] = _logit_fit(x, y)
    pu = d[(d["play_type"] == "punt") & d["next_yl"].notna() & (d["next_pos"] != d["posteam"])].copy()
    pu["yl_b"] = yl_bin(pu["yardline_100"])
    pu["res_yl"] = pu["next_yl"]                          # receiving team's yard line to go
    pu = pu.reset_index(drop=True)
    tables["punts"] = pu[["yl_b", "res_yl", "punt_blocked"]].copy()
    tables["punt_sampler"] = Sampler(["yl_b"]).fit(tables["punts"])
    # on a kickoff the feed's `posteam` is the RECEIVING team, so the next play
    # (their first snap) has the same posteam
    ko = d[(d["play_type"] == "kickoff") & d["season"].isin(KICKOFF_SEASONS) & d["next_yl"].notna()
           & (d["next_pos"] == d["posteam"]) & (d["return_touchdown"] == 0)]
    tables["kickoff_yl"] = ko["next_yl"].to_numpy(float)
    tables["kickoff_td_rate"] = float(d[(d["play_type"] == "kickoff") & d["season"].isin(KICKOFF_SEASONS)]
                                      ["return_touchdown"].mean())
    xp = d[d["extra_point_result"].notna()]
    tables["xp_rate"] = float((xp["extra_point_result"] == "good").mean())
    tp = d[d["two_point_attempt"] == 1]
    tables["two_pt_rate"] = float((tp["two_point_conv_result"] == "success").mean())
    # go-for-two decision: the try's row carries the score with the six points
    # already on the board, so bin on that differential and the time
    tries = d[(d["two_point_attempt"] == 1) | d["extra_point_result"].notna()].copy()
    tries["try"] = np.where(tries["two_point_attempt"] == 1, "two", "one")
    tries["sd_b"] = sd_bin(tries["score_differential"].fillna(0))
    tries["t_b"] = time_bin(tries["game_half"], tries["half_seconds_remaining"].fillna(900))
    tables["try_call"] = Table(["sd_b", "t_b"], ["one", "two"]).fit(tries, "try")
    return tables


def _logit_fit(x, y, iters=200, lr=0.05):
    """Tiny logistic regression P(y=1) = sigmoid(a + b·x), gradient descent on
    standardised x — no dependency."""
    mu, sd = x.mean(), x.std()
    z = (x - mu) / sd
    a, b = 0.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a + b * z)))
        a -= lr * (p - y).mean() * 4
        b -= lr * ((p - y) * z).mean() * 4
    return dict(a=a, b=b, mu=mu, sd=sd)


def fg_make_prob(coef: dict, dist):
    z = (np.asarray(dist, float) - coef["mu"]) / coef["sd"]
    return 1 / (1 + np.exp(-(coef["a"] + coef["b"] * z)))


# ---------------------------------------------------------------------------
# Stage 2: the game, played one snap at a time for every simulation at once
# ---------------------------------------------------------------------------
# State lives in flat numpy arrays of length n (one entry per simulated game)
# and each step advances every live game by one event. Games in the same state
# bin share a draw from the same empirical cell, which is what makes this fast
# enough: a few hundred unique cells per step, ~300 steps, ~10k games.

HALF_SECS = 1800
OT_SECS = 600
MAX_STEPS = 420
PH_KICK, PH_PLAY, PH_TRY, PH_OVER = 0, 1, 2, 3

STAT_COLS = ["plays", "pass_att", "comp", "pass_yds", "rush_att", "rush_yds", "sacks", "sack_yds",
             "ints", "fum_lost", "pass_td", "rush_td", "fg_made", "fg_att", "punts", "drives",
             "first_downs", "def_td", "penalties"]
# pass_att excludes sacks (a sack is a dropback, not an attempt); pass_yds is
# GROSS (sack yardage in sack_yds), so it equals the sum of the receivers' yards


def _draw_table(rng, table: Table, keys: np.ndarray) -> np.ndarray:
    """Sample one category per row from `table`, rows grouped by their key."""
    out = np.zeros(len(keys), dtype=int)
    if len(keys) == 0:
        return out
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    u = rng.random(len(keys))
    for i, key in enumerate(uniq):
        m = inv == i
        cum = np.cumsum(table.probs(tuple(int(x) for x in key)))
        out[m] = np.minimum(np.searchsorted(cum, u[m], side="right"), len(cum) - 1)
    return out


def _draw_table_shift(rng, table: Table, keys: np.ndarray, pass_shift: np.ndarray) -> np.ndarray:
    """Like `_draw_table`, but category 0 (pass) gets a per-row logit shift and
    the other categories share the remainder in their original proportions."""
    out = np.zeros(len(keys), dtype=int)
    if len(keys) == 0:
        return out
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    u = rng.random(len(keys))
    for i, key in enumerate(uniq):
        m = np.where(inv == i)[0]
        pr = table.probs(tuple(int(x) for x in key))
        p0 = np.clip(pr[0], 1e-4, 1 - 1e-4)
        p_new = 1 / (1 + np.exp(-(np.log(p0 / (1 - p0)) + pass_shift[m])))       # per row
        rest = pr[1:] / max(1 - p0, 1e-9)
        cum = np.cumsum(np.c_[p_new, np.outer(1 - p_new, rest)], axis=1)
        out[m] = np.minimum((cum < u[m, None]).sum(axis=1), len(pr) - 1)
    return out


def _draw_rows(rng, sampler: Sampler, keys: np.ndarray) -> np.ndarray:
    """Sample one outcome row index per row from `sampler`."""
    out = np.zeros(len(keys), dtype=int)
    if len(keys) == 0:
        return out
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    for i, key in enumerate(uniq):
        m = inv == i
        pool = sampler.pool(tuple(int(k) for k in key))
        out[m] = pool[rng.integers(0, len(pool), size=int(m.sum()))]
    return out


class _Arrays:
    """The outcome / clock tables as numpy columns, for fast gathers."""

    def __init__(self, tables: dict):
        o = tables["outcomes"]
        self.call_code = {"pass": 0, "run": 1}
        self.gain = o["gain"].to_numpy(float)
        self.rel = o["rel"].to_numpy(float)
        self.is_pen = o["is_pen"].to_numpy(int)
        self.to_int = o["to_int"].to_numpy(int)
        self.to_fum = o["to_fum"].to_numpy(int)
        self.ret_td = o["ret_td"].to_numpy(int)
        self.stop = o["clock_stop"].to_numpy(int)
        self.auto_fd = o["auto_fd"].to_numpy(int)
        self.safety = o["safety"].to_numpy(int)
        self.sack = o["sack"].to_numpy(int)
        self.comp = o["complete_pass"].to_numpy(int)
        self.to_yl = o["to_yl"].to_numpy(float)
        self.dt = tables["clock"]["dt"].to_numpy(float)
        self.punt_yl = tables["punts"]["res_yl"].to_numpy(float)


def simulate(tables: dict, n: int = 10000, seed: int | None = None,
             shift: dict | None = None, home_receives_first: float = 0.5,
             tendency: dict | None = None) -> dict:
    """Play `n` games. `shift` (stage 3) is a per-team yards-per-play shift:
    {'pass': (home, away), 'run': (home, away)} applied to non-penalty gains,
    with the defence's allowance folded in by the caller. Returns team scores
    and per-team box-score counts, one entry per simulation."""
    rng = np.random.default_rng(seed)
    A = _Arrays(tables)
    shift = shift or {"pass": (0.0, 0.0), "run": (0.0, 0.0)}
    sh = {c: np.array(v, float) for c, v in shift.items()}
    # team tendencies: a logit shift on the pass call (pass rate over expectation)
    # and a multiplier on the clock a play consumes (tempo), per side
    tendency = tendency or {}
    proe = np.array(tendency.get("proe", (0.0, 0.0)), float)
    tempo = np.array(tendency.get("tempo", (1.0, 1.0)), float)

    pos = np.zeros(n, int)                      # 0 home has the ball, 1 away
    yl = np.full(n, 75.0)
    down = np.ones(n, int); ytg = np.full(n, 10.0)
    half = np.ones(n, int); secs = np.full(n, float(HALF_SECS))
    score = np.zeros((n, 2), int)
    phase = np.full(n, PH_KICK)
    first_receiver = (rng.random(n) >= home_receives_first).astype(int)   # 0 = home receives first
    kick_team = 1 - first_receiver
    stats = {c: np.zeros((n, 2), int) for c in STAT_COLS}
    drive_open = np.zeros(n, bool)

    def opp(t):
        return 1 - t

    def start_drive(m):
        stats["drives"][m, pos[m]] += 1

    def half_over(m):
        """The clock has run out for games `m`. Masks are taken BEFORE any
        state changes so a game does not fall through two branches at once."""
        h1 = m & (half == 1)
        h2 = m & (half == 2)
        ot = m & (half == 3)
        tied = h2 & (score[:, 0] == score[:, 1])
        # first half -> second half, kicked off by the team that received first
        half[h1] = 2; secs[h1] = HALF_SECS
        kick_team[h1] = first_receiver[h1]
        phase[h1] = PH_KICK
        # regulation over: decided, or into a single overtime period
        phase[h2 & ~tied] = PH_OVER
        half[tied] = 3; secs[tied] = OT_SECS
        kick_team[tied] = (rng.random(n) < 0.5).astype(int)[tied]
        phase[tied] = PH_KICK
        # overtime clock out: a tie
        phase[ot] = PH_OVER

    def score_points(team_idx, pts, m):
        score[m, team_idx[m]] += pts

    for _ in range(MAX_STEPS):
        live = phase != PH_OVER
        if not live.any():
            break

        # ---- kickoff ---------------------------------------------------
        k = live & (phase == PH_KICK)
        if k.any():
            idx = np.where(k)[0]
            rec = opp(kick_team[idx])
            td = rng.random(len(idx)) < tables["kickoff_td_rate"]
            pos[idx] = rec
            # return touchdown: the receiving team scores, then tries
            if td.any():
                i_td = idx[td]
                score[i_td, rec[td]] += 6
                stats["def_td"][i_td, rec[td]] += 1
                phase[i_td] = PH_TRY
            i_no = idx[~td]
            yl[i_no] = tables["kickoff_yl"][rng.integers(0, len(tables["kickoff_yl"]), len(i_no))]
            down[i_no] = 1; ytg[i_no] = 10
            secs[i_no] -= 5
            phase[i_no] = PH_PLAY
            start_drive(np.isin(np.arange(n), i_no))
            ended = k & (secs <= 0)
            if ended.any():
                half_over(ended)

        # ---- the try after a touchdown ---------------------------------
        t = live & (phase == PH_TRY)
        if t.any():
            idx = np.where(t)[0]
            ot_end = half[idx] == 3                   # a touchdown ends overtime
            sd = score[idx, pos[idx]] - score[idx, opp(pos[idx])]
            keys = np.c_[sd_bin(sd), time_bin(half[idx], secs[idx])]
            two = _draw_table(rng, tables["try_call"], keys) == 1
            made1 = (~two) & (rng.random(len(idx)) < tables["xp_rate"])
            made2 = two & (rng.random(len(idx)) < tables["two_pt_rate"])
            score[idx[made1], pos[idx[made1]]] += 1
            score[idx[made2], pos[idx[made2]]] += 2
            kick_team[idx] = pos[idx]                 # the scoring team kicks off
            phase[idx] = np.where(ot_end, PH_OVER, PH_KICK)
            # the half can end on a touchdown with no time left
            ended = t & (secs <= 0) & (phase == PH_KICK)
            if ended.any():
                half_over(ended)

        # ---- a scrimmage play ------------------------------------------
        p = live & (phase == PH_PLAY)
        if not p.any():
            continue
        idx = np.where(p)[0]
        me, them = pos[idx], opp(pos[idx])
        sd = score[idx, me] - score[idx, them]
        tb = time_bin(half[idx], secs[idx])
        yb, gb, sb = yl_bin(yl[idx]), ytg_bin(ytg[idx]), sd_bin(sd)

        call = np.zeros(len(idx), int)                 # 0 pass 1 run 2 kneel 3 spike 4 punt 5 fg
        early = down[idx] <= 3
        if early.any():
            keys = np.c_[down[idx][early], gb[early], yb[early], tb[early], sb[early]]
            ce = _draw_table_shift(rng, tables["call_early"], keys, proe[me[early]])
            ce = np.where((ce == 5) & (yl[idx][early] + 17 > 66), 0, ce)
            call[early] = ce
        fourth = ~early
        if fourth.any():
            keys = np.c_[gb[fourth], yb[fourth], tb[fourth], sb[fourth]]
            c4 = _draw_table_shift(rng, tables["call_fourth"], keys, proe[me[fourth]])   # 0 pass 1 run 2 punt 3 fg
            c4 = np.where(c4 == 2, 4, np.where(c4 == 3, 5, c4))
            # no field goals from beyond 65 yards
            c4 = np.where((c4 == 5) & (yl[idx][fourth] + 17 > 66), 4, c4)
            call[fourth] = c4
        stats["plays"][idx, me] += 1

        new_yl = yl[idx].copy(); new_down = down[idx].copy(); new_ytg = ytg[idx].copy()
        flip = np.zeros(len(idx), bool); flip_yl = np.zeros(len(idx))
        td = np.zeros(len(idx), bool); dtd = np.zeros(len(idx), bool); saf = np.zeros(len(idx), bool)
        fg_made = np.zeros(len(idx), bool); stop = np.zeros(len(idx), int)
        dt = np.zeros(len(idx))

        # pass / run --------------------------------------------------
        pr = call <= 1
        if pr.any():
            j = np.where(pr)[0]
            cname = call[j]                                  # 0 pass, 1 run (CALL_CODE)
            keys = np.c_[cname, np.minimum(down[idx][j], 4), gb[j], yb[j]].astype(int)
            rows = _draw_rows(rng, tables["outcome_sampler"], keys)
            gain = A.gain[rows].copy()
            pen = A.is_pen[rows] == 1
            # downs 2-4: yards relative to the sticks, plus this state's distance
            # (first down: the distance is almost always 10, so absolute yards)
            rel_ok = (down[idx][j] >= 2) & ~pen & (A.sack[rows] == 0)
            gain = np.where(rel_ok, ytg[idx][j] + A.rel[rows], gain)
            # matchup shift on real plays only. Gains are whole yards, so a
            # fractional shift is applied as its floor plus one extra yard with
            # probability equal to the fraction (rounding it would erase it).
            for cn, code in (("pass", 0), ("run", 1)):
                mm = (call[j] == code) & ~pen
                sv = sh[cn][me[j][mm]]
                base_s = np.floor(sv)
                gain[mm] += base_s + (rng.random(int(mm.sum())) < (sv - base_s))
            gain = np.round(gain)
            is_pass = call[j] == 0
            is_sack = (A.sack[rows] == 1) & ~pen
            att = is_pass & ~pen & ~is_sack
            stats["pass_att"][idx[j][att], me[j][att]] += 1
            stats["rush_att"][idx[j][~is_pass & ~pen], me[j][~is_pass & ~pen]] += 1
            stats["comp"][idx[j], me[j]] += (A.comp[rows] == 1) & ~pen
            stats["sacks"][idx[j], me[j]] += is_sack
            stats["sack_yds"][idx[j], me[j]] += np.where(is_sack, -gain, 0).astype(int)
            stats["penalties"][idx[j], me[j]] += pen
            stats["pass_yds"][idx[j], me[j]] += np.where(att, gain, 0).astype(int)
            stats["rush_yds"][idx[j], me[j]] += np.where(~is_pass & ~pen, gain, 0).astype(int)
            stop[j] = A.stop[rows]
            to = ((A.to_int[rows] == 1) | (A.to_fum[rows] == 1)) & ~pen
            stats["ints"][idx[j], me[j]] += (A.to_int[rows] == 1) & ~pen
            stats["fum_lost"][idx[j], me[j]] += (A.to_fum[rows] == 1) & ~pen
            rtd = to & (A.ret_td[rows] == 1)
            sfy = (A.safety[rows] == 1) & ~pen & ~to
            # turnover: the ball goes the other way at the return's end
            flip[j[to & ~rtd]] = True
            tyl = A.to_yl[rows]
            fallback = np.clip(100 - (yl[idx][j] - gain), 1, 99)
            flip_yl[j[to & ~rtd]] = np.where(np.isnan(tyl), fallback, tyl)[to & ~rtd]
            dtd[j[rtd]] = True
            saf[j[sfy]] = True
            ok = ~to & ~sfy
            # penalties: yards, then the down repeats unless it was an automatic first
            pj = ok & pen
            new_yl[j[pj]] = np.clip(yl[idx][j][pj] - gain[pj], 1, 99)
            afd = pj & (A.auto_fd[rows] == 1)
            new_down[j[afd]] = 1; new_ytg[j[afd]] = np.minimum(10, new_yl[j[afd]])
            keep = pj & ~afd
            new_ytg[j[keep]] = np.maximum(ytg[idx][j][keep] - gain[keep], 1)
            fd_pen = keep & (ytg[idx][j] - gain <= 0)
            new_down[j[fd_pen]] = 1; new_ytg[j[fd_pen]] = np.minimum(10, new_yl[j[fd_pen]])
            # real plays
            rj = ok & ~pen
            ny = yl[idx][j] - gain
            scored = rj & (ny <= 0)
            td[j[scored]] = True
            stats["pass_td"][idx[j][scored & is_pass], me[j][scored & is_pass]] += 1
            stats["rush_td"][idx[j][scored & ~is_pass], me[j][scored & ~is_pass]] += 1
            own_saf = rj & ~scored & (ny >= 100)
            saf[j[own_saf]] = True
            adv = rj & ~scored & ~own_saf
            new_yl[j[adv]] = ny[adv]
            first = adv & (gain >= ytg[idx][j])
            stats["first_downs"][idx[j][first], me[j][first]] += 1
            new_down[j[first]] = 1; new_ytg[j[first]] = np.minimum(10, ny[first])
            nofirst = adv & ~first
            new_down[j[nofirst]] = down[idx][j][nofirst] + 1
            new_ytg[j[nofirst]] = ytg[idx][j][nofirst] - gain[nofirst]
            # turnover on downs
            tod = nofirst & (new_down[j] > 4)
            flip[j[tod]] = True; flip_yl[j[tod]] = np.clip(100 - ny[tod], 1, 99)
            # clock
            lead = lead_class(sd[j])
            ck = np.c_[cname, stop[j], tb[j], lead].astype(int)
            dt[j] = A.dt[_draw_rows(rng, tables["clock_sampler"], ck)] * tempo[me[j]]

        # kneel / spike ------------------------------------------------
        for code, cname, gain_v, stopv in ((2, 2, -1.0, 0), (3, 3, 0.0, 1)):
            kk = call == code
            if kk.any():
                j = np.where(kk)[0]
                new_yl[j] = np.clip(yl[idx][j] - gain_v, 1, 99)
                new_down[j] = down[idx][j] + 1; new_ytg[j] = ytg[idx][j] - gain_v
                tod = new_down[j] > 4
                flip[j[tod]] = True; flip_yl[j[tod]] = np.clip(100 - new_yl[j[tod]], 1, 99)
                ck = np.c_[np.full(len(j), cname), np.full(len(j), stopv), tb[j], lead_class(sd[j])].astype(int)
                dt[j] = A.dt[_draw_rows(rng, tables["clock_sampler"], ck)]

        # punt -----------------------------------------------------------
        pu = call == 4
        if pu.any():
            j = np.where(pu)[0]
            rows = _draw_rows(rng, tables["punt_sampler"], yb[j].reshape(-1, 1))
            flip[j] = True; flip_yl[j] = np.clip(A.punt_yl[rows], 1, 99)
            stats["punts"][idx[j], me[j]] += 1
            dt[j] = 7.0

        # field goal -----------------------------------------------------
        fg = call == 5
        if fg.any():
            j = np.where(fg)[0]
            dist = yl[idx][j] + 17
            made = rng.random(len(j)) < fg_make_prob(tables["fg_coef"], dist)
            stats["fg_att"][idx[j], me[j]] += 1
            stats["fg_made"][idx[j][made], me[j][made]] += 1
            fg_made[j[made]] = True
            miss = j[~made]
            flip[miss] = True
            spot = yl[idx][miss] + 7                       # opponent takes over at the spot
            flip_yl[miss] = np.where(spot >= 80, 80, np.clip(100 - spot, 1, 99))
            dt[j] = 5.0

        # ---- apply -----------------------------------------------------
        secs[idx] -= dt
        # scores
        i_td = idx[td]
        score[i_td, me[td]] += 6
        phase[i_td] = PH_TRY
        i_dtd = idx[dtd]                                   # defensive return touchdown
        score[i_dtd, them[dtd]] += 6
        stats["def_td"][i_dtd, them[dtd]] += 1
        pos[i_dtd] = them[dtd]; phase[i_dtd] = PH_TRY
        i_saf = idx[saf]
        score[i_saf, them[saf]] += 2
        kick_team[i_saf] = me[saf]                          # the team that conceded kicks
        phase[i_saf] = PH_KICK
        i_fg = idx[fg_made]
        score[i_fg, me[fg_made]] += 3
        kick_team[i_fg] = me[fg_made]; phase[i_fg] = PH_KICK
        # possession changes
        i_fl = idx[flip]
        pos[i_fl] = them[flip]; yl[i_fl] = flip_yl[flip]; down[i_fl] = 1; ytg[i_fl] = np.minimum(10, flip_yl[flip])
        start_drive(np.isin(np.arange(n), i_fl))
        # ordinary continuation
        cont = ~(td | dtd | saf | fg_made | flip)
        i_c = idx[cont]
        yl[i_c] = new_yl[cont]; down[i_c] = new_down[cont]; ytg[i_c] = np.maximum(new_ytg[cont], 1)
        # overtime ends on any score
        ot_scored = (half[idx] == 3) & (td | dtd | saf | fg_made)
        phase[idx[ot_scored & ~(td | dtd)]] = PH_OVER          # FG / safety end it now; a TD ends after the try
        # clock ran out during a play that did not score
        ended = p & (secs <= 0) & (phase == PH_PLAY)
        if ended.any():
            half_over(ended)

    home, away = score[:, 0], score[:, 1]
    return dict(points_home=home, points_away=away, margin=home - away, total=home + away,
                stats=stats, steps=_, finished=int((phase == PH_OVER).sum()))


def league_check(sim: dict, plays: pd.DataFrame) -> pd.DataFrame:
    """Stage-2 gate: the simulated league against the real one (2016-25)."""
    m, t = sim["margin"].astype(float), sim["total"].astype(float)
    g = plays.drop_duplicates("game_id")
    real_m = None
    # real finals from the play feed: last score_differential per game is hard to
    # read here; use the schedule instead
    sched = D.load_schedule(tuple(sorted(plays["season"].unique())))
    rm = (sched["home_score"] - sched["away_score"]).dropna().to_numpy(float)
    rt = (sched["home_score"] + sched["away_score"]).dropna().to_numpy(float)
    st = sim["stats"]
    per_team = lambda c: st[c].sum(axis=1).mean() / 2
    real_plays = plays[plays["play_type"].isin(["pass", "run"])].groupby("game_id").size().mean() / 2
    pp = plays[plays["play_type"] == "pass"]
    real_pass = (len(pp) - pp["sack"].sum()) / plays["game_id"].nunique() / 2      # attempts, sacks excluded
    rows = [
        ("points per team", (sim["points_home"].mean() + sim["points_away"].mean()) / 2, (rm.size and (rt.mean() / 2))),
        ("margin sd", m.std(), rm.std()),
        ("total sd", t.std(), rt.std()),
        ("home margin mean", m.mean(), rm.mean()),
        ("P(|margin| = 3)", (np.abs(m) == 3).mean(), (np.abs(rm) == 3).mean()),
        ("P(|margin| = 7)", (np.abs(m) == 7).mean(), (np.abs(rm) == 7).mean()),
        ("P(|margin| = 6)", (np.abs(m) == 6).mean(), (np.abs(rm) == 6).mean()),
        ("P(|margin| = 10)", (np.abs(m) == 10).mean(), (np.abs(rm) == 10).mean()),
        ("P(tie)", (m == 0).mean(), (rm == 0).mean()),
        ("corr(home, away pts)", np.corrcoef(sim["points_home"], sim["points_away"])[0, 1],
         np.corrcoef(sched["home_score"].dropna(), sched["away_score"].dropna())[0, 1]),
        ("offensive plays / team", per_team("pass_att") + per_team("sacks") + per_team("rush_att"), real_plays),
        ("pass attempts / team", per_team("pass_att"), real_pass),
        ("gross pass yards / team", per_team("pass_yds"),
         float(pp[pp["sack"] == 0]["yards_gained"].sum()) / plays["game_id"].nunique() / 2),
        ("rush yards / team", per_team("rush_yds"),
         float(plays[plays["play_type"] == "run"]["yards_gained"].sum()) / plays["game_id"].nunique() / 2),
        ("drives / team", per_team("drives"), 11.4),
        ("punts / team", per_team("punts"), plays[plays["play_type"] == "punt"].groupby("game_id").size().mean() / 2),
        ("FG attempts / team", per_team("fg_att"), plays[plays["play_type"] == "field_goal"].groupby("game_id").size().mean() / 2),
        ("INTs / team", per_team("ints"), plays["interception"].sum() / plays["game_id"].nunique() / 2),
        ("sacks / team", per_team("sacks"), plays["sack"].sum() / plays["game_id"].nunique() / 2),
    ]
    for k in (1, 2, 4, 5, 8, 14):
        rows.append((f"P(|margin| = {k})", (np.abs(m) == k).mean(), (np.abs(rm) == k).mean()))
    return pd.DataFrame(rows, columns=["metric", "engine", "real"])


# ---------------------------------------------------------------------------
# Stage 3: team strength — steer the engine to the ratings' expected game
# ---------------------------------------------------------------------------
# The ratings layer (`teams.expected_points`: blended EPA / drive ratings, home
# field, availability, wind) already carries the validated view of what a
# matchup is worth on average. The engine takes that as its target: a yards-
# per-play shift for each offense is solved so the simulated mean margin and
# total land on the ratings' numbers. The shift is the ONLY thing borrowed;
# the spread of outcomes, the key numbers, the score correlation and every
# play count are the engine's own. Sensitivities (points of margin / total
# per yard of shift) are measured on the neutral engine once per table build.

SENS_SHIFT = 0.6            # yards per play used to measure the sensitivities
SENS_N = 6000


def sensitivity(tables: dict, seed: int = 11) -> dict:
    """d(margin)/d(home shift), d(total)/d(shift) and the neutral means."""
    base = simulate(tables, SENS_N, seed=seed)
    up = simulate(tables, SENS_N, seed=seed, shift={"pass": (SENS_SHIFT, 0.0), "run": (SENS_SHIFT, 0.0)})
    m0, t0 = float(base["margin"].mean()), float(base["total"].mean())
    dm = (float(up["margin"].mean()) - m0) / SENS_SHIFT          # margin per yard, one side
    dt = (float(up["total"].mean()) - t0) / SENS_SHIFT           # total per yard, one side
    return dict(m0=m0, t0=t0, dm=dm, dt=dt)


def shifts_for(sens: dict, margin: float, total: float) -> dict:
    """Per-side yards shifts that move the neutral engine to (margin, total).
    Home shift s_h and away shift s_a: margin = m0 + dm·(s_h − s_a),
    total = t0 + dt·(s_h + s_a)."""
    diff = (float(margin) - sens["m0"]) / max(sens["dm"], 1e-6)
    summ = (float(total) - sens["t0"]) / max(sens["dt"], 1e-6)
    s_h, s_a = 0.5 * (summ + diff), 0.5 * (summ - diff)
    s_h, s_a = float(np.clip(s_h, -2.5, 2.5)), float(np.clip(s_a, -2.5, 2.5))
    return {"pass": (s_h, s_a), "run": (s_h, s_a)}


def simulate_matchup(tables: dict, sens: dict, ratings: dict, home: str, away: str,
                     n: int = 10000, seed: int | None = None, avail=None, wind=None,
                     roof=None, neutral_site: bool = False, tendency: dict | None = None) -> dict:
    """The engine steered to the ratings' expected margin and total for one game."""
    from . import teams as T
    e = T.expected_points(ratings, home, away, home=None if neutral_site else "a",
                          avail=avail, wind=wind, roof=roof)
    sh = shifts_for(sens, e["margin"], e["total"])
    sim = simulate(tables, n, seed=seed, shift=sh, tendency=tendency)
    sim.update(target_margin=float(e["margin"]), target_total=float(e["total"]),
               shift=sh, home=home, away=away,
               win_home=float((sim["margin"] > 0).mean() + 0.5 * (sim["margin"] == 0).mean()))
    return sim


def cover_prob(sim: dict, spread_home: float) -> tuple[float, float]:
    """P(home covers), P(push) against a home spread (positive = home favoured)."""
    m = sim["margin"]
    return float((m > spread_home).mean()), float((m == spread_home).mean())


def total_prob(sim: dict, line: float) -> tuple[float, float]:
    t = sim["total"]
    return float((t > line).mean()), float((t == line).mean())


# ---------------------------------------------------------------------------
# Stage-3 gate: the engine's shape against a normal curve on the same means
# ---------------------------------------------------------------------------

def shape_backtest(tables: dict, sens: dict, bt: pd.DataFrame, n: int = 3000,
                   seed: int = 3, progress=None) -> pd.DataFrame:
    """For every game in a team backtest (out-of-sample expected margin and
    total already computed), steer the engine to those means and record its
    win probability, cover probability against the closing spread and
    over probability against the closing total — beside the normal-curve
    equivalents the harness uses. Same means, different shapes."""
    from . import backtest as B
    from math import erf, sqrt
    rows = []
    for i, g in bt.reset_index(drop=True).iterrows():
        if progress and i % 20 == 0:
            progress(i / len(bt), f"{g['away']} @ {g['home']}")
        sh = shifts_for(sens, g["pred_margin"], g["pred_total"])
        sim = simulate(tables, n, seed=seed + i, shift=sh)
        m, t = sim["margin"], sim["total"]
        row = dict(game_id=g["game_id"], season=g["season"], week=g["week"],
                   eng_margin=float(m.mean()), eng_total=float(t.mean()),
                   eng_win=float((m > 0).mean() + 0.5 * (m == 0).mean()),
                   norm_win=0.5 * (1 + erf(g["pred_margin"] / B.MARGIN_SD / sqrt(2))),
                   actual_margin=g["actual_margin"], actual_total=g["actual_total"],
                   line_margin=g.get("line_margin", np.nan), line_total=g.get("line_total", np.nan))
        if pd.notna(row["line_margin"]):
            pc, pp = cover_prob(sim, float(row["line_margin"]))
            row["eng_cover"] = pc / max(1 - pp, 1e-9)
            row["norm_cover"] = 0.5 * (1 + erf((g["pred_margin"] - row["line_margin"]) / B.MARGIN_SD / sqrt(2)))
        if pd.notna(row["line_total"]):
            po, pp = total_prob(sim, float(row["line_total"]))
            row["eng_over"] = po / max(1 - pp, 1e-9)
            row["norm_over"] = 0.5 * (1 + erf((g["pred_total"] - row["line_total"]) / 13.4 / sqrt(2)))
        rows.append(row)
    return pd.DataFrame(rows)


def shape_metrics(sb: pd.DataFrame) -> pd.DataFrame:
    def ll(p, y):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); y = np.asarray(y, float)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    def brier(p, y):
        return float(((np.asarray(p, float) - np.asarray(y, float)) ** 2).mean())
    y_win = (sb["actual_margin"] > 0).astype(float)
    out = [("winner log-loss", ll(sb["eng_win"], y_win), ll(sb["norm_win"], y_win)),
           ("winner Brier", brier(sb["eng_win"], y_win), brier(sb["norm_win"], y_win))]
    c = sb.dropna(subset=["eng_cover"]); c = c[c["actual_margin"] != c["line_margin"]]
    y_c = (c["actual_margin"] > c["line_margin"]).astype(float)
    out += [("cover log-loss", ll(c["eng_cover"], y_c), ll(c["norm_cover"], y_c)),
            ("cover Brier", brier(c["eng_cover"], y_c), brier(c["norm_cover"], y_c))]
    o = sb.dropna(subset=["eng_over"]); o = o[o["actual_total"] != o["line_total"]]
    y_o = (o["actual_total"] > o["line_total"]).astype(float)
    out += [("over log-loss", ll(o["eng_over"], y_o), ll(o["norm_over"], y_o)),
            ("over Brier", brier(o["eng_over"], y_o), brier(o["norm_over"], y_o)),
            ("margin RMSE (means agree by construction)", float(np.sqrt(((sb["eng_margin"] - sb["actual_margin"]) ** 2).mean())), np.nan)]
    return pd.DataFrame(out, columns=["metric", "engine", "normal"])


# ---------------------------------------------------------------------------
# Stage 4: the depth chart on top of the simulated game
# ---------------------------------------------------------------------------
# Every simulation already carries each team's attempts, completions, gross
# passing yards, carries, rushing yards, touchdowns, sacks and interceptions —
# from the clock and the game state, not from a normal draw. Allocation to the
# players follows the drive engine exactly (Dirichlet shares, the receiving and
# rushing yardage mechanics, multinomial touchdown splits) and then the players'
# yards are rescaled so that, in EVERY simulation, the receivers sum to the
# team's gross passing yards and the rushers to its rushing yards. That is what
# ties a receiver's game to his quarterback's and to the game's total.

TENDENCY_SEASONS_BACK = 2          # seasons of plays behind a team's tendencies
TENDENCY_DECAY = 0.7               # per season


def team_tendencies(tables: dict, plays: pd.DataFrame, latest: int | None = None) -> pd.DataFrame:
    """Per team: pass rate over expectation (as a logit shift) and tempo.

    PROE: the team's pass calls on downs 1-3 minus what the league table
    expects in the same states, over recent seasons; converted to a logit shift
    at the league's typical pass rate. Tempo: seconds per running-clock play
    outside the two-minute drill, relative to league.
    """
    d = plays.copy()
    latest = int(latest or d["season"].max())
    d = d[d["season"] >= latest - TENDENCY_SEASONS_BACK]
    d["w"] = TENDENCY_DECAY ** (latest - d["season"])
    d["ytg_b"] = ytg_bin(d["ydstogo"].fillna(10)); d["yl_b"] = yl_bin(d["yardline_100"].fillna(50))
    d["sd_b"] = sd_bin(d["score_differential"].fillna(0))
    d["t_b"] = time_bin(d["game_half"], d["half_seconds_remaining"].fillna(900))
    d["down_i"] = d["down"].fillna(0).astype(int)
    e = d[d["down_i"].between(1, 3) & d["play_type"].isin(["pass", "run"])].copy()
    keys = e[["down_i", "ytg_b", "yl_b", "t_b", "sd_b"]].to_numpy(int)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    exp_pass = np.array([tables["call_early"].probs(tuple(int(x) for x in k))[0] for k in uniq])[inv.ravel()]
    e["exp_pass"] = exp_pass; e["is_pass"] = (e["play_type"] == "pass").astype(float)
    g = e.groupby("posteam").apply(lambda q: pd.Series(dict(
        n=len(q), proe=float(np.average(q["is_pass"] - q["exp_pass"], weights=q["w"])),
        pass_rate=float(np.average(q["is_pass"], weights=q["w"])))))
    pbar = float(np.average(e["is_pass"], weights=e["w"]))
    g["proe_logit"] = g["proe"] / (pbar * (1 - pbar))
    # tempo
    ck = tables["clock"]
    c = d[d["play_type"].isin(["pass", "run"]) & (d["incomplete_pass"] == 0) & (d["out_of_bounds"] == 0)
          & d["t_b"].isin([0, 2, 3])].copy()
    c["dt"] = c["half_seconds_remaining"] - c["next_half_secs"]
    c = c[c["dt"].between(5, 60)]
    lg_dt = float(np.average(c["dt"], weights=c["w"]))
    tempo = c.groupby("posteam").apply(lambda q: float(np.average(q["dt"], weights=q["w"])) / lg_dt)
    g["tempo"] = tempo.reindex(g.index).fillna(1.0).clip(0.9, 1.1)
    return g


def tendency_for(tend: pd.DataFrame, home: str, away: str) -> dict:
    def get(team, col, default):
        return float(tend.loc[team, col]) if team in tend.index else default
    return dict(proe=(get(home, "proe_logit", 0.0), get(away, "proe_logit", 0.0)),
                tempo=(get(home, "tempo", 1.0), get(away, "tempo", 1.0)))


def allocate_players(rng, ctx: dict, roster: pd.DataFrame, team: str, opponent: str,
                     st: dict, side: int) -> dict:
    """One team's player lines from the engine's per-simulation team totals.
    Returns the same dict shape as `game._side_box`."""
    from . import game as G, rushing as R
    n = len(st["pass_att"]); k = len(roster)
    attempts = st["pass_att"][:, side].astype(int)
    sacks = st["sacks"][:, side].astype(int)
    ints = st["ints"][:, side].astype(int)
    carries_team = st["rush_att"][:, side].astype(int)
    pass_yds_team = st["pass_yds"][:, side].astype(float)
    rush_yds_team = st["rush_yds"][:, side].astype(float)
    n_pass_td = st["pass_td"][:, side].astype(int)
    n_rush_td = st["rush_td"][:, side].astype(int)
    dropbacks = attempts + sacks

    tgt_w = rng.dirichlet(np.clip(roster["target_share"].values, 1e-4, None)
                          * G.TARGET_CONCENTRATION, size=n)
    car_w = rng.dirichlet(np.clip(roster["carry_share"].values, 1e-4, None)
                          * G.CARRY_CONCENTRATION, size=n)
    target_rate = float(ctx.get("target_rate", G.TARGET_PER_ATTEMPT))
    targeted = rng.binomial(attempts, float(np.clip(target_rate, 0.8, 1.0)))
    targets = G._split_counts(targeted, tgt_w)
    player_car = G._split_counts(carries_team, car_w)

    receptions = np.zeros((n, k), dtype=int)
    rec_yards = np.zeros((n, k)); rush_yards = np.zeros((n, k))
    rush_def = ctx.get("rush_def") or {}
    for j, pl in roster.reset_index(drop=True).iterrows():
        t = targets[:, j]
        rec = rng.binomial(t, pl["catch_rate"])
        receptions[:, j] = rec
        ypc = pl["ypt"] / max(pl["catch_rate"], 1e-6)
        kk = 1.0 / (G.YPC_CV ** 2)
        rec_yards[:, j] = np.where(rec > 0, rng.gamma(np.clip(rec * kk, 1e-9, None), 1.0) * (ypc / kk), 0.0)
        c = player_car[:, j]
        if pl["rush_priors"] is not None and c.max() > 0:
            rush_yards[:, j] = R.yards_from_carries(rng, pl["rush_priors"], c, rush_def.get(opponent))["yards"]
        elif c.max() > 0:
            rush_yards[:, j] = c * 4.2 * rng.lognormal(-0.08, 0.40, n)

    # rescale to the game's totals: the receivers ARE the passing yards
    def rescale(player, team_total):
        s_ = player.sum(axis=1)
        f = np.where(s_ > 0, team_total / np.where(s_ > 0, s_, 1.0), 0.0)
        out = player * f[:, None]
        # a team with yards but no receiver drew a catch: give them to the top share
        orphan = (s_ <= 0) & (team_total > 0)
        if orphan.any():
            top = np.argmax(tgt_w[orphan], axis=1) if player is rec_yards else np.argmax(car_w[orphan], axis=1)
            out[np.where(orphan)[0], top] = team_total[orphan]
        return np.round(out, 1)
    rec_yards = rescale(rec_yards, pass_yds_team)
    rush_yards = rescale(rush_yards, rush_yds_team)

    rec_tds = G._multinomial_alloc(rng, n_pass_td, np.tile(roster["rec_td_weight"].values, (n, 1)))
    rush_tds = G._multinomial_alloc(rng, n_rush_td, np.tile(roster["rush_td_weight"].values, (n, 1)))
    plays = dropbacks + carries_team
    return dict(
        roster=roster, targets=targets, receptions=receptions, rec_yards=rec_yards,
        carries=player_car, rush_yards=rush_yards, rec_tds=rec_tds, rush_tds=rush_tds,
        dropbacks=dropbacks, attempts=attempts, sacks=sacks, ints=ints,
        team_pass_yards=pass_yds_team, team_rush_yards=rush_yds_team,
        pass_frac=dropbacks / np.maximum(plays, 1), qb_priors=None,
        n_pass_td=n_pass_td, n_rush_td=n_rush_td, engine="play",
    )


def simulate_game_players(tables: dict, sens: dict, ctx: dict,
                          roster_a: pd.DataFrame, roster_b: pd.DataFrame,
                          team_a: str, team_b: str, n: int = 10000, seed: int | None = None,
                          avail=None, wind=None, roof=None, home: str | None = "a") -> dict:
    """The play engine with both depth charts on top — the same output shape as
    `game.simulate_game`, so fantasy, props and the pick'em can consume either."""
    rng = np.random.default_rng(seed)
    tend = ctx.get("play_tendencies")
    tendency = tendency_for(tend, team_a, team_b) if tend is not None else None
    sim = simulate_matchup(tables, sens, ctx["ratings"], team_a, team_b, n=n, seed=seed,
                           avail=avail, wind=wind, roof=roof, neutral_site=(home is None),
                           tendency=tendency)
    box_a = allocate_players(rng, ctx, roster_a, team_a, team_b, sim["stats"], 0)
    box_b = allocate_players(rng, ctx, roster_b, team_b, team_a, sim["stats"], 1)
    return dict(
        team_a=team_a, team_b=team_b, points_a=sim["points_home"], points_b=sim["points_away"],
        drives_a=sim["stats"]["drives"][:, 0], drives_b=sim["stats"]["drives"][:, 1],
        box_a=box_a, box_b=box_b, pace=dict(mean=float(sim["stats"]["drives"].mean())),
        n_sims=n, home=home, engine="play", team_stats=sim["stats"],
        target_margin=sim["target_margin"], target_total=sim["target_total"],
    )


_ENGINE: dict = {}


def attach(ctx: dict) -> dict:
    """Build (once per process) and attach the play engine's tables to a context."""
    if "tables" not in _ENGINE:
        plays = load_plays()
        _ENGINE["tables"] = build_tables(plays)
        _ENGINE["sens"] = sensitivity(_ENGINE["tables"])
        # tendencies use the most recent seasons INCLUDING the one being played
        try:
            latest = int(ctx.get("depth_seasons", (plays["season"].max(),))[-1])
            recent = load_plays(tuple(range(latest - TENDENCY_SEASONS_BACK, latest + 1)))
        except Exception:
            recent, latest = plays, None
        _ENGINE["tendencies"] = team_tendencies(_ENGINE["tables"], recent, latest)
    ctx["play_engine"] = (_ENGINE["tables"], _ENGINE["sens"])
    ctx["play_tendencies"] = _ENGINE["tendencies"]
    return ctx


# ---------------------------------------------------------------------------
# Stage-1 report: do the tables match what we know about the league?
# ---------------------------------------------------------------------------

def report(tables: dict, plays: pd.DataFrame) -> str:
    lines = []
    ce = tables["call_early"]
    for down, ytg in ((1, 4), (2, 2), (2, 3), (3, 0), (3, 2), (3, 5)):
        p = ce.probs((down, ytg, 5, 0, sd_bin([0])[0]))
        lines.append(f"pass rate, down {down} ytg-bin {ytg} at midfield, tied, H1: {p[0]:.0%}")
    cf = tables["call_fourth"]
    for ytg, yl, lab in ((0, 5, "4th & 1 at midfield"), (2, 3, "4th & 3-5 at opp 35"), (5, 6, "4th & 11+ at own 40"),
                         (0, 0, "4th & 1 inside the 10"), (2, 2, "4th & 3-5 at opp 25 (FG range)")):
        p = cf.probs((ytg, yl, 2, sd_bin([0])[0]))
        lines.append(f"4th down, {lab}, H2 early, tied: go {p[0]+p[1]:.0%} punt {p[2]:.0%} FG {p[3]:.0%}")
    o = tables["outcomes"]
    for call in ("pass", "run"):
        q = o[o["call"] == CALL_CODE[call]]
        lines.append(f"{call}: n={len(q):,} mean gain {q['gain'].mean():.2f} | sack {q['sack'].mean():.1%} "
                     f"| INT {q['to_int'].mean():.1%} | fumble lost {q['to_fum'].mean():.1%} | penalty {q['is_pen'].mean():.1%}")
    ck = tables["clock"]
    for call, stop, tb in (("run", 0, 0), ("pass", 0, 0), ("pass", 1, 0), ("run", 0, 5), ("pass", 1, 5), ("kneel", 0, 5)):
        q = ck[(ck["call"] == CALL_CODE[call]) & (ck["stop"] == stop) & (ck["t_b"] == tb)]
        lines.append(f"clock: {call} {'stop' if stop else 'runs'} t_b={tb}: median {q['dt'].median():.0f}s (n={len(q)})")
    for dist in (30, 40, 50, 55, 60):
        lines.append(f"FG from {dist}: {fg_make_prob(tables['fg_coef'], dist):.0%}")
    ko = tables["kickoff_yl"]
    lines.append(f"kickoff start: median own {100 - np.median(ko):.0f} (touchback to the 35: {np.isclose(ko, 65).mean():.0%}, n={len(ko):,}) | return TD {tables['kickoff_td_rate']:.2%}")
    lines.append(f"XP {tables['xp_rate']:.1%} | 2-pt {tables['two_pt_rate']:.1%} | go for two when down 8 after the TD (sd -2), H2 late: "
                 f"{tables['try_call'].probs((sd_bin([-2])[0], 4))[1]:.0%}; when tied (sd 0) H1: {tables['try_call'].probs((sd_bin([0])[0], 0))[1]:.0%}")
    g = plays[plays["play_type"].isin(["pass", "run"])].groupby("game_id").size()
    lines.append(f"offensive plays per game (both teams): {g.mean():.1f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import time
    t = time.time()
    p = load_plays()
    print(f"{len(p):,} plays, {p['game_id'].nunique():,} games, loaded in {time.time()-t:.0f}s")
    t = time.time()
    tb = build_tables(p)
    print(f"tables built in {time.time()-t:.0f}s | outcome cells {len(tb['outcome_sampler'].cells):,} "
          f"| decision cells {len(tb['call_early'].cells):,} + {len(tb['call_fourth'].cells):,}")
    print(report(tb, p))
    t = time.time()
    sim = simulate(tb, n=4000, seed=1)
    print(f"\nsimulated 4,000 games in {time.time()-t:.0f}s ({sim['steps']} steps, {sim['finished']} finished)")
    pd.set_option("display.width", 160)
    print(league_check(sim, p).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    t = time.time()
    sens = sensitivity(tb)
    print(f"\nsensitivity ({time.time()-t:.0f}s): neutral margin {sens['m0']:+.2f} total {sens['t0']:.2f} | "
          f"margin {sens['dm']:.2f} pts and total {sens['dt']:.2f} pts per yard-per-play of one side")
    sh = shifts_for(sens, 7.0, 47.0)
    chk = simulate(tb, 6000, seed=5, shift=sh)
    print(f"steer to margin +7 / total 47: engine margin {chk['margin'].mean():+.2f} total {chk['total'].mean():.2f} "
          f"| shifts {sh['pass'][0]:+.2f} / {sh['pass'][1]:+.2f} yds | P(home by exactly 3) {(chk['margin']==3).mean():.1%}, by 7 {(chk['margin']==7).mean():.1%}")
