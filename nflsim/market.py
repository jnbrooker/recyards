"""
nflsim/market.py — probabilities from the market and history, for the pick'em.

The replay on page 9 showed the engine's own view of a game's margin adds
nothing to the closing line (slope +0.01, correlation +0.004 on 2025). A
confidence pool is not beaten by out-predicting the market; it is beaten by
taking the market's probabilities as the truth and finding where the POOL's
prices differ from them. Two such places, both measurable on 2010-25 closing
lines (4,191 games):

  * **Key numbers.** 14.5% of games land exactly on 3, 8.7% on 7. A side at
    +3.5 when the market says +3 covers 59%, not 50%; at +7.5 vs +7, 53%. A
    normal distribution cannot see this; the empirical distribution of margins
    given the spread can.
  * **Teasers.** Six-point legs that cross both 3 and 7 — dogs of +1.5..+2.5
    teased up, favourites of -7.5..-8.5 teased down ("Wong" legs) — hit 75.7%
    and 72.7%; every other teaser leg is 64-71%, a loser at +160 (breakeven
    72.7% per leg for three legs). The empirical distribution prices this
    without a special rule.

So every side probability here is P(margin beats the POOL's line | the MARKET
spread), read off games with the same closing spread (a +-0.5 kernel, with a
shifted-normal fallback when history is thin). Totals use the empirical
distribution of (total - closing total), which has no key-number structure to
speak of. Moneylines use the vig-free price when the pool gives one, else the
same empirical margin distribution at zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D

HISTORY_FROM = 2010          # closing lines are reliable in the feed from here
MIN_KERNEL_N = 60            # below this, fall back to the shifted normal
MARGIN_SD = 13.2             # residual sd of margin around the closing spread
TOTAL_SD = 13.5              # residual sd of total around the closing total

_SCHED_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
              "schedules/games.csv")


@D.ttl_cache(maxsize=2)
def load_history(seasons_from: int = HISTORY_FROM, seasons_to: int | None = None) -> pd.DataFrame:
    """Played regular-season games with closing spread/total and the result.
    `sp` follows the feed: positive = home favoured. `m` = home margin."""
    try:
        d = pd.read_csv(_SCHED_URL, low_memory=False)
    except Exception:
        return pd.DataFrame()
    d = d[(d["game_type"] == "REG") & d["spread_line"].notna() & d["home_score"].notna()]
    d = d[d["season"] >= int(seasons_from)]
    if seasons_to is not None:
        d = d[d["season"] <= int(seasons_to)]
    out = pd.DataFrame({
        "season": d["season"].astype(int), "sp": d["spread_line"].astype(float),
        "tot": pd.to_numeric(d["total_line"], errors="coerce"),
        "m": (d["home_score"] - d["away_score"]).astype(float),
        "t": (d["home_score"] + d["away_score"]).astype(float),
    })
    out["res_t"] = out["t"] - out["tot"]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sides: P(margin beats a line | the market spread)
# ---------------------------------------------------------------------------

def _kernel(hist: pd.DataFrame, market_sp: float):
    """Games at the same closing spread, with half-point neighbours at half
    weight. Returns (margins, weights) or None if too thin."""
    dsp = (hist["sp"] - float(market_sp)).abs()
    w = np.where(dsp < 0.01, 1.0, np.where(dsp <= 0.51, 0.5, 0.0))
    if w.sum() < MIN_KERNEL_N:
        return None
    return hist["m"].to_numpy(float), w


def cover_prob(hist: pd.DataFrame, market_sp: float, pool_sp: float, side: str) -> tuple[float, float, int]:
    """P(win), P(push), effective n for `side` ('home' or 'away') against the
    POOL's home spread, given the MARKET's home spread."""
    k = _kernel(hist, market_sp) if hist is not None and not hist.empty else None
    if k is None:
        from math import erf, sqrt
        z = (float(pool_sp) - float(market_sp)) / MARGIN_SD
        p_home = 0.5 * (1 - erf(z / sqrt(2)))          # P(m > pool_sp) under N(market_sp, sd)
        return (p_home if side == "home" else 1 - p_home), 0.0, 0
    m, w = k
    win = (m > pool_sp) if side == "home" else (m < pool_sp)
    push = np.isclose(m, pool_sp)
    W = w.sum()
    return float((w * win).sum() / W), float((w * push).sum() / W), int(round(W))


def win_prob(hist: pd.DataFrame, market_sp: float, side: str) -> tuple[float, float]:
    """P(win outright), P(tie) for a side given the market spread."""
    p, push, _ = cover_prob(hist, market_sp, 0.0, side)
    return p, push


def vig_free(odds_a, odds_b) -> tuple[float, float]:
    """Two-way market -> fair probabilities (multiplicative vig removal)."""
    from .pickem import implied_prob
    a, b = implied_prob(odds_a), implied_prob(odds_b)
    s = a + b
    return (a / s, b / s) if s > 0 else (0.5, 0.5)


# ---------------------------------------------------------------------------
# Totals: P(total beats a line | the market total)
# ---------------------------------------------------------------------------

def total_prob(hist: pd.DataFrame, market_tot: float, pool_tot: float, side: str) -> tuple[float, float]:
    """P(win), P(push) for 'over' / 'under' the POOL total given the MARKET
    total: the empirical residual (total - closing total) shifted so its
    centre sits at the market number."""
    if hist is None or hist.empty or hist["res_t"].notna().sum() < MIN_KERNEL_N:
        from math import erf, sqrt
        z = (float(pool_tot) - float(market_tot)) / TOTAL_SD
        p_over = 0.5 * (1 - erf(z / sqrt(2)))
        return (p_over if side == "over" else 1 - p_over), 0.0
    t = float(market_tot) + hist["res_t"].dropna().to_numpy(float)
    win = (t > pool_tot) if side == "over" else (t < pool_tot)
    return float(win.mean()), float(np.isclose(t, pool_tot).mean())


# ---------------------------------------------------------------------------
# Live market lines (The Odds API, same key as the props page)
# ---------------------------------------------------------------------------

def fetch_game_lines(api_key: str, regions: str = "us") -> tuple[pd.DataFrame, dict]:
    """Current spreads / totals / moneylines for upcoming NFL games, one row
    per (game, bookmaker). Returns (lines, quota). One request (~3 credits)."""
    from .props import _get, API, SPORT, TEAM_ABBR
    js, quota = _get(f"{API}/sports/{SPORT}/odds",
                     dict(apiKey=api_key, regions=regions,
                          markets="spreads,totals,h2h", oddsFormat="american"))
    rows = []
    for ev in js or []:
        home = TEAM_ABBR.get(ev.get("home_team"), ev.get("home_team"))
        away = TEAM_ABBR.get(ev.get("away_team"), ev.get("away_team"))
        for bk in ev.get("bookmakers", []):
            r = dict(home=home, away=away, commence=ev.get("commence_time"),
                     bookmaker=bk.get("key"), spread_home=np.nan, spread_price_home=np.nan,
                     spread_price_away=np.nan, total=np.nan, over_price=np.nan,
                     under_price=np.nan, ml_home=np.nan, ml_away=np.nan)
            for mk in bk.get("markets", []):
                for o in mk.get("outcomes", []):
                    nm = TEAM_ABBR.get(o.get("name"), o.get("name"))
                    if mk["key"] == "spreads":
                        if nm == home:
                            r["spread_home"] = -float(o["point"]); r["spread_price_home"] = o["price"]
                        elif nm == away:
                            r["spread_price_away"] = o["price"]
                    elif mk["key"] == "totals":
                        if o.get("name") == "Over":
                            r["total"] = float(o["point"]); r["over_price"] = o["price"]
                        else:
                            r["under_price"] = o["price"]
                    elif mk["key"] == "h2h":
                        if nm == home:
                            r["ml_home"] = o["price"]
                        elif nm == away:
                            r["ml_away"] = o["price"]
            rows.append(r)
    return pd.DataFrame(rows), quota


def consensus_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """One row per game: the median spread / total across books (the market's
    centre) and the median moneylines. Medians resist one book's stale number."""
    if lines is None or lines.empty:
        return pd.DataFrame()
    g = (lines.groupby(["home", "away"], as_index=False)
              .agg(spread_home=("spread_home", "median"), total=("total", "median"),
                   ml_home=("ml_home", "median"), ml_away=("ml_away", "median"),
                   books=("bookmaker", "nunique"), commence=("commence", "first")))
    for c in ("spread_home", "total"):
        g[c] = (g[c] * 2).round() / 2            # back onto the half-point grid
    return g
