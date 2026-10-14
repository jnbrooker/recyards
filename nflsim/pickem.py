"""
nflsim/pickem.py — a confidence pick'em card built from the game engine.

The pool: twenty slots, each given a unique confidence from 20 down to 1.

  * one pick per game — either side against the spread, or the UNDERDOG on the
    moneyline (favourites on the moneyline are not allowed);
  * a 3-team ATS parlay, a 3-team moneyline parlay and a 3-team 6-point teaser;
  * over/unders fill whatever slots remain (20 − games − 3, so more in a bye week).

The engine gives every game a full distribution of margins and totals, so each
candidate has a model probability directly — P(cover) is just the share of
simulations where the margin clears the line, no normal approximation. Parlay
legs are taken from different games, so their probabilities multiply exactly
(real games are independent, and so are the simulations).

Two scoring conventions are supported, because they lead to different cards:

  * **flat** — a winning pick earns its confidence points. Rank by P(win).
  * **odds-weighted** — a winning pick earns confidence × decimal payout. Rank
    by P(win) × payout, which is where underdogs and parlays can earn a slot.

In both cases the assignment problem has a closed form: sort the twenty picks
by expected points per confidence point and hand out 20, 19, … in that order
(the rearrangement inequality — larger multipliers go with larger values).

Lines default to the closing market from the schedule feed but are meant to be
overwritten with whatever the pool actually offers; the model never reads them
for anything but grading the picks.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from . import game as G

N_SLOTS = 20
DEFAULT_JUICE = -110          # ATS and totals price
DEFAULT_TEASER_POINTS = 6.0
DEFAULT_TEASER_ODDS = 160     # 3-team 6-point teaser, American
DEFAULT_PARLAY_ODDS = 600     # if the pool pays a fixed 3-team price
SPREAD_TO_ML_SD = 13.5        # only to synthesise a moneyline when the feed lacks one


# ---------------------------------------------------------------------------
# Odds helpers
# ---------------------------------------------------------------------------

def to_decimal(american) -> float:
    a = float(american)
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / abs(a))


def to_american(decimal: float) -> str:
    d = float(decimal)
    if d <= 1.0:
        return "—"
    return f"+{round((d - 1) * 100)}" if d >= 2.0 else f"-{round(100 / (d - 1))}"


def implied_prob(american) -> float:
    return 1.0 / to_decimal(american)


def spread_to_moneyline(spread_home: float) -> tuple[float, float]:
    """A placeholder moneyline pair from a spread, for feeds missing one."""
    from math import erf, sqrt
    p_home = 0.5 * (1 + erf((spread_home / SPREAD_TO_ML_SD) / sqrt(2)))
    p_home = float(np.clip(p_home, 0.03, 0.97))
    def am(p):
        d = 1.0 / p
        return round((d - 1) * 100) if d >= 2 else -round(100 / (d - 1))
    return am(p_home), am(1 - p_home)


# ---------------------------------------------------------------------------
# Simulating the slate
# ---------------------------------------------------------------------------

def simulate_slate(ctx: dict, games: pd.DataFrame, n_sims: int = 10000,
                   use_injuries: bool = True, seed: int = 23, engine: str = "drive", progress=None) -> dict:
    """game_id -> dict(home, away, pts_home, pts_away, label) for every game."""
    ratings = ctx["ratings"]
    out, rosters = {}, {}

    def roster(team):
        if team not in rosters:
            rosters[team] = G.roster_for(ctx, team, use_injuries=use_injuries)
        return rosters[team]

    n_games = max(len(games), 1)
    for i, g in games.reset_index(drop=True).iterrows():
        home, away = g["home_team"], g["away_team"]
        if home not in ratings["off"].index or away not in ratings["off"].index:
            continue
        sub = (lambda f, t, i=i: progress((i + f) / n_games, f"{away} @ {home}")) if progress else None
        if progress:
            progress(i / n_games, f"{away} @ {home}")
        try:
            sim = G.run_game(ctx, roster(home), roster(away), home, away, n_sims=n_sims,
                             seed=seed + i, home="a", avail=ctx.get("avail"),
                             wind=g.get("wind"), roof=g.get("roof"), engine=engine, progress=sub)
        except ValueError:
            continue
        out[g["game_id"]] = dict(
            game_id=g["game_id"], home=home, away=away,
            pts_home=sim["points_a"].astype(np.int16),
            pts_away=sim["points_b"].astype(np.int16),
            played=bool(g.get("played", False)),
            home_score=g.get("home_score"), away_score=g.get("away_score"),
        )
    return out


def default_lines(games: pd.DataFrame) -> pd.DataFrame:
    """The editable lines table, seeded from the closing market where present.

    `spread_home` follows the feed's convention: positive = home favoured by
    that many points (so CHI -3 at CAR is spread_home = -3).
    """
    rows = []
    for _, g in games.iterrows():
        sp = g.get("spread_line")
        tot = g.get("total_line")
        ml_h, ml_a = g.get("home_moneyline"), g.get("away_moneyline")
        if (pd.isna(ml_h) or pd.isna(ml_a)) and pd.notna(sp):
            ml_h, ml_a = spread_to_moneyline(float(sp))
        row = dict(
            game_id=g["game_id"], Game=f"{g['away_team']} @ {g['home_team']}",
            home=g["home_team"], away=g["away_team"],
            spread_home=float(sp) if pd.notna(sp) else 0.0,
            total=float(tot) if pd.notna(tot) else 45.0,
            ml_home=int(ml_h) if pd.notna(ml_h) else -110,
            ml_away=int(ml_a) if pd.notna(ml_a) else -110,
            played=bool(g.get("played", False)),
        )
        # the market's own numbers: what the pool's lines are judged AGAINST.
        # Seeded equal to the pool columns; a live fetch or an edit separates them.
        row.update(mkt_spread_home=row["spread_home"], mkt_total=row["total"],
                   mkt_ml_home=row["ml_home"], mkt_ml_away=row["ml_away"])
        rows.append(row)
    return pd.DataFrame(rows)


POOL_FILE = "pickem/lines.csv"
_POOL_COLS = ["season", "week", "game_id", "home", "away", "spread_home", "total",
              "ml_home", "ml_away", "mkt_spread_home", "mkt_total", "mkt_ml_home",
              "mkt_ml_away", "ou_in_pool", "saved_at"]


# ---------------------------------------------------------------------------
# The locked card: the picks as they stood before kickoff, graded afterwards
# ---------------------------------------------------------------------------
# The page rebuilds the card on every view, so grading it after the games
# would use post-game ratings and lines. The card (and the model's margin and
# total for every game) is written here before the first kickoff — the same
# rule as the props ledger — and results are read against that.

CARD_FILE = "pickem/cards.csv"
GAMES_FILE = "pickem/model_games.csv"
_CARD_COLS = ["season", "week", "slot", "kind", "game_id", "team", "text", "line_margin", "line_total",
              "p", "p_push", "dec", "market_p", "edge", "confidence", "payout", "exp_points", "market_exp",
              "mode", "source", "legs", "locked_at"]
_GAME_COLS = ["season", "week", "game_id", "home", "away", "model_margin", "line_margin", "model_total",
              "line_total", "p_home_cover", "p_home_win", "p_over", "source", "locked_at"]


def lock_card(card: pd.DataFrame, candidates: pd.DataFrame, season: int, week: int,
              mode: str, source: str, played: set | None = None) -> None:
    """Replace this week's locked card and game view with the ones given.
    Slots and games already played are left out — a lock made mid-week
    covers the games still to come, never a pick made in hindsight."""
    import json
    import os
    now = pd.Timestamp.now(tz="UTC").isoformat()
    played = set(played or [])
    c = card.copy()
    if played:
        def touches_played(r):
            if r["kind"] == "PARLAY":
                return any(l["game_id"] in played for l in (r.get("legs") or []))
            return r["game_id"] in played
        c = c[[not touches_played(r) for _, r in c.iterrows()]]
        candidates = candidates[~candidates["game_id"].isin(played)]
    c["legs"] = [json.dumps(l) if isinstance(l, list) else "" for l in c.get("legs", [""] * len(c))]
    if "payout" not in c.columns:
        c["payout"] = c["dec"]
    c = c.assign(season=int(season), week=int(week), mode=mode, source=source, locked_at=now)
    for col in _CARD_COLS:
        if col not in c.columns:
            c[col] = np.nan
    rows = []
    for gid, g in candidates.groupby("game_id", sort=False):
        f = g.iloc[0]
        def P(kind, team=None):
            q = g[g["kind"] == kind]
            if team is not None:
                q = q[q["team"] == team]
            return float(q["p"].iloc[0]) if len(q) else np.nan
        rows.append(dict(season=int(season), week=int(week), game_id=gid, home=f["home"], away=f["away"],
                         model_margin=f["model_margin"], line_margin=f["line_margin"],
                         model_total=f["model_total"], line_total=f["line_total"],
                         p_home_cover=P("ATS", f["home"]), p_home_win=P("ML", f["home"]), p_over=P("OVER"),
                         source=source, locked_at=now))
    gv = pd.DataFrame(rows, columns=_GAME_COLS)
    os.makedirs(os.path.dirname(CARD_FILE), exist_ok=True)
    for path, new, cols in ((CARD_FILE, c[_CARD_COLS], _CARD_COLS), (GAMES_FILE, gv, _GAME_COLS)):
        old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=cols)
        old = old[~((old["season"] == int(season)) & (old["week"] == int(week)))]
        pd.concat([old, new], ignore_index=True).sort_values(["season", "week"]).to_csv(path, index=False)


def load_locked(season: int, week: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """This week's locked card and game view, or (None, None)."""
    import json
    import os
    if not os.path.exists(CARD_FILE):
        return None, None
    c = pd.read_csv(CARD_FILE); c = c[(c["season"] == int(season)) & (c["week"] == int(week))]
    if c.empty:
        return None, None
    c = c.copy()
    c["legs"] = [json.loads(l) if isinstance(l, str) and l.startswith("[") else None for l in c["legs"]]
    g = pd.read_csv(GAMES_FILE) if os.path.exists(GAMES_FILE) else pd.DataFrame(columns=_GAME_COLS)
    g = g[(g["season"] == int(season)) & (g["week"] == int(week))]
    return c.reset_index(drop=True), g.reset_index(drop=True)


def load_locked_all() -> pd.DataFrame:
    import os
    return pd.read_csv(CARD_FILE) if os.path.exists(CARD_FILE) else pd.DataFrame(columns=_CARD_COLS)


def week_scores(sched_week: pd.DataFrame) -> dict:
    """game_id -> (home_score, away_score, home, away) for the played games."""
    q = sched_week[sched_week["played"] & sched_week["home_score"].notna()]
    return {r["game_id"]: (float(r["home_score"]), float(r["away_score"]), r["home_team"], r["away_team"])
            for _, r in q.iterrows()}


def game_results(games: pd.DataFrame, sched_week: pd.DataFrame) -> pd.DataFrame:
    """The locked game view beside the actual scores: model, line and actual
    margin and total, with each one's error, for the played games."""
    sc = week_scores(sched_week)
    g = games[games["game_id"].isin(sc)].copy()
    if g.empty:
        return g
    g["actual_margin"] = [sc[i][0] - sc[i][1] for i in g["game_id"]]
    g["actual_total"] = [sc[i][0] + sc[i][1] for i in g["game_id"]]
    g["score"] = [f"{sc[i][3]} {sc[i][1]:.0f} @ {sc[i][2]} {sc[i][0]:.0f}" for i in g["game_id"]]
    g["model_margin_err"] = g["model_margin"] - g["actual_margin"]
    g["line_margin_err"] = g["line_margin"] - g["actual_margin"]
    g["model_total_err"] = g["model_total"] - g["actual_total"]
    g["line_total_err"] = g["line_total"] - g["actual_total"]
    g["model_side_right"] = np.sign(g["model_margin"] - g["line_margin"]) == np.sign(g["actual_margin"] - g["line_margin"])
    return g


def load_pool_lines() -> pd.DataFrame:
    """The pool's posted lines, one row per game, as saved from the page (or
    entered by hand). Also the opener log: the market columns are the market
    at the moment the pool's lines were saved."""
    import os
    if not os.path.exists(POOL_FILE):
        return pd.DataFrame(columns=_POOL_COLS)
    d = pd.read_csv(POOL_FILE)
    for c in _POOL_COLS:
        if c not in d.columns:
            d[c] = np.nan
    return d


def save_pool_lines(lines: pd.DataFrame, season: int, week: int,
                    ou_games: list | None = None) -> pd.DataFrame:
    """Replace this week's rows with `lines` (the page's lines frame: pool and
    mkt_* columns). Returns the full file."""
    import os
    d = load_pool_lines()
    d = d[~((d["season"] == int(season)) & (d["week"] == int(week)))]
    ou = set(ou_games or [])
    rows = lines.assign(season=int(season), week=int(week),
                        ou_in_pool=lines["game_id"].isin(ou),
                        saved_at=pd.Timestamp.now(tz="UTC").isoformat())[_POOL_COLS]
    d = pd.concat([d, rows], ignore_index=True).sort_values(["season", "week", "game_id"])
    os.makedirs(os.path.dirname(POOL_FILE), exist_ok=True)
    d.to_csv(POOL_FILE, index=False)
    return d


def check_lines(lines: pd.DataFrame, max_gap: float = 3.0) -> pd.DataFrame:
    """Sanity-check hand-entered pool lines. One row per problem found.

    A mistyped spread sign is the dangerous error: it does not look wrong, it
    looks like a huge edge (a flipped 6.5 against a market of −7 reads as a
    13-point gift and a 94% pick). Two independent checks catch it — the
    pool's own moneylines disagree with its spread about who is favoured, and
    the pool's number is implausibly far from the market's.
    """
    out = []
    for _, r in lines.iterrows():
        g = f"{r['away']} @ {r['home']}"
        sp, mh, ma = float(r["spread_home"]), float(r["ml_home"]), float(r["ml_away"])
        # who each field says is favoured (shorter price / positive home spread)
        if sp != 0 and mh != ma:
            spread_says = "home" if sp > 0 else "away"
            ml_says = "home" if mh < ma else "away"
            if spread_says != ml_says:
                out.append(dict(game=g, problem="spread and moneylines disagree on the favourite",
                                detail=f"spread_home {sp:+g} says {spread_says}, "
                                       f"ML {mh:+.0f}/{ma:+.0f} says {ml_says}"))
        msp = r.get("mkt_spread_home")
        if pd.notna(msp) and abs(sp - float(msp)) > max_gap:
            out.append(dict(game=g, problem=f"pool spread is {abs(sp - float(msp)):.1f} pts from the market",
                            detail=f"pool {sp:+g} vs market {float(msp):+g}"
                                   + (" — sign flipped?" if sp * float(msp) < 0 else "")))
        mt = r.get("mkt_total")
        if pd.notna(mt) and abs(float(r["total"]) - float(mt)) > max_gap:
            out.append(dict(game=g, problem=f"pool total is {abs(float(r['total']) - float(mt)):.1f} pts from the market",
                            detail=f"pool {float(r['total']):g} vs market {float(mt):g}"))
    return pd.DataFrame(out, columns=["game", "problem", "detail"])


def apply_pool(lines: pd.DataFrame, saved: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Overwrite the POOL columns of the page's lines frame from the saved
    file for this week; games not in the file keep the seed. Market columns
    are left alone (they track the live market, not the file)."""
    if saved is None or saved.empty:
        return lines
    s = saved[(saved["season"] == int(season)) & (saved["week"] == int(week))]
    if s.empty:
        return lines
    out = lines.copy()
    key = s.set_index("game_id")
    for i, r in out.iterrows():
        if r["game_id"] not in key.index:
            continue
        m = key.loc[r["game_id"]]
        for c in ("spread_home", "total"):
            if pd.notna(m.get(c)):
                out.loc[i, c] = float(m[c])
        for c in ("ml_home", "ml_away"):
            if pd.notna(m.get(c)):
                out.loc[i, c] = int(m[c])
    return out


def apply_market(lines: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Overwrite the mkt_* columns from a consensus-lines frame (home, away,
    spread_home, total, ml_home, ml_away). Pool columns are left alone."""
    if market is None or market.empty:
        return lines
    out = lines.copy()
    key = market.set_index(["home", "away"])
    for i, r in out.iterrows():
        k = (r["home"], r["away"])
        if k not in key.index:
            continue
        m = key.loc[k]
        for src, dst in (("spread_home", "mkt_spread_home"), ("total", "mkt_total"),
                         ("ml_home", "mkt_ml_home"), ("ml_away", "mkt_ml_away")):
            if pd.notna(m.get(src)):
                out.loc[i, dst] = float(m[src]) if src in ("spread_home", "total") else int(m[src])
    return out


# ---------------------------------------------------------------------------
# Grading every candidate against the simulations
# ---------------------------------------------------------------------------

def game_candidates(sim: dict, line: pd.Series, juice: int = DEFAULT_JUICE,
                    teaser_pts: float = DEFAULT_TEASER_POINTS,
                    market_weight: float = 0.0) -> list[dict]:
    """Every bettable side of one game, with its probability.

    `market_weight` shifts the simulated margins and totals part-way toward the
    market's expected margin (the spread) and total: 0 grades every candidate
    on the pure model, 1 on a distribution centred where the market is. The
    model's SHAPE is kept either way — only the centre moves — so ATS, ML,
    teaser and total probabilities all come from one consistent distribution.
    """
    m = sim["pts_home"].astype(float) - sim["pts_away"].astype(float)
    t = sim["pts_home"].astype(float) + sim["pts_away"].astype(float)
    sp, tot = float(line["spread_home"]), float(line["total"])
    model_margin, model_total = float(m.mean()), float(t.mean())
    w = float(np.clip(market_weight, 0.0, 1.0))
    if w > 0:
        m = m + w * (sp - model_margin)
        t = t + w * (tot - model_total)
    home, away = sim["home"], sim["away"]
    fav, dog = (home, away) if sp > 0 else (away, home) if sp < 0 else (None, None)
    ml_home, ml_away = int(line["ml_home"]), int(line["ml_away"])
    if fav is None:                                   # pick'em: dog = worse ML
        dog = home if ml_home > ml_away else away
    dec_j = to_decimal(juice)

    def side(kind, team, p, push, odds, text, extra=None):
        d = dict(game_id=sim["game_id"], home=home, away=away, kind=kind, team=team,
                 text=text, p=float(p), p_push=float(push), odds=int(odds),
                 dec=to_decimal(odds), market_p=implied_prob(odds),
                 model_margin=model_margin, model_total=model_total,
                 line_margin=sp, line_total=tot)
        d.update(extra or {})
        return d

    c = []
    # against the spread
    c.append(side("ATS", home, (m > sp).mean(), (m == sp).mean(), juice,
                  f"{home} {-sp:+g}" if sp != 0 else f"{home} PK"))
    c.append(side("ATS", away, (m < sp).mean(), (m == sp).mean(), juice,
                  f"{away} {sp:+g}" if sp != 0 else f"{away} PK"))
    # moneylines (the dog is a single-pick candidate; both are parlay legs)
    c.append(side("ML", home, (m > 0).mean(), (m == 0).mean(), ml_home,
                  f"{home} ML {ml_home:+d}", dict(is_dog=home == dog)))
    c.append(side("ML", away, (m < 0).mean(), (m == 0).mean(), ml_away,
                  f"{away} ML {ml_away:+d}", dict(is_dog=away == dog)))
    # totals
    c.append(side("OVER", None, (t > tot).mean(), (t == tot).mean(), juice, f"Over {tot:g}"))
    c.append(side("UNDER", None, (t < tot).mean(), (t == tot).mean(), juice, f"Under {tot:g}"))
    # teaser legs: the spread moves teaser_pts in the picked side's favour
    c.append(side("TEASER", home, (m > sp - teaser_pts).mean(), (m == sp - teaser_pts).mean(),
                  juice, f"{home} {-(sp - teaser_pts):+g} (teased)"))
    c.append(side("TEASER", away, (m < sp + teaser_pts).mean(), (m == sp + teaser_pts).mean(),
                  juice, f"{away} {(sp + teaser_pts):+g} (teased)"))
    return c


def market_candidates(hist: pd.DataFrame, game_id: str, home: str, away: str,
                      line: pd.Series, juice: int = DEFAULT_JUICE,
                      teaser_pts: float = DEFAULT_TEASER_POINTS) -> list[dict]:
    """Every bettable side of one game, priced by the market and history
    (`nflsim.market`): P(the POOL's line is beaten | the MARKET's line).

    `line` carries the pool's numbers (spread_home, total, ml_home, ml_away)
    and the market's (mkt_*). When they are equal a side is ~50% and the only
    value in the game is a Wong teaser leg; when the pool's number is better,
    the probability moves by the key-number mass between the two.
    """
    from . import market as MK
    sp, tot = float(line["spread_home"]), float(line["total"])
    msp = float(line.get("mkt_spread_home", sp)); mtot = float(line.get("mkt_total", tot))
    ml_home, ml_away = int(line["ml_home"]), int(line["ml_away"])
    mml_h, mml_a = line.get("mkt_ml_home", ml_home), line.get("mkt_ml_away", ml_away)
    fav, dog = (home, away) if msp > 0 else (away, home) if msp < 0 else (None, None)
    if fav is None:
        dog = home if ml_home > ml_away else away

    def side(kind, team, p, push, odds, text, n=0, extra=None):
        d = dict(game_id=game_id, home=home, away=away, kind=kind, team=team, text=text,
                 p=float(p), p_push=float(push), odds=int(odds), dec=to_decimal(odds),
                 market_p=implied_prob(odds), model_margin=msp, model_total=mtot,
                 line_margin=sp, line_total=tot, n_hist=int(n), source="market")
        d.update(extra or {})
        return d

    c = []
    for team, sd in ((home, "home"), (away, "away")):
        p, pu, n = MK.cover_prob(hist, msp, sp, sd)
        txt = f"{team} {(-sp if sd == 'home' else sp):+g}" if sp != 0 else f"{team} PK"
        c.append(side("ATS", team, p, pu, juice, txt, n,
                      dict(number_edge=float(sp - msp) * (-1 if sd == "home" else 1))))
    # moneyline: the market's vig-free price is the truth when we have one
    if pd.notna(mml_h) and pd.notna(mml_a):
        ph, pa = MK.vig_free(mml_h, mml_a)
        tie = 0.0
    else:
        ph, tie = MK.win_prob(hist, msp, "home"); pa = 1 - ph - tie
    c.append(side("ML", home, ph, tie, ml_home, f"{home} ML {ml_home:+d}", 0, dict(is_dog=home == dog)))
    c.append(side("ML", away, pa, tie, ml_away, f"{away} ML {ml_away:+d}", 0, dict(is_dog=away == dog)))
    for kind, sd in (("OVER", "over"), ("UNDER", "under")):
        p, pu = MK.total_prob(hist, mtot, tot, sd)
        c.append(side(kind, None, p, pu, juice, f"{kind.title()} {tot:g}", 0,
                      dict(number_edge=float(mtot - tot) if sd == "over" else float(tot - mtot))))
    for team, sd, tl in ((home, "home", sp - teaser_pts), (away, "away", sp + teaser_pts)):
        p, pu, n = MK.cover_prob(hist, msp, tl, sd)
        txt = f"{team} {(-tl if sd == 'home' else tl):+g} (teased)"
        c.append(side("TEASER", team, p, pu, juice, txt, n))
    return c


def _value(p: float, dec: float, mode: str) -> float:
    """Expected points per confidence point."""
    return p * dec if mode == "odds" else p


def _best_combo(legs: list[dict], mode: str, k: int = 3) -> list[dict]:
    """Best k legs from distinct games. The expected return of a parlay is the
    product of its legs' values, so it is the k legs with the largest values —
    subject to one leg per game, which the greedy pass over sorted legs gives."""
    picked, used = [], set()
    for leg in sorted(legs, key=lambda x: _value(x["p"], x["dec"], mode), reverse=True):
        if leg["game_id"] in used:
            continue
        picked.append(leg); used.add(leg["game_id"])
        if len(picked) == k:
            break
    return picked


def build_card(sims: dict, lines: pd.DataFrame, mode: str = "flat",
               juice: int = DEFAULT_JUICE, teaser_pts: float = DEFAULT_TEASER_POINTS,
               teaser_odds: int = DEFAULT_TEASER_ODDS,
               parlay_pricing: str = "true", parlay_odds: int = DEFAULT_PARLAY_ODDS,
               n_slots: int = N_SLOTS, exclude_played: bool = False,
               market_weight: float = 0.0, ou_games: list | None = None,
               source: str = "engine", hist: pd.DataFrame | None = None) -> dict:
    """Choose the picks and assign confidence. Returns the card and the full
    candidate table so the reasoning is inspectable.

    `source`: 'engine' prices every side from the simulated games in `sims`
    (blended toward the market by `market_weight`); 'market' prices them from
    the market's lines and the empirical margin/total distributions in `hist`
    (`nflsim.market`) — the pool's numbers graded against the market's. With
    'market', `sims` only needs game_id / home / away / played per game.
    """
    lines = lines.set_index("game_id")
    cands = []
    for gid, sim in sims.items():
        if gid not in lines.index:
            continue
        if exclude_played and sim.get("played"):
            continue
        if source == "market":
            cands.extend(market_candidates(hist, gid, sim["home"], sim["away"],
                                           lines.loc[gid], juice, teaser_pts))
        else:
            cands.extend(game_candidates(sim, lines.loc[gid], juice, teaser_pts, market_weight))
    cand = pd.DataFrame(cands)
    if cand.empty:
        return dict(card=pd.DataFrame(), candidates=cand, notes=["No games to pick."])
    cand["value"] = [_value(p, d, mode) for p, d in zip(cand["p"], cand["dec"])]
    cand["edge"] = cand["p"] - cand["market_p"]

    picks, notes = [], []
    games = list(dict.fromkeys(cand["game_id"]))

    # 1. one pick per game: best of home ATS / away ATS / dog ML
    for gid in games:
        g = cand[(cand["game_id"] == gid)]
        opts = pd.concat([g[g["kind"] == "ATS"], g[(g["kind"] == "ML") & g["is_dog"].fillna(False)]])
        best = opts.sort_values("value", ascending=False).iloc[0]
        picks.append(dict(slot="Game pick", **best.to_dict(), payout=best["dec"]))

    # 2. the three combos
    ats = _best_combo(cand[cand["kind"] == "ATS"].to_dict("records"), mode)
    mls = _best_combo(cand[cand["kind"] == "ML"].to_dict("records"), mode)
    tsr = _best_combo(cand[cand["kind"] == "TEASER"].to_dict("records"), "flat")
    for name, legs, pay in (("3-team ATS parlay", ats, None),
                            ("3-team ML parlay", mls, None),
                            (f"3-team {teaser_pts:g}-pt teaser", tsr, to_decimal(teaser_odds))):
        if len(legs) < 3:
            notes.append(f"Not enough games for the {name}.")
            continue
        p = float(np.prod([l["p"] for l in legs]))
        if pay is None:
            pay = (float(np.prod([l["dec"] for l in legs])) if parlay_pricing == "true"
                   else to_decimal(parlay_odds))
        picks.append(dict(slot=name, game_id="+".join(l["game_id"] for l in legs),
                          kind="PARLAY", team=None, text=" / ".join(l["text"] for l in legs),
                          p=p, p_push=0.0, odds=0, dec=pay, market_p=1.0 / pay, payout=pay,
                          value=_value(p, pay, mode), edge=p - 1.0 / pay,
                          legs=[dict(game_id=l["game_id"], kind=l["kind"], team=l["team"],
                                     line_margin=l["line_margin"], line_total=l["line_total"])
                                for l in legs]))

    # 3. totals: the pool names the games (ou_games); the model picks over or
    #    under on each. Without a list, the best-value games fill the slots.
    n_ou = n_slots - len(picks)
    ou = cand[cand["kind"].isin(["OVER", "UNDER"])]
    if ou_games is not None:
        chosen = ou[ou["game_id"].isin(list(ou_games))]
        best_ou = chosen.sort_values("value", ascending=False).drop_duplicates("game_id")
        if len(best_ou) != n_ou:
            notes.append(f"The pool has {n_ou} total slot{'s' if n_ou != 1 else ''} this "
                         f"week but {len(best_ou)} game{'s are' if len(best_ou) != 1 else ' is'} "
                         "selected for totals.")
        for _, r in best_ou.iterrows():
            picks.append(dict(slot="Total", **r.to_dict(), payout=r["dec"]))
    elif n_ou > 0:
        best_ou = (ou.sort_values("value", ascending=False)
                     .drop_duplicates("game_id").head(n_ou))
        for _, r in best_ou.iterrows():
            picks.append(dict(slot="Total", **r.to_dict(), payout=r["dec"]))
        if len(best_ou) < n_ou:
            notes.append(f"Only {len(best_ou)} totals available for {n_ou} slots.")
    if len(picks) > n_slots:
        notes.append(f"{len(picks)} picks exceed {n_slots} slots — the lowest-value "
                     "picks are dropped.")
        picks = sorted(picks, key=lambda x: x["value"], reverse=True)[:n_slots]

    card = pd.DataFrame(picks).sort_values("value", ascending=False).reset_index(drop=True)
    card["confidence"] = list(range(n_slots, n_slots - len(card), -1))
    card["n_ou_slots"] = n_ou
    card["exp_points"] = card["confidence"] * card["value"]
    card["market_exp"] = card["confidence"] * [
        _value(mp, d, mode) for mp, d in zip(card["market_p"], card["dec"])]
    return dict(card=card, candidates=cand, notes=notes,
                total_exp=float(card["exp_points"].sum()),
                market_exp=float(card["market_exp"].sum()),
                mode=mode, market_weight=float(market_weight), source=source)


def game_view(candidates: pd.DataFrame) -> pd.DataFrame:
    """One row per game: model vs market centre, and the probabilities behind
    every candidate — so the card's reasoning can be read game by game."""
    rows = []
    for gid, g in candidates.groupby("game_id", sort=False):
        f = g.iloc[0]
        def P(kind, team=None):
            q = g[g["kind"] == kind]
            if team is not None:
                q = q[q["team"] == team]
            return float(q["p"].iloc[0]) if len(q) else np.nan
        rows.append(dict(
            Game=f"{f['away']} @ {f['home']}",
            **{"Model margin (home)": f["model_margin"], "Line (home)": f["line_margin"],
               "Model total": f["model_total"], "Line total": f["line_total"],
               f"{f['home']} covers": P("ATS", f["home"]), f"{f['away']} covers": P("ATS", f["away"]),
               f"{f['home']} wins": P("ML", f["home"]), "Over": P("OVER")}))
    return pd.DataFrame(rows)


def card_display(card: pd.DataFrame) -> pd.DataFrame:
    """The card as the reader sees it."""
    d = pd.DataFrame({
        "Conf": card["confidence"],
        "Slot": card["slot"],
        "Pick": card["text"],
        "P(win)": card["p"],
        "Push": card["p_push"],
        "Price": [to_american(x) for x in card["dec"]],
        "Price implies": card["market_p"],
        "Edge": card["edge"],
        "Exp. pts": card["exp_points"],
    })
    if "number_edge" in card.columns:
        d.insert(3, "Number vs market", card["number_edge"].fillna(0.0))
    return d


# ---------------------------------------------------------------------------
# Grading a card against real results, and replaying past weeks honestly
# ---------------------------------------------------------------------------
# A past week's card is rebuilt from the team backtest (`backtest.team_backtest`):
# for every game the model's margin and total with the ratings, home field,
# availability and calibration refit on games BEFORE that week, and the closing
# line. The engine's full simulation cannot be replayed for a past week (depth
# charts and injury reports are as-of-now), so each game's distribution is the
# harness's own: Normal around the model's margin (sd MARGIN_SD) and total
# (sd HIST_TOTAL_SD, the 2025 out-of-sample residual), independent - which is
# what page 10 scores too. Pushes score zero. A parlay with a pushed leg counts
# as lost (pools differ; this is the conservative reading).

HIST_TOTAL_SD = 13.4
HIST_SAMPLES = 8000


def settle(kind: str, team, home: str, away: str, line_margin: float, line_total: float,
           hs: float, as_: float, teaser_pts: float = DEFAULT_TEASER_POINTS) -> str:
    """'win' / 'loss' / 'push' for one side given the final score."""
    m, t = float(hs) - float(as_), float(hs) + float(as_)
    sp, tot = float(line_margin), float(line_total)
    if kind == "ATS":
        edge = (m - sp) if team == home else (sp - m)
    elif kind == "ML":
        edge = m if team == home else -m
    elif kind == "TEASER":
        edge = (m - (sp - teaser_pts)) if team == home else ((sp + teaser_pts) - m)
    elif kind == "OVER":
        edge = t - tot
    elif kind == "UNDER":
        edge = tot - t
    else:
        return "loss"
    return "win" if edge > 0 else "loss" if edge < 0 else "push"


def grade_card(card: pd.DataFrame, scores: dict, mode: str,
               teaser_pts: float = DEFAULT_TEASER_POINTS) -> pd.DataFrame:
    """Add `result` and `points` to a card.

    `scores`: game_id -> (home_score, away_score, home, away). Points =
    confidence (x payout in odds mode) on a win, else 0. A pick whose game has
    no score yet is 'open'.
    """
    out = card.copy()
    results, points = [], []
    for _, r in out.iterrows():
        if r["kind"] == "PARLAY":
            legs = r.get("legs") or []
            res = []
            for l in legs:
                if l["game_id"] not in scores:
                    res.append("open"); continue
                hs, as_, home, away = scores[l["game_id"]]
                res.append(settle(l["kind"], l["team"], home, away,
                                  l["line_margin"], l["line_total"], hs, as_, teaser_pts))
            result = ("open" if "open" in res
                      else "win" if res and all(x == "win" for x in res) else "loss")
        elif r["game_id"] in scores:
            hs, as_, home, away = scores[r["game_id"]]
            result = settle(r["kind"], r["team"], home, away, r["line_margin"],
                            r["line_total"], hs, as_, teaser_pts)
        else:
            result = "open"
        pay = float(r["payout"]) if mode == "odds" else 1.0
        results.append(result)
        points.append(float(r["confidence"]) * pay if result == "win" else 0.0)
    out["result"] = results
    out["points"] = points
    return out


def history_sims(bt_week: pd.DataFrame, n: int = HIST_SAMPLES, seed: int = 5,
                 margin_sd: float | None = None) -> dict:
    """Per-game (margin, total) samples from the backtest's out-of-sample
    predictions, in the same shape `simulate_slate` returns."""
    from . import backtest as B
    rng = np.random.default_rng(seed)
    msd = float(margin_sd if margin_sd is not None else B.MARGIN_SD)
    out = {}
    for _, g in bt_week.iterrows():
        m = rng.normal(float(g["pred_margin"]), msd, n)
        t = rng.normal(float(g["pred_total"]), HIST_TOTAL_SD, n)
        out[g["game_id"]] = dict(
            game_id=g["game_id"], home=g["home"], away=g["away"],
            pts_home=(t + m) / 2.0, pts_away=(t - m) / 2.0, played=True,
            home_score=g["actual_home"], away_score=g["actual_away"])
    return out


def card_history(bt: pd.DataFrame, sched: pd.DataFrame, mode: str = "odds",
                 juice: int = DEFAULT_JUICE, teaser_pts: float = DEFAULT_TEASER_POINTS,
                 teaser_odds: int = DEFAULT_TEASER_ODDS, parlay_pricing: str = "true",
                 parlay_odds: int = DEFAULT_PARLAY_ODDS, n_slots: int = N_SLOTS,
                 market_weight: float = 0.0,
                 n_samples: int = HIST_SAMPLES, source: str = "engine",
                 hist: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild and grade the card for every (season, week) in a team backtest.

    Returns (weeks, picks): one summary row per week, and every graded pick.
    Totals slots are filled by the model's best-value games (the pool's actual
    choices for past weeks are unknown).
    """
    if bt is None or bt.empty:
        return pd.DataFrame(), pd.DataFrame()
    weeks, picks = [], []
    sched = sched.set_index("game_id")
    for (S, w), g in bt.groupby(["season", "week"]):
        sims = history_sims(g, n_samples, seed=int(S * 100 + w))
        gids = [gid for gid in g["game_id"] if gid in sched.index]
        if not gids:
            continue
        lines = default_lines(sched.loc[gids].reset_index())
        h = hist[hist["season"] < int(S)] if (source == "market" and hist is not None) else hist
        res = build_card(sims, lines, mode=mode, juice=juice, teaser_pts=teaser_pts,
                         teaser_odds=teaser_odds, parlay_pricing=parlay_pricing,
                         parlay_odds=parlay_odds, n_slots=n_slots, market_weight=market_weight,
                         source=source, hist=h)
        card = res["card"]
        if card.empty:
            continue
        scores = {r["game_id"]: (float(r["actual_home"]), float(r["actual_away"]),
                                 r["home"], r["away"]) for _, r in g.iterrows()}
        graded = grade_card(card, scores, mode, teaser_pts)
        graded["season"], graded["week"] = int(S), int(w)
        picks.append(graded)
        st = graded[graded["result"] != "open"]
        gp, tt = st[st["slot"] == "Game pick"], st[st["slot"] == "Total"]
        weeks.append(dict(
            season=int(S), week=int(w), games=int(len(g)), picks=int(len(st)),
            points=float(st["points"].sum()),
            expected=float(res["total_exp"]), market_expected=float(res["market_exp"]),
            wins=int((st["result"] == "win").sum()), pushes=int((st["result"] == "push").sum()),
            exp_wins=float(st["p"].sum()),
            top5_wins=int((st.head(5)["result"] == "win").sum()),
            game_pick_hit=float((gp["result"] == "win").mean()) if len(gp) else np.nan,
            totals_hit=float((tt["result"] == "win").mean()) if len(tt) else np.nan,
            combos_won=int(((st["kind"] == "PARLAY") & (st["result"] == "win")).sum()),
        ))
    wk = pd.DataFrame(weeks)
    if not wk.empty:
        wk["cum_points"] = wk["points"].cumsum()
        wk["cum_expected"] = wk["expected"].cumsum()
    return wk, (pd.concat(picks, ignore_index=True) if picks else pd.DataFrame())
