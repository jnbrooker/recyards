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
                   use_injuries: bool = True, seed: int = 23) -> dict:
    """game_id -> dict(home, away, pts_home, pts_away, label) for every game."""
    ratings = ctx["ratings"]
    out, rosters = {}, {}

    def roster(team):
        if team not in rosters:
            rosters[team] = G.roster_for(ctx, team, use_injuries=use_injuries)
        return rosters[team]

    for i, g in games.reset_index(drop=True).iterrows():
        home, away = g["home_team"], g["away_team"]
        if home not in ratings["off"].index or away not in ratings["off"].index:
            continue
        try:
            sim = G.simulate_game(ratings, ctx["wk"], roster(home), roster(away),
                                  home, away, ctx["pass_vol"], ctx["rush_vol"],
                                  ctx["rush_def"], ctx["lg_pass"], home="a",
                                  n_sims=n_sims, seed=seed + i, avail=ctx.get("avail"))
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
        rows.append(dict(
            game_id=g["game_id"], Game=f"{g['away_team']} @ {g['home_team']}",
            home=g["home_team"], away=g["away_team"],
            spread_home=float(sp) if pd.notna(sp) else 0.0,
            total=float(tot) if pd.notna(tot) else 45.0,
            ml_home=int(ml_h) if pd.notna(ml_h) else -110,
            ml_away=int(ml_a) if pd.notna(ml_a) else -110,
            played=bool(g.get("played", False)),
        ))
    return pd.DataFrame(rows)


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
               market_weight: float = 0.0, ou_games: list | None = None) -> dict:
    """Choose the picks and assign confidence. Returns the card and the full
    candidate table so the reasoning is inspectable."""
    lines = lines.set_index("game_id")
    cands = []
    for gid, sim in sims.items():
        if gid not in lines.index:
            continue
        if exclude_played and sim.get("played"):
            continue
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
                          value=_value(p, pay, mode), edge=p - 1.0 / pay))

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
                mode=mode, market_weight=float(market_weight))


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
        "Model P": card["p"],
        "Push": card["p_push"],
        "Price": [to_american(x) for x in card["dec"]],
        "Market P": card["market_p"],
        "Edge": card["edge"],
        "Exp. pts": card["exp_points"],
    })
    return d
