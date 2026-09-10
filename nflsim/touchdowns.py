"""
nflsim/touchdowns.py — Monte Carlo model for a player's total touchdowns,
combining BOTH rushing and receiving.

Approach = opportunities × conversion (the accurate part is the expected count,
not the count distribution):

  1. Opportunities  — simulate the player's receptions and carries this game
     from their own game-to-game volume (Negative Binomial, so a hot game and a
     quiet game both show up).
  2. Conversion     — each reception scores with prob p_rec_td, each carry with
     prob p_rush_td. Both rates are REGRESSED toward the positional league
     average (TD rates are noisy), which is what stops a small-sample fluke from
     dominating. Goal-line role falls straight out of a high rush-TD-per-carry.
  3. Total TDs      = Binomial(receptions, p_rec_td) + Binomial(carries, p_rush_td).

Drawing TDs per-opportunity (rather than a flat Poisson) correlates scoring with
volume — a big workload game is also a bigger TD chance — and still aggregates to
the familiar near-Poisson shape for the count. Anytime-TD = P(total ≥ 1).

Defense adjustment (shrunk hard, because TD-allowed splits are very noisy):
nudges the conversion rates by the TDs the opponent allows to that position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D

# Pseudo-counts for regressing a player's TD rate toward the league mean.
# Higher = more regression. ~ "this many opportunities of league-average prior".
REC_TD_PRIOR_N = 40.0      # receptions-worth of prior
RUSH_TD_PRIOR_N = 45.0     # carries-worth of prior

# TD-allowed defense splits are noisy; default shrink is strong.
DEFAULT_TD_DEF_SHRINK = 0.35


# ---------------------------------------------------------------------------
# League positional TD rates (the regression targets)
# ---------------------------------------------------------------------------

def league_td_rates(wk: pd.DataFrame) -> dict:
    """League mean rec-TD-per-reception and rush-TD-per-carry, by position."""
    out = {}
    for pos, g in wk.groupby("position"):
        rec = g["receptions"].sum()
        car = g["carries"].sum()
        out[pos] = dict(
            rec_td_per_rec=float(g["receiving_tds"].sum() / rec) if rec > 0 else 0.05,
            rush_td_per_car=float(g["rushing_tds"].sum() / car) if car > 0 else 0.025,
        )
    # sensible global fallbacks
    out["_ALL_"] = dict(
        rec_td_per_rec=float(wk["receiving_tds"].sum() / max(wk["receptions"].sum(), 1)),
        rush_td_per_car=float(wk["rushing_tds"].sum() / max(wk["carries"].sum(), 1)),
    )
    return out


# ---------------------------------------------------------------------------
# Player priors: expected volume + regressed conversion rates
# ---------------------------------------------------------------------------

def _nb_params(mean, var):
    """Negative-Binomial (n, p) from a mean and variance. Falls back to Poisson
    behaviour when not overdispersed."""
    mean = max(float(mean), 1e-6)
    var = float(var)
    if var <= mean * 1.05:            # not overdispersed -> ~Poisson
        return None, mean
    n = mean * mean / (var - mean)
    p = n / (n + mean)
    return max(n, 1e-3), p


def player_td_priors(wk: pd.DataFrame, player_id: str, lg: dict) -> dict:
    p = wk[wk["player_id"] == player_id].copy()
    if p.empty:
        raise ValueError("No games for this player.")
    pos = p["position"].iloc[-1]
    w = p["season_w"].values
    prior = lg.get(pos, lg["_ALL_"])

    # --- expected volume per game (mean & variance for the NB draw) ---
    rec_g = p["receptions"].values
    car_g = p["carries"].values
    mu_rec = D.wmean(rec_g, w) if len(rec_g) else 0.0
    mu_car = D.wmean(car_g, w) if len(car_g) else 0.0
    var_rec = np.average((rec_g - mu_rec) ** 2, weights=w) if len(rec_g) > 1 else mu_rec
    var_car = np.average((car_g - mu_car) ** 2, weights=w) if len(car_g) > 1 else mu_car

    # --- regressed conversion rates ---
    rec_tot = p["receptions"].sum()
    car_tot = p["carries"].sum()
    rec_td = p["receiving_tds"].sum()
    rush_td = p["rushing_tds"].sum()
    p_rec_td = (rec_td + REC_TD_PRIOR_N * prior["rec_td_per_rec"]) / (rec_tot + REC_TD_PRIOR_N)
    p_rush_td = (rush_td + RUSH_TD_PRIOR_N * prior["rush_td_per_car"]) / (car_tot + RUSH_TD_PRIOR_N)

    return dict(
        player_id=player_id, name=p["player_display_name"].iloc[-1], position=pos,
        team=p["recent_team"].iloc[-1], games=int(len(p)),
        mu_rec=float(max(mu_rec, 0.0)), var_rec=float(max(var_rec, mu_rec)),
        mu_car=float(max(mu_car, 0.0)), var_car=float(max(var_car, mu_car)),
        p_rec_td=float(np.clip(p_rec_td, 0.0, 0.4)),
        p_rush_td=float(np.clip(p_rush_td, 0.0, 0.25)),
        raw_rec_td_per_rec=float(rec_td / rec_tot) if rec_tot > 0 else 0.0,
        raw_rush_td_per_car=float(rush_td / car_tot) if car_tot > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Defense TD-allowed profiles (rec & rush), vs league, per position
# ---------------------------------------------------------------------------

def td_defense_profiles(wk: pd.DataFrame) -> dict:
    """(defense, position) -> ratios of rec-TD/reception and rush-TD/carry
    allowed vs league. >1 = gives up more scores than average."""
    d = (wk.groupby(["opponent_team", "position"], as_index=False)
           .agg(rec=("receptions", "sum"), rtd=("receiving_tds", "sum"),
                car=("carries", "sum"), rutd=("rushing_tds", "sum")))
    lg = {}
    for pos, g in d.groupby("position"):
        lg[pos] = dict(
            rec=float(g["rtd"].sum() / max(g["rec"].sum(), 1)),
            rush=float(g["rutd"].sum() / max(g["car"].sum(), 1)),
        )
    prof = {}
    for _, r in d.iterrows():
        pos = r["position"]; base = lg[pos]
        rec_rate = r["rtd"] / r["rec"] if r["rec"] >= 25 else base["rec"]
        rush_rate = r["rutd"] / r["car"] if r["car"] >= 25 else base["rush"]
        prof[(r["opponent_team"], pos)] = dict(
            r_rec_td=float(rec_rate / base["rec"]) if base["rec"] > 0 else 1.0,
            r_rush_td=float(rush_rate / base["rush"]) if base["rush"] > 0 else 1.0,
        )
    return prof


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _draw_counts(rng, mean, var, n):
    nb_n, param = _nb_params(mean, var)
    if nb_n is None:
        return rng.poisson(param, n)
    return rng.negative_binomial(nb_n, param, n)


def simulate(priors: dict, def_prof: dict | None, n_sims: int = 40000,
             def_shrink: float = DEFAULT_TD_DEF_SHRINK, seed: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n = int(n_sims)

    m_rec, m_rush = 1.0, 1.0
    if def_prof is not None:
        m_rec = D.shrink(def_prof["r_rec_td"], def_shrink)
        m_rush = D.shrink(def_prof["r_rush_td"], def_shrink)

    receptions = _draw_counts(rng, priors["mu_rec"], priors["var_rec"], n)
    carries = _draw_counts(rng, priors["mu_car"], priors["var_car"], n)

    p_rec = float(np.clip(priors["p_rec_td"] * m_rec, 0.0, 0.6))
    p_rush = float(np.clip(priors["p_rush_td"] * m_rush, 0.0, 0.4))
    rec_tds = rng.binomial(receptions, p_rec)
    rush_tds = rng.binomial(carries, p_rush)
    total = rec_tds + rush_tds

    return dict(total=total, rec_tds=rec_tds, rush_tds=rush_tds,
                receptions=receptions, carries=carries,
                adj=dict(m_rec=m_rec, m_rush=m_rush),
                exp_rec_td=float(rec_tds.mean()), exp_rush_td=float(rush_tds.mean()))


def summarize(sim: dict, line: float = 0.5) -> dict:
    t = sim["total"]
    p_over = float((t > line).mean())        # e.g. line 0.5 -> P(anytime TD)
    maxk = int(min(t.max(), 4))
    dist = {k: float((t == k).mean()) for k in range(maxk)}
    dist[f"{maxk}+"] = float((t >= maxk).mean())
    return dict(
        mean=float(t.mean()), exp_rec=float(sim["rec_tds"].mean()),
        exp_rush=float(sim["rush_tds"].mean()),
        p_anytime=float((t >= 1).mean()), p_2plus=float((t >= 2).mean()),
        line=float(line), p_over=p_over, p_under=1 - p_over,
        fair_over_odds=D.american(p_over), fair_under_odds=D.american(1 - p_over),
        dist=dist,
    )


if __name__ == "__main__":
    seasons = (2024, 2025)
    print("Loading data...")
    wk = D.load_weekly(seasons)
    lg = league_td_rates(wk)
    defs = td_defense_profiles(wk)

    # a goal-line RB, a dual-threat RB, and a WR
    names = ["Derrick Henry", "Christian McCaffrey", "Ja'Marr Chase", "Jahmyr Gibbs"]
    ids = (wk[wk["player_display_name"].isin(names)]
           .groupby("player_display_name")["player_id"].last())
    for nm, pid in ids.items():
        pri = player_td_priors(wk, pid, lg)
        dp = defs.get(("SF", pri["position"]))
        sim = simulate(pri, dp, seed=3)
        s = summarize(sim, 0.5)
        print(f"\n{pri['name']} ({pri['position']}, {pri['games']}g) vs SF")
        print(f"  E[rec]={pri['mu_rec']:.1f} p_rec_td={pri['p_rec_td']:.3f} | "
              f"E[car]={pri['mu_car']:.1f} p_rush_td={pri['p_rush_td']:.3f}")
        print(f"  anytime TD {s['p_anytime']:.1%} (fair {s['fair_over_odds']}) | "
              f"2+ {s['p_2plus']:.1%} | E[TD]={s['mean']:.2f} "
              f"(rec {s['exp_rec']:.2f} + rush {s['exp_rush']:.2f})")
