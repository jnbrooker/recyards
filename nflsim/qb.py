"""
nflsim/qb.py — Monte Carlo models for the two QB "disruption" stats, both seen
from the OFFENSE side so they read clean offensive feeds, not defensive box
scores (roadmap §3.4–3.5):

  * SACKS TAKEN  — sacks per DROPBACK. Expected = dropbacks × sack_rate, where
    sack_rate combines the QB's own sacks-allowed rate with the defense's
    sack-generating rate in LOG-ODDS, then is scaled by the QB's NGS average
    time to throw (a quick release takes fewer sacks). Counts are low and
    overdispersed, so a game-level rate wobble is added before the count draw.

  * INTs THROWN  — INTs per ATTEMPT. Expected = attempts × int_rate. INT rate is
    one of the noisiest stats in football, so the QB's own rate is regressed
    HARD toward the league mean before any defense is applied.

Both stats follow the house template: fit a rate with its own game-to-game
variance, adjust for the opponent with shrinkage, simulate the event chain.
Volume (dropbacks / attempts) is drawn per game from the QB's own distribution
so the count correlates with how much he throws.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D

# Regression strength: pseudo-observations of a league-average prior.
SACK_PRIOR_N = 220.0     # dropbacks-worth of prior on sack rate
INT_PRIOR_N = 400.0      # attempts-worth of prior on INT rate (heavy: INT is noisy)

# Defense shrink defaults (INT splits noisier than sack splits).
DEFAULT_SACK_DEF_SHRINK = 0.5
DEFAULT_INT_DEF_SHRINK = 0.35

# Game-to-game rate wobble (log-normal sigma) → overdispersed counts.
SACK_GAME_SIGMA = 0.35
INT_GAME_SIGMA = 0.45

# NGS time-to-throw scaling of sack rate: (ttt / league)**exponent.
TTT_EXPONENT = 1.5
TTT_CLIP = (0.7, 1.4)


# ---------------------------------------------------------------------------
# League rates
# ---------------------------------------------------------------------------

def league_pass_rates(wk: pd.DataFrame) -> dict:
    """League sack rate per dropback and INT rate per attempt, QBs only.

    Falls back to the constants in `data` if a feed is missing the columns, so a
    schema change degrades to league-average rather than crashing a page.
    """
    q = wk[wk["position"] == "QB"]
    at = float(q["attempts"].sum()) if "attempts" in q.columns else 0.0
    db = float(q["dropbacks"].sum()) if "dropbacks" in q.columns else at
    sacks = float(q["sacks"].sum()) if "sacks" in q.columns else 0.0
    ints = float(q["interceptions"].sum()) if "interceptions" in q.columns else 0.0
    return dict(
        sack=sacks / db if db > 0 and sacks > 0 else D.LG_SACK_RATE,
        intr=ints / at if at > 0 and ints > 0 else D.LG_INT_RATE,
    )


# ---------------------------------------------------------------------------
# QB priors: volume (mean & var) + regressed sack / INT rates
# ---------------------------------------------------------------------------

def qb_priors(wk: pd.DataFrame, player_id: str, lg: dict,
              ttt_map: dict | None = None) -> dict:
    missing = [c for c in ("attempts", "sacks", "interceptions")
               if c not in wk.columns]
    if missing:
        raise ValueError(
            "The weekly feed is missing the passing columns "
            f"{missing} — the sack/INT models need them (check nflsim.data._ALIASES "
            "against the current nflverse release).")

    p = wk[wk["player_id"] == player_id].copy()
    p = p[p["attempts"] > 0]
    if p.empty:
        raise ValueError("No usable passing games for this QB.")
    w = p["season_w"].values
    has_db = "dropbacks" in p.columns and p["dropbacks"].sum() > 0

    att = p["attempts"].values.astype(float)
    db = (p["dropbacks"].values.astype(float) if has_db
          else att + p["sacks"].values.astype(float))   # dropbacks ≈ atts + sacks

    mu_att = D.wmean(att, w)
    var_att = np.average((att - mu_att) ** 2, weights=w) if len(att) > 1 else mu_att
    mu_db = D.wmean(db, w)
    var_db = np.average((db - mu_db) ** 2, weights=w) if len(db) > 1 else mu_db

    sacks = p["sacks"].sum()
    ints = p["interceptions"].sum()
    db_tot = db.sum()
    att_tot = att.sum()

    p_sack = (sacks + SACK_PRIOR_N * lg["sack"]) / (db_tot + SACK_PRIOR_N)
    p_int = (ints + INT_PRIOR_N * lg["intr"]) / (att_tot + INT_PRIOR_N)

    # NGS average time to throw (optional) → sack multiplier
    ttt_mult, ttt_val = 1.0, None
    if ttt_map:
        lg_ttt = ttt_map.get("_LEAGUE_", D.LG_TIME_TO_THROW)
        ttt_val = ttt_map.get(str(player_id))
        if ttt_val is not None and lg_ttt > 0:
            ttt_mult = float(np.clip((ttt_val / lg_ttt) ** TTT_EXPONENT, *TTT_CLIP))

    return dict(
        player_id=player_id, name=p["player_display_name"].iloc[-1], position="QB",
        team=p["recent_team"].iloc[-1], games=int(len(p)),
        mu_att=float(max(mu_att, 1.0)), var_att=float(max(var_att, mu_att)),
        mu_db=float(max(mu_db, 1.0)), var_db=float(max(var_db, mu_db)),
        p_sack=float(np.clip(p_sack, 0.005, 0.25)),
        p_int=float(np.clip(p_int, 0.002, 0.12)),
        raw_sack=float(sacks / db_tot) if db_tot > 0 else 0.0,
        raw_int=float(ints / att_tot) if att_tot > 0 else 0.0,
        ttt_val=ttt_val, ttt_mult=float(ttt_mult),
    )


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _nb_params(mean, var):
    mean = max(float(mean), 1e-6)
    var = float(var)
    if var <= mean * 1.05:
        return None, mean
    n = mean * mean / (var - mean)
    return max(n, 1e-3), n / (n + mean)


def _draw_volume(rng, mean, var, n):
    nb_n, param = _nb_params(mean, var)
    if nb_n is None:
        return rng.poisson(param, n)
    return rng.negative_binomial(nb_n, param, n)


# ---------------------------------------------------------------------------
# Sacks
# ---------------------------------------------------------------------------

def simulate_sacks(priors: dict, def_prof: dict | None, lg: dict,
                   n_sims: int = 40000, def_shrink: float = DEFAULT_SACK_DEF_SHRINK,
                   seed: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n = int(n_sims)

    base = priors["p_sack"] * priors["ttt_mult"]
    r_sack = def_prof.get("r_sack", 1.0) if def_prof else 1.0
    rate = D.combine_rate_logodds(base, r_sack, lg["sack"], def_shrink) if def_prof \
        else np.clip(base, 1e-4, 0.6)
    rate = float(np.clip(rate, 1e-4, 0.6))

    dropbacks = np.clip(_draw_volume(rng, priors["mu_db"], priors["var_db"], n), 1, None)
    # game-level rate wobble (log-normal, mean 1) for overdispersion
    wobble = rng.lognormal(-0.5 * SACK_GAME_SIGMA ** 2, SACK_GAME_SIGMA, n)
    game_rate = np.clip(rate * wobble, 1e-4, 0.8)
    sacks = rng.binomial(dropbacks.astype(int), game_rate)

    return dict(count=sacks, dropbacks=dropbacks, rate=rate,
                m_sack=r_sack if def_prof else 1.0,
                adj_rate=float(game_rate.mean()), exp=float(sacks.mean()))


# ---------------------------------------------------------------------------
# Interceptions
# ---------------------------------------------------------------------------

def simulate_ints(priors: dict, def_prof: dict | None, lg: dict,
                  n_sims: int = 40000, def_shrink: float = DEFAULT_INT_DEF_SHRINK,
                  seed: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n = int(n_sims)

    base = priors["p_int"]
    r_int = def_prof.get("r_int", 1.0) if def_prof else 1.0
    rate = D.combine_rate_logodds(base, r_int, lg["intr"], def_shrink) if def_prof \
        else np.clip(base, 1e-4, 0.4)
    rate = float(np.clip(rate, 1e-4, 0.4))

    attempts = np.clip(_draw_volume(rng, priors["mu_att"], priors["var_att"], n), 1, None)
    wobble = rng.lognormal(-0.5 * INT_GAME_SIGMA ** 2, INT_GAME_SIGMA, n)
    game_rate = np.clip(rate * wobble, 1e-4, 0.6)
    ints = rng.binomial(attempts.astype(int), game_rate)

    return dict(count=ints, attempts=attempts, rate=rate,
                m_int=r_int if def_prof else 1.0,
                adj_rate=float(game_rate.mean()), exp=float(ints.mean()))


# ---------------------------------------------------------------------------
# Shared summary (works for either count sim)
# ---------------------------------------------------------------------------

def summarize(sim: dict, line: float = 0.5) -> dict:
    c = sim["count"]
    maxk = int(min(c.max(), 5)) if c.max() > 0 else 1
    dist = {str(k): float((c == k).mean()) for k in range(maxk)}
    dist[f"{maxk}+"] = float((c >= maxk).mean())
    p_over = float((c > line).mean())
    return dict(
        mean=float(c.mean()), median=float(np.median(c)),
        p_1plus=float((c >= 1).mean()), p_2plus=float((c >= 2).mean()),
        line=float(line), p_over=p_over, p_under=1 - p_over,
        fair_over_odds=D.american(p_over), fair_under_odds=D.american(1 - p_over),
        dist=dist, rate=float(sim["rate"]),
    )


if __name__ == "__main__":
    seasons = (2024, 2025)
    print("Loading data...")
    wk = D.load_weekly(seasons)
    lg = league_pass_rates(wk)
    dpr = D.def_pass_rates(wk)
    ttt = D.ngs_time_to_throw(D.load_ngs_pass(seasons))
    print(f"league sack rate {lg['sack']:.3f} / dropback | INT rate {lg['intr']:.3f} / att | "
          f"NGS QBs: {len(ttt)-1}")

    qbs = D.list_players(wk[wk['position'] == 'QB'], stat="attempts", min_vol=150)
    for _, row in qbs.head(4).iterrows():
        pri = qb_priors(wk, row["player_id"], lg, ttt)
        dp = dpr.get("SF")
        ss = summarize(simulate_sacks(pri, dp, lg, seed=2), 1.5)
        si = summarize(simulate_ints(pri, dp, lg, seed=2), 0.5)
        ttt_txt = f"{pri['ttt_val']:.2f}s" if pri["ttt_val"] is not None else "n/a"
        print(f"\n{pri['name']} ({pri['team']}, {pri['games']}g) vs SF  "
              f"ttt={ttt_txt} x{pri['ttt_mult']:.2f}")
        print(f"  sacks: E={ss['mean']:.2f} rate={ss['rate']:.3f} "
              f"P(2+)={ss['p_2plus']:.1%} fair(o1.5) {ss['fair_over_odds']}")
        print(f"  ints:  E={si['mean']:.2f} rate={si['rate']:.3f} "
              f"anytime={si['p_1plus']:.1%} fair {si['fair_over_odds']}")
