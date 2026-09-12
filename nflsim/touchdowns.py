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

# Goal-line role (roadmap 8.2): a carry inside the 10 scores ~29% of the time
# vs ~1% elsewhere, a target inside the 10 ~39% vs ~2.6%. So a player's TD
# rate is his goal-line SHARE of touches times those conversions — and the
# share is measured from touches (tens per season), not from the rare TDs.
# The share is regressed toward the positional mean over GL_PRIOR_N touches.
# When the TD-rate prior is role-based it is trusted over TD_ROLE_PRIOR_N
# touches-worth (vs 40-45 for the positional mean): on the 2025 backtest the
# anytime-TD Brier went 0.2078 -> 0.2063, log-loss 0.605 -> 0.602 and the
# correlation of projected with actual TDs 0.24 -> 0.26; the same shrinkage
# toward the positional mean was worse, so it is the role signal, not the
# shrinkage, that helps.
GL_PRIOR_N = 40.0
TD_ROLE_PRIOR_N = 300.0


# ---------------------------------------------------------------------------
# League positional TD rates (the regression targets)
# ---------------------------------------------------------------------------

def _wsum(g: pd.DataFrame, col: str) -> float:
    """Recency-weighted total of `col` (plain total if the frame is unweighted)."""
    w = g["w"] if "w" in g.columns else 1.0
    return float((g[col] * w).sum())


def goal_line_profiles(touches: pd.DataFrame, wk: pd.DataFrame | None = None) -> dict:
    """From `data.load_touches`: per player the recency-weighted goal-line
    share of his carries and targets (regressed toward his position's mean),
    plus the league conversion rates inside and outside the 10.

    Returns {player_id: dict(gl_car_frac, gl_tgt_frac, car, tgt), "_LEAGUE_":
    dict(p_gl_rush, p_ngl_rush, p_gl_pass, p_ngl_pass, gl_car_frac{pos},
    gl_tgt_frac{pos})}. Empty dict if there are no touches."""
    if touches is None or touches.empty:
        return {}
    t = touches
    w = t["w"]
    car, glc = float((t["car"] * w).sum()), float((t["gl_car"] * w).sum())
    tgt, glt = float((t["tgt"] * w).sum()), float((t["gl_tgt"] * w).sum())
    # conversions inside / outside the 10 (league-wide; TDs come with the touches)
    rtd, ptd = float((t["rush_td"] * w).sum()), float((t["rec_td"] * w).sum())
    # a TD outside the 10 is rare; split the TDs by where the touches were
    # using the league conversion ratio measured directly on the plays
    lg = dict(p_gl_rush=0.29, p_ngl_rush=0.0094, p_gl_pass=0.39, p_ngl_pass=0.026)
    if car > 0 and tgt > 0:
        # solve the two-rate split so the totals reconcile: total = ngl*(n-gl) + gl*g
        # with the inside/outside ratio fixed at the measured 30x / 15x
        k_r, k_p = 0.29 / 0.0094, 0.39 / 0.026
        ngl_r = rtd / ((car - glc) + k_r * glc); ngl_p = ptd / ((tgt - glt) + k_p * glt)
        lg = dict(p_gl_rush=float(k_r * ngl_r), p_ngl_rush=float(ngl_r),
                  p_gl_pass=float(k_p * ngl_p), p_ngl_pass=float(ngl_p))
    pos_of = {}
    if wk is not None and not wk.empty:
        pos_of = wk.drop_duplicates("player_id").set_index(wk.drop_duplicates("player_id")["player_id"].astype(str))["position"].to_dict()
    t = t.assign(pos=t["player_id"].map(pos_of).fillna("_ALL_"))
    pos_frac = {}
    for pos, g in list(t.groupby("pos")) + [("_ALL_", t)]:
        c, gc = float((g["car"] * g["w"]).sum()), float((g["gl_car"] * g["w"]).sum())
        tg, gt = float((g["tgt"] * g["w"]).sum()), float((g["gl_tgt"] * g["w"]).sum())
        pos_frac[pos] = (gc / c if c > 0 else glc / max(car, 1), gt / tg if tg > 0 else glt / max(tgt, 1))
    lg["gl_car_frac"] = {k: v[0] for k, v in pos_frac.items()}
    lg["gl_tgt_frac"] = {k: v[1] for k, v in pos_frac.items()}
    out = {"_LEAGUE_": lg}
    g = t.assign(wc=t["car"] * w, wgc=t["gl_car"] * w, wt=t["tgt"] * w, wgt=t["gl_tgt"] * w) \
         .groupby("player_id")[["wc", "wgc", "wt", "wgt", "pos"]].agg(
             wc=("wc", "sum"), wgc=("wgc", "sum"), wt=("wt", "sum"), wgt=("wgt", "sum"), pos=("pos", "last"))
    for pid, r in g.iterrows():
        pc, pt = pos_frac.get(r["pos"], pos_frac["_ALL_"])
        out[str(pid)] = dict(
            gl_car_frac=float((r["wgc"] + GL_PRIOR_N * pc) / (r["wc"] + GL_PRIOR_N)),
            gl_tgt_frac=float((r["wgt"] + GL_PRIOR_N * pt) / (r["wt"] + GL_PRIOR_N)),
            car=float(r["wc"]), tgt=float(r["wt"]),
            raw_gl_car_frac=float(r["wgc"] / r["wc"]) if r["wc"] > 0 else np.nan,
            raw_gl_tgt_frac=float(r["wgt"] / r["wt"]) if r["wt"] > 0 else np.nan)
    return out


def role_td_rates(pid: str, position: str, gl: dict, catch_rate: float) -> dict | None:
    """Player-specific TD-rate priors from goal-line role: per carry, and per
    RECEPTION (the model's unit), converting the per-target rate with the
    player's catch rate. None if the profile has nothing on him."""
    if not gl or "_LEAGUE_" not in gl:
        return None
    lg = gl["_LEAGUE_"]
    p = gl.get(str(pid))
    if p is None:
        fc = lg["gl_car_frac"].get(position, lg["gl_car_frac"]["_ALL_"])
        ft = lg["gl_tgt_frac"].get(position, lg["gl_tgt_frac"]["_ALL_"])
    else:
        fc, ft = p["gl_car_frac"], p["gl_tgt_frac"]
    per_car = (1 - fc) * lg["p_ngl_rush"] + fc * lg["p_gl_rush"]
    per_tgt = (1 - ft) * lg["p_ngl_pass"] + ft * lg["p_gl_pass"]
    return dict(rush_td_per_car=float(per_car),
                rec_td_per_rec=float(per_tgt / max(catch_rate, 0.3)),
                gl_car_frac=float(fc), gl_tgt_frac=float(ft), from_profile=p is not None)


def league_td_rates(wk: pd.DataFrame) -> dict:
    """League mean rec-TD-per-reception and rush-TD-per-carry, by position
    (recency-weighted)."""
    out = {}
    for pos, g in wk.groupby("position"):
        rec, car = _wsum(g, "receptions"), _wsum(g, "carries")
        out[pos] = dict(
            rec_td_per_rec=float(_wsum(g, "receiving_tds") / rec) if rec > 0 else 0.05,
            rush_td_per_car=float(_wsum(g, "rushing_tds") / car) if car > 0 else 0.025,
        )
    # sensible global fallbacks
    out["_ALL_"] = dict(
        rec_td_per_rec=float(_wsum(wk, "receiving_tds") / max(_wsum(wk, "receptions"), 1e-9)),
        rush_td_per_car=float(_wsum(wk, "rushing_tds") / max(_wsum(wk, "carries"), 1e-9)),
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


def player_td_priors(wk: pd.DataFrame, player_id: str, lg: dict,
                     gl: dict | None = None) -> dict:
    """`gl` is `goal_line_profiles(...)`: when given, the regression target for
    the player's TD rates is his own goal-line role x league conversion
    instead of the positional mean."""
    p = wk[wk["player_id"] == player_id].copy()
    if p.empty:
        raise ValueError("No games for this player.")
    pos = p["position"].iloc[-1]
    w = p["w"].values
    prior = lg.get(pos, lg["_ALL_"])
    role = None
    if gl:
        catch = float(_wsum(p, "receptions") / max(_wsum(p, "targets"), 1e-9)) if _wsum(p, "targets") > 0 else 0.65
        role = role_td_rates(str(player_id), pos, gl, catch)
        if role is not None:
            prior = dict(rec_td_per_rec=role["rec_td_per_rec"], rush_td_per_car=role["rush_td_per_car"])

    # --- expected volume per game (mean & variance for the NB draw) ---
    rec_g = p["receptions"].values
    car_g = p["carries"].values
    mu_rec = D.wmean(rec_g, w) if len(rec_g) else 0.0
    mu_car = D.wmean(car_g, w) if len(car_g) else 0.0
    var_rec = np.average((rec_g - mu_rec) ** 2, weights=w) if len(rec_g) > 1 else mu_rec
    var_car = np.average((car_g - mu_car) ** 2, weights=w) if len(car_g) > 1 else mu_car

    # --- regressed conversion rates (on recency-weighted totals) ---
    rec_tot = _wsum(p, "receptions")
    car_tot = _wsum(p, "carries")
    rec_td = _wsum(p, "receiving_tds")
    rush_td = _wsum(p, "rushing_tds")
    n_rec = TD_ROLE_PRIOR_N if role else REC_TD_PRIOR_N
    n_rush = TD_ROLE_PRIOR_N if role else RUSH_TD_PRIOR_N
    p_rec_td = (rec_td + n_rec * prior["rec_td_per_rec"]) / (rec_tot + n_rec)
    p_rush_td = (rush_td + n_rush * prior["rush_td_per_car"]) / (car_tot + n_rush)

    return dict(
        player_id=player_id, name=p["player_display_name"].iloc[-1], position=pos,
        team=p["recent_team"].iloc[-1], games=int(len(p)),
        mu_rec=float(max(mu_rec, 0.0)), var_rec=float(max(var_rec, mu_rec)),
        mu_car=float(max(mu_car, 0.0)), var_car=float(max(var_car, mu_car)),
        p_rec_td=float(np.clip(p_rec_td, 0.0, 0.4)),
        p_rush_td=float(np.clip(p_rush_td, 0.0, 0.25)),
        raw_rec_td_per_rec=float(rec_td / rec_tot) if rec_tot > 0 else 0.0,
        raw_rush_td_per_car=float(rush_td / car_tot) if car_tot > 0 else 0.0,
        prior_rec_td=float(prior["rec_td_per_rec"]), prior_rush_td=float(prior["rush_td_per_car"]),
        gl_car_frac=role["gl_car_frac"] if role else np.nan,
        gl_tgt_frac=role["gl_tgt_frac"] if role else np.nan,
        role_source="goal-line role" if role else "positional mean",
    )


# ---------------------------------------------------------------------------
# Defense TD-allowed profiles (rec & rush), vs league, per position
# ---------------------------------------------------------------------------

def td_defense_profiles(wk: pd.DataFrame) -> dict:
    """(defense, position) -> ratios of rec-TD/reception and rush-TD/carry
    allowed vs league. >1 = gives up more scores than average."""
    # Weighted totals give the rates; raw counts guard the sample size.
    x = wk.assign(_w=wk["w"] if "w" in wk.columns else 1.0)
    for c in ("receptions", "receiving_tds", "carries", "rushing_tds"):
        x[f"w_{c}"] = x[c] * x["_w"]
    d = (x.groupby(["opponent_team", "position"], as_index=False)
          .agg(rec=("w_receptions", "sum"), rtd=("w_receiving_tds", "sum"),
               car=("w_carries", "sum"), rutd=("w_rushing_tds", "sum"),
               raw_rec=("receptions", "sum"), raw_car=("carries", "sum")))
    lg = {}
    for pos, g in d.groupby("position"):
        lg[pos] = dict(
            rec=float(g["rtd"].sum() / max(g["rec"].sum(), 1e-9)),
            rush=float(g["rutd"].sum() / max(g["car"].sum(), 1e-9)),
        )
    prof = {}
    for _, r in d.iterrows():
        pos = r["position"]; base = lg[pos]
        rec_rate = r["rtd"] / r["rec"] if r["raw_rec"] >= 25 and r["rec"] > 0 else base["rec"]
        rush_rate = r["rutd"] / r["car"] if r["raw_car"] >= 25 and r["car"] > 0 else base["rush"]
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
    dist = {str(k): float((t == k).mean()) for k in range(maxk)}
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
