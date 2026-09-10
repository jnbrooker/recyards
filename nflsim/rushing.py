"""
nflsim/rushing.py — Monte Carlo model for a player's rushing yards, built from
the *mechanics* of a run rather than a single yards-per-carry number.

Each carry's gain is decomposed the way the receiving model splits aDOT + YAC:

    gain = yards BEFORE contact  +  yards AFTER contact (+ broken-tackle bonus)

  * Yards before contact (YBC/att)  — blocking, scheme and the defensive front.
    Modelled as a shifted Gamma so stuffed runs go negative (tackles for loss).
    Nudged by the opponent's run front.
  * Yards after contact (YAC/att)   — the back's own doing. A small baseline
    Gamma PLUS, on the carries where he breaks a tackle, an explosive bonus.
  * Broken tackles                  — drawn per carry (Binomial at the player's
    broken-tackle rate); each one adds a chunk of YAC and creates the fat right
    tail you see in real rushing lines.

Every one of these has its OWN game-to-game variance estimated from the player's
real games (YBC/att ±, YAC/att ±, broken-tackle rate), so the spread is earned
from data, not assumed. Advanced inputs come from PFR via nflverse; if a player
has no PFR history we fall back to splitting their yards-per-carry into a
league-typical before/after mix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D

# Per-carry yardage shape constants.
YBC_SHIFT = 3.0          # before-contact floor: a carry can lose up to ~this
YBC_CV = 0.95            # per-carry spread of before-contact yards
YAC_CV = 1.00            # per-carry spread of baseline after-contact yards
BRK_BONUS = 6.0          # mean extra YAC yards from a broken tackle
BRK_CV = 0.90            # spread of that broken-tackle bonus

FALLBACK_SHARE_SD = 0.06
FALLBACK_YPC = 4.2
YBC_FRACTION = 0.60      # fallback split of YPC into before/after contact
LG_YBC_SD = 1.4          # fallback game-to-game SDs when no PFR history
LG_YAC_SD = 1.0
LG_BRK_RATE = 0.06

BRK_PRIOR_N = 60.0       # carries-worth of regression on broken-tackle rate


# ---------------------------------------------------------------------------
# Team rush volume
# ---------------------------------------------------------------------------

def team_rush_volume(wk: pd.DataFrame) -> dict:
    """Mean & std of total team carries per game, keyed by team (plus league)."""
    tg = (wk.groupby(["recent_team", "season", "week"], as_index=False)
            .agg(team_car=("carries", "sum")))
    out = {}
    for team, grp in tg.groupby("recent_team"):
        out[team] = (float(grp["team_car"].mean()),
                     float(grp["team_car"].std(ddof=1) or 4.0))
    out["_LEAGUE_"] = (float(tg["team_car"].mean()),
                       float(tg["team_car"].std(ddof=1) or 4.0))
    return out


# ---------------------------------------------------------------------------
# PFR advanced aggregates (per player, with variance) + league means
# ---------------------------------------------------------------------------

def pfr_rush_aggregates(pfr: pd.DataFrame) -> tuple[dict, dict]:
    """Per-gsis-player means & game-to-game SDs of YBC/att, YAC/att, broken-tackle
    rate; plus league means for regression / fallback."""
    have = pfr.dropna(subset=["gsis_id"])
    g = (have.groupby("gsis_id")
             .agg(pfr_games=("carries", "count"), car=("carries", "sum"),
                  mu_ybc=("ybc_att", "mean"), sd_ybc=("ybc_att", "std"),
                  mu_yac=("yac_att", "mean"), sd_yac=("yac_att", "std"),
                  brk=("brk_rate", "mean")))
    agg = {}
    for gid, r in g.iterrows():
        agg[gid] = dict(
            pfr_games=int(r["pfr_games"]), car=float(r["car"]),
            mu_ybc=float(r["mu_ybc"]), sd_ybc=float(r["sd_ybc"] if r["sd_ybc"] == r["sd_ybc"] else LG_YBC_SD),
            mu_yac=float(r["mu_yac"]), sd_yac=float(r["sd_yac"] if r["sd_yac"] == r["sd_yac"] else LG_YAC_SD),
            brk=float(r["brk"]),
        )
    lg = dict(
        mu_ybc=float((pfr["rushing_yards_before_contact"].sum() / pfr["carries"].sum())),
        mu_yac=float((pfr["rushing_yards_after_contact"].sum() / pfr["carries"].sum())),
        brk=float(pfr["rushing_broken_tackles"].sum() / pfr["carries"].sum()),
    )
    return agg, lg


# ---------------------------------------------------------------------------
# Player priors
# ---------------------------------------------------------------------------

def player_rush_priors(wk: pd.DataFrame, player_id: str,
                       pfr_agg: dict | None = None,
                       pfr_lg: dict | None = None) -> dict:
    """Carry share (+variance) from the main feed, decomposed per-carry yardage
    (YBC / YAC / broken tackles, each +variance) from PFR when available."""
    p = wk[wk["player_id"] == player_id].copy()
    p = p[p["carries"] > 0]
    if p.empty:
        raise ValueError("No usable rushing games for this player.")
    w = p["season_w"].values

    team_car = (wk.groupby(["recent_team", "season", "week"])["carries"]
                  .sum().rename("team_car").reset_index())
    p = p.merge(team_car, on=["recent_team", "season", "week"], how="left")
    share = (p["carries"] / p["team_car"]).clip(0, 1).values
    mu_share = D.wmean(share, w)
    sd_share = D.wstd(share, w, FALLBACK_SHARE_SD)
    mu_ypc = D.wmean((p["rushing_yards"] / p["carries"]).values, p["carries"].values * w)
    if not np.isfinite(mu_ypc):
        mu_ypc = FALLBACK_YPC

    a = (pfr_agg or {}).get(player_id)
    lg = pfr_lg or dict(mu_ybc=mu_ypc * YBC_FRACTION, mu_yac=mu_ypc * (1 - YBC_FRACTION), brk=LG_BRK_RATE)
    if a is not None and a["pfr_games"] >= 3:
        mu_ybc, sd_ybc = a["mu_ybc"], a["sd_ybc"]
        mu_yac, sd_yac = a["mu_yac"], a["sd_yac"]
        # regress broken-tackle rate toward league
        brk = (a["brk"] * a["car"] + LG_BRK_RATE * BRK_PRIOR_N) / (a["car"] + BRK_PRIOR_N)
        source = f"PFR ({a['pfr_games']} g)"
    else:
        # fallback: split YPC into league-typical before/after contact
        mu_ybc = mu_ypc * YBC_FRACTION
        mu_yac = mu_ypc * (1 - YBC_FRACTION)
        sd_ybc, sd_yac, brk = LG_YBC_SD, LG_YAC_SD, lg["brk"]
        source = "YPC split (no PFR history)"

    return dict(
        player_id=player_id, name=p["player_display_name"].iloc[-1],
        position=p["position"].iloc[-1], team=p["recent_team"].iloc[-1],
        games=int(len(p)),
        mu_share=float(np.clip(mu_share, 0.01, 0.95)), sd_share=float(sd_share),
        mu_ypc=float(mu_ypc),
        mu_ybc=float(np.clip(mu_ybc, 0.5, 6.0)), sd_ybc=float(np.clip(sd_ybc, 0.4, 3.0)),
        mu_yac=float(np.clip(mu_yac, 0.5, 5.0)), sd_yac=float(np.clip(sd_yac, 0.3, 3.0)),
        brk_rate=float(np.clip(brk, 0.0, 0.25)),
        adv_source=source,
    )


# ---------------------------------------------------------------------------
# Rush-defense profiles (advanced: YBC / YAC / broken tackles allowed)
# ---------------------------------------------------------------------------

def rush_defense_profiles(wk: pd.DataFrame, pfr: pd.DataFrame | None = None) -> dict:
    """Per defense: before-contact, after-contact and broken-tackle rate allowed
    to rushers, as ratios vs league. r_ybc>1 = poor front (backs reach the second
    level); r_yac>1 = poor tackling."""
    if pfr is None or pfr.empty:
        # basic fallback: yards-per-carry allowed to RBs
        rb = wk[wk["position"].isin(["RB", "FB"])]
        d = (rb.groupby("opponent_team", as_index=False)
               .agg(car=("carries", "sum"), ry=("rushing_yards", "sum")))
        d = d[d["car"] >= 40]
        d["ypc"] = d["ry"] / d["car"]
        lg = float(d["ry"].sum() / d["car"].sum())
        return {r["opponent_team"]: dict(r_ybc=float(r["ypc"] / lg), r_yac=float(r["ypc"] / lg),
                                         r_brk=1.0, ybc_allowed=float(r["ypc"]) * YBC_FRACTION,
                                         yac_allowed=float(r["ypc"]) * (1 - YBC_FRACTION),
                                         lg_ybc=lg * YBC_FRACTION, lg_yac=lg * (1 - YBC_FRACTION))
                for _, r in d.iterrows()}

    d = (pfr.groupby("opponent", as_index=False)
           .agg(car=("carries", "sum"),
                ybc=("rushing_yards_before_contact", "sum"),
                yac=("rushing_yards_after_contact", "sum"),
                brk=("rushing_broken_tackles", "sum")))
    d = d[d["car"] >= 60]
    d["ybc_att"] = d["ybc"] / d["car"]
    d["yac_att"] = d["yac"] / d["car"]
    d["brk_rate"] = d["brk"] / d["car"]
    lg_ybc = float(d["ybc"].sum() / d["car"].sum())
    lg_yac = float(d["yac"].sum() / d["car"].sum())
    lg_brk = float(d["brk"].sum() / d["car"].sum())
    prof = {}
    for _, r in d.iterrows():
        prof[r["opponent"]] = dict(
            r_ybc=float(r["ybc_att"] / lg_ybc), r_yac=float(r["yac_att"] / lg_yac),
            r_brk=float(r["brk_rate"] / lg_brk) if lg_brk > 0 else 1.0,
            ybc_allowed=float(r["ybc_att"]), yac_allowed=float(r["yac_att"]),
            lg_ybc=lg_ybc, lg_yac=lg_yac)
    return prof


def rush_scheme_label(prof: dict) -> str:
    front = ("soft front — backs reach the second level" if prof["r_ybc"] > 1.06
             else "stout front — stuffs runs at the line" if prof["r_ybc"] < 0.94
             else "average front")
    tackle = ("tackles poorly (gives up YAC)" if prof["r_yac"] > 1.06
              else "tackles well (limits YAC)" if prof["r_yac"] < 0.94
              else "average tackling")
    return f"{front}; {tackle}"


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(priors: dict, team_vol: tuple[float, float], def_prof: dict | None,
             n_sims: int = 20000, def_shrink: float = D.DEFAULT_DEF_SHRINK,
             seed: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n = int(n_sims)

    m_ybc, m_yac, m_brk = 1.0, 1.0, 1.0
    if def_prof is not None:
        m_ybc = D.shrink(def_prof["r_ybc"], def_shrink)
        m_yac = D.shrink(def_prof["r_yac"], def_shrink)
        m_brk = D.shrink(def_prof.get("r_brk", 1.0), def_shrink)

    # 1. carries this game
    tmean, tsd = team_vol
    team_car = np.clip(rng.normal(tmean, tsd, n), 8, None)
    a, b = D.beta_params(priors["mu_share"], priors["sd_share"])
    share = rng.beta(a, b, n)
    carries = rng.poisson(np.clip(team_car * share, 0, None))
    C = carries.astype(float)

    # 2. game-level per-attempt YBC and YAC, each with the player's own variance
    game_ybc = np.clip(rng.normal(priors["mu_ybc"] * m_ybc, priors["sd_ybc"], n), 0.2, None)
    game_yac = np.clip(rng.normal(priors["mu_yac"] * m_yac, priors["sd_yac"], n), 0.2, None)
    p_brk = float(np.clip(priors["brk_rate"] * m_brk, 0.0, 0.4))

    # 3a. before-contact yards: shifted-Gamma sum (allows tackles for loss)
    k_ybc = 1.0 / (YBC_CV ** 2)
    theta_ybc = (game_ybc + YBC_SHIFT) / k_ybc
    ybc_total = np.where(C > 0, rng.gamma(np.clip(C * k_ybc, 1e-9, None), 1.0) * theta_ybc, 0.0) \
        - C * YBC_SHIFT

    # 3b. after-contact: small baseline + broken-tackle explosions
    n_brk = rng.binomial(carries, p_brk).astype(float)
    baseline = np.clip(game_yac - p_brk * BRK_BONUS, 0.3, None)   # keep mean YAC = game_yac
    k_yac = 1.0 / (YAC_CV ** 2)
    theta_yac = baseline / k_yac
    yac_base = np.where(C > 0, rng.gamma(np.clip(C * k_yac, 1e-9, None), 1.0) * theta_yac, 0.0)
    k_brk = 1.0 / (BRK_CV ** 2)
    theta_brk = BRK_BONUS / k_brk
    yac_brk = np.where(n_brk > 0, rng.gamma(np.clip(n_brk * k_brk, 1e-9, None), 1.0) * theta_brk, 0.0)

    yards = np.round(ybc_total + yac_base + yac_brk, 1)

    return dict(
        yards=yards, carries=carries, broken_tackles=n_brk,
        adj=dict(m_ybc=m_ybc, m_yac=m_yac, m_brk=m_brk),
        exp_carries=float((team_car * share).mean()),
        exp_broken=float(n_brk.mean()),
    )


def summarize(sim: dict, line: float | None = None) -> dict:
    y = sim["yards"]
    out = dict(
        mean=float(y.mean()), median=float(np.median(y)), std=float(y.std()),
        p10=float(np.percentile(y, 10)), p25=float(np.percentile(y, 25)),
        p75=float(np.percentile(y, 75)), p90=float(np.percentile(y, 90)),
        mean_carries=float(sim["carries"].mean()),
        mean_broken=float(sim["broken_tackles"].mean()),
    )
    if line is not None:
        p_over = float((y > line).mean())
        out.update(line=float(line), p_over=p_over, p_under=1 - p_over,
                   fair_over_odds=D.american(p_over), fair_under_odds=D.american(1 - p_over))
    return out


if __name__ == "__main__":
    seasons = (2024, 2025)
    print("Loading data (first run downloads a few MB)...")
    wk = D.load_weekly(seasons)
    pfr = D.load_pfr_rush(seasons)
    agg, pfr_lg = pfr_rush_aggregates(pfr)
    tv = team_rush_volume(wk)
    defs = rush_defense_profiles(wk, pfr)
    players = D.list_players(wk, stat="carries", min_vol=50)

    for _, row in players.head(3).iterrows():
        pri = player_rush_priors(wk, row["player_id"], agg, pfr_lg)
        dp = defs.get("SF")
        tvt = tv.get(pri["team"], tv["_LEAGUE_"])
        sim = simulate(pri, tvt, dp, n_sims=40000, seed=1)
        s = summarize(sim, 65.5)
        print(f"\n{pri['name']} ({pri['team']}, {pri['games']}g) vs SF — {pri['adv_source']}")
        print(f"  YBC/att {pri['mu_ybc']:.2f}±{pri['sd_ybc']:.2f} | "
              f"YAC/att {pri['mu_yac']:.2f}±{pri['sd_yac']:.2f} | brk {pri['brk_rate']:.3f}")
        print(f"  def: front x{sim['adj']['m_ybc']:.2f}, tackling x{sim['adj']['m_yac']:.2f} "
              f"({rush_scheme_label(dp)})")
        print(f"  mean {s['mean']:.1f} yds on {s['mean_carries']:.1f} car "
              f"({s['mean_broken']:.1f} broken) | median {s['median']:.0f} | "
              f"10-90%: {s['p10']:.0f}-{s['p90']:.0f} | P(>65.5)={s['p_over']:.1%}")
