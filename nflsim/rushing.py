"""
nflsim/rushing.py — Monte Carlo model for a player's rushing yards, built from
the *mechanics* of a run rather than a single yards-per-carry number.

Each carry now resolves into one of three outcomes, and the defense shapes both
how OFTEN each happens and how FAR the back gets:

  * STUFFED run  — stopped at or behind the line (TFL / no gain). A defense that
    stuffs runs does it more often (pbp `r_stuff`). This is the negative left
    tail, and it is now defense-driven rather than a fixed shift constant.
  * NORMAL run   — the bread-and-butter gain, decomposed the receiving-model way:
        gain = yards BEFORE contact (front, pbp/PFR `r_ybc`)
             + yards AFTER contact  (tackling, PFR `r_yac`, + broken tackles)
    and scaled gently by the defense's overall run EFFICIENCY allowed
    (success rate / EPA, pbp `r_eff`).
  * EXPLOSIVE run — a breakaway (10+ yds). A soft defense surrenders these more
    often (pbp `r_expl`); this is the fat right tail, separate from the player's
    own broken-tackle skill.

The three outcome rates form a DECOMPOSITION of the player's real yards-per-carry
(against a league-average defense the simulated mean reproduces the player's YPC),
so the new factors reshape the distribution without inflating it. A big-play back
gets a fatter explosive rate and fewer stuffs from his own YPC even before the
opponent is applied.

Every per-attempt input keeps its OWN game-to-game variance estimated from the
player's real games. Advanced YBC/YAC/broken-tackle inputs come from PFR; the
stuff / explosive / efficiency factors come from play-by-play (nflverse). Any
feed that is missing degrades to a neutral (league-average) factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D

# Per-carry yardage shape constants.
YBC_SHIFT = 1.0          # small before-contact floor on NORMAL carries (stuffs
                         # are now their own bucket, so this no longer carries
                         # the whole negative tail)
YBC_CV = 0.90            # per-carry spread of before-contact yards
YAC_CV = 1.00            # per-carry spread of baseline after-contact yards
BRK_BONUS = 6.0          # mean extra YAC yards from a broken tackle
BRK_CV = 0.90            # spread of that broken-tackle bonus

# Outcome-bucket constants.
LG_YPC = 4.3             # league-typical yards per carry (base-rate anchor)
STUFF_RATE_LG = 0.19     # league share of runs stopped at/behind the line
EXPL_RATE_LG = 0.11      # league share of runs going 10+ yards
STUFF_GAIN = -1.1        # mean yards on a stuffed carry (small loss)
STUFF_CV = 0.75          # spread of the stuffed-carry loss
EXPL_BONUS = 10.0        # mean breakaway yards ADDED on an explosive carry
EXPL_CV = 0.80           # spread of the breakaway bonus
EFF_SHRINK_SCALE = 0.6   # efficiency factor overlaps YBC/YAC → apply it softer

FALLBACK_SHARE_SD = 0.06
FALLBACK_YPC = 4.2
YBC_FRACTION = 0.60      # fallback split of YPC into before/after contact
LG_YBC_SD = 0.6          # fallback game-to-game SDs when no PFR history
LG_YAC_SD = 0.5          # (between-game, net of per-carry noise)
MIN_SHARE_SD = 0.03      # floors on the between-game spread (data.between_sd)
MIN_YBC_SD = 0.4
MIN_YAC_SD = 0.3
LG_BRK_RATE = 0.06

BRK_PRIOR_N = 60.0       # carries-worth of regression on broken-tackle rate


# ---------------------------------------------------------------------------
# Team rush volume
# ---------------------------------------------------------------------------

def team_rush_volume(wk: pd.DataFrame) -> dict:
    """Recency-weighted mean & std of total team carries per game, keyed by
    team (plus league)."""
    tg = team_game_volume(wk, "carries")
    out = {team: (D.wmean(grp["vol"], grp["w"]), D.wstd(grp["vol"], grp["w"], 4.0))
           for team, grp in tg.groupby("recent_team")}
    out["_LEAGUE_"] = (D.wmean(tg["vol"], tg["w"]), D.wstd(tg["vol"], tg["w"], 4.0))
    return out


def team_game_volume(wk: pd.DataFrame, col: str) -> pd.DataFrame:
    """One row per team-game: the team's total of `col` as `vol`, and the
    game's recency weight `w` (1.0 if the frame is unweighted)."""
    if "w" not in wk.columns:
        wk = wk.assign(w=1.0)
    return (wk.groupby(["recent_team", "season", "week"], as_index=False)
              .agg(vol=(col, "sum"), w=("w", "first")))


# ---------------------------------------------------------------------------
# PFR advanced aggregates (per player, with variance) + league means
# ---------------------------------------------------------------------------

def pfr_rush_aggregates(pfr: pd.DataFrame) -> tuple[dict, dict]:
    """Per-gsis-player means & game-to-game SDs of YBC/att, YAC/att, broken-tackle
    rate; plus league means for regression / fallback."""
    lg_default = dict(mu_ybc=FALLBACK_YPC * YBC_FRACTION,
                      mu_yac=FALLBACK_YPC * (1 - YBC_FRACTION), brk=LG_BRK_RATE)
    if pfr is None or pfr.empty or "gsis_id" not in pfr.columns:
        return {}, lg_default
    if "w" not in pfr.columns:
        pfr = pfr.assign(w=1.0)
    have = pfr.dropna(subset=["gsis_id"])
    if have.empty:
        return {}, lg_default
    # Per-game means and SDs are recency-weighted; `car` is the weighted carry
    # total the broken-tackle regression acts on.
    agg = {}
    for gid, g in have.groupby("gsis_id"):
        w = g["w"].values
        car = g["carries"].values.astype(float)
        mu_ybc, mu_yac = D.wmean(g["ybc_att"], w), D.wmean(g["yac_att"], w)
        # per-game SDs net of the per-carry Gamma noise the simulator draws
        agg[gid] = dict(
            pfr_games=int(len(g)), car=float((car * w).sum()),
            mu_ybc=mu_ybc,
            sd_ybc=D.between_sd(g["ybc_att"], w, (YBC_CV * max(mu_ybc, 0.5)) ** 2 / car,
                                MIN_YBC_SD, LG_YBC_SD),
            mu_yac=mu_yac,
            sd_yac=D.between_sd(g["yac_att"], w, (YAC_CV * max(mu_yac, 0.5)) ** 2 / car,
                                MIN_YAC_SD, LG_YAC_SD),
            brk=D.wmean(g["brk_rate"], w),
        )
    wcar = float((pfr["carries"] * pfr["w"]).sum())
    lg = dict(
        mu_ybc=float((pfr["rushing_yards_before_contact"] * pfr["w"]).sum() / wcar),
        mu_yac=float((pfr["rushing_yards_after_contact"] * pfr["w"]).sum() / wcar),
        brk=float((pfr["rushing_broken_tackles"] * pfr["w"]).sum() / wcar),
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
    w = p["w"].values

    team_car = (wk.groupby(["recent_team", "season", "week"])["carries"]
                  .sum().rename("team_car").reset_index())
    p = p.merge(team_car, on=["recent_team", "season", "week"], how="left")
    share = (p["carries"] / p["team_car"]).clip(0, 1).values
    mu_share = D.wmean(share, w)
    # Poisson noise on the carries behind a share: var(share) ~ share / team carries
    sd_share = D.between_sd(share, w, np.clip(mu_share, 0.01, None) / np.clip(p["team_car"].values, 5, None),
                            MIN_SHARE_SD, FALLBACK_SHARE_SD)
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
# Rush-defense profiles: PFR (front / tackling / broken tackles) merged with
# pbp (stuff / explosive / efficiency)
# ---------------------------------------------------------------------------

def _pfr_defense(wk: pd.DataFrame, pfr: pd.DataFrame | None) -> dict:
    """PFR-based front/tackling/broken-tackle ratios (the original three)."""
    if pfr is None or pfr.empty:
        rb = wk[wk["position"].isin(["RB", "FB"])]
        rb = rb.assign(_w=rb["w"] if "w" in rb.columns else 1.0)
        d = (rb.assign(w_car=rb["carries"] * rb["_w"], w_ry=rb["rushing_yards"] * rb["_w"])
               .groupby("opponent_team", as_index=False)
               .agg(car=("w_car", "sum"), ry=("w_ry", "sum"), raw_car=("carries", "sum")))
        d = d[d["raw_car"] >= 40]
        d["ypc"] = d["ry"] / d["car"]
        lg = float(d["ry"].sum() / d["car"].sum())
        return {r["opponent_team"]: dict(r_ybc=float(r["ypc"] / lg), r_yac=float(r["ypc"] / lg),
                                         r_brk=1.0, ybc_allowed=float(r["ypc"]) * YBC_FRACTION,
                                         yac_allowed=float(r["ypc"]) * (1 - YBC_FRACTION),
                                         lg_ybc=lg * YBC_FRACTION, lg_yac=lg * (1 - YBC_FRACTION))
                for _, r in d.iterrows()}

    # Weighted totals give the rates; raw carries guard the sample size.
    x = pfr.assign(_w=pfr["w"] if "w" in pfr.columns else 1.0)
    d = (x.assign(w_car=x["carries"] * x["_w"],
                  w_ybc=x["rushing_yards_before_contact"] * x["_w"],
                  w_yac=x["rushing_yards_after_contact"] * x["_w"],
                  w_brk=x["rushing_broken_tackles"] * x["_w"])
          .groupby("opponent", as_index=False)
          .agg(car=("w_car", "sum"), ybc=("w_ybc", "sum"), yac=("w_yac", "sum"),
               brk=("w_brk", "sum"), raw_car=("carries", "sum")))
    d = d[d["raw_car"] >= 60]
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


_NEUTRAL_PBP = dict(r_stuff=1.0, r_expl=1.0, r_eff=1.0)


def rush_defense_profiles(wk: pd.DataFrame, pfr: pd.DataFrame | None = None,
                          pbp: pd.DataFrame | None = None) -> dict:
    """Per defense: the three PFR factors (front / tackling / broken tackles)
    merged with the three pbp factors (stuff / explosive / efficiency).

    A team missing from either feed keeps neutral (league-average) values for
    that feed's factors, so the simulator always has a complete profile.
    """
    base = _pfr_defense(wk, pfr)
    pbp_prof = D.rush_defense_pbp(pbp) if pbp is not None else {}

    teams = set(base) | set(pbp_prof)
    out = {}
    for t in teams:
        prof = dict(base.get(t, dict(r_ybc=1.0, r_yac=1.0, r_brk=1.0)))
        prof.update(pbp_prof.get(t, dict(_NEUTRAL_PBP)))
        prof.setdefault("has_pbp", t in pbp_prof)
        out[t] = prof
    return out


def rush_scheme_label(prof: dict) -> str:
    parts = []
    r_ybc = prof.get("r_ybc", 1.0)
    parts.append("soft front — backs reach the second level" if r_ybc > 1.06
                 else "stout front — stuffs runs at the line" if r_ybc < 0.94
                 else "average front")
    r_yac = prof.get("r_yac", 1.0)
    parts.append("tackles poorly (gives up YAC)" if r_yac > 1.06
                 else "tackles well (limits YAC)" if r_yac < 0.94
                 else "average tackling")
    if prof.get("has_pbp"):
        r_stuff = prof.get("r_stuff", 1.0)
        parts.append("stuffs runs often" if r_stuff > 1.08
                     else "rarely stuffs runs" if r_stuff < 0.92
                     else "average stuff rate")
        r_expl = prof.get("r_expl", 1.0)
        parts.append("gives up big runs" if r_expl > 1.10
                     else "limits big runs" if r_expl < 0.90
                     else "average explosive rate")
        r_eff = prof.get("r_eff", 1.0)
        parts.append("easy yards (soft overall)" if r_eff > 1.05
                     else "stingy overall" if r_eff < 0.95
                     else "average efficiency")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _player_base_rates(mu_ypc: float) -> tuple[float, float]:
    """A back's own stuff / explosive base rates, scaled off his YPC (better
    backs are stuffed less and break more) before any defense is applied."""
    ypc = float(np.clip(mu_ypc, 2.0, 7.0))
    p_stuff = STUFF_RATE_LG * float(np.clip(LG_YPC / ypc, 0.6, 1.6))
    p_expl = EXPL_RATE_LG * float(np.clip(ypc / LG_YPC, 0.5, 1.8))
    # keep the positive/normal bucket dominant
    if p_stuff + p_expl > 0.55:
        scale = 0.55 / (p_stuff + p_expl)
        p_stuff *= scale
        p_expl *= scale
    return p_stuff, p_expl


def _def_multipliers(def_prof: dict | None, def_shrink: float) -> tuple:
    """The six run-defense factors, each shrunk toward 1 (= no effect)."""
    if def_prof is None:
        return 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    return (
        D.shrink(def_prof.get("r_ybc", 1.0), def_shrink),
        D.shrink(def_prof.get("r_yac", 1.0), def_shrink),
        D.shrink(def_prof.get("r_brk", 1.0), def_shrink),
        D.shrink(def_prof.get("r_stuff", 1.0), def_shrink),
        D.shrink(def_prof.get("r_expl", 1.0), def_shrink),
        # efficiency overlaps YBC/YAC, so apply it more softly
        D.shrink(def_prof.get("r_eff", 1.0), def_shrink * EFF_SHRINK_SCALE),
    )


def yards_from_carries(rng, priors: dict, carries, def_prof: dict | None = None,
                       def_shrink: float = D.DEFAULT_DEF_SHRINK) -> dict:
    """Rushing yards for a carry count the CALLER has already decided.

    The rushing page draws its own carries from team volume x share; the game
    engine allocates them out of the simulated drive sequence instead. Both go
    through this function, so the yardage mechanics are identical either way.
    """
    carries = np.asarray(carries).astype(int)
    n = int(len(carries))
    m_ybc, m_yac, m_brk, m_stuff, m_expl, m_eff = _def_multipliers(def_prof, def_shrink)

    # outcome-bucket rates: the back's own base rates → defense-adjusted rates
    p_stuff0, p_expl0 = _player_base_rates(priors["mu_ypc"])
    p_stuff = float(np.clip(p_stuff0 * m_stuff, 0.02, 0.5))
    p_expl = float(np.clip(p_expl0 * m_expl, 0.01, 0.4))

    # The three buckets are a DECOMPOSITION of the player's YPC. Solve the normal
    # bucket's mean so that, at the player's OWN base rates, the mix reproduces
    # his YPC — then defense reshapes it via the adjusted rates and multipliers.
    T = priors["mu_ybc"] + priors["mu_yac"]           # player's per-carry mean
    p_norm0 = max(1.0 - p_stuff0 - p_expl0, 0.2)
    expl_mean0 = T + EXPL_BONUS                        # a breakaway ≈ normal + bonus
    normal_mean = (T - p_stuff0 * STUFF_GAIN - p_expl0 * expl_mean0) / p_norm0
    normal_mean = float(np.clip(normal_mean, 1.0, 8.0))
    # split the normal-bucket mean into before/after contact in the player's ratio
    ybc_frac = float(np.clip(priors["mu_ybc"] / T, 0.2, 0.9)) if T > 0 else YBC_FRACTION
    norm_ybc = normal_mean * ybc_frac
    norm_yac = normal_mean * (1 - ybc_frac)

    # game-level per-attempt means (each with the player's own game-to-game noise)
    game_ybc = np.clip(rng.normal(norm_ybc * m_ybc * m_eff, priors["sd_ybc"], n), 0.1, None)
    game_yac = np.clip(rng.normal(norm_yac * m_yac * m_eff, priors["sd_yac"], n), 0.1, None)
    p_brk = float(np.clip(priors["brk_rate"] * m_brk, 0.0, 0.4))

    # 3. split each game's carries into stuffed / explosive / normal
    n_stuff = rng.binomial(carries, p_stuff)
    rem = carries - n_stuff
    p_expl_cond = float(np.clip(p_expl / max(1.0 - p_stuff, 1e-6), 0.0, 0.9))
    n_expl = rng.binomial(rem, p_expl_cond)
    n_norm = (rem - n_expl).astype(float)
    Ns, Ne = n_stuff.astype(float), n_expl.astype(float)

    # 3a. stuffed carries: small loss each (this is the defense-driven left tail)
    k_s = 1.0 / (STUFF_CV ** 2)
    loss_mean = -STUFF_GAIN + 0.6          # gamma centred so mean gain = STUFF_GAIN
    theta_s = loss_mean / k_s
    stuff_yards = np.where(Ns > 0,
                           Ns * 0.6 - rng.gamma(np.clip(Ns * k_s, 1e-9, None), 1.0) * theta_s,
                           0.0)

    # 3b. normal carries: before-contact + after-contact (+ broken tackles)
    k_ybc = 1.0 / (YBC_CV ** 2)
    theta_ybc = (game_ybc + YBC_SHIFT) / k_ybc
    ybc_total = np.where(n_norm > 0,
                         rng.gamma(np.clip(n_norm * k_ybc, 1e-9, None), 1.0) * theta_ybc,
                         0.0) - n_norm * YBC_SHIFT
    n_brk = rng.binomial(n_norm.astype(int), p_brk).astype(float)
    baseline = np.clip(game_yac - p_brk * BRK_BONUS, 0.2, None)
    k_yac = 1.0 / (YAC_CV ** 2)
    theta_yac = baseline / k_yac
    yac_base = np.where(n_norm > 0,
                        rng.gamma(np.clip(n_norm * k_yac, 1e-9, None), 1.0) * theta_yac, 0.0)
    k_brk = 1.0 / (BRK_CV ** 2)
    theta_brk = BRK_BONUS / k_brk
    yac_brk = np.where(n_brk > 0,
                       rng.gamma(np.clip(n_brk * k_brk, 1e-9, None), 1.0) * theta_brk, 0.0)

    # 3c. explosive carries: a normal-ish gain plus a breakaway bonus
    k_e = 1.0 / (EXPL_CV ** 2)
    theta_e = EXPL_BONUS / k_e
    expl_base = Ne * (game_ybc + game_yac)
    expl_bonus = np.where(Ne > 0,
                          rng.gamma(np.clip(Ne * k_e, 1e-9, None), 1.0) * theta_e, 0.0)
    expl_yards = expl_base + expl_bonus

    yards = np.round(stuff_yards + ybc_total + yac_base + yac_brk + expl_yards, 1)

    return dict(
        yards=yards, carries=carries, broken_tackles=n_brk,
        stuffs=n_stuff, explosives=n_expl,
        adj=dict(m_ybc=m_ybc, m_yac=m_yac, m_brk=m_brk,
                 m_stuff=m_stuff, m_expl=m_expl, m_eff=m_eff),
        rates=dict(p_stuff=p_stuff, p_expl=p_expl, p_brk=p_brk),
        exp_broken=float(n_brk.mean()),
        has_pbp=bool(def_prof.get("has_pbp")) if def_prof else False,
    )


def simulate(priors: dict, team_vol: tuple[float, float], def_prof: dict | None,
             n_sims: int = 20000, def_shrink: float = D.DEFAULT_DEF_SHRINK,
             seed: int | None = None) -> dict:
    """A single player's rushing game: draw the carries, then fill in the yards."""
    rng = np.random.default_rng(seed)
    n = int(n_sims)
    tmean, tsd = team_vol
    team_car = np.clip(rng.normal(tmean, tsd, n), 8, None)
    a, b = D.beta_params(priors["mu_share"], priors["sd_share"])
    share = rng.beta(a, b, n)
    carries = rng.poisson(np.clip(team_car * share, 0, None))

    out = yards_from_carries(rng, priors, carries, def_prof, def_shrink)
    out["exp_carries"] = float((team_car * share).mean())
    return out


def summarize(sim: dict, line: float | None = None) -> dict:
    y = sim["yards"]
    out = dict(
        mean=float(y.mean()), median=float(np.median(y)), std=float(y.std()),
        p10=float(np.percentile(y, 10)), p25=float(np.percentile(y, 25)),
        p75=float(np.percentile(y, 75)), p90=float(np.percentile(y, 90)),
        mean_carries=float(sim["carries"].mean()),
        mean_broken=float(sim["broken_tackles"].mean()),
        mean_stuffs=float(sim["stuffs"].mean()),
        mean_explosives=float(sim["explosives"].mean()),
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
    pbp = D.load_pbp(seasons)
    agg, pfr_lg = pfr_rush_aggregates(pfr)
    tv = team_rush_volume(wk)
    defs = rush_defense_profiles(wk, pfr, pbp)
    players = D.list_players(wk, stat="carries", min_vol=50)

    for _, row in players.head(3).iterrows():
        pri = player_rush_priors(wk, row["player_id"], agg, pfr_lg)
        dp = defs.get("SF")
        tvt = tv.get(pri["team"], tv["_LEAGUE_"])
        sim = simulate(pri, tvt, dp, n_sims=40000, seed=1)
        s = summarize(sim, 65.5)
        print(f"\n{pri['name']} ({pri['team']}, {pri['games']}g) vs SF — {pri['adv_source']}")
        print(f"  mean {s['mean']:.1f} yds on {s['mean_carries']:.1f} car | "
              f"median {s['median']:.0f} | 10-90%: {s['p10']:.0f}-{s['p90']:.0f} | "
              f"stuffs {s['mean_stuffs']:.1f} / expl {s['mean_explosives']:.1f} | "
              f"P(>65.5)={s['p_over']:.1%}")
        print(f"  def: {rush_scheme_label(dp)}")
