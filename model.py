"""
model.py — Monte Carlo model for NFL receiving yards.

Everything the dashboard needs lives here:
  * pulling & caching NFL data (nfl_data_py / nflverse)
  * building per-player priors (target share, catch %, aDOT, YAC) with variance
  * building per-defense "allowed" profiles by position (yards & aDOT allowed)
  * running the Monte Carlo simulation for one player vs one defense

The model in one sentence:
  simulate a game's team pass volume -> the player's share of targets ->
  how many are caught -> the yards on each catch, drawing every step from a
  distribution fit to that player's real game-to-game variance, and nudging
  the depth / catch rate / efficiency toward what the opposing defense allows.

You can run this file directly for a quick sanity check:
    python model.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config / knobs
# ---------------------------------------------------------------------------

RECEIVING_POSITIONS = ["WR", "TE", "RB"]

# How hard to lean on a single defense's numbers. 0 = ignore the defense
# entirely, 1 = take its (noisy) season splits at face value. 0.6 is a sane
# middle: one season of defense-vs-position data is a small sample.
DEFAULT_DEF_SHRINK = 0.6

# Fallback game-to-game spread when a player has too few games to estimate it.
FALLBACK_TS_SD = 0.05        # target share std
FALLBACK_CATCH_SD = 0.10     # catch rate std
FALLBACK_ADOT_SD = 3.0       # aDOT std (yards)
FALLBACK_AIR_SD = 1.0        # completed air yards per catch std (yards)

# Floors on the game-to-game (between-game) spread of each rate once the
# sampling noise the simulator draws itself has been removed — see
# `nflsim.data.between_sd`. The raw per-game SDs are mostly sampling noise.
MIN_TS_SD = 0.03
MIN_CATCH_SD = 0.04
MIN_AIR_SD = 0.8

# Regression toward the positional mean (roadmap 8.1b). A player's own history
# is not taken at face value: his target share is blended with the position's
# mean share over `TS_PRIOR_N` games-worth of prior, and his catch rate / air
# yards per catch / YAC per catch over `CATCH_PRIOR_N` targets-worth and
# `EFF_PRIOR_N` receptions-worth. All act on recency-WEIGHTED totals, so an old
# history is regressed harder than a fresh one. Tuned on the 2025 backtest.
TS_PRIOR_N = 3.0
CATCH_PRIOR_N = 25.0
EFF_PRIOR_N = 20.0

# Per-catch yards are very right-skewed (a 5-yard slant vs a 60-yard bomb),
# so we model them with a Gamma. This is the coefficient of variation of a
# single catch's yardage; ~1.1 matches league-wide yards-per-reception spread.
YPR_CV = 1.10

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
# The download, caching and recency weighting live in the shared layer
# (`nflsim.data`), so this page speaks the same data — and the same game
# weights `w` — as every other model in the suite. Only the receiving
# positions are kept here.

from nflsim import data as _D

Recency = _D.Recency
RECENCY_DEFAULT = _D.RECENCY_DEFAULT


def load_weekly(seasons: tuple[int, ...],
                recency: Recency = RECENCY_DEFAULT) -> pd.DataFrame:
    """Regular-season weekly receiving lines for the given seasons, with the
    recency weight `w` on every row (recent seasons and recent games count
    more — see `nflsim.data.game_weights`).

    Seasons that aren't available yet are skipped; if none are available a
    clear error is raised. `wk.attrs["loaded_seasons"]` says what loaded.
    """
    df = _D.load_weekly(tuple(sorted(int(s) for s in seasons)), recency)
    out = df[df["position"].isin(RECEIVING_POSITIONS)].copy()
    out.attrs.update(df.attrs)
    return out


def list_players(wk: pd.DataFrame, min_targets: int = 20) -> pd.DataFrame:
    """Players with enough volume to model, most-targeted first."""
    g = (wk.groupby(["player_id", "player_display_name", "position"],
                    as_index=False)
           .agg(tgt=("targets", "sum"),
                team=("recent_team", "last"),
                games=("week", "count")))
    g = g[g["tgt"] >= min_targets].sort_values("tgt", ascending=False)
    g["label"] = g["player_display_name"] + " (" + g["position"] + ", " + g["team"] + ")"
    return g.reset_index(drop=True)


def list_defenses(wk: pd.DataFrame) -> list[str]:
    return sorted(wk["opponent_team"].dropna().unique().tolist())


# ---------------------------------------------------------------------------
# Priors: turn a player's real games into means + variances
# ---------------------------------------------------------------------------

def _wmean(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not m.any():
        return np.nan
    return np.average(x[m], weights=w[m])


def _wstd(x, w, fallback):
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if m.sum() < 2:
        return fallback
    mu = np.average(x[m], weights=w[m])
    var = np.average((x[m] - mu) ** 2, weights=w[m])
    # small-sample correction
    var *= m.sum() / (m.sum() - 1)
    sd = float(np.sqrt(var))
    return sd if sd > 1e-6 else fallback


# Slot-aware target-share prior (roadmap 8.1b follow-up): a player is
# regressed toward the share of his RANK among his team's receivers, not the
# positional mean — one number for all WRs pulled stars down 10% and scrubs
# up. Ranks are by weighted target share within the player's latest team;
# the slot values are the unconditional shares from `nflsim.roster`.
def _slot_prior(rank: int, position: str) -> float:
    from nflsim import roster as _RO
    table = _RO.TARGET_ROLE_PRIOR
    for r in range(int(rank), 0, -1):
        if (position, r) in table:
            return float(table[(position, r)])
    return 0.02


def league_priors(wk: pd.DataFrame) -> dict:
    """What the player priors are regressed toward: catch rate, completed air
    yards per catch and YAC per catch by position (recency-weighted), plus each
    player's slot-aware target-share prior under "_slot_" (see `_slot_prior`).
    Keyed by position plus "_ALL_"."""
    d = wk[wk["targets"] > 0]
    if "w" not in d.columns:
        d = d.assign(w=1.0)
    # rank each player among his latest team's receivers by weighted share
    full = wk if "w" in wk.columns else wk.assign(w=1.0)
    latest_team = full.sort_values(["season", "week"]).groupby("player_id")["recent_team"].last()
    sh = (full.assign(x=full["target_share"].fillna(0.0) * full["w"])
              .groupby("player_id").agg(x=("x", "sum"), w=("w", "sum"), position=("position", "last")))
    sh["share"] = sh["x"] / sh["w"].clip(lower=1e-9)
    sh["team"] = latest_team
    sh["rank"] = sh.groupby(["team", "position"])["share"].rank(ascending=False, method="first")
    slot = {str(pid): _slot_prior(int(r["rank"]), str(r["position"])) for pid, r in sh.iterrows()}
    yac_col = d.get("receiving_yards_after_catch", pd.Series(0.0, index=d.index))
    out = {}
    for pos, g in list(d.groupby("position")) + [("_ALL_", d)]:
        w = g["w"].values
        tg = float((g["targets"] * g["w"]).sum())
        rec = float((g["receptions"] * g["w"]).sum())
        yac = float((yac_col.loc[g.index] * g["w"]).sum())
        out[pos] = dict(
            ts=float(_wmean(g["target_share"].values, w)) if g["target_share"].notna().any() else 0.13,
            catch=rec / tg if tg > 0 else 0.65,
            air=(float((g["receiving_yards"] * g["w"]).sum()) - yac) / rec if rec > 0 else 5.7,
            yac=yac / rec if rec > 0 else 5.2,
        )
    out["_slot_"] = slot
    return out


def player_priors(wk: pd.DataFrame, player_id: str, lg: dict | None = None) -> dict:
    """Estimate a player's per-game distributions from their game logs.

    `lg` is `league_priors(wk)`; computed here if not supplied (pass it when
    calling in a loop).
    """
    allg = wk[wk["player_id"] == player_id].copy()   # every game he appeared in
    p = allg[allg["targets"] > 0]                    # games with a target (for the rates)
    if p.empty:
        raise ValueError("No usable games for this player.")
    lg = lg or league_priors(wk)
    prior = dict(lg.get(str(p["position"].iloc[-1]), lg["_ALL_"]))
    prior["ts"] = lg.get("_slot_", {}).get(str(player_id), prior["ts"])

    w = p["w"].values                            # recency weight per game
    n_eff = float(allg["w"].sum())               # games-worth of weighted history

    # Target share: the player's (weighted) targets over his team's (weighted)
    # targets across EVERY game he appeared in — a zero-target game while
    # active is a real outcome (the backtest scores it, and a prop would have
    # paid on it), and leaving those out overstated mid-tier receivers' volume
    # by ~15%. Volume-weighted rather than a mean of per-game shares: only
    # share x mean team targets then reproduces his actual targets per game
    # (a per-game mean runs high for a receiver whose share peaks in his
    # team's low-volume games). Same definition as the roster layer.
    team_t = (wk.groupby(["recent_team", "season", "week"])["targets"].sum()
                .rename("team_tgt").reset_index())
    ag = allg.merge(team_t, on=["recent_team", "season", "week"], how="left")
    den = float((ag["team_tgt"].fillna(0.0) * ag["w"]).sum())
    ts = p["target_share"].fillna(0.0).values     # targeted games, for the spread
    mu_ts_raw = float((ag["targets"] * ag["w"]).sum() / den) if den > 0 else _wmean(ts, w)
    mu_ts = ((mu_ts_raw * n_eff + prior["ts"] * TS_PRIOR_N) / (n_eff + TS_PRIOR_N)
             if n_eff + TS_PRIOR_N > 0 else mu_ts_raw)
    # sampling noise on a share of ~T team targets: Poisson on the player's
    # targets -> var(share) ~ share / T
    team_t = np.where(ts > 0, p["targets"].values / np.clip(ts, 1e-6, None), 30.0)
    sd_ts = _D.between_sd(ts, w, np.clip(mu_ts, 0.01, None) / np.clip(team_t, 5, None),
                          MIN_TS_SD, FALLBACK_TS_SD)

    # Catch rate per game.
    catch_g = (p["receptions"] / p["targets"]).clip(0, 1).values
    tg_w = float((p["targets"] * p["w"]).sum())
    mu_catch_raw = _wmean(catch_g, p["targets"].values * w)   # weight by volume
    mu_catch = ((mu_catch_raw * tg_w + prior["catch"] * CATCH_PRIOR_N) / (tg_w + CATCH_PRIOR_N)
                if tg_w + CATCH_PRIOR_N > 0 else mu_catch_raw)
    pc = float(np.clip(mu_catch, 0.05, 0.95))
    sd_catch = _D.between_sd(catch_g, w, pc * (1 - pc) / p["targets"].values,
                             MIN_CATCH_SD, FALLBACK_CATCH_SD)   # net of binomial noise

    # aDOT per game (average depth of target) — kept for display and for the
    # defense's depth ratio, but NOT used as the air yards on a catch: deep
    # targets are the ones that fall incomplete, so completed passes travel
    # ~2 yards less than the average target. Using aDOT as the per-catch air
    # component over-projected every receiver by ~10 yards a game (found by the
    # backtest harness, 2025 out of sample).
    adot_g = (p["receiving_air_yards"] / p["targets"]).values
    mu_adot = _wmean(adot_g, p["targets"].values * w)
    sd_adot = _wstd(adot_g, w, FALLBACK_ADOT_SD)

    # Completed air yards per reception = receiving yards minus YAC, over the
    # catches; YAC per reception is the rest. Both on recency-weighted totals.
    yac_col = p.get("receiving_yards_after_catch", pd.Series(0.0, index=p.index))
    rec_tot = float((p["receptions"] * p["w"]).sum())
    yac_tot = float((yac_col * p["w"]).sum())
    air_tot = float(((p["receiving_yards"] - yac_col) * p["w"]).sum())
    den = rec_tot + EFF_PRIOR_N
    yac_per_rec = (yac_tot + prior["yac"] * EFF_PRIOR_N) / den if den > 0 else prior["yac"]
    mu_air = (air_tot + prior["air"] * EFF_PRIOR_N) / den if den > 0 else prior["air"]
    caught = p[p["receptions"] > 0]
    air_g = ((caught["receiving_yards"] - yac_col.loc[caught.index]) / caught["receptions"]).values
    # per-catch Gamma noise (cv YPR_CV) averaged over the game's catches
    sd_air = _D.between_sd(air_g, caught["w"].values,
                           (YPR_CV * max(mu_air, 1.0)) ** 2 / caught["receptions"].values,
                           MIN_AIR_SD, FALLBACK_AIR_SD)

    return dict(
        player_id=player_id,
        name=p["player_display_name"].iloc[-1],
        position=p["position"].iloc[-1],
        team=p["recent_team"].iloc[-1],
        games=int(len(allg)),
        mu_ts=float(np.clip(mu_ts, 0.01, 0.6)),
        sd_ts=float(sd_ts),
        mu_catch=float(np.clip(mu_catch, 0.3, 0.95)),
        sd_catch=float(sd_catch),
        mu_adot=float(mu_adot),
        sd_adot=float(sd_adot),
        mu_air=float(max(0.0, mu_air)),
        sd_air=float(sd_air),
        yac_per_rec=float(max(0.0, yac_per_rec)),
        # unregressed, for display
        raw_ts=float(mu_ts_raw), raw_catch=float(mu_catch_raw),
        raw_ypr=float((air_tot + yac_tot) / rec_tot) if rec_tot > 0 else np.nan,
        games_eff=n_eff,
    )


def team_pass_volume(wk: pd.DataFrame) -> dict:
    """Mean & std of total team targets per game, keyed by team.

    Total team targets ~= team pass attempts, which sets how many chances the
    player has to be targeted.
    """
    if "w" not in wk.columns:
        wk = wk.assign(w=1.0)
    team_game = (wk.groupby(["recent_team", "season", "week"], as_index=False)
                   .agg(team_tgt=("targets", "sum"), w=("w", "first")))
    out = {}
    for team, grp in team_game.groupby("recent_team"):
        out[team] = (_wmean(grp["team_tgt"], grp["w"]),
                     _wstd(grp["team_tgt"], grp["w"], 5.0))
    out["_LEAGUE_"] = (_wmean(team_game["team_tgt"], team_game["w"]),
                       _wstd(team_game["team_tgt"], team_game["w"], 5.0))
    return out


# ---------------------------------------------------------------------------
# Defense profiles: what each defense allows, by position
# ---------------------------------------------------------------------------

def defense_profiles(wk: pd.DataFrame) -> dict:
    """For each (defense, position): catch rate, aDOT and yards-per-target
    allowed, expressed as a ratio vs the league average for that position.

    Ratio > 1 means "gives up more than average".
    """
    # Weighted totals give the rates; raw targets guard the sample size.
    x = wk.assign(_w=wk["w"] if "w" in wk.columns else 1.0)
    d = (x.assign(w_tgt=x["targets"] * x["_w"], w_rec=x["receptions"] * x["_w"],
                  w_ry=x["receiving_yards"] * x["_w"],
                  w_air=x["receiving_air_yards"] * x["_w"])
          .groupby(["opponent_team", "position"], as_index=False)
          .agg(tgt=("w_tgt", "sum"), rec=("w_rec", "sum"), ry=("w_ry", "sum"),
               air=("w_air", "sum"), raw_tgt=("targets", "sum"),
               games=("week", "count"), w_games=("_w", "sum")))
    d = d[(d["raw_tgt"] >= 30) & (d["tgt"] > 0)]
    d["catch_allowed"] = d["rec"] / d["tgt"]
    d["adot_allowed"] = d["air"] / d["tgt"]
    d["ypt_allowed"] = d["ry"] / d["tgt"]          # yards per target = the key efficiency stat
    d["ypg_allowed"] = d["ry"] / d["w_games"]

    league = (d.groupby("position")
                .agg(catch=("catch_allowed", "mean"),
                     adot=("adot_allowed", "mean"),
                     ypt=("ypt_allowed", "mean"),
                     ypg=("ypg_allowed", "mean"))
                .to_dict("index"))

    prof = {}
    for _, r in d.iterrows():
        pos = r["position"]; lg = league[pos]
        prof[(r["opponent_team"], pos)] = dict(
            catch_allowed=float(r["catch_allowed"]),
            adot_allowed=float(r["adot_allowed"]),
            ypt_allowed=float(r["ypt_allowed"]),
            ypg_allowed=float(r["ypg_allowed"]),
            r_catch=float(r["catch_allowed"] / lg["catch"]),
            r_adot=float(r["adot_allowed"] / lg["adot"]),
            r_ypt=float(r["ypt_allowed"] / lg["ypt"]),
            lg_catch=float(lg["catch"]),
            lg_adot=float(lg["adot"]),
            lg_ypt=float(lg["ypt"]),
            lg_ypg=float(lg["ypg"]),
            games=int(r["games"]),
        )
    return prof


def scheme_label(r_adot: float, r_ypt: float) -> str:
    """A plain-language read on the defense from what it allows."""
    depth = ("lets receivers get downfield" if r_adot > 1.05
             else "keeps everything short/underneath" if r_adot < 0.95
             else "average target depth")
    gen = ("soft — gives up yards" if r_ypt > 1.08
           else "stingy — takes yards away" if r_ypt < 0.92
           else "roughly league-average")
    return f"{gen}; {depth}"


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def _beta_params(mean, sd):
    """Beta(a,b) matched to a mean and std, guarded to stay valid."""
    mean = float(np.clip(mean, 1e-3, 1 - 1e-3))
    max_sd = np.sqrt(mean * (1 - mean)) * 0.99
    sd = float(np.clip(sd, 1e-4, max_sd))
    v = sd ** 2
    k = mean * (1 - mean) / v - 1
    return max(mean * k, 0.1), max((1 - mean) * k, 0.1)


def _shrink(ratio, shrink):
    """Pull a defense ratio toward 1 (no effect) by (1 - shrink)."""
    return 1.0 + shrink * (ratio - 1.0)


# ---------------------------------------------------------------------------
# The simulation
# ---------------------------------------------------------------------------

def simulate(priors: dict,
             team_vol: tuple[float, float],
             def_prof: dict | None,
             n_sims: int = 20000,
             def_shrink: float = DEFAULT_DEF_SHRINK,
             seed: int | None = None) -> dict:
    """Run n_sims games and return the receiving-yards distribution.

    priors    : output of player_priors()
    team_vol  : (mean, sd) of the player's team's targets per game
    def_prof  : output of defense_profiles()[(opp, pos)], or None to skip the
                defense adjustment entirely
    """
    rng = np.random.default_rng(seed)
    n = int(n_sims)

    # --- defense adjustment factors (all default to 1.0 = neutral) ----------
    m_adot, m_catch, m_yac_env = 1.0, 1.0, 1.0
    if def_prof is not None:
        m_adot = _shrink(def_prof["r_adot"], def_shrink)
        m_catch = _shrink(def_prof["r_catch"], def_shrink)
        # Residual: the yards-per-target a defense allows beyond what its catch
        # rate and depth explain lands on YAC / efficiency.
        residual = def_prof["r_ypt"] / (def_prof["r_catch"] * def_prof["r_adot"])
        m_yac_env = float(np.clip(_shrink(residual, def_shrink), 0.6, 1.6))

    # --- 1. team pass volume for the game -----------------------------------
    tmean, tsd = team_vol
    team_tgt = rng.normal(tmean, tsd, n)
    team_tgt = np.clip(team_tgt, 10, None)

    # --- 2. the player's share of those targets -----------------------------
    a_ts, b_ts = _beta_params(priors["mu_ts"], priors["sd_ts"])
    ts = rng.beta(a_ts, b_ts, n)
    exp_targets = team_tgt * ts
    targets = rng.poisson(np.clip(exp_targets, 0, None))     # integer targets

    # --- 3. how many are caught ---------------------------------------------
    catch_mu = float(np.clip(priors["mu_catch"] * m_catch, 0.05, 0.98))
    a_c, b_c = _beta_params(catch_mu, priors["sd_catch"])
    catch_rate = rng.beta(a_c, b_c, n)
    receptions = rng.binomial(targets, catch_rate)

    # --- 4. yards on the catches --------------------------------------------
    # Game-level completed air yards per catch, nudged by the defense's depth
    # ratio. (Older priors dicts carry only aDOT; fall back to it.)
    mu_air = priors.get("mu_air", priors["mu_adot"])
    sd_air = priors.get("sd_air", priors["sd_adot"])
    air_g = rng.normal(mu_air * m_adot, sd_air, n)
    air_component = np.clip(air_g, 0.0, None)                # air yards per catch
    yac_component = priors["yac_per_rec"] * m_yac_env
    ypr_mean = np.clip(air_component + yac_component, 1.0, None)  # mean yards per catch

    # Total yards = sum of `receptions` iid Gamma(mean=ypr_mean, cv=YPR_CV).
    # Sum of Gammas with shape k each is Gamma(shape = R*k). Vectorized, exact.
    k = 1.0 / (YPR_CV ** 2)                 # per-catch shape
    theta = ypr_mean / k                    # per-catch scale (varies by sim)
    total_shape = receptions * k
    yards = np.where(total_shape > 0,
                     rng.gamma(np.clip(total_shape, 1e-9, None), 1.0) * theta,
                     0.0)
    yards = np.round(yards, 1)

    return dict(
        yards=yards,
        targets=targets,
        receptions=receptions,
        adj=dict(m_adot=m_adot, m_catch=m_catch, m_yac_env=m_yac_env),
        exp_targets=float(exp_targets.mean()),
    )


def summarize(sim: dict, line: float | None = None) -> dict:
    y = sim["yards"]
    out = dict(
        mean=float(y.mean()),
        median=float(np.median(y)),
        std=float(y.std()),
        p10=float(np.percentile(y, 10)),
        p25=float(np.percentile(y, 25)),
        p75=float(np.percentile(y, 75)),
        p90=float(np.percentile(y, 90)),
        mean_targets=float(sim["targets"].mean()),
        mean_receptions=float(sim["receptions"].mean()),
    )
    if line is not None:
        p_over = float((y > line).mean())
        out["line"] = float(line)
        out["p_over"] = p_over
        out["p_under"] = 1.0 - p_over
        # fair American odds for the over
        out["fair_over_odds"] = _american(p_over)
        out["fair_under_odds"] = _american(1 - p_over)
    return out


def _american(p: float) -> str:
    p = min(max(p, 1e-4), 1 - 1e-4)
    dec = 1.0 / p
    if dec >= 2.0:
        return f"+{round((dec - 1) * 100)}"
    return f"-{round(100 / (dec - 1))}"


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seasons = (2023, 2024)
    print("Loading data (first run downloads a few MB)...")
    wk = load_weekly(seasons)
    tv = team_pass_volume(wk)
    defs = defense_profiles(wk)

    players = list_players(wk)
    row = players.iloc[0]
    pri = player_priors(wk, row["player_id"])
    print("\nPlayer:", pri["name"], pri["position"], pri["team"],
          f"({pri['games']} games)")
    print(f"  target share {pri['mu_ts']:.1%} ± {pri['sd_ts']:.1%} | "
          f"catch {pri['mu_catch']:.1%} ± {pri['sd_catch']:.1%} | "
          f"aDOT {pri['mu_adot']:.1f} ± {pri['sd_adot']:.1f} | "
          f"YAC/rec {pri['yac_per_rec']:.1f}")

    opp = "SF"
    dp = defs.get((opp, pri["position"]))
    tv_team = tv.get(pri["team"], tv["_LEAGUE_"])
    sim = simulate(pri, tv_team, dp, n_sims=40000, seed=1)
    line = 65.5
    s = summarize(sim, line)
    print(f"\nvs {opp} defense — {scheme_label(dp['r_adot'], dp['r_ypt'])}")
    print(f"  adj: aDOT x{sim['adj']['m_adot']:.2f}, catch x{sim['adj']['m_catch']:.2f}, "
          f"YAC-env x{sim['adj']['m_yac_env']:.2f}")
    print(f"  mean {s['mean']:.1f} | median {s['median']:.1f} yds | "
          f"{s['mean_receptions']:.1f} rec on {s['mean_targets']:.1f} tgt")
    print(f"  P(over {line}) = {s['p_over']:.1%}  (fair {s['fair_over_odds']})")
    print("  band 10-90%:", round(s["p10"]), "-", round(s["p90"]))
