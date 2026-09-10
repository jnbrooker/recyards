"""
nflsim/game.py — the drive-based game engine (roadmap §4.2–4.4).

Two depth charts in, a simulated offensive box score out. Each simulation runs
top-down, so the box score can never disagree with the scoreboard:

  1. **Pace.** Draw how many drives the game has. Both teams get essentially the
     same number of possessions, so it is one shared draw.
  2. **Drives.** Each drive resolves to touchdown / field goal / turnover /
     nothing, using the matchup rates from the team-strength layer (§4.1) —
     which are already calibrated so their point value equals the efficiency
     rating. Add the defense's own scores and the return/safety residual, and
     the game has a final score.
  3. **Game script.** Now that the margin is known, split the team's plays into
     dropbacks and carries: a team that is simulated to trail throws more, a
     team that leads runs more. This is the shared state the roadmap asks for —
     volume, pass/run split and points all come from the same simulation.
  4. **Allocation.** Dirichlet shares spread targets and carries across the
     players actually listed on the depth chart, then each player's own priors
     fill in catches and yards (the same yardage mechanics the single-stat
     pages use). Team touchdowns are split across players by a multinomial over
     role weights, so player touchdowns always sum to the team's.

**Depth chart to role (§4.4)** is resolved the way the rest of this codebase
resolves everything: a player's share comes from HIS OWN history when he has
enough of it, and falls back to a positional-rank prior (WR1 / WR2 / RB1 …) when
he does not. The depth chart decides who is on the field and in what slot; the
history decides what that slot is worth. So a rookie WR1 inherits the role's
prior, while a veteran WR3 who really commands targets keeps his own number.

Not modelled: an explicit clock, field position, or drive-by-drive sequencing —
drives are exchangeable within a game. Sacks and interceptions come from the
Phase 4 models and are consistent with the drive engine's turnover rate in
expectation, but are not linked to it play by play.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import rushing as R
from . import qb as Q
from . import teams as T

# ---------------------------------------------------------------------------
# Role priors: what a depth-chart slot is worth when a player has no history.
# Shares of team targets / team carries; they are renormalised per roster.
# ---------------------------------------------------------------------------

TARGET_ROLE_PRIOR = {
    ("WR", 1): 0.22, ("WR", 2): 0.16, ("WR", 3): 0.10, ("WR", 4): 0.05, ("WR", 5): 0.02,
    ("TE", 1): 0.17, ("TE", 2): 0.05, ("TE", 3): 0.02,
    ("RB", 1): 0.13, ("RB", 2): 0.07, ("RB", 3): 0.03, ("RB", 4): 0.01,
    ("FB", 1): 0.02,
}
CARRY_ROLE_PRIOR = {
    ("RB", 1): 0.50, ("RB", 2): 0.23, ("RB", 3): 0.09, ("RB", 4): 0.03,
    ("QB", 1): 0.09, ("QB", 2): 0.02,
    ("FB", 1): 0.03,
    ("WR", 1): 0.02, ("WR", 2): 0.01, ("WR", 3): 0.01,
    ("TE", 1): 0.005,
}

# How fast a player's own history takes over from the role prior (games).
ROLE_BLEND_N = 10.0

# Dirichlet concentration on the allocated shares: higher = less week-to-week
# wobble in who gets the ball.
TARGET_CONCENTRATION = 45.0
CARRY_CONCENTRATION = 30.0

# Receiving fallbacks / regression.
LG_CATCH_RATE = 0.645
LG_YPT = 7.6
CATCH_PRIOR_N = 25.0        # targets-worth of prior on catch rate
YPT_PRIOR_N = 30.0          # targets-worth of prior on yards per target
YPC_CV = 1.05               # per-catch yardage spread

# Touchdown role weighting.
TD_RATE_PRIOR_N = 30.0
LG_PASS_TD_SHARE = 0.62     # share of offensive TDs that come through the air

# How pace feeds through to scoring and to volume. Both are MEASURED on the
# 2024-25 drive table rather than assumed, and they are very different:
#
#   * Points per team-game barely move with the number of drives (corr -0.04;
#     teams with 8 drives averaged 21.6 points, teams with 13 averaged 18.8).
#     Extra drives are extra three-and-outs, not extra scoring, so the per-drive
#     scoring rate is scaled DOWN when a simulation draws a lot of drives.
#     Treating drives as a plain multiplier on points inflates the score
#     distribution badly and invents a positive correlation between the two
#     teams' scores that real games do not have.
#   * Plays DO grow with drives, but sublinearly (log-log elasticity +0.34).
PACE_POINTS_ELASTICITY = 0.0     # expected points are pace-invariant
PACE_VOLUME_ELASTICITY = 0.344   # plays ~ drives ** 0.344
PACE_SCALE_CLIP = (0.55, 1.8)

# Game script: extra pass share per point of deficit.
GAME_SCRIPT_BETA = 0.006
PASS_FRAC_CLIP = (0.25, 0.80)

XP_RATE = 0.96              # so a touchdown averages T.TD_POINTS


# ---------------------------------------------------------------------------
# Team volume
# ---------------------------------------------------------------------------

def team_pass_volume(wk: pd.DataFrame) -> dict:
    """Mean & std of team DROPBACKS per game (attempts + sacks), by team."""
    col = "dropbacks" if "dropbacks" in wk.columns else "attempts"
    tg = (wk.groupby(["recent_team", "season", "week"], as_index=False)
            .agg(db=(col, "sum")))
    out = {}
    for team, grp in tg.groupby("recent_team"):
        out[team] = (float(grp["db"].mean()), float(grp["db"].std(ddof=1) or 5.0))
    out["_LEAGUE_"] = (float(tg["db"].mean()), float(tg["db"].std(ddof=1) or 5.0))
    return out


# ---------------------------------------------------------------------------
# Roster construction: depth chart + history -> shares and priors
# ---------------------------------------------------------------------------

def _team_totals(wk: pd.DataFrame) -> pd.DataFrame:
    """Team targets and carries per game, for turning counts into shares."""
    return (wk.groupby(["recent_team", "season", "week"], as_index=False)
              .agg(team_tgt=("targets", "sum"), team_car=("carries", "sum")))


def build_roster(wk: pd.DataFrame, snapshot: pd.DataFrame, team: str,
                 ruled_out: set | None = None,
                 pfr_agg: dict | None = None, pfr_lg: dict | None = None,
                 max_per_pos: dict | None = None) -> pd.DataFrame:
    """Turn one depth-chart snapshot into a table of players with usage shares.

    Each player gets a target share and a carry share that blend his own history
    with the prior for his depth-chart slot, plus the efficiency priors needed
    to turn volume into yards.
    """
    ruled_out = ruled_out or set()
    caps = {"WR": 5, "TE": 3, "RB": 4, "QB": 1, "FB": 1}
    caps.update(max_per_pos or {})

    snap = snapshot[~snapshot["player_id"].astype(str).isin(ruled_out)].copy()
    snap["depth"] = snap["depth"].astype(int)
    snap = snap[snap.apply(lambda r: r["depth"] <= caps.get(r["position"], 0), axis=1)]
    if snap.empty:
        raise ValueError(f"No usable depth chart for {team}.")

    totals = _team_totals(wk)
    lg = _league_rates(wk)
    rows = []
    for _, pl in snap.iterrows():
        pid = str(pl["player_id"])
        h = wk[wk["player_id"].astype(str) == pid]
        rows.append(_player_row(pl, h, totals, lg, pfr_agg, pfr_lg))

    r = pd.DataFrame(rows)
    # Shares are renormalised so the roster's volume adds up to the team's.
    for col in ("target_share", "carry_share"):
        tot = r[col].sum()
        r[col] = r[col] / tot if tot > 0 else 0.0
    for col, weight in (("rec_td_weight", "target_share"), ("rush_td_weight", "carry_share")):
        tot = r[col].sum()
        r[col] = r[col] / tot if tot > 0 else r[weight]
    return r.sort_values(["position", "depth"]).reset_index(drop=True)


def _league_rates(wk: pd.DataFrame) -> dict:
    tg, rec = wk["targets"].sum(), wk["receptions"].sum()
    ry, car = wk["receiving_yards"].sum(), wk["carries"].sum()
    return dict(
        catch=float(rec / tg) if tg > 0 else LG_CATCH_RATE,
        ypt=float(ry / tg) if tg > 0 else LG_YPT,
        rec_td=float(wk["receiving_tds"].sum() / rec) if rec > 0 else 0.075,
        rush_td=float(wk["rushing_tds"].sum() / car) if car > 0 else 0.025,
    )


def _player_row(pl: pd.Series, h: pd.DataFrame, totals: pd.DataFrame,
                lg: dict, pfr_agg, pfr_lg) -> dict:
    pos, depth = pl["position"], int(pl["depth"])
    games = int(len(h))
    blend = games / (games + ROLE_BLEND_N)

    tgt_prior = TARGET_ROLE_PRIOR.get((pos, depth), 0.01)
    car_prior = CARRY_ROLE_PRIOR.get((pos, depth), 0.005)

    if games:
        w = h["season_w"].values
        own_tgt = float(D.wmean(h["target_share"].values, w)) \
            if h["target_share"].sum() > 0 else _share_from_counts(h, totals, "targets")
        own_car = _share_from_counts(h, totals, "carries")
        tgt = blend * _safe(own_tgt, tgt_prior) + (1 - blend) * tgt_prior
        car = blend * _safe(own_car, car_prior) + (1 - blend) * car_prior

        tg_tot, rec_tot = float(h["targets"].sum()), float(h["receptions"].sum())
        ry_tot, car_tot = float(h["receiving_yards"].sum()), float(h["carries"].sum())
        catch = (rec_tot + CATCH_PRIOR_N * lg["catch"]) / (tg_tot + CATCH_PRIOR_N)
        ypt = (ry_tot + YPT_PRIOR_N * lg["ypt"]) / (tg_tot + YPT_PRIOR_N)
        rec_td = ((float(h["receiving_tds"].sum()) + TD_RATE_PRIOR_N * lg["rec_td"])
                  / (rec_tot + TD_RATE_PRIOR_N))
        rush_td = ((float(h["rushing_tds"].sum()) + TD_RATE_PRIOR_N * lg["rush_td"])
                   / (car_tot + TD_RATE_PRIOR_N))
        name = str(h["player_display_name"].iloc[-1])
    else:
        tgt, car = tgt_prior, car_prior
        catch, ypt = lg["catch"], lg["ypt"]
        rec_td, rush_td = lg["rec_td"], lg["rush_td"]
        name = str(pl.get("player_name") or pl["player_id"])

    rush_priors = None
    if car > 0.02 and games:
        try:
            rush_priors = R.player_rush_priors(h, str(pl["player_id"]), pfr_agg, pfr_lg)
        except Exception:
            rush_priors = None

    return dict(
        player_id=str(pl["player_id"]), name=name, position=pos, depth=depth,
        games=games, from_history=bool(games), blend=float(blend),
        target_share=float(max(tgt, 0.0)), carry_share=float(max(car, 0.0)),
        catch_rate=float(np.clip(catch, 0.30, 0.90)),
        ypt=float(np.clip(ypt, 3.0, 14.0)),
        rec_td_rate=float(np.clip(rec_td, 0.005, 0.30)),
        rush_td_rate=float(np.clip(rush_td, 0.002, 0.20)),
        # Role weight for splitting team touchdowns: opportunity x conversion.
        rec_td_weight=float(max(tgt, 0.0) * np.clip(rec_td, 0.005, 0.30) / max(lg["rec_td"], 1e-6)),
        rush_td_weight=float(max(car, 0.0) * np.clip(rush_td, 0.002, 0.20) / max(lg["rush_td"], 1e-6)),
        rush_priors=rush_priors,
    )


def _safe(x, fallback):
    return float(x) if x is not None and np.isfinite(x) and x > 0 else float(fallback)


def _share_from_counts(h: pd.DataFrame, totals: pd.DataFrame, col: str) -> float:
    """A player's share of team volume across the games he actually played."""
    j = h.merge(totals, on=["recent_team", "season", "week"], how="left")
    tot = j["team_tgt" if col == "targets" else "team_car"].sum()
    return float(j[col].sum() / tot) if tot and tot > 0 else 0.0


# ---------------------------------------------------------------------------
# Allocation helpers
# ---------------------------------------------------------------------------

def _split_counts(total: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Split an integer total across players by weight, keeping the sum exact.

    Largest-remainder apportionment: floor everything, then hand the leftovers
    to whoever was rounded down hardest. This is what makes the box score add up
    to the team total in every single simulation.
    """
    raw = total[:, None] * weights
    base = np.floor(raw).astype(int)
    rem = (total - base.sum(axis=1)).astype(int)
    frac = raw - base
    rank = np.argsort(np.argsort(-frac, axis=1), axis=1)
    return base + (rank < np.clip(rem, 0, weights.shape[1])[:, None]).astype(int)


def _multinomial_alloc(rng, counts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Hand out `counts` touchdowns per simulation, one at a time, by weight."""
    n, k = weights.shape
    out = np.zeros((n, k), dtype=int)
    if counts.max() <= 0:
        return out
    cum = np.cumsum(weights, axis=1)
    cum = cum / np.clip(cum[:, -1:], 1e-12, None)
    for slot in range(int(counts.max())):
        live = counts > slot
        if not live.any():
            break
        u = rng.random(n)
        idx = (cum < u[:, None]).sum(axis=1).clip(0, k - 1)
        out[live, idx[live]] += 1
    return out


def _apply_hfa(mix: dict, delta_points: float, drives: float) -> dict:
    """Nudge a drive outcome mix so the game's expected points move by
    `delta_points` — home field expressed as scoring rate, not a fudge added to
    a discrete final score."""
    if drives <= 0:
        return mix
    target = mix["ppd"] + delta_points / drives
    cur = T.TD_POINTS * mix["p_td"] + T.FG_POINTS * mix["p_fg"]
    if cur <= 1e-6:
        return mix
    k = float(np.clip(target / cur, 0.5, 1.6))
    out = dict(mix)
    out["p_td"] = float(np.clip(mix["p_td"] * k, *T.TD_CLIP))
    out["p_fg"] = float(np.clip(mix["p_fg"] * k, *T.FG_CLIP))
    out["ppd"] = T.TD_POINTS * out["p_td"] + T.FG_POINTS * out["p_fg"]
    return out


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

def simulate_game(ratings: dict, wk: pd.DataFrame,
                  roster_a: pd.DataFrame, roster_b: pd.DataFrame,
                  team_a: str, team_b: str,
                  pass_vol: dict, rush_vol: dict,
                  rush_def: dict | None = None,
                  lg_pass: dict | None = None,
                  home: str | None = "a",
                  n_sims: int = 20000, seed: int | None = None) -> dict:
    """Simulate the game `n_sims` times and return scores plus both box scores."""
    rng = np.random.default_rng(seed)
    n = int(n_sims)

    mix_a = T.matchup(ratings, team_a, team_b)
    mix_b = T.matchup(ratings, team_b, team_a)
    pace = T.game_pace(ratings, team_a, team_b)

    half = 0.5 * float(ratings.get("hfa", T.HFA_DEFAULT))
    if home in ("a", team_a):
        mix_a = _apply_hfa(mix_a, +half, pace["mean"])
        mix_b = _apply_hfa(mix_b, -half, pace["mean"])
    elif home in ("b", team_b):
        mix_a = _apply_hfa(mix_a, -half, pace["mean"])
        mix_b = _apply_hfa(mix_b, +half, pace["mean"])

    # 1. pace — one shared draw, both teams within a possession of each other
    base = rng.normal(pace["mean"], pace["sd"], n)
    drives_a = np.clip(np.round(base + rng.normal(0, 0.35, n)), 6, 16).astype(int)
    drives_b = np.clip(np.round(base + rng.normal(0, 0.35, n)), 6, 16).astype(int)

    # 2. drives -> score
    sa = _score_side(rng, drives_a, mix_a, ratings, team_b, pace["mean"])
    sb = _score_side(rng, drives_b, mix_b, ratings, team_a, pace["mean"])
    # a defensive score is produced by the DEFENSE, so it belongs to the other team
    pts_a = sa["off_points"] + sb["takeaway_points"]
    pts_b = sb["off_points"] + sa["takeaway_points"]

    # 3. game script, then 4. allocation
    box_a = _side_box(rng, wk, roster_a, team_a, team_b, drives_a, sa,
                      pts_a - pts_b, pass_vol, rush_vol, rush_def, lg_pass, ratings)
    box_b = _side_box(rng, wk, roster_b, team_b, team_a, drives_b, sb,
                      pts_b - pts_a, pass_vol, rush_vol, rush_def, lg_pass, ratings)

    return dict(
        team_a=team_a, team_b=team_b, points_a=pts_a, points_b=pts_b,
        drives_a=drives_a, drives_b=drives_b, box_a=box_a, box_b=box_b,
        mix_a=mix_a, mix_b=mix_b, pace=pace, n_sims=n, home=home,
    )


def _score_side(rng, drives: np.ndarray, mix: dict, r: dict, defense: str,
                pace_mean: float) -> dict:
    """Resolve one team's drives into touchdowns, field goals and points.

    Scoring rates are scaled by how many drives this simulation drew, so that
    expected points stay pace-invariant (see PACE_POINTS_ELASTICITY). A 13-drive
    game has the same expected points as an 8-drive one, spread thinner.
    """
    p_to = mix["p_turnover"]
    scale = np.clip((pace_mean / np.maximum(drives, 1))
                    ** (1.0 - PACE_POINTS_ELASTICITY), *PACE_SCALE_CLIP)
    p_td = np.clip(mix["p_td"] * scale, 1e-4, 0.8)
    p_fg = np.clip(mix["p_fg"] * scale, 1e-4, 0.8)

    n_td = rng.binomial(drives, p_td)
    rest = drives - n_td
    n_fg = rng.binomial(rest, np.clip(p_fg / np.maximum(1 - p_td, 1e-6), 0, 1))
    rest2 = rest - n_fg
    n_to = rng.binomial(rest2, np.clip(p_to / np.maximum(1 - p_td - p_fg, 1e-6), 0, 1))

    xp = rng.binomial(n_td, XP_RATE)
    off_points = 6 * n_td + xp + 3 * n_fg

    # Points this team's DEFENSE takes the other way, plus the league's
    # return/safety residual that belongs to no drive at all.
    d_rate = float(r["def_score_rate"].get(defense, r["lg_def_score"])) \
        if hasattr(r["def_score_rate"], "get") else r["lg_def_score"]
    n_def_td = rng.poisson(np.clip(d_rate * drives, 0, None))
    n_other = rng.poisson(np.full(len(drives), max(r.get("lg_other_ppg", 0.0), 0.0) / 7.0))
    take = n_def_td + n_other
    takeaway_points = 6 * take + rng.binomial(take, XP_RATE)

    return dict(n_td=n_td, n_fg=n_fg, n_to=n_to, off_points=off_points,
                takeaway_points=takeaway_points)


def _side_box(rng, wk, roster, team, opponent, drives, score, margin,
              pass_vol, rush_vol, rush_def, lg_pass, ratings) -> dict:
    """Volume, game script and player allocation for one team."""
    n = len(drives)
    k = len(roster)

    # --- volume, scaled to this simulation's pace -------------------------
    pace_team = float(ratings["pace"].get(team, ratings["lg_pace"]))
    # plays grow with drives, but sublinearly — a 13-drive game is not 30% more
    # snaps than a 10-drive one, it is about 10% more
    scale = np.clip((drives / max(pace_team, 1e-6)) ** PACE_VOLUME_ELASTICITY,
                    *PACE_SCALE_CLIP)
    mu_db, sd_db = pass_vol.get(team, pass_vol["_LEAGUE_"])
    mu_car, sd_car = rush_vol.get(team, rush_vol["_LEAGUE_"])
    db = np.clip(rng.normal(mu_db * scale, sd_db), 12, None)
    car = np.clip(rng.normal(mu_car * scale, sd_car), 6, None)

    # --- game script: trailing teams throw more ---------------------------
    plays = db + car
    pass_frac = np.clip(db / plays + GAME_SCRIPT_BETA * (-margin), *PASS_FRAC_CLIP)
    dropbacks = np.round(plays * pass_frac).astype(int)
    carries = np.round(plays * (1 - pass_frac)).astype(int)

    # --- sacks & INTs (Phase 4 models) ------------------------------------
    qb_row = roster[roster["position"] == "QB"]
    sacks = np.zeros(n, dtype=int)
    ints = np.zeros(n, dtype=int)
    qb_priors = None
    if len(qb_row) and lg_pass is not None:
        try:
            qb_priors = Q.qb_priors(wk, qb_row.iloc[0]["player_id"], lg_pass)
            dprof = (ratings.get("pass_def") or {}).get(opponent)
            s_rate = (D.combine_rate_logodds(qb_priors["p_sack"] * qb_priors["ttt_mult"],
                                             dprof["r_sack"], lg_pass["sack"],
                                             Q.DEFAULT_SACK_DEF_SHRINK)
                      if dprof else qb_priors["p_sack"] * qb_priors["ttt_mult"])
            i_rate = (D.combine_rate_logodds(qb_priors["p_int"], dprof["r_int"],
                                             lg_pass["intr"], Q.DEFAULT_INT_DEF_SHRINK)
                      if dprof else qb_priors["p_int"])
            sacks = rng.binomial(dropbacks, float(np.clip(s_rate, 1e-4, 0.5)))
            attempts = np.clip(dropbacks - sacks, 0, None)
            ints = rng.binomial(attempts, float(np.clip(i_rate, 1e-4, 0.4)))
        except Exception:
            qb_priors = None
    attempts = np.clip(dropbacks - sacks, 0, None)

    # --- split targets and carries across the depth chart ------------------
    tgt_w = rng.dirichlet(np.clip(roster["target_share"].values, 1e-4, None)
                          * TARGET_CONCENTRATION, size=n)
    car_w = rng.dirichlet(np.clip(roster["carry_share"].values, 1e-4, None)
                          * CARRY_CONCENTRATION, size=n)
    targets = _split_counts(attempts, tgt_w)
    player_car = _split_counts(carries, car_w)

    # --- catches and yards -------------------------------------------------
    receptions = np.zeros((n, k), dtype=int)
    rec_yards = np.zeros((n, k))
    rush_yards = np.zeros((n, k))
    for j, pl in roster.reset_index(drop=True).iterrows():
        t = targets[:, j]
        rec = rng.binomial(t, pl["catch_rate"])
        receptions[:, j] = rec
        ypc = pl["ypt"] / max(pl["catch_rate"], 1e-6)
        kk = 1.0 / (YPC_CV ** 2)
        rec_yards[:, j] = np.where(
            rec > 0, rng.gamma(np.clip(rec * kk, 1e-9, None), 1.0) * (ypc / kk), 0.0)

        c = player_car[:, j]
        if pl["rush_priors"] is not None and c.max() > 0:
            dp = (rush_def or {}).get(opponent)
            rush_yards[:, j] = R.yards_from_carries(rng, pl["rush_priors"], c, dp)["yards"]
        elif c.max() > 0:
            rush_yards[:, j] = c * 4.2 * rng.lognormal(-0.08, 0.40, n)

    # --- touchdowns: team total, then who ----------------------------------
    n_td = score["n_td"]
    pass_share = _team_pass_td_share(wk, team)
    n_pass_td = rng.binomial(n_td, pass_share)
    n_rush_td = n_td - n_pass_td
    rec_tds = _multinomial_alloc(rng, n_pass_td,
                                 np.tile(roster["rec_td_weight"].values, (n, 1)))
    rush_tds = _multinomial_alloc(rng, n_rush_td,
                                  np.tile(roster["rush_td_weight"].values, (n, 1)))

    return dict(
        roster=roster, targets=targets, receptions=receptions, rec_yards=rec_yards,
        carries=player_car, rush_yards=rush_yards, rec_tds=rec_tds, rush_tds=rush_tds,
        dropbacks=dropbacks, attempts=attempts, sacks=sacks, ints=ints,
        team_pass_yards=rec_yards.sum(axis=1), team_rush_yards=rush_yards.sum(axis=1),
        pass_frac=pass_frac, qb_priors=qb_priors,
        n_pass_td=n_pass_td, n_rush_td=n_rush_td,
    )


def _team_pass_td_share(wk: pd.DataFrame, team: str) -> float:
    """Share of a team's offensive touchdowns that come through the air."""
    t = wk[wk["recent_team"] == team]
    rec, rush = float(t["receiving_tds"].sum()), float(t["rushing_tds"].sum())
    if rec + rush < 20:
        return LG_PASS_TD_SHARE
    return float(np.clip(rec / (rec + rush), 0.35, 0.85))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def box_score(side: dict) -> pd.DataFrame:
    """Mean projected line for every player on one side."""
    r = side["roster"].reset_index(drop=True)
    out = pd.DataFrame({
        "Player": r["name"], "Pos": r["position"], "Depth": r["depth"],
        "Tgt": side["targets"].mean(axis=0),
        "Rec": side["receptions"].mean(axis=0),
        "RecYds": side["rec_yards"].mean(axis=0),
        "RecTD": side["rec_tds"].mean(axis=0),
        "Car": side["carries"].mean(axis=0),
        "RushYds": side["rush_yards"].mean(axis=0),
        "RushTD": side["rush_tds"].mean(axis=0),
    })
    out["TD"] = out["RecTD"] + out["RushTD"]
    out["Yds"] = out["RecYds"] + out["RushYds"]
    out["Source"] = np.where(r["games"] > 0, r["games"].astype(str) + "g history", "role prior")
    return out.sort_values("Yds", ascending=False).reset_index(drop=True)


def qb_line(side: dict) -> dict:
    """The QB's line, assembled from the receivers rather than modelled apart."""
    r = side["roster"].reset_index(drop=True)
    qb = r[r["position"] == "QB"]
    name = str(qb.iloc[0]["name"]) if len(qb) else "QB"
    return dict(
        name=name,
        attempts=float(side["attempts"].mean()),
        completions=float(side["receptions"].sum(axis=1).mean()),
        pass_yards=float(side["team_pass_yards"].mean()),
        pass_tds=float(side["n_pass_td"].mean()),
        sacks=float(side["sacks"].mean()),
        ints=float(side["ints"].mean()),
    )


def summarize(sim: dict) -> dict:
    a, b = sim["points_a"], sim["points_b"]
    margin = a - b
    return dict(
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        median_a=float(np.median(a)), median_b=float(np.median(b)),
        win_a=float((margin > 0).mean()), win_b=float((margin < 0).mean()),
        tie=float((margin == 0).mean()),
        mean_margin=float(margin.mean()), median_margin=float(np.median(margin)),
        margin_sd=float(margin.std()),
        mean_total=float((a + b).mean()), total_sd=float((a + b).std()),
        cover_odds=D.american(float((margin > 0).mean())),
        p10_total=float(np.percentile(a + b, 10)),
        p90_total=float(np.percentile(a + b, 90)),
    )


# ---------------------------------------------------------------------------
# One-call convenience wrapper
# ---------------------------------------------------------------------------

def prepare(seasons: tuple[int, ...], depth_seasons: tuple[int, ...] | None = None) -> dict:
    """Load every feed the engine needs once, and build the shared pieces.

    `seasons` is the priors window (how teams and players have played).
    `depth_seasons` is who is on the roster NOW — a different question, so it
    defaults to the season after the priors window (the one being played) and
    falls back to the last priors season if that file does not exist yet.
    """
    wk = D.load_weekly(seasons)
    drives = D.load_drives(seasons)
    ratings = T.team_ratings(drives, D.load_games(seasons))
    ratings["pass_def"] = D.def_pass_rates(wk)
    pfr = D.load_pfr_rush(seasons)
    agg, pfr_lg = R.pfr_rush_aggregates(pfr)
    depth_seasons, depth, injuries = _current_rosters(seasons, depth_seasons)
    return dict(
        wk=wk, ratings=ratings,
        pass_vol=team_pass_volume(wk), rush_vol=R.team_rush_volume(wk),
        rush_def=R.rush_defense_profiles(wk, pfr, D.load_pbp(seasons)),
        lg_pass=Q.league_pass_rates(wk), pfr_agg=agg, pfr_lg=pfr_lg,
        depth=depth, injuries=injuries, depth_seasons=depth_seasons,
    )


def _current_rosters(seasons: tuple[int, ...],
                     depth_seasons: tuple[int, ...] | None):
    """Depth charts and injury reports for the season being played."""
    if depth_seasons is None:
        depth_seasons = (max(int(s) for s in seasons) + 1,)
        if D.load_depth_charts(depth_seasons).empty:
            depth_seasons = (max(int(s) for s in seasons),)
    depth_seasons = tuple(sorted(int(s) for s in depth_seasons))
    return depth_seasons, D.load_depth_charts(depth_seasons), D.load_injuries(depth_seasons)


def roster_for(ctx: dict, team: str, use_injuries: bool = True,
               week: int | None = None) -> pd.DataFrame:
    """Build one team's roster from its most recent depth chart.

    The injury filter uses the latest report of the CURRENT season (or `week` if
    given). It is only meaningful in season — out of season the last report
    belongs to a game already played.
    """
    snap = D.depth_chart_snapshot(ctx["depth"], team)
    out = D.players_ruled_out(ctx["injuries"], team, week=week) if use_injuries else set()
    return build_roster(ctx["wk"], snap, team, out, ctx["pfr_agg"], ctx["pfr_lg"])


if __name__ == "__main__":
    seasons = (2024, 2025)
    print("Loading every feed (first run downloads a few tens of MB)...")
    ctx = prepare(seasons)
    team_a, team_b = "BAL", "SF"
    ra = roster_for(ctx, team_a)
    rb = roster_for(ctx, team_b)

    sim = simulate_game(ctx["ratings"], ctx["wk"], ra, rb, team_a, team_b,
                        ctx["pass_vol"], ctx["rush_vol"], ctx["rush_def"],
                        ctx["lg_pass"], home="a", n_sims=20000, seed=5)
    s = summarize(sim)
    print(f"\n{team_a} (home) {s['mean_a']:.1f} - {s['mean_b']:.1f} {team_b}   "
          f"win {s['win_a']:.1%} / {s['win_b']:.1%} | margin {s['mean_margin']:+.1f} "
          f"(sd {s['margin_sd']:.1f}) | total {s['mean_total']:.1f} "
          f"({s['p10_total']:.0f}-{s['p90_total']:.0f})")

    for team, side in ((team_a, sim["box_a"]), (team_b, sim["box_b"])):
        q = qb_line(side)
        print(f"\n{team} — {q['name']}: {q['completions']:.1f}/{q['attempts']:.1f}, "
              f"{q['pass_yards']:.0f} yds, {q['pass_tds']:.1f} TD, "
              f"{q['ints']:.1f} INT, {q['sacks']:.1f} sacks")
        bs = box_score(side)
        print(bs[bs["Yds"] > 5][["Player", "Pos", "Depth", "Tgt", "Rec", "RecYds",
                                 "Car", "RushYds", "TD", "Source"]]
              .to_string(index=False, float_format=lambda x: f"{x:.1f}"))
