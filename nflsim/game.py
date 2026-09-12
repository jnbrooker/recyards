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

**Depth chart to role (§4.4)** lives in `nflsim/roster.py`, shared with the
single-stat pages: the depth chart decides who is on the field and in what
slot; the player's own history decides what that slot is worth.

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
from . import roster as RO
from .roster import build_roster   # re-exported: the engine's roster contract

# Dirichlet concentration on the allocated shares: higher = less week-to-week
# wobble in who gets the ball.
TARGET_CONCENTRATION = 45.0
CARRY_CONCENTRATION = 30.0

YPC_CV = 1.05               # per-catch yardage spread
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

# Game script: extra pass share per point of deficit, applied to the REALISED
# margin in each simulation. Measured on 2024-25 (roadmap 8.9): pass share
# moves -0.0034 per point of realised margin, but only -0.0011 per point of the
# model's pre-game EXPECTED margin — the realised slope includes the reverse
# direction (teams that throw more lose by more). 0.002 sits between the two
# so the engine's mean volume shift for an expected 14-point dog (~3 pp) is
# near the measured 1.5 pp while within-sim correlation stays realistic. The
# hand-set 0.006 it replaces overstated the script ~2-4x.
GAME_SCRIPT_BETA = 0.002
PASS_FRAC_CLIP = (0.25, 0.80)

# Pre-game script for the single-stat pages' GAME VIEW: how a team's volume
# moves with the model's expected margin, measured on 2024-25 out-of-sample
# expectations (carries +0.113 per point, se 0.039; dropbacks -0.006, se
# 0.044 — i.e. flat).
SCRIPT_CARRIES_PER_POINT = 0.113
SCRIPT_DROPBACKS_PER_POINT = 0.0

XP_RATE = 0.96              # so a touchdown averages T.TD_POINTS

# Not every pass attempt is a target: throwaways, spikes and batted balls are
# ~5% of attempts. `prepare` measures the ratio from the weekly feed; this is
# the fallback. Without it the engine ran 5% high on targets and 8% on
# receiving yards per team-game.
TARGET_PER_ATTEMPT = 0.95


# ---------------------------------------------------------------------------
# Team volume
# ---------------------------------------------------------------------------

def team_pass_volume(wk: pd.DataFrame) -> dict:
    """Recency-weighted mean & std of team DROPBACKS per game (attempts +
    sacks), by team."""
    col = "dropbacks" if "dropbacks" in wk.columns else "attempts"
    tg = R.team_game_volume(wk, col)
    out = {team: (D.wmean(grp["vol"], grp["w"]), D.wstd(grp["vol"], grp["w"], 5.0))
           for team, grp in tg.groupby("recent_team")}
    out["_LEAGUE_"] = (D.wmean(tg["vol"], tg["w"]), D.wstd(tg["vol"], tg["w"], 5.0))
    return out


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
                  n_sims: int = 20000, seed: int | None = None,
                  avail: dict | None = None, wind=None, roof=None,
                  target_rate: float | None = None) -> dict:
    """Simulate the game `n_sims` times and return scores plus both box scores.
    `avail` is `{team: availability indices}` (see `availability.py`) and
    `wind` / `roof` the weather — the same shifts `teams.expected_points`
    applies (availability on the margin, wind on the total)."""
    rng = np.random.default_rng(seed)
    n = int(n_sims)

    mix_a = T.matchup(ratings, team_a, team_b)
    mix_b = T.matchup(ratings, team_b, team_a)
    pace = T.game_pace(ratings, team_a, team_b)

    shift_a = shift_b = 0.0
    if avail:
        from . import availability as AV
        shift_a = 0.5 * AV.margin_shift(avail.get(team_a), avail.get(team_b))
        shift_b = -shift_a
    wx = 0.5 * T.weather_total_shift(wind, roof)
    shift_a, shift_b = shift_a + wx, shift_b + wx
    half = 0.5 * float(ratings.get("hfa", T.HFA_DEFAULT))
    if home in ("a", team_a):
        shift_a, shift_b = shift_a + half, shift_b - half
    elif home in ("b", team_b):
        shift_a, shift_b = shift_a - half, shift_b + half
    mix_a = _apply_hfa(mix_a, shift_a, pace["mean"])
    mix_b = _apply_hfa(mix_b, shift_b, pace["mean"])

    # 1. pace — one shared draw, both teams within a possession of each other
    base = rng.normal(pace["mean"], pace["sd"], n)
    drives_a = np.clip(np.round(base + rng.normal(0, 0.35, n)), 6, 16).astype(int)
    drives_b = np.clip(np.round(base + rng.normal(0, 0.35, n)), 6, 16).astype(int)

    # 2. drives -> score, in sequence with the running score feeding back
    sa, sb = _score_sides(rng, drives_a, drives_b, mix_a, mix_b, ratings,
                          team_a, team_b, pace["mean"])
    # a defensive score is produced by the DEFENSE, so it belongs to the other team
    pts_a = sa["off_points"] + sb["takeaway_points"]
    pts_b = sb["off_points"] + sa["takeaway_points"]

    # 3. game script, then 4. allocation
    tr = float(target_rate) if target_rate else TARGET_PER_ATTEMPT
    box_a = _side_box(rng, wk, roster_a, team_a, team_b, drives_a, sa,
                      pts_a - pts_b, pass_vol, rush_vol, rush_def, lg_pass, ratings, tr)
    box_b = _side_box(rng, wk, roster_b, team_b, team_a, drives_b, sb,
                      pts_b - pts_a, pass_vol, rush_vol, rush_def, lg_pass, ratings, tr)

    return dict(
        team_a=team_a, team_b=team_b, points_a=pts_a, points_b=pts_b,
        drives_a=drives_a, drives_b=drives_b, box_a=box_a, box_b=box_b,
        mix_a=mix_a, mix_b=mix_b, pace=pace, n_sims=n, home=home,
        avail=avail, avail_shift_a=float(shift_a), avail_shift_b=float(shift_b),
        weather_shift=float(2 * wx), wind=wind, roof=roof,
    )


# Game script within the game (roadmap 8.8). Drives are resolved in sequence,
# alternating possessions, and a team's scoring rate on each drive is scaled
# by exp(-LEAD_BETA x (lead - expected lead so far) / 7): a side a touchdown
# ahead of where the ratings expected it to be scores at exp(-LEAD_BETA) of
# its rate, a side a touchdown behind at exp(+LEAD_BETA). Centring on the
# EXPECTED lead keeps the mean margin where the ratings put it (they are
# fitted to actual points, which already include favourites sitting on
# leads) and changes only the spread. Independent drives give a team-points
# sd of ~9.6 (pure binomial); the real CONDITIONAL spread — the residual
# around the model's own prediction, 2024-25 out of sample — is ~9.3 per
# team, 12.8 on the margin, with the two teams' points correlated +0.05.
# Leading teams sitting on the ball and trailing teams pressing produces
# both, and LEAD_BETA is calibrated to those moments.
LEAD_BETA = 0.03
MAX_DRIVES = 16


def _score_sides(rng, drives_a: np.ndarray, drives_b: np.ndarray, mix_a: dict, mix_b: dict,
                 r: dict, team_a: str, team_b: str, pace_mean: float) -> tuple[dict, dict]:
    """Resolve both teams' drives into touchdowns, field goals and points,
    drive by drive with the running score feeding back into the rates.

    Scoring rates are also scaled by how many drives this simulation drew, so
    expected points stay pace-invariant (see PACE_POINTS_ELASTICITY): a
    13-drive game has the same expected points as an 8-drive one, spread thinner.
    """
    n = len(drives_a)

    def base(mix, drives):
        scale = np.clip((pace_mean / np.maximum(drives, 1))
                        ** (1.0 - PACE_POINTS_ELASTICITY), *PACE_SCALE_CLIP)
        return (np.clip(mix["p_td"] * scale, 1e-4, 0.8),
                np.clip(mix["p_fg"] * scale, 1e-4, 0.8),
                np.full(n, float(mix["p_turnover"])))

    sides = {}
    for key, drives, mix in (("a", drives_a, mix_a), ("b", drives_b, mix_b)):
        p_td, p_fg, p_to = base(mix, drives)
        sides[key] = dict(drives=drives, p_td=p_td, p_fg=p_fg, p_to=p_to,
                          n_td=np.zeros(n, int), n_fg=np.zeros(n, int), n_to=np.zeros(n, int),
                          pts=np.zeros(n, int))

    # expected points per drive for each side (for the expected lead so far)
    ppd = {key: T.TD_POINTS * sides[key]["p_td"] + T.FG_POINTS * sides[key]["p_fg"]
           for key in ("a", "b")}
    for k in range(MAX_DRIVES):
        for key, other in (("a", "b"), ("b", "a")):
            s, o = sides[key], sides[other]
            live = s["drives"] > k
            if not live.any():
                continue
            # this side has had k drives, the other k (or k+1) — expected lead now
            exp_lead = ppd[key] * k - ppd[other] * (k if key == "a" else k + 1)
            lead = (s["pts"] - o["pts"] - exp_lead) / 7.0
            f = np.exp(-LEAD_BETA * lead) if LEAD_BETA else 1.0
            p_td = np.clip(s["p_td"] * f, 1e-4, 0.85)
            p_fg = np.clip(s["p_fg"] * f, 1e-4, 0.85)
            scoring = p_td + p_fg
            over = scoring > 0.9
            if np.any(over):
                p_td = np.where(over, p_td * 0.9 / scoring, p_td)
                p_fg = np.where(over, p_fg * 0.9 / scoring, p_fg)
            u = rng.random(n)
            td = live & (u < p_td)
            fg = live & ~td & (u < p_td + p_fg)
            to = live & ~td & ~fg & (u < p_td + p_fg + s["p_to"])
            s["n_td"] += td; s["n_fg"] += fg; s["n_to"] += to
            xp = td & (rng.random(n) < XP_RATE)
            s["pts"] += 6 * td + xp + 3 * fg

    out = []
    for key, defense, drives in (("a", team_b, drives_a), ("b", team_a, drives_b)):
        s = sides[key]
        # Points this team's DEFENSE takes the other way, plus the league's
        # return/safety residual that belongs to no drive at all.
        d_rate = float(r["def_score_rate"].get(defense, r["lg_def_score"])) \
            if hasattr(r["def_score_rate"], "get") else r["lg_def_score"]
        n_def_td = rng.poisson(np.clip(d_rate * drives, 0, None))
        n_other = rng.poisson(np.full(n, max(r.get("lg_other_ppg", 0.0), 0.0) / 7.0))
        take = n_def_td + n_other
        takeaway_points = 6 * take + rng.binomial(take, XP_RATE)
        out.append(dict(n_td=s["n_td"], n_fg=s["n_fg"], n_to=s["n_to"],
                        off_points=s["pts"], takeaway_points=takeaway_points))
    return out[0], out[1]


def _side_box(rng, wk, roster, team, opponent, drives, score, margin,
              pass_vol, rush_vol, rush_def, lg_pass, ratings,
              target_rate: float = TARGET_PER_ATTEMPT) -> dict:
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
    targeted = rng.binomial(attempts, float(np.clip(target_rate, 0.8, 1.0)))
    targets = _split_counts(targeted, tgt_w)
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
    if float(t["receiving_tds"].sum() + t["rushing_tds"].sum()) < 20:
        return LG_PASS_TD_SHARE
    w = t["w"] if "w" in t.columns else 1.0
    rec, rush = float((t["receiving_tds"] * w).sum()), float((t["rushing_tds"] * w).sum())
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

def prepare(seasons: tuple[int, ...], depth_seasons: tuple[int, ...] | None = None,
            recency: D.Recency = D.RECENCY_DEFAULT) -> dict:
    """Load every feed the engine needs once, and build the shared pieces.

    `seasons` is the priors window (how teams and players have played); depth
    charts and injuries come from the season being played (see `roster.py`).
    `recency` sets how much recent games count on every feed (`data.Recency`).
    """
    live = RO.load_live(seasons, depth_seasons, recency)
    wk = live["wk"]
    seasons = live["seasons"]
    drives = D.load_drives(seasons, recency)
    ratings = T.team_ratings(drives, D.load_games(seasons))
    ratings["pass_def"] = D.def_pass_rates(wk)
    ctx = dict(live)
    ctx.update(
        ratings=ratings,
        pass_vol=team_pass_volume(wk), rush_vol=R.team_rush_volume(wk),
        rush_def=R.rush_defense_profiles(wk, D.load_pfr_rush(seasons, recency),
                                         D.load_pbp(seasons, recency)),
        lg_pass=Q.league_pass_rates(wk),
        target_rate=float(np.clip(wk["targets"].sum() / max(wk["attempts"].sum(), 1), 0.85, 1.0))
        if "attempts" in wk.columns else TARGET_PER_ATTEMPT,
    )
    # who is actually playing this week (§8.5): QB familiarity and defensive
    # starters ruled out, from the live depth charts and injury report
    try:
        from . import availability as AV
        ctx["avail"] = AV.current_indices(live)
    except Exception:
        ctx["avail"] = {}
    return ctx


def script_factors(ctx: dict, team: str, opp: str, home: str | None = "a",
                   wind=None, roof=None) -> dict:
    """The pre-game view of one matchup for the single-stat pages: expected
    margin and win probability (ratings + availability + home field + wind),
    the team's expected carries / dropbacks / targets in THIS game vs its
    typical game (`SCRIPT_*_PER_POINT`), and a touchdown factor — this game's
    expected points over the team's typical points, which scales per-touch
    TD rates (more scoring drives, more red-zone touches)."""
    r = ctx["ratings"]
    e = T.expected_points(r, team, opp, home=home, avail=ctx.get("avail"), wind=wind, roof=roof)
    margin = float(e["margin"])
    mu_db, sd_db = ctx["pass_vol"].get(team, ctx["pass_vol"]["_LEAGUE_"])
    mu_car, sd_car = ctx["rush_vol"].get(team, ctx["rush_vol"]["_LEAGUE_"])
    car = max(mu_car + SCRIPT_CARRIES_PER_POINT * margin, 8.0)
    db = max(mu_db + SCRIPT_DROPBACKS_PER_POINT * margin, 12.0)
    tv = ctx["team_vol"].get(team, ctx["team_vol"]["_LEAGUE_"])
    tgt_per_db = tv["targets"] / max(tv["dropbacks"], 1e-6)
    typical = (r["lg_ppd"] + float(r["off"].get(team, 0.0))) * r["lg_pace"] + float(r.get("lg_other_ppg", 0.0))
    td_factor = float(np.clip(e["points_a"] / max(typical, 1e-6), 0.6, 1.5))
    from math import erf, sqrt
    win = 0.5 * (1 + erf(margin / 12.8 / sqrt(2)))
    return dict(team=team, opp=opp, exp_margin=margin, win=float(win),
                points_for=float(e["points_a"]), points_against=float(e["points_b"]),
                carries=(float(car), float(sd_car)), carries_typical=float(mu_car),
                dropbacks=(float(db), float(sd_db)), dropbacks_typical=float(mu_db),
                targets=(float(db * tgt_per_db), float(sd_db * tgt_per_db)),
                targets_typical=float(mu_db * tgt_per_db),
                td_factor=td_factor, typical_points=float(typical),
                avail_shift=float(e.get("avail_shift_a", 0.0)), weather_shift=float(e.get("weather_shift", 0.0)))


def roster_for(ctx: dict, team: str, use_injuries: bool = True,
               week: int | None = None) -> pd.DataFrame:
    """One team's roster; inactive (ruled-out) players are dropped for the engine."""
    r = RO.roster_for(ctx, team, use_injuries, week)
    return r[r["active"]].reset_index(drop=True)


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
