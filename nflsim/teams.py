"""
nflsim/teams.py — the team-strength layer (roadmap §4.1), the one genuinely new
module the game engine needs.

Because the model is SELF-CONTAINED (no Vegas total, no spread), it has to form
its own opinion of how good each team is. The atom is a DRIVE:

    points this drive produced  =  league average
                                +  how good the offense is   (off rating)
                                +  how bad the defense is    (def rating)

Both ratings are solved together by iterating: a team's offensive rating is the
average of what its drives produced after subtracting the defensive ratings of
the units it actually faced, and vice versa. That is the SRS idea applied to
points per drive, so a good offense that played a brutal schedule is not
punished for it. Every rating is shrunk toward league average by how many drives
it rests on, and recent seasons and recent games are weighted more heavily.

The layer outputs, per matchup:

  * expected POINTS PER DRIVE for each offense against that defense,
  * expected DRIVES for the game (pace — a shared property of the two teams),
  * a per-drive OUTCOME MIX (touchdown / field goal / turnover / nothing),
    rescaled so its point value equals the expected points per drive above.

That last step is the §4.3 reconciliation in miniature: the efficiency rating
decides how many points a drive is worth, and the outcome mix is calibrated to
agree with it, so the drive engine can never produce a box score whose
touchdowns contradict the team ratings.

Defensive/special-teams scores are tracked separately (`def_score_rate`),
because points a defense scores itself belong to that team's total but must not
be credited to its offense.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D

# Shrinkage: drives-worth of league-average prior on each rating.
RATING_PRIOR_N = 35.0      # drives-worth; was 200 — see roadmap 8.8 (out-of-sample slope 1.97 → 1.0)
RATE_PRIOR_N = 250.0       # outcome rates are noisier per drive
PACE_PRIOR_N = 8.0         # games-worth of prior on pace

SOLVE_ITERS = 60
SOLVE_DAMPING = 0.5        # blend each iteration with the previous one

# Home-field advantage, in points of margin. Fitted from the games supplied to
# `team_ratings`; this is only the fallback when no scores are available.
HFA_DEFAULT = 1.8
HFA_CLIP = (0.0, 4.0)

# Guard rails on a single matchup (points per drive).
PPD_CLIP = (0.30, 4.50)
TD_CLIP = (0.05, 0.55)
FG_CLIP = (0.04, 0.34)
TO_CLIP = (0.02, 0.30)

TD_POINTS = D.DRIVE_POINTS["Touchdown"]
FG_POINTS = D.DRIVE_POINTS["Field goal"]

# Weather (roadmap 8.6). Wind is the one weather variable with a signal in the
# out-of-sample total residuals: -0.52 points of total per mph above 10 (t =
# -1.6, same sign and size in 2024 and 2025, and in line with what is known
# about wind and scoring); cold was non-monotone and domes showed nothing, so
# only wind is priced, shrunk to -0.4/mph and capped at 25 mph. nflverse only
# records wind after the game, so for an upcoming game it has to be a forecast.
WIND_COEF = -0.4
WIND_FREE_MPH = 10.0
WIND_CAP_MPH = 25.0
INDOOR_ROOFS = ("dome", "closed")


def weather_total_shift(wind=None, roof=None) -> float:
    """Points of expected TOTAL from the weather (0 indoors or unknown)."""
    if roof is not None and str(roof).lower() in INDOOR_ROOFS:
        return 0.0
    if wind is None or not np.isfinite(float(wind)):
        return 0.0
    w = float(np.clip(float(wind), 0.0, WIND_CAP_MPH))
    return float(WIND_COEF * max(w - WIND_FREE_MPH, 0.0))


# ---------------------------------------------------------------------------
# Weighted helpers
# ---------------------------------------------------------------------------

def _wsum_by(keys: pd.Series, vals: pd.Series, w: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Weighted sum and weight total of `vals` grouped by `keys`."""
    num = (vals * w).groupby(keys).sum()
    den = w.groupby(keys).sum()
    return num, den


def _shrunk_mean(num: pd.Series, den: pd.Series, prior_n: float,
                 prior_mean: float = 0.0) -> pd.Series:
    """Weighted mean pulled toward `prior_mean` by how little data backs it."""
    return (num + prior_n * prior_mean) / (den + prior_n)


# ---------------------------------------------------------------------------
# The rating solve
# ---------------------------------------------------------------------------

def team_ratings(drives: pd.DataFrame, games: pd.DataFrame | None = None,
                 iters: int = SOLVE_ITERS) -> dict:
    """Opponent-adjusted offensive and defensive strength from a drive table.

    Returns a dict with, per team: `off` / `def` (points per drive above league,
    positive = better offense / worse defense), the raw unadjusted rates, the
    outcome-rate ratings, pace, and the league baselines everything hangs off.
    """
    if drives is None or drives.empty:
        raise ValueError("No drive data — the pbp feed did not load.")

    d = drives[drives["live"]].copy()
    if d.empty:
        raise ValueError("Drive table has no live drives.")
    # `w` is the recency weight from `data.load_drives` (season curve x
    # within-season decay); an unweighted table counts every drive equally.
    d["w"] = d["w"].astype(float) if "w" in d.columns else d.get("season_w", 1.0)

    lg_ppd = float((d["points"] * d["w"]).sum() / d["w"].sum())
    lg_td = float((d["is_td"] * d["w"]).sum() / d["w"].sum())
    lg_fg = float((d["is_fg"] * d["w"]).sum() / d["w"].sum())
    lg_to = float((d["is_turnover"] * d["w"]).sum() / d["w"].sum())

    teams = sorted(set(d["posteam"]) | set(d["defteam"]))
    off = pd.Series(0.0, index=teams)
    dfn = pd.Series(0.0, index=teams)

    # Denominators never change, so compute them once.
    _, off_den = _wsum_by(d["posteam"], d["points"], d["w"])
    _, def_den = _wsum_by(d["defteam"], d["points"], d["w"])
    off_den = off_den.reindex(teams).fillna(0.0)
    def_den = def_den.reindex(teams).fillna(0.0)

    for _ in range(int(iters)):
        # offense = what its drives produced, minus the defenses it faced
        resid = d["points"] - lg_ppd - d["defteam"].map(dfn).astype(float)
        num, _ = _wsum_by(d["posteam"], resid, d["w"])
        new_off = _shrunk_mean(num.reindex(teams).fillna(0.0), off_den, RATING_PRIOR_N)
        new_off -= new_off.mean()

        # defense = what it allowed, minus the offenses it faced
        resid = d["points"] - lg_ppd - d["posteam"].map(new_off).astype(float)
        num, _ = _wsum_by(d["defteam"], resid, d["w"])
        new_def = _shrunk_mean(num.reindex(teams).fillna(0.0), def_den, RATING_PRIOR_N)
        new_def -= new_def.mean()

        off = SOLVE_DAMPING * new_off + (1 - SOLVE_DAMPING) * off
        dfn = SOLVE_DAMPING * new_def + (1 - SOLVE_DAMPING) * dfn

    rates = _rate_ratings(d, teams, dict(td=lg_td, fg=lg_fg, to=lg_to))
    pace = _pace(d, teams)
    raw = _raw_rates(d, teams)
    sched_off, sched_def = _strength_of_schedule(d, teams, off, dfn)
    other = _other_scoring(d, games, pace["_LEAGUE_"], rates["lg_def_score"])

    out = dict(
        teams=teams, off=off, dfn=dfn,
        lg_ppd=lg_ppd, lg_td=lg_td, lg_fg=lg_fg, lg_to=lg_to,
        lg_pace=pace["_LEAGUE_"], pace=pace["team"], pace_sd=pace["sd"],
        off_td=rates["off_td"], def_td=rates["def_td"],
        off_fg=rates["off_fg"], def_fg=rates["def_fg"],
        off_to=rates["off_to"], def_to=rates["def_to"],
        def_score_rate=rates["def_score_rate"], lg_def_score=rates["lg_def_score"],
        lg_other_ppg=other, sched_off=sched_off, sched_def=sched_def,
        hfa=HFA_DEFAULT,
        raw=raw, drives=int(len(d)), seasons=sorted(d["season"].unique().tolist()),
    )
    # Home field is measured against the ratings themselves, so it has to be
    # fitted once they exist: it is whatever margin the home side wins by that
    # the team ratings do not already explain.
    out["hfa"] = _fit_home_field(out, games)
    return out


def _fit_home_field(r: dict, games: pd.DataFrame | None) -> float:
    """Mean unexplained home margin across the supplied games."""
    if games is None or len(games) == 0:
        return HFA_DEFAULT
    resid = []
    idx = r["off"].index
    for home, away, hs, as_ in zip(games["home_team"], games["away_team"],
                                   games["home_score"], games["away_score"]):
        if home not in idx or away not in idx:
            continue
        e = expected_points(r, home, away)              # neutral-site margin:
        model = e["points_a"] - e["points_b"]           # no home term applied
        resid.append((float(hs) - float(as_)) - model)
    if not resid:
        return HFA_DEFAULT
    return float(np.clip(np.mean(resid), *HFA_CLIP))


def _rate_ratings(d: pd.DataFrame, teams: list, lg: dict) -> dict:
    """Opponent-adjusted touchdown / field-goal / turnover rates per drive.

    Same additive residual idea as the points solve, but run once rather than
    iterated: the point ratings already carry the schedule adjustment, and these
    rates only need to be good enough to shape the outcome mix (which is then
    rescaled to agree with the points rating anyway).
    """
    out = {}
    for key, col in (("td", "is_td"), ("fg", "is_fg"), ("to", "is_turnover")):
        base = lg[key]
        num, den = _wsum_by(d["posteam"], d[col] - base, d["w"])
        out[f"off_{key}"] = _shrunk_mean(num.reindex(teams).fillna(0.0),
                                         den.reindex(teams).fillna(0.0), RATE_PRIOR_N)
        num, den = _wsum_by(d["defteam"], d[col] - base, d["w"])
        out[f"def_{key}"] = _shrunk_mean(num.reindex(teams).fillna(0.0),
                                         den.reindex(teams).fillna(0.0), RATE_PRIOR_N)

    # Points a DEFENSE scores itself (pick-six etc.): credited to that team, and
    # deliberately kept out of its offense's rating.
    ds = (d["result"] == "Opp touchdown").astype(float)
    lg_ds = float((ds * d["w"]).sum() / d["w"].sum())
    num, den = _wsum_by(d["defteam"], ds - lg_ds, d["w"])
    out["def_score_rate"] = (_shrunk_mean(num.reindex(teams).fillna(0.0),
                                          den.reindex(teams).fillna(0.0),
                                          RATE_PRIOR_N) + lg_ds).clip(0.0, 0.10)
    out["lg_def_score"] = lg_ds
    return out


def _pace(d: pd.DataFrame, teams: list) -> dict:
    """Live drives per team-game: league mean, each team's shrunk mean, and the
    game-to-game spread (pace is a property of the pairing, not one team)."""
    per_game = (d.groupby(["season", "game_id", "posteam"], as_index=False)
                  .agg(drives=("drive", "size"), w=("w", "first")))
    lg = float((per_game["drives"] * per_game["w"]).sum() / per_game["w"].sum())
    num, den = _wsum_by(per_game["posteam"], per_game["drives"] - lg, per_game["w"])
    team = (_shrunk_mean(num.reindex(teams).fillna(0.0),
                         den.reindex(teams).fillna(0.0), PACE_PRIOR_N) + lg)
    sd = float(per_game["drives"].std())
    return dict(_LEAGUE_=lg, team=team, sd=max(sd, 0.5))


def _strength_of_schedule(d: pd.DataFrame, teams: list, off: pd.Series,
                          dfn: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Average opponent rating each unit actually faced.

    `sched_off` is the mean defensive rating this offense played (positive =
    faced weak defenses, so its raw numbers flatter it); `sched_def` is the mean
    offensive rating this defense played.
    """
    num, den = _wsum_by(d["posteam"], d["defteam"].map(dfn).astype(float), d["w"])
    sched_off = (num / den).reindex(teams).fillna(0.0)
    num, den = _wsum_by(d["defteam"], d["posteam"].map(off).astype(float), d["w"])
    sched_def = (num / den).reindex(teams).fillna(0.0)
    return sched_off, sched_def


def _other_scoring(d: pd.DataFrame, games: pd.DataFrame | None,
                   lg_pace: float, lg_def_score: float) -> float:
    """Points per team-game that belong to no drive at all.

    Kick and punt return touchdowns, and safeties scored by the defense, never
    appear in the drive table, so a drive-only model runs a couple of points
    light per team. Measure that residual against the real final scores instead
    of guessing it, and let the engine add it back. Returns 0.0 when no game
    scores are supplied (the ratings are then relative-only).
    """
    if games is None or len(games) == 0:
        return 0.0
    tg = D.team_game_points(games)
    if tg.empty:
        return 0.0
    actual = float(tg["points"].mean())
    drive_pts = float(d.groupby(["season", "game_id", "posteam"])["points"].sum().mean())
    def_pts = float(lg_def_score) * float(lg_pace) * TD_POINTS
    return float(np.clip(actual - drive_pts - def_pts, 0.0, 4.0))


def _raw_rates(d: pd.DataFrame, teams: list) -> pd.DataFrame:
    """Unadjusted points per drive for and against — for display and as a
    sanity check on how much the opponent adjustment actually moved."""
    o_num, o_den = _wsum_by(d["posteam"], d["points"], d["w"])
    d_num, d_den = _wsum_by(d["defteam"], d["points"], d["w"])
    return pd.DataFrame({
        "off_ppd_raw": (o_num / o_den).reindex(teams),
        "def_ppd_raw": (d_num / d_den).reindex(teams),
        "off_drives": o_den.reindex(teams).fillna(0.0),
    })


# ---------------------------------------------------------------------------
# Matchup: what this offense does against that defense
# ---------------------------------------------------------------------------

def matchup(r: dict, offense: str, defense: str) -> dict:
    """Expected per-drive production for one offense against one defense.

    The outcome mix is rescaled so that its point value equals the expected
    points per drive from the efficiency ratings — the reconciliation the drive
    engine relies on (§4.3).
    """
    if offense not in r["off"].index:
        raise ValueError(f"No rating for offense {offense!r}.")
    if defense not in r["dfn"].index:
        raise ValueError(f"No rating for defense {defense!r}.")

    ppd = float(np.clip(r["lg_ppd"] + r["off"][offense] + r["dfn"][defense], *PPD_CLIP))

    p_td = float(np.clip(r["lg_td"] + r["off_td"][offense] + r["def_td"][defense], *TD_CLIP))
    p_fg = float(np.clip(r["lg_fg"] + r["off_fg"][offense] + r["def_fg"][defense], *FG_CLIP))
    p_to = float(np.clip(r["lg_to"] + r["off_to"][offense] + r["def_to"][defense], *TO_CLIP))

    # Reconcile the mix with the points rating: scale TD and FG rates together
    # so 6.96*p_td + 3*p_fg lands on `ppd`, then re-clip and renormalise.
    mix_pts = TD_POINTS * p_td + FG_POINTS * p_fg
    if mix_pts > 1e-6:
        k = ppd / mix_pts
        p_td = float(np.clip(p_td * k, *TD_CLIP))
        p_fg = float(np.clip(p_fg * k, *FG_CLIP))
    scoring = p_td + p_fg
    if scoring + p_to > 0.97:                      # leave room for punts
        room = 0.97 - p_to
        p_td *= room / scoring
        p_fg *= room / scoring
    p_none = max(1.0 - p_td - p_fg - p_to, 0.0)

    return dict(
        offense=offense, defense=defense,
        ppd=ppd, mix_ppd=TD_POINTS * p_td + FG_POINTS * p_fg,
        p_td=p_td, p_fg=p_fg, p_turnover=p_to, p_none=p_none,
        def_score_rate=float(r["def_score_rate"][defense]),
        off_rating=float(r["off"][offense]), def_rating=float(r["dfn"][defense]),
    )


def game_pace(r: dict, team_a: str, team_b: str) -> dict:
    """Expected drives per team for a game between two teams, plus the spread.

    Both teams get very nearly the same number of possessions, so the pairing's
    pace is the average of the two teams' own, shrunk toward league.
    """
    pa = float(r["pace"].get(team_a, r["lg_pace"]))
    pb = float(r["pace"].get(team_b, r["lg_pace"]))
    mean = 0.5 * (pa + pb)
    return dict(mean=float(np.clip(mean, 7.0, 14.0)), sd=float(r["pace_sd"]),
                pace_a=pa, pace_b=pb)


def expected_points(r: dict, team_a: str, team_b: str,
                    home: str | None = None, avail: dict | None = None,
                    wind=None, roof=None) -> dict:
    """A first, non-simulated read on the game: expected points for each team.

    `home` names which side is at home ("a", "b", or a team abbreviation); pass
    None for a neutral site. `avail` is `{team: availability indices}` (see
    `availability.py`): the margin is shifted by the QB-familiarity and
    defensive-starter differences with fitted coefficients, half to each side.
    This is the deterministic check that the ratings behave — the drive engine
    (§4.2) replaces it with a simulation, but the means should agree.
    """
    pace = game_pace(r, team_a, team_b)
    a = matchup(r, team_a, team_b)
    b = matchup(r, team_b, team_a)
    other = float(r.get("lg_other_ppg", 0.0))
    # A defensive score belongs to the team whose DEFENSE made it. `a` is team_a
    # on offense facing team_b's defense, so a["def_score_rate"] is team_b's
    # defense — it scores for team_b, not team_a. Crossing these over is an easy
    # mistake and shows up as the two teams' totals being swapped by ~0.7 points.
    pts_a = (a["ppd"] * pace["mean"]
             + b["def_score_rate"] * pace["mean"] * TD_POINTS + other)
    pts_b = (b["ppd"] * pace["mean"]
             + a["def_score_rate"] * pace["mean"] * TD_POINTS + other)

    # Availability (§8.5): an unfamiliar QB or missing defensive starters move
    # the MARGIN (fitted; no effect on the total), split like home field.
    shift_a = shift_b = 0.0
    if avail:
        from . import availability as AV
        shift_a = 0.5 * AV.margin_shift(avail.get(team_a), avail.get(team_b))
        shift_b = -shift_a
        pts_a, pts_b = pts_a + shift_a, pts_b + shift_b

    # Weather moves the total, split evenly; the margin is untouched.
    wx = 0.5 * weather_total_shift(wind, roof)
    pts_a, pts_b = pts_a + wx, pts_b + wx

    # Home field is split evenly: the home side gains half, the road side loses
    # half, so the total is untouched and only the margin moves.
    half = 0.5 * float(r.get("hfa", HFA_DEFAULT))
    side = ("a" if home in ("a", team_a) else
            "b" if home in ("b", team_b) else None)
    if side == "a":
        pts_a, pts_b = pts_a + half, pts_b - half
    elif side == "b":
        pts_a, pts_b = pts_a - half, pts_b + half

    return dict(team_a=team_a, team_b=team_b, drives=pace["mean"], home=side,
                points_a=float(pts_a), points_b=float(pts_b),
                margin=float(pts_a - pts_b), total=float(pts_a + pts_b),
                avail_shift_a=float(shift_a), avail_shift_b=float(shift_b),
                weather_shift=float(2 * wx), a=a, b=b)


def strength_table(r: dict) -> pd.DataFrame:
    """Every team's ratings, sorted by net strength — the layer's headline view."""
    t = pd.DataFrame({
        "off_ppd": r["lg_ppd"] + r["off"],
        "def_ppd": r["lg_ppd"] + r["dfn"],
        "off_adj": r["off"], "def_adj": r["dfn"],
        "net": r["off"] - r["dfn"],
        "pace": r["pace"],
        "off_ppd_raw": r["raw"]["off_ppd_raw"],
        "def_ppd_raw": r["raw"]["def_ppd_raw"],
        "sched_off": r["sched_off"],
        "sched_def": r["sched_def"],
    })
    return t.sort_values("net", ascending=False)


if __name__ == "__main__":
    seasons = (2024, 2025)
    print("Loading play-by-play (first run downloads a few tens of MB)...")
    dr = D.load_drives(seasons)
    r = team_ratings(dr, D.load_games(seasons))
    print(f"{r['drives']:,} live drives, seasons {r['seasons']} | "
          f"league {r['lg_ppd']:.3f} pts/drive, {r['lg_pace']:.2f} drives/team-game, "
          f"TD {r['lg_td']:.1%} FG {r['lg_fg']:.1%} TO {r['lg_to']:.1%}")
    print(f"scoring level: {r['lg_ppd'] * r['lg_pace']:.2f} offensive + "
          f"{r['lg_def_score'] * r['lg_pace'] * TD_POINTS:.2f} defensive + "
          f"{r['lg_other_ppg']:.2f} return/safety = "
          f"{r['lg_ppd'] * r['lg_pace'] + r['lg_def_score'] * r['lg_pace'] * TD_POINTS + r['lg_other_ppg']:.2f} "
          f"pts/team-game")

    t = strength_table(r)
    show = ["off_ppd", "def_ppd", "net", "pace", "sched_off"]
    print("\nTop 6 by net rating:")
    print(t.head(6)[show].to_string(float_format=lambda x: f"{x:6.3f}"))
    print("\nBottom 4:")
    print(t.tail(4)[show].to_string(float_format=lambda x: f"{x:6.3f}"))

    print(f"home-field advantage fitted at {r['hfa']:+.2f} pts of margin")

    for a, b in (("BAL", "SF"), ("DET", "CHI")):
        e = expected_points(r, a, b, home="a")
        print(f"\n{a} (home) vs {b}: {e['points_a']:.1f} - {e['points_b']:.1f} "
              f"(margin {e['margin']:+.1f}, total {e['total']:.1f}, "
              f"{e['drives']:.1f} drives each)")
        for side in ("a", "b"):
            m = e[side]
            print(f"  {m['offense']} offense: {m['ppd']:.2f} pts/drive | "
                  f"TD {m['p_td']:.1%} FG {m['p_fg']:.1%} TO {m['p_turnover']:.1%} "
                  f"| mix checks to {m['mix_ppd']:.2f}")
