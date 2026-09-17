"""
nflsim/enginetest.py — the two game engines scored on PLAYERS, out of sample.

The backtest harness scores the single-stat models; the play engine's stage 4
had never been scored on anyone. This runs both engines on the same games with
the same as-of inputs — ratings refit on games before the week, rosters from
the depth chart as of kickoff, priors from prior games only, injuries as of
that week — and scores every player who actually played against his simulated
receiving yards, rushing yards and receptions. Same rosters, same means; the
engines differ only in how team volume and its split are generated, so any
gap is the engine's.

Not included, for both engines alike: availability shifts, goal-line roles,
team tendencies (they use the season being scored). Absolute numbers are
therefore a touch worse than the production pages; the comparison is fair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtest as B
from . import data as D
from . import game as G
from . import roster as RO
from . import rushing as R
from . import qb as Q
from . import teams as T

STATS = {
    "rec_yards": ("rec_yards", "receiving_yards"),
    "rush_yards": ("rush_yards", "rushing_yards"),
    "receptions": ("receptions", "receptions"),
}
MIN_EXPECTED = dict(rec_yards=8.0, rush_yards=8.0, receptions=1.0)   # who is worth scoring


def _asof_context(S: int, w: int, feeds: dict, recency) -> dict | None:
    """Ratings, volumes and defensive profiles from games before (S, w)."""
    wk_prior = B.before(feeds["wk"], S, w)
    if wk_prior.empty:
        return None
    wk_prior = D.game_weights(wk_prior, "recent_team", recency)
    d = D.game_weights(B.before(feeds["drives"], S, w), "posteam", recency)
    pl = (D.game_weights(B.before(feeds["plays"], S, w), "posteam", recency)
          if feeds["plays"] is not None and not feeds["plays"].empty else None)
    try:
        ratings = T.team_ratings(d, B.before(feeds["games"], S, w), plays=pl)
    except ValueError:
        return None
    ratings["pass_def"] = D.def_pass_rates(wk_prior)
    pfr_prior = D.game_weights(B.before(feeds["pfr"], S, w), "team", recency)
    pbp_prior = D.game_weights(B.before(feeds["pbp"], S, w), "defteam", recency)
    agg, pfr_lg = R.pfr_rush_aggregates(pfr_prior)
    snaps = RO.snap_roles(B.before(feeds["snaps"], S, w), recency) if not feeds["snaps"].empty else None
    return dict(
        wk=wk_prior, ratings=ratings, pfr_agg=agg, pfr_lg=pfr_lg, snaps=snaps, goal_line={},
        pass_vol=G.team_pass_volume(wk_prior), rush_vol=R.team_rush_volume(wk_prior),
        rush_def=R.rush_defense_profiles(wk_prior, pfr_prior, pbp_prior),
        lg_pass=Q.league_pass_rates(wk_prior),
        target_rate=float(np.clip(wk_prior["targets"].sum() / max(wk_prior["attempts"].sum(), 1), 0.85, 1.0)),
        avail={}, play_tendencies=None,
    )


def _roster(ctx: dict, feeds: dict, team: str, S: int, w: int, kickoff) -> pd.DataFrame | None:
    snap = D.depth_chart_snapshot(feeds["depth"], team, as_of=kickoff)
    if snap.empty:
        return None
    out = D.players_ruled_out(feeds["inj"], team, season=S, week=w)
    status = RO.injury_status(feeds["inj"], team, week=w)
    try:
        r = RO.build_roster(ctx["wk"], snap, team, out, ctx["pfr_agg"], ctx["pfr_lg"],
                            status=status, snaps=ctx["snaps"], goal_line=None)
    except ValueError:
        return None
    return r[r["active"]].reset_index(drop=True)


def _score_side(rows: list, side: dict, actual: pd.DataFrame, engine: str,
                S: int, w: int, game_id: str, team: str, opp: str) -> None:
    r = side["roster"].reset_index(drop=True)
    act = actual.set_index(actual["player_id"].astype(str))
    arrays = dict(rec_yards=side["rec_yards"], rush_yards=side["rush_yards"],
                  receptions=side["receptions"])
    for j, pl in r.iterrows():
        pid = str(pl["player_id"])
        if pid not in act.index:
            continue               # did not play (or no stat line): no score
        a = act.loc[pid]
        if isinstance(a, pd.DataFrame):
            a = a.iloc[0]
        for stat, (_, col) in STATS.items():
            x = np.asarray(arrays[stat][:, j], float)
            if x.mean() < MIN_EXPECTED[stat]:
                continue
            y = float(a[col])
            rows.append(dict(
                engine=engine, stat=stat, season=S, week=w, game_id=game_id, team=team, opp=opp,
                player_id=pid, name=pl["name"], position=pl["position"], depth=int(pl["depth"]),
                actual=y, pred_mean=float(x.mean()), pred_median=float(np.median(x)),
                p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)),
                sd=float(x.std()), crps=B._crps(x, y),
                pin90=_pinball(np.percentile(x, 90), y, 0.9), pin10=_pinball(np.percentile(x, 10), y, 0.1),
            ))


def _pinball(q: float, y: float, tau: float) -> float:
    d = y - q
    return float(tau * d if d >= 0 else (tau - 1) * d)


def run(score_season: int = 2025, weeks=None, n_prior: int = 2,
        recency: D.Recency = D.RECENCY_DEFAULT, n_sims: int = 2000,
        games_per_week: int | None = None, seed: int = 0,
        engines=("drive", "play"), progress=None) -> pd.DataFrame:
    """One row per (engine, stat, player-game). `games_per_week` samples the
    slate to keep the play engine's runtime in check."""
    seasons = B.window([score_season], n_prior)
    wk_all = D.load_weekly(seasons)
    feeds = dict(
        wk=wk_all, drives=D.load_drives(seasons), games=D.load_games(seasons),
        plays=D.load_plays(seasons) if T.RATING_METHOD in ("epa", "blend") else None,
        pfr=D.load_pfr_rush(seasons), pbp=D.load_pbp(seasons),
        snaps=D.load_snap_counts(seasons),
        depth=D.load_depth_charts((score_season,)), inj=D.load_injuries((score_season,)),
    )
    sched = D.load_schedule((score_season,))
    if "play" in engines:
        from . import playengine as PE
        probe = dict(depth_seasons=(score_season - 1,))   # tendencies are not used; nothing to leak
        PE.attach(probe)
        tables, sens = probe["play_engine"]
    rng = np.random.default_rng(seed)
    steps = list(B._score_weeks(sched, [score_season], weeks))
    rows = []
    for k, (S, w) in enumerate(steps):
        if progress:
            progress(k / max(len(steps), 1), f"{S} week {w}")
        ctx = _asof_context(S, w, feeds, recency)
        if ctx is None:
            continue
        if "play" in engines:
            ctx["play_engine"] = (tables, sens)
        wk_games = sched[(sched["season"] == S) & (sched["week"] == w) & sched["played"]]
        if games_per_week and len(wk_games) > games_per_week:
            wk_games = wk_games.sample(n=games_per_week, random_state=int(rng.integers(1 << 31)))
        this = wk_all[(wk_all["season"] == S) & (wk_all["week"] == w)]
        for _, gm in wk_games.iterrows():
            h, a = gm["home_team"], gm["away_team"]
            if h not in ctx["ratings"]["off"].index or a not in ctx["ratings"]["off"].index:
                continue
            rh, ra = _roster(ctx, feeds, h, S, w, gm["kickoff"]), _roster(ctx, feeds, a, S, w, gm["kickoff"])
            if rh is None or ra is None:
                continue
            gseed = int(rng.integers(1 << 31))
            for eng in engines:
                try:
                    sim = G.run_game(ctx, rh, ra, h, a, n_sims=n_sims, seed=gseed, home="a",
                                     wind=gm.get("wind"), roof=gm.get("roof"), engine=eng)
                except Exception as e:      # one bad game must not kill the run
                    rows.append(dict(engine=eng, stat="error", season=S, week=w, game_id=gm["game_id"],
                                     name=str(e)[:120]))
                    continue
                _score_side(rows, sim["box_a"], this[this["recent_team"] == h], eng, S, w, gm["game_id"], h, a)
                _score_side(rows, sim["box_b"], this[this["recent_team"] == a], eng, S, w, gm["game_id"], a, h)
    return pd.DataFrame(rows)


def metrics(res: pd.DataFrame) -> pd.DataFrame:
    """Engine-by-stat scorecard on the player-games BOTH engines scored."""
    res = res[res["stat"] != "error"]
    key = ["stat", "game_id", "player_id"]
    both = res.groupby(key)["engine"].nunique()
    both = both[both == res["engine"].nunique()].index
    r = res.set_index(key).loc[both].reset_index()
    out = []
    for (stat, eng), d in r.groupby(["stat", "engine"]):
        out.append(dict(
            stat=stat, engine=eng, n=len(d),
            mae_mean=float((d["pred_mean"] - d["actual"]).abs().mean()),
            mae_median=float((d["pred_median"] - d["actual"]).abs().mean()),
            bias_mean=float((d["pred_mean"] - d["actual"]).mean()),
            crps=float(d["crps"].mean()),
            corr=float(np.corrcoef(d["pred_mean"], d["actual"])[0, 1]) if len(d) > 2 else np.nan,
            over_median=float((d["actual"] > d["pred_median"]).mean()),
            cover80=float(((d["actual"] >= d["p10"]) & (d["actual"] <= d["p90"])).mean()),
            above_p90=float((d["actual"] > d["p90"]).mean()),
            below_p10=float((d["actual"] < d["p10"]).mean()),
            pinball90=float(d["pin90"].mean()), pinball10=float(d["pin10"].mean()),
            sd=float(d["sd"].mean()),
        ))
    return pd.DataFrame(out).sort_values(["stat", "engine"]).reset_index(drop=True)


def by_tier(res: pd.DataFrame, stat: str, bins=(0, 20, 40, 60, 500)) -> pd.DataFrame:
    """Calibration by projection tier — where the engines' shapes differ most."""
    d = res[(res["stat"] == stat)].copy()
    d["tier"] = pd.cut(d["pred_mean"], bins)
    return (d.groupby(["tier", "engine"], observed=True)
             .agg(n=("actual", "size"), mae=("pred_mean", lambda x: (x - d.loc[x.index, "actual"]).abs().mean()),
                  bias=("pred_mean", lambda x: (x - d.loc[x.index, "actual"]).mean()),
                  over_median=("actual", lambda y: (y > d.loc[y.index, "pred_median"]).mean()),
                  above_p90=("actual", lambda y: (y > d.loc[y.index, "p90"]).mean()),
                  crps=("crps", "mean"))
             .reset_index())
