"""
nflsim/backtest.py — the out-of-sample harness (roadmap §8.1).

Every number in the roadmap's validation tables was fitted and scored on the
same games. This module scores the models the way they are actually used: for
each week `w` of a scored season, everything is fitted on games played BEFORE
that week (earlier seasons in the window plus weeks < w of the current one),
then week `w` is predicted and compared with what happened.

Two layers are scored:

  * TEAM — the team-strength layer (`teams.py`): expected margin and total per
    game from `expected_points`, scored by RMSE / MAE / bias, straight-up
    accuracy, and log-loss / Brier on P(home win) with a normal margin
    approximation. The closing line from the schedule is scored on the same
    games as the benchmark, plus the model's record against the spread.

  * PLAYER — the single-stat models (receiving yards, rushing yards,
    touchdowns, sacks, interceptions) for every player who played that week
    and had enough prior history, using the history-only path (no depth chart,
    no injury report — those feeds are not reliably reconstructable for past
    weeks). Each prediction is a full simulated distribution, scored by MAE /
    RMSE / bias of the mean, coverage of the 10–90 band, whether the median
    really splits outcomes 50/50, CRPS, and — for count stats — the Brier score
    of P(at least one). A trailing weighted average of the stat is the naive
    baseline every model has to beat.

The harness takes a `Recency` so the presets can be compared honestly; it is
the tool for tuning any constant in this codebase.
"""

from __future__ import annotations

from math import erf, sqrt

import numpy as np
import pandas as pd

from . import data as D
from . import teams as T
from . import rushing as R
from . import touchdowns as TD
from . import qb as Q

# Normal approximation to the margin distribution for win probabilities. The
# drive engine's simulated margin sd is ~14.2 and the 2024-25 residual sd 12.7;
# 13.5 sits between them. `team_metrics` also reports the fitted residual sd.
MARGIN_SD = 13.5

# Which players are worth scoring: enough prior games to have priors, and
# enough expected volume that the prop would exist.
MIN_PRIOR_GAMES = 5
MIN_EXP_TARGETS = 3.0
MIN_EXP_CARRIES = 5.0
MIN_EXP_TOUCHES = 4.0
MIN_EXP_ATTEMPTS = 15.0

# Goal-line role (8.2) in the touchdown priors; False scores the old positional-
# mean regression for A/B.
USE_GOAL_LINE = True

PLAYER_STATS = {
    "rec_yards": dict(label="Receiving yards", positions=("WR", "TE", "RB"), count=False),
    "rush_yards": dict(label="Rushing yards", positions=("RB", "QB", "WR", "FB"), count=False),
    "tds": dict(label="Touchdowns (rush + rec)", positions=("RB", "WR", "TE", "QB", "FB"), count=True),
    "sacks": dict(label="Sacks taken", positions=("QB",), count=True),
    "ints": dict(label="Interceptions thrown", positions=("QB",), count=True),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_cdf(x):
    x = np.asarray(x, float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def before(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Rows from games played strictly before (season, week)."""
    if df is None or df.empty:
        return df
    s, w = df["season"].astype(int), df["week"].astype(int)
    return df[(s < season) | ((s == season) & (w < week))]


def window(score_seasons, n_prior: int) -> tuple[int, ...]:
    """Seasons to load: `n_prior` before the earliest scored season, through the latest."""
    return tuple(range(min(score_seasons) - int(n_prior), max(score_seasons) + 1))


def _crps(samples: np.ndarray, y: float) -> float:
    """Continuous ranked probability score from samples (lower is better)."""
    x = np.sort(np.asarray(samples, float))
    n = len(x)
    if n == 0:
        return np.nan
    term1 = float(np.abs(x - y).mean())
    i = np.arange(1, n + 1)
    term2 = float((2.0 / (n * n)) * np.sum((2 * i - n - 1) * x))
    return term1 - 0.5 * term2


def _score_weeks(sched: pd.DataFrame, score_seasons, weeks=None):
    for S in sorted(int(s) for s in score_seasons):
        played = sched[(sched["season"] == S) & sched["played"]]
        for w in sorted(played["week"].unique()):
            if weeks and int(w) not in weeks:
                continue
            yield S, int(w)


# ---------------------------------------------------------------------------
# Team layer
# ---------------------------------------------------------------------------

def team_backtest(score_seasons, n_prior: int = 2,
                  recency: D.Recency = D.RECENCY_DEFAULT,
                  weeks=None, progress=None, availability: bool = False) -> pd.DataFrame:
    """One row per scored game: model margin/total vs actual and the closing line.

    Ratings (and home field, and the scoring-level calibration) are refitted
    for every week on drives and finals from before that week only. With
    `availability`, the QB-familiarity / defensive-starter margin shift
    (`availability.py`, as knowable before each kickoff) is applied and the
    indices are kept in the output.
    """
    seasons = window(score_seasons, n_prior)
    drives = D.load_drives(seasons)
    games = D.load_games(seasons)
    sched = D.load_schedule(tuple(int(s) for s in score_seasons))
    if drives.empty or sched.empty:
        return pd.DataFrame()

    steps = list(_score_weeks(sched, score_seasons, weeks))
    rows = []
    for k, (S, w) in enumerate(steps):
        if progress:
            progress(k / max(len(steps), 1), f"teams {S} week {w}")
        d = before(drives, S, w)
        if d.empty:
            continue
        d = D.game_weights(d, "posteam", recency)
        try:
            r = T.team_ratings(d, before(games, S, w))
        except ValueError:
            continue
        wk_games = sched[(sched["season"] == S) & (sched["week"] == w) & sched["played"]]
        for _, gm in wk_games.iterrows():
            h, a = gm["home_team"], gm["away_team"]
            if h not in r["off"].index or a not in r["off"].index:
                continue
            e = T.expected_points(r, h, a, home="a")
            rows.append(dict(
                season=S, week=w, game_id=gm["game_id"], home=h, away=a,
                pred_home=e["points_a"], pred_away=e["points_b"],
                pred_margin=e["margin"], pred_total=e["total"],
                actual_home=float(gm["home_score"]), actual_away=float(gm["away_score"]),
                actual_margin=float(gm["home_score"] - gm["away_score"]),
                actual_total=float(gm["home_score"] + gm["away_score"]),
                line_margin=gm.get("spread_line", np.nan),
                line_total=gm.get("total_line", np.nan),
                drives_fit=int(r["drives"]), hfa=float(r["hfa"]),
            ))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if availability:
        from . import availability as AV
        idx = AV.historical_indices(score_seasons, n_prior, recency, weeks, progress)
        key = idx.set_index(["game_id", "team"])
        shift, qb_h, qb_a, df_h, df_a = [], [], [], [], []
        for g, h, a in zip(out["game_id"], out["home"], out["away"]):
            try:
                ih, ia = key.loc[(g, h)].to_dict(), key.loc[(g, a)].to_dict()
            except KeyError:
                ih, ia = {}, {}
            shift.append(0.5 * AV.margin_shift(ih, ia))
            qb_h.append(ih.get("qb_idx", np.nan)); qb_a.append(ia.get("qb_idx", np.nan))
            df_h.append(ih.get("def_idx", np.nan)); df_a.append(ia.get("def_idx", np.nan))
        shift = np.array(shift)
        out["pred_home"] += shift
        out["pred_away"] -= shift
        out["pred_margin"] = out["pred_home"] - out["pred_away"]
        out["avail_shift"] = 2 * shift
        out["qb_idx_home"], out["qb_idx_away"] = qb_h, qb_a
        out["def_idx_home"], out["def_idx_away"] = df_h, df_a
    out["p_home"] = _norm_cdf(out["pred_margin"] / MARGIN_SD)
    out["line_p_home"] = _norm_cdf(out["line_margin"] / MARGIN_SD)
    return out


def _margin_scores(pred: pd.Series, p_home: pd.Series, bt: pd.DataFrame) -> dict:
    actual = bt["actual_margin"]
    resid = pred - actual
    decided = actual != 0
    y = (actual > 0).astype(float)
    p = p_home.clip(1e-6, 1 - 1e-6)
    return dict(
        games=int(len(bt)),
        margin_rmse=float(np.sqrt((resid ** 2).mean())),
        margin_mae=float(resid.abs().mean()),
        margin_bias=float(resid.mean()),
        margin_corr=float(np.corrcoef(pred, actual)[0, 1]) if len(bt) > 2 else np.nan,
        resid_sd=float(resid.std(ddof=1)) if len(bt) > 2 else np.nan,
        su_acc=float((np.sign(pred[decided]) == np.sign(actual[decided])).mean()),
        log_loss=float(-(y[decided] * np.log(p[decided])
                         + (1 - y[decided]) * np.log(1 - p[decided])).mean()),
        brier=float(((p[decided] - y[decided]) ** 2).mean()),
    )


def team_metrics(bt: pd.DataFrame) -> pd.DataFrame:
    """Model vs closing line on the same games; one row each."""
    if bt is None or bt.empty:
        return pd.DataFrame()
    b = bt.dropna(subset=["actual_margin"])
    model = _margin_scores(b["pred_margin"], b["p_home"], b)
    tot = b["pred_total"] - b["actual_total"]
    model.update(total_rmse=float(np.sqrt((tot ** 2).mean())), total_bias=float(tot.mean()))

    has_line = b.dropna(subset=["line_margin"])
    # against the spread: back the home side when the model likes it more than the market
    live = has_line[has_line["actual_margin"] != has_line["line_margin"]]
    pick_home = live["pred_margin"] > live["line_margin"]
    home_covers = live["actual_margin"] > live["line_margin"]
    model.update(ats_pct=float((pick_home == home_covers).mean()) if len(live) else np.nan,
                 ats_n=int(len(live)))

    rows = {"Model": model}
    if len(has_line):
        mk = _margin_scores(has_line["line_margin"], has_line["line_p_home"], has_line)
        tl = has_line.dropna(subset=["line_total"])
        tot = tl["line_total"] - tl["actual_total"]
        mk.update(total_rmse=float(np.sqrt((tot ** 2).mean())), total_bias=float(tot.mean()),
                  ats_pct=np.nan, ats_n=0)
        rows["Closing line"] = mk
    return pd.DataFrame(rows).T


def calibration(bt: pd.DataFrame, col: str = "p_home", bins: int = 8) -> pd.DataFrame:
    """Observed home-win rate by predicted-probability bin."""
    b = bt[bt["actual_margin"] != 0].copy()
    b["bin"] = pd.cut(b[col], np.linspace(0, 1, bins + 1), include_lowest=True)
    g = (b.groupby("bin", observed=True)
           .agg(n=(col, "size"), predicted=(col, "mean"),
                observed=("actual_margin", lambda m: float((m > 0).mean())))
           .reset_index())
    return g[g["n"] > 0]


def weekly(bt: pd.DataFrame) -> pd.DataFrame:
    """Margin RMSE per scored week, model and line side by side."""
    def f(g):
        return pd.Series(dict(
            games=len(g),
            model_rmse=float(np.sqrt(((g["pred_margin"] - g["actual_margin"]) ** 2).mean())),
            line_rmse=float(np.sqrt(((g["line_margin"] - g["actual_margin"]) ** 2).mean())),
            model_su=float((np.sign(g["pred_margin"]) == np.sign(g["actual_margin"])).mean()),
        ))
    return bt.groupby(["season", "week"]).apply(f).reset_index()


# ---------------------------------------------------------------------------
# Player layer
# ---------------------------------------------------------------------------

def _prior_context(stat: str, wk_prior: pd.DataFrame, pfr_prior, pbp_prior,
                   touches_prior=None) -> dict:
    """The league / defense / volume inputs each model needs, from prior games only."""
    if stat == "rec_yards":
        import model as M
        rec = wk_prior[wk_prior["position"].isin(M.RECEIVING_POSITIONS)]
        return dict(M=M, tv=M.team_pass_volume(rec), defs=M.defense_profiles(rec),
                    lg=M.league_priors(rec))
    if stat == "rush_yards":
        agg, lg = R.pfr_rush_aggregates(pfr_prior)
        return dict(agg=agg, lg=lg, tv=R.team_rush_volume(wk_prior),
                    defs=R.rush_defense_profiles(wk_prior, pfr_prior, pbp_prior),
                    rush_lg=R.league_rush_priors(wk_prior))
    if stat == "tds":
        gl = (TD.goal_line_profiles(touches_prior, wk_prior)
              if USE_GOAL_LINE and touches_prior is not None and not touches_prior.empty else None)
        return dict(lg=TD.league_td_rates(wk_prior), defs=TD.td_defense_profiles(wk_prior), gl=gl)
    if stat in ("sacks", "ints"):
        return dict(lg=Q.league_pass_rates(wk_prior), defs=D.def_pass_rates(wk_prior))
    raise KeyError(stat)


def _predict(stat: str, ctx: dict, wk_prior: pd.DataFrame, row: pd.Series,
             n_sims: int, seed: int):
    """Simulated samples for one player-game, or None if he doesn't qualify."""
    pid, team, opp = str(row["player_id"]), row["recent_team"], row["opponent_team"]
    if stat == "rec_yards":
        M = ctx["M"]
        pri = M.player_priors(wk_prior, pid, ctx["lg"])
        tv = ctx["tv"].get(team, ctx["tv"]["_LEAGUE_"])
        # select on the player's own (unregressed) share so the candidate set
        # does not move when the regression constants are tuned
        if pri.get("raw_ts", pri["mu_ts"]) * tv[0] < MIN_EXP_TARGETS:
            return None
        return M.simulate(pri, tv, ctx["defs"].get((opp, pri["position"])),
                          n_sims=n_sims, seed=seed)["yards"]
    if stat == "rush_yards":
        pri = R.player_rush_priors(wk_prior, pid, ctx["agg"], ctx["lg"], lg=ctx["rush_lg"])
        tv = ctx["tv"].get(team, ctx["tv"]["_LEAGUE_"])
        if pri.get("raw_share", pri["mu_share"]) * tv[0] < MIN_EXP_CARRIES:
            return None
        return R.simulate(pri, tv, ctx["defs"].get(opp), n_sims=n_sims, seed=seed)["yards"]
    if stat == "tds":
        pri = TD.player_td_priors(wk_prior, pid, ctx["lg"], gl=ctx.get("gl"))
        if pri["mu_rec"] + pri["mu_car"] < MIN_EXP_TOUCHES:
            return None
        return TD.simulate(pri, ctx["defs"].get((opp, pri["position"])),
                           n_sims=n_sims, seed=seed)["total"]
    pri = Q.qb_priors(wk_prior, pid, ctx["lg"])
    if pri["mu_att"] < MIN_EXP_ATTEMPTS:
        return None
    dp = ctx["defs"].get(opp)
    if stat == "sacks":
        return Q.simulate_sacks(pri, dp, ctx["lg"], n_sims=n_sims, seed=seed)["count"]
    return Q.simulate_ints(pri, dp, ctx["lg"], n_sims=n_sims, seed=seed)["count"]


def _actual(stat: str, row: pd.Series) -> float:
    if stat == "rec_yards":
        return float(row["receiving_yards"])
    if stat == "rush_yards":
        return float(row["rushing_yards"])
    if stat == "tds":
        return float(row["receiving_tds"] + row["rushing_tds"])
    if stat == "sacks":
        return float(row["sacks"])
    return float(row["interceptions"])


def _naive(stat: str, h: pd.DataFrame) -> float:
    """Trailing recency-weighted mean of the stat: the baseline to beat."""
    if stat == "tds":
        x = h["receiving_tds"] + h["rushing_tds"]
    else:
        x = h[{"rec_yards": "receiving_yards", "rush_yards": "rushing_yards",
               "sacks": "sacks", "ints": "interceptions"}[stat]]
    return float(D.wmean(x, h["w"]))


def player_backtest(score_seasons, stats=("rec_yards", "rush_yards", "tds"),
                    n_prior: int = 2, recency: D.Recency = D.RECENCY_DEFAULT,
                    n_sims: int = 4000, min_games: int = MIN_PRIOR_GAMES,
                    weeks=None, seed: int = 0, progress=None) -> pd.DataFrame:
    """One row per (stat, player-game) with the simulated distribution's summary
    and the actual outcome. History-only path: priors, volumes and defensive
    profiles are rebuilt each week from games before it."""
    stats = [s for s in stats if s in PLAYER_STATS]
    seasons = window(score_seasons, n_prior)
    wk_all = D.load_weekly(seasons)
    need_rush = "rush_yards" in stats
    pfr_all = D.load_pfr_rush(seasons) if need_rush else pd.DataFrame()
    pbp_all = D.load_pbp(seasons) if need_rush else pd.DataFrame()
    touches_all = D.load_touches(seasons) if ("tds" in stats and USE_GOAL_LINE) else pd.DataFrame()
    sched = D.load_schedule(tuple(int(s) for s in score_seasons))
    rng = np.random.default_rng(seed)

    steps = list(_score_weeks(sched, score_seasons, weeks))
    rows = []
    for k, (S, w) in enumerate(steps):
        if progress:
            progress(k / max(len(steps), 1), f"players {S} week {w}")
        wk_prior = before(wk_all, S, w)
        if wk_prior.empty:
            continue
        wk_prior = D.game_weights(wk_prior, "recent_team", recency)
        pfr_prior = (D.game_weights(before(pfr_all, S, w), "team", recency)
                     if need_rush and not pfr_all.empty else pd.DataFrame())
        pbp_prior = (D.game_weights(before(pbp_all, S, w), "defteam", recency)
                     if need_rush and not pbp_all.empty else pd.DataFrame())
        touches_prior = (D.game_weights(before(touches_all, S, w), "posteam", recency)
                         if not touches_all.empty else None)
        this = wk_all[(wk_all["season"] == S) & (wk_all["week"] == w)]
        games_prior = wk_prior.groupby("player_id").size()

        for stat in stats:
            spec = PLAYER_STATS[stat]
            ctx = _prior_context(stat, wk_prior, pfr_prior, pbp_prior, touches_prior)
            cands = this[this["position"].isin(spec["positions"])]
            for _, row in cands.iterrows():
                pid = str(row["player_id"])
                if games_prior.get(pid, 0) < min_games:
                    continue
                try:
                    x = _predict(stat, ctx, wk_prior, row, n_sims, int(rng.integers(1 << 31)))
                except (ValueError, KeyError):
                    continue
                if x is None:
                    continue
                x = np.asarray(x, float)
                y = _actual(stat, row)
                med = float(np.median(x))
                rows.append(dict(
                    stat=stat, season=S, week=w, player_id=pid,
                    name=row["player_display_name"], position=row["position"],
                    team=row["recent_team"], opp=row["opponent_team"],
                    actual=y, pred_mean=float(x.mean()), pred_median=med,
                    p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)),
                    p_ge1=float((x >= 1).mean()), crps=_crps(x, y),
                    naive=_naive(stat, wk_prior[wk_prior["player_id"] == pid]),
                ))
    return pd.DataFrame(rows)


def player_metrics(pb: pd.DataFrame) -> pd.DataFrame:
    """One row per stat. `over_median` should sit near 0.50 and `cover80` near
    0.80 if the distributions are calibrated; `mae` vs `naive_mae` is whether
    the model beats a trailing average."""
    if pb is None or pb.empty:
        return pd.DataFrame()
    out = []
    for stat, g in pb.groupby("stat", sort=False):
        err = g["pred_mean"] - g["actual"]
        d = dict(
            stat=PLAYER_STATS[stat]["label"], n=int(len(g)),
            mae=float(err.abs().mean()), rmse=float(np.sqrt((err ** 2).mean())),
            bias=float(err.mean()),
            corr=float(np.corrcoef(g["pred_mean"], g["actual"])[0, 1]) if len(g) > 2 else np.nan,
            cover80=float(((g["actual"] >= g["p10"]) & (g["actual"] <= g["p90"])).mean()),
            over_median=float((g["actual"] > g["pred_median"]).mean()),
            crps=float(g["crps"].mean()),
            naive_mae=float((g["naive"] - g["actual"]).abs().mean()),
        )
        if PLAYER_STATS[stat]["count"]:
            y1 = (g["actual"] >= 1).astype(float)
            d["brier_ge1"] = float(((g["p_ge1"] - y1) ** 2).mean())
            d["base_rate_ge1"] = float(y1.mean())
        out.append(d)
    return pd.DataFrame(out).set_index("stat")


# ---------------------------------------------------------------------------
# Whole run
# ---------------------------------------------------------------------------

def run(score_seasons, presets: dict | None = None, stats=("rec_yards", "rush_yards", "tds"),
        n_prior: int = 2, n_sims: int = 4000, weeks=None, progress=None,
        availability: bool = False) -> dict:
    """Team and player backtests for each recency preset.

    Returns {"team": {preset: games df}, "player": {preset: rows df}}. With
    `availability`, the team layer is scored with the §8.5 margin shift and
    a "<preset> (no availability)" row is added for comparison.
    """
    presets = presets or {"Balanced": D.RECENCY_DEFAULT}
    out = dict(team={}, player={})
    n = len(presets)
    for i, (name, rec) in enumerate(presets.items()):
        def prog(f, txt, i=i):
            if progress:
                progress((i + f) / n, f"{name}: {txt}")
        bt = team_backtest(score_seasons, n_prior, rec, weeks, prog, availability=availability)
        out["team"][name] = bt
        if availability and not bt.empty:
            plain = bt.copy()
            plain["pred_home"] -= plain["avail_shift"] / 2
            plain["pred_away"] += plain["avail_shift"] / 2
            plain["pred_margin"] = plain["pred_home"] - plain["pred_away"]
            plain["p_home"] = _norm_cdf(plain["pred_margin"] / MARGIN_SD)
            out["team"][f"{name} (no availability)"] = plain
        if stats:
            out["player"][name] = player_backtest(score_seasons, stats, n_prior, rec,
                                                  n_sims, weeks=weeks, progress=prog)
    return out


def compare(results: dict, layer: str = "team") -> pd.DataFrame:
    """Headline metrics per preset, side by side."""
    rows = {}
    if layer == "team":
        for name, bt in results["team"].items():
            m = team_metrics(bt)
            if not m.empty:
                rows[name] = m.loc["Model"]
                if "Closing line" in m.index and "Closing line" not in rows:
                    rows["Closing line"] = m.loc["Closing line"]
        return pd.DataFrame(rows).T
    for name, pb in results["player"].items():
        m = player_metrics(pb)
        for stat, r in m.iterrows():
            rows[(stat, name)] = r
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    import sys
    seasons = [int(s) for s in sys.argv[1:]] or [2025]
    print(f"Scoring {seasons} out of sample (fit on the two seasons before + prior weeks)...")
    res = run(seasons, D.RECENCY_PRESETS, stats=("rec_yards", "rush_yards", "tds"),
              n_sims=2000, progress=lambda f, t: print(f"  {f:5.0%} {t}", end="\r"))
    print()
    pd.set_option("display.width", 160)
    print("\nTEAM — margin / total / winner, model vs closing line")
    print(compare(res, "team").to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nPLAYER — distribution accuracy by stat and preset")
    print(compare(res, "player").to_string(float_format=lambda x: f"{x:.3f}"))
