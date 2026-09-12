"""
nflsim/availability.py — who is actually playing (roadmap §8.5).

Team strength is a rating of a UNIT over its recent drives; it does not know
who is on the field this week. Two things move a unit most and are knowable
before kickoff from the depth chart and the injury report:

  * **The QB.** `qb_index` = 1 − the share of the team's (recency-weighted)
    dropbacks in the priors window that this week's starter took. An
    established starter scores ~0; a backup or a newly signed QB scores ~1 —
    the offense rating was built by someone else.

  * **The defense.** `def_index` = importance of the defensive starters ruled
    Out / Doubtful ÷ importance of all starters, where a starter's importance
    is his recency-weighted share of defensive snaps over the priors window
    (snap counts, joined to the depth chart by gsis id). Twelve starters —
    the base front seven plus the nickel — from the depth chart as of kickoff.

Both indices are scored on the out-of-sample residuals the backtest produces
(`backtest.team_backtest` + `historical_indices`, then `fit` / `fit_margin`),
so the coefficients below are FITTED, not guessed. They enter
`teams.expected_points` and the drive engine as a shift of the expected
MARGIN, split evenly across the two sides like home field — because that is
what the data says the effect is: a backup QB moves the margin (−8.8 points
per unit of index difference, t = −7.2 on 2024–25) and not the total (t < 1.2
either season). Fitted on 2024 and tested on 2025 the QB term alone takes
margin RMSE 13.38 → 12.92 and winners 59.8% → 63.1%; the reverse direction
agrees (13.66 → 12.88). The defensive term is directionally consistent and
monotone by bin but only t ≈ 2 pooled; it adds ~0.04 of RMSE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D

# Points of expected MARGIN per unit of index DIFFERENCE between the two
# teams, fitted on 2024-25 out-of-sample residuals (`fit_margin`, pooled
# n = 544 games). A side whose QB index is higher than its opponent's loses
# QB_MARGIN_COEF x (difference) of margin; a side whose defense is missing more
# starter importance gives up DEF_MARGIN_COEF x (difference).
QB_MARGIN_COEF = -8.84      # t = -7.2
DEF_MARGIN_COEF = 12.10     # t = +2.2

DEFAULT_SNAP_SHARE = 0.7     # a listed starter with no snap history
N_DEF_STARTERS = 12          # base front seven + secondary + nickel
OUT_STATUSES = ("Out", "Doubtful")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def snap_shares(snaps: pd.DataFrame, recency: D.Recency = D.RECENCY_DEFAULT,
                col: str = "defense_pct") -> dict:
    """player_id -> recency-weighted mean snap share over the games supplied."""
    if snaps is None or snaps.empty or col not in snaps.columns:
        return {}
    d = snaps[snaps[col] > 0]
    if d.empty:
        return {}
    d = D.game_weights(d, "team", recency)
    num = (d[col] * d["w"]).groupby(d["player_id"]).sum()
    den = d["w"].groupby(d["player_id"]).sum()
    return (num / den).to_dict()


def def_starters(depth_def: pd.DataFrame, team: str, as_of=None) -> pd.DataFrame:
    """The defensive starters (depth 1) on the team's chart as of a date."""
    snap = D.depth_chart_snapshot(depth_def, team, as_of)
    if snap.empty:
        return snap
    return snap[snap["depth"] == 1].head(N_DEF_STARTERS + 2).reset_index(drop=True)


def def_index(starters: pd.DataFrame, out_ids: set, share: dict) -> dict:
    """Share of the defensive starters' importance that is ruled out."""
    if starters is None or starters.empty:
        return dict(def_idx=0.0, def_missing=[], n_starters=0, known=False)
    imp = np.array([float(share.get(str(p), DEFAULT_SNAP_SHARE)) for p in starters["player_id"]])
    out = np.array([str(p) in out_ids for p in starters["player_id"]])
    total = float(imp.sum())
    missing = [f"{n} ({pos})" for n, pos, o in
               zip(starters["player_name"], starters["position"], out) if o]
    return dict(def_idx=float(imp[out].sum() / total) if total > 0 else 0.0,
                def_missing=missing, n_starters=int(len(starters)), known=True)


def starting_qb(depth_off: pd.DataFrame, team: str, out_ids: set, as_of=None):
    """(player_id, name) of the first QB on the chart who is not ruled out."""
    snap = D.depth_chart_snapshot(depth_off, team, as_of)
    if snap.empty:
        return None, None
    qbs = snap[snap["position"] == "QB"].sort_values("depth")
    for _, q in qbs.iterrows():
        if str(q["player_id"]) not in out_ids:
            return str(q["player_id"]), str(q.get("player_name") or q["player_id"])
    return None, None


def qb_index(wk_prior: pd.DataFrame, team: str, qb_id) -> dict:
    """1 − this QB's share of the team's weighted dropbacks in the window."""
    t = wk_prior[(wk_prior["recent_team"] == team) & (wk_prior["position"] == "QB")]
    col = "dropbacks" if "dropbacks" in t.columns else "attempts"
    w = t["w"] if "w" in t.columns else pd.Series(1.0, index=t.index)
    total = float((t[col] * w).sum())
    if qb_id is None or total <= 0:
        # no chart QB (index unknowable -> 0) or a team with no history (-> 1)
        return dict(qb_idx=1.0 if qb_id is not None else 0.0, qb_share=0.0)
    mine = t["player_id"].astype(str) == str(qb_id)
    share = float((t.loc[mine, col] * w[mine]).sum()) / total
    return dict(qb_idx=float(np.clip(1.0 - share, 0.0, 1.0)), qb_share=float(share))


def team_indices(team: str, wk_prior: pd.DataFrame, depth_off: pd.DataFrame,
                 depth_def: pd.DataFrame, out_ids: set, share: dict, as_of=None) -> dict:
    """Both indices for one team, plus the names behind them."""
    qb_id, qb_name = starting_qb(depth_off, team, out_ids, as_of)
    out = dict(team=team, qb_id=qb_id, qb_name=qb_name)
    out.update(qb_index(wk_prior, team, qb_id))
    out.update(def_index(def_starters(depth_def, team, as_of), out_ids, share))
    return out


def margin_shift(idx_a: dict | None, idx_b: dict | None) -> float:
    """Expected-margin shift for team A against team B (points, A minus B).
    Callers split it evenly: A gains half, B loses half, so the total is
    untouched and only the margin moves — the effect the data supports."""
    a, b = idx_a or {}, idx_b or {}
    dq = float(a.get("qb_idx", 0.0)) - float(b.get("qb_idx", 0.0))
    dd = float(b.get("def_idx", 0.0)) - float(a.get("def_idx", 0.0))
    return float(QB_MARGIN_COEF * dq + DEF_MARGIN_COEF * dd)


# ---------------------------------------------------------------------------
# Current week (for the engine) and historical weeks (for the fit)
# ---------------------------------------------------------------------------

def current_indices(live: dict, week: int | None = None) -> dict:
    """team -> indices for the upcoming games, from the live depth charts,
    the latest injury report and the priors window's snap counts.
    `live` is `roster.load_live(...)`."""
    seasons, depth_seasons = live["seasons"], live["depth_seasons"]
    recency = live.get("recency", D.RECENCY_DEFAULT)
    depth_def = D.load_depth_charts(depth_seasons, side="defense")
    snaps = D.load_snap_counts(tuple(sorted(set(seasons) | set(depth_seasons))))
    share = snap_shares(snaps, recency)
    inj = live["injuries"]
    out = {}
    teams = sorted(set(live["depth"]["team"].dropna())) if not live["depth"].empty else []
    for team in teams:
        ruled = D.players_ruled_out(inj, team, week=week)
        out[team] = team_indices(team, live["wk"], live["depth"], depth_def, ruled, share)
    return out


def historical_indices(score_seasons, n_prior: int = 2,
                       recency: D.Recency = D.RECENCY_DEFAULT,
                       weeks=None, progress=None) -> pd.DataFrame:
    """One row per (season, week, team) with the indices as they were knowable
    before kickoff: depth chart as of the game date, that week's injury
    report, snap shares and QB dropbacks from games before that week."""
    from . import backtest as B
    seasons = B.window(score_seasons, n_prior)
    wk_all = D.load_weekly(seasons)
    depth_off = D.load_depth_charts(tuple(int(s) for s in score_seasons), side="offense")
    depth_def = D.load_depth_charts(tuple(int(s) for s in score_seasons), side="defense")
    inj = D.load_injuries(tuple(int(s) for s in score_seasons))
    snaps = D.load_snap_counts(seasons)
    sched = D.load_schedule(tuple(int(s) for s in score_seasons))
    rows = []
    steps = list(B._score_weeks(sched, score_seasons, weeks))
    for k, (S, w) in enumerate(steps):
        if progress:
            progress(k / max(len(steps), 1), f"availability {S} week {w}")
        wk_prior = D.game_weights(B.before(wk_all, S, w), "recent_team", recency)
        share = snap_shares(B.before(snaps, S, w), recency)
        games = sched[(sched["season"] == S) & (sched["week"] == w)]
        for _, g in games.iterrows():
            as_of = pd.to_datetime(g["kickoff"], utc=True) if pd.notna(g["kickoff"]) else None
            for team in (g["home_team"], g["away_team"]):
                ruled = D.players_ruled_out(inj, team, season=S, week=w)
                r = team_indices(team, wk_prior, depth_off, depth_def, ruled, share, as_of)
                r.update(season=S, week=w, game_id=g["game_id"])
                rows.append(r)
    return pd.DataFrame(rows)


def fit_margin(bt: pd.DataFrame, idx: pd.DataFrame) -> dict:
    """OLS of the out-of-sample MARGIN residual (home minus away) on the QB
    index difference and the defensive index difference — the form the
    module's coefficients take. `bt` is `backtest.team_backtest` output, `idx`
    is `historical_indices`."""
    key = idx.set_index(["game_id", "team"])
    rows = []
    for _, g in bt.iterrows():
        try:
            h, a = key.loc[(g["game_id"], g["home"])], key.loc[(g["game_id"], g["away"])]
        except KeyError:
            continue
        rows.append(dict(resid=float(g["actual_margin"] - g["pred_margin"]),
                         dq=float(h["qb_idx"] - a["qb_idx"]),
                         dd=float(a["def_idx"] - h["def_idx"])))
    d = pd.DataFrame(rows)
    if len(d) < 30:
        return dict(n=int(len(d)))
    X = np.column_stack([np.ones(len(d)), d["dq"], d["dd"]])
    beta, *_ = np.linalg.lstsq(X, d["resid"].values, rcond=None)
    resid = d["resid"].values - X @ beta
    se = np.sqrt(np.diag(float((resid ** 2).sum() / max(len(d) - 3, 1)) * np.linalg.inv(X.T @ X)))
    return dict(n=int(len(d)), intercept=float(beta[0]),
                qb_coef=float(beta[1]), qb_t=float(beta[1] / se[1]),
                def_coef=float(beta[2]), def_t=float(beta[2] / se[2]), rows=d)


def fit(bt: pd.DataFrame, idx: pd.DataFrame) -> dict:
    """OLS of each side's out-of-sample POINTS residual on its own QB index and
    the opponent's defensive index — the diagnostic form (does each unit's
    availability move its own scoring?). `fit_margin` is what the coefficients
    use. Returns coefficients, t-stats, n."""
    key = idx.set_index(["game_id", "team"])
    rows = []
    for _, g in bt.iterrows():
        for side, opp, pred, act in (("home", "away", "pred_home", "actual_home"),
                                     ("away", "home", "pred_away", "actual_away")):
            try:
                own = key.loc[(g["game_id"], g[side])]
                other = key.loc[(g["game_id"], g[opp])]
            except KeyError:
                continue
            rows.append(dict(resid=float(g[act] - g[pred]),
                             qb=float(own["qb_idx"]), dfn=float(other["def_idx"])))
    d = pd.DataFrame(rows)
    if len(d) < 30:
        return dict(n=int(len(d)), qb_coef=np.nan, def_coef=np.nan)
    X = np.column_stack([np.ones(len(d)), d["qb"], d["dfn"]])
    y = d["resid"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = float((resid ** 2).sum() / max(len(d) - 3, 1))
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return dict(n=int(len(d)), intercept=float(beta[0]),
                qb_coef=float(beta[1]), qb_t=float(beta[1] / se[1]),
                def_coef=float(beta[2]), def_t=float(beta[2] / se[2]),
                mean_qb_idx=float(d["qb"].mean()), mean_def_idx=float(d["dfn"].mean()),
                share_qb_gt_half=float((d["qb"] > 0.5).mean()),
                share_def_gt_0=float((d["dfn"] > 0).mean()),
                rows=d)
