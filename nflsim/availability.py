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

The QB term has two parts. The familiarity DUMMY (`qb_idx`) is the main
effect; the quality SWING (`qb_swing = qb_idx × (starter ANY/A − incumbents'
ANY/A)`) refines it so a proven starter who changed teams is not charged the
full backup penalty. With both in the fit the swing is t = +3.4 and the
dummy stays t = −6.1; swing alone is worse than the dummy alone, so
unfamiliarity costs something beyond measured quality. Both together take
the out-of-sample RMSE a further 0.09 in each direction.
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
QB_MARGIN_COEF = -7.74      # t = -6.1  (familiarity dummy)
QB_SWING_COEF = 3.29        # t = +3.4  (x ANY/A gap, starter minus incumbents)
DEF_MARGIN_COEF = 12.36     # t = +2.3
# Offensive line: fitted at -4.6 margin per unit of own-line index difference
# (t = -1.1, n = 544; +0.02 RMSE out of sample) — right sign, not distinguishable
# from zero on two seasons, so it is DISPLAY-ONLY until a third season says
# otherwise. The index is still computed and shown.
OL_MARGIN_COEF = 0.0

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


def _unit_index(starters: pd.DataFrame, out_ids: set, share: dict, prefix: str) -> dict:
    """Share of a unit's starters' (snap-weighted) importance that is ruled out."""
    if starters is None or starters.empty:
        return {f"{prefix}_idx": 0.0, f"{prefix}_missing": [], f"n_{prefix}": 0}
    imp = np.array([float(share.get(str(p), DEFAULT_SNAP_SHARE)) for p in starters["player_id"]])
    out = np.array([str(p) in out_ids for p in starters["player_id"]])
    total = float(imp.sum())
    missing = [f"{n} ({pos})" for n, pos, o in
               zip(starters["player_name"], starters["position"], out) if o]
    return {f"{prefix}_idx": float(imp[out].sum() / total) if total > 0 else 0.0,
            f"{prefix}_missing": missing, f"n_{prefix}": int(len(starters))}


def def_index(starters: pd.DataFrame, out_ids: set, share: dict) -> dict:
    """Share of the defensive starters' importance that is ruled out."""
    d = _unit_index(starters, out_ids, share, "def")
    d["n_starters"] = d.pop("n_def")
    return d


def ol_starters(depth_ol: pd.DataFrame, team: str, as_of=None) -> pd.DataFrame:
    """The five offensive-line starters on the team's chart as of a date."""
    snap = D.depth_chart_snapshot(depth_ol, team, as_of)
    if snap.empty:
        return snap
    return snap[snap["depth"] == 1].reset_index(drop=True)


def ol_index(starters: pd.DataFrame, out_ids: set, share: dict) -> dict:
    """Share of the line starters' (offensive-snap-weighted) importance ruled out."""
    return _unit_index(starters, out_ids, share, "ol")


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


# QB quality: adjusted net yards per attempt, recency-weighted over the
# window and regressed toward a replacement level with QB_QUALITY_PRIOR_N
# dropbacks-worth of prior, so a QB with no history is "a replacement", not
# "league average".
QB_QUALITY_PRIOR_N = 150.0
REPLACEMENT_BELOW_LEAGUE = 0.8    # ANY/A below the league mean


def qb_quality(wk_prior: pd.DataFrame) -> dict:
    """player_id -> regressed ANY/A for every QB in the window, plus
    "_LEAGUE_" (the league mean) and "_REPLACEMENT_"."""
    q = wk_prior[wk_prior["position"] == "QB"]
    need = {"attempts", "passing_yards", "passing_tds", "interceptions", "sacks", "sack_yards"}
    if q.empty or not need <= set(q.columns):
        return {"_LEAGUE_": 6.0, "_REPLACEMENT_": 6.0 - REPLACEMENT_BELOW_LEAGUE}
    w = q["w"] if "w" in q.columns else pd.Series(1.0, index=q.index)
    num = (q["passing_yards"] + 20 * q["passing_tds"] - 45 * q["interceptions"] - q["sack_yards"]) * w
    den = (q["attempts"] + q["sacks"]) * w
    lg = float(num.sum() / den.sum()) if den.sum() > 0 else 6.0
    rep = lg - REPLACEMENT_BELOW_LEAGUE
    g = pd.DataFrame({"num": num, "den": den, "pid": q["player_id"].astype(str)}).groupby("pid").sum()
    out = ((g["num"] + rep * QB_QUALITY_PRIOR_N) / (g["den"] + QB_QUALITY_PRIOR_N)).to_dict()
    out["_LEAGUE_"] = lg
    out["_REPLACEMENT_"] = rep
    return out


def qb_index(wk_prior: pd.DataFrame, team: str, qb_id, quality: dict | None = None) -> dict:
    """QB familiarity and quality swing for a team's starter.

    qb_idx   = 1 − this QB's share of the team's weighted dropbacks in the
               window (how much of the offense rating someone else built).
    qb_swing = qb_idx × (starter's ANY/A − the other QBs' dropback-weighted
               ANY/A): the familiarity gap signed and scaled by how much
               better or worse the new man is than the men behind the rating.
    """
    t = wk_prior[(wk_prior["recent_team"] == team) & (wk_prior["position"] == "QB")]
    col = "dropbacks" if "dropbacks" in t.columns else "attempts"
    w = t["w"] if "w" in t.columns else pd.Series(1.0, index=t.index)
    total = float((t[col] * w).sum())
    quality = quality or {}
    rep = float(quality.get("_REPLACEMENT_", 5.2))
    if qb_id is None or total <= 0:
        # no chart QB (index unknowable -> 0) or a team with no history (-> 1)
        idx = 1.0 if qb_id is not None else 0.0
        return dict(qb_idx=idx, qb_share=0.0, qb_swing=0.0,
                    qb_quality=float(quality.get(str(qb_id), rep)) if qb_id else np.nan,
                    incumbent_quality=np.nan)
    pid = t["player_id"].astype(str)
    mine = pid == str(qb_id)
    share = float((t.loc[mine, col] * w[mine]).sum()) / total
    idx = float(np.clip(1.0 - share, 0.0, 1.0))
    # the other QBs behind the rating, weighted by the dropbacks they took
    others = t[~mine]
    ow = (others[col] * w[~mine])
    if float(ow.sum()) > 0:
        inc = float(sum(quality.get(p, rep) * x for p, x in zip(pid[~mine], ow)) / ow.sum())
    else:
        inc = float(quality.get(str(qb_id), rep))
    mine_q = float(quality.get(str(qb_id), rep))
    return dict(qb_idx=idx, qb_share=float(share), qb_swing=float(idx * (mine_q - inc)),
                qb_quality=mine_q, incumbent_quality=inc)


def team_indices(team: str, wk_prior: pd.DataFrame, depth_off: pd.DataFrame,
                 depth_def: pd.DataFrame, out_ids: set, share: dict, as_of=None,
                 quality: dict | None = None, depth_ol: pd.DataFrame | None = None,
                 off_share: dict | None = None) -> dict:
    """All indices for one team, plus the names behind them."""
    qb_id, qb_name = starting_qb(depth_off, team, out_ids, as_of)
    out = dict(team=team, qb_id=qb_id, qb_name=qb_name)
    out.update(qb_index(wk_prior, team, qb_id, quality))
    out.update(def_index(def_starters(depth_def, team, as_of), out_ids, share))
    if depth_ol is not None and not depth_ol.empty:
        out.update(ol_index(ol_starters(depth_ol, team, as_of), out_ids, off_share or {}))
    else:
        out.update(ol_idx=0.0, ol_missing=[], n_ol=0)
    return out


def margin_shift(idx_a: dict | None, idx_b: dict | None) -> float:
    """Expected-margin shift for team A against team B (points, A minus B).
    Callers split it evenly: A gains half, B loses half, so the total is
    untouched and only the margin moves — the effect the data supports."""
    a, b = idx_a or {}, idx_b or {}
    dq = float(a.get("qb_idx", 0.0)) - float(b.get("qb_idx", 0.0))
    ds = float(a.get("qb_swing", 0.0)) - float(b.get("qb_swing", 0.0))
    dd = float(b.get("def_idx", 0.0)) - float(a.get("def_idx", 0.0))
    dl = float(a.get("ol_idx", 0.0)) - float(b.get("ol_idx", 0.0))
    return float(QB_MARGIN_COEF * dq + QB_SWING_COEF * ds + DEF_MARGIN_COEF * dd
                 + OL_MARGIN_COEF * dl)


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
    depth_ol = D.load_depth_charts(depth_seasons, side="oline")
    snaps = D.load_snap_counts(tuple(sorted(set(seasons) | set(depth_seasons))))
    share = snap_shares(snaps, recency)
    off_share = snap_shares(snaps, recency, col="offense_pct")
    inj = live["injuries"]
    quality = qb_quality(live["wk"])
    out = {}
    teams = sorted(set(live["depth"]["team"].dropna())) if not live["depth"].empty else []
    for team in teams:
        ruled = D.players_ruled_out(inj, team, week=week)
        out[team] = team_indices(team, live["wk"], live["depth"], depth_def, ruled, share,
                                 quality=quality, depth_ol=depth_ol, off_share=off_share)
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
    depth_ol = D.load_depth_charts(tuple(int(s) for s in score_seasons), side="oline")
    inj = D.load_injuries(tuple(int(s) for s in score_seasons))
    snaps = D.load_snap_counts(seasons)
    sched = D.load_schedule(tuple(int(s) for s in score_seasons))
    rows = []
    steps = list(B._score_weeks(sched, score_seasons, weeks))
    for k, (S, w) in enumerate(steps):
        if progress:
            progress(k / max(len(steps), 1), f"availability {S} week {w}")
        wk_prior = D.game_weights(B.before(wk_all, S, w), "recent_team", recency)
        snaps_prior = B.before(snaps, S, w)
        share = snap_shares(snaps_prior, recency)
        off_share = snap_shares(snaps_prior, recency, col="offense_pct")
        quality = qb_quality(wk_prior)
        games = sched[(sched["season"] == S) & (sched["week"] == w)]
        for _, g in games.iterrows():
            as_of = pd.to_datetime(g["kickoff"], utc=True) if pd.notna(g["kickoff"]) else None
            for team in (g["home_team"], g["away_team"]):
                ruled = D.players_ruled_out(inj, team, season=S, week=w)
                r = team_indices(team, wk_prior, depth_off, depth_def, ruled, share, as_of,
                                 quality=quality, depth_ol=depth_ol, off_share=off_share)
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
                         ds=float(h.get("qb_swing", 0.0) - a.get("qb_swing", 0.0)),
                         dd=float(a["def_idx"] - h["def_idx"])))
    d = pd.DataFrame(rows)
    if len(d) < 30:
        return dict(n=int(len(d)))
    X = np.column_stack([np.ones(len(d)), d["dq"], d["ds"], d["dd"]])
    beta, *_ = np.linalg.lstsq(X, d["resid"].values, rcond=None)
    resid = d["resid"].values - X @ beta
    se = np.sqrt(np.diag(float((resid ** 2).sum() / max(len(d) - 4, 1)) * np.linalg.inv(X.T @ X)))
    return dict(n=int(len(d)), intercept=float(beta[0]),
                qb_coef=float(beta[1]), qb_t=float(beta[1] / se[1]),
                swing_coef=float(beta[2]), swing_t=float(beta[2] / se[2]),
                def_coef=float(beta[3]), def_t=float(beta[3] / se[3]), rows=d)


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
