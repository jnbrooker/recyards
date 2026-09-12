"""
nflsim/roster.py — live depth charts → who is on the field, and what each slot
is worth (roadmap §4.4). Shared by the game engine and every single-stat page.

A name isn't a workload. The depth chart (auto-pulled from nflverse for the
season being PLAYED, not the priors window) decides who is on the field and in
what slot; the player's own history decides what that slot is worth. Each usage
share is a blend, weighted `games / (games + ROLE_BLEND_N)`, of

  * the player's own target / carry share over the priors window, and
  * the prior for his positional rank (WR1 0.22, WR2 0.16, TE1 0.17, RB1 0.50
    of carries, …),

then renormalised across the roster so the team's volume adds up. A rookie WR1
inherits his slot's prior; a veteran WR3 who really commands targets keeps his
own number. Players ruled Out / Doubtful on the current season's latest injury
report are dropped and their share redistributed to the players who remain —
which is exactly what the single-stat pages need when a teammate goes down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import rushing as R

# ---------------------------------------------------------------------------
# Role priors: what a depth-chart slot is worth when a player has no history.
# Shares of team targets / team carries; they are renormalised per roster.
# ---------------------------------------------------------------------------

# UNCONDITIONAL slot shares — measured over every listed player not ruled
# out, 2025, with a game he did not appear in counting as zero (WR5s appear
# in 71% of weeks, RB4s in 23%). Unconditional is what the allocation needs:
# a roster's listed slots then sum to the position group's real share of the
# ball (WR 0.56, RB 0.81) instead of over-filling it and scaling every starter
# down. These are what a slot is worth with no history and, via the
# consistency weighting below, the anchor a player's history is checked against.
TARGET_ROLE_PRIOR = {
    ("WR", 1): 0.242, ("WR", 2): 0.169, ("WR", 3): 0.105, ("WR", 4): 0.058, ("WR", 5): 0.028,
    ("TE", 1): 0.161, ("TE", 2): 0.063, ("TE", 3): 0.024,
    ("RB", 1): 0.106, ("RB", 2): 0.054, ("RB", 3): 0.020, ("RB", 4): 0.002,
    ("FB", 1): 0.012,
}
CARRY_ROLE_PRIOR = {
    ("RB", 1): 0.540, ("RB", 2): 0.231, ("RB", 3): 0.061, ("RB", 4): 0.012,
    ("QB", 1): 0.135, ("QB", 2): 0.01,
    ("FB", 1): 0.008,
    ("WR", 1): 0.006, ("WR", 2): 0.006, ("WR", 3): 0.005, ("WR", 4): 0.006, ("WR", 5): 0.003,
    ("TE", 1): 0.003, ("TE", 2): 0.002, ("TE", 3): 0.003,
}

# How fast a player's own history takes over from the role prior (games).
ROLE_BLEND_N = 10.0

# Snap counts as a second role signal (roadmap 8.4). A player's expected usage
# share is roughly proportional to his offensive snap share — fitted on
# 2024-25 player-games: target share ~ 0.25 x snaps for a WR (r = 0.72), 0.20
# for a TE, 0.17 for a RB; carry share ~ 0.81 x snaps for a RB (r = 0.87).
# The role prior is the AVERAGE of the depth-chart-rank prior and the snap
# prior once the player has SNAP_MIN_GAMES of snaps; backtested on 2025 that
# cut target-share error 5% for players with < 5 games of history and 2.5%
# for 5-15, and was neutral for veterans (whose own history dominates).
SNAP_TARGET_SLOPE = {"WR": 0.25, "TE": 0.20, "RB": 0.17, "FB": 0.05, "QB": 0.0}
SNAP_CARRY_SLOPE = {"RB": 0.81, "QB": 0.15, "WR": 0.01, "FB": 0.10, "TE": 0.005}
SNAP_MIN_GAMES = 2

# A player's history is only as informative as it is CONSISTENT with the slot
# the depth chart now gives him. A back who was a lead back somewhere and is
# now listed RB4 (a returning veteran behind a rookie), or a backup promoted
# to RB1, has history that describes a different role, so the weight on it is
# scaled by (smaller / larger of own share vs slot prior) ** CONSISTENCY_POW.
# Then each position group is normalised to the team's own share of touches
# for that position (shrunk toward league). Before this, three veteran
# backups with big histories could squeeze a rookie RB1 to 31% of the
# carries; listed RB1s actually take 55%. Backtested over 5 489 listed
# player-weeks of 2025 (roadmap 8.4): RB1 carry error -14%, all carries -6%.
CONSISTENCY_POW = 1.0
GROUP_PRIOR_GAMES = 8.0     # team-games of prior on the team's group split
LEAGUE_CARRY_GROUP = {"RB": 0.81, "QB": 0.15, "WR": 0.02, "FB": 0.01, "TE": 0.01}
LEAGUE_TARGET_GROUP = {"WR": 0.57, "TE": 0.22, "RB": 0.19, "FB": 0.01, "QB": 0.01}

# Receiving fallbacks / regression.
LG_CATCH_RATE = 0.645
LG_YPT = 7.6
CATCH_PRIOR_N = 25.0        # targets-worth of prior on catch rate
YPT_PRIOR_N = 30.0          # targets-worth of prior on yards per target

# Touchdown role weighting. With a goal-line profile (touchdowns.py) the TD
# rates regress toward the player's own role x conversion over
# TD_ROLE_PRIOR_N touches; otherwise toward league over TD_RATE_PRIOR_N.
TD_RATE_PRIOR_N = 30.0

# Depth-chart slots the pages and engine consider per position.
DEPTH_CAPS = {"WR": 5, "TE": 3, "RB": 4, "QB": 1, "FB": 1}
POSITION_ORDER = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "FB": 4}

INJURY_TAG = {"Out": "OUT", "Doubtful": "DOUBTFUL", "Questionable": "Q"}


# ---------------------------------------------------------------------------
# Loading: everything the role layer needs, once
# ---------------------------------------------------------------------------

def current_season(seasons: tuple[int, ...],
                   depth_seasons: tuple[int, ...] | None = None) -> tuple[int, ...]:
    """The season whose depth charts describe today's rosters.

    Defaults to the season after the priors window (the one being played) and
    falls back to the last priors season if that file does not exist yet.
    """
    if depth_seasons is None:
        depth_seasons = (max(int(s) for s in seasons) + 1,)
        if D.load_depth_charts(depth_seasons).empty:
            depth_seasons = (max(int(s) for s in seasons),)
    return tuple(sorted(int(s) for s in depth_seasons))


def load_live(seasons: tuple[int, ...],
              depth_seasons: tuple[int, ...] | None = None,
              recency: D.Recency = D.RECENCY_DEFAULT) -> dict:
    """Weekly priors + current depth charts + injuries + PFR, in one dict.
    `recency` sets the game weights on every feed (see `data.game_weights`)."""
    seasons = tuple(sorted(int(s) for s in seasons))
    wk = D.load_weekly(seasons, recency)
    depth_seasons = current_season(seasons, depth_seasons)
    pfr = D.load_pfr_rush(seasons, recency)
    agg, pfr_lg = R.pfr_rush_aggregates(pfr)
    return dict(
        seasons=seasons, depth_seasons=depth_seasons, recency=D.Recency(*recency), wk=wk,
        depth=D.load_depth_charts(depth_seasons),
        injuries=D.load_injuries(depth_seasons),
        pfr_agg=agg, pfr_lg=pfr_lg,
        team_vol=team_volumes(wk),
        snaps=snap_roles(D.load_snap_counts(tuple(sorted(set(seasons) | set(depth_seasons)))),
                         recency),
        goal_line=_goal_line(seasons, recency, wk),
    )


def _goal_line(seasons, recency, wk) -> dict:
    """Goal-line role profiles from pbp touches (empty if pbp is unavailable)."""
    from . import touchdowns as TDm
    try:
        return TDm.goal_line_profiles(D.load_touches(seasons, recency), wk)
    except Exception:
        return {}


def snap_roles(snaps: pd.DataFrame, recency: D.Recency = D.RECENCY_DEFAULT) -> dict:
    """player_id -> (recency-weighted offensive snap share, games of snaps)."""
    if snaps is None or snaps.empty or "offense_pct" not in snaps.columns:
        return {}
    d = D.game_weights(snaps[snaps["offense_pct"] > 0], "team", recency)
    if d.empty:
        return {}
    num = (d["offense_pct"] * d["w"]).groupby(d["player_id"]).sum()
    den = d["w"].groupby(d["player_id"]).sum()
    n = d.groupby("player_id").size()
    return {p: (float(num[p] / den[p]), int(n[p])) for p in num.index}


def team_games(wk: pd.DataFrame, **sums) -> pd.DataFrame:
    """One row per team-game with the requested column sums and the game's
    recency weight `w`, e.g. `team_games(wk, tgt=("targets", "sum"))`."""
    if "w" not in wk.columns:
        wk = wk.assign(w=1.0)
    return (wk.groupby(["recent_team", "season", "week"], as_index=False)
              .agg(**dict(sums), w=("w", "first")))


def team_volumes(wk: pd.DataFrame) -> dict:
    """Per team: recency-weighted mean attempts, dropbacks and carries per game
    (plus league)."""
    db = "dropbacks" if "dropbacks" in wk.columns else "attempts"
    tg = team_games(wk, att=("attempts", "sum"), db=(db, "sum"),
                    car=("carries", "sum"), tgt=("targets", "sum"))

    def _row(g):
        return dict(attempts=D.wmean(g["att"], g["w"]), dropbacks=D.wmean(g["db"], g["w"]),
                    carries=D.wmean(g["car"], g["w"]), targets=D.wmean(g["tgt"], g["w"]))

    out = {t: _row(g) for t, g in tg.groupby("recent_team")}
    out["_LEAGUE_"] = _row(tg)
    return out


def injury_status(inj: pd.DataFrame, team: str, week: int | None = None) -> dict:
    """gsis_id -> report status on the team's latest (or given) injury report."""
    if inj is None or inj.empty or "report_status" not in inj.columns:
        return {}
    t = inj[inj["team"] == team]
    if "season" in t.columns and not t.empty:
        t = t[t["season"] == t["season"].max()]
    if "week" in t.columns and not t.empty:
        t = t[t["week"] == (int(week) if week is not None else t["week"].max())]
    t = t.dropna(subset=["report_status"])
    idcol = "gsis_id" if "gsis_id" in t.columns else "player_id"
    return {str(p): str(s) for p, s in zip(t[idcol], t["report_status"])}


# ---------------------------------------------------------------------------
# Roster construction: depth chart + history -> shares and priors
# ---------------------------------------------------------------------------

def _team_totals(wk: pd.DataFrame) -> pd.DataFrame:
    """Team targets and carries per game, for turning counts into shares."""
    return team_games(wk, team_tgt=("targets", "sum"), team_car=("carries", "sum"))


def _wsum(h: pd.DataFrame, col: str) -> float:
    """Recency-weighted total of `col` over a player's games."""
    return float((h[col] * h["w"]).sum())


def build_roster(wk: pd.DataFrame, snapshot: pd.DataFrame, team: str,
                 ruled_out: set | None = None,
                 pfr_agg: dict | None = None, pfr_lg: dict | None = None,
                 max_per_pos: dict | None = None,
                 status: dict | None = None,
                 snaps: dict | None = None,
                 goal_line: dict | None = None) -> pd.DataFrame:
    """Turn one depth-chart snapshot into a table of players with usage shares.

    Each player gets a target share and a carry share that blend his own history
    with the prior for his depth-chart slot, plus the efficiency priors needed
    to turn volume into yards. Players in `ruled_out` are excluded from the
    share renormalisation but kept in the table with `active = False`, so a
    picker can still show them (tagged) without them soaking up volume.
    """
    ruled_out = {str(p) for p in (ruled_out or set())}
    status = status or {}
    caps = dict(DEPTH_CAPS)
    caps.update(max_per_pos or {})

    snap = snapshot.copy()
    snap["depth"] = snap["depth"].astype(int)
    snap = snap[snap.apply(lambda r: r["depth"] <= caps.get(r["position"], 0), axis=1)]
    if snap.empty:
        raise ValueError(f"No usable depth chart for {team}.")

    totals = _team_totals(wk)
    lg = _league_rates(wk)
    lg["rush"] = R.league_rush_priors(wk)
    lg["goal_line"] = goal_line or {}
    rows = []
    for _, pl in snap.iterrows():
        pid = str(pl["player_id"])
        h = wk[wk["player_id"].astype(str) == pid]
        row = _player_row(pl, h, totals, lg, pfr_agg, pfr_lg, (snaps or {}).get(pid))
        row["team"] = team
        row["status"] = status.get(pid, "")
        row["active"] = pid not in ruled_out
        rows.append(row)

    r = pd.DataFrame(rows)
    # A player's own-history share, kept for display and for the case where a
    # page insists on simulating someone who is listed out.
    r["own_target_share"] = r["target_share"]
    r["own_carry_share"] = r["carry_share"]

    # Depth-chart order within each position group, group normalised to the
    # team's split of touches; inactive players carry zero share.
    r = allocate_shares(r, team_group_shares(wk, team))
    act = r["active"].values
    # Role weights for splitting team touchdowns follow the FINAL shares:
    # opportunity x conversion, normalised over active players.
    r["rec_td_weight"] = r["target_share"] * r["rec_td_rate"] / max(lg["rec_td"], 1e-6)
    r["rush_td_weight"] = r["carry_share"] * r["rush_td_rate"] / max(lg["rush_td"], 1e-6)
    for col, weight in (("rec_td_weight", "target_share"), ("rush_td_weight", "carry_share")):
        tot = r.loc[act, col].sum()
        r[col] = np.where(act, r[col] / tot if tot > 0 else r[weight], 0.0)

    r["pos_order"] = r["position"].map(POSITION_ORDER).fillna(9)
    return (r.sort_values(["pos_order", "depth"]).drop(columns="pos_order")
             .reset_index(drop=True))


def _league_rates(wk: pd.DataFrame) -> dict:
    tg, rec = _wsum(wk, "targets"), _wsum(wk, "receptions")
    ry, car = _wsum(wk, "receiving_yards"), _wsum(wk, "carries")
    return dict(
        catch=float(rec / tg) if tg > 0 else LG_CATCH_RATE,
        ypt=float(ry / tg) if tg > 0 else LG_YPT,
        rec_td=float(_wsum(wk, "receiving_tds") / rec) if rec > 0 else 0.075,
        rush_td=float(_wsum(wk, "rushing_tds") / car) if car > 0 else 0.025,
    )


def _player_row(pl: pd.Series, h: pd.DataFrame, totals: pd.DataFrame,
                lg: dict, pfr_agg, pfr_lg, snap: tuple | None = None) -> dict:
    pos, depth = pl["position"], int(pl["depth"])
    games = int(len(h))
    blend = games / (games + ROLE_BLEND_N)

    rank_tgt = TARGET_ROLE_PRIOR.get((pos, depth), 0.01)
    rank_car = CARRY_ROLE_PRIOR.get((pos, depth), 0.005)
    tgt_prior, car_prior = rank_tgt, rank_car
    snap_share, role_source = None, "rank"
    if snap is not None and snap[1] >= SNAP_MIN_GAMES:
        # the snap prior is history too: weight it by its consistency with the slot
        snap_share = float(snap[0])
        tgt_prior = _blend(0.5, SNAP_TARGET_SLOPE.get(pos, 0.01) * snap_share, rank_tgt, rank_tgt, depth == 1)
        car_prior = _blend(0.5, SNAP_CARRY_SLOPE.get(pos, 0.005) * snap_share, rank_car, rank_car, depth == 1)
        role_source = "rank + snaps"

    if games:
        w = h["w"].values
        own_tgt = _share_from_counts(h, totals, "targets")
        own_car = _share_from_counts(h, totals, "carries")
        # the snap prior is per game appeared; make it unconditional too
        avail = _stint_appearance_rate(h, totals)
        avail_rate = avail
        if snap_share is not None:
            tgt_prior = _blend(0.5, SNAP_TARGET_SLOPE.get(pos, 0.01) * snap_share * avail, rank_tgt, rank_tgt, depth == 1)
            car_prior = _blend(0.5, SNAP_CARRY_SLOPE.get(pos, 0.005) * snap_share * avail, rank_car, rank_car, depth == 1)
        tgt = _blend(blend, _safe(own_tgt, tgt_prior), tgt_prior, rank_tgt, depth == 1)
        car = _blend(blend, _safe(own_car, car_prior), car_prior, rank_car, depth == 1)
        b_tgt = _history_weight(blend, _safe(own_tgt, tgt_prior), rank_tgt, depth == 1)
        b_car = _history_weight(blend, _safe(own_car, car_prior), rank_car, depth == 1)

        # Efficiency rates on recency-weighted totals, regressed toward league.
        tg_tot, rec_tot = _wsum(h, "targets"), _wsum(h, "receptions")
        ry_tot, car_tot = _wsum(h, "receiving_yards"), _wsum(h, "carries")
        catch = (rec_tot + CATCH_PRIOR_N * lg["catch"]) / (tg_tot + CATCH_PRIOR_N)
        ypt = (ry_tot + YPT_PRIOR_N * lg["ypt"]) / (tg_tot + YPT_PRIOR_N)
        # TD rates: toward the player's goal-line role when pbp knows it
        from . import touchdowns as TDm
        role = TDm.role_td_rates(str(pl["player_id"]), pos, lg.get("goal_line"), float(np.clip(catch, 0.3, 0.95)))
        if role is not None:
            n_td, t_rec, t_rush = TDm.TD_ROLE_PRIOR_N, role["rec_td_per_rec"], role["rush_td_per_car"]
        else:
            n_td, t_rec, t_rush = TD_RATE_PRIOR_N, lg["rec_td"], lg["rush_td"]
        rec_td = (_wsum(h, "receiving_tds") + n_td * t_rec) / (rec_tot + n_td)
        rush_td = (_wsum(h, "rushing_tds") + n_td * t_rush) / (car_tot + n_td)
        name = str(h["player_display_name"].iloc[-1])
        prev_team = str(h["recent_team"].iloc[-1])
    else:
        tgt, car = tgt_prior, car_prior
        b_tgt = b_car = 0.0
        avail_rate = 1.0
        catch, ypt = lg["catch"], lg["ypt"]
        rec_td, rush_td = lg["rec_td"], lg["rush_td"]
        from . import touchdowns as TDm
        role = TDm.role_td_rates(str(pl["player_id"]), pos, lg.get("goal_line"), catch)
        if role is not None and role.get("from_profile"):
            rec_td, rush_td = role["rec_td_per_rec"], role["rush_td_per_car"]
        name = str(pl.get("player_name") or pl["player_id"])
        prev_team = ""

    if pos == "QB":
        tgt = 0.0            # a QB is not a target; trick plays are noise
    rush_priors = None
    if car > 0.02 and games:
        try:
            rush_priors = R.player_rush_priors(h, str(pl["player_id"]), pfr_agg, pfr_lg,
                                               lg=lg.get("rush"))
        except Exception:
            rush_priors = None

    return dict(
        player_id=str(pl["player_id"]), name=name, position=pos, depth=depth,
        games=games, from_history=bool(games), blend=float(blend),
        prev_team=prev_team, snap_share=snap_share, role_source=role_source,
        b_tgt=float(b_tgt), b_car=float(b_car),
        # games appeared / team games during his stints: the shares above are
        # UNCONDITIONAL (missed games count as zero); divide by this for
        # "if he plays" (the single-stat pages do, via ui.live_share)
        avail_rate=float(avail_rate),
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


def _blend(weight: float, own: float, prior: float, anchor: float,
           top_slot: bool = False) -> float:
    """`own` (a history-based estimate) vs `prior`, with the history's weight
    scaled by how consistent it is with the SLOT's rank prior `anchor` (see
    CONSISTENCY_POW). For the top slot, history ABOVE the anchor is fully
    consistent — a star who out-produces the WR1 prior is exactly what a WR1
    looks like; only history below it (a promoted backup) is discounted.
    CONSISTENCY_POW = 0 recovers the plain games blend."""
    if own <= 0 or anchor <= 0:
        return weight * own + (1 - weight) * prior
    if top_slot and own >= anchor:
        consistency = 1.0
    else:
        consistency = (min(own, anchor) / max(own, anchor)) ** CONSISTENCY_POW
    b = weight * consistency
    return b * own + (1 - b) * prior


def _history_weight(weight: float, own: float, anchor: float, top_slot: bool) -> float:
    """The weight `_blend` puts on history — how much evidence sits behind a
    share (0 = pure prior). Used to decide who gives way when a group
    over- or under-fills."""
    if own <= 0 or anchor <= 0:
        return 0.0
    if top_slot and own >= anchor:
        return float(weight)
    return float(weight * (min(own, anchor) / max(own, anchor)) ** CONSISTENCY_POW)


def team_group_shares(wk: pd.DataFrame, team: str) -> dict:
    """The team's own split of carries and targets by position group,
    recency-weighted and shrunk toward league over GROUP_PRIOR_GAMES."""
    t = wk[wk["recent_team"] == team]
    w = t["w"] if "w" in t.columns else pd.Series(1.0, index=t.index)
    out = {}
    for col, league in (("carries", LEAGUE_CARRY_GROUP), ("targets", LEAGUE_TARGET_GROUP)):
        tot = float((t[col] * w).sum())
        n_games = float(t.drop_duplicates(["season", "week"])["w"].sum()) if "w" in t.columns else float(t[["season", "week"]].drop_duplicates().shape[0])
        k = n_games / (n_games + GROUP_PRIOR_GAMES)
        grp = {}
        for pos in league:
            own = float((t.loc[t["position"] == pos, col] * w[t["position"] == pos]).sum()) / tot if tot > 0 else league[pos]
            grp[pos] = k * own + (1 - k) * league[pos]
        z = sum(grp.values())
        out[col] = {p: v / z for p, v in grp.items()}
    return out


def _fit_group(vals: np.ndarray, evidence: np.ndarray, want: float) -> np.ndarray:
    """Scale a group's shares to sum to `want`, taking the adjustment from the
    least-evidenced shares first: each share moves in proportion to
    (1 - evidence), so a star's well-measured 34% is not cut to make room for
    a fifth receiver's guessed slot. Falls back to proportional scaling for
    whatever the low-evidence shares cannot absorb."""
    tot = float(vals.sum())
    if tot <= 0:
        return vals
    gap = want - tot
    give = vals * (1.0 - np.clip(evidence, 0.0, 1.0))
    if gap < 0:
        # shrink: the low-evidence mass can absorb at most its own size
        absorb = min(-gap, float(give.sum()) * 0.95)
        out = vals - give * (absorb / give.sum() if give.sum() > 0 else 0.0)
        rest = -gap - absorb
        return out * (1.0 - rest / max(out.sum(), 1e-9)) if rest > 0 else out
    out = vals + give * (gap / give.sum()) if give.sum() > 0 else vals
    return out * (want / out.sum())


def allocate_shares(r: pd.DataFrame, group_shares: dict | None = None) -> pd.DataFrame:
    """Turn per-player blended shares into a roster allocation: each position
    group fitted to its share of the team's touches, low-evidence shares giving
    way first (`_fit_group`). Operates on `target_share` / `carry_share` of the
    ACTIVE players in `r`; inactive players get zero."""
    r = r.copy()
    act = r["active"].values.astype(bool)
    for col, bcol, key, league in (("carry_share", "b_car", "carries", LEAGUE_CARRY_GROUP),
                                   ("target_share", "b_tgt", "targets", LEAGUE_TARGET_GROUP)):
        gs = (group_shares or {}).get(key, league)
        new = np.zeros(len(r))
        present = {}
        for pos, g in r[act].groupby("position"):
            ev = g[bcol].values.astype(float) if bcol in g.columns else np.zeros(len(g))
            present[pos] = (g.index, g[col].values.astype(float), ev)
        # groups the roster lacks (no FB listed) give their share back pro rata
        total_share = sum(gs.get(p, 0.0) for p in present) or 1.0
        for pos, (idx, vals, ev) in present.items():
            want = gs.get(pos, 0.0) / total_share
            new[r.index.get_indexer(idx)] = _fit_group(vals, ev, want)
        r[col] = new
    return r


def _share_from_counts(h: pd.DataFrame, totals: pd.DataFrame, col: str) -> float:
    """A player's UNCONDITIONAL share of team volume: his (weighted) touches
    over the team's (weighted) touches in every team game during his stint —
    first to last appearance with each team — so a game he missed counts as
    zero, the way the slot priors are measured."""
    tcol = "team_tgt" if col == "targets" else "team_car"
    num = float((h[col] * h["w"]).sum())
    den = 0.0
    for team, g in h.groupby("recent_team"):
        stamp = g["season"] * 100 + g["week"]
        t = totals[totals["recent_team"] == team]
        ts = t["season"] * 100 + t["week"]
        span = t[(ts >= stamp.min()) & (ts <= stamp.max())]
        den += float((span[tcol] * span["w"]).sum())
    return num / den if den > 0 else 0.0


def _stint_appearance_rate(h: pd.DataFrame, totals: pd.DataFrame) -> float:
    """Games appeared / team games during the player's stint(s)."""
    n_team = 0
    for team, g in h.groupby("recent_team"):
        stamp = g["season"] * 100 + g["week"]
        t = totals[totals["recent_team"] == team]
        ts = t["season"] * 100 + t["week"]
        n_team += int(((ts >= stamp.min()) & (ts <= stamp.max())).sum())
    return float(len(h) / n_team) if n_team > 0 else 1.0


# ---------------------------------------------------------------------------
# Team- and league-level convenience
# ---------------------------------------------------------------------------

def roster_for(live: dict, team: str, use_injuries: bool = True,
               week: int | None = None) -> pd.DataFrame:
    """One team's roster from its most recent depth chart.

    The injury filter uses the latest report of the CURRENT season (or `week`).
    It is only meaningful in season — out of season the last report belongs to
    a game already played.
    """
    snap = D.depth_chart_snapshot(live["depth"], team)
    status = injury_status(live["injuries"], team, week=week)
    out = ({p for p, s in status.items() if s in ("Out", "Doubtful")}
           if use_injuries else set())
    return build_roster(live["wk"], snap, team, out, live["pfr_agg"], live["pfr_lg"],
                        status=status, snaps=live.get("snaps"),
                        goal_line=live.get("goal_line"))


def league_rosters(live: dict, use_injuries: bool = True) -> pd.DataFrame:
    """Every team's roster in one table, with a picker-ready `label`."""
    depth = live["depth"]
    if depth is None or depth.empty:
        return pd.DataFrame()
    frames = []
    for team in sorted(depth["team"].dropna().unique()):
        try:
            frames.append(roster_for(live, team, use_injuries))
        except ValueError:
            continue
    if not frames:
        return pd.DataFrame()
    r = pd.concat(frames, ignore_index=True)
    r["label"] = [player_label(row) for _, row in r.iterrows()]
    r["pos_order"] = r["position"].map(POSITION_ORDER).fillna(9)
    return (r.sort_values(["team", "pos_order", "depth"]).drop(columns="pos_order")
             .reset_index(drop=True))


def player_label(row) -> str:
    """'Zay Flowers — BAL WR1' with an injury tag and a moved-teams marker."""
    tag = INJURY_TAG.get(str(row.get("status", "")), "")
    moved = (row.get("prev_team") and row["prev_team"] != row["team"])
    s = f"{row['name']} — {row['team']} {row['position']}{int(row['depth'])}"
    if tag:
        s += f" · {tag}"
    if moved:
        s += f" (was {row['prev_team']})"
    if not row.get("from_history", True):
        s += " · no history"
    return s


# ---------------------------------------------------------------------------
# Role-only priors: what to simulate when a player has NO history in the
# priors window (a rookie starter, a returning veteran). Each returns the exact
# dict shape the corresponding model's `simulate` expects, built from the
# player's depth-chart slot and league-average efficiency for his position.
# ---------------------------------------------------------------------------

def _pos_league(wk: pd.DataFrame, position: str) -> dict:
    p = wk[wk["position"] == position]
    if p.empty or p["targets"].sum() == 0:
        p = wk
    tg, rec = _wsum(p, "targets"), _wsum(p, "receptions")
    yac = _wsum(p, "receiving_yards_after_catch")
    return dict(
        adot=float(_wsum(p, "receiving_air_yards") / tg) if tg > 0 else 8.0,
        air_per_rec=float((_wsum(p, "receiving_yards") - yac) / rec) if rec > 0 else 5.7,
        yac_per_rec=float(yac / rec) if rec > 0 else 5.0,
        catch=float(rec / tg) if tg > 0 else LG_CATCH_RATE,
    )


def receiving_priors_from_role(wk: pd.DataFrame, row: pd.Series) -> dict:
    """Priors in the shape of `model.player_priors` for a history-less player."""
    lg = _pos_league(wk, row["position"])
    return dict(
        player_id=str(row["player_id"]), name=str(row["name"]),
        position=str(row["position"]), team=str(row["team"]), games=0,
        mu_ts=float(np.clip(row["target_share"], 0.01, 0.6)), sd_ts=0.05,
        mu_catch=float(np.clip(row.get("catch_rate", lg["catch"]), 0.3, 0.95)),
        sd_catch=0.10,
        mu_adot=lg["adot"], sd_adot=3.0,
        mu_air=lg["air_per_rec"], sd_air=2.5,
        yac_per_rec=lg["yac_per_rec"],
    )


def rushing_priors_from_role(row: pd.Series, pfr_lg: dict | None = None) -> dict:
    """Priors in the shape of `rushing.player_rush_priors` for a history-less back."""
    lg = pfr_lg or {}
    ybc = float(lg.get("mu_ybc", R.FALLBACK_YPC * R.YBC_FRACTION))
    yac = float(lg.get("mu_yac", R.FALLBACK_YPC * (1 - R.YBC_FRACTION)))
    return dict(
        player_id=str(row["player_id"]), name=str(row["name"]),
        position=str(row["position"]), team=str(row["team"]), games=0,
        mu_share=float(np.clip(row["carry_share"], 0.01, 0.95)),
        sd_share=R.FALLBACK_SHARE_SD,
        mu_ypc=ybc + yac,
        mu_ybc=ybc, sd_ybc=R.LG_YBC_SD, mu_yac=yac, sd_yac=R.LG_YAC_SD,
        brk_rate=float(lg.get("brk", R.LG_BRK_RATE)),
        adv_source="league average (no history)",
    )


def td_priors_from_role(row: pd.Series, team_vol: dict, lg: dict) -> dict:
    """Priors in the shape of `touchdowns.player_td_priors` for a history-less player."""
    tv = team_vol.get(row["team"], team_vol["_LEAGUE_"])
    mu_rec = float(row["target_share"] * tv["targets"] * row.get("catch_rate", LG_CATCH_RATE))
    mu_car = float(row["carry_share"] * tv["carries"])
    pos = str(row["position"])
    # `lg` is touchdowns.league_td_rates: {pos: {rec_td_per_rec, rush_td_per_car}, "_ALL_": ...}
    rates = lg.get(pos, lg.get("_ALL_", {}))
    p_rec = float(rates.get("rec_td_per_rec", 0.07))
    p_rush = float(rates.get("rush_td_per_car", 0.03))
    return dict(
        player_id=str(row["player_id"]), name=str(row["name"]), position=pos,
        team=str(row["team"]), games=0,
        mu_rec=mu_rec, var_rec=max(mu_rec * 1.5, 0.5),
        mu_car=mu_car, var_car=max(mu_car * 2.0, 0.5),
        p_rec_td=p_rec, p_rush_td=p_rush,
        raw_rec_td_per_rec=0.0, raw_rush_td_per_car=0.0,
        prior_rec_td=p_rec, prior_rush_td=p_rush,
        gl_car_frac=np.nan, gl_tgt_frac=np.nan, role_source="positional mean (no history)",
    )


def scale_volume_priors(pri: dict, mu_key: str, var_key: str, new_mu: float) -> dict:
    """Move a count prior's mean to `new_mu`, scaling its variance with it so
    the dispersion (var/mean) the player's history showed is preserved."""
    old = float(pri.get(mu_key, 0.0))
    out = dict(pri)
    if old > 0 and new_mu > 0:
        out[var_key] = float(pri[var_key]) * (new_mu / old)
    out[mu_key] = float(max(new_mu, 0.0))
    if var_key in out:
        out[var_key] = float(max(out[var_key], out[mu_key]))
    return out
