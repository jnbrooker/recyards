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

    # Shares are renormalised over ACTIVE players so the team's volume adds up;
    # inactive players carry zero share.
    act = r["active"].values
    for col in ("target_share", "carry_share"):
        tot = r.loc[act, col].sum()
        r[col] = np.where(act, r[col] / tot if tot > 0 else 0.0, 0.0)
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

    tgt_prior = TARGET_ROLE_PRIOR.get((pos, depth), 0.01)
    car_prior = CARRY_ROLE_PRIOR.get((pos, depth), 0.005)
    snap_share, role_source = None, "rank"
    if snap is not None and snap[1] >= SNAP_MIN_GAMES:
        snap_share = float(snap[0])
        tgt_prior = 0.5 * tgt_prior + 0.5 * SNAP_TARGET_SLOPE.get(pos, 0.01) * snap_share
        car_prior = 0.5 * car_prior + 0.5 * SNAP_CARRY_SLOPE.get(pos, 0.005) * snap_share
        role_source = "rank + snaps"

    if games:
        w = h["w"].values
        own_tgt = float(D.wmean(h["target_share"].values, w)) \
            if h["target_share"].sum() > 0 else _share_from_counts(h, totals, "targets")
        own_car = _share_from_counts(h, totals, "carries")
        tgt = blend * _safe(own_tgt, tgt_prior) + (1 - blend) * tgt_prior
        car = blend * _safe(own_car, car_prior) + (1 - blend) * car_prior

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
    """A player's share of team volume across the games he actually played,
    recent games counting more."""
    j = h.merge(totals.drop(columns="w", errors="ignore"),
                on=["recent_team", "season", "week"], how="left")
    tot = float((j["team_tgt" if col == "targets" else "team_car"] * j["w"]).sum())
    return float((j[col] * j["w"]).sum() / tot) if tot and tot > 0 else 0.0


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
