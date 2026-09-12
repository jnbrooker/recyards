"""
nflsim/ui.py — the sidebar widgets the single-stat pages share, so "pick a
player from the live depth chart" behaves identically on every page.

The pages differ in what they simulate, not in how a player is chosen: team,
then a player from that team's current depth chart, labelled with slot, injury
status and whether he changed teams. The row handed back carries the live
usage share (already redistributed for teammates ruled out) and the player's
CURRENT team, which is what the team-volume lookups should use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from . import roster as RO


def pick_player(rosters: pd.DataFrame, positions: list[str], key: str = "pick",
                default_team: str = "BAL") -> pd.Series:
    """Team + player selectboxes over the live rosters; returns the roster row."""
    teams = sorted(rosters["team"].dropna().unique().tolist())
    team = st.sidebar.selectbox(
        "Team", teams, index=teams.index(default_team) if default_team in teams else 0,
        key=f"{key}_team")
    pool = rosters[(rosters["team"] == team) & rosters["position"].isin(positions)].copy()
    # list the page's primary position first (WR on the receiving page, RB on rushing)
    pool["_order"] = pool["position"].map({p: i for i, p in enumerate(positions)})
    pool = pool.sort_values(["_order", "depth"]).drop(columns="_order")
    if pool.empty:
        st.sidebar.error(f"No {'/'.join(positions)} on the {team} depth chart.")
        st.stop()
    label = st.sidebar.selectbox(
        "Player", pool["label"].tolist(), key=f"{key}_player",
        help="Current depth chart, best slot first. Tags: OUT / DOUBTFUL / Q from "
             "the latest injury report; (was XXX) = changed teams since the priors "
             "window; 'no history' = role prior only.")
    return pool[pool["label"] == label].iloc[0]


def role_caption(row: pd.Series, share_col: str, what: str) -> str:
    """One line explaining where this player's live share came from."""
    own = float(row.get(f"own_{share_col}", row[share_col]))
    live = float(row[share_col])
    games = int(row.get("games", 0))
    bits = [f"**{row['team']} {row['position']}{int(row['depth'])}**",
            f"live share of team {what}: **{live:.0%}**"]
    prior_txt = ("slot prior" if row.get("role_source", "rank") == "rank"
                 else f"slot + snap prior ({row.get('snap_share', 0):.0%} of snaps)")
    if games:
        bits.append(f"own history {own:.0%} over {games} games, "
                    f"blended {row.get('blend', 0):.0%} toward it, rest {prior_txt}")
    else:
        bits.append(f"no history in the priors window — {prior_txt} only")
    if row.get("prev_team") and row["prev_team"] != row["team"]:
        bits.append(f"history is from **{row['prev_team']}**; volume now uses "
                    f"**{row['team']}**")
    return " · ".join(bits)


def status_warning(row: pd.Series) -> None:
    """Flag a player who is listed out, or whose share was inflated by teammates
    being out, so the reader knows why the number moved."""
    status = str(row.get("status", "") or "")
    if not bool(row.get("active", True)):
        st.warning(
            f"**{row['name']} is listed {status.upper()}** on the latest injury "
            "report. His live share is zero; the simulation below uses his usual "
            "role instead, so treat it as 'if he plays'.")
    elif status == "Questionable":
        st.info(f"{row['name']} is **Questionable** on the latest injury report.")


def live_share(row: pd.Series, share_col: str) -> float:
    """The share to simulate with on a single-stat page — "if he plays":
    the live (redistributed) share if active, own history if ruled out, either
    way conditional on playing (the roster's shares are unconditional, with a
    player's historically missed games counted as zero; that haircut belongs
    in a season projection, not a game he is simulated to be in)."""
    avail = float(row.get("avail_rate", 1.0) or 1.0)
    avail = min(max(avail, 0.5), 1.0)
    if bool(row.get("active", True)) and float(row[share_col]) > 0:
        return float(row[share_col]) / avail
    return float(row.get(f"own_{share_col}", row[share_col])) / avail


def priors_picker(label: str = "Seasons used to build priors",
                  key: str = "seasons") -> tuple:
    """The priors window every page shares: `(seasons, recency)`.

    The visible control is "how much to trust this season" — a preset that
    sets both the season curve and the within-season half-life (see
    `data.Recency`), applied to every rate, share, volume and team rating. The
    seasons multiselect sits under an *Advanced* expander: its default is the
    most recent seasons nflverse has actually published, so the current season
    joins the week its first stats file lands. Stops the page if no season is
    selected.
    """
    from . import data as D
    names = list(D.RECENCY_PRESETS)
    choice = st.sidebar.select_slider(
        "How much to trust this season", names, value="Balanced", key=f"{key}_recency",
        help="**Long memory**: seasons count as flat blocks (×0.7 a year) — the "
             "original curve; this season is under half the model until week 17.  \n"
             "**Balanced**: ×0.85 a year, and a game 12 back counts half — this "
             "season is half the model by week 9.  \n**Recent form**: ×0.7 a year, a "
             "game 6 back counts half — the last six weeks dominate.  \n"
             "Applies to every rate, share, volume and team rating on every page.")
    recency = D.RECENCY_PRESETS[choice]

    options, defaults = D.season_choices()
    with st.sidebar.expander("Advanced: seasons in the window"):
        seasons = st.multiselect(
            label, options, default=defaults, key=key,
            help="Which seasons feed the priors. The current season is included "
                 "automatically once it has data; how much each season counts is "
                 "set by the recency control above.")
    if not seasons:
        st.sidebar.error("Pick at least one season.")
        st.stop()
    return tuple(int(s) for s in seasons), recency


def season_picker(label: str = "Seasons used to build priors",
                  key: str = "seasons") -> tuple:
    """Seasons only (default recency) — kept for callers that don't thread the
    recency through; new pages should use `priors_picker`."""
    return priors_picker(label, key)[0]


def recency_caption(wk: pd.DataFrame | None, recency, shares: dict | None = None) -> None:
    """One sidebar line saying how the weight is spread across seasons —
    computed from `wk`, or passed in as `shares` (`data.weight_shares`)."""
    from . import data as D
    shares = shares if shares is not None else D.weight_shares(wk)
    if not shares:
        return
    latest = max(shares)
    parts = ", ".join(f"{s}: {v:.0%}" for s, v in sorted(shares.items(), reverse=True))
    r = D.Recency(*recency)
    tail = ("every game in a season counts equally" if not np.isfinite(r.half_life)
            else f"each game back counts less (half after {r.half_life:g})")
    st.sidebar.caption(f"Weight by season — {parts}. {latest} is "
                       f"**{shares[latest]:.0%}** of the model; {tail}.")


def cached_live(seasons: tuple[int, ...], recency=None) -> dict:
    from . import data as D
    return RO.load_live(tuple(sorted(seasons)), recency=recency or D.RECENCY_DEFAULT)


def cached_rosters(seasons: tuple[int, ...], use_injuries: bool,
                   recency=None) -> pd.DataFrame:
    return RO.league_rosters(cached_live(seasons, recency), use_injuries)
