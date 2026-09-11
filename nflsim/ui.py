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
    if games:
        bits.append(f"own history {own:.0%} over {games} games, "
                    f"blended {row.get('blend', 0):.0%} toward it")
    else:
        bits.append("no history in the priors window — slot prior only")
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
    """The share to simulate with: live if active, own history if ruled out."""
    if bool(row.get("active", True)) and float(row[share_col]) > 0:
        return float(row[share_col])
    return float(row.get(f"own_{share_col}", row[share_col]))


def cached_live(seasons: tuple[int, ...]) -> dict:
    return RO.load_live(tuple(sorted(seasons)))


def cached_rosters(seasons: tuple[int, ...], use_injuries: bool) -> pd.DataFrame:
    return RO.league_rosters(cached_live(seasons), use_injuries)


def season_picker(label: str = "Seasons used to build priors",
                  key: str = "seasons") -> tuple:
    """The seasons multiselect every page shares.

    Options run back from the current calendar year; the default is the most
    recent seasons nflverse has actually published, so the current season joins
    the defaults the week its first stats file lands and is weighted most
    heavily from then on. Stops the page if nothing is selected.
    """
    from . import data as D
    options, defaults = D.season_choices()
    seasons = st.sidebar.multiselect(
        label, options, default=defaults, key=key,
        help="Recent seasons are weighted more heavily (1.0 / 0.7 / 0.45 / 0.3). "
             "The current season is included automatically once it has data; "
             "early in the year it carries few games, so its influence grows "
             "week by week.")
    if not seasons:
        st.sidebar.error("Pick at least one season.")
        st.stop()
    return tuple(int(s) for s in seasons)
