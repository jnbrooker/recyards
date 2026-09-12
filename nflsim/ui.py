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


@st.cache_data(ttl=6 * 3600, show_spinner="Loading play-by-play, depth charts and injuries…")
def cached_context(seasons: tuple[int, ...], recency) -> dict:
    """The full game-engine context (`game.prepare`), shared by every page
    that offers the game view — one download and one set of ratings per
    (seasons, recency) for the whole app."""
    from . import game as G
    return G.prepare(tuple(sorted(seasons)), recency=recency)


@st.cache_data(ttl=6 * 3600, show_spinner="Building rosters…")
def cached_team_roster(seasons: tuple[int, ...], recency, team: str,
                       use_injuries: bool) -> pd.DataFrame:
    from . import game as G
    return RO.roster_for(cached_context(seasons, recency), team, use_injuries)


def view_picker(key: str = "view") -> str:
    """Game view (a real fixture: opponent, home/away, availability and game
    script from the engine) or Season view (the player's typical game vs a
    chosen defense)."""
    return st.sidebar.radio(
        "View", ["Game", "Season"], horizontal=True, key=f"{key}_view",
        help="**Game**: pick a fixture; the opponent, home field, who is playing "
             "(QB familiarity, defensive starters out) and the game script — a "
             "team expected to trail runs less — all come from the game engine.  \n"
             "**Season**: the player's typical game, against whichever defense you "
             "pick. Closer to a season-long average.")


def game_picker(ctx: dict, seasons, recency, positions: list[str], key: str = "game") -> dict:
    """Week → game → side → player, from the live rosters. Returns the roster
    row plus the fixture: team, opponent, is_home, game_row, use_injuries."""
    from . import data as D
    sched = D.load_schedule((int(ctx["depth_seasons"][-1]),))
    if sched.empty:
        st.sidebar.error("The schedule did not load."); st.stop()
    weeks = sorted(sched["week"].unique().tolist())
    cur = D.current_week(sched)
    week = st.sidebar.selectbox("Week", weeks, index=weeks.index(cur) if cur in weeks else 0,
                                key=f"{key}_week")
    wg = sched[sched["week"] == week].reset_index(drop=True)
    labels = [D.game_label(r) for _, r in wg.iterrows()]
    unplayed = [i for i, r in wg.iterrows() if not bool(r["played"])]
    pick = st.sidebar.selectbox("Game", labels, index=unplayed[0] if unplayed else 0,
                                key=f"{key}_game")
    game_row = wg.iloc[labels.index(pick)]
    home, away = game_row["home_team"], game_row["away_team"]
    use_inj = st.sidebar.toggle("Drop players ruled out", value=True, key=f"{key}_inj")
    frames = []
    for team in (home, away):
        try:
            frames.append(cached_team_roster(tuple(sorted(seasons)), recency, team, use_inj))
        except ValueError:
            continue
    if not frames:
        st.sidebar.error("No usable depth charts for this game."); st.stop()
    pool = pd.concat(frames, ignore_index=True)
    pool = pool[pool["position"].isin(positions)].copy()
    pool["_order"] = pool["position"].map({p: i for i, p in enumerate(positions)})
    pool["_side"] = (pool["team"] == away).astype(int)      # home first
    pool = pool.sort_values(["_side", "_order", "depth"])
    pool["label"] = [RO.player_label(r) for _, r in pool.iterrows()]
    if pool.empty:
        st.sidebar.error(f"No {'/'.join(positions)} on either depth chart."); st.stop()
    label = st.sidebar.selectbox(
        "Player", pool["label"].tolist(), key=f"{key}_player",
        help="Both teams' depth charts, home team first. Tags: OUT / DOUBTFUL / Q "
             "from the latest injury report; 'no history' = role prior only.")
    row = pool[pool["label"] == label].iloc[0]
    team = row["team"]
    return dict(row=row, team=team, opp=away if team == home else home,
                is_home=(team == home), game_row=game_row, week=int(week),
                use_injuries=use_inj)


def script_caption(f: dict, game_row, what: str, typical: float, this_game: float,
                   player_typical: float, player_game: float) -> str:
    """One line on what the game view changed for this player."""
    side = "home" if f.get("is_home", True) else "away"
    fav = f["team"] if f["exp_margin"] >= 0 else f["opp"]
    return (f"**Game view — {f['team']} vs {f['opp']}:** expected margin "
            f"{fav} by {abs(f['exp_margin']):.1f} ({f['team']} win {f['win']:.0%}), "
            f"{f['team']} projected {f['points_for']:.1f} points (typical {f['typical_points']:.1f}). "
            f"Team {what} {typical:.1f} → **{this_game:.1f}** in this game; the player's expected "
            f"{what} {player_typical:.1f} → **{player_game:.1f}**.")


def cached_live(seasons: tuple[int, ...], recency=None) -> dict:
    from . import data as D
    return RO.load_live(tuple(sorted(seasons)), recency=recency or D.RECENCY_DEFAULT)


def cached_rosters(seasons: tuple[int, ...], use_injuries: bool,
                   recency=None) -> pd.DataFrame:
    return RO.league_rosters(cached_live(seasons, recency), use_injuries)
