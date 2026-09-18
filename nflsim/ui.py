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

import threading
import time

import numpy as np
import pandas as pd
import streamlit as st

from . import roster as RO


# ---------------------------------------------------------------------------
# Progress with a time estimate
# ---------------------------------------------------------------------------
# A cached function must not draw Streamlit elements (they would be recorded
# and replayed on every cache hit), so the long computations report a fraction
# into a per-thread slot and the page thread draws ONE bar from it. The cached
# function runs in a worker thread; a cache hit returns before any bar is drawn.

_PROGRESS: dict[int, dict] = {}


def report(frac: float, text: str = "") -> None:
    """Called from inside the long computations: how far along, and what is
    being worked on. A no-op unless the call is under `run_with_progress`."""
    state = _PROGRESS.get(threading.get_ident())
    if state is not None:
        state["frac"], state["text"] = float(frac), str(text)


def _fmt_secs(x: float) -> str:
    x = max(int(round(x)), 0)
    return f"{x // 60}:{x % 60:02d}" if x >= 60 else f"{x} s"


def run_with_progress(label: str, fn, *args, **kwargs):
    """Run `fn(*args)` (normally one of the cached functions below) and, while
    it runs, show a bar with what it is doing, the time elapsed and an estimate
    of the time left. Returns its result; re-raises its exception."""
    from streamlit.runtime.scriptrunner import add_script_run_ctx
    state = {"frac": 0.0, "text": "", "result": None, "error": None, "done": False}

    def work():
        _PROGRESS[threading.get_ident()] = state
        try:
            state["result"] = fn(*args, **kwargs)
        except BaseException as e:            # noqa: BLE001 — re-raised on the page thread
            state["error"] = e
        finally:
            _PROGRESS.pop(threading.get_ident(), None)
            state["done"] = True

    th = threading.Thread(target=work, daemon=True)
    add_script_run_ctx(th)
    th.start()
    th.join(0.3)                              # a cache hit is back before a bar is worth drawing
    bar, t0 = None, time.time()
    while not state["done"]:
        if bar is None:
            bar = st.progress(0.0, text=label)
        frac, el = min(state["frac"], 1.0), time.time() - t0
        eta = f" · about {_fmt_secs(el / frac - el)} left" if frac > 0.03 else ""
        what = f" — {state['text']}" if state["text"] else ""
        bar.progress(frac, text=f"{label}{what} · {_fmt_secs(el)} elapsed{eta}")
        th.join(0.5)
    if bar is not None:
        bar.empty()
    if state["error"] is not None:
        raise state["error"]
    return state["result"]


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
    """The share to simulate with on a single-stat page — "if he plays": the
    player's share IN THE GAMES HE PLAYED, blended and redistributed exactly
    like the roster's unconditional share (`*_cond` columns). The roster's own
    share counts his historically missed games as zero — right for a season
    projection, wrong for a game he is simulated to be in. Live if active, own
    history if ruled out."""
    cond = f"{share_col}_cond"
    if bool(row.get("active", True)) and float(row.get(cond, row[share_col])) > 0:
        return float(row.get(cond, row[share_col]))
    return float(row.get(f"own_{cond}", row.get(f"own_{share_col}", row[share_col])))


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
def _cached_context(seasons: tuple, recency) -> dict:
    from . import game as G
    return G.prepare(seasons, recency=recency)


def cached_context(seasons: tuple[int, ...], recency) -> dict:
    """The full game-engine context (`game.prepare`), shared by every page
    that offers the game view — one download and one set of ratings per
    (seasons, recency) for the whole app."""
    return _cached_context(_norm_seasons(seasons), recency)


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


@st.cache_data(ttl=6 * 3600, show_spinner="Loading depth charts and injury reports…")
def _cached_live(seasons: tuple, recency) -> dict:
    return RO.load_live(seasons, recency=recency)


def cached_live(seasons: tuple[int, ...], recency=None) -> dict:
    from . import data as D
    return _cached_live(_norm_seasons(seasons), recency or D.RECENCY_DEFAULT)


@st.cache_data(ttl=6 * 3600, show_spinner="Building every team's roster…")
def _cached_rosters(seasons: tuple, use_injuries: bool, recency) -> pd.DataFrame:
    return RO.league_rosters(_cached_live(seasons, recency), use_injuries)


def cached_rosters(seasons: tuple[int, ...], use_injuries: bool,
                   recency=None) -> pd.DataFrame:
    from . import data as D
    return _cached_rosters(_norm_seasons(seasons), bool(use_injuries), recency or D.RECENCY_DEFAULT)


def _norm_seasons(seasons) -> tuple:
    """Cache keys are built from the raw arguments, so every caller must hand
    over the same thing: a sorted tuple of ints, whatever order the widget gave."""
    return tuple(sorted(int(s) for s in seasons))


def engine_picker(key: str = "engine") -> str:
    """Drive engine (default) or the play-level engine, on the pages that
    simulate whole games. The play engine builds its tables on first use
    (~2-3 minutes once per app process) and runs ~10x slower per game."""
    opts = ["Drive", "Play-level"]
    lbl = st.sidebar.radio(
        "Game engine", opts, index=1 if DEFAULT_ENGINE == "play" else 0, horizontal=True, key=f"{key}_engine",
        help="**Drive**: one draw per possession; the engine every page was built on.  \n"
             "**Play-level**: every snap simulated from ten seasons of play-by-play — "
             "volume comes from the clock and the score, so a quarterback's yards move "
             "with the game total and a trailing team really does throw more. Same "
             "expected margin and total (both are steered to the ratings layer); "
             "different shape. Not yet shown to beat the drive engine on props — "
             "see page 12 for the gates.")
    return "play" if lbl.startswith("Play") else "drive"


# ---------------------------------------------------------------------------
# One cache for the whole app
# ---------------------------------------------------------------------------
# Streamlit's caches are per PROCESS but keyed per FUNCTION, so two pages that
# wrap the same computation in their own @st.cache_data each pay for it. Every
# page that simulates a game or a slate now goes through the functions below,
# and Home.py can run them all once for the default settings ("warm-up") so
# that the pages open instantly afterwards. Pages must call with the same
# arguments the warm-up used — DEFAULTS is the single source of those.

DEFAULTS = dict(
    n_sims_game=20000,        # page 7, one fixture
    n_sims_slate=10000,       # page 8 fantasy, page 9 pick'em, drive engine
    n_sims_slate_play=5000,   # the same slates on the play engine (~7 s a game; the
                              # Monte-Carlo error on a player mean is well under a yard)
    use_injuries=True,
    scoring="PPR",
)
DEFAULT_ENGINE = "play"       # the engine every page opens on — one line to flip; the props page's gate is the tripwire (ROADMAP §18)


def sims_picker(engine: str, key: str) -> int:
    """Simulations per game for a slate, with the engine's own default so the
    warm-up's cache is what the page opens on."""
    if engine == "play":
        return int(st.sidebar.select_slider("Simulations per game", [2000, 5000, 10000],
                                            value=DEFAULTS["n_sims_slate_play"], key=f"{key}_sims_play"))
    return int(st.sidebar.select_slider("Simulations per game", [4000, 10000, 20000],
                                        value=DEFAULTS["n_sims_slate"], key=f"{key}_sims_drive"))


def default_priors() -> tuple:
    """(seasons, recency) exactly as `priors_picker` returns them untouched."""
    from . import data as D
    _, defaults = D.season_choices()
    # sorted, because every page passes tuple(sorted(seasons)) — the cache key must match
    return tuple(sorted(int(s) for s in defaults)), D.RECENCY_PRESETS["Balanced"]


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_schedule(season: int) -> pd.DataFrame:
    from . import data as D
    return D.load_schedule((int(season),))


@st.cache_data(ttl=24 * 3600, show_spinner="Loading 15 seasons of closing lines…")
def cached_history() -> pd.DataFrame:
    from . import market as MK
    return MK.load_history()


@st.cache_resource(show_spinner=False)
def cached_play_engine(latest_season: int) -> dict:
    """The play engine's tables, sensitivities and team tendencies, built once
    per process (from disk when this code version has built them before) and
    shared with `game.run_game` through the module registry."""
    from . import playengine as PE
    PE.attach({"depth_seasons": (int(latest_season),)}, progress=report)
    return PE._ENGINE


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _cached_game(seasons: tuple, recency, home: str, away: str, n_sims: int, neutral: bool,
                use_injuries: bool, wind: float, roof: str, engine: str) -> dict:
    from . import game as G
    ctx = cached_context(seasons, recency)
    if engine == "play":
        cached_play_engine(int(ctx["depth_seasons"][-1]))
    r_home = G.roster_for(ctx, home, use_injuries=use_injuries)
    r_away = G.roster_for(ctx, away, use_injuries=use_injuries)
    return G.run_game(ctx, r_home, r_away, home, away, n_sims=int(n_sims), seed=11,
                      home=None if neutral else "a", avail=ctx.get("avail"),
                      wind=wind, roof=roof, engine=engine, progress=report)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _cached_week(seasons: tuple, recency, week: int, rules_items: tuple, n_sims: int,
                use_injuries: bool, engine: str):
    from . import fantasy as F
    ctx = cached_context(seasons, recency)
    if engine == "play":
        cached_play_engine(int(ctx["depth_seasons"][-1]))
    sched = cached_schedule(int(ctx["depth_seasons"][-1]))
    games = sched[sched["week"] == int(week)]
    return F.week_projections(ctx, games, dict(rules_items), n_sims=int(n_sims),
                              use_injuries=use_injuries, engine=engine, progress=report)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _cached_slate(seasons: tuple, recency, week: int, n_sims: int, use_injuries: bool, engine: str) -> dict:
    from . import pickem as P
    ctx = cached_context(seasons, recency)
    if engine == "play":
        cached_play_engine(int(ctx["depth_seasons"][-1]))
    sched = cached_schedule(int(ctx["depth_seasons"][-1]))
    games = sched[sched["week"] == int(week)]
    return P.simulate_slate(ctx, games, n_sims=int(n_sims), use_injuries=use_injuries, engine=engine, progress=report)


@st.cache_data(ttl=6 * 3600, show_spinner="Refitting the model week by week for past games…")
def _cached_team_backtest(seasons: tuple, recency) -> pd.DataFrame:
    from . import backtest as B
    return B.team_backtest(list(seasons), recency=recency, availability=True)


def warm_up(progress=None, engines=("drive", "play")) -> list[str]:
    """Compute everything the pages need for the default settings, so every
    page opens from cache. Returns a log of what was built and how long it took."""
    import time
    from . import data as D, fantasy as F
    log = []
    seasons, recency = default_priors()

    def step(label, fn):
        t = time.time()
        if progress:
            progress(label)
        run_with_progress(label, fn)
        log.append(f"{label} — {time.time() - t:.0f}s")

    step("Priors, ratings, depth charts and injuries", lambda: cached_context(seasons, recency))
    ctx = cached_context(seasons, recency)
    season = int(ctx["depth_seasons"][-1])
    step("Schedule", lambda: cached_schedule(season))
    sched = cached_schedule(season)
    week = D.current_week(sched)
    step("Live rosters for every team", lambda: cached_rosters(seasons, DEFAULTS["use_injuries"], recency))
    step("Fifteen seasons of closing lines", cached_history)
    step(f"Week {week} slate, drive engine (pick'em)",
         lambda: cached_slate(seasons, recency, week, DEFAULTS["n_sims_slate"], DEFAULTS["use_injuries"], "drive"))
    rules = tuple(sorted(F.PRESETS[DEFAULTS["scoring"]].items()))
    step(f"Week {week} fantasy projections, drive engine",
         lambda: cached_week(seasons, recency, week, rules, DEFAULTS["n_sims_slate"], DEFAULTS["use_injuries"], "drive"))
    step(f"Card history for {season} (pick'em replay)", lambda: cached_team_backtest((season,), recency))
    wg = sched[(sched["week"] == week) & ~sched["played"]]
    g = (wg if len(wg) else sched[sched["week"] == week]).iloc[0]
    roof = str(g["roof"]) if pd.notna(g.get("roof")) else "outdoors"
    wind = float(g["wind"]) if pd.notna(g.get("wind")) else 0.0
    step(f"{g['away_team']} @ {g['home_team']}, drive engine (game page default)",
         lambda: cached_game(seasons, recency, g["home_team"], g["away_team"], DEFAULTS["n_sims_game"],
                             False, DEFAULTS["use_injuries"], wind, roof, "drive"))
    if "play" in engines:
        step("Play-level engine tables (from disk after the first build)", lambda: cached_play_engine(season))
        step(f"{g['away_team']} @ {g['home_team']}, play engine",
             lambda: cached_game(seasons, recency, g["home_team"], g["away_team"], DEFAULTS["n_sims_game"],
                                 False, DEFAULTS["use_injuries"], wind, roof, "play"))
        step(f"Week {week} slate, play engine (pick'em)",
             lambda: cached_slate(seasons, recency, week, DEFAULTS["n_sims_slate_play"], DEFAULTS["use_injuries"], "play"))
        step(f"Week {week} fantasy projections, play engine",
             lambda: cached_week(seasons, recency, week, rules, DEFAULTS["n_sims_slate_play"], DEFAULTS["use_injuries"], "play"))
    return log


def cached_game(seasons: tuple, recency, home: str, away: str, n_sims: int, neutral: bool,
                use_injuries: bool, wind: float, roof: str, engine: str):
    return _cached_game(_norm_seasons(seasons), recency, str(home), str(away), int(n_sims), bool(neutral), bool(use_injuries), float(wind), str(roof), str(engine))


def cached_week(seasons: tuple, recency, week: int, rules_items: tuple, n_sims: int,
                use_injuries: bool, engine: str):
    return _cached_week(_norm_seasons(seasons), recency, int(week), tuple(sorted((str(k), float(v)) for k, v in dict(rules_items).items())), int(n_sims), bool(use_injuries), str(engine))


def cached_slate(seasons: tuple, recency, week: int, n_sims: int, use_injuries: bool, engine: str):
    return _cached_slate(_norm_seasons(seasons), recency, int(week), int(n_sims), bool(use_injuries), str(engine))


def cached_team_backtest(seasons: tuple, recency):
    return _cached_team_backtest(_norm_seasons(seasons), recency)
