"""Prop Evaluation — this week's player-prop lines beside the model's frozen
Game-view projection and the play engine's projection of the same line,
settled automatically once the games are played, and graded against the book.

Lines come from The Odds API only when you press Fetch (never on page load):
games kicking off within the next 7 days, two markets by default, skipping
what the ledger already has. Predictions are frozen at fetch time."""

import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from nflsim import data as D, props as P, ui as UI

st.set_page_config(page_title="Prop Evaluation", page_icon="🏈", layout="wide")


def _api_key():
    try:
        k = st.secrets.get("ODDS_API_KEY", "")
    except Exception as e:
        # a secrets file that exists but will not parse (a stray BOM, a bad
        # quote) would otherwise silently disable the fetch buttons
        st.sidebar.error(f"`.streamlit/secrets.toml` could not be read: {e}")
        k = ""
    return k or os.environ.get("ODDS_API_KEY", "")


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner=False)
def get_schedule(season):
    return D.load_schedule((int(season),))


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Building rosters…")
def get_rosters(seasons, recency):
    return UI.cached_rosters(tuple(sorted(seasons)), True, recency)


@st.cache_data(ttl=3600, show_spinner=False)
def get_events(key, days_ahead):
    """The free events call, cached an hour so page reruns don't repeat it."""
    return P.list_events(key, days_ahead)


# --- sidebar ----------------------------------------------------------------
st.sidebar.header("Setup")
seasons, recency = UI.priors_picker("Seasons used to build priors")
ctx = UI.cached_context(tuple(seasons), recency)
season = int(ctx["depth_seasons"][-1])
sched = get_schedule(season)

st.sidebar.divider()
st.sidebar.subheader("Fetch lines")
key = _api_key()
markets = st.sidebar.multiselect(
    "Markets", list(P.MARKETS), default=list(P.DEFAULT_MARKETS),
    format_func=lambda m: P.MARKETS[m][0],
    help="Each market costs one credit per game per fetch. Two markets for a "
         "16-game week is ~32 of the free tier's 500 a month.")
days_ahead = st.sidebar.slider("Games within (days)", 1, 10, 7)
refresh = st.sidebar.toggle("Re-fetch games already in the ledger", value=False,
                            help="Off: games with these markets already recorded are skipped "
                                 "(no credits). On: their lines are replaced by the current "
                                 "ones — do this close to kickoff to record the closing line. "
                                 "Frozen predictions are kept where the line has not moved.")
if not key:
    st.sidebar.warning("No API key. Paste it into `.streamlit/secrets.toml` as "
                       "`ODDS_API_KEY = \"...\"` and restart the app.")
n_events = 0
if key:
    try:
        ev, quota0 = get_events(key, days_ahead)
        n_events = len(ev)
        st.sidebar.caption(f"{n_events} games in the window · a fetch costs up to "
                           f"**{n_events * len(markets)} credits** · "
                           f"{quota0.get('remaining') or '?'} credits remaining")
    except Exception as e:
        st.sidebar.error(f"Could not reach The Odds API: {e}")
go_fetch = st.sidebar.button("Fetch this week's lines", type="primary",
                             disabled=not key or not markets or n_events == 0)

st.sidebar.caption("Predictions are frozen when a line is recorded. After a model change, "
                   "re-project the lines whose games have not started; started or settled "
                   "lines keep the prediction they were graded on.")
go_reproj = st.sidebar.button("Re-project unplayed lines")
if go_reproj:
    UI.cached_play_engine(season)
    bar = st.progress(0.0, text="Re-projecting…")
    led0, n = P.reproject_unplayed(P.load_ledger(), ctx,
                                   progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
    bar.empty(); P.save_ledger(led0)
    st.success(f"Re-projected {n} unplayed lines with the current model.")

if go_fetch:
    UI.cached_play_engine(season)
    bar = st.progress(0.0, text="Fetching…")
    rosters = get_rosters(tuple(seasons), recency)
    led, quota, log = P.fetch_week(key, ctx, sched, rosters, tuple(markets), days_ahead,
                                   refresh=refresh, progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
    bar.empty()
    st.success(f"Ledger now has {len(led)} lines · {quota.get('remaining') or '?'} credits remaining")
    with st.expander("Fetch log"):
        st.write("\n".join(f"- {l}" for l in log))

# --- ledger, settled and graded ---------------------------------------------
led = P.load_ledger()
st.title("🏈 Prop Evaluation")
if led.empty:
    st.info("No lines recorded yet. Fetch this week's lines from the sidebar; each fetch is "
            "added to `props/ledger.csv`, the model's Game-view projection is frozen at that "
            "moment, and the actual stat fills in once the game has been played.")
    st.stop()

# settle anything now playable; re-project unplayed lines predicted by an older
# model version (started / settled lines keep the prediction they were graded on)
wk_all = D.load_weekly(tuple(sorted(set(int(s) for s in led["season"].dropna().unique()))), recency)
if led["player_id"].isna().any():
    led, _ = P.rematch(led, get_rosters(tuple(seasons), recency))
led = P.fill_actuals(led, wk_all)
# anything open (not kicked off) that the current model has not projected — a
# stale version, or no play-engine projection yet because the line was recorded
# before the play engine joined the ledger — is projected now, never afterwards
_open = led["result"].isna() & led["player_id"].notna() & \
        (pd.to_datetime(led["commence"], utc=True, errors="coerce") > pd.Timestamp.now(tz="UTC"))
_todo = _open & ((led["model_version"].astype(str) != P.model_version()) |
                 led["pred_mean"].isna() | led["play_mean"].isna())
if _todo.any():
    UI.cached_play_engine(season)       # builds the tables once per process, with its own spinner
    bar = st.progress(0.0, text="Checking predictions against the current model…")
    led, n_stale = P.reproject_unplayed(led, ctx, only_stale=True,
                                        progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
    if n_stale:
        st.info(f"The model changed since {n_stale} unplayed lines were projected — they have been "
                f"re-projected with version `{P.model_version()}`. Started and settled lines are untouched.")
    if (_open & led["play_mean"].isna()).any():
        bar.progress(0.0, text="Play-engine projections for lines that have none (one simulated game per fixture)…")
        led = P.predict_missing(led, ctx, progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
    elif (_open & led["pred_mean"].isna()).any():
        led = P.predict_missing(led, ctx)
    bar.empty()
P.save_ledger(led)

st.sidebar.divider()
st.sidebar.subheader("Evaluate")
engine = st.sidebar.radio("Projection graded", list(P.ENGINES), horizontal=True,
                          format_func=lambda e: P.ENGINE_LABELS[e],
                          help="**Game view**: the single-stat models with this game's script "
                               "factors — what the player pages show.  \n**Play engine**: every "
                               "snap simulated and the player read off the box score, so his "
                               "volume comes from the clock and the score. Both are frozen when a "
                               "line is recorded; the comparison below scores them on the same lines.")
use = st.sidebar.radio("Compare the line with the model's", ["median", "mean"], horizontal=True,
                       help="Median is the fair over/under point of a right-skewed yardage "
                            "distribution; the mean is what a book's line usually tracks.")
edge = st.sidebar.slider("Edge to count as a pick", 0.0, 0.15, 0.03, 0.01,
                         help="The model only 'picks' a side when its P(over) differs from the "
                              "book's vig-free probability by at least this much.")
books = sorted(led["bookmaker"].dropna().unique().tolist())
book = st.sidebar.selectbox("Bookmaker", ["all (median line)"] + books)
weeks = sorted(led["week"].dropna().astype(int).unique().tolist())
week_sel = st.sidebar.multiselect("Weeks", weeks, default=weeks)
mkt_sel = st.sidebar.multiselect("Markets shown", list(P.MARKETS), default=[m for m in P.MARKETS if m in led["market"].unique()],
                                 format_func=lambda m: P.MARKETS[m][0])

d = led[led["week"].isin(week_sel) & led["market"].isin(mkt_sel)].copy()
if book != "all (median line)":
    d = d[d["bookmaker"] == book]
else:
    # consensus: median line and median prices across books, one row per player-market-game
    keys = ["season", "week", "game_id", "home", "away", "market", "player", "player_id", "team"]
    agg = d.groupby(keys, dropna=False).agg(
        line=("line", "median"), over_price=("over_price", "median"), under_price=("under_price", "median"),
        pred_mean=("pred_mean", "median"), pred_median=("pred_median", "median"), p_over=("p_over", "median"),
        play_mean=("play_mean", "median"), play_median=("play_median", "median"), play_p_over=("play_p_over", "median"),
        actual=("actual", "first"), result=("result", "first"), commence=("commence", "first"),
        model_version=("model_version", "first"), bookmaker=("bookmaker", "nunique")).reset_index()
    d = agg
g = P.grade(d, use=use, edge=edge, engine=engine)
games_sel = st.sidebar.multiselect("Games", sorted((g["away"] + " @ " + g["home"]).unique().tolist()), default=[])
if games_sel:
    g = g[(g["away"] + " @ " + g["home"]).isin(games_sel)]

s = P.summary(g)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Lines", f"{len(g):,}", help="After the filters; one row per player-market-book, or per player-market when the consensus line is selected.")
c2.metric("Settled", f"{s.get('settled', 0):,}")
if s.get("settled", 0):
    c3.metric(f"{P.ENGINE_LABELS[engine]} {use} vs line — hit rate", f"{s['centre_hit']:.1%}" if np.isfinite(s.get("centre_hit", np.nan)) else "—",
              help=f"How often the side the {P.ENGINE_LABELS[engine]} {use} sits on came in. 52.4% breaks even at -110. n = {s['centre_n']}.")
    c4.metric(f"Picks (edge ≥ {edge:.0%}) — hit rate", f"{s['pick_hit']:.1%}" if np.isfinite(s.get("pick_hit", np.nan)) else "—",
              help=f"Only lines where the model disagrees with the book by the edge. n = {s['pick_n']}.")
    c5.metric("Brier: model / book", f"{s['model_brier']:.3f} / {s['book_brier']:.3f}",
              help="Squared error of P(over) against the outcome; lower is better. The book's "
                   "vig-free probability is the benchmark.")
    if "mae_line" in s:
        st.caption(f"Yardage markets: MAE of the model's mean **{s['mae_mean']:.1f}**, median "
                   f"**{s['mae_median']:.1f}**, the line itself **{s['mae_line']:.1f}** · outcomes went "
                   f"over {s['over_rate']:.0%} of the time; the model had the over favoured on "
                   f"{s['model_over']:.0%} of lines; the book's favoured side hit {s['book_hit']:.1%}.")
else:
    st.caption("Nothing settled yet — actuals fill in automatically once the games are in the "
               "weekly feed (nflverse publishes within hours of the final whistle).")

# the two engines head to head, on lines where both projections were frozen
cmp_ = P.compare_engines(d, use=use, edge=edge)
if not cmp_.empty:
    with st.expander("Game view vs play engine, on the same settled lines", expanded=True):
        st.dataframe(cmp_.rename(columns={"engine": "Projection", "settled": "Settled",
                                          "centre_hit": f"{use.title()} vs line hit", "centre_n": "n",
                                          "pick_hit": f"Picks ≥{edge:.0%} hit", "pick_n": "n picks",
                                          "brier": "Brier", "mae_mean": "MAE mean", "mae_median": "MAE median",
                                          "model_over": "Favours over"})
                     .style.format({f"{use.title()} vs line hit": "{:.1%}", f"Picks ≥{edge:.0%} hit": "{:.1%}",
                                    "Brier": "{:.3f}", "MAE mean": "{:.1f}", "MAE median": "{:.1f}",
                                    "Favours over": "{:.0%}"}, na_rep="—"),
                     hide_index=True, width="stretch")
        st.caption("Only lines with both projections frozen before kickoff count, so neither engine "
                   "gets an easier subset. The book row uses the line itself (MAE) and its favoured "
                   "side (hit rate). The play engine joined the ledger in week 2; on the 2025 replay "
                   "(roadmap §12) the two were a dead heat on yards, the play engine fixing the drive "
                   "engine's low medians for 40–60-yard receivers.")
else:
    st.caption("The play engine's projections are frozen beside the Game view from week 2 on; once "
               "those lines settle, the two engines are compared here on the same lines.")

# where the model's centre sits against the book's number, by model version —
# this is the thing to watch as new weeks come in
yd = g[g["market"].isin(["player_rush_yds", "player_reception_yds"]) & g["eng_mean"].notna()]
if not yd.empty:
    with st.expander(f"{P.ENGINE_LABELS[engine]} centre vs the book's line, by model version"):
        bias = (yd.assign(mean_line=yd["eng_mean"] - yd["line"],
                          median_line=yd["eng_median"] - yd["line"],
                          actual_line=yd["actual"] - yd["line"])
                  .groupby(["model_version", "market"])
                  .agg(lines=("line", "size"), settled=("actual", "count"),
                       mean_minus_line=("mean_line", "mean"), median_minus_line=("median_line", "mean"),
                       actual_minus_line=("actual_line", "mean"),
                       model_over=("eng_p", lambda x: float((x > 0.5).mean())))
                  .reset_index())
        bias["market"] = bias["market"].map(lambda m: P.MARKETS[m][0])
        bias["current"] = np.where(bias["model_version"] == P.model_version(), "◀ current", "")
        st.dataframe(bias.style.format({"mean_minus_line": "{:+.1f}", "median_minus_line": "{:+.1f}",
                                        "actual_minus_line": "{:+.1f}", "model_over": "{:.0%}"}, na_rep="—"),
                     hide_index=True, width="stretch")
        st.caption("Yards. A calibrated model has *mean − line* near zero and *median − line* a "
                   "little below it (yardage is right-skewed, and books tend to set the number "
                   "near the mean). Week 1 (version 47d6…) had medians 8 yards under the line "
                   "on receiving — the shape bug fixed on 14 Sep. `model_over` is the share of "
                   "lines where the model favours the over; `actual − line` shows how the week "
                   "itself ran.")

# --- the table -----------------------------------------------------------------
show = g.copy()
show["Game"] = show["away"] + " @ " + show["home"]
show["Market"] = show["market"].map(lambda m: P.MARKETS[m][0])
show["Edge"] = show["edge"]
cols = ["week", "Game", "player", "team", "Market", "line", "pred_median", "pred_mean", "p_over",
        "play_median", "play_mean", "play_p_over",
        "book_p_over", "Edge", "pick", "actual", "result", "hit"]
if book != "all (median line)":
    cols.insert(6, "over_price"); cols.insert(7, "under_price")
else:
    show["books"] = show["bookmaker"]; cols.append("books")
view = show[cols].rename(columns={"week": "Wk", "player": "Player", "team": "Team", "line": "Line",
                                  "pred_median": "Game median", "pred_mean": "Game mean", "p_over": "Game P(over)",
                                  "play_median": "Play median", "play_mean": "Play mean", "play_p_over": "Play P(over)",
                                  "book_p_over": "Book P(over)", "pick": "Pick",
                                  "actual": "Actual", "result": "Result", "hit": "Hit",
                                  "over_price": "Over", "under_price": "Under", "books": "Books"})
view = view.sort_values(["Wk", "Game", "Market", "Edge"], ascending=[True, True, True, False])
fmt = {"Line": "{:.1f}", "Game median": "{:.1f}", "Game mean": "{:.1f}", "Game P(over)": "{:.0%}",
       "Play median": "{:.1f}", "Play mean": "{:.1f}", "Play P(over)": "{:.0%}",
       "Book P(over)": "{:.0%}", "Edge": "{:+.0%}", "Actual": "{:.0f}"}
st.dataframe(view.style.format(fmt, na_rep="—"), width="stretch", hide_index=True, height=560)
st.caption("The model columns were frozen when the line was recorded (see `predicted_at` in the "
           "ledger) — nothing is re-projected after the fact. *Game* is the Game-view projection, "
           "*Play* the play engine's (week 2 on; blank where the line was recorded before the play "
           f"engine joined the ledger). *Edge* and *Pick* are the {P.ENGINE_LABELS[engine]}'s: the side "
           "it backs when its P(over) beats the book's vig-free probability by the edge; *Hit* "
           "whether it came in. Void = the player did not play.")

# --- charts ----------------------------------------------------------------------
settled = g[g["result"].isin(["over", "under"])]
if len(settled) >= 20:
    left, right = st.columns(2)
    with left:
        st.subheader("Calibration of P(over)")
        b = settled[settled["eng_p"].notna()].copy()
        b["bin"] = pd.cut(b["eng_p"], np.linspace(0, 1, 6), include_lowest=True)
        cal = b.groupby("bin", observed=True).agg(n=("eng_p", "size"), pred=("eng_p", "mean"),
                                                   obs=("result", lambda r: float((r == "over").mean()))).reset_index()
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#888"), showlegend=False)
        fig.add_scatter(x=cal["pred"], y=cal["obs"], mode="markers+lines", marker=dict(size=np.sqrt(cal["n"]) * 4, color="#2e7d5b"),
                        text=cal["n"], hovertemplate="model %{x:.0%}<br>observed %{y:.0%}<br>%{text} lines<extra></extra>", showlegend=False)
        fig.update_layout(height=360, xaxis_title="Model P(over)", yaxis_title="Share that went over",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%", margin=dict(t=20, b=50, l=60, r=20))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Model vs line vs actual (yards)")
        y = settled[settled["market"].isin(["player_rush_yds", "player_reception_yds"]) & settled["eng_mean"].notna()]
        if not y.empty:
            centre = y["eng_median"] if use == "median" else y["eng_mean"]
            fig = go.Figure()
            fig.add_scatter(x=y["line"], y=y["actual"], mode="markers", name="line", marker=dict(color="#888", size=6, opacity=0.6),
                            text=y["player"], hovertemplate="%{text}<br>line %{x:.1f} · actual %{y:.0f}<extra></extra>")
            fig.add_scatter(x=centre, y=y["actual"], mode="markers", name=f"model {use}", marker=dict(color="#2e7d5b", size=6, opacity=0.6),
                            text=y["player"], hovertemplate="%{text}<br>model %{x:.1f} · actual %{y:.0f}<extra></extra>")
            m = float(max(y["actual"].max(), y["line"].max()))
            fig.add_scatter(x=[0, m], y=[0, m], mode="lines", line=dict(dash="dot", color="#888"), showlegend=False)
            fig.update_layout(height=360, xaxis_title="Projected", yaxis_title="Actual yards",
                              margin=dict(t=20, b=50, l=60, r=20), legend=dict(orientation="h"))
            st.plotly_chart(fig, width="stretch")

with st.expander("Biggest misses and best calls"):
    y = settled[settled["market"].isin(["player_rush_yds", "player_reception_yds"]) & settled["eng_mean"].notna()].copy()
    if not y.empty:
        centre = y["eng_median"] if use == "median" else y["eng_mean"]
        y["err"] = centre - y["actual"]; y["line_err"] = y["line"] - y["actual"]
        cols2 = ["week", "player", "market", "line", "eng_median", "eng_mean", "actual", "err", "line_err"]
        a, b_ = st.columns(2)
        a.markdown("**Model too high**"); a.dataframe(y.nlargest(10, "err")[cols2].round(1), hide_index=True, width="stretch")
        b_.markdown("**Model too low**"); b_.dataframe(y.nsmallest(10, "err")[cols2].round(1), hide_index=True, width="stretch")
    else:
        st.write("Nothing settled yet.")

st.caption("Lines from The Odds API, fetched only on request and recorded in `props/ledger.csv`. "
           "For research/entertainment.")
