"""Play Engine — the play-level simulator beside the drive engine.

A lab page: nothing here changes what the other pages do. It shows the engine's
league-level gate (does a season of simulated games look like a real one, with
no tuning?), runs both engines on the same fixture side by side, and can score
the engine's shape against the harness's normal curve on a sample of games.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from nflsim import data as D, game as G, teams as T, playengine as PE, backtest as B
from nflsim import ui as UI

st.set_page_config(page_title="Play Engine", page_icon="🏈", layout="wide")


def get_engine(latest_season):
    eng = UI.cached_play_engine(int(latest_season))
    return PE.load_plays(), eng["tables"], eng["sens"]


def get_schedule(season):
    return UI.cached_schedule(int(season))


def get_context(seasons, recency):
    return UI.cached_context(tuple(sorted(seasons)), recency)


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner=False)
def get_roster(seasons, recency, team, use_injuries):
    ctx = get_context(seasons, recency)
    return G.roster_for(ctx, team, use_injuries=use_injuries)


@st.cache_data(show_spinner="Simulating a neutral season…")
def league_gate(n, latest_season):
    plays, tables, _ = get_engine(latest_season)
    sim = PE.simulate(tables, n=int(n), seed=1)
    return PE.league_check(sim, plays)


# --- sidebar ----------------------------------------------------------------
st.sidebar.header("Setup")
seasons, recency = UI.priors_picker("Seasons used to build priors")
ctx = get_context(tuple(seasons), recency)
ratings = ctx["ratings"]
sched = get_schedule(ctx["depth_seasons"][-1])
n_sims = st.sidebar.select_slider("Simulations", [2000, 5000, 10000, 20000], value=5000)
use_inj = st.sidebar.toggle("Drop players ruled out (drive engine)", value=True)

plays, tables, sens = get_engine(ctx["depth_seasons"][-1])

st.title("🏈 Play Engine")
st.caption(f"Ten seasons of play-by-play ({len(plays):,} plays, {plays['game_id'].nunique():,} games) "
           f"→ {len(tables['outcome_sampler'].cells):,} outcome cells, "
           f"{len(tables['pass_base'].cells) + len(tables['special_early'].cells) + len(tables['fourth_base'].cells):,} decision cells. "
           "The drive engine on every other page is untouched.")

# --- 1. league gate -------------------------------------------------------------
st.subheader("1 · Does a simulated season look like a real one?")
st.caption("Every team identical, no home field, **no tuning**: the numbers below come out of "
           "the play mechanics alone. Spread (margin / total sd) is narrower than real because "
           "real teams differ — team strength is added in §2. Mass at exactly 3 is still short: "
           "the last of it needs timeouts as a resource and the two-minute warning.")
lg = league_gate(4000, int(ctx["depth_seasons"][-1]))
lg["ratio"] = lg["engine"] / lg["real"].replace(0, np.nan)
st.dataframe(lg.style.format({"engine": "{:.3f}", "real": "{:.3f}", "ratio": "{:.2f}"}, na_rep="—"),
             hide_index=True, width="stretch", height=min(700, 38 * len(lg) + 40))

# --- 2. one fixture, both engines -----------------------------------------------
st.subheader("2 · One game, both engines")
weeks = sorted(sched["week"].unique().tolist()) if not sched.empty else [1]
cur = D.current_week(sched) if not sched.empty else 1
c1, c2 = st.columns([1, 3])
with c1:
    week = st.selectbox("Week", weeks, index=weeks.index(cur) if cur in weeks else 0)
wg = sched[sched["week"] == week].reset_index(drop=True)
labels = [D.game_label(r) for _, r in wg.iterrows()]
with c2:
    pick = st.selectbox("Game", labels, index=0)
g = wg.iloc[labels.index(pick)]
home, away = g["home_team"], g["away_team"]
roof = str(g["roof"]) if pd.notna(g.get("roof")) else "outdoors"
wind = float(g["wind"]) if pd.notna(g.get("wind")) else 0.0

with st.spinner("Running both engines…"):
    ps = PE.simulate_matchup(tables, sens, ratings, home, away, n=int(n_sims), seed=7,
                             avail=ctx.get("avail"), wind=wind, roof=roof)
    try:
        r_home = get_roster(tuple(seasons), recency, home, use_inj)
        r_away = get_roster(tuple(seasons), recency, away, use_inj)
        ds = G.simulate_game(ratings, ctx["wk"], r_home, r_away, home, away,
                             ctx["pass_vol"], ctx["rush_vol"], ctx["rush_def"], ctx["lg_pass"],
                             home="a", n_sims=int(n_sims), seed=7, avail=ctx.get("avail"),
                             wind=wind, roof=roof, target_rate=ctx.get("target_rate"))
        dm, dt_ = ds["points_a"] - ds["points_b"], ds["points_a"] + ds["points_b"]
    except Exception as e:
        ds, dm, dt_ = None, None, None
        st.warning(f"Drive engine could not run this fixture: {e}")

pm, pt = ps["margin"], ps["total"]
rows = [("expected margin (ratings target)", ps["target_margin"], ps["target_margin"]),
        (f"{home} points", ps["points_home"].mean(), ds["points_a"].mean() if ds else np.nan),
        (f"{away} points", ps["points_away"].mean(), ds["points_b"].mean() if ds else np.nan),
        (f"{home} win probability", ps["win_home"], float((dm > 0).mean() + 0.5 * (dm == 0).mean()) if ds else np.nan),
        ("margin sd", pm.std(), dm.std() if ds else np.nan),
        ("total sd", pt.std(), dt_.std() if ds else np.nan),
        ("P(margin exactly 3)", (np.abs(pm) == 3).mean(), (np.abs(dm) == 3).mean() if ds else np.nan),
        ("P(margin exactly 7)", (np.abs(pm) == 7).mean(), (np.abs(dm) == 7).mean() if ds else np.nan),
        ("P(tie)", (pm == 0).mean(), (dm == 0).mean() if ds else np.nan),
        ("corr(home pts, away pts)", np.corrcoef(ps["points_home"], ps["points_away"])[0, 1],
         np.corrcoef(ds["points_a"], ds["points_b"])[0, 1] if ds else np.nan)]
if pd.notna(g.get("spread_line")):
    sp = float(g["spread_line"])
    pc, pp = PE.cover_prob(ps, sp)
    rows.append((f"P({home} covers {-sp:+g})", pc / max(1 - pp, 1e-9),
                 float((dm > sp).mean() / max(1 - (dm == sp).mean(), 1e-9)) if ds else np.nan))
if pd.notna(g.get("total_line")):
    tl = float(g["total_line"])
    po, pp = PE.total_prob(ps, tl)
    rows.append((f"P(over {tl:g})", po / max(1 - pp, 1e-9),
                 float((dt_ > tl).mean() / max(1 - (dt_ == tl).mean(), 1e-9)) if ds else np.nan))
cmp = pd.DataFrame(rows, columns=["metric", "play engine", "drive engine"])
st.dataframe(cmp.style.format({"play engine": "{:.3f}", "drive engine": "{:.3f}"}, na_rep="—"),
             hide_index=True, width="stretch", height=min(600, 38 * len(cmp) + 40))
if bool(g.get("played")):
    st.caption(f"Final: {away} {int(g['away_score'])} — {home} {int(g['home_score'])}.")

# margin distributions overlaid
fig = go.Figure()
edges = np.arange(-45.5, 46.5, 1.0)
h1, _ = np.histogram(pm, bins=edges)
fig.add_bar(x=(edges[:-1] + edges[1:]) / 2, y=h1 / h1.sum(), name="Play engine", marker_color="#2e7d5b", opacity=0.75)
if ds is not None:
    h2, _ = np.histogram(dm, bins=edges)
    fig.add_bar(x=(edges[:-1] + edges[1:]) / 2, y=h2 / h2.sum(), name="Drive engine", marker_color="#c0563b", opacity=0.55)
fig.update_layout(barmode="overlay", height=380, xaxis_title=f"{home} margin", yaxis_title="Share of simulations",
                  yaxis_tickformat=".1%", legend=dict(orientation="h", y=1.1), margin=dict(t=20, b=45, l=60, r=20))
st.plotly_chart(fig, width="stretch")
st.caption("Both engines are steered to the same expected margin and total from the ratings layer; "
           "the shapes are their own. The play engine's spikes at 3 and 7 come from how games end.")

# team box from the play engine
with st.expander("Play engine team box (means)"):
    stt = ps["stats"]
    box = pd.DataFrame({k: [stt[k][:, 0].mean(), stt[k][:, 1].mean()] for k in
                        ("plays", "pass_att", "comp", "pass_yds", "rush_att", "rush_yds", "sacks", "ints",
                         "fum_lost", "pass_td", "rush_td", "fg_att", "fg_made", "punts", "drives",
                         "first_downs", "penalties")}, index=[home, away]).T
    st.dataframe(box.style.format("{:.1f}"), width="stretch")
    st.caption("Team-level only until stage 4 puts the depth chart on top. `plays` counts every "
               "snap including kicks and penalties.")

# --- 3. shape gate on a sample --------------------------------------------------
st.subheader("3 · Shape gate: engine vs the normal curve, same means")
st.caption("Takes the harness's out-of-sample expected margin and total for real games, steers "
           "the engine to them, and scores the engine's winner / cover / over probabilities "
           "against the normal-curve versions the harness uses. Same means — only the shape differs.")
n_games = st.slider("Games to sample from 2025", 20, 120, 40, 10)
if st.button("Run the shape gate"):
    with st.spinner("Refitting the ratings week by week and simulating…"):
        bt = B.team_backtest([2025], recency=recency, availability=True)
        bt = bt.sample(n=min(n_games, len(bt)), random_state=1)
        sb = PE.shape_backtest(tables, sens, bt, n=2000, seed=3)
        st.session_state["shape_gate"] = PE.shape_metrics(sb)
if "shape_gate" in st.session_state:
    st.dataframe(st.session_state["shape_gate"].style.format({"engine": "{:.4f}", "normal": "{:.4f}"}, na_rep="—"),
                 hide_index=True, width="stretch")
    st.caption("Lower is better on every row. On a small sample the two will be close; the "
               "engine has to win on the full season before it replaces the curve anywhere.")

st.caption("Stage 1 tables · Stage 2 loop · Stage 3 team steering. Stage 4 (players on the "
           "depth chart) is what lets this engine drive fantasy and props. For research/entertainment.")
