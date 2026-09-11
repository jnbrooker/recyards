"""Fantasy Projections — every player in a week's slate, scored per simulation.

Runs the game engine for every game on the schedule that week, applies the
league's scoring rules to each simulation, and shows the projection with its
floor and ceiling and a breakdown of where the points come from.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from nflsim import data as D, game as G, fantasy as F
from nflsim import ui as UI

st.set_page_config(page_title="Fantasy Projections", page_icon="🏈", layout="wide")


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Loading play-by-play, depth charts and injuries…")
def get_context(seasons):
    return G.prepare(tuple(sorted(seasons)))


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner=False)
def get_schedule(season):
    return D.load_schedule((int(season),))


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Simulating every game this week…")
def get_week(seasons, week, rules_items, n_sims, use_injuries):
    ctx = get_context(seasons)
    sched = get_schedule(ctx["depth_seasons"][-1])
    games = sched[sched["week"] == int(week)]
    return F.week_projections(ctx, games, dict(rules_items), n_sims=n_sims,
                              use_injuries=use_injuries)


# --- sidebar ----------------------------------------------------------------
st.sidebar.header("Setup")
seasons = UI.season_picker("Seasons used to build priors")
ctx = get_context(tuple(seasons))
sched = get_schedule(ctx["depth_seasons"][-1])
if sched.empty:
    st.error("The schedule did not load."); st.stop()

st.sidebar.divider()
weeks = sorted(sched["week"].unique().tolist())
cur = D.current_week(sched)
week = st.sidebar.selectbox("Week", weeks, index=weeks.index(cur) if cur in weeks else 0)

preset = st.sidebar.selectbox("Scoring", list(F.PRESETS.keys()), index=0)
rules = dict(F.PRESETS[preset])
with st.sidebar.expander("Edit scoring rules"):
    for key, label in F.COMPONENT_LABELS.items():
        rules[key] = st.number_input(f"{label} (pts per unit)", value=float(rules[key]),
                                     step=0.05 if "yd" in key else 0.5, format="%.2f",
                                     key=f"rule_{key}")
use_inj = st.sidebar.toggle("Drop players ruled out", value=True)
n_sims = st.sidebar.select_slider("Simulations per game", [4000, 10000, 20000], value=10000)

st.sidebar.divider()
positions = st.sidebar.multiselect("Positions", ["QB", "RB", "WR", "TE", "FB"],
                                   default=["QB", "RB", "WR", "TE"])
team_filter = st.sidebar.multiselect("Teams (blank = all)", sorted(ctx["ratings"]["off"].index))
min_proj = st.sidebar.slider("Hide players projected under", 0.0, 10.0, 2.0, 0.5)

# --- run --------------------------------------------------------------------
table, samples, summaries = get_week(tuple(seasons), int(week),
                                     tuple(sorted(rules.items())), int(n_sims), use_inj)

st.title("🏈 Fantasy Projections")
ok = [g for g in summaries if g["ok"]]
skipped = [g for g in summaries if not g["ok"]]
st.caption(f"Week {week} · {len(ok)} games simulated {n_sims:,} times each · "
           f"{preset} scoring · depth charts from {ctx['depth_seasons'][-1]}")
if skipped:
    st.warning("Skipped: " + ", ".join(f"{g['away']} @ {g['home']} ({g['error']})" for g in skipped))
if table.empty:
    st.info("No games to simulate this week."); st.stop()

view = table[table["Pos"].isin(positions)]
if team_filter:
    view = view[view["Team"].isin(team_filter)]
view = view[view["Proj"] >= min_proj].reset_index(drop=True)

# --- the board --------------------------------------------------------------
st.subheader("Projection board")
st.caption("**Proj** is the mean of every simulation; **Floor / Ceiling** are the 10th "
           "and 90th percentiles; **P(20+)** is the chance of a 20-point game. The "
           "points columns show where the projection comes from and always add up to it.")
pts_cols = {f"pts_{k}": v for k, v in F.COMPONENT_LABELS.items()}
show = view[["Player", "Team", "Pos", "Opp", "Proj", "Floor", "Ceiling", "P20"]
            + list(pts_cols.keys()) + ["Source"]].rename(columns=pts_cols)
fmt = {"Proj": "{:.1f}", "Floor": "{:.1f}", "Ceiling": "{:.1f}", "P20": "{:.0%}"}
fmt.update({v: "{:.1f}" for v in pts_cols.values()})
st.dataframe(show.style.format(fmt), hide_index=True, width="stretch", height=560)

# --- top of the board, stacked by component -----------------------------------
st.subheader("Where the top projections come from")
top_n = st.slider("Players shown", 10, 40, 20, 5)
top = view.head(top_n).iloc[::-1]
fig = go.Figure()
palette = {"rec": "#2e7d5b", "rec_yd": "#5aa87f", "rec_td": "#9ad1b0",
           "rush_yd": "#c0563b", "rush_td": "#e08a72",
           "pass_yd": "#3a6ea5", "pass_td": "#7fa6d1", "int": "#888"}
for key, label in F.COMPONENT_LABELS.items():
    col = f"pts_{key}"
    if (top[col].abs() > 0.05).any():
        fig.add_bar(name=label, y=top["Player"] + " (" + top["Pos"] + ")", x=top[col],
                    orientation="h", marker_color=palette[key],
                    hovertemplate=label + ": %{x:.1f}<extra></extra>")
fig.update_layout(barmode="relative", height=max(420, 24 * top_n + 80),
                  xaxis_title="Projected fantasy points", legend_title="",
                  margin=dict(t=20, b=40, l=10, r=20))
st.plotly_chart(fig, width="stretch")

# --- one player in depth ------------------------------------------------------
st.subheader("One player, in depth")
labels = (view["Player"] + " — " + view["Team"] + " " + view["Pos"] + " vs " + view["Opp"]).tolist()
pick = st.selectbox("Player", labels, index=0)
row = view.iloc[labels.index(pick)]
p = samples.get(row["player_id"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Projection", f"{row['Proj']:.1f}")
c2.metric("Floor (p10)", f"{row['Floor']:.1f}")
c3.metric("Ceiling (p90)", f"{row['Ceiling']:.1f}")
c4.metric("20+ points", f"{row['P20']:.0%}")

left, right = st.columns([1, 1])
with left:
    st.markdown("**How it got there**")
    bd = F.breakdown_table(row)
    bd.loc[len(bd)] = ["Total", np.nan, float(row["Proj"])]
    st.dataframe(bd.style.format({"Projected": "{:.1f}", "Points": "{:.1f}"}, na_rep=""),
                 hide_index=True, width="stretch")
    st.caption(f"{row['Source']} · {row['Team']} {row['Pos']}{int(row['Depth'])} vs {row['Opp']}")
with right:
    if p is not None:
        hi = float(np.percentile(p, 99.5))
        edges = np.arange(0, hi + 2, 1.0)
        cnt, _ = np.histogram(np.clip(p, 0, hi), bins=edges)
        hfig = go.Figure()
        hfig.add_bar(x=(edges[:-1] + edges[1:]) / 2, y=cnt / cnt.sum(), marker_color="#2e7d5b",
                     hovertemplate="%{x:.0f} pts<br>%{y:.1%} of games<extra></extra>")
        hfig.add_vline(x=row["Proj"], line_dash="dash", line_color="#222")
        hfig.update_layout(height=320, bargap=0.05, xaxis_title="Fantasy points",
                           yaxis_title="Share of simulations", yaxis_tickformat=".0%",
                           margin=dict(t=20, b=40, l=60, r=20))
        st.plotly_chart(hfig, width="stretch")
        st.caption(f"Chance of at least: 10 pts {float((p >= 10).mean()):.0%} · "
                   f"15 pts {float((p >= 15).mean()):.0%} · 25 pts {float((p >= 25).mean()):.0%}")

# --- head-to-head -------------------------------------------------------------
st.subheader("Head-to-head: two lineups, who wins?")
st.caption("Pick the starters on each side from this week's slate. Totals are summed "
           "**per simulation**, so a QB stacked with his own receiver, or two players "
           "in the same game, keep the correlation the engine gave them — the win "
           "chance is not just a comparison of two projections. Players on bye are "
           "not listed. Kickers and defenses are not modelled: add them as a constant.")

pool = table[table["Pos"].isin(["QB", "RB", "WR", "TE", "FB"])].reset_index(drop=True)
pool_labels = (pool["Player"] + " — " + pool["Team"] + " " + pool["Pos"]
               + " (" + pool["Proj"].map("{:.1f}".format) + ")").tolist()
label_to_id = dict(zip(pool_labels, pool["player_id"]))

ca, cb = st.columns(2)
with ca:
    name_a = st.text_input("Team A name", "Team A")
    picks_a = st.multiselect("Team A starters", pool_labels, key="h2h_a")
    extra_a = st.number_input("Team A other points (K, DST, …)", value=0.0, step=0.5, key="h2h_xa")
with cb:
    name_b = st.text_input("Team B name", "Team B")
    picks_b = st.multiselect("Team B starters", pool_labels, key="h2h_b")
    extra_b = st.number_input("Team B other points (K, DST, …)", value=0.0, step=0.5, key="h2h_xb")

ids_a = [label_to_id[l] for l in picks_a]
ids_b = [label_to_id[l] for l in picks_b]
dup = set(ids_a) & set(ids_b)
if dup:
    st.warning("The same player is on both sides: " +
               ", ".join(pool.set_index("player_id").loc[list(dup), "Player"].tolist()))

h2h = F.matchup(samples, ids_a, ids_b, extra_a, extra_b) if ids_a and ids_b else None
if h2h is None:
    st.info("Pick at least one player on each side.")
else:
    fav_name = name_a if h2h["win_a"] >= h2h["win_b"] else name_b
    fav_p = max(h2h["win_a"], h2h["win_b"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{name_a} projected", f"{h2h['mean_a']:.1f}",
              help=f"80% range {h2h['p10_a']:.0f}–{h2h['p90_a']:.0f}")
    m2.metric(f"{name_b} projected", f"{h2h['mean_b']:.1f}",
              help=f"80% range {h2h['p10_b']:.0f}–{h2h['p90_b']:.0f}")
    m3.metric(f"{fav_name} wins", f"{fav_p:.0%}", help=f"Fair odds {D.american(fav_p)}")
    m4.metric("Typical margin", f"{h2h['mean_margin']:+.1f}",
              help=f"sd {h2h['margin_sd']:.1f} · 80% of weeks between "
                   f"{h2h['p10_margin']:+.0f} and {h2h['p90_margin']:+.0f}")

    mg = h2h["margin"]
    edges = np.arange(np.floor(mg.min()) - 0.5, np.ceil(mg.max()) + 1.5, 2.0)
    cnt, _ = np.histogram(mg, bins=edges)
    ctr = (edges[:-1] + edges[1:]) / 2
    mfig = go.Figure()
    mfig.add_bar(x=ctr, y=cnt / cnt.sum(),
                 marker_color=["#2e7d5b" if c > 0 else "#c0563b" for c in ctr],
                 hovertemplate="margin %{x:+.0f}<br>%{y:.1%} of simulations<extra></extra>")
    mfig.add_vline(x=0, line_width=2, line_dash="dash", line_color="#222")
    mfig.update_layout(height=340, bargap=0.05, showlegend=False,
                       xaxis_title=f"{name_a} minus {name_b} (fantasy points)",
                       yaxis_title="Share of simulations", yaxis_tickformat=".1%",
                       margin=dict(t=20, b=45, l=60, r=20))
    st.plotly_chart(mfig, width="stretch")

    ta, tb = st.columns(2)
    with ta:
        st.markdown(f"**{name_a}**")
        st.dataframe(F.lineup_table(table, ids_a).style.format(
            {"Proj": "{:.1f}", "Floor": "{:.1f}", "Ceiling": "{:.1f}"}),
            hide_index=True, width="stretch")
    with tb:
        st.markdown(f"**{name_b}**")
        st.dataframe(F.lineup_table(table, ids_b).style.format(
            {"Proj": "{:.1f}", "Floor": "{:.1f}", "Ceiling": "{:.1f}"}),
            hide_index=True, width="stretch")

with st.expander("This week's games as the model sees them"):
    g = pd.DataFrame(ok)
    if not g.empty:
        g["Model"] = g.apply(lambda r: f"{r['away']} {r['away_pts']:.1f} @ {r['home']} {r['home_pts']:.1f}", axis=1)
        g["Home win"] = g["win_home"]
        g["Market spread (home)"] = g["spread_line"]
        g["Market total"] = g["total_line"]
        st.dataframe(g[["label", "Model", "Home win", "total", "Market spread (home)", "Market total"]]
                     .rename(columns={"label": "Game", "total": "Model total"})
                     .style.format({"Home win": "{:.0%}", "Model total": "{:.1f}",
                                    "Market spread (home)": "{:+.1f}", "Market total": "{:.1f}"}, na_rep="—"),
                     hide_index=True, width="stretch")
        st.caption("Market lines are shown for comparison only — the model never reads them.")

st.caption("Not scored: fumbles, two-point conversions and return yards (not modelled; "
           "small and very noisy). For research/entertainment.")
