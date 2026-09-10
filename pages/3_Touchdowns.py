"""Touchdowns simulator page — total TDs combining rushing AND receiving.
Same template as the yardage pages: line check (default 0.5 = anytime TD),
fair odds, split distribution chart, cumulative view, inputs & outcome tables."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from nflsim import data as D, touchdowns as T

st.set_page_config(page_title="Touchdowns Simulator", page_icon="🏈", layout="wide")


@st.cache_data(show_spinner="Downloading NFL data…")
def get_weekly(seasons):
    return D.load_weekly(tuple(sorted(seasons)))


@st.cache_data(show_spinner=False)
def get_derived(seasons):
    wk = get_weekly(seasons)
    g = (wk.groupby(["player_id", "player_display_name", "position"], as_index=False)
           .agg(rec=("receptions", "sum"), car=("carries", "sum"),
                team=("recent_team", "last"), games=("week", "count")))
    g["opp"] = g["rec"] + g["car"]
    g = g[g["opp"] >= 30].sort_values("opp", ascending=False)
    g["label"] = g["player_display_name"] + " (" + g["position"] + ", " + g["team"] + ")"
    return (g.reset_index(drop=True), D.list_defenses(wk),
            T.league_td_rates(wk), T.td_defense_profiles(wk))


st.sidebar.header("Setup")
ALL_SEASONS = [2026, 2025, 2024, 2023, 2022]
seasons = st.sidebar.multiselect(
    "Seasons used to build priors", ALL_SEASONS, default=[2025, 2024],
    help="More seasons = steadier estimates; fewer = more current form.")
if not seasons:
    st.sidebar.error("Pick at least one season."); st.stop()

try:
    wk = get_weekly(tuple(seasons))
except ValueError as e:
    st.error(str(e)); st.stop()

loaded = sorted(int(s) for s in wk["season"].unique())
skipped = [s for s in seasons if s not in loaded]
if skipped:
    st.sidebar.warning(f"No data yet for {', '.join(map(str, skipped))} — "
                       f"using {', '.join(map(str, loaded))}.")

players, defenses, lg, def_profiles = get_derived(tuple(seasons))

player_label = st.sidebar.selectbox(
    "Player", players["label"].tolist(),
    help="Players with 30+ combined carries and catches in the selected seasons.")
player_row = players[players["label"] == player_label].iloc[0]

opp = st.sidebar.selectbox("Opponent defense", defenses,
                           index=defenses.index("SF") if "SF" in defenses else 0)
line = st.sidebar.number_input("TD line", 0.5, 3.5, 0.5, 1.0,
                               help="0.5 = anytime TD. 1.5 = two or more, etc.")

st.sidebar.divider()
use_def = st.sidebar.toggle("Adjust for opponent defense", value=True)
shrink = st.sidebar.slider(
    "Defense adjustment strength", 0.0, 1.0, T.DEFAULT_TD_DEF_SHRINK, 0.05,
    disabled=not use_def,
    help="TD-allowed splits are very noisy, so this defaults low (heavily shrunk).")
n_sims = st.sidebar.select_slider("Simulations", [10000, 20000, 40000, 100000], value=40000)

pri = T.player_td_priors(wk, player_row["player_id"], lg)
pos = pri["position"]
dprof = def_profiles.get((opp, pos)) if use_def else None
sim = T.simulate(pri, dprof, n_sims=n_sims, def_shrink=shrink, seed=7)
s = T.summarize(sim, line)

st.title("🏈 Touchdowns Simulator")
st.caption(f"**{pri['name']}** ({pos}, {pri['team']}) vs **{opp}** defense — "
           f"rushing + receiving, {pri['games']} games of history, {n_sims:,} simulations")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Anytime TD", f"{s['p_anytime']:.0%}",
          help=f"Fair odds  {D.american(s['p_anytime'])}")
c2.metric("Expected TDs", f"{s['mean']:.2f}")
c3.metric("2+ TDs", f"{s['p_2plus']:.0%}")
c4.metric("Rush / Rec split", f"{s['exp_rush']:.2f} / {s['exp_rec']:.2f}")

if use_def and dprof is not None:
    st.info(f"**{opp} vs {pos}s:** allows rushing TDs ×{dprof['r_rush_td']:.2f} and "
            f"receiving TDs ×{dprof['r_rec_td']:.2f} vs league (before shrink).  "
            f"Applied → rush ×{sim['adj']['m_rush']:.2f}, rec ×{sim['adj']['m_rec']:.2f}.")
elif not use_def:
    st.info("Defense adjustment is **off** — baseline vs a league-average defense.")

# Distribution chart (discrete TD counts, split at the line)
dist = s["dist"]
labels = list(dist.keys())
vals = [dist[k] for k in labels]
def _num(lbl):
    s = str(lbl)
    return int(s[:-1]) if s.endswith("+") else int(s)
colors = ["#2e7d5b" if _num(lbl) > line else "#c0563b" for lbl in labels]

fig = go.Figure()
fig.add_bar(x=labels, y=vals, marker_color=colors,
            hovertemplate="%{x} TD<br>%{y:.1%} of games<extra></extra>")
fig.update_layout(
    title="Simulated total touchdowns — probability of each outcome",
    xaxis_title="Touchdowns in the game (rushing + receiving)",
    yaxis_title="Probability", yaxis_tickformat=".0%", bargap=0.25, height=440,
    margin=dict(t=70, b=40, l=60, r=20), showlegend=False)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Chance of scoring at least N touchdowns (cumulative view)"):
    t = sim["total"]
    xs = list(range(0, int(min(t.max(), 4)) + 1))
    p_at_least = [float((t >= x).mean()) for x in xs]
    cfig = go.Figure()
    cfig.add_bar(x=[str(x) for x in xs], y=p_at_least, marker_color="#2e7d5b",
                 hovertemplate="≥ %{x} TD<br>%{y:.1%}<extra></extra>")
    cfig.update_layout(xaxis_title="At least N touchdowns", yaxis_title="Chance",
                       yaxis_tickformat=".0%", height=320,
                       margin=dict(t=20, b=40, l=60, r=20))
    st.plotly_chart(cfig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Player profile (inputs)")
    st.dataframe(pd.DataFrame({
        "Metric": ["Expected receptions", "Rec TD / reception", "Expected carries",
                   "Rush TD / carry", "Games"],
        "Value":  [f"{pri['mu_rec']:.1f}", f"{pri['p_rec_td']:.1%}",
                   f"{pri['mu_car']:.1f}", f"{pri['p_rush_td']:.1%}", str(pri["games"])],
        "Raw (unregressed)": ["—", f"{pri['raw_rec_td_per_rec']:.1%}", "—",
                              f"{pri['raw_rush_td_per_car']:.1%}", "—"],
    }), hide_index=True, use_container_width=True)
    st.caption("TD rates are regressed toward the positional league average.")
with right:
    st.subheader("Outcome probabilities")
    st.dataframe(pd.DataFrame({
        "Outcome": [f"{k} TD" for k in dist.keys()],
        "Chance": [f"{v:.1%}" for v in dist.values()],
    }), hide_index=True, use_container_width=True)
    st.metric(f"Fair odds — {'anytime' if line == 0.5 else f'over {line:g}'}",
              f"Over {s['fair_over_odds']}  /  Under {s['fair_under_odds']}")

st.caption("Total TDs = Binomial(receptions, rec-TD rate) + Binomial(carries, "
           "rush-TD rate), so scoring scales with volume and both phases count. "
           "For research/entertainment.")
