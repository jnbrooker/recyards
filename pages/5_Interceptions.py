"""Interceptions simulator — INTs THROWN, viewed from the offense/QB side.
INT rate per attempt is regressed HARD toward the league mean (INT rate is one
of the noisiest stats in football), then nudged by the defense's ball-hawking
in log-odds. Same dashboard template as the other pages."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from nflsim import data as D, qb as Q

st.set_page_config(page_title="Interceptions Simulator", page_icon="🏈", layout="wide")


@st.cache_data(show_spinner="Downloading NFL data…")
def get_weekly(seasons):
    return D.load_weekly(tuple(sorted(seasons)))


@st.cache_data(show_spinner=False)
def get_derived(seasons):
    wk = get_weekly(seasons)
    qbs = D.list_players(wk[wk["position"] == "QB"], stat="attempts", min_vol=120)
    return qbs, D.list_defenses(wk), Q.league_pass_rates(wk), D.def_pass_rates(wk)


st.sidebar.header("Setup")
ALL_SEASONS = [2026, 2025, 2024, 2023, 2022]
seasons = st.sidebar.multiselect("Seasons used to build priors", ALL_SEASONS,
                                 default=[2025, 2024],
                                 help="Recent seasons are weighted more heavily.")
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

qbs, defenses, lg, def_profiles = get_derived(tuple(seasons))

player_label = st.sidebar.selectbox("Quarterback", qbs["label"].tolist(),
                                    help="QBs with 120+ attempts in the selected seasons.")
player_row = qbs[qbs["label"] == player_label].iloc[0]
opp = st.sidebar.selectbox("Opponent defense", defenses,
                           index=defenses.index("SF") if "SF" in defenses else 0)
line = st.sidebar.number_input("Interceptions line", 0.5, 4.5, 0.5, 1.0,
                               help="0.5 = at least one INT. 1.5 = two or more, etc.")
st.sidebar.divider()
use_def = st.sidebar.toggle("Adjust for opponent secondary", value=True)
shrink = st.sidebar.slider("Defense adjustment strength", 0.0, 1.0,
                           Q.DEFAULT_INT_DEF_SHRINK, 0.05, disabled=not use_def,
                           help="INT-allowed splits are very noisy, so this defaults low.")
n_sims = st.sidebar.select_slider("Simulations", [10000, 20000, 40000, 100000], value=40000)

pri = Q.qb_priors(wk, player_row["player_id"], lg)
dprof = def_profiles.get(opp) if use_def else None
sim = Q.simulate_ints(pri, dprof, lg, n_sims=n_sims, def_shrink=shrink, seed=7)
s = Q.summarize(sim, line)

st.title("🏈 Interceptions Simulator")
st.caption(f"**{pri['name']}** (QB, {pri['team']}) vs **{opp}** secondary — "
           f"INTs thrown, {pri['games']} games of history, {n_sims:,} simulations")

c1, c2, c3, c4 = st.columns(4)
c1.metric("At least one INT", f"{s['p_1plus']:.0%}",
          help=f"Fair odds  {D.american(s['p_1plus'])}")
c2.metric("Expected INTs", f"{s['mean']:.2f}")
c3.metric("2+ INTs", f"{s['p_2plus']:.0%}")
c4.metric("Mean attempts", f"{sim['attempts'].mean():.1f}")

if use_def and dprof is not None:
    st.info(f"**{opp} secondary:** generates INTs ×{dprof['r_int']:.2f} vs league "
            f"({dprof['int_rate_allowed']:.1%} per attempt vs {dprof['lg_int']:.1%}).  \n"
            f"QB base INT rate {pri['p_int']:.1%} (raw {pri['raw_int']:.1%}, "
            f"regressed hard) → game rate {sim['rate']:.1%} after the matchup.")
elif not use_def:
    st.info("Defense adjustment is **off** — baseline vs a league-average secondary.")

dist = s["dist"]
labels = list(dist.keys()); vals = [dist[k] for k in labels]
def _num(l): return int(str(l)[:-1]) if str(l).endswith("+") else int(l)
colors = ["#2e7d5b" if _num(l) > line else "#c0563b" for l in labels]
fig = go.Figure()
fig.add_bar(x=labels, y=vals, marker_color=colors,
            hovertemplate="%{x} INT<br>%{y:.1%} of games<extra></extra>")
fig.add_vline(x=line, line_width=2, line_dash="dash", line_color="#222")
fig.update_layout(title="Simulated interceptions thrown — probability of each outcome",
                  xaxis_title="Interceptions in the game", yaxis_title="Probability",
                  yaxis_tickformat=".0%", bargap=0.25, height=440,
                  margin=dict(t=70, b=40, l=60, r=20), showlegend=False)
st.plotly_chart(fig, width="stretch")

with st.expander("Chance of at least N interceptions (cumulative view)"):
    c = sim["count"]; xs = list(range(0, int(min(c.max(), 5)) + 1))
    p_at_least = [float((c >= x).mean()) for x in xs]
    cfig = go.Figure()
    cfig.add_bar(x=[str(x) for x in xs], y=p_at_least, marker_color="#2e7d5b",
                 hovertemplate="≥ %{x} INT<br>%{y:.1%}<extra></extra>")
    cfig.update_layout(xaxis_title="At least N interceptions", yaxis_title="Chance",
                       yaxis_tickformat=".0%", height=320,
                       margin=dict(t=20, b=40, l=60, r=20))
    st.plotly_chart(cfig, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("QB profile (inputs)")
    st.dataframe(pd.DataFrame({
        "Metric": ["Expected attempts", "INT rate / attempt", "Games"],
        "Value":  [f"{pri['mu_att']:.1f}", f"{pri['p_int']:.1%}", str(pri["games"])],
        "Raw (unregressed)": ["—", f"{pri['raw_int']:.1%}", "—"],
    }), hide_index=True, width="stretch")
    st.caption("INT rate is heavily regressed toward the league mean — one of the "
               "noisiest stats in football — then combined with the defense in log-odds.")
with right:
    st.subheader("Outcome probabilities")
    st.dataframe(pd.DataFrame({
        "Outcome": [f"{k} INT" for k in dist.keys()],
        "Chance": [f"{v:.1%}" for v in dist.values()],
    }), hide_index=True, width="stretch")
    st.metric(f"Fair odds — {'anytime' if line == 0.5 else f'over {line:g}'}",
              f"Over {s['fair_over_odds']}  /  Under {s['fair_under_odds']}")

st.caption("INTs are viewed from the offense/QB side (INTs thrown per attempt), a "
           "turnover output rather than a defender stat. For research/entertainment.")
