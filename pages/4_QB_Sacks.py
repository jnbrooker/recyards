"""QB Sacks simulator — sacks TAKEN, viewed from the offense/QB side.
Sack rate per dropback combines the QB's own rate with the defense's pass rush
(log-odds) and is scaled by NGS average time to throw. Same dashboard template
as the yardage and TD pages."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from nflsim import data as D, qb as Q
from nflsim import roster as RO, ui as UI

st.set_page_config(page_title="QB Sacks Simulator", page_icon="🏈", layout="wide")


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Downloading NFL data…")
def get_weekly(seasons, recency):
    return D.load_weekly(tuple(sorted(seasons)), recency)


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Loading Next Gen Stats…")
def get_ttt(seasons, recency):
    return D.ngs_time_to_throw(D.load_ngs_pass(tuple(sorted(seasons))), recency)


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner=False)
def get_derived(seasons, recency):
    wk = get_weekly(seasons, recency)
    qbs = D.list_players(wk[wk["position"] == "QB"], stat="attempts", min_vol=120)
    return qbs, D.list_defenses(wk), Q.league_pass_rates(wk), D.def_pass_rates(wk)


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Loading live depth charts and injury reports…")
def get_rosters(seasons, recency, use_injuries):
    return UI.cached_rosters(tuple(sorted(seasons)), use_injuries, recency)


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner=False)
def get_live_team_vol(seasons, recency):
    return UI.cached_live(tuple(sorted(seasons)), recency)["team_vol"]


st.sidebar.header("Setup")
seasons, recency = UI.priors_picker("Seasons used to build priors")
try:
    wk = get_weekly(tuple(seasons), recency)
except ValueError as e:
    st.error(str(e)); st.stop()

loaded = sorted(int(s) for s in wk["season"].unique())
skipped = [s for s in seasons if s not in loaded]
if skipped:
    st.sidebar.warning(f"No data yet for {', '.join(map(str, skipped))} — "
                       f"using {', '.join(map(str, loaded))}.")
UI.recency_caption(wk, recency)

qbs, defenses, lg, def_profiles = get_derived(tuple(seasons), recency)
ttt = get_ttt(tuple(seasons), recency)

use_live = st.sidebar.toggle(
    "Pick from live depth charts", value=True,
    help="Current-season depth chart and injury report. The player's usage share is "
         "redistributed when teammates are ruled out, and team volume follows his "
         "CURRENT team, not the one in his history.")
live_row = None
if use_live:
    use_inj = st.sidebar.toggle("Drop players ruled out", value=True)
    rosters = get_rosters(tuple(seasons), recency, use_inj)
    if rosters.empty:
        st.sidebar.error("Depth charts did not load — using history only."); use_live = False
    else:
        live_row = UI.pick_player(rosters, ['QB'], key='qb')
        player_id = live_row["player_id"]
if not use_live:
    player_label = st.sidebar.selectbox("Quarterback", qbs["label"].tolist(),
                                        help="QBs with 120+ attempts in the selected seasons.")
    player_row = qbs[qbs["label"] == player_label].iloc[0]
    player_id = player_row["player_id"]

opp = st.sidebar.selectbox("Opponent defense", defenses,
                           index=defenses.index("SF") if "SF" in defenses else 0)
line = st.sidebar.number_input("Sacks line", 0.5, 8.5, 2.5, 1.0,
                               help="Prop line for sacks taken. 2.5 = three or more.")
st.sidebar.divider()
use_def = st.sidebar.toggle("Adjust for opponent pass rush", value=True)
shrink = st.sidebar.slider("Defense adjustment strength", 0.0, 1.0,
                           Q.DEFAULT_SACK_DEF_SHRINK, 0.05, disabled=not use_def,
                           help="How much to trust one defense's sack-rate splits.")
n_sims = st.sidebar.select_slider("Simulations", [10000, 20000, 40000, 100000], value=40000)

try:
    pri = Q.qb_priors(wk, player_id, lg, ttt)
except ValueError:
    if live_row is None:
        raise
    st.error(f"{live_row['name']} has no passing history in the selected seasons."); st.stop()
if live_row is not None:
    pri["team"] = live_row["team"]
dprof = def_profiles.get(opp) if use_def else None
sim = Q.simulate_sacks(pri, dprof, lg, n_sims=n_sims, def_shrink=shrink, seed=7)
s = Q.summarize(sim, line)

st.title("🏈 QB Sacks Simulator")
if live_row is not None:
    UI.status_warning(live_row)
    st.caption(f"**{live_row['team']} QB{int(live_row['depth'])}**" + (f" · history is from **{live_row['prev_team']}**" if live_row.get("prev_team") and live_row["prev_team"] != live_row["team"] else ""))
st.caption(f"**{pri['name']}** (QB, {pri['team']}) vs **{opp}** pass rush — "
           f"sacks taken, {pri['games']} games of history, {n_sims:,} simulations")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Chance over {line:g}", f"{s['p_over']:.0%}",
          help=f"Fair odds  Over {s['fair_over_odds']} / Under {s['fair_under_odds']}")
c2.metric("Expected sacks", f"{s['mean']:.2f}")
c3.metric("Sacked at least once", f"{s['p_1plus']:.0%}")
c4.metric("Mean dropbacks", f"{sim['dropbacks'].mean():.1f}")

ttt_txt = (f" · NGS time to throw {pri['ttt_val']:.2f}s (×{pri['ttt_mult']:.2f} on sack rate)"
           if pri["ttt_val"] is not None else " · NGS time-to-throw unavailable (neutral)")
if use_def and dprof is not None:
    st.info(f"**{opp} pass rush:** generates sacks ×{dprof['r_sack']:.2f} vs league "
            f"({dprof['sack_rate_allowed']:.1%} per dropback vs {dprof['lg_sack']:.1%}).  \n"
            f"QB base sack rate {pri['p_sack']:.1%} → game rate {sim['rate']:.1%} after "
            f"pass rush + release.{ttt_txt}.")
elif not use_def:
    st.info(f"Defense adjustment is **off** — baseline vs a league-average pass rush."
            f"{ttt_txt}.")

# Discrete distribution chart
dist = s["dist"]
labels = list(dist.keys()); vals = [dist[k] for k in labels]
def _num(l): return int(str(l)[:-1]) if str(l).endswith("+") else int(l)
colors = ["#2e7d5b" if _num(l) > line else "#c0563b" for l in labels]
fig = go.Figure()
fig.add_bar(x=labels, y=vals, marker_color=colors,
            hovertemplate="%{x} sacks<br>%{y:.1%} of games<extra></extra>")
fig.add_vline(x=line, line_width=2, line_dash="dash", line_color="#222")
fig.update_layout(title="Simulated sacks taken — probability of each outcome",
                  xaxis_title="Sacks in the game", yaxis_title="Probability",
                  yaxis_tickformat=".0%", bargap=0.25, height=440,
                  margin=dict(t=70, b=40, l=60, r=20), showlegend=False)
st.plotly_chart(fig, width="stretch")

with st.expander("Chance of at least N sacks (cumulative view)"):
    c = sim["count"]; xs = list(range(0, int(min(c.max(), 6)) + 1))
    p_at_least = [float((c >= x).mean()) for x in xs]
    cfig = go.Figure()
    cfig.add_bar(x=[str(x) for x in xs], y=p_at_least, marker_color="#2e7d5b",
                 hovertemplate="≥ %{x} sacks<br>%{y:.1%}<extra></extra>")
    cfig.update_layout(xaxis_title="At least N sacks", yaxis_title="Chance",
                       yaxis_tickformat=".0%", height=320,
                       margin=dict(t=20, b=40, l=60, r=20))
    st.plotly_chart(cfig, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("QB profile (inputs)")
    st.dataframe(pd.DataFrame({
        "Metric": ["Expected dropbacks", "Sack rate / dropback", "Time to throw",
                   "Games"],
        "Value":  [f"{pri['mu_db']:.1f}", f"{pri['p_sack']:.1%}",
                   f"{pri['ttt_val']:.2f}s" if pri['ttt_val'] is not None else "—",
                   str(pri["games"])],
        "Raw (unregressed)": ["—", f"{pri['raw_sack']:.1%}", "—", "—"],
    }), hide_index=True, width="stretch")
    st.caption("Sack rate is regressed toward the league mean, then combined with "
               "the opponent's pass rush in log-odds and scaled by time to throw.")
with right:
    st.subheader("Outcome probabilities")
    st.dataframe(pd.DataFrame({
        "Outcome": [f"{k} sacks" for k in dist.keys()],
        "Chance": [f"{v:.1%}" for v in dist.values()],
    }), hide_index=True, width="stretch")
    st.metric("Fair prop odds",
              f"Over {s['fair_over_odds']}  /  Under {s['fair_under_odds']}")

st.caption("Sacks are viewed from the offense/QB side (sacks taken per dropback), "
           "so the model reads clean offensive feeds, not defensive box scores. "
           "For research/entertainment.")
