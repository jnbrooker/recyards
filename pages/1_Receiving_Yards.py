"""Receiving Yards simulator page — the original receiving dashboard, unchanged,
now living inside the multi-page app. Reuses the original model.py at the repo root."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import model

st.set_page_config(page_title="Receiving Yards Simulator", page_icon="🏈", layout="wide")


@st.cache_data(show_spinner="Downloading NFL data…")
def get_weekly(seasons):
    return model.load_weekly(tuple(sorted(seasons)))


@st.cache_data(show_spinner=False)
def get_derived(seasons):
    wk = get_weekly(seasons)
    return (model.list_players(wk), model.list_defenses(wk),
            model.team_pass_volume(wk), model.defense_profiles(wk))


st.sidebar.header("Setup")
ALL_SEASONS = [2026, 2025, 2024, 2023, 2022]
seasons = st.sidebar.multiselect(
    "Seasons used to build priors", ALL_SEASONS, default=[2025, 2024],
    help="More seasons = steadier estimates; fewer = more current form. "
         "Recent seasons are weighted more heavily either way.")
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

players, defenses, team_vol, def_profiles = get_derived(tuple(seasons))

player_label = st.sidebar.selectbox(
    "Player", players["label"].tolist(),
    help="Only players with 20+ targets in the selected seasons are listed.")
player_row = players[players["label"] == player_label].iloc[0]

opp = st.sidebar.selectbox("Opponent defense", defenses,
                           index=defenses.index("SF") if "SF" in defenses else 0)
line = st.sidebar.number_input("Prop line (receiving yards)", 0.0, 250.0, 59.5, 0.5)

st.sidebar.divider()
use_def = st.sidebar.toggle("Adjust for opponent defense", value=True)
shrink = st.sidebar.slider(
    "Defense adjustment strength", 0.0, 1.0, model.DEFAULT_DEF_SHRINK, 0.05,
    disabled=not use_def,
    help="How much to trust one season of defense-vs-position splits. "
         "0 = ignore, 1 = take at face value.")
n_sims = st.sidebar.select_slider("Simulations", [5000, 10000, 20000, 50000, 100000],
                                  value=20000)

pri = model.player_priors(wk, player_row["player_id"])
pos = pri["position"]
tv = team_vol.get(pri["team"], team_vol["_LEAGUE_"])
dprof = def_profiles.get((opp, pos)) if use_def else None

sim = model.simulate(pri, tv, dprof, n_sims=n_sims, def_shrink=shrink, seed=7)
s = model.summarize(sim, line)

st.title("🏈 Receiving Yards Simulator")
st.caption(f"**{pri['name']}** ({pos}, {pri['team']}) vs **{opp}** defense — "
           f"{pri['games']} games of history, {n_sims:,} simulations")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Chance over {line:g}", f"{s['p_over']:.0%}",
          help=f"Fair odds  Over {s['fair_over_odds']} / Under {s['fair_under_odds']}")
c2.metric("Projected yards (mean)", f"{s['mean']:.0f}")
c3.metric("Most likely (median)", f"{s['median']:.0f}")
c4.metric("Catches / targets", f"{s['mean_receptions']:.1f} / {s['mean_targets']:.1f}")

if use_def and dprof is not None:
    st.info(f"**{opp} defense vs {pos}s:** {model.scheme_label(dprof['r_adot'], dprof['r_ypt'])}.  "
            f"Allows {dprof['ypg_allowed']:.0f} yds/game "
            f"(league avg {dprof['lg_ypg']:.0f}), "
            f"aDOT {dprof['adot_allowed']:.1f} vs league {dprof['lg_adot']:.1f}.  "
            f"Adjustments applied → depth ×{sim['adj']['m_adot']:.2f}, "
            f"catch rate ×{sim['adj']['m_catch']:.2f}, "
            f"efficiency ×{sim['adj']['m_yac_env']:.2f}.")
elif not use_def:
    st.info("Defense adjustment is **off** — this is the player's baseline "
            "distribution against a league-average defense.")

# Distribution chart
y = sim["yards"]
cutoff = np.percentile(y, 99.5)
yv = y[y <= cutoff]
nbins = 60
counts, edges = np.histogram(yv, bins=nbins)
centers = (edges[:-1] + edges[1:]) / 2
probs = counts / len(y)
over_mask = centers > line

fig = go.Figure()
fig.add_bar(x=centers[~over_mask], y=probs[~over_mask], name=f"Under {line:g}",
            marker_color="#c0563b", width=(edges[1]-edges[0])*0.95,
            hovertemplate="%{x:.0f} yds<br>%{y:.1%} of games<extra></extra>")
fig.add_bar(x=centers[over_mask], y=probs[over_mask], name=f"Over {line:g}",
            marker_color="#2e7d5b", width=(edges[1]-edges[0])*0.95,
            hovertemplate="%{x:.0f} yds<br>%{y:.1%} of games<extra></extra>")
fig.add_vline(x=line, line_width=2, line_dash="dash", line_color="#222",
              annotation_text=f"Line {line:g}", annotation_position="top")
fig.add_vline(x=s["mean"], line_width=1.5, line_dash="dot", line_color="#1f6feb",
              annotation_text="mean", annotation_position="top right")
fig.update_layout(
    title="Simulated receiving yards — probability of each outcome",
    xaxis_title="Receiving yards in the game", yaxis_title="Probability",
    yaxis_tickformat=".1%", bargap=0.02, height=440,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(t=70, b=40, l=60, r=20))
st.plotly_chart(fig, use_container_width=True)

with st.expander("Chance of clearing any line (cumulative view)"):
    xs = np.arange(0, int(cutoff) + 5, 5)
    p_at_least = [float((y >= x).mean()) for x in xs]
    cfig = go.Figure()
    cfig.add_scatter(x=xs, y=p_at_least, mode="lines", line=dict(color="#2e7d5b", width=3),
                     hovertemplate="≥ %{x} yds<br>%{y:.1%}<extra></extra>")
    cfig.add_vline(x=line, line_dash="dash", line_color="#222",
                   annotation_text=f"Line {line:g}")
    cfig.update_layout(xaxis_title="Yards line", yaxis_title="Chance of going over",
                       yaxis_tickformat=".0%", height=340,
                       margin=dict(t=20, b=40, l=60, r=20))
    st.plotly_chart(cfig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Player profile (inputs)")
    st.dataframe(pd.DataFrame({
        "Metric": ["Target share", "Catch %", "aDOT (depth)", "YAC / catch", "Games"],
        "Mean":   [f"{pri['mu_ts']:.1%}", f"{pri['mu_catch']:.1%}",
                   f"{pri['mu_adot']:.1f}", f"{pri['yac_per_rec']:.1f}", str(pri["games"])],
        "Std (variance)": [f"±{pri['sd_ts']:.1%}", f"±{pri['sd_catch']:.1%}",
                           f"±{pri['sd_adot']:.1f}", "—", "—"],
    }), hide_index=True, use_container_width=True)
with right:
    st.subheader("Outcome percentiles")
    st.dataframe(pd.DataFrame({
        "Percentile": ["10th (floor)", "25th", "Median", "75th", "90th (ceiling)"],
        "Yards": [f"{s['p10']:.0f}", f"{s['p25']:.0f}", f"{s['median']:.0f}",
                  f"{s['p75']:.0f}", f"{s['p90']:.0f}"],
    }), hide_index=True, use_container_width=True)
    st.metric("Fair prop odds",
              f"Over {s['fair_over_odds']}  /  Under {s['fair_under_odds']}")

st.caption("Model is for research/entertainment. Priors come from nflverse "
           "regular-season data; defense adjustments use season-level "
           "splits and are deliberately shrunk toward league average.")
