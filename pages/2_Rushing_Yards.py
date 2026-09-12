"""Rushing Yards simulator page — the same dashboard as the receiving page,
feature-for-feature, applied to the ground game. Same line-checking, fair odds,
split distribution chart, cumulative curve, inputs & percentile tables.

Run defense is modelled on six factors: front (yards before contact) and
tackling (yards after contact) and broken tackles allowed from PFR, plus stuff
rate, explosive-run rate and overall efficiency (success rate / EPA) allowed
from play-by-play."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from nflsim import data as D, rushing as R
from nflsim import roster as RO, ui as UI

st.set_page_config(page_title="Rushing Yards Simulator", page_icon="🏈", layout="wide")


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Downloading NFL data…")
def get_weekly(seasons, recency):
    return D.load_weekly(tuple(sorted(seasons)), recency)


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Loading advanced rushing data…")
def get_pfr(seasons, recency):
    pfr = D.load_pfr_rush(tuple(sorted(seasons)), recency)
    agg, lg = R.pfr_rush_aggregates(pfr)
    return pfr, agg, lg


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Loading play-by-play run-defense data…")
def get_pbp(seasons, recency):
    return D.load_pbp(tuple(sorted(seasons)), recency)


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner=False)
def get_derived(seasons, recency):
    wk = get_weekly(seasons, recency)
    pfr, _, _ = get_pfr(seasons, recency)
    pbp = get_pbp(seasons, recency)
    return (D.list_players(wk, stat="carries", min_vol=40), D.list_defenses(wk),
            R.team_rush_volume(wk), R.rush_defense_profiles(wk, pfr, pbp))


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

players, defenses, team_vol, def_profiles = get_derived(tuple(seasons), recency)

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
        live_row = UI.pick_player(rosters, ['RB', 'QB', 'WR', 'FB'], key='rush')
        player_id = live_row["player_id"]
if not use_live:
    player_label = st.sidebar.selectbox(
        "Rusher", players["label"].tolist(),
        help="Only players with 40+ carries in the selected seasons are listed.")
    player_row = players[players["label"] == player_label].iloc[0]
    player_id = player_row["player_id"]


opp = st.sidebar.selectbox("Opponent defense", defenses,
                           index=defenses.index("SF") if "SF" in defenses else 0)
line = st.sidebar.number_input("Prop line (rushing yards)", 0.0, 250.0, 59.5, 0.5)

st.sidebar.divider()
use_def = st.sidebar.toggle("Adjust for opponent run defense", value=True)
shrink = st.sidebar.slider(
    "Defense adjustment strength", 0.0, 1.0, D.DEFAULT_DEF_SHRINK, 0.05,
    disabled=not use_def,
    help="How much to trust one season of run-defense splits. "
         "0 = ignore, 1 = take at face value.")
n_sims = st.sidebar.select_slider("Simulations", [5000, 10000, 20000, 50000, 100000],
                                  value=20000)

_, pfr_agg, pfr_lg = get_pfr(tuple(seasons), recency)
try:
    pri = R.player_rush_priors(wk, player_id, pfr_agg, pfr_lg)
except ValueError:
    if live_row is None:
        raise
    pri = RO.rushing_priors_from_role(live_row, pfr_lg)
if live_row is not None:
    pri["team"] = live_row["team"]
    pri["mu_share"] = UI.live_share(live_row, "carry_share")
pos = pri["position"]
tv = team_vol.get(pri["team"], team_vol["_LEAGUE_"])
dprof = def_profiles.get(opp) if use_def else None

sim = R.simulate(pri, tv, dprof, n_sims=n_sims, def_shrink=shrink, seed=7)
s = R.summarize(sim, line)

st.title("🏈 Rushing Yards Simulator")
if live_row is not None:
    UI.status_warning(live_row)
    st.caption(UI.role_caption(live_row, "carry_share", "carries"))
st.caption(f"**{pri['name']}** ({pos}, {pri['team']}) vs **{opp}** defense — "
           f"{pri['games']} games of history, {n_sims:,} simulations")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Chance over {line:g}", f"{s['p_over']:.0%}",
          help=f"Fair odds  Over {s['fair_over_odds']} / Under {s['fair_under_odds']}")
c2.metric("Projected yards (mean)", f"{s['mean']:.0f}")
c3.metric("Most likely (median)", f"{s['median']:.0f}")
c4.metric("Carries / broken tkl", f"{s['mean_carries']:.1f} / {s['mean_broken']:.1f}")

if use_def and dprof is not None:
    adj = sim["adj"]
    msg = (f"**{opp} run defense:** {R.rush_scheme_label(dprof)}.  \n"
           f"Allows {dprof.get('ybc_allowed', float('nan')):.2f} yds before contact / "
           f"{dprof.get('yac_allowed', float('nan')):.2f} after "
           f"(league {dprof.get('lg_ybc', float('nan')):.2f} / {dprof.get('lg_yac', float('nan')):.2f}).  \n"
           f"Front ×{adj['m_ybc']:.2f} · tackling ×{adj['m_yac']:.2f} · "
           f"broken-tkl ×{adj['m_brk']:.2f}")
    if dprof.get("has_pbp"):
        msg += (f" · stuff ×{adj['m_stuff']:.2f} · explosive ×{adj['m_expl']:.2f} · "
                f"efficiency ×{adj['m_eff']:.2f}.  \n"
                f"Play-by-play ({dprof.get('def_runs', 0)} runs faced): "
                f"stuffs {dprof['stuff_allowed']:.0%} of runs (league {dprof['lg_stuff']:.0%}), "
                f"10+ yd runs {dprof['expl_allowed']:.0%} (league {dprof['lg_expl']:.0%}), "
                f"EPA/rush {dprof['epa_allowed']:+.3f}.")
    else:
        msg += ".  \n_No play-by-play profile for this defense in the selected seasons — "
        msg += "stuff / explosive / efficiency left at league average._"
    st.info(msg)
elif not use_def:
    st.info("Defense adjustment is **off** — this is the player's baseline "
            "distribution against a league-average run defense.")

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
    title="Simulated rushing yards — probability of each outcome",
    xaxis_title="Rushing yards in the game", yaxis_title="Probability",
    yaxis_tickformat=".1%", bargap=0.02, height=440,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(t=70, b=40, l=60, r=20))
st.plotly_chart(fig, width="stretch")

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
    st.plotly_chart(cfig, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Player profile (inputs)")
    st.dataframe(pd.DataFrame({
        "Metric": ["Carry share", "Yards before contact / att", "Yards after contact / att",
                   "Broken-tackle rate", "Yards / carry", "Games"],
        "Mean":   [f"{pri['mu_share']:.1%}", f"{pri['mu_ybc']:.2f}", f"{pri['mu_yac']:.2f}",
                   f"{pri['brk_rate']:.1%}", f"{pri['mu_ypc']:.2f}", str(pri["games"])],
        "Std (variance)": [f"±{pri['sd_share']:.1%}", f"±{pri['sd_ybc']:.2f}", f"±{pri['sd_yac']:.2f}",
                           "—", "—", "—"],
    }), hide_index=True, width="stretch")
    st.caption(f"Advanced inputs: {pri['adv_source']}.  "
               f"Simulated per game: ~{s['mean_stuffs']:.1f} stuffed runs, "
               f"~{s['mean_explosives']:.1f} explosive (10+) runs.")
with right:
    st.subheader("Outcome percentiles")
    st.dataframe(pd.DataFrame({
        "Percentile": ["10th (floor)", "25th", "Median", "75th", "90th (ceiling)"],
        "Yards": [f"{s['p10']:.0f}", f"{s['p25']:.0f}", f"{s['median']:.0f}",
                  f"{s['p75']:.0f}", f"{s['p90']:.0f}"],
    }), hide_index=True, width="stretch")
    st.metric("Fair prop odds",
              f"Over {s['fair_over_odds']}  /  Under {s['fair_under_odds']}")

st.caption("Model is for research/entertainment. Priors come from nflverse "
           "regular-season data; each carry resolves into a stuffed, normal or "
           "explosive run whose rates and yards are shrunk toward league average "
           "by the opponent's run defense (front, tackling, broken tackles, stuff "
           "rate, explosive rate and efficiency).")
