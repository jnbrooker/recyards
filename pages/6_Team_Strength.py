"""Team Strength — the self-contained view of how good each team is (roadmap §4.1).

Every rating here comes from drive outcomes in the play-by-play feed, opponent-
adjusted by solving offense and defense together. Nothing is anchored to a Vegas
total or spread. This is the layer the drive-based game engine sits on top of.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from nflsim import data as D, teams as T
from nflsim import ui as UI

st.set_page_config(page_title="Team Strength", page_icon="🏈", layout="wide")


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Downloading play-by-play (this one is a big file)…")
def get_ratings(seasons, recency):
    seasons = tuple(sorted(seasons))
    drives = D.load_drives(seasons, recency)
    if drives.empty:
        return None
    r = T.team_ratings(drives, D.load_games(seasons))
    r["weight_shares"] = D.weight_shares(drives[drives["live"]], team_col="posteam")
    return r


st.sidebar.header("Setup")
seasons, recency = UI.priors_picker("Seasons used to rate teams")

r = get_ratings(tuple(seasons), recency)
if r is None:
    st.error("Play-by-play data did not load for those seasons — try another season.")
    st.stop()
UI.recency_caption(None, recency, shares=r.get("weight_shares"))

teams = list(r["off"].index)
st.sidebar.divider()
team_a = st.sidebar.selectbox("Home team", teams,
                              index=teams.index("BAL") if "BAL" in teams else 0)
team_b = st.sidebar.selectbox("Away team", teams,
                              index=teams.index("SF") if "SF" in teams else 1)
neutral = st.sidebar.toggle("Neutral site", value=False,
                            help="Turn off home-field advantage for this matchup.")

st.title("🏈 Team Strength")
st.caption(f"Opponent-adjusted points per drive from {r['drives']:,} drives across "
           f"{', '.join(str(s) for s in r['seasons'])} — no Vegas anchor.")

if team_a == team_b:
    st.warning("Pick two different teams to see a matchup.")
    st.stop()

e = T.expected_points(r, team_a, team_b, home=None if neutral else "a")
a, b = e["a"], e["b"]

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{team_a} projected", f"{e['points_a']:.1f}")
c2.metric(f"{team_b} projected", f"{e['points_b']:.1f}")
c3.metric("Margin", f"{e['margin']:+.1f}",
          help=f"{team_a} minus {team_b}" + ("" if neutral else
               f", including {r['hfa']:+.2f} of home field"))
c4.metric("Total", f"{e['total']:.1f}", help=f"{e['drives']:.1f} drives each")

st.info(
    f"**{team_a} offense vs {team_b} defense:** {a['ppd']:.2f} points per drive "
    f"(league {r['lg_ppd']:.2f}).  \n"
    f"**{team_b} offense vs {team_a} defense:** {b['ppd']:.2f} points per drive.  \n"
    f"Pace: {e['drives']:.1f} drives each. Home field is fitted from real results "
    f"at **{r['hfa']:+.2f}** points of margin"
    + (" — switched off for this neutral-site matchup." if neutral else ".")
)

# --- per-drive outcome mix -------------------------------------------------
st.subheader("What each drive is worth")
mix = go.Figure()
labels = ["Touchdown", "Field goal", "Turnover", "No score"]
for side, m, colr in ((team_a, a, "#2e7d5b"), (team_b, b, "#c0563b")):
    mix.add_bar(name=side, x=labels, marker_color=colr,
                y=[m["p_td"], m["p_fg"], m["p_turnover"], m["p_none"]],
                hovertemplate="%{x}: %{y:.1%} of drives<extra>" + side + "</extra>")
mix.update_layout(barmode="group", yaxis_tickformat=".0%", height=380,
                  yaxis_title="Share of drives", bargap=0.3,
                  margin=dict(t=30, b=40, l=60, r=20))
st.plotly_chart(mix, width="stretch")
st.caption(
    "The outcome mix is rescaled so its point value equals the efficiency "
    f"rating: {team_a} checks to {a['mix_ppd']:.2f} vs {a['ppd']:.2f} expected, "
    f"{team_b} to {b['mix_ppd']:.2f} vs {b['ppd']:.2f}. That reconciliation is what "
    "stops the simulated box score's touchdowns from contradicting the team ratings."
)

# --- league map ------------------------------------------------------------
t = T.strength_table(r)
st.subheader("The league, offense vs defense")
fig = go.Figure()
fig.add_scatter(x=t["off_ppd"], y=t["def_ppd"], mode="markers+text",
                text=t.index, textposition="top center", textfont=dict(size=9),
                marker=dict(size=9, color=t["net"], colorscale="RdYlGn",
                            cmid=0, showscale=True,
                            colorbar=dict(title="Net", thickness=12)),
                hovertemplate=("%{text}<br>offense %{x:.2f} pts/drive"
                               "<br>defense allows %{y:.2f}<extra></extra>"))
fig.add_vline(x=r["lg_ppd"], line_dash="dot", line_color="#888")
fig.add_hline(y=r["lg_ppd"], line_dash="dot", line_color="#888")
for team, colr in ((team_a, "#2e7d5b"), (team_b, "#c0563b")):
    fig.add_scatter(x=[t.loc[team, "off_ppd"]], y=[t.loc[team, "def_ppd"]],
                    mode="markers", marker=dict(size=16, color=colr,
                                                line=dict(width=2, color="#fff")),
                    name=team, hoverinfo="skip")
fig.update_layout(height=560, showlegend=False,
                  xaxis_title="Offense — points produced per drive",
                  yaxis_title="Defense — points allowed per drive",
                  yaxis_autorange="reversed",
                  margin=dict(t=30, b=50, l=70, r=20))
st.plotly_chart(fig, width="stretch")
st.caption("Up and to the right is good: strong offense, stingy defense. "
           "Dotted lines are league average.")

with st.expander("Full rating table"):
    show = t[["off_ppd", "def_ppd", "net", "pace", "off_ppd_raw", "def_ppd_raw",
              "sched_off", "sched_def"]].copy()
    show.columns = ["Off pts/drive", "Def pts/drive allowed", "Net", "Drives/game",
                    "Off (unadjusted)", "Def (unadjusted)",
                    "Schedule faced (off)", "Schedule faced (def)"]
    st.dataframe(show.style.format("{:.3f}"), width="stretch")
    st.caption(
        "*Unadjusted* columns are the raw per-drive numbers. The adjusted ones "
        "subtract the quality of the units each team actually faced and shrink "
        "toward league average. *Schedule faced* is the mean opponent rating: "
        "positive on offense means it played weak defenses.")

with st.expander("How the scoring level is calibrated"):
    off_pts = r["lg_ppd"] * r["lg_pace"]
    def_pts = r["lg_def_score"] * r["lg_pace"] * T.TD_POINTS
    st.markdown(
        f"A league-average team scores **{off_pts + def_pts + r['lg_other_ppg']:.2f}** "
        "points a game in this model:\n\n"
        f"- **{off_pts:.2f}** on offense — {r['lg_ppd']:.3f} points per drive "
        f"× {r['lg_pace']:.2f} drives\n"
        f"- **{def_pts:.2f}** from the defense scoring itself (pick-sixes and the like)\n"
        f"- **{r['lg_other_ppg']:.2f}** from kick/punt return touchdowns and safeties, "
        "which belong to no drive at all and are measured against real final scores\n\n"
        "Ratings are solved on drives, so that last piece has to be added back "
        "explicitly or every projected total comes in light.")

st.caption("Team strength is built only from drive outcomes — no market lines. "
           "For research/entertainment.")
