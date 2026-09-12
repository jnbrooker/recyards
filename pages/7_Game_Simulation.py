"""Game Simulation — two depth charts in, a simulated box score out (roadmap §4.2–4.4).

The whole game is simulated top-down thousands of times: pace, then drive
outcomes from the team-strength ratings, then the game script, then the players
on each depth chart. Because touchdowns are allocated out of the team's
simulated total, the box score always reconciles to the scoreboard.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from nflsim import data as D, game as G, teams as T
from nflsim import ui as UI

st.set_page_config(page_title="Game Simulation", page_icon="🏈", layout="wide")


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Loading play-by-play, depth charts and injuries…")
def get_context(seasons, recency):
    return G.prepare(tuple(sorted(seasons)), recency=recency)


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner=False)
def get_schedule(season):
    return D.load_schedule((int(season),))


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner=False)
def get_roster(seasons, recency, team, use_injuries):
    ctx = get_context(seasons, recency)
    return G.roster_for(ctx, team, use_injuries=use_injuries)


st.sidebar.header("Setup")
seasons, recency = UI.priors_picker("Seasons used to build priors")

ctx = get_context(tuple(seasons), recency)
UI.recency_caption(ctx["wk"], recency)
ratings = ctx["ratings"]
teams = list(ratings["off"].index)

st.sidebar.divider()
sched = get_schedule(ctx["depth_seasons"][-1])
from_sched = st.sidebar.toggle("Pick a game from the schedule", value=not sched.empty,
                               disabled=sched.empty)
game_row = None
if from_sched:
    weeks = sorted(sched["week"].unique().tolist())
    cur = D.current_week(sched)
    week = st.sidebar.selectbox("Week", weeks, index=weeks.index(cur) if cur in weeks else 0)
    wg = sched[sched["week"] == week].reset_index(drop=True)
    labels = [D.game_label(r) for _, r in wg.iterrows()]
    unplayed = [i for i, r in wg.iterrows() if not bool(r["played"])]
    pick = st.sidebar.selectbox("Game", labels, index=unplayed[0] if unplayed else 0,
                                help="Kickoff times are local to the venue. Played "
                                     "games show the final, so you can compare.")
    game_row = wg.iloc[labels.index(pick)]
    home, away = game_row["home_team"], game_row["away_team"]
    neutral = False
else:
    home = st.sidebar.selectbox("Home team", teams,
                                index=teams.index("BAL") if "BAL" in teams else 0)
    away = st.sidebar.selectbox("Away team", teams,
                                index=teams.index("SF") if "SF" in teams else 1)
    neutral = st.sidebar.toggle("Neutral site", value=False)
use_inj = st.sidebar.toggle("Drop players ruled out", value=True,
                            help="Uses the latest injury report of the current "
                                 "season (Out and Doubtful).")
n_sims = st.sidebar.select_slider("Simulations", [4000, 10000, 20000, 50000], value=20000)

st.title("🏈 Game Simulation")
if home == away:
    st.warning("Pick two different teams."); st.stop()

dseasons = ", ".join(str(s) for s in ctx["depth_seasons"])
st.caption(f"Depth charts from the {dseasons} season · priors from "
           f"{', '.join(str(s) for s in ratings['seasons'])} · {n_sims:,} simulations")

try:
    r_home = get_roster(tuple(seasons), recency, home, use_inj)
    r_away = get_roster(tuple(seasons), recency, away, use_inj)
except ValueError as e:
    st.error(str(e)); st.stop()

sim = G.simulate_game(ratings, ctx["wk"], r_home, r_away, home, away,
                      ctx["pass_vol"], ctx["rush_vol"], ctx["rush_def"],
                      ctx["lg_pass"], home=None if neutral else "a",
                      n_sims=n_sims, seed=11)
s = G.summarize(sim)

fav, dog = (home, away) if s["mean_margin"] >= 0 else (away, home)
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{home} (home)" if not neutral else home, f"{s['mean_a']:.1f}")
c2.metric(f"{away}", f"{s['mean_b']:.1f}")
c3.metric(f"{fav} win probability", f"{max(s['win_a'], s['win_b']):.0%}",
          help=f"Fair odds {D.american(max(s['win_a'], s['win_b']))}")
c4.metric("Projected total", f"{s['mean_total']:.1f}",
          help=f"80% of simulations land between {s['p10_total']:.0f} and {s['p90_total']:.0f}")

st.info(
    f"**{fav} by {abs(s['mean_margin']):.1f}.** "
    f"{home} wins {s['win_a']:.1%}, {away} wins {s['win_b']:.1%}, ties {s['tie']:.1%}.  \n"
    f"Each team gets about {sim['pace']['mean']:.1f} drives. "
    + ("Neutral site — no home-field adjustment."
       if neutral else
       f"Home field is worth {ratings['hfa']:+.2f} points of margin here.")
)

if game_row is not None:
    bits = []
    if pd.notna(game_row.get("spread_line")) and pd.notna(game_row.get("total_line")):
        sp = float(game_row["spread_line"])
        fav_txt = f"{home} -{sp:g}" if sp > 0 else f"{away} -{-sp:g}" if sp < 0 else "pick"
        bits.append(f"**Closing market:** {fav_txt}, total {game_row['total_line']:g} — "
                    f"model says {fav} by {abs(s['mean_margin']):.1f}, total "
                    f"{s['mean_total']:.1f}. Shown for comparison only; the model "
                    "never reads the line.")
    if bool(game_row.get("played")):
        bits.append(f"**Final:** {away} {int(game_row['away_score'])} — "
                    f"{home} {int(game_row['home_score'])}.")
    if pd.notna(game_row.get("roof")):
        wx = f"{game_row['roof']}"
        if pd.notna(game_row.get("wind")):
            wx += f", wind {game_row['wind']:.0f} mph"
        if pd.notna(game_row.get("temp")):
            wx += f", {game_row['temp']:.0f}°F"
        bits.append(f"Venue: {wx}.")
    if bits:
        st.caption(("  " + chr(10)).join(bits))

# --- score distribution ----------------------------------------------------
st.subheader("How the game finishes")
margin = sim["points_a"] - sim["points_b"]
edges = np.arange(margin.min() - 0.5, margin.max() + 1.5, 1.0)
counts, _ = np.histogram(margin, bins=edges)
centres = (edges[:-1] + edges[1:]) / 2
colors = ["#2e7d5b" if c > 0 else "#c0563b" if c < 0 else "#888" for c in centres]
fig = go.Figure()
fig.add_bar(x=centres, y=counts / counts.sum(), marker_color=colors,
            hovertemplate="margin %{x:+.0f}<br>%{y:.2%} of simulations<extra></extra>")
fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="#222")
fig.update_layout(height=380, bargap=0.05, showlegend=False,
                  xaxis_title=f"{home} margin (negative = {away} wins)",
                  yaxis_title="Share of simulations", yaxis_tickformat=".1%",
                  margin=dict(t=20, b=45, l=60, r=20))
st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    st.metric("Median margin", f"{s['median_margin']:+.0f}",
              help=f"Spread of outcomes: sd {s['margin_sd']:.1f} points")
with right:
    st.metric("Fair moneyline",
              f"{home} {D.american(s['win_a'])} / {away} {D.american(s['win_b'])}")

with st.expander("Total points distribution"):
    tot = sim["points_a"] + sim["points_b"]
    tedges = np.arange(tot.min() - 0.5, tot.max() + 1.5, 2.0)
    tc, _ = np.histogram(tot, bins=tedges)
    tfig = go.Figure()
    tfig.add_bar(x=(tedges[:-1] + tedges[1:]) / 2, y=tc / tc.sum(),
                 marker_color="#2e7d5b",
                 hovertemplate="%{x:.0f} points<br>%{y:.2%}<extra></extra>")
    tfig.update_layout(height=320, bargap=0.05, xaxis_title="Combined points",
                       yaxis_title="Share of simulations", yaxis_tickformat=".1%",
                       margin=dict(t=20, b=45, l=60, r=20))
    st.plotly_chart(tfig, width="stretch")

# --- box scores ------------------------------------------------------------
st.subheader("Projected offensive box score")
st.caption("Mean of every simulation. Touchdowns are split out of each team's "
           "simulated total, so the two columns always add back to the score.")

cols = st.columns(2)
for col, team, side in ((cols[0], home, sim["box_a"]), (cols[1], away, sim["box_b"])):
    with col:
        q = G.qb_line(side)
        st.markdown(f"#### {team}")
        st.markdown(
            f"**{q['name']}** — {q['completions']:.1f}/{q['attempts']:.1f}, "
            f"{q['pass_yards']:.0f} yds, {q['pass_tds']:.1f} TD, {q['ints']:.1f} INT, "
            f"{q['sacks']:.1f} sacks taken")
        bs = G.box_score(side)
        show = bs[bs["Yds"] >= 1.0][["Player", "Pos", "Depth", "Tgt", "Rec",
                                     "RecYds", "Car", "RushYds", "TD", "Source"]]
        st.dataframe(show.style.format({
            "Tgt": "{:.1f}", "Rec": "{:.1f}", "RecYds": "{:.0f}",
            "Car": "{:.1f}", "RushYds": "{:.0f}", "TD": "{:.2f}"}),
            hide_index=True, width="stretch")
        st.caption(f"Team: {side['team_pass_yards'].mean():.0f} pass + "
                   f"{side['team_rush_yards'].mean():.0f} rush = "
                   f"{side['team_pass_yards'].mean() + side['team_rush_yards'].mean():.0f} yards "
                   f"· {side['pass_frac'].mean():.0%} of plays are dropbacks")

with st.expander("How a simulation runs, and what it does not model"):
    st.markdown(
        "1. **Pace** — one shared draw of how many drives the game has; both "
        "teams get within a possession of each other.\n"
        "2. **Drives** — each one resolves to touchdown / field goal / turnover "
        "/ nothing at the matchup rates from the Team Strength page. Expected "
        "points are held *pace-invariant*: a 13-drive game has the same expected "
        "points as an 8-drive one, because in real football extra drives are "
        "extra three-and-outs (measured correlation between drives and points "
        "per game: −0.04).\n"
        "3. **Game script** — once the margin is known, the team that trails "
        "throws more and the team that leads runs more.\n"
        "4. **Allocation** — Dirichlet shares spread targets and carries across "
        "the depth chart; each player's own priors turn that volume into catches "
        "and yards; team touchdowns are split by a multinomial over role weights.\n\n"
        "**Roles** come from a player's own history when he has one, and from a "
        "positional-rank prior (WR1 / WR2 / RB1 …) when he does not — the "
        "*Source* column says which. So a rookie starter inherits his slot's "
        "prior while a veteran keeps his own numbers.\n\n"
        "**Not modelled:** an explicit clock, field position, or drive ordering — "
        "drives are exchangeable within a game. Sacks and interceptions come from "
        "the Phase 4 models and match the drive engine's turnover rate on average, "
        "but are not linked to it play by play.")

st.caption("Everything here is self-contained — team strength is solved from "
           "drive outcomes, never from a market line. For research/entertainment.")
