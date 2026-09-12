"""Touchdowns simulator page — total TDs combining rushing AND receiving.
Same template as the yardage pages: line check (default 0.5 = anytime TD),
fair odds, split distribution chart, cumulative view, inputs & outcome tables."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from nflsim import data as D, touchdowns as T, game as G
from nflsim import roster as RO, ui as UI

st.set_page_config(page_title="Touchdowns Simulator", page_icon="🏈", layout="wide")


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Downloading NFL data…")
def get_weekly(seasons, recency):
    return D.load_weekly(tuple(sorted(seasons)), recency)


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner=False)
def get_derived(seasons, recency):
    wk = get_weekly(seasons, recency)
    g = (wk.groupby(["player_id", "player_display_name", "position"], as_index=False)
           .agg(rec=("receptions", "sum"), car=("carries", "sum"),
                team=("recent_team", "last"), games=("week", "count")))
    g["opp"] = g["rec"] + g["car"]
    g = g[g["opp"] >= 30].sort_values("opp", ascending=False)
    g["label"] = g["player_display_name"] + " (" + g["position"] + ", " + g["team"] + ")"
    return (g.reset_index(drop=True), D.list_defenses(wk),
            T.league_td_rates(wk), T.td_defense_profiles(wk))


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Loading play-by-play for goal-line roles…")
def get_goal_line(seasons, recency):
    wk = get_weekly(seasons, recency)
    return T.goal_line_profiles(D.load_touches(tuple(sorted(seasons)), recency), wk)


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Loading live depth charts and injury reports…")
def get_rosters(seasons, recency, use_injuries):
    return UI.cached_rosters(tuple(sorted(seasons)), use_injuries, recency)


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner=False)
def get_live_team_vol(seasons, recency):
    return UI.cached_live(tuple(sorted(seasons)), recency)["team_vol"]


st.sidebar.header("Setup")
seasons, recency = UI.priors_picker("Seasons used to build priors")
view = UI.view_picker("td")

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

players, defenses, lg, def_profiles = get_derived(tuple(seasons), recency)

live_row, game, ctx = None, None, None
if view == "Game":
    ctx = UI.cached_context(tuple(seasons), recency)
    game = UI.game_picker(ctx, seasons, recency, ['RB', 'WR', 'TE', 'QB', 'FB'], key='td')
    live_row, player_id, opp = game["row"], game["row"]["player_id"], game["opp"]
else:
    use_live = st.sidebar.toggle(
        "Pick from live depth charts", value=True,
        help="Current-season depth chart and injury report. The player's usage share is "
             "redistributed when teammates are ruled out, and team volume follows his "
             "CURRENT team, not the one in his history.")
    if use_live:
        use_inj = st.sidebar.toggle("Drop players ruled out", value=True)
        rosters = get_rosters(tuple(seasons), recency, use_inj)
        if rosters.empty:
            st.sidebar.error("Depth charts did not load — using history only."); use_live = False
        else:
            live_row = UI.pick_player(rosters, ['RB', 'WR', 'TE', 'QB', 'FB'], key='td')
            player_id = live_row["player_id"]
    if not use_live:
        player_label = st.sidebar.selectbox(
            "Player", players["label"].tolist(),
            help="Players with 30+ combined carries and catches in the selected seasons.")
        player_row = players[players["label"] == player_label].iloc[0]
        player_id = player_row["player_id"]
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

try:
    pri = T.player_td_priors(wk, player_id, lg, gl=get_goal_line(tuple(seasons), recency))
except ValueError:
    if live_row is None:
        raise
    pri = RO.td_priors_from_role(live_row, get_live_team_vol(tuple(seasons), recency), lg)
if live_row is not None:
    tv = get_live_team_vol(tuple(seasons), recency)
    tv = tv.get(live_row["team"], tv["_LEAGUE_"])
    pri["team"] = live_row["team"]
    pri = RO.scale_volume_priors(pri, "mu_rec", "var_rec",
        UI.live_share(live_row, "target_share") * tv["targets"] * float(live_row["catch_rate"]))
    pri = RO.scale_volume_priors(pri, "mu_car", "var_car",
        UI.live_share(live_row, "carry_share") * tv["carries"])
pos = pri["position"]
factors = None
if game is not None:
    gr = game["game_row"]
    factors = G.script_factors(ctx, game["team"], opp, home="a" if game["is_home"] else "b",
                               wind=gr.get("wind"), roof=gr.get("roof"))
    factors["is_home"] = game["is_home"]
    typical = dict(rec=pri["mu_rec"], car=pri["mu_car"], p_rec=pri["p_rec_td"], p_rush=pri["p_rush_td"])
    # volume follows the game script; per-touch conversion follows this
    # game's expected scoring vs the team's typical (more scoring drives, more
    # red-zone touches)
    k_db = factors["dropbacks"][0] / max(factors["dropbacks_typical"], 1e-6)
    k_car = factors["carries"][0] / max(factors["carries_typical"], 1e-6)
    pri = RO.scale_volume_priors(pri, "mu_rec", "var_rec", pri["mu_rec"] * k_db)
    pri = RO.scale_volume_priors(pri, "mu_car", "var_car", pri["mu_car"] * k_car)
    pri["p_rec_td"] = float(np.clip(pri["p_rec_td"] * factors["td_factor"], 0.0, 0.5))
    pri["p_rush_td"] = float(np.clip(pri["p_rush_td"] * factors["td_factor"], 0.0, 0.4))
dprof = def_profiles.get((opp, pos)) if use_def else None
sim = T.simulate(pri, dprof, n_sims=n_sims, def_shrink=shrink, seed=7)
s = T.summarize(sim, line)

st.title("🏈 Touchdowns Simulator")
if live_row is not None:
    UI.status_warning(live_row)
    st.caption(UI.role_caption(live_row, "target_share", "targets") + "  " + chr(10)
               + UI.role_caption(live_row, "carry_share", "carries"))
if factors is not None:
    fav = factors["team"] if factors["exp_margin"] >= 0 else factors["opp"]
    st.caption(f"**Game view — {factors['team']} vs {factors['opp']}:** expected margin {fav} by "
               f"{abs(factors['exp_margin']):.1f} ({factors['team']} win {factors['win']:.0%}), "
               f"{factors['team']} projected {factors['points_for']:.1f} points (typical "
               f"{factors['typical_points']:.1f}). Expected receptions {typical['rec']:.1f} → "
               f"**{pri['mu_rec']:.1f}**, carries {typical['car']:.1f} → **{pri['mu_car']:.1f}**; "
               f"TD rates ×{factors['td_factor']:.2f} for this game's scoring outlook.")
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
st.plotly_chart(fig, width="stretch")

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
    st.plotly_chart(cfig, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Player profile (inputs)")
    st.dataframe(pd.DataFrame({
        "Metric": ["Expected receptions", "Rec TD / reception", "Expected carries",
                   "Rush TD / carry", "Goal-line share of targets", "Goal-line share of carries",
                   "Games"],
        "Value":  [f"{pri['mu_rec']:.1f}", f"{pri['p_rec_td']:.1%}",
                   f"{pri['mu_car']:.1f}", f"{pri['p_rush_td']:.1%}",
                   f"{pri['gl_tgt_frac']:.1%}" if np.isfinite(pri.get("gl_tgt_frac", np.nan)) else "—",
                   f"{pri['gl_car_frac']:.1%}" if np.isfinite(pri.get("gl_car_frac", np.nan)) else "—",
                   str(pri["games"])],
        "Prior / raw": [ "—", f"prior {pri['prior_rec_td']:.1%} · raw {pri['raw_rec_td_per_rec']:.1%}", "—",
                         f"prior {pri['prior_rush_td']:.1%} · raw {pri['raw_rush_td_per_car']:.1%}",
                         "—", "—", "—"],
    }), hide_index=True, width="stretch")
    st.caption(f"TD rates are regressed toward the **{pri.get('role_source', 'positional mean')}** "
               "— with play-by-play, that is the player's share of his team's touches inside "
               "the 10 times the league conversion there (~29% per carry, ~39% per target), "
               "which is far more stable than his own touchdown count.")
with right:
    st.subheader("Outcome probabilities")
    st.dataframe(pd.DataFrame({
        "Outcome": [f"{k} TD" for k in dist.keys()],
        "Chance": [f"{v:.1%}" for v in dist.values()],
    }), hide_index=True, width="stretch")
    st.metric(f"Fair odds — {'anytime' if line == 0.5 else f'over {line:g}'}",
              f"Over {s['fair_over_odds']}  /  Under {s['fair_under_odds']}")

st.caption("Total TDs = Binomial(receptions, rec-TD rate) + Binomial(carries, "
           "rush-TD rate), so scoring scales with volume and both phases count. "
           "For research/entertainment.")
