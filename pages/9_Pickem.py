"""Pick'em — a confidence card for the week, graded by the game engine.

Twenty slots, confidence 20 down to 1. One pick per game (either side against
the spread, or the underdog moneyline), a 3-team ATS parlay, a 3-team ML parlay,
a 3-team 6-point teaser, and over/unders on the games the pool names. Lines are
editable so the card is graded against what the pool actually offers.
"""

import numpy as np
import pandas as pd
import streamlit as st

from nflsim import data as D, game as G, pickem as P
from nflsim import ui as UI

st.set_page_config(page_title="Pick'em", page_icon="🏈", layout="wide")


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Loading play-by-play, depth charts and injuries…")
def get_context(seasons):
    return G.prepare(tuple(sorted(seasons)))


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner=False)
def get_schedule(season):
    return D.load_schedule((int(season),))


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Simulating every game this week…")
def get_slate(seasons, week, n_sims, use_injuries):
    ctx = get_context(seasons)
    sched = get_schedule(ctx["depth_seasons"][-1])
    games = sched[sched["week"] == int(week)]
    return P.simulate_slate(ctx, games, n_sims=n_sims, use_injuries=use_injuries)


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
exclude_played = st.sidebar.toggle("Skip games already played", value=True)
use_inj = st.sidebar.toggle("Drop players ruled out", value=True)
n_sims = st.sidebar.select_slider("Simulations per game", [4000, 10000, 20000], value=10000)

st.sidebar.divider()
st.sidebar.subheader("Pool rules")
mode = st.sidebar.radio("A winning pick earns…", ["confidence × odds", "confidence (flat)"],
                        index=0, help="Odds-weighted is where underdogs and parlays can earn "
                                      "a high slot; flat scoring ranks purely by win chance.")
mode_key = "odds" if mode.startswith("confidence ×") else "flat"
n_slots = st.sidebar.number_input("Slots", 5, 40, P.N_SLOTS, 1)
juice = st.sidebar.number_input("Spread / total price", -200, 100, P.DEFAULT_JUICE, 5)
teaser_pts = st.sidebar.number_input("Teaser points", 4.0, 10.0, P.DEFAULT_TEASER_POINTS, 0.5)
teaser_odds = st.sidebar.number_input("3-team teaser price", -200, 400, P.DEFAULT_TEASER_ODDS, 5)
parlay_pricing = st.sidebar.radio("Parlays pay…", ["true odds of the legs", "a fixed 3-team price"], index=0)
parlay_odds = st.sidebar.number_input("Fixed 3-team parlay price", 100, 2000, P.DEFAULT_PARLAY_ODDS, 25,
                                      disabled=parlay_pricing.startswith("true"))

st.sidebar.divider()
market_w = st.sidebar.slider(
    "Lean on the market", 0.0, 1.0, 0.5, 0.05,
    help="0 = grade every pick on the pure model. 1 = centre each game where the "
         "line is and keep only the model's shape. The model's ratings are shrunk "
         "toward average, so it sees games as closer than the market does — which "
         "flatters every underdog. Blending is the honest hedge against that.")

# --- run ----------------------------------------------------------------------
sims = get_slate(tuple(seasons), int(week), int(n_sims), use_inj)
games = sched[sched["week"] == int(week)].copy()
games = games[games["game_id"].isin(sims.keys())]
if exclude_played:
    games = games[~games["played"]]

st.title("🏈 Pick'em Card")
st.caption(f"Week {week} · {len(games)} games in play · {n_sims:,} simulations each · "
           f"scoring: {mode} · lines below are graded, never learned from")
if games.empty:
    st.info("No games left to pick this week."); st.stop()

# --- lines, editable ----------------------------------------------------------
st.subheader("Lines")
st.caption("Seeded from the closing market. **Overwrite them with your pool's numbers.** "
           "Home spread: positive means the home team is favoured by that many "
           "(CHI −3 at CAR → −3.0).")
seed_lines = P.default_lines(games)
key = f"lines_{week}_{exclude_played}"
if st.button("Reset to closing lines"):
    st.session_state.pop(key, None)
edited = st.data_editor(
    seed_lines[["Game", "spread_home", "total", "ml_home", "ml_away"]],
    key=key, hide_index=True, width="stretch",
    column_config={
        "Game": st.column_config.TextColumn(disabled=True),
        "spread_home": st.column_config.NumberColumn("Home spread", step=0.5, format="%.1f"),
        "total": st.column_config.NumberColumn("Total", step=0.5, format="%.1f"),
        "ml_home": st.column_config.NumberColumn("Home ML", step=5, format="%d"),
        "ml_away": st.column_config.NumberColumn("Away ML", step=5, format="%d"),
    })
lines = seed_lines.copy()
for c in ("spread_home", "total", "ml_home", "ml_away"):
    lines[c] = edited[c].values

# --- which games carry the totals --------------------------------------------
n_required = len(games) + 3
n_ou = max(int(n_slots) - n_required, 0)
st.subheader(f"Totals: {n_ou} slot{'s' if n_ou != 1 else ''} this week")
prelim = P.build_card(sims, lines, mode=mode_key, juice=int(juice), teaser_pts=float(teaser_pts),
                      teaser_odds=int(teaser_odds),
                      parlay_pricing="true" if parlay_pricing.startswith("true") else "fixed",
                      parlay_odds=int(parlay_odds), n_slots=int(n_slots),
                      exclude_played=exclude_played, market_weight=float(market_w))
auto_ou = prelim["card"].loc[prelim["card"]["slot"] == "Total", "game_id"].tolist()
game_opts = dict(zip(lines["Game"], lines["game_id"]))
ou_pick = st.multiselect(
    "Games whose over/unders are in the pool", list(game_opts.keys()),
    default=[g for g, gid in game_opts.items() if gid in auto_ou],
    help="The pool names these; the model then picks over or under on each. "
         "Pre-filled with the games where the model sees the most value.")
ou_games = [game_opts[g] for g in ou_pick]

res = P.build_card(sims, lines, mode=mode_key, juice=int(juice), teaser_pts=float(teaser_pts),
                   teaser_odds=int(teaser_odds),
                   parlay_pricing="true" if parlay_pricing.startswith("true") else "fixed",
                   parlay_odds=int(parlay_odds), n_slots=int(n_slots),
                   exclude_played=exclude_played, market_weight=float(market_w),
                   ou_games=ou_games)
card = res["card"]
for n in res["notes"]:
    st.warning(n)

# --- the card -----------------------------------------------------------------
st.subheader("The card")
c1, c2, c3 = st.columns(3)
c1.metric("Expected points", f"{res['total_exp']:.1f}",
          help="Sum over slots of confidence × P(win)" + (" × payout" if mode_key == "odds" else ""))
c2.metric("Same card at market prices", f"{res['market_exp']:.1f}",
          help="What the card would be worth if every pick had exactly the probability "
               "its price implies — the model's claimed edge is the gap.")
c3.metric("Expected wins", f"{card['p'].sum():.1f} of {len(card)}")

disp = P.card_display(card)
st.dataframe(disp.style.format({"Model P": "{:.1%}", "Push": "{:.1%}", "Market P": "{:.1%}",
                                "Edge": "{:+.1%}", "Exp. pts": "{:.1f}"}),
             hide_index=True, width="stretch", height=min(760, 38 * len(disp) + 40))
if mode_key == "odds":
    st.caption("Odds-weighted scoring puts long shots and parlays at the top because their "
               "payout is large — that is the highest **expected** return, but it is also "
               "the highest variance. The Model P column is the chance each slot pays at all.")

# --- the reasoning, game by game --------------------------------------------------
with st.expander("Every game: model vs line, and every candidate's probability"):
    gv = P.game_view(res["candidates"])
    fmt = {c: "{:.1%}" for c in gv.columns if c.endswith(("covers", "wins")) or c == "Over"}
    fmt.update({"Model margin (home)": "{:+.1f}", "Line (home)": "{:+.1f}",
                "Model total": "{:.1f}", "Line total": "{:.1f}"})
    st.dataframe(gv.style.format(fmt, na_rep="—"), hide_index=True, width="stretch")
    st.caption("Probabilities already reflect the market lean above. 'Model margin' and "
               "'Model total' are the pure model's centres; the line columns are what "
               "the picks are graded against.")

with st.expander("All candidates, ranked"):
    cand = res["candidates"].sort_values("value", ascending=False)
    show = pd.DataFrame({
        "Game": cand["away"] + " @ " + cand["home"], "Type": cand["kind"], "Pick": cand["text"],
        "P": cand["p"], "Push": cand["p_push"], "Price": [P.to_american(d) for d in cand["dec"]],
        "Market P": cand["market_p"], "Edge": cand["edge"], "Value / conf pt": cand["value"]})
    st.dataframe(show.style.format({"P": "{:.1%}", "Push": "{:.1%}", "Market P": "{:.1%}",
                                    "Edge": "{:+.1%}", "Value / conf pt": "{:.3f}"}),
                 hide_index=True, width="stretch", height=500)

st.caption("The card maximises expected points: picks are sorted by expected points per "
           "confidence point and confidence is handed out in that order, which is optimal "
           "for a sum of confidence × value. Pushes count as losses. Parlay legs come from "
           "different games so their probabilities multiply exactly. For research/entertainment.")
