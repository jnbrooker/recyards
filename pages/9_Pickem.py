"""Pick'em — a confidence card for the week, graded by the game engine.

Twenty slots, confidence 20 down to 1. One pick per game (either side against
the spread, or the underdog moneyline), a 3-team ATS parlay, a 3-team ML parlay,
a 3-team 6-point teaser, and over/unders on the games the pool names. Lines are
editable so the card is graded against what the pool actually offers.
"""

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go

from nflsim import data as D, game as G, pickem as P, backtest as B
from nflsim import ui as UI

st.set_page_config(page_title="Pick'em", page_icon="🏈", layout="wide")


def get_context(seasons, recency):
    return UI.cached_context(tuple(sorted(seasons)), recency)


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner=False)
def get_schedule(season):
    return D.load_schedule((int(season),))


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Simulating every game this week…")
def get_slate(seasons, recency, week, n_sims, use_injuries):
    ctx = get_context(seasons, recency)
    sched = get_schedule(ctx["depth_seasons"][-1])
    games = sched[sched["week"] == int(week)]
    return P.simulate_slate(ctx, games, n_sims=n_sims, use_injuries=use_injuries)


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Refitting the model week by week for past games…")
def get_team_backtest(seasons, recency):
    """Out-of-sample margin and total for every played game: ratings, home
    field, availability and calibration refit on games BEFORE each week."""
    return B.team_backtest(list(seasons), recency=recency, availability=True)


# --- sidebar ----------------------------------------------------------------
st.sidebar.header("Setup")
seasons, recency = UI.priors_picker("Seasons used to build priors")
ctx = get_context(tuple(seasons), recency)
UI.recency_caption(ctx["wk"], recency)
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
sims = get_slate(tuple(seasons), recency, int(week), int(n_sims), use_inj)
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

# --- how the card has done --------------------------------------------------------
st.subheader("How the card has done")
st.caption(
    "Every played week is replayed **honestly**: the ratings, home field, availability "
    "and scoring calibration are refit on games before that week only, each game's "
    "distribution is the harness's (Normal, sd 12.8 on the margin and 13.4 on the total), "
    "the card is built with the pool rules and market lean set above against the closing "
    "lines, and graded against the final scores. Totals slots use the model's best games "
    "(the pool's actual choices are unknown). Pushes score zero; a parlay with a pushed leg "
    "counts as lost.")
season_now = int(ctx["depth_seasons"][-1])
hist_seasons = st.multiselect("Seasons to replay", [season_now, season_now - 1],
                              default=[season_now],
                              help="Add last season for a sample big enough to mean something — "
                                   "one week is noise either way.")
if hist_seasons:
    bt = get_team_backtest(tuple(sorted(hist_seasons)), recency)
    hsched = D.load_schedule(tuple(sorted(hist_seasons)))
    kw = dict(mode=mode_key, juice=int(juice), teaser_pts=float(teaser_pts),
              teaser_odds=int(teaser_odds),
              parlay_pricing="true" if parlay_pricing.startswith("true") else "fixed",
              parlay_odds=int(parlay_odds), n_slots=int(n_slots))
    hw, hp = P.card_history(bt, hsched, market_weight=float(market_w), **kw)
    mw, _ = P.card_history(bt, hsched, market_weight=1.0, **kw)
    if hw.empty:
        st.info("No played weeks to replay yet.")
    else:
        pts, exp, mkt = hw["points"].sum(), hw["expected"].sum(), hw["market_expected"].sum()
        mpts = mw["points"].sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Points scored", f"{pts:,.0f}", delta=f"{pts - exp:+,.0f} vs expected",
                  delta_color="normal", help=f"The card expected {exp:,.0f}; at pure market "
                                             f"prices it would have expected {mkt:,.0f}.")
        c2.metric("Market-only card", f"{mpts:,.0f}",
                  delta=f"{pts - mpts:+,.0f} model vs market", delta_color="normal",
                  help="The same rules with the market lean at 100% — every game centred on "
                       "the closing line. The honest baseline: is the model adding anything?")
        c3.metric("Wins", f"{int(hw['wins'].sum())} of {int(hw['picks'].sum())}",
                  delta=f"{hw['wins'].sum() - hw['exp_wins'].sum():+.1f} vs expected",
                  delta_color="normal")
        gp = hp[(hp["slot"] == "Game pick") & (hp["result"].isin(["win", "loss"]))]
        c4.metric("Game picks hit", f"{(gp['result'] == 'win').mean():.1%}" if len(gp) else "—",
                  help=f"{len(gp)} single game picks (ATS or underdog ML), pushes excluded. "
                       "Breakeven at −110 is 52.4%; a moneyline dog is expected to hit far less.")

        show = hw[["season", "week", "games", "points", "expected", "market_expected", "wins",
                   "picks", "exp_wins", "top5_wins", "game_pick_hit", "totals_hit",
                   "combos_won", "cum_points", "cum_expected"]].rename(columns={
            "season": "Season", "week": "Week", "games": "Games", "points": "Points",
            "expected": "Expected", "market_expected": "Market exp.", "wins": "Wins",
            "picks": "Picks", "exp_wins": "Exp. wins", "top5_wins": "Top-5 wins",
            "game_pick_hit": "Game picks", "totals_hit": "Totals", "combos_won": "Combos won",
            "cum_points": "Cum. points", "cum_expected": "Cum. expected"})
        st.dataframe(show.style.format({"Points": "{:.0f}", "Expected": "{:.0f}",
                                        "Market exp.": "{:.0f}", "Exp. wins": "{:.1f}",
                                        "Game picks": "{:.0%}", "Totals": "{:.0%}",
                                        "Cum. points": "{:.0f}", "Cum. expected": "{:.0f}"},
                                       na_rep="—"),
                     hide_index=True, width="stretch")

        if len(hw) > 1:
            x = [f"{s}-{w:02d}" for s, w in zip(hw["season"], hw["week"])]
            fig = go.Figure()
            fig.add_scatter(x=x, y=hw["cum_points"], name="Model card, realised",
                            mode="lines+markers", line=dict(color="#2e7d5b", width=3))
            fig.add_scatter(x=x, y=hw["cum_expected"], name="Model card, expected",
                            mode="lines", line=dict(color="#2e7d5b", dash="dot"))
            fig.add_scatter(x=x, y=mw["cum_points"], name="Market-only card, realised",
                            mode="lines+markers", line=dict(color="#888", width=2))
            fig.update_layout(height=360, yaxis_title="Cumulative points",
                              legend=dict(orientation="h", y=1.12),
                              margin=dict(t=20, b=40, l=60, r=20))
            st.plotly_chart(fig, width="stretch")

        st.caption(
            "Read the gap between **expected** and **realised** as the model's optimism about "
            "the sides it picks: when you always take the side the model likes, its stated "
            "probability on that side runs high (selection). If the model card is not beating "
            "the market-only card over a season, the model is not adding to the market for "
            "this pool — that is the honest test, and one week cannot pass or fail it.")

        with st.expander("Each week's graded card"):
            for (S, w), g in hp.groupby(["season", "week"]):
                st.markdown(f"**{S} week {w}** — {g['points'].sum():.0f} points, "
                            f"{int((g['result'] == 'win').sum())} of {len(g)} won")
                gd = pd.DataFrame({"Conf": g["confidence"], "Slot": g["slot"], "Pick": g["text"],
                                   "Model P": g["p"], "Price": [P.to_american(x) for x in g["dec"]],
                                   "Result": g["result"], "Points": g["points"]})
                st.dataframe(gd.style.format({"Model P": "{:.1%}", "Points": "{:.1f}"}),
                             hide_index=True, width="stretch", height=min(480, 38 * len(gd) + 40))

st.caption("The card maximises expected points: picks are sorted by expected points per "
           "confidence point and confidence is handed out in that order, which is optimal "
           "for a sum of confidence × value. Pushes count as losses. Parlay legs come from "
           "different games so their probabilities multiply exactly. For research/entertainment.")
