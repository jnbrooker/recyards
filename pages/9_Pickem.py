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

import os

from nflsim import data as D, game as G, pickem as P, backtest as B, market as MK
from nflsim import ui as UI

st.set_page_config(page_title="Pick'em", page_icon="🏈", layout="wide")


def get_context(seasons, recency):
    return UI.cached_context(tuple(sorted(seasons)), recency)


def get_schedule(season):
    return UI.cached_schedule(int(season))


def get_slate(seasons, recency, week, n_sims, use_injuries, engine="drive"):
    return UI.cached_slate(tuple(seasons), recency, int(week), int(n_sims), bool(use_injuries), engine)


def _api_key():
    try:
        k = st.secrets.get("ODDS_API_KEY", "")
    except Exception as e:
        # a secrets file that exists but will not parse (a stray BOM, a bad
        # quote) would otherwise silently disable the fetch buttons
        st.sidebar.error(f"`.streamlit/secrets.toml` could not be read: {e}")
        k = ""
    return k or os.environ.get("ODDS_API_KEY", "")


def get_history():
    return UI.cached_history()


def get_team_backtest(seasons, recency):
    return UI.cached_team_backtest(tuple(seasons), recency)


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
source_lbl = st.sidebar.radio(
    "Probabilities from", ["Market + history (recommended)", "Game engine"], index=0,
    help="**Market + history**: each side's chance is P(the pool's line is beaten | the "
         "market's line), read off 4,000+ games at the same closing spread — so it knows "
         "that 14.5% of games land on 3, and that +3.5 against a market of +3 is a 59% "
         "pick, not 50%. Six-point teaser legs that cross 3 and 7 price themselves. This is "
         "how a pool priced at openers is beaten.  \n**Game engine**: the model's own "
         "simulation, blended toward the market — on 2025 the model's disagreement with "
         "the closing line carried no information (slope 0.01), so use this only to see "
         "what the engine thinks.")
source = "market" if source_lbl.startswith("Market") else "engine"
market_w = 1.0
game_engine = "drive"
n_sims = UI.DEFAULTS["n_sims_slate"]
if source == "engine":
    game_engine = UI.engine_picker("p9")
    n_sims = UI.sims_picker(game_engine, "p9")
    market_w = st.sidebar.slider(
        "Lean on the market", 0.0, 1.0, 0.5, 0.05,
        help="0 = grade every pick on the pure model. 1 = centre each game where the "
             "line is and keep only the model's shape.")

# --- run ----------------------------------------------------------------------
sims = UI.run_with_progress(f"Simulating week {week} on the {'play' if game_engine == 'play' else 'drive'} engine",
                            get_slate, tuple(seasons), recency, int(week), int(n_sims), use_inj, game_engine)
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
st.subheader("Lines: the pool's numbers against the market's")
st.caption("**Your pool's lines** (editable — type in exactly what the pool posts) are graded "
           "against the **market's** current lines. A pool that prices at openers hands you "
           "half-points across 3 and 7 all week; that gap is where the edge is. Home spread: "
           "positive = home favoured (CHI −3 at CAR → −3.0).")
seed_lines = P.default_lines(games)
mkey = f"market_{week}"
c_fetch, c_reset, c_info = st.columns([1, 1, 3])
with c_fetch:
    if st.button("Fetch current market lines", disabled=not _api_key(),
                 help="The Odds API, same key as the props page (~3 credits). Median of "
                      "every book's spread / total / moneyline."):
        try:
            raw, quota = MK.fetch_game_lines(_api_key())
            st.session_state[mkey] = MK.consensus_lines(raw)
            st.session_state[mkey + "_at"] = pd.Timestamp.now(tz="UTC")
            st.session_state[mkey + "_quota"] = quota
        except Exception as e:
            st.error(f"Fetch failed: {e}")
with c_reset:
    if st.button("Reset pool lines to market"):
        st.session_state.pop(f"lines_{week}_{exclude_played}", None)
with c_info:
    if not _api_key():
        st.caption("Set `ODDS_API_KEY` (secrets or environment) to fetch live market lines; "
                   "until then the market columns use the schedule feed's current line.")
    elif mkey in st.session_state:
        q = st.session_state.get(mkey + "_quota", {}) or {}
        st.caption(f"Market lines fetched {st.session_state[mkey + '_at']:%a %H:%M} UTC across "
                   f"{int(st.session_state[mkey]['books'].median())} books · API credits left "
                   f"{q.get('remaining', '?')}")
if mkey in st.session_state:
    seed_lines = P.apply_market(seed_lines, st.session_state[mkey])
saved_pool = P.load_pool_lines()
season_of_week = int(games["season"].iloc[0]) if len(games) else int(ctx["depth_seasons"][-1])
seed_lines = P.apply_pool(seed_lines, saved_pool, season_of_week, int(week))
n_saved = int(((saved_pool["season"] == season_of_week) & (saved_pool["week"] == int(week))).sum())
if n_saved:
    st.caption(f"Pool lines for {n_saved} games loaded from `{P.POOL_FILE}`; the market columns "
               "are live. Edit and save to replace them.")

key = f"lines_{week}_{exclude_played}"
edited = st.data_editor(
    seed_lines[["Game", "spread_home", "mkt_spread_home", "total", "mkt_total",
                "ml_home", "mkt_ml_home", "ml_away", "mkt_ml_away"]],
    key=key, hide_index=True, width="stretch",
    column_config={
        "Game": st.column_config.TextColumn(disabled=True),
        "spread_home": st.column_config.NumberColumn("Pool home spread", step=0.5, format="%.1f"),
        "mkt_spread_home": st.column_config.NumberColumn("Market", step=0.5, format="%.1f"),
        "total": st.column_config.NumberColumn("Pool total", step=0.5, format="%.1f"),
        "mkt_total": st.column_config.NumberColumn("Market", step=0.5, format="%.1f"),
        "ml_home": st.column_config.NumberColumn("Pool home ML", step=5, format="%d"),
        "mkt_ml_home": st.column_config.NumberColumn("Market", step=5, format="%d"),
        "ml_away": st.column_config.NumberColumn("Pool away ML", step=5, format="%d"),
        "mkt_ml_away": st.column_config.NumberColumn("Market", step=5, format="%d"),
    })
lines = seed_lines.copy()
for c in ("spread_home", "total", "ml_home", "ml_away",
          "mkt_spread_home", "mkt_total", "mkt_ml_home", "mkt_ml_away"):
    lines[c] = edited[c].values
problems = P.check_lines(lines)
if not problems.empty:
    st.error("These pool lines look mistyped — a flipped spread sign shows up as a huge edge, "
             "not as an obvious error. Fix them above before trusting the card.")
    st.dataframe(problems, hide_index=True, width="stretch")

hist = get_history() if source == "market" else None
n_number_edges = int(((lines["spread_home"] != lines["mkt_spread_home"])
                      | (lines["total"] != lines["mkt_total"])).sum())
if source == "market":
    st.caption(f"{n_number_edges} of {len(lines)} games have a pool number different from the "
               "market's." + (" Edit the pool columns, or fetch the market, to find them." if n_number_edges == 0 else ""))

# --- which games carry the totals --------------------------------------------
n_required = len(games) + 3
n_ou = max(int(n_slots) - n_required, 0)
st.subheader(f"Totals: {n_ou} slot{'s' if n_ou != 1 else ''} this week")
prelim = P.build_card(sims, lines, mode=mode_key, juice=int(juice), teaser_pts=float(teaser_pts),
                      teaser_odds=int(teaser_odds),
                      parlay_pricing="true" if parlay_pricing.startswith("true") else "fixed",
                      parlay_odds=int(parlay_odds), n_slots=int(n_slots),
                      exclude_played=exclude_played, market_weight=float(market_w),
                      source=source, hist=hist)
auto_ou = prelim["card"].loc[prelim["card"]["slot"] == "Total", "game_id"].tolist()
game_opts = dict(zip(lines["Game"], lines["game_id"]))
saved_ou = set(saved_pool.loc[(saved_pool["season"] == season_of_week) & (saved_pool["week"] == int(week))
                              & (saved_pool["ou_in_pool"] == True), "game_id"]) if n_saved else set()
ou_pick = st.multiselect(
    "Games whose over/unders are in the pool", list(game_opts.keys()),
    default=[g for g, gid in game_opts.items() if gid in (saved_ou or auto_ou)],
    help="The pool names these; the model then picks over or under on each. "
         "Pre-filled from the saved pool lines, else with the games where the model sees the most value.")
ou_games = [game_opts[g] for g in ou_pick]
save_clicked = st.button("Save pool lines", help=f"Writes the pool columns, the totals games and the market "
                                                  f"as it stands now to `{P.POOL_FILE}` — so they survive a "
                                                  "restart and the opener-vs-closer log accumulates — and "
                                                  "locks the card as it stands (if no game has kicked off).")
if save_clicked:
    P.save_pool_lines(lines, season_of_week, int(week), ou_games)
    st.success(f"Saved {len(lines)} games for week {week}.")

res = P.build_card(sims, lines, mode=mode_key, juice=int(juice), teaser_pts=float(teaser_pts),
                   teaser_odds=int(teaser_odds),
                   parlay_pricing="true" if parlay_pricing.startswith("true") else "fixed",
                   parlay_odds=int(parlay_odds), n_slots=int(n_slots),
                   exclude_played=exclude_played, market_weight=float(market_w),
                   ou_games=ou_games, source=source, hist=hist)
card = res["card"]
for n in res["notes"]:
    st.warning(n)

# --- lock the card before kickoff, so results can be read against it -----------
week_games = sched[sched["week"] == int(week)]
any_played = bool(week_games["played"].any())
locked_card, locked_games = P.load_locked(season_of_week, int(week))
played_ids = set(week_games.loc[week_games["played"], "game_id"])
relock = st.button("Lock the card as it stands", help="Writes this card (and the model's margin and total "
                   "for every game) to `pickem/cards.csv` so this week's results are graded against it. Games "
                   "already played are left out. The first view of a week locks automatically; use this to "
                   "re-lock after changing the settings, before kickoff.")
if locked_card is None or relock or (save_clicked and not any_played):
    # the first view of a week (or a save before kickoff, or the button): lock
    # the card as it stands; games already played are left out, so a mid-week
    # lock covers only what is still to come
    P.lock_card(card, res["candidates"], season_of_week, int(week), mode_key, source, played=played_ids)
    locked_card, locked_games = P.load_locked(season_of_week, int(week))
    if relock:
        st.success(f"Locked {len(locked_card)} slots for the {len(locked_games)} games still to play.")

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
fmt = {"P(win)": "{:.1%}", "Push": "{:.1%}", "Price implies": "{:.1%}",
       "Edge": "{:+.1%}", "Exp. pts": "{:.1f}", "Number vs market": "{:+.1f}"}
st.dataframe(disp.style.format({k: v for k, v in fmt.items() if k in disp.columns}),
             hide_index=True, width="stretch", height=min(760, 38 * len(disp) + 40))
if source == "market":
    st.caption("**Number vs market** is how many points better than the market's line the pool "
               "is giving you on that side (a 6-point teaser shows the teased line's chance "
               "directly). **P(win)** comes from history at the market's spread; **Edge** is "
               "P(win) minus what the price implies. With the pool at the market's numbers, "
               "sides sit at ~50% and only the teaser legs that cross 3 and 7 carry an edge.")
if mode_key == "odds":
    st.caption("Odds-weighted scoring puts long shots and parlays at the top because their "
               "payout is large — that is the highest **expected** return, but it is also "
               "the highest variance. The Model P column is the chance each slot pays at all.")

# --- this week's results against the locked card -----------------------------------
if any_played:
    st.subheader(f"Week {week} so far: results against the locked card")
    if locked_card is None or locked_card.empty:
        st.info("Nothing locked for this week: every game had been played when it was first viewed.")
    else:
        graded = P.grade_card(locked_card, P.week_scores(week_games), mode_key)   # the pool's scoring, as set above
        settled = graded[graded["result"] != "open"]
        wins = int((settled["result"] == "win").sum()); pushes = int((settled["result"] == "push").sum())
        exp_settled = float(settled["exp_points"].sum()); exp_all = float(graded["exp_points"].sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Points so far", f"{graded['points'].sum():.1f}",
                  help=f"From the {len(settled)} settled slots; the locked card expected {exp_settled:.1f} from them.")
        m2.metric("Expected from settled slots", f"{exp_settled:.1f}")
        m3.metric("Picks hit", f"{wins} of {len(settled) - pushes}" + (f" ({pushes} push)" if pushes else ""))
        locked_when = pd.to_datetime(locked_card["locked_at"].iloc[0], errors="coerce", utc=True)
        locked_txt = locked_when.strftime("%d %b %H:%M") if pd.notna(locked_when) else "—"
        m4.metric("Card total expected", f"{exp_all:.1f}", help=f"All {len(graded)} slots, as locked at {locked_txt} UTC.")
        n_lock = len(locked_games) if locked_games is not None else 0
        if n_lock < len(week_games):
            st.caption(f"Locked at {locked_txt} UTC "
                       f"with {len(week_games) - n_lock} game{'s' if len(week_games) - n_lock != 1 else ''} already "
                       "played and left out; the card and the comparison cover the rest of the week.")
        show = pd.DataFrame({"Conf": graded["confidence"], "Slot": graded["slot"], "Pick": graded["text"],
                             "P(win)": graded["p"], "Exp. pts": graded["exp_points"],
                             "Result": graded["result"].map({"win": "✅ win", "loss": "❌ loss", "push": "push", "open": "—"}),
                             "Points": np.where(graded["result"] == "open", np.nan, graded["points"])})
        st.dataframe(show.style.format({"P(win)": "{:.1%}", "Exp. pts": "{:.1f}", "Points": "{:.1f}"}, na_rep="—"),
                     hide_index=True, width="stretch", height=min(600, 38 * len(show) + 40))
        gr = P.game_results(locked_games, week_games) if locked_games is not None else pd.DataFrame()
        if not gr.empty:
            st.markdown("**The model against the scores, game by game**")
            gt = pd.DataFrame({"Game": gr["away"] + " @ " + gr["home"], "Score": gr["score"],
                               "Model margin": gr["model_margin"], "Line": gr["line_margin"], "Actual margin": gr["actual_margin"],
                               "Model total": gr["model_total"], "Line total": gr["line_total"], "Actual total": gr["actual_total"],
                               "Model on the right side": np.where(gr["model_side_right"], "✅", "❌")})
            st.dataframe(gt.style.format({"Model margin": "{:+.1f}", "Line": "{:+.1f}", "Actual margin": "{:+.0f}",
                                          "Model total": "{:.1f}", "Line total": "{:.1f}", "Actual total": "{:.0f}"}),
                         hide_index=True, width="stretch")
            n = len(gr)
            st.caption(f"{n} game{'s' if n != 1 else ''} played. Margin error — model {gr['model_margin_err'].abs().mean():.1f}, "
                       f"line {gr['line_margin_err'].abs().mean():.1f}; total error — model {gr['model_total_err'].abs().mean():.1f}, "
                       f"line {gr['line_total_err'].abs().mean():.1f}. The model sat on the right side of the line in "
                       f"{int(gr['model_side_right'].sum())} of {n}. Margins are home minus away.")
    # the season so far, from every locked week
    allc = P.load_locked_all()
    allc = allc[(allc["season"] == season_of_week) & (allc["week"] < int(week))] if len(allc) else allc
    if len(allc):
        rows = []
        for w_, cw in allc.groupby("week"):
            cw = cw.copy(); cw["legs"] = [__import__("json").loads(l) if isinstance(l, str) and l.startswith("[") else None for l in cw["legs"]]
            g_ = P.grade_card(cw, P.week_scores(sched[sched["week"] == int(w_)]), mode_key)
            rows.append(dict(Week=int(w_), Slots=len(g_), Hit=int((g_["result"] == "win").sum()),
                             Expected=float(g_["exp_points"].sum()), Realised=float(g_["points"].sum())))
        st.markdown("**Earlier weeks, locked cards**")
        st.dataframe(pd.DataFrame(rows).style.format({"Expected": "{:.1f}", "Realised": "{:.1f}"}), hide_index=True, width="stretch")

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
    "Every played week is replayed **honestly** with the probability source chosen above. "
    "Engine: ratings, home field, availability and calibration refit on games before that "
    "week only (Normal, sd 12.8 on the margin, 13.4 on the total). Market + history: closing "
    "lines and the empirical distributions from seasons before that one. The pool's lines are "
    "the closing lines here — history has no openers — so **no number edge is available in "
    "the replay**; it measures calibration and slot selection only. Totals slots use the "
    "model's best games. Pushes score zero; a parlay with a pushed leg counts as lost.")
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
    hw, hp = P.card_history(bt, hsched, market_weight=float(market_w), source=source,
                            hist=get_history() if source == "market" else None, **kw)
    mw, _ = P.card_history(bt, hsched, market_weight=1.0, source="engine", **kw)
    if hw.empty:
        st.info("No played weeks to replay yet.")
    else:
        pts, exp, mkt = hw["points"].sum(), hw["expected"].sum(), hw["market_expected"].sum()
        mpts = mw["points"].sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Points scored", f"{pts:,.0f}", delta=f"{pts - exp:+,.0f} vs expected",
                  delta_color="normal", help=f"The card expected {exp:,.0f}; at pure market "
                                             f"prices it would have expected {mkt:,.0f}.")
        c2.metric("Engine card at 100% market lean", f"{mpts:,.0f}",
                  delta=f"{pts - mpts:+,.0f} vs it", delta_color="normal",
                  help="The engine's shape centred on the closing line — the old page's best "
                       "case. In the replay the pool's lines ARE the closing lines, so no "
                       "number edge exists; the difference is calibration and teaser selection.")
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
