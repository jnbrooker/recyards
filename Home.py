"""
Home.py — landing page for the NFL model suite (multi-page Streamlit app).

Run the whole suite with:
    streamlit run Home.py

Each stat lives on its own page (see the sidebar). They all share the data and
math layer in `nflsim/`, and they all feed the forthcoming Game Simulation page.
"""

import streamlit as st

from nflsim import ui as UI

st.set_page_config(page_title="NFL Model Suite", page_icon="🏈", layout="wide")

st.title("🏈 NFL Monte Carlo Model Suite")
st.write(
    "Pick a stat from the sidebar. Each page simulates a single game thousands "
    "of times from distributions fit to a player's real game-to-game numbers, "
    "adjusts for the opponent, and shows the full distribution of outcomes plus "
    "the chance of beating a line."
)

st.subheader("Pages")
st.markdown(
    "- **Receiving Yards** — targets → catches → yards, vs a defense.\n"
    "- **Rushing Yards** — carries → yards before/after contact + broken tackles, vs a run defense.\n"
    "- **Touchdowns** — rushing + receiving, opportunity × conversion (goal-line role).\n"
    "- **QB Sacks** — sacks taken per dropback vs the opponent's pass rush, scaled by NGS time to throw.\n"
    "- **Interceptions** — INTs thrown per attempt, heavily regressed, vs the opponent's secondary.\n"
    "- **Team Strength** — opponent-adjusted points per drive, pace and home field; the base the game model is built on.\n"
    "- **Game Simulation** — pick a game from the schedule; simulated box score, score distribution and win probability, with the closing line beside it for comparison.\n"
    "- **Fantasy Projections** — every player on the week's slate scored per simulation (PPR / half / standard, editable), with floor, ceiling and a breakdown of where the points come from.\n"
    "- **Pick'em Card** — a 20-slot confidence card for the week (ATS / underdog ML per game, three 3-team combos, the pool's totals), graded against editable lines and ranked by expected return.\n"
    "- **Backtest** — every model scored out of sample, week by week, against the closing line and a trailing average. The only honest way to tune anything here.\n"
)

# --- warm-up: compute this week for the default settings, once per process ------
st.subheader("Getting this week ready")
c1, c2 = st.columns([1, 3])
with c1:
    both = st.toggle("Include the play-level engine", value=True,
                     help="Adds the ten-season play tables and a full slate on the play "
                          "engine — the slow part (10-20 minutes). Off = drive engine only "
                          "(2-3 minutes).")
    run = st.button("Pre-compute this week", type="primary",
                    help="Runs every page's default computation once so the pages open "
                         "instantly. Results live in the app's cache for six hours (or until "
                         "the app restarts).")
with c2:
    st.caption("Runs automatically the first time the app opens, and on demand after that. "
               "Every page simulates and caches; this does all of it up front for the default settings — priors, rosters, this week's slate on "
               "both engines, the pick'em replay and the game page's first fixture — so you can "
               "move between pages without waiting. Change a setting on a page and only that "
               "computation reruns. Leave this tab open while it works.")
@st.cache_resource
def _warm_state():
    return {"done": False}


auto = _warm_state()
if run or not auto["done"]:
    auto["done"] = True                      # once per app process; the button re-runs it
    status = st.status("Warming up…", expanded=True)
    log = UI.warm_up(progress=lambda label: status.write(f"• {label}"),
                     engines=("drive", "play") if both else ("drive",))
    status.update(label="Ready — every page now opens from cache.", state="complete")
    st.session_state["warmed"] = True
    with st.expander("What was built"):
        st.write("\n".join(f"- {l}" for l in log))
elif st.session_state.get("warmed"):
    st.success("This week is pre-computed. Pages open from cache.")

st.subheader("How it works")
st.write(
    "Everything is self-contained: priors come from nflverse data, and the game "
    "model forms its own view of team strength rather than anchoring to Vegas. "
    "Recent games count more than old ones (the sidebar's recency control), the "
    "engine knows who is actually playing (QB familiarity, defensive starters "
    "ruled out) and every constant that could be fitted was fitted on "
    "out-of-sample residuals. See `README.md` for the model and `ROADMAP.md` for "
    "how it got here."
)
st.caption("For research and entertainment.")
