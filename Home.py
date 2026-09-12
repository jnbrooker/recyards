"""
Home.py — landing page for the NFL model suite (multi-page Streamlit app).

Run the whole suite with:
    streamlit run Home.py

Each stat lives on its own page (see the sidebar). They all share the data and
math layer in `nflsim/`, and they all feed the forthcoming Game Simulation page.
"""

import streamlit as st

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
