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
    "- *QB Sacks / Interceptions* — coming next (offense-side rates).\n"
    "- *Game Simulation* — upload/auto-pull two depth charts → simulated box "
    "score, score distribution and win probability."
)

st.subheader("How it works")
st.write(
    "Everything is self-contained: priors come from nflverse data, and the game "
    "model forms its own view of team strength rather than anchoring to Vegas. "
    "See `ROADMAP.md` for the full build plan."
)
st.caption("For research and entertainment.")
