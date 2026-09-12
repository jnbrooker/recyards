"""Backtest — out-of-sample scoring of the team layer and the player models
(roadmap §8.1). Every week is predicted with models fitted only on games played
before it, then compared with what happened and with the closing line."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from nflsim import data as D, backtest as B

st.set_page_config(page_title="Backtest", page_icon="🏈", layout="wide")


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner=False)
def run_backtest(score_seasons, preset_names, stats, n_prior, n_sims, weeks, availability,
                 _progress=None):
    presets = {n: D.RECENCY_PRESETS[n] for n in preset_names}
    return B.run(list(score_seasons), presets, stats=list(stats), n_prior=int(n_prior),
                 n_sims=int(n_sims), weeks=list(weeks) or None, progress=_progress,
                 availability=bool(availability))


# --- sidebar ----------------------------------------------------------------
st.sidebar.header("Setup")
options, defaults = D.season_choices()
score_seasons = st.sidebar.multiselect(
    "Seasons to score", options, default=defaults[:2],
    help="Each week of these seasons is predicted from games before it.")
n_prior = st.sidebar.slider("Prior seasons in the fit window", 1, 3, 2,
                            help="How many earlier seasons the models see, matching the "
                                 "pages' default window of three.")
preset_names = st.sidebar.multiselect("Recency presets to compare", list(D.RECENCY_PRESETS),
                                      default=list(D.RECENCY_PRESETS))
stats = st.sidebar.multiselect(
    "Player stats", list(B.PLAYER_STATS), default=["rec_yards", "rush_yards", "tds"],
    format_func=lambda s: B.PLAYER_STATS[s]["label"])
n_sims = st.sidebar.select_slider("Simulations per player-game", [1000, 2000, 4000, 8000],
                                  value=4000)
availability = st.sidebar.toggle(
    "Apply availability (QB / defensive starters)", value=True,
    help="Shift each game's margin by the QB-familiarity and missing-defensive-"
         "starter indices as they were knowable before kickoff (roadmap §8.5). "
         "Adds a 'no availability' row per preset for comparison.")
week_range = st.sidebar.slider("Weeks", 1, 18, (1, 18))
weeks = tuple(range(week_range[0], week_range[1] + 1)) if week_range != (1, 18) else ()
go_btn = st.sidebar.button("Run backtest", type="primary",
                           disabled=not score_seasons or not preset_names)

st.title("🏈 Backtest")
st.caption("Rolling out-of-sample scoring: fit on weeks before, predict the week, "
           "compare with the result and with the closing line. This is the only "
           "honest way to tune any constant in the model.")

if "bt_args" not in st.session_state and not go_btn:
    st.info("Pick the seasons and presets, then **Run backtest**. A season with three "
            "presets and three player stats takes a few minutes the first time.")
    st.stop()
if go_btn:
    st.session_state["bt_args"] = (tuple(sorted(int(s) for s in score_seasons)),
                                   tuple(preset_names), tuple(stats), int(n_prior),
                                   int(n_sims), weeks, bool(availability))
args = st.session_state["bt_args"]
bar = st.progress(0.0, text="Fitting week by week…")
res = run_backtest(*args, _progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
bar.empty()
seasons_txt = ", ".join(str(s) for s in args[0])

# --- team layer --------------------------------------------------------------
st.header("Team layer — margins, totals, winners")
team_cmp = B.compare(res, "team")
if team_cmp.empty:
    st.error("No games were scored — the play-by-play or schedule feed did not load.")
    st.stop()

n_games = int(team_cmp["games"].max())
st.caption(f"{n_games} games of {seasons_txt}, each predicted before kickoff from "
           f"{args[3]} prior season(s) plus the earlier weeks. Win probabilities use a "
           f"normal margin with sd {B.MARGIN_SD}.")
show = team_cmp[["margin_rmse", "margin_mae", "margin_bias", "su_acc", "log_loss",
                 "brier", "total_rmse", "total_bias", "ats_pct", "resid_sd"]].copy()
show.columns = ["Margin RMSE", "Margin MAE", "Margin bias", "Winner %", "Log-loss",
                "Brier", "Total RMSE", "Total bias", "ATS %", "Residual sd"]
st.dataframe(show.style.format({"Margin RMSE": "{:.2f}", "Margin MAE": "{:.2f}",
                                "Margin bias": "{:+.2f}", "Winner %": "{:.1%}",
                                "Log-loss": "{:.3f}", "Brier": "{:.3f}",
                                "Total RMSE": "{:.2f}", "Total bias": "{:+.2f}",
                                "ATS %": "{:.1%}", "Residual sd": "{:.2f}"}),
             width="stretch")
st.caption("Lower is better for RMSE / MAE / log-loss / Brier; bias is prediction minus "
           "actual (home margin, total). *ATS %* is how often backing the side the model "
           "likes more than the market covered the closing spread — 52.4% breaks even at "
           "-110. The **Closing line** row is the market scored on the same games.")

best = min((n for n in res["team"] if "(no availability)" not in n),
           key=lambda n: B.team_metrics(res["team"][n]).loc["Model", "margin_rmse"])
bt = res["team"][best]

c1, c2 = st.columns(2)
with c1:
    st.subheader(f"Calibration — {best}")
    cal = B.calibration(bt)
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#888"),
                    name="perfect")
    fig.add_scatter(x=cal["predicted"], y=cal["observed"], mode="markers+lines",
                    marker=dict(size=np.sqrt(cal["n"]) * 3, color="#2e7d5b"),
                    text=cal["n"], name="model",
                    hovertemplate="predicted %{x:.0%}<br>observed %{y:.0%}<br>%{text} games<extra></extra>")
    fig.update_layout(height=380, xaxis_title="Predicted P(home win)",
                      yaxis_title="Observed home-win rate", xaxis_tickformat=".0%",
                      yaxis_tickformat=".0%", showlegend=False,
                      margin=dict(t=20, b=50, l=60, r=20))
    st.plotly_chart(fig, width="stretch")
    st.caption("Points on the diagonal mean the win probabilities are honest. "
               "Marker size is the number of games in the bin.")
with c2:
    st.subheader("Margin RMSE by week")
    fig = go.Figure()
    for name, b in res["team"].items():
        wk = B.weekly(b)
        wk["x"] = wk["season"].astype(str) + " wk" + wk["week"].astype(str)
        fig.add_scatter(x=wk["x"], y=wk["model_rmse"], mode="lines+markers", name=name)
    wk = B.weekly(bt); wk["x"] = wk["season"].astype(str) + " wk" + wk["week"].astype(str)
    fig.add_scatter(x=wk["x"], y=wk["line_rmse"], mode="lines", name="Closing line",
                    line=dict(color="#333", dash="dot"))
    fig.update_layout(height=380, yaxis_title="Margin RMSE (points)",
                      margin=dict(t=20, b=80, l=60, r=20), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Early weeks lean on last season; the gap to the line is the cost of "
               "not knowing what the market knows (injuries, QB changes, weather).")

with st.expander("Every scored game"):
    view = bt[["season", "week", "home", "away", "pred_home", "pred_away", "actual_home",
               "actual_away", "pred_margin", "line_margin", "actual_margin",
               "pred_total", "line_total", "actual_total", "p_home"]].copy()
    view.columns = ["Season", "Week", "Home", "Away", "Model home", "Model away",
                    "Home pts", "Away pts", "Model margin", "Line", "Margin",
                    "Model total", "Line total", "Total", "P(home)"]
    st.dataframe(view.style.format(precision=1).format({"P(home)": "{:.0%}"}),
                 width="stretch", hide_index=True)

# --- player layer --------------------------------------------------------------
if res["player"]:
    st.header("Player layer — the single-stat models")
    pl = B.compare(res, "player")
    if not pl.empty:
        pl = pl.reset_index().rename(columns={"level_0": "Stat", "level_1": "Preset"})
        cols = ["Stat", "Preset", "n", "mae", "naive_mae", "rmse", "bias", "corr", "cover80",
                "over_median", "crps"]
        if "brier_ge1" in pl.columns:
            cols += ["brier_ge1", "base_rate_ge1"]
        view = pl[cols].copy()
        view.columns = ["Stat", "Preset", "n", "MAE", "Naive MAE", "RMSE", "Bias", "Corr",
                        "10–90 cover", "Over median", "CRPS"] + \
                       (["Brier ≥1", "Base rate ≥1"] if "brier_ge1" in pl.columns else [])
        fmt = {"MAE": "{:.2f}", "Naive MAE": "{:.2f}", "RMSE": "{:.2f}", "Bias": "{:+.2f}",
               "Corr": "{:.2f}", "10–90 cover": "{:.0%}", "Over median": "{:.0%}",
               "CRPS": "{:.2f}", "Brier ≥1": "{:.3f}", "Base rate ≥1": "{:.0%}"}
        st.dataframe(view.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)
        st.caption(
            "History-only path (no depth chart or injury report — those cannot be "
            "reconstructed for past weeks), players with 5+ prior games and enough "
            "expected volume for a prop to exist. **Naive MAE** is a trailing weighted "
            "average of the stat: the model must beat it. **10–90 cover** should be ~80% "
            "and **Over median** ~50% if the distributions are honest (count stats sit "
            "below 50% because the median is often 0). **CRPS** scores the whole "
            "distribution; **Brier ≥1** is the anytime-TD / at-least-one market against "
            "the base rate's own Brier of p(1−p).")

    with st.expander("Largest misses and best calls"):
        name = st.selectbox("Preset", list(res["player"]))
        pb = res["player"][name]
        stat = st.selectbox("Stat", sorted(pb["stat"].unique()),
                            format_func=lambda s: B.PLAYER_STATS[s]["label"])
        g = pb[pb["stat"] == stat].copy()
        g["err"] = g["pred_mean"] - g["actual"]
        cols = ["season", "week", "name", "team", "opp", "actual", "pred_mean",
                "pred_median", "p10", "p90", "naive", "err"]
        left, right = st.columns(2)
        with left:
            st.markdown("**Biggest over-projections**")
            st.dataframe(g.nlargest(12, "err")[cols].round(1), hide_index=True, width="stretch")
        with right:
            st.markdown("**Biggest under-projections**")
            st.dataframe(g.nsmallest(12, "err")[cols].round(1), hide_index=True, width="stretch")

st.caption("Nothing here feeds the models; it only scores them. For research/entertainment.")
