# NFL Monte Carlo Model Suite

A self-contained NFL game and player-prop model, as a multi-page Streamlit app.
Pick a game and it simulates it thousands of times from distributions fitted to
real game-to-game data — pace, drives, who is on the field, who gets the ball,
what each touch is worth — and reports the score distribution, win
probability, a projected box score, and the chance of any player beating a
line. Nothing anchors to Vegas; the closing line is shown beside the model
only as a comparator.

Every model in it is scored **out of sample** by its own backtest harness,
and every constant that could be fitted was fitted on those residuals. The
numbers below are what the harness says, not what the fit said.

## Pages

| Page | What it does |
|---|---|
| Receiving Yards | targets → catches → yards (completed air yards + YAC), vs a pass defense |
| Rushing Yards | carries → stuffed / normal / explosive runs, before and after contact, vs six run-defense factors |
| Touchdowns | rushing + receiving; TD rate = goal-line role × conversion, scoring scales with volume |
| QB Sacks | sacks taken per dropback vs the opponent's pass rush, scaled by time to throw |
| Interceptions | INTs thrown per attempt, heavily regressed, vs the opponent's secondary |
| Team Strength | opponent-adjusted points per drive, pace, home field — the base of the game model |
| Game Simulation | a real fixture: score distribution, win probability, box score, availability, wind |
| Fantasy Projections | the whole slate scored per simulation (PPR / half / standard), floor and ceiling |
| Pick'em | a 20-slot confidence card graded from the simulated margins and totals |
| Backtest | every model scored week by week out of sample, vs the closing line and a trailing average |
| Prop Evaluation | this week's player-prop lines (The Odds API, on request) beside the model's frozen projection, settled automatically, graded against the book |

Pages 1–5 pick players from the **live depth chart** with injury tags; a
player's usage share blends his own history with his slot's prior (depth-chart
rank averaged with a snap-count prior). Each has a **Game / Season** view:
*Game* picks a fixture (week → game → player) and takes the opponent, home
field, availability, wind and the pre-game script from the engine; *Season* is
the player's typical game against any defense.

**Player props** need a line source the free data does not have. Put an
Odds API key (the-odds-api.com, free tier) in `.streamlit/secrets.toml` as
`ODDS_API_KEY = "..."`; the Prop Evaluation page fetches this week's lines
only when asked (~32 credits for two markets across a slate), freezes the
model's projection at that moment, fills in the actuals once games are
played, and grades mean or median against the line and P(over) against the
book's vig-free probability. The ledger is `props/ledger.csv`.

## How the game model works

1. **Team strength** (`nflsim/teams.py`) — offensive and defensive points per
   drive, solved jointly so a schedule of tough defenses is not held against an
   offense, shrunk toward league by the drives behind each rating, recent
   games weighted more. Home field and the return/safety scoring residual are
   measured, not assumed.
2. **Availability** (`nflsim/availability.py`) — the ratings describe a
   *unit*; this layer asks who is actually playing. *QB familiarity* (the
   starter's share of the dropbacks behind the rating, refined by his
   efficiency vs the incumbents') and *defensive starters ruled out*
   (snap-weighted, from the defensive depth chart as of kickoff) shift the
   expected margin with coefficients fitted on out-of-sample residuals. The
   offensive line is indexed and shown but not priced — the fitted effect is
   within noise.
3. **Drive engine** (`nflsim/game.py`) — one shared pace draw, then drives
   resolved in sequence for both teams with a lead-dependent scoring rate
   (leading teams sit on the ball, trailing teams press), calibrated so the
   simulated spread equals the real conditional spread. Expected points are
   pace-invariant, as the data says they are.
4. **Allocation** (`nflsim/roster.py`) — Dirichlet shares spread targets and
   carries across the depth chart; each player's own priors fill in catches
   and yards; team touchdowns are split by goal-line role. Every box-score
   identity holds in every simulation: targets sum to attempts, player
   touchdowns to team touchdowns, passing yards to the receivers' yards.

Recency: every feed carries one weight column — a season curve times a
per-team, bye-aware game decay — and every rate, share and volume is a
weighted estimate. The sidebar's *How much to trust this season* control sets
both.

## Out-of-sample performance (2025 season, fitted only on games before each week)

**Team layer** — 272 games:

| | margin RMSE | winners | log-loss |
|---|---|---|---|
| Model | 12.79 | 62.0% | 0.644 |
| Closing line | 12.27 | 65.3% | 0.610 |

2024 (fitted on 2022–23 + prior weeks): model 12.71 / 68.0% vs the line's
12.61 / 71.3%. Win probabilities are calibrated; margins regress on
predictions with slope ≈ 1.

**Player models** vs a trailing weighted average of the stat (the baseline any
model has to beat):

| stat | MAE | naive MAE | 10–90 band covers | notes |
|---|---|---|---|---|
| Receiving yards | 22.8 | 23.1 | 87% | bias 0.0 in every quintile |
| Rushing yards | 24.2 | 24.6 | 83% | |
| Touchdowns | — | — | — | anytime-TD Brier 0.207 vs base rate 0.217 |

Sacks and interceptions sit at their naive baselines. Everything above comes
from the Backtest page (`nflsim/backtest.py`), which also shows the largest
misses so the model can be argued with.

## Setup & run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run Home.py
```

Data is pulled from nflverse releases (weekly player stats, play-by-play,
depth charts, injuries, snap counts, PFR advanced rushing, NGS, schedules)
and cached; a season's play-by-play is a few tens of MB on first load. The
caches expire every six hours, so a long-running app picks up a new week on
its own.

Command-line checks, no dashboard:

```bash
python -m nflsim.game            # simulate one game and print the box score
python -m nflsim.backtest 2025   # score the season out of sample
python -m nflsim.calibrate 2024 2025          # refit report for the fitted constants
python -m nflsim.calibrate 2024 2025 --write  # ...and update them in place
```

## Maintaining it

The fitted constants rest on two seasons and should be refitted each
off-season with `nflsim.calibrate`: the availability coefficients, the
ratings' shrinkage (read off the calibration slope), the spread the engine
reproduces, and the wind coefficient. The report prints current beside
refitted with t-stats; `--write` changes only what is clearly different from
zero. Then re-run the harness and commit.

`ROADMAP.md` records every design decision and every finding, including the
ones that did not pan out.

## Caveats

- The QB index measures "did not take the dropbacks behind this rating"; a
  proven starter who changed teams is charged like a backup until he has
  played, softened by his efficiency gap.
- nflverse only records wind after the game; for an upcoming game, type the
  forecast.
- Depth charts and injury reports cannot be replayed for past weeks, so the
  player backtests use the history-only path.
- For research and entertainment.
