# Receiving Yards Monte Carlo Model

A dashboard that predicts an NFL player's receiving yards for a single game by
simulating it thousands of times, then shows the chance of beating a prop line
and the full distribution of outcomes.

## What it does

Pick a player, pick the opposing defense, and set a yards line. The model
simulates the game ~20,000 times and reports:

- the **percentage chance of going over** the line (with fair betting odds),
- an interactive **graph of the probability of every yardage outcome**,
- projected mean / median yards, catches and targets,
- outcome percentiles (floor, median, ceiling),
- a plain-language read on the opposing defense.

## How the model works

Each simulated game is built one step at a time, drawing from distributions fit
to the player's real game-to-game numbers (so **variance** is baked in, not just
averages):

1. **Team pass volume** — how many targets the player's offense throws (Normal
   around the team's per-game average).
2. **Target share** — the player's slice of those targets (Beta fit to their
   mean & variance). Targets = volume × share.
3. **Catch rate** — how many targets become catches (Beta fit to mean &
   variance → Binomial).
4. **Yards per catch** — driven by **average depth of target (aDOT)** plus
   **yards-after-catch**, each catch drawn from a right-skewed Gamma so the odd
   big play shows up like it does in real life.

**Defense adjustment** (toggleable): for the chosen opponent it compares what
that defense allows *to that position* — catch rate, aDOT, and yards per target —
against the league average, and nudges the player's depth, catch rate and
efficiency accordingly. Because one season of defense-vs-position data is a
small sample, the adjustment is **shrunk toward league average** (strength is a
slider; default 0.6).

Data comes from **nflverse** via `nfl_data_py` (regular-season only), and more
recent seasons are weighted more heavily when building priors.

## Files

- `app.py` — the Streamlit dashboard (run this).
- `model.py` — all the data + simulation logic (importable / runnable on its own).
- `requirements.txt` — dependencies.

## Setup & run

```bash
cd recyardsmodel
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

streamlit run app.py
```

It opens in your browser. The first data load downloads a few MB from nflverse
and is cached, so later interactions are fast.

Quick command-line sanity check without the dashboard:

```bash
python model.py
```

## Notes & caveats

- Uses **season-level** player and defense numbers — it does not yet account for
  injuries, weather, a new team/role mid-season, or specific coverage schemes
  beyond what shows up in yards/aDOT allowed.
- Defense splits vs a single position over one season are noisy; that's why the
  adjustment is shrunk. Turn the strength down (or off) if you want the player's
  pure baseline.
- For research and entertainment.
