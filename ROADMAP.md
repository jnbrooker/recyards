# NFL Game Model — Build Roadmap

A plan to grow the receiving-yards Monte Carlo model (`recyards`) into a full,
**self-contained** game simulator: upload two depth charts, and it simulates the
game thousands of times to produce a projected **offensive box score**, a
**score distribution**, and a **win-probability** read — using the same
"fit priors with variance → simulate the event chain → adjust for the
opponent with shrinkage" pattern the receiving model already uses.

*Self-contained* means the model forms its own opinion from team-efficiency data
and never anchors to a Vegas total or spread.

---

## 1. Design principles (inherited from `recyards`)

Every stat module follows the existing template:

1. **Fit priors with variance**, not just averages — draw each step of a game
   from a distribution fit to the player's real game-to-game numbers.
2. **Right distribution per step**: Normal for volume, Beta for rates,
   Poisson / Negative Binomial for counts, Gamma (or a skewed/mixture form) for
   yards.
3. **Opponent adjustment with shrinkage** — compare what a defense allows vs
   league average, shrink toward league because one season of splits is noisy.
4. **Recent seasons weighted more heavily** when building priors.

The game model adds two ideas on top:

5. **Consistency by construction** — player shares sum to team totals (Dirichlet),
   and team TDs are split across players (multinomial) so the box score always
   reconciles to the score.
6. **A shared game-script state per simulation** links volume, pass/run split,
   and points, so a team that's simulated to fall behind throws more, and a team
   ahead runs more.

---

## 2. Scope (locked)

- **Offense-only box score.** No defensive-player stats (tackles, defender INTs
  dropped — noisiest data, least value for predicting the score).
- **Sacks and INTs are viewed from the offense/QB side**: sacks *taken*, INTs
  *thrown*. These use clean offensive feeds, not defensive box scores.
- **Passing yards / passing TDs are emergent, not a separate model** — team
  passing yards ≈ sum of the receivers' simulated yards; QB passing TDs = sum of
  allocated receiving TDs. The QB line is *assembled* from the receiver
  allocation plus the sack/INT models, so it can never contradict the receivers.
- **Self-contained scoring** — team expected points come from opponent-adjusted
  efficiency ratings, never from Vegas.

---

## 3. The stat modules

### 3.1 Receiving yards — DONE (`recyards`)
Team pass volume (Normal) → target share (Beta) → catches (Beta→Binomial) →
per-catch yards (aDOT + YAC, Gamma). Defense adjustment on depth / catch rate /
efficiency, shrunk. This is the reference implementation.

### 3.2 Rushing yards — *Phase 1, direct clone*
Team rush volume (Normal; more game-script sensitive than passing — leading
teams run more) → carry share (Beta) → per-carry yards. **Difference from
receiving:** YPC is more skewed (breakaway runs) and can go **negative** (TFLs),
so model per-carry yards as a shifted / mixture distribution rather than a plain
Gamma. Defense adjustment: rush yards allowed per carry, yards before contact.

### 3.3 Player touchdowns
Model **expected TDs = opportunities × conversion rate**, then draw the count.
The distribution (Poisson is fine for a single game; NB if overdispersed) is not
where the accuracy lives — the expected-TD estimate is.
- **Rushing TDs:** goal-line/short-yardage carries × conversion. Goal-line
  *role* dominates — a backup short-yardage back scores with almost no yards.
- **Receiving TDs:** red-zone target share × TD/target.
- **In the game model, do NOT simulate each player's TDs independently.**
  Simulate the team's total TDs (tied to the score), then split across players
  via a multinomial over TD-share weights. This is what keeps box-score TDs
  summing to the final score.

### 3.4 QB sacks (taken)
Model the **sack rate per dropback**, not raw counts. Expected sacks =
dropbacks × sack_rate, where sack_rate combines the offense's sacks-allowed rate
and the defense's pressure/sack rate (log-odds / odds-ratio combination), scaled
by **NGS `avg_time_to_throw`** — quick-release offenses take fewer sacks. Counts
are low and overdispersed → **Negative Binomial**. Sacks cost yards and
dropbacks in the game engine.

### 3.5 QB / team INTs (thrown)
A **turnover** output, not a defender stat. Expected INTs = attempts ×
int_rate, with the rate **heavily regressed** toward league/positional mean
(INT rate is one of the noisiest stats in football). Poisson/NB count. Feeds the
turnover mechanism in the drive engine (ends drives, flips field position).

---

## 4. The self-contained scoring engine (the new work)

### 4.1 Team-strength layer (the one genuinely new module)
Because we don't anchor to Vegas, we need our own view of team strength:
**opponent-adjusted offensive and defensive efficiency** — points and yards per
drive, adjusted for the quality of opponents each team actually faced (iterative
SRS-style averaging, or a ridge regression of drive outcomes on offense/defense
indicators). Output per matchup: expected points for each offense vs the other
defense, plus **pace** (drives per game).

### 4.2 Drive-based game engine (top-down hierarchical)
Per simulation:
1. Draw a shared **game-script state**: each team's drives (pace) and pass/run
   split, correlated with the running score.
2. Simulate **drive outcomes** (TD / FG / punt / turnover / downs) whose means
   are set by the matchup efficiency from 4.1, with turnovers informed by the
   INT model (3.5) and sacks (3.4) suppressing drive success.
3. **Allocate to the depth chart:** Dirichlet target/carry shares split volume
   across listed players; run each player's per-stat priors (3.1–3.3) to fill
   yards and catches; multinomial-split the team's scoring drives' TDs across
   players by role weight.
4. **Score:** TDs + FGs → points per team → final score for this sim.

### 4.3 The core engineering tension to design around
Two routes to TDs must be reconciled: **top-down** (efficiency → team points →
implied TDs) and **bottom-up** (player red-zone → TDs → sum). Recommended split:
the drive engine decides *how many* scoring drives and TD-vs-FG; the player
models decide *who* scores and the yardage; calibrate so aggregate TDs match the
efficiency ratings. Getting this reconciliation right is the main challenge of
the whole build.

### 4.4 Depth-chart → roles
A name isn't a workload. Uploaded depth charts must map to target/carry/red-zone
shares via snap-share and role priors, or "WR1" and "WR3" look identical to the
model. Decide the input format (positional slots + expected snap%, or infer from
each player's own history) early.

---

## 5. Data sources & the one spike to do first

- **Rushing / receiving / passing weekly** — already used (nflverse
  `stats_player_week`).
- **NGS** — `avg_time_to_throw` for sacks; rush/receiving efficiency (nflverse
  NGS releases).
- **Play-by-play** — pace, drives, red-zone, game script (nflverse pbp).

**Spike before Phase 3+:** confirm NGS time-to-throw and pbp drive data load
cleanly and cover the seasons you want. This derisks sacks and the entire game
engine in one short check.

---

## 6. Build order

| Phase | Deliverable | Notes |
|------|-------------|-------|
| 1 | Rushing yards page | Direct clone of `recyards`; proves the template generalizes. |
| 2 | Data spike | NGS time-to-throw + pbp drives/red-zone/pace load cleanly. |
| 3 | Player TDs page | Opportunity × conversion; goal-line role. |
| 4 | QB sacks + INTs pages | Offense-side rates; NGS time-to-throw. |
| 5 | Team-strength layer + drive-based game engine | The reconciliation work (§4). |
| 6 | Game dashboard | Simulated offensive box score, score distribution, win probability. |

Keep it as a **multi-page Streamlit app** — `recyards` becomes one page among
several, sharing a common data/model utility layer.

---

## 7. Open questions

1. **Project structure** — extend the existing `recyards` folder into a
   multi-page app, or start a fresh project folder that imports the receiving
   model? (Affects Phase 1 file layout.)
2. **Depth-chart input format** for the game model (§4.4) — positional slots
   with expected snap%, or infer roles from each player's own history?
3. **Seasons window** — how many seasons of priors, and how hard to weight the
   current one for in-season form.

---

*Decisions locked so far: offense-only box score · QB-perspective sacks & INTs ·
fully self-contained scoring (no Vegas anchor).*
