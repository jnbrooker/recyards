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

### 3.2 Rushing yards — DONE (extended beyond the direct clone)
Team rush volume (Normal; more game-script sensitive than passing — leading
teams run more) → carry share (Beta) → **each carry resolves into a stuffed /
normal / explosive outcome**, a mixture whose three bucket means/rates form a
decomposition of the player's real YPC (so the neutral-defense mean reproduces
his YPC exactly), giving the negative TFL tail and the breakaway right tail
without inflating the average.

**Defense is now modelled on six factors, each shrunk toward league average:**
- *front* — yards before contact allowed (PFR `r_ybc`)
- *tackling* — yards after contact allowed (PFR `r_yac`)
- *broken tackles* allowed (PFR `r_brk`)
- *stuff / TFL rate* allowed (pbp `r_stuff`) — drives the negative tail, which
  used to be a fixed shift constant
- *explosive-run rate* (10+ yds) allowed (pbp `r_expl`) — drives the right tail
- *overall efficiency* — success rate / EPA per rush allowed (pbp `r_eff`),
  applied softly because it overlaps the front/tackling factors

The stuff / explosive / efficiency factors pulled the Phase 2 pbp spike (§5)
forward. Any missing feed (a defense with no pbp history) degrades gracefully to
neutral factors. `nflsim/rushing.py` + `nflsim/data.py` (`load_pbp`,
`rush_defense_pbp`).

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

### 3.4 QB sacks (taken) — DONE (`nflsim/qb.py`, page 4)
Model the **sack rate per dropback**, not raw counts. Expected sacks =
dropbacks × sack_rate, where sack_rate combines the offense's sacks-allowed rate
and the defense's pressure/sack rate (log-odds / odds-ratio combination), scaled
by **NGS `avg_time_to_throw`** — quick-release offenses take fewer sacks. Counts
are low and overdispersed → **Negative Binomial**. Sacks cost yards and
dropbacks in the game engine.

### 3.5 QB / team INTs (thrown) — DONE (`nflsim/qb.py`, page 5)
A **turnover** output, not a defender stat. Expected INTs = attempts ×
int_rate, with the rate **heavily regressed** toward league/positional mean
(INT rate is one of the noisiest stats in football). Poisson/NB count. Feeds the
turnover mechanism in the drive engine (ends drives, flips field position).

---

## 4. The self-contained scoring engine (the new work)

### 4.1 Team-strength layer — DONE (`nflsim/teams.py`, page 6)
Because we don't anchor to Vegas, we need our own view of team strength:
**opponent-adjusted offensive and defensive efficiency**, built on a drive table
(`data.load_drives` — one row per `fixed_drive` with its result and points).
Offense and defense ratings are solved together by iterating the SRS idea on
points per drive (a team's offensive rating is what its drives produced after
subtracting the defensive ratings it actually faced, and vice versa), each rating
shrunk toward league by how many drives back it, recent seasons weighted more.

Per matchup it returns expected points per drive each way, **pace** (drives per
game, a property of the pairing), and a per-drive **outcome mix** (TD / FG /
turnover / nothing) that is *rescaled so its point value equals the efficiency
rating* — §4.3's reconciliation, done at the drive level.

Two things a drive-only model gets wrong unless they are handled explicitly, both
now measured from data rather than assumed:

- **Non-drive points.** Kick/punt return TDs and safeties belong to no drive, and
  a defense's own scores must not be credited to its offense. Drive points come
  to 20.7 of the real 23.0 points per team-game; the layer measures the residual
  against actual final scores and adds it back (20.67 offensive + 1.58 defensive
  + 0.69 return/safety = 22.94 vs 22.96 actual).
- **Home field**, fitted as the home margin the ratings don't already explain:
  **+1.97 points** over 2024–25.

*Validated on all 544 regular-season games of 2024–25:* margin correlation
**0.50**, margin RMSE **12.7**, straight-up winner **67.1%**, mean total 45.8 vs
45.9 actual, zero margin bias. The residual spread (sd 12.7 points) is the
target the drive engine's simulated score distribution has to reproduce.

### 4.2 Drive-based game engine — DONE (`nflsim/game.py`, page 7)
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

**One finding that changed the design.** Points per team-game turn out to be
essentially *independent* of how many drives a game has (correlation −0.04;
teams with 8 drives averaged 21.6 points, teams with 13 averaged 18.8). Extra
drives are extra three-and-outs, not extra scoring. Treating drives as a plain
multiplier on points — the obvious way to build this — inflated the score
distribution by ~25% and invented a positive correlation between the two teams'
scores that real games do not have. The engine therefore holds expected points
**pace-invariant** and scales *volume* by a measured elasticity of +0.34
instead. After that fix the simulated distribution matches reality:

| | real 2024–25 | engine |
|---|---|---|
| team points sd | 9.86 | 10.27 |
| corr(team A, team B) | −0.061 | −0.006 |
| total sd | 13.44 | 14.10 |
| margin sd | 14.29 | 14.19 |

Simulated means reproduce §4.1's analytic expected points to within 0.09 points.

### 4.3 The core engineering tension to design around
Two routes to TDs must be reconciled: **top-down** (efficiency → team points →
implied TDs) and **bottom-up** (player red-zone → TDs → sum). Recommended split:
the drive engine decides *how many* scoring drives and TD-vs-FG; the player
models decide *who* scores and the yardage; calibrate so aggregate TDs match the
efficiency ratings. Getting this reconciliation right is the main challenge of
the whole build.

### 4.4 Depth-chart → roles — DECIDED and DONE
A name isn't a workload. **Resolution: do both, and let the data pick.** The
depth chart (auto-pulled from nflverse, current season) decides *who is on the
field and in what slot*; the player's own history decides *what that slot is
worth*. Each share is a blend, weighted `games / (games + 10)`, of the player's
own target/carry share and the prior for his positional rank (WR1 0.22, WR2
0.16, TE1 0.17, RB1 0.50 of carries, …). Shares are then renormalised across the
roster, so a rookie WR1 inherits his slot's prior while a veteran WR3 who really
commands targets keeps his own number. The box score labels which applied.

Players ruled Out or Doubtful on the current season's latest injury report are
dropped and their share redistributed.

---

## 5. Data sources & the one spike to do first

- **Rushing / receiving / passing weekly** — already used (nflverse
  `stats_player_week`).
- **NGS** — `avg_time_to_throw` for sacks; rush/receiving efficiency (nflverse
  NGS releases).
- **Play-by-play** — pace, drives, red-zone, game script (nflverse pbp).

**Spike — DONE.** NGS time-to-throw and the pbp drive data both load and cover
2018→present. Two schema traps found and fixed, worth remembering:

- **NGS ships one all-seasons file per stat group** (`nextgen_stats/ngs_passing.parquet`),
  not one per season; season-level rows are the `week == 0` rows.
- **The weekly release renamed the passing columns**: `sacks` → `sacks_suffered`,
  `interceptions` → `passing_interceptions`, `sack_yards` → `sack_yards_lost`,
  and **`dropbacks` no longer exists** (derived as attempts + sacks, which is the
  conventional sack-rate denominator anyway). `data._ALIASES` normalises the old
  and new schemas to one canonical set — check it first if a passing model
  suddenly reads zeros or raises a `KeyError`.

One cached pbp read (`data.load_pbp_raw`) now serves both the run-defense factors
and the drive table, so a session downloads each season's play-by-play once.

---

## 6. Build order

| Phase | Deliverable | Notes |
|------|-------------|-------|
| 1 | Rushing yards page ✅ | Done, and extended with three pbp run-defense factors (stuff / explosive / efficiency). |
| 2 | Data spike ✅ | pbp runs, pbp drives/red-zone/pace, and NGS time-to-throw all load and are in use. Schema traps documented in §5. |
| 3 | Player TDs page ✅ | Opportunity × conversion; `nflsim/touchdowns.py`, page 3. |
| 4 | QB sacks + INTs pages ✅ | Offense-side rates, log-odds defense combine, NGS time-to-throw (optional). `nflsim/qb.py`, pages 4 & 5. |
| 5a | Team-strength layer ✅ | `nflsim/teams.py` + page 6. Opponent-adjusted points per drive, pace, fitted home field, calibrated scoring level; validated on 544 games (§4.1). |
| 5b | Drive-based game engine ✅ | `nflsim/game.py`. Pace-invariant drive simulation, game script, Dirichlet allocation to the depth chart, multinomial TD split (§4.2–4.4). |
| 6 | Game dashboard ✅ | Page 7: projected box score, margin and total distributions, win probability and fair moneyline. |

Keep it as a **multi-page Streamlit app** — `recyards` becomes one page among
several, sharing a common data/model utility layer.

---

## 7. Open questions

1. **Project structure** — extend the existing `recyards` folder into a
   multi-page app, or start a fresh project folder that imports the receiving
   model? (Affects Phase 1 file layout.)
2. ~~**Depth-chart input format**~~ — answered in §4.4: auto-pulled nflverse
   depth charts for the current season, with each player's share blending his
   own history and his positional-rank prior. Uploading a custom depth chart is
   still worth adding for hypotheticals ("what if this WR were the WR1").
3. **Seasons window** — how many seasons of priors, and how hard to weight the
   current one for in-season form.

---

*Phase 5b note: every box-score identity is asserted to hold in EVERY
simulation, not on average — player targets sum to team attempts, player
touchdowns sum to the drive engine's team touchdowns, and team passing yards are
literally the sum of the receivers' yards. Largest-remainder apportionment on
the Dirichlet shares is what makes the counts add up exactly.*

*Depth-chart feed note: the nflverse depth charts are now a stream of dated
snapshots keyed on `pos_abb` / `pos_rank` (no `week`, no `depth_team`), and
`gsis_id` joins straight to the weekly feed's `player_id`. `data.load_depth_charts`
normalises both that and the older weekly layout. Depth charts and injuries are
loaded for the season being PLAYED, which is deliberately not the priors window.*

*Phase 5a note: the team-strength layer is deliberately points-per-drive rather
than yards-per-drive — points are what the engine needs, and the outcome mix
carries the TD/FG detail. Yards per drive is still worth adding as a second
rating if the box-score yardage needs its own anchor.*

*Phase 4 note: sacks & INTs are built on the weekly offense feed (sacks/dropbacks, INTs/attempts) with an optional NGS time-to-throw scaler that degrades to neutral if NGS doesn't load. Rates combine offense + defense in log-odds; game-level rate wobble gives overdispersed counts.*

*Decisions locked so far: offense-only box score · QB-perspective sacks & INTs ·
fully self-contained scoring (no Vegas anchor).*
