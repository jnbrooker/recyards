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
4. **Recent seasons — and recent games — weighted more heavily** when building
   priors (one weight column on every feed; §8.3).

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

*Originally validated in-sample on 2024–25 (RMSE 12.7, winners 67.1%). Those
numbers are superseded by the out-of-sample harness (§8.1): fitted on prior
seasons and weeks only, the layer scores 12.71 / 68.0% in 2024 and 12.79 /
62.0% in 2025 against the closing line's 12.61 / 71.3% and 12.27 / 65.3%. The
residual spread (margin sd 12.8) is the target the drive engine reproduces.*

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

### 4.4 Depth-chart → roles — DECIDED, DONE, and rebuilt 2026-09-12
A name isn't a workload. The depth chart (auto-pulled from nflverse, current
season) decides *who is on the field and in what slot*; the player's own
history decides *what that slot is worth*. The original blend — own share
weighted `games / (games + 10)`, the rest the slot's rank prior, everyone
renormalised equally — had three defects that a user spotted from the output
(a rookie RB1 projected 7 carries; top receivers low) and a role backtest
over 5 489 listed player-weeks of 2025 then measured: listed **RB1s took 55%
of carries and were projected 45%; WR1s 25% and projected 22%**.

1. *History was weighted by games alone, never by whether it matched the
   slot.* A back who was a lead back somewhere and is now listed RB4 (James
   Conner behind a rookie) kept 65% weight on a 41% share; three such backups
   squeezed the rookie to 31%. Now the weight on any history-based estimate
   (own share, snap-count prior) is scaled by its **consistency with the
   slot's rank prior**, `(smaller / larger) ** 1`; for the top slot, history
   *above* the anchor is fully consistent (that is what a WR1 looks like) —
   only a promoted backup's history below it is discounted.
2. *Slot shares were guesses, and conditional.* They are now **measured,
   unconditional** shares — over every listed player not ruled out, with a
   game he did not appear in counting as zero (WR5s appear in 71% of weeks,
   RB4s in 23%) — so a roster's listed slots sum to the position group's real
   share of the ball instead of over-filling it and scaling every starter
   down. A player's own share is unconditional the same way: his touches over
   the team's touches in every game of his stint, missed games included.
3. *Renormalisation was over the whole roster.* Each position group is now
   fitted to the team's own split of touches (shrunk toward league), so the
   backfield cannot absorb the QB's carries — and the adjustment is taken
   from the **least-evidenced shares first** (`_fit_group`): when five
   listed receivers over-fill the group, the guessed WR4/WR5 slots give way,
   not a WR1's measured 34%. Depth-order pooling was tested and made no
   difference, so it is not there.

   *Two paths, one definition.* The single-stat pages simulate "if he plays",
   so `ui.live_share` divides the roster's unconditional share by the
   player's appearance rate; and the props path's own share is now the same
   volume-weighted quantity (targets ÷ team targets across his games) as the
   roster's — a mean of per-game shares ran ~10% high for receivers whose
   share peaks in their team's low-volume games.

Role backtest, 2025: RB1 carry-share error −15%, all carries −7%, targets
level; projected RB1 share 0.455 → 0.51 (actual 0.55; the residual is
mostly that the backtest's "actual" conditions on playing while the
projection does not), WR1 0.216 → 0.235. Selection-free level check, PPR
points per team-game: engine 82.5 vs 82.8 actual, every position group
within a point; targets 30.5 vs 30.4 once the engine stopped treating every
attempt as a target (`TARGET_PER_ATTEMPT`, measured 0.952). The single-stat
pages had the same disease in a different coat — regression toward one
positional mean pulled the 95th–99th percentile of receivers 10% low — and
now regress toward the player's **slot within his own team** (rank by share
→ the same slot table): receiving MAE 22.8 → 22.65, extreme-top bias −8.6 →
−4.2; rushing likewise.

Players ruled Out or Doubtful on the current season's latest injury report are
dropped and their share redistributed.

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
| 6b | Schedule + fantasy ✅ | `data.load_schedule` (nflverse `games.csv`): pick a real game by week on page 7, closing line shown as a comparator. `nflsim/fantasy.py` + page 8: per-simulation PPR/half/standard scoring for the whole slate with floor/ceiling and an exact breakdown, plus a head-to-head lineup simulator that sums each side PER SIMULATION so stacks and same-game players keep their correlation (Burrow–Chase +0.41; players in different games 0.00). |
| 6c | Pick'em card ✅ | `nflsim/pickem.py` + page 9. One pick per game (ATS or dog ML), 3-team ATS/ML parlays and 6-pt teaser, pool-named totals; every candidate graded from the simulated margin/total distribution, confidence 20→1 by expected return (flat or odds-weighted). Lines editable. A `market_weight` blend shifts each game's centre toward the line — the model's shrunk ratings see games as closer than the market, which flatters underdogs under odds-weighted scoring. |
| 6 | Game dashboard ✅ | Page 7: projected box score, margin and total distributions, win probability and fair moneyline. |
| 7 | Backtest harness ✅ | `nflsim/backtest.py` + page 10: rolling out-of-sample scoring of the team layer (vs the closing line) and the player models (vs a trailing average), per recency preset. First run fixed two receiving-model defects (§8.1). |
| 8 | Availability layer ✅ | `nflsim/availability.py`: QB familiarity + defensive starters out, with a QB quality swing, fitted on out-of-sample residuals as a margin shift (§8.5). 2025: 13.38 → 12.75 RMSE, 59% → 63% winners. |

Keep it as a **multi-page Streamlit app** — `recyards` becomes one page among
several, sharing a common data/model utility layer.

---

## 7. Open questions

1. ~~**Project structure**~~ — answered: the `recyards` repo became the
   multi-page app, `model.py` (the original receiving model) is page 1 and
   now delegates its loader to the shared layer.
2. ~~**Depth-chart input format**~~ — answered in §4.4: auto-pulled nflverse
   depth charts for the current season, with each player's share blending his
   own history and his slot's prior (rank averaged with a snap-count prior).
   *The one thing still open in this document:* uploading a custom depth chart
   for hypotheticals ("what if this WR were the WR1").
3. ~~**Seasons window**~~ — answered. The window defaults to the three most
   recent seasons nflverse has published (`data.season_choices`), so the current
   season joins automatically the week its first file lands; how much each
   season and each game counts is the recency control (§8.3).

---

*Live depth charts everywhere (`nflsim/roster.py`, `nflsim/ui.py`): pages 1–5
now pick players from the current season's depth chart with injury tags, use
the player's CURRENT team for volume (a traded receiver gets his new team's
pass rate), and use the live, injury-redistributed usage share. A player with no
history gets role-only priors; one listed Out is simulated "if he plays" with a
warning. The old history-only picker is still there behind a toggle.*

*Freshness: every nflverse loader sits behind a time-bucketed cache
(`data.ttl_cache`, `REFRESH_HOURS = 6`) and the page caches expire on the same
clock, so a long-running app picks up a new week without a restart. nflverse
republishes weekly stats within hours of games and depth charts / injuries
daily; nothing here needs a manual refresh.*

*Phase 5b note: every box-score identity is asserted to hold in EVERY
simulation, not on average — player targets sum to the team's targeted attempts, player
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

---

## 8. What would make the models better next (ranked)

1. ~~**An out-of-sample backtest harness.**~~ *Done 2026-09-12* —
   `nflsim/backtest.py` + page 10. For each week of a scored season everything
   is refitted on games before it (two prior seasons + earlier weeks), then the
   week is predicted. Team layer: `expected_points` margin / total per game,
   scored against the result and against the **closing line on the same
   games**. Player layer: the five single-stat models on every player who
   played that week with 5+ prior games and prop-worthy expected volume
   (history-only path — depth charts and injury reports cannot be replayed
   for past weeks), each as a full simulated distribution scored on MAE /
   RMSE / bias, 10–90 coverage, median split, CRPS, and for counts the Brier
   of P(≥1); a trailing weighted average is the naive baseline. Takes a
   `Recency`, so it compares the presets.

   **First results — 2025, 272 games, out of sample.** The in-sample table in
   §4.1 said margin RMSE 12.7 / winner 67%; honestly scored it is:

   | | margin RMSE | winner | log-loss | total RMSE | ATS vs close |
   |---|---|---|---|---|---|
   | Closing line | 12.27 | 65.3% | 0.611 | 13.19 | — |
   | Long memory | 13.47 | 60.5% | 0.665 | 13.40 | 46.9% |
   | **Balanced** | 13.38 | 59.0% | 0.658 | 13.38 | 49.8% |
   | Recent form | 13.42 | 60.1% | 0.657 | 13.50 | 50.2% |

   The model is ~1.1 points of RMSE behind the market and has no edge against
   the spread — as a self-contained rating with no injury / QB / weather
   information should be. The presets are within noise of each other on
   teams; recency helps the margin *correlation* (0.32 → 0.41) but not RMSE.
   Win probabilities are well calibrated at sd 13.5.

   **What the harness found in the player models, and what was fixed.**
   - **Receiving yards over-projected every receiver by ~10 yards** (bias
     +9.5, MAE 24.8 vs a trailing average's 22.1). Cause: yards per catch was
     built as *aDOT + YAC*, but aDOT is air yards per **target** and the deep
     targets are the incomplete ones — completed passes travel 5.7 air yards
     vs 7.8 per target, +2.1 yards on every catch. `model.player_priors` now
     carries `mu_air` (completed air yards per reception) and simulates from
     it; aDOT stays for display and the defense's depth ratio.
   - **Every 80% band covered ~93%.** Cause: the per-game SD of a rate (catch
     rate on six targets, yards per carry on twelve carries) is almost all
     sampling noise — a receiver's per-game catch-rate SD is 0.209 observed,
     0.205 of it binomial — and the simulator draws that noise itself, so it
     was counted twice. `data.between_sd` removes the sampling component
     (method of moments, floored) and receiving (`sd_ts`, `sd_catch`,
     `sd_air`) and rushing (`sd_share`, `sd_ybc`, `sd_yac`) priors use it.
     Coverage went 93% → 87% (receiving) and 86% → 81% (rushing).

   After both fixes, Balanced: receiving MAE 22.9 (naive 22.1), bias +3.8,
   CRPS 15.5; rushing MAE 24.6 (naive 24.0), bias +3.0, cover 81%; anytime-TD
   Brier 0.208 vs the base rate's 0.217; sacks and INTs sit at their naive
   baselines. The player models are calibrated but do **not yet beat a
   trailing average on the mean** — that is the next item.

1b. ~~**Regress player priors toward the positional mean.**~~ *Done
   2026-09-12, tuned by #1.* Two causes behind the residual bias, both fixed:
   - **Stars taken at face value.** The receiving priors now blend target
     share with the position's mean over `TS_PRIOR_N = 3` games-worth, and
     catch rate / air per catch / YAC per catch over 25 targets / 20
     receptions-worth (`model.league_priors`); rushing blends carry share
     over `SHARE_PRIOR_N = 2` games and YPC / the PFR contact split over 40
     carries (`rushing.league_rush_priors`). All on recency-weighted totals.
   - **Share estimated only from targeted (or carried) games.** A zero-target
     game while active is a real outcome that the backtest — and a prop —
     scores, and dropping those inflated mid-tier receivers' volume by ~15%.
     Shares now use every appearance; rates still use the games with a touch.

   The grid (2025, 2 000 sims, stable candidate set) put the pseudo-counts
   where overall and top-quintile bias cross zero; MAE was flat beyond that.
   Result, Balanced, 2025 out of sample: receiving MAE **22.8 vs naive 23.1**,
   bias −0.0 (was +9.5 before #1's fixes), every quintile within ±2.5 yards;
   rushing MAE **24.2 vs naive 24.6**, bias −0.6, 10–90 coverage 83%. The
   harness selects candidates on the *unregressed* share (`raw_ts`,
   `raw_share`) so the scored set does not move when the constants do.

   Still open: regression toward one positional mean pulls low-share players
   *up* (second quintile +2.5); a level-aware prior (WR1 vs WR4 — the depth
   chart's role prior, once it can be replayed historically) would fix the
   tail both ways. Count stats (TDs, sacks, INTs) sit at their naive
   baselines and have not had this treatment.

2. ~~**Goal-line role for touchdowns.**~~ *Done 2026-09-12.* The pbp now
   keeps the ball-carrier ids, and `data.load_touches` counts each player's
   carries and targets inside the 10 per game. Measured on 2024–25: a carry
   inside the 10 scores **29%** of the time vs 0.9% elsewhere, a target
   **39%** vs 2.6% — and for receivers the goal-line share is twice as stable
   year to year as the TD rate itself (r = 0.40 vs 0.21). So
   `touchdowns.goal_line_profiles` / `role_td_rates` build each player's TD
   rate as *role × conversion* (his regressed goal-line share of touches
   times the league conversion inside and outside the 10) and that, not the
   positional mean, is what his own TD history is regressed toward — over
   300 touches-worth, because the role is trusted more than the rare TDs.
   On the 2025 harness: anytime-TD Brier 0.2078 → 0.2063, log-loss 0.605 →
   0.602, correlation of projected with actual TDs 0.24 → 0.26, top-quintile
   bias 0.075 → 0.028; the same shrinkage toward the positional mean was
   worse, so the role signal is what helps. Modest — anytime-TD is mostly a
   volume-and-team-scoring question — but consistent on every metric. Page 3
   shows the goal-line shares and the prior; the game engine's TD split
   (`roster` rec/rush TD weights) uses the same role rates.

3. ~~**Recency weighting — make it consistent, then add within-season decay.**~~
   *Done 2026-09-12.* Every estimate in the suite is now a weighted one, and
   every feed carries the same weight column.

   **What changed.**
   - One helper, `data.game_weights(df, team_col, recency)`, stamps `season_w`
     and `w` on any per-game frame. `w = season_decay ** (seasons ago) ×
     0.5 ** (games_ago / half_life)`, where `games_ago` is counted **per team
     over the games it actually played** (a dense rank of distinct
     season-weeks, so a bye is not a game) and runs **continuously back
     through earlier seasons** — last season's finale is a few games staler
     than this season's opener, its week 1 a whole season staler. The first
     draft decayed only the latest season and left earlier ones flat; that
     inverted the ordering (a completed season weighed *less* than the one
     before it), so it was replaced.
   - The weekly feed (`recent_team`), drive table (`posteam`), designed-run
     pbp (`defteam`) and PFR rush feed (`team`) all get `w` at load time. The
     PFR loader now also keeps regular-season games only, like every other
     feed.
   - Every previously unweighted `.sum()` in a rate — sack/INT rates, catch
     rate, yards per target, TD-per-touch, YAC per reception, every defensive
     profile, team volume means and SDs, the pass-TD share, PFR per-player
     aggregates and league means — is a `w`-weighted total. Sample-size
     guards (`db >= 150`, `tgt >= 30`, …) still test raw counts, so a defence
     is not dropped for being recent. The regression pseudo-counts act on the
     weighted totals, so a rate built on old games is regressed harder — the
     intended behaviour.
   - `model.py` (page 1) no longer has its own loader; it filters
     `data.load_weekly` to the receiving positions, so the receiving page
     shares the cache, the schema aliases and the weights.
   - The season curve is geometric (`season_decay ** gap`) rather than the
     old `{1, 0.7, 0.45, 0.3}` table; at 0.7 the two differ by 0.04 at two
     seasons back.

   **The control.** `ui.priors_picker` replaces the seasons multiselect with
   one slider, *How much to trust this season*, mapped to `data.Recency`
   presets; the multiselect survives under an *Advanced* expander. The
   sidebar prints the resulting share of weight per season. Current-season
   share with two prior seasons in the window:

   | preset | (decay, half-life) | wk 1 | wk 4 | wk 8 | wk 12 | wk 17 |
   |---|---|---|---|---|---|---|
   | Long memory | (0.70, ∞) | 5% | 17% | 28% | 37% | 46% |
   | **Balanced** (default) | (0.85, 12) | 8% | 27% | 46% | 59% | 70% |
   | Recent form | (0.70, 6) | 16% | 47% | 70% | 82% | 90% |

   *Long memory* reproduces the old flat curve to within a point, so it is the
   "before" for any A/B. Scored by the harness (#1) on 2025: the three are
   within noise of each other on team margins (RMSE 13.47 / 13.38 / 13.42) and
   on player props; Balanced is marginally best on receiving and rushing,
   Recent form on sacks. Recency was not the lever the audit hoped — the
   player models' error is dominated by unregressed means (#1b), not stale
   ones. *Post-calibration (§8.8), 2025 with availability:* Recent form edges
   Balanced on the team layer (margin RMSE 12.67 vs 12.79, log-loss 0.631 vs
   0.644) and is within noise elsewhere; one season is not enough to move the
   default, but it is the first preset to check when a third season arrives.

4. ~~**Snap counts as the role signal.**~~ *Done 2026-09-12 — smaller than
   hoped, and the roadmap's premise was wrong.* Snap share is a strong proxy
   for usage (WR target share ≈ 0.25 × snaps, r = 0.72; RB carry share ≈
   0.81 × snaps, r = 0.87), but it does not "predict targets much better"
   than what the role layer already had: backtested over every listed player
   in 2025 (5 489 player-weeks), a snap-only prior was *worse* than the
   depth-chart-rank prior once a player had 15+ games, because his own
   target share is the better signal by then. Where a prior matters — under
   15 games — averaging the rank prior with the snap prior cut target-share
   error 5% (< 5 games) and 2.5% (5–15) and removed the rank prior's small
   WR/TE under-prediction; carries were neutral. That average is what
   `roster._player_row` now uses when the player has two games of snaps
   (`SNAP_TARGET_SLOPE`, `SNAP_CARRY_SLOPE`, `snap_roles`); the pages say
   "slot + snap prior" and show the snap share.

5. ~~**Unit availability — who is actually playing.**~~ *Built and fitted
   2026-09-12* — `nflsim/availability.py`, applied in `teams.expected_points`,
   the drive engine, fantasy, pick'em and page 7; scored on page 10.

   **Two indices, knowable before kickoff.** *QB familiarity* = 1 − the share
   of the team's recency-weighted dropbacks in the priors window taken by this
   week's starter (depth-chart QB1, or QB2 if he is Out). *Defensive
   availability* = importance of the defensive starters ruled Out / Doubtful ÷
   importance of all twelve starters (base front seven, secondary, nickel —
   the depth chart's `Base 3-4 D` / `Base 4-3 D` group as of kickoff), where a
   starter's importance is his recency-weighted defensive snap share
   (`data.load_snap_counts`, joined by gsis id). `data.load_depth_charts` now
   takes `side="defense"` and reconciles the two feed generations when
   seasons are mixed.

   **Fitted, not guessed.** Both indices were reconstructed for every 2024–25
   team-game (`historical_indices`) and regressed on the out-of-sample
   residuals from the harness (#1). The finding that set the design: **the QB
   effect is a margin effect, not a scoring effect** — −8.8 points of margin
   per unit of QB-index difference (t = −7.2, n = 544), with no effect on the
   total (t < 1.2 in either season). A backup QB costs his own side *and*
   hands the opponent short fields. So the shift is applied to the margin,
   half to each side, exactly like home field. The defensive index is
   directionally consistent and monotone by bin (+12.1 margin per unit
   difference; one or two key starters out ≈ +1.7 points to the opponent) but
   only t ≈ 2 pooled — real, small.

   **Out of sample, both directions:** fit on 2024 → 2025 takes margin RMSE
   13.38 → 12.92 and winners 59.8% → 63.1%; fit on 2025 → 2024 takes 13.66 →
   12.88. Almost all of it is the QB term; the defensive term adds ~0.04 of
   RMSE. With the pooled coefficients the 2025 harness reads **12.89 RMSE /
   63.1% winners vs the closing line's 12.27 / 65.3%** — half the gap to the
   market closed, from one feature.

   **Refinement, same session.** The dummy says only "did not take the
   dropbacks behind this rating", so a proven starter who changed teams would
   be charged like an injury backup. A *quality swing* — `qb_idx × (starter
   ANY/A − incumbents' ANY/A)`, ANY/A recency-weighted over the window and
   regressed toward replacement level — was tested against it: with both in
   the fit the swing is t = +3.4 and the dummy stays t = −6.1; swing alone is
   worse than the dummy alone. So unfamiliarity costs something beyond
   measured quality, and quality refines it. Both are in
   (`QB_MARGIN_COEF = −7.74`, `QB_SWING_COEF = +3.29`, `DEF_MARGIN_COEF =
   +12.36`); out of sample each direction improved a further 0.09 of RMSE.
   **2025 harness: 12.75 RMSE / 63.1% winners vs the line's 12.27 / 65.3%**
   (13.38 / 59.0% without the layer).

   **Offensive line — indexed, display-only.** Same construction (five line
   slots from the `3WR 1TE` group / `depth_position`, offensive snap shares).
   A line starter is out in 20% of team-games. Fitted alongside the other
   terms: −4.6 margin per unit of own-line index difference, **t = −1.1**,
   +0.02 RMSE out of sample either direction. Right sign, not distinguishable
   from zero on two seasons, so `OL_MARGIN_COEF = 0`; page 7 names the missing
   linemen with "shown but not priced". Revisit with a third season.

   Still open: 2026 week 1 shows the limit of the swing — Tua at ATL measures
   the same ANY/A as the incumbents he replaced, so he takes the full
   unfamiliarity penalty until he has played.

6. ~~**Weather and roof.**~~ *Evaluated and built 2026-09-12 — wind only.*
   Regressing the out-of-sample total residuals on the schedule's weather:
   wind above 10 mph costs **−0.52 points of total per mph** (t = −1.6,
   same sign and size in both seasons, and in line with what is known about
   wind and scoring); cold is non-monotone (32–45°F −3.4, below 32 +0.2) and
   domes show nothing, so only wind is priced — `teams.weather_total_shift`,
   shrunk to −0.4/mph and capped at 25 mph, applied to the total (both sides
   equally) in `expected_points` and the engine. On the harness it is a
   wash (2024 +0.01, 2025 −0.03 on total RMSE, 79 windy games): priced but
   unproven. Practical limit: nflverse only records wind after the game, so
   page 7 has a wind input (prefilled when the schedule has it) and an
   upcoming game needs the forecast typed in; fantasy and pick'em use the
   schedule value when present.

7. ~~**The market line as a comparator, not an anchor.**~~ Done — pages 7 and 8
   show the closing spread/total beside the model, never feeding it in.

8. ~~**Within-game drive correlation.**~~ *Done 2026-09-12, and it turned
   into the biggest calibration fix of the day.* Evaluating the engine's
   spread against the harness's residuals exposed two things:

   - **The team ratings were under-dispersed by half.** Out of sample, actual
     margin regressed on predicted margin had a slope of **1.97** in both
     seasons: `RATING_PRIOR_N = 200` drives of shrinkage (on recency-weighted
     drives, which shrinks harder still) was far too much. Grid on 2024–25:
     200 → 50 takes margin RMSE 13.52 → 13.26, winners 61.1% → 63.9%,
     log-loss 0.655 → 0.643; 25 gives slope 0.97. Set to **35** (slope 1.03).
     The availability coefficients had absorbed part of the missing spread
     and were refitted on the corrected residuals: QB familiarity −7.74 →
     **−4.66** (t = −3.7), quality swing **+3.88** (t = 3.9), defense
     **+10.73** (t = 2.0). Final, both seasons out of sample:

     | | margin RMSE | winners | log-loss | closing line |
     |---|---|---|---|---|
     | 2024 | **12.71** | 68.0% | 0.604 | 12.61 / 71.3% / 0.592 |
     | 2025 | **12.79** | 62.0% | 0.644 | 12.27 / 65.3% / 0.610 |

     Pooled the model is ~0.3 points of RMSE behind the market, from ~1.2 at
     the start of the day; in 2024 it is within 0.1.

   - **The engine's spread was ~9% too wide on margins.** Independent drives
     give a team-points sd of ~9.6 (pure binomial); the real *conditional*
     spread — residuals around the model's own prediction — is 9.3 per team,
     12.8 on the margin, with the two teams' points correlated +0.05. Real
     games are less variable than independent drives because leading teams
     sit on the ball and trailing teams press. `game._score_sides` now
     resolves drives in sequence, alternating possessions, with each side's
     scoring rate scaled by `exp(−LEAD_BETA × (lead − expected lead so far)
     / 7)`. Centring on the *expected* lead is essential: centring on zero
     compressed the mean margin by 0.8 points, double-counting behaviour the
     ratings (fitted to actual points) already contain. `LEAD_BETA = 0.03`
     reproduces the targets (margin sd 12.72, team 9.28, corr +0.06, mean
     shift +0.04); totals remain ~3% wide. The harness's win-probability sd
     is 12.8 to match. A 7-point favourite is now 70.8% rather than 68.6%.

9. **Game view on the single-stat pages.** *(Added 2026-09-12.)* A prop is
   priced for one game, so pages 1–5 now have a **Game / Season** view.
   *Game*: pick week → fixture → player (both depth charts); the opponent,
   home field, availability, wind and the **pre-game script** come from the
   engine — the team's expected carries / dropbacks in this game via the
   measured elasticities on the model's expected margin (`game.script_factors`:
   carries +0.113 per point, se 0.039; dropbacks flat), and on the TD page the
   per-touch rates scale with this game's expected points over the team's
   typical. *Season*: the old flow, any player vs any defense.

   Measuring those elasticities also recalibrated the engine: pass share
   moves −0.0034 per point of *realised* margin but only −0.0011 per point of
   the model's *expected* margin (the realised slope includes the reverse
   direction — teams that throw lose by more), so the hand-set
   `GAME_SCRIPT_BETA = 0.006` overstated the script 2–4×; it is 0.002 now,
   between the two. Judkins as a 14-point road dog: 13.8 carries in season
   view, 12.8 in game view, where the old engine had him at 11.1.

10. **Prop evaluation.** *(Added 2026-09-12.)* Player-prop lines from The
    Odds API (user's key, fetched only on request, ~32 credits a week for two
    markets), recorded in `props/ledger.csv` with the model's Game-view
    projection **frozen at fetch time** so nothing is re-projected after the
    fact; actuals fill in from the weekly feed; graded as mean/median vs the
    line and P(over) vs the book's vig-free probability (Brier, calibration,
    hit rate at a chosen edge), by week, game, market and book. The history
    accumulates from the first fetch — the free tier has no historical props.
    Re-fetching near kickoff records the closing line, the honest benchmark.

## 9. Maintenance

Several constants are fitted on out-of-sample residuals and rest on two
seasons (2024–25). Refit them each off-season, once a season's schedule,
depth charts, injuries and snap counts are complete:

```bash
python -m nflsim.calibrate 2024 2025 2026            # report
python -m nflsim.calibrate 2024 2025 2026 --write    # update the constants
```

The report shows, with t-stats: the availability coefficients
(`availability.QB_MARGIN_COEF`, `QB_SWING_COEF`, `DEF_MARGIN_COEF`, and the
display-only `OL_MARGIN_COEF` — price it if it reaches |t| ≥ 2), the
ratings' calibration slope (adjust `teams.RATING_PRIOR_N` toward slope 1.0),
the residual spreads (`backtest.MARGIN_SD`; check `game.LEAD_BETA` still
reproduces the margin / team sd and correlation with a few simulated
matchups), and the wind coefficient (`teams.WIND_COEF`). Then re-run the
harness (page 10 or `python -m nflsim.backtest`) and commit.

The recency presets (§8.3) and the player-prior pseudo-counts (§8.1b) were
grid-searched on 2025; the same grids are worth re-running with a third
season, but they moved little and are not expected to move.

*Decisions locked: offense-only box score · QB-perspective sacks & INTs ·
fully self-contained scoring (no Vegas anchor) · every fitted constant is
fitted out of sample, and a constant the harness cannot distinguish from zero
is shown but not priced.*
