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

11. **Week 1 post-mortem: the prop distributions were too skewed, and usage
    needs its own memory.** *(2026-09-14, after settling the 1,167 week-1 lines.)*

    **What the ledger said.** Model MAE 20.4 vs the line's 19.7 — the *means*
    were fine. But overs hit 54% while the model put P(over) > 0.5 on only 17%
    of lines: `P(actual > model median)` was 0.64 when a calibrated median
    gives 0.50. The frozen projections had a median/mean ratio of 0.67
    (receiving); real game logs sit at 0.75–0.95 depending on volume, ~0.10
    higher at every tier. The props page picks on the median, so it said
    "under" on 83% of lines. That was the tough week — shape, not level.

    **Cause, by stage.** Real per-game receiving yards have CV 0.80; the
    model's had 1.05. Targets were drawn `Poisson(team × share)` — a share of
    a fixed number of attempts is Binomial, `(1 − share)` less variance, and
    the Poisson over-dispersed target counts 13% (CV 0.62 vs 0.56). The SD
    floors (`MIN_TS_SD` 0.03 etc.) bound for 50–70% of players and implied a
    21% game-to-game wobble in role on top of the sampling noise already drawn.
    And `YPR_CV = 1.10` is a Gamma with shape < 1 (mode at zero): the measured
    within-player per-catch CV is 0.82–0.86 (204 receivers). Even at 0.86 an
    iid Gamma-sum is more right-skewed than real game logs at the same CV
    (catches and yards per catch are not independent within a game), so
    `YPR_CV` is now an *effective* 0.65, chosen on the out-of-sample shape.
    Rushing's shape was already close (coverage 80%, median 0.49) and was left.

    **Usage vs efficiency memory.** Week 1's largest misses were role
    surprises (Golden 3 → 12 targets; the market had him at 41.5 to the
    model's 22): 40% of week-1 error variance was target volume (7.7 yards per
    unexpected target). Measured 2023→25, week-1 share deserves ~25% of the
    weight on a target role and ~50% on a carry role against a full prior
    season; on raw shares the best exponential half-life is 3 games (targets)
    and 1.5 (carries), flat memory 6% / 21% worse. The single `half_life`
    (12) gave week 1 8%. So `Recency` now carries `target_half_life` and
    `carry_half_life`, `game_weights` stamps `w_tgt` / `w_car`, and every share
    and per-game-volume estimate (`model.player_priors`,
    `rushing.player_rush_priors`, `roster._share_from_counts` /
    `_share_when_playing`, `touchdowns` volume) uses them; rates keep `w`.
    The regression toward the slot prior still keys off `n_eff` from `w`, so
    the change moves *which games a role leans on*, not how hard it is
    regressed. Swept inside the model, out of sample (2025 weeks 2–9, 2 000
    sims), **5 games for both** beat the raw optima:

    | receiving (973 games) | MAE | naive | RMSE | bias | corr | cover80 | over_median | CRPS |
    |---|---|---|---|---|---|---|---|---|
    | before (1.10 / Poisson / floors / HL ∞) | 22.47 | 22.64 | 29.78 | −0.82 | 0.481 | 0.864 | 0.555 | 16.10 |
    | shape fixes only | 22.48 | 22.64 | 29.80 | −0.75 | 0.480 | 0.829 | 0.529 | 15.98 |
    | **shape + usage HL 5** | **22.33** | 22.54 | **29.35** | +0.15 | **0.498** | **0.825** | **0.502** | **15.66** |

    | rushing (418 games) | MAE | naive | RMSE | corr | cover80 | over_median |
    |---|---|---|---|---|---|---|
    | before (HL ∞) | 24.88 | 24.88 | 33.47 | 0.469 | 0.795 | 0.496 |
    | **usage HL 5** | **23.95** | 24.75 | **32.49** | **0.505** | 0.809 | 0.507 |

    Both models now beat a trailing average on the mean for the first time,
    and the receiving median is calibrated in every projection quartile
    (0.51 / 0.53 / 0.51 / 0.47). Carry HL 1.5 over-reacted inside the model
    (top-quartile bias +8.8); 5 is the compromise the harness picked.

    **Lessons for the harness.** `over_median` and `cover80` were already
    reported and read 0.555 / 0.864 on the shipped configuration; the pages
    pick on the median, so those two numbers are the props page's hit rate in
    disguise and should gate a release. The week-1 ledger predictions stay
    frozen (that is the point of the ledger); the version stamp changes from
    here.

    **Still open.** The top projection quartile still runs +3.4 (receiving)
    and +6 (rushing) high — the slot-prior regression pulls stars down and
    then the usage memory pushes recent big games up; a level-aware prior is
    the fix. Rushing's game-level spread is also ~20% wide in-sample (yards CV
    0.81 vs 0.68) even though its coverage is fine out of sample; worth the
    same per-stage decomposition when there is time.

    **Week 1 final, one row per player-market (225 lines, 16 Sep).** Side the
    median sat on hit 52.0%; picks at ≥3% edge 50.0% (n=192); the book's own
    favoured side 47.6%. Brier 0.270 vs the book's 0.250 — the book was better,
    largely because the model's under-skew put it on the wrong side of a week
    where outcomes ran +5 yards over the lines. Level was right (mean − line
    +0.4 receiving, +1.5 rushing); shape was the whole story (median − line
    −8.2 / −4.9). The 18 DEN@KC lines re-projected with the fixed shape sat
    at median − line −4, mean − line +1: medians now a normal 0.85 of the mean,
    still below the book's number. Whether the book's number is the mean or
    the median of outcomes is the open question, and it decides whether the
    model should pick on the median at all; it takes several weeks of settled
    lines to answer. The props page now shows this table by model version.
    The current version (17b6…) has no graded lines until week 2's are fetched.
    Also fixed: book nicknames ("Joshua Palmer", "Hollywood Brown") now match
    via an alias table and a loose first-name key; late matches on played
    games are settled but never predicted after the fact.

12. **Play-level game engine — built, gated, kept as a lab.** *(2026-09-16.)*
    `nflsim/playengine.py` + page 12. Ten seasons of play-by-play (462,773
    plays, cached locally in `cache/`) → empirical tables: play call by state
    (down, distance, field position, exact score within a field goal, part of
    game), actual outcomes by state, clock use by kind of play, kicks and tries.
    A vectorised loop plays every simulation one snap at a time (4,000 games in
    ~15 s). Team strength enters as per-side yards-per-play shifts solved so the
    engine's mean margin and total land on the ratings layer's expected game;
    fractional shifts are applied stochastically (rounding erased them — the
    first sensitivity measured was 14.6 points per yard because +0.6 rounded to
    a full yard on every play).

    **League gate, no tuning, identical teams:** drives 11.0 vs 11.4, punts 3.9
    vs 4.1, FG attempts 1.97 vs 1.97, INTs 0.78 vs 0.78, sacks 2.36 vs 2.38,
    plays 63 vs 62, mass at 7 / 6 / 10 points right, home–away score correlation
    right. Open: mass at exactly 3 is 10% vs 14% (needs timeouts as a resource
    and the two-minute warning); ties 1.0% vs 0.4% (overtime is single-period
    sudden death here).

    **Shape gate, 2025, engine vs a normal curve on the same out-of-sample
    means:** winner log-loss 0.6368 vs 0.6342, cover 0.7282 vs 0.7270, over
    0.6958 vs 0.6943 — a hair worse everywhere, inside noise. A normal with the
    right sd is already a good model of a margin, and the engine's missing mass
    at 3 costs it what its mechanics gain. More telling: the cover calibration
    is inverted for BOTH (predicted 66% → actual 49%), because the ratings'
    disagreement with the closing line carries no signal; no shape fixes that.
    **So the drive engine stays the default for scores and spreads.**

    **Stage 4 — players on top (built 2026-09-16).** `playengine.allocate_players`
    takes each simulation's team attempts, completions, gross passing yards,
    carries, rushing yards, touchdowns, sacks and interceptions from the clock
    and game state and allocates them to the depth chart with the drive
    engine's own machinery (Dirichlet shares, the receiving and rushing
    yardage mechanics, multinomial touchdown splits), then rescales so that in
    EVERY simulation the receivers sum to the team's gross passing yards and
    the rushers to its rushing yards. Every box-score identity holds on both
    engines. The structural difference is the point: under the play engine a
    quarterback's passing yards correlate **+0.35** with the game total (drive
    engine 0.00; real +0.44) and rushing attempts **+0.61** with the margin
    (drive +0.20; real +0.50) — because volume comes from the game, not from a
    normal draw. Team tendencies were added so teams differ in call mix and
    pace: pass rate over expectation as a logit shift on the play call (BAL
    −0.34 to CIN +0.29) and tempo as a clock multiplier, from the last three
    seasons of plays. `game.run_game(..., engine=)` dispatches; the output
    shape is identical so fantasy, the pick'em slate and the game page take an
    Engine toggle (`ui.engine_picker`, default Drive).

    **Stage-4 gate (5,278 real team-games vs the engine):** pass attempts 32.4
    vs 34.1, rush attempts 27.9 vs 25.9, gross pass yards 236 vs 243 — close
    but leaning run; and the outcome dependence of volume is too strong: real
    teams run 60 snaps when losing by 14+ and 63 when winning by 14+, the
    engine 57 and 67 (slope 0.22 snaps per point vs a real 0.05). Traced to
    the losing side's drives being too short (5.1 snaps vs 5.8). Two fixes
    tried and kept because they are right, neither closed it: the clock now
    keys on the size of the lead (trailing 9+ snaps every 28 s in the second
    half against 37 s for a team up 9+); outcomes on downs 2-4 are sampled
    relative to the sticks so conversion rates match the data exactly. **Open:**
    the trailing team's volume, ~3-4 snaps at the extremes. Until it closes,
    the play engine over-states a trailing team's receivers slightly less than
    the drive engine's normal draw does, but is not yet validated as the
    better props engine — no player-level historical replay is possible with
    as-of-now depth charts, so the gate is the team-box distributions above.

    **Player-level replay, done after all (`nflsim/enginetest.py`, 2026-09-17).**
    The 2025 depth charts are daily snapshots, so rosters *can* be rebuilt as
    of each kickoff. Both engines on the same 136 games (8 a week, weeks 2–18),
    same as-of ratings, rosters, priors and injuries, 2,000 sims, scored on
    every player who played (1,946 receiving lines, 847 rushing, 1,762
    receptions; ~35 min for the pair):

    | | drive | play |
    |---|---|---|
    | receiving MAE / CRPS | 19.02 / 13.21 | 19.04 / 13.15 |
    | receiving median beaten | 0.525 | 0.505 |
    | receiving 40–60 tier: bias / median beaten | −4.8 / 0.580 | −2.7 / 0.509 |
    | receiving 80% cover / above p90 | 0.810 / 0.112 | 0.795 / 0.112 |
    | rushing MAE / CRPS | 19.18 / 13.14 | 19.26 / 13.22 |
    | receptions MAE / CRPS | 1.50 / 1.05 | 1.52 / 1.06 |

    A dead heat: the CRPS gap on receiving is −0.06 (se 0.04), on rushing
    +0.08 (se 0.11). The one real difference is the drive engine's medians for
    40–60-yard receivers (the WR1/WR2 tier that carries most prop lines)
    running low — beaten 58% of the time — which the play engine fixes
    (50.9%). Nothing else separates them, tails included. Verdict: the play
    engine is *as good* on players, not better; it earns its place on
    correlated markets (its QB-yards↔total correlation is real), not on
    single-player lines. Not adopted as default.

    Also live / any-state pricing (not started).

13. **Learned mean (`nflsim/learn.py`) — tested, not adopted.** *(2026-09-15.)*
    Gradient boosting on 2016–25 player-weeks with routes run and targets per
    route (participation feed), snap share, the QB's EPA per dropback, opponent
    man-coverage and pressure rates, and the player's own history at three
    horizons; walk-forward by season. Receiving yards: MAE 23.28 vs a trailing
    average's 23.77 (2024), 22.69 vs 23.15 (2025) — and **23.06 vs the current
    model's 22.99 on matched rows, correlation 0.93 between the two.** No
    advantage for rookies, role changes or stars; a 50/50 ensemble gains 0.1.
    Targets alone: 3.7% better than the trailing average. Conclusion: the mean
    of a receiver's yards is at the ceiling of box-score and participation data
    (noise floor ~19–20, both models at 23); the gap to the book (~0.7 yards) is
    information, not maths. Routes: far more stable week to week than target
    share (0.70 vs 0.56) but targets per route is noisy (0.27), so route% × TPRR
    does not beat the EW share (0.0510 vs 0.0506). Kept as the experiment
    harness; the next real lever on props is injury-week redistribution, and
    the model's structural edge is the joint distribution (SGP, fantasy).

14. **Pick'em: probabilities from the market and history.** *(2026-09-15.)*
    `nflsim/market.py`. The engine's disagreement with the closing spread
    carried no information on 2025 (slope +0.01, r +0.004; ATS 46%). Page 9 now
    prices each side as P(the pool's line is beaten | the market's line) from
    4,191 games at the same closing spread (±0.5 kernel, anchored at 50/50 on
    the market's own number), totals from the empirical residual, moneylines
    vig-free. Out of sample 2023–25: pool a point worse than market → predicted
    44.3%, actual 44.7%; Wong teaser legs predicted 74.1%, actual 76.7% (n=180,
    breakeven 72.7%). Replay of 19 weeks, odds-weighted: 3,831 points realised
    vs 4,041 expected (engine at 50% lean: 2,731 vs 5,100); teasers 8/19 vs
    4/19. Pool lines are editable beside live market lines (Odds API); a
    "Number vs market" column shows the half-point edge. The pool prices at
    openers, so that edge is real but unmeasured until the weekly ledger
    accumulates closing lines.

    *Replay re-run 2026-09-16 with the current ratings layer.* 2025, 18 weeks,
    odds-weighted: market card 3,596 realised vs 3,697 expected (126 of 360
    wins, 118 expected; parlays/teasers 13 of 54; totals 20 of 34); engine card
    3,295 vs 4,226 expected (3 of 54 combos) — the engine's expectation is the
    over-confidence the market source removes. 2026 week 1 live: market card
    144 vs 205 expected (4 of 20 — the odds-weighted card is 16 moneyline dogs,
    and the dogs went 4-12), engine card 209 vs 226. One week is noise; the
    market card's 2025 expectation was within 3% of realised, the engine's was
    28% high.

15. **Warm-up and one cache for the app.** *(2026-09-16.)* Every page that
    simulates a game or a slate goes through the shared cached functions in
    `ui.py` (`cached_context`, `cached_schedule`, `cached_rosters`,
    `cached_game`, `cached_week`, `cached_slate`, `cached_team_backtest`,
    `cached_history`, `cached_play_engine`), keyed on the same `DEFAULTS`, so
    two pages never compute the same thing twice. `Home.py` runs `ui.warm_up`
    once per app process on first open (drive engine 2-3 minutes; with the
    play engine 10-20 minutes) — priors, rosters, this week's slate on both
    engines, the pick'em replay and the game page's first fixture — after
    which every page opens from cache at its default settings. A toggle skips
    the play engine; a button re-runs it. Caches live six hours.

16. **Play engine in the props ledger.** *(2026-09-17.)* The 2025 replay
    (§12) could only say the engines were level on players; the ledger can say
    it on the lines that matter. Every ledger row now freezes projections
    beside the Game view — `play_*` (and from §17, `drive_*`) — from one
    engine simulation of the fixture with the player read off the box score,
    same active roster, same rule that nothing is projected after kickoff. The
    page grades any of them (a *Projection graded* radio drives Edge, Pick and
    the charts), shows each one's median and P(over) per line, and once lines
    settle compares them head to head on the SAME lines
    (`props.compare_engines`). Week 1's lines have blank engine columns; week
    2's were frozen on 17 Sep before the Thursday game.

    *Week 1 in hindsight (a replay, never written to the ledger).* Both engines
    on the 225 week-1 consensus lines with everything as of kickoff: median-side
    hit 50.9% (frozen Game view, old shape) / 52.9% (drive) / 56.9% (play);
    picks at ≥3% edge 50.5 / 53.6 / 57.3%, and the play engine held ~57% at
    every edge threshold while the drive engine faded to 53.8% at ≥10%. On the
    41 lines where the engines took opposite sides, the play engine's side hit
    61%. The mean beat the median as a pick centre for the Game view and the
    drive engine (+3.6 / +4.4 points) and not for the play engine (−1.3): a
    calibrated model can only be picked on its median, so the mean "winning" is
    a diagnostic that the median is biased low (actuals above the median 62% /
    57% / 54%), not a reason to switch.

17. **Play engine rebuilt against a state-conditioned gate; timeouts; the
    ledger on `run_game`.** *(2026-09-18.)* The stage-4 gate was rebuilt as
    code (`playengine.box_gate`, `python -m nflsim.playengine --gate`): a
    *dispersed* league (matchups steered to margins drawn N(0, 6), so time
    spent trailing or leading big is comparable) scored against the last three
    seasons on snaps, pass rate and drive length **by live score state**,
    drive-ending events, down shares and the key numbers, with written
    tolerances. The first run localised everything: snaps and downs right;
    pass rate damped by 3-4 pp in both directions; trailing drives too short
    (5.5 vs 6.1 snaps) because the hierarchical tables dropped the score key
    first whenever a cell was thin — which is precisely the late, lopsided
    cells — so a trailing team punted like an average one (real fourth-down
    go rates trailing 9+ in the second half: 52-83%; even: 15-44%).

    **What changed, in order of effect on props.**
    - *Decisions as a dense physical table plus additive clock-and-score
      shifts* (`ShiftTable`, the PROE idea applied to the situation), fitted
      with recency weights (pass rate 0.60 → 0.57 and fourth-down go 0.13 →
      0.23 over 2016-25 had made the pooled tables 1.5 pp too pass-happy and
      8 pp too timid) and shrinkage n/(n+25) toward the parent instead of a
      40-play cutoff. Fourth down keys on a *need* class (a kick ties or wins /
      one score / two scores…) and a field-position zone, because late and
      trailing the rule is "kick if in range, otherwise go" — a uniform shift
      predicted 24% punts for a team down 1-3 in the last two minutes (real
      1.4%). Kneels, spikes and early-down kicks have their own
      clock-and-score-first table. Calibrated on real fourth downs in every
      score state.
    - *Timeouts as a resource* (3 a half, 2 in overtime), called at the data's
      rate by situation among teams that have one (`timeout_off/def`, from
      timeout rows charged to the play whose clock window holds them), clock
      pools split by whether a timeout followed the play, the two-minute
      warning, and the victory formation as a rule: a leading team kneels out
      only when the clock it can burn covers the time left. Used 3.7 a
      team-game (real 3.65). This was the missing mass at 3: in a one-score
      game at 2:00 the trailing team got 5.9 snaps and 0.34 field-goal
      attempts against 7.25 and 0.67 real.
    - *Hurry-up outcomes*: the last two minutes of a half are a different
      pool (deep shots, sideline throws — fewer completions, more clock stops,
      more chunk gains), so outcomes carry a hurry flag ahead of field
      position. Within-team score effects on yards in normal time (a trailing
      team's runs +0.18, its passes −0.15; overtime passes +0.44) as a shrunk
      mean shift — measured after removing team-season means so the strength
      shifts are not double-counted (the raw conditional pools would have
      imported "trailing teams are worse").
    - *Game-level variance*: play-level noise under-disperses games (neutral
      margin sd 12.0 vs the 12.8 residual the harness calibrated to; total sd
      12.0 vs 13.4). Each side draws a yards-per-play shift for the day plus a
      factor common to both (it moves the total, not the margin); both sds are
      solved in `sensitivity` so the neutral engine lands on the targets. This
      is what the shape gate's "hair worse" cover log-loss was: over-confidence.
    - *Overtime under the current rule* (both teams possess, then sudden
      death; a matching field goal does not end it — the first cut declared
      a tie there) and the last two minutes of OT mapped onto the late bins,
      since they are the last two minutes of a tied game. Ties per OT 21% →
      14% (real ~7%).
    - *Pace*: clock pools thinned toward recent seasons (63.5 snaps a
      team-game in 2016, 61.5 in 2023-25); a definition mismatch fixed
      (turnovers were "clock stops" in the outcome table but not in the clock
      table, so they drew from the short pool — a snap a team-game); the own
      1-5 split off as its own field-position bin (safeties 0.18 → 0.12 a
      game, real 0.054); drives counted at their first real play, as the
      reference counts them.
    - *Shares*: both engines now allocate with the roster's conditional ("if
      he plays") share for active players (`roster.sim_shares`), the quantity
      the single-stat pages already used; the unconditional share fills the
      roster's slots but under-projects a player who missed games and is now
      in. And the play engine's rescale to the team's totals is robust: a
      near-cancelling rushing sum had multiplied one back's yards by 10¹⁶.

    **Gate after (engine vs 2023-25, gap):** snaps +0.9, pass attempts +0.7,
    rush +0.35, gross pass yards +0.3, rush yards +0.2, pass rate by state
    within 0.8 pp, snap share by state within 1 pp, every drive-ending event
    within 0.1 a team-game, down shares within 0.3 pp, P(7) exact. **Known
    gaps** (reported, not failed): mass at 3 (9.3% vs 14.2% — the halftime
    distribution matches; the drift is second-half accumulation of 1/4/6-point
    margins, TD-versus-two-FG offsets, not the end-game any more), margin sd
    −0.8, safeties 2×, ties 0.6% vs 0.4%.

    **Players (2025 weeks 2-18, 5 games a week, 5,552 player-games paired
    old code vs new on identical as-of inputs).** The engine rebuild is
    NEUTRAL on single-player marginals: play engine receiving CRPS +0.01
    (se 0.01), rushing 0.00 (se 0.03) before the share change — the ratings
    steer each game's mean volume, and the rebuild changed *when* volume
    arrives, which is joint structure (game scripts, correlated markets, live
    pricing), not a player's marginal. The **conditional-share change is a
    measured gain for both engines**: receiving CRPS 13.09 → 12.92 (drive,
    t −2.3) and 12.98 → 12.82 (play, t −2.2), MAE −0.26 / −0.22, the mean
    bias closing (−0.22 → −0.01 drive); rushing unchanged. Play vs drive on
    the final code: receiving 12.82 vs 12.92, rushing 13.76 vs 14.19 — the
    play engine's rushing lead was already there on this sample with the old
    code, so it is sample variation against §12's run, not the rebuild. Verdict
    unchanged from §12: as good on single lines, with a better-behaved team
    layer; the ledger decides.

    **The ledger's primary projection is now `run_game` on the default
    engine.** Three projections per line: `drive_*` (the drive engine's own
    box score — what pages 7-9 ship), `play_*`, and the single-stat Game view
    (`pred_*`) kept as the third. `ui.DEFAULT_ENGINE` (one line, "drive") is
    what the page grades by default and what `engine_picker` opens on. The
    **promotion gate** is written down and scored on the page
    (`props.promotion_gate`): ≥600 settled lines with both engines frozen,
    play ≤ drive on Brier and median MAE, actuals above the play median in
    0.47-0.53 on both yardage markets, play within 2 points of drive in every
    line tier. When every row is met, flip the constant.

    **App.** Long computations report a fraction into a per-thread slot; the
    page thread draws one bar with what is being simulated, the time elapsed
    and an estimate of what is left (`ui.run_with_progress`) — cached
    functions never draw, so nothing is replayed on a cache hit. The built
    engine (tables, sensitivities, game-level sds) is pickled in `cache/`
    keyed on this file's hash, so a process starts from disk; the play
    engine's slates run at 5,000 sims (`DEFAULTS["n_sims_slate_play"]`, ~7 s a
    game), with the sims slider per engine so the warm-up's cache is what the
    page opens on.

    **Open, in order:** the second-half drift away from 3 (joint TD/FG
    differential structure); safeties; the 14% OT tie rate under one season of
    the new rule; drive length while trailing still −0.35 snaps (real trailing
    drives are the longest; penalties in the two-minute drill are 0.59 a game
    real, 0.32 engine — pass interference on deep shots is not in the outcome
    pool as a separate event).

18. **Play engine the default; kickers in the fantasy projections.**
    *(2026-09-18.)* `ui.DEFAULT_ENGINE = "play"`. The decision, on the
    evidence: equal on single-player lines (§17's paired replay), better on
    games (state-conditioned volumes, timeouts, calibrated variance), and the
    only engine that can score kickers and defences honestly. The props page's
    gate is now a *tripwire* rather than a promotion test: the play engine
    holds the default while it is at least as good as the drive engine on the
    lines both projected (Brier, median MAE, calibration on both yardage
    markets, every line tier); a failed row is the signal to revisit. Cost:
    a play-engine slate is ~80 s against 6 s; the warm-up absorbs it.

    **Kickers.** Every field goal in the play engine already has a distance
    and a make probability; the fantasy line is a read-off, correlated with
    the game the way it should be (a red-zone stall is an attempt, a blowout
    fewer). Added: attempts and makes by the fantasy bands (to 39 / 40-49 /
    50+) per simulation; per-kicker accuracy as shrunk logit shifts on the
    league curve by band and on the extra point (`fit_kickers`, from every
    attempt 2016-25 with a recency weight, 40 attempts to full weight —
    Dicker, Folk, Aubrey at the top, Moody, Rosas at the bottom); the league
    curve and the XP rate recency-weighted too (kicking improved: 83% pooled,
    85% recent); each team's kicker taken from whoever took its most recent
    kicks in the play feed (the depth-chart and weekly feeds are
    offence-only). The drive engine gets the same rows from its made field
    goals and touchdowns per simulation — bands by the league's split of
    makes, misses by band, the same kicker shifts — so the toggle stays
    symmetric. `fantasy.KICKER_RULES` (3/4/5, 0 for a miss, 1 an XP) in every
    preset; K in page 8's positions, pool and breakdown. **League level:**
    7.95 kicker points a team-game against 8.26 real (2023-25); FG made 1.66
    vs 1.70, XP 2.08 vs 2.16 — the residual is the engine's known −0.1
    offensive touchdowns a game, not the kicking.

    **Next: defence / special teams** (two to three days). Points-allowed
    tiers, sacks, interceptions, fumbles lost, defensive touchdowns and
    safeties are already per-simulation counters in the play engine, and they
    are the *opponent's* offensive counters — a defence's interceptions are
    the opposing quarterback's, which is the joint structure the drive engine
    cannot give. To add: return touchdowns split from defensive ones (a
    counter; kickoff-return TDs are already drawn), blocked kicks (in the punt
    table as `punt_blocked`, unused), defensive strength on sack and INT
    rates (the `qb.py` profiles the drive engine already uses, as shifts),
    D/ST scoring rules and rows, drive-engine parity (its defensive TDs are a
    rate, not tied to the opponent's turnover draws), and a 2023-25 backtest
    of D/ST points per team-game (derivable from the play feed). Then the
    same markets in the props ledger.

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
