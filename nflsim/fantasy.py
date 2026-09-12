"""
nflsim/fantasy.py — fantasy points from the game engine, per simulation.

The game engine already produces every player's line in every simulation, so
fantasy points are just a scoring rule applied to those arrays. Because it is
done per simulation rather than on the means, the output carries a real
distribution: a floor and a ceiling for every player, not just a projection,
and the projection itself is decomposed into where the points come from.

The QB line is assembled the way the roadmap insists: his passing yards are the
sum of his receivers' yards, his passing touchdowns are the receiving
touchdowns that were allocated, and his interceptions and sacks come from the
Phase 4 models — so a QB's fantasy points can never contradict his receivers'.

Not modelled (and therefore not scored): fumbles, two-point conversions, return
yards. They are small and very noisy; if a league scores them they are a
constant to subtract, not a distribution to simulate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import game as G

# Points per unit. Keys are the breakdown components shown on the page.
PRESETS = {
    "PPR": dict(rec=1.0, rec_yd=0.1, rec_td=6.0, rush_yd=0.1, rush_td=6.0,
                pass_yd=0.04, pass_td=4.0, int=-2.0),
    "Half PPR": dict(rec=0.5, rec_yd=0.1, rec_td=6.0, rush_yd=0.1, rush_td=6.0,
                     pass_yd=0.04, pass_td=4.0, int=-2.0),
    "Standard": dict(rec=0.0, rec_yd=0.1, rec_td=6.0, rush_yd=0.1, rush_td=6.0,
                     pass_yd=0.04, pass_td=4.0, int=-2.0),
    "PPR, 6-pt pass TD": dict(rec=1.0, rec_yd=0.1, rec_td=6.0, rush_yd=0.1, rush_td=6.0,
                              pass_yd=0.04, pass_td=6.0, int=-2.0),
}

COMPONENT_LABELS = {
    "rec": "Receptions", "rec_yd": "Rec yards", "rec_td": "Rec TD",
    "rush_yd": "Rush yards", "rush_td": "Rush TD",
    "pass_yd": "Pass yards", "pass_td": "Pass TD", "int": "INT",
}


def side_points(side: dict, rules: dict) -> tuple[np.ndarray, dict]:
    """Per-simulation fantasy points for every player on one side.

    Returns (points[n_sims, n_players], components) where `components` maps each
    scoring component to its own [n_sims, n_players] array so the breakdown is
    exact rather than reconstructed from means.
    """
    r = side["roster"].reset_index(drop=True)
    n, k = side["targets"].shape
    comp = {
        "rec": rules["rec"] * side["receptions"],
        "rec_yd": rules["rec_yd"] * side["rec_yards"],
        "rec_td": rules["rec_td"] * side["rec_tds"],
        "rush_yd": rules["rush_yd"] * side["rush_yards"],
        "rush_td": rules["rush_td"] * side["rush_tds"],
        "pass_yd": np.zeros((n, k)), "pass_td": np.zeros((n, k)), "int": np.zeros((n, k)),
    }
    # The team's passing line belongs to its QB1 (the engine caps QBs at one).
    qb = np.where(r["position"].values == "QB")[0]
    if len(qb):
        j = int(qb[0])
        comp["pass_yd"][:, j] = rules["pass_yd"] * side["team_pass_yards"]
        comp["pass_td"][:, j] = rules["pass_td"] * side["n_pass_td"]
        comp["int"][:, j] = rules["int"] * side["ints"]
    total = sum(comp.values())
    return total, comp


def game_projections(sim: dict, rules: dict, game_id: str = "",
                     week: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Projection table for both teams in one simulated game.

    Returns (table, samples): the table has one row per player with the mean,
    floor (p10), median and ceiling (p90) plus the mean points from each
    component; `samples` maps player_id -> per-simulation points (float32) so a
    page can draw the full distribution on demand.
    """
    rows, samples = [], {}
    for team, opp, side in ((sim["team_a"], sim["team_b"], sim["box_a"]),
                            (sim["team_b"], sim["team_a"], sim["box_b"])):
        pts, comp = side_points(side, rules)
        r = side["roster"].reset_index(drop=True)
        for j, pl in r.iterrows():
            p = pts[:, j]
            row = dict(
                player_id=pl["player_id"], Player=pl["name"], Team=team, Opp=opp,
                Pos=pl["position"], Depth=int(pl["depth"]),
                Proj=float(p.mean()), Floor=float(np.percentile(p, 10)),
                Median=float(np.median(p)), Ceiling=float(np.percentile(p, 90)),
                P20=float((p >= 20).mean()),
                Source=(f"{int(pl['games'])}g history" if pl["games"] > 0 else "role prior"),
                game_id=game_id, week=week,
                # raw line, for the breakdown table
                Tgt=float(side["targets"][:, j].mean()),
                Rec=float(side["receptions"][:, j].mean()),
                RecYds=float(side["rec_yards"][:, j].mean()),
                RecTD=float(side["rec_tds"][:, j].mean()),
                Car=float(side["carries"][:, j].mean()),
                RushYds=float(side["rush_yards"][:, j].mean()),
                RushTD=float(side["rush_tds"][:, j].mean()),
            )
            for key, arr in comp.items():
                row[f"pts_{key}"] = float(arr[:, j].mean())
            if pl["position"] == "QB":
                row["PassYds"] = float(side["team_pass_yards"].mean())
                row["PassTD"] = float(side["n_pass_td"].mean())
                row["INT"] = float(side["ints"].mean())
            rows.append(row)
            samples[pl["player_id"]] = p.astype(np.float32)
    return pd.DataFrame(rows), samples


def week_projections(ctx: dict, games: pd.DataFrame, rules: dict,
                     n_sims: int = 10000, use_injuries: bool = True,
                     seed: int = 11) -> tuple[pd.DataFrame, dict, list]:
    """Simulate every game in `games` (a schedule slice) and stack the tables.

    Returns (table, samples, game_summaries). Games whose teams have no rating
    or no depth chart are skipped and reported in the summaries.
    """
    ratings = ctx["ratings"]
    tables, samples, summaries = [], {}, []
    rosters = {}

    def roster(team):
        if team not in rosters:
            rosters[team] = G.roster_for(ctx, team, use_injuries=use_injuries)
        return rosters[team]

    for i, g in games.reset_index(drop=True).iterrows():
        home, away = g["home_team"], g["away_team"]
        try:
            if home not in ratings["off"].index or away not in ratings["off"].index:
                raise ValueError("no rating")
            sim = G.simulate_game(ratings, ctx["wk"], roster(home), roster(away),
                                  home, away, ctx["pass_vol"], ctx["rush_vol"],
                                  ctx["rush_def"], ctx["lg_pass"], home="a",
                                  n_sims=n_sims, seed=seed + i, avail=ctx.get("avail"),
                                  wind=g.get("wind"), roof=g.get("roof"),
                                  target_rate=ctx.get("target_rate"))
        except ValueError as e:
            summaries.append(dict(game_id=g["game_id"], home=home, away=away,
                                  ok=False, error=str(e)))
            continue
        t, smp = game_projections(sim, rules, g["game_id"], int(g.get("week", 0)))
        s = G.summarize(sim)
        tables.append(t)
        samples.update(smp)
        summaries.append(dict(
            game_id=g["game_id"], home=home, away=away, ok=True,
            home_pts=s["mean_a"], away_pts=s["mean_b"], win_home=s["win_a"],
            total=s["mean_total"], margin=s["mean_margin"],
            spread_line=g.get("spread_line"), total_line=g.get("total_line"),
            label=D.game_label(g) if "kickoff" in g else f"{away} @ {home}",
        ))
    table = (pd.concat(tables, ignore_index=True).sort_values("Proj", ascending=False)
             .reset_index(drop=True)) if tables else pd.DataFrame()
    return table, samples, summaries


def breakdown_table(row: pd.Series) -> pd.DataFrame:
    """'How it got there' for one player: stat line, rule, points."""
    items = [
        ("Receptions", row.get("Rec", 0.0), "rec", ""),
        ("Receiving yards", row.get("RecYds", 0.0), "rec_yd", ""),
        ("Receiving TD", row.get("RecTD", 0.0), "rec_td", ""),
        ("Rushing yards", row.get("RushYds", 0.0), "rush_yd", ""),
        ("Rushing TD", row.get("RushTD", 0.0), "rush_td", ""),
    ]
    if row.get("Pos") == "QB":
        items += [
            ("Passing yards", row.get("PassYds", 0.0), "pass_yd", ""),
            ("Passing TD", row.get("PassTD", 0.0), "pass_td", ""),
            ("Interceptions", row.get("INT", 0.0), "int", ""),
        ]
    out = pd.DataFrame([dict(Stat=name, Projected=float(val),
                             Points=float(row.get(f"pts_{key}", 0.0)))
                        for name, val, key, _ in items])
    out = out[(out["Projected"].abs() > 0.005) | (out["Points"].abs() > 0.005)]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Head-to-head: two lineups, who wins?
# ---------------------------------------------------------------------------
# Every player's samples are indexed by simulation, and simulation k of one game
# is independent of simulation k of another — exactly as real games are — while
# players in the SAME game share simulation k and therefore its pace, script and
# scoring. So summing a lineup along the simulation axis gives a joint draw of
# the lineup's total with the right correlation structure: a QB stacked with his
# WR1 is riskier than the same projection spread across four games, and the
# win probability reflects that.

def lineup_totals(samples: dict, player_ids: list, extra: float = 0.0) -> np.ndarray:
    """Per-simulation total for a lineup (plus a constant for unmodelled slots)."""
    arrs = [samples[p] for p in player_ids if p in samples]
    if not arrs:
        return None
    n = min(len(a) for a in arrs)
    return np.sum([a[:n] for a in arrs], axis=0).astype(float) + float(extra)


def matchup(samples: dict, ids_a: list, ids_b: list,
            extra_a: float = 0.0, extra_b: float = 0.0) -> dict | None:
    """Simulate lineup A against lineup B; returns totals and win chances."""
    a = lineup_totals(samples, ids_a, extra_a)
    b = lineup_totals(samples, ids_b, extra_b)
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    margin = a - b
    return dict(
        total_a=a, total_b=b, margin=margin,
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        p10_a=float(np.percentile(a, 10)), p90_a=float(np.percentile(a, 90)),
        p10_b=float(np.percentile(b, 10)), p90_b=float(np.percentile(b, 90)),
        win_a=float((margin > 0).mean()), win_b=float((margin < 0).mean()),
        tie=float((margin == 0).mean()),
        mean_margin=float(margin.mean()), margin_sd=float(margin.std()),
        p10_margin=float(np.percentile(margin, 10)),
        p90_margin=float(np.percentile(margin, 90)),
    )


def lineup_table(table: pd.DataFrame, player_ids: list) -> pd.DataFrame:
    """The board rows for a lineup, in projection order."""
    t = table[table["player_id"].isin(player_ids)]
    return t.sort_values("Proj", ascending=False)[
        ["Player", "Team", "Pos", "Opp", "Proj", "Floor", "Ceiling"]].reset_index(drop=True)
