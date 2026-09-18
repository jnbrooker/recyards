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
_K = dict(fg_0_39=3.0, fg_40_49=4.0, fg_50=5.0, fg_miss=0.0, xp=1.0)     # kicker scoring, every preset
_D = dict(dst_sack=1.0, dst_int=2.0, dst_fum=2.0, dst_td=6.0, dst_safety=2.0, dst_block=2.0)   # D/ST events
# points allowed, the standard tiers: (up to, points)
DST_PA_TIERS = ((0, 10.0), (6, 7.0), (13, 4.0), (20, 1.0), (27, 0.0), (34, -1.0), (999, -4.0))
PRESETS = {
    "PPR": dict(rec=1.0, rec_yd=0.1, rec_td=6.0, rush_yd=0.1, rush_td=6.0,
                pass_yd=0.04, pass_td=4.0, int=-2.0, **_K, **_D),
    "Half PPR": dict(rec=0.5, rec_yd=0.1, rec_td=6.0, rush_yd=0.1, rush_td=6.0,
                     pass_yd=0.04, pass_td=4.0, int=-2.0, **_K, **_D),
    "Standard": dict(rec=0.0, rec_yd=0.1, rec_td=6.0, rush_yd=0.1, rush_td=6.0,
                     pass_yd=0.04, pass_td=4.0, int=-2.0, **_K, **_D),
    "PPR, 6-pt pass TD": dict(rec=1.0, rec_yd=0.1, rec_td=6.0, rush_yd=0.1, rush_td=6.0,
                              pass_yd=0.04, pass_td=6.0, int=-2.0, **_K, **_D),
}

COMPONENT_LABELS = {
    "rec": "Receptions", "rec_yd": "Rec yards", "rec_td": "Rec TD",
    "rush_yd": "Rush yards", "rush_td": "Rush TD",
    "pass_yd": "Pass yards", "pass_td": "Pass TD", "int": "INT",
    "fg_0_39": "FG 0-39", "fg_40_49": "FG 40-49", "fg_50": "FG 50+", "fg_miss": "FG missed", "xp": "XP",
    "dst_sack": "D/ST sack", "dst_int": "D/ST INT", "dst_fum": "D/ST fumble rec", "dst_td": "D/ST TD",
    "dst_safety": "D/ST safety", "dst_block": "D/ST blocked kick", "dst_pa": "D/ST points allowed",
}
KICKER_KEYS = ("fg_0_39", "fg_40_49", "fg_50", "fg_miss", "xp")
DST_KEYS = ("dst_sack", "dst_int", "dst_fum", "dst_td", "dst_safety", "dst_block")


def pa_points(pa: np.ndarray) -> np.ndarray:
    """Points-allowed tier points for an array of points allowed."""
    out = np.full(len(pa), DST_PA_TIERS[-1][1])
    for upto, pts in reversed(DST_PA_TIERS):
        out = np.where(pa <= upto, pts, out)
    return out


def dst_points(dst: dict, rules: dict) -> tuple[np.ndarray, dict]:
    """Per-simulation points for one team's defence / special teams."""
    r = {k: rules.get(k, _D[k]) for k in _D}
    comp = {
        "dst_sack": r["dst_sack"] * dst["sacks"], "dst_int": r["dst_int"] * dst["ints"],
        "dst_fum": r["dst_fum"] * dst["fum"], "dst_td": r["dst_td"] * dst["td"],
        "dst_safety": r["dst_safety"] * dst["safeties"], "dst_block": r["dst_block"] * dst["blocks"],
        "dst_pa": pa_points(np.asarray(dst["pa"])),
    }
    return sum(comp.values()).astype(float), comp
KICKER_RULES = dict(fg_0_39=3.0, fg_40_49=4.0, fg_50=5.0, fg_miss=0.0, xp=1.0)


def kicker_points(kick: dict, rules: dict) -> tuple[np.ndarray, dict]:
    """Per-simulation points for one team's kicker from his made / attempted
    field goals by band and extra points."""
    r = {k: rules.get(k, KICKER_RULES[k]) for k in KICKER_RULES}
    comp = {
        "fg_0_39": r["fg_0_39"] * kick["fg_made"][:, 0],
        "fg_40_49": r["fg_40_49"] * kick["fg_made"][:, 1],
        "fg_50": r["fg_50"] * kick["fg_made"][:, 2],
        "fg_miss": r["fg_miss"] * (kick["fg_att"] - kick["fg_made"]).sum(axis=1),
        "xp": r["xp"] * kick["xp_made"],
    }
    return sum(comp.values()).astype(float), comp


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
    # the kickers, one per team, from the game's field goals and tries
    for team, opp, kick in ((sim["team_a"], sim["team_b"], sim.get("kick_a")),
                            (sim["team_b"], sim["team_a"], sim.get("kick_b"))):
        if not kick or kick.get("player_id") is None:
            continue
        p, comp = kicker_points(kick, rules)
        row = dict(player_id=kick["player_id"], Player=kick["name"], Team=team, Opp=opp, Pos="K", Depth=1,
                   Proj=float(p.mean()), Floor=float(np.percentile(p, 10)), Median=float(np.median(p)),
                   Ceiling=float(np.percentile(p, 90)), P20=float((p >= 20).mean()), Source="kicks",
                   game_id=game_id, week=week,
                   FGA=float(kick["fg_att"].sum(axis=1).mean()), FGM=float(kick["fg_made"].sum(axis=1).mean()),
                   FG50=float(kick["fg_made"][:, 2].mean()), XP=float(kick["xp_made"].mean()))
        for key, arr in comp.items():
            row[f"pts_{key}"] = float(arr.mean())
        rows.append(row)
        samples[kick["player_id"]] = p.astype(np.float32)
    # the defences, one per team
    for team, opp, dst in ((sim["team_a"], sim["team_b"], sim.get("dst_a")),
                           (sim["team_b"], sim["team_a"], sim.get("dst_b"))):
        if not dst:
            continue
        p, comp = dst_points(dst, rules)
        pid = f"DST_{team}"
        row = dict(player_id=pid, Player=f"{team} D/ST", Team=team, Opp=opp, Pos="DST", Depth=1,
                   Proj=float(p.mean()), Floor=float(np.percentile(p, 10)), Median=float(np.median(p)),
                   Ceiling=float(np.percentile(p, 90)), P20=float((p >= 20).mean()), Source="game",
                   game_id=game_id, week=week,
                   Sacks=float(np.mean(dst["sacks"])), INTs=float(np.mean(dst["ints"])), FumRec=float(np.mean(dst["fum"])),
                   DefTD=float(np.mean(dst["td"])), Safeties=float(np.mean(dst["safeties"])), Blocks=float(np.mean(dst["blocks"])),
                   PA=float(np.mean(dst["pa"])))
        for key, arr in comp.items():
            row[f"pts_{key}"] = float(arr.mean())
        rows.append(row)
        samples[pid] = p.astype(np.float32)
    return pd.DataFrame(rows), samples


def week_projections(ctx: dict, games: pd.DataFrame, rules: dict,
                     n_sims: int = 10000, use_injuries: bool = True,
                     seed: int = 11, engine: str = "drive", progress=None) -> tuple[pd.DataFrame, dict, list]:
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

    n_games = max(len(games), 1)
    for i, g in games.reset_index(drop=True).iterrows():
        home, away = g["home_team"], g["away_team"]
        sub = (lambda f, t, i=i: progress((i + f) / n_games, f"{away} @ {home}")) if progress else None
        if progress:
            progress(i / n_games, f"{away} @ {home}")
        try:
            if home not in ratings["off"].index or away not in ratings["off"].index:
                raise ValueError("no rating")
            sim = G.run_game(ctx, roster(home), roster(away), home, away, n_sims=n_sims,
                             seed=seed + i, home="a", avail=ctx.get("avail"),
                             wind=g.get("wind"), roof=g.get("roof"), engine=engine, progress=sub)
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
    if row.get("Pos") == "DST":
        items = [
            ("Sacks", row.get("Sacks", 0.0), "dst_sack", ""), ("Interceptions", row.get("INTs", 0.0), "dst_int", ""),
            ("Fumbles recovered", row.get("FumRec", 0.0), "dst_fum", ""), ("Defensive / return TD", row.get("DefTD", 0.0), "dst_td", ""),
            ("Safeties", row.get("Safeties", 0.0), "dst_safety", ""), ("Blocked kicks", row.get("Blocks", 0.0), "dst_block", ""),
            ("Points allowed", row.get("PA", 0.0), "dst_pa", ""),
        ]
    if row.get("Pos") == "K":
        items = [
            ("FG made, to 39", row.get("pts_fg_0_39", 0.0) / 3.0 if row.get("pts_fg_0_39") else 0.0, "fg_0_39", ""),
            ("FG made, 40-49", row.get("pts_fg_40_49", 0.0) / 4.0 if row.get("pts_fg_40_49") else 0.0, "fg_40_49", ""),
            ("FG made, 50+", row.get("FG50", 0.0), "fg_50", ""),
            ("FG missed", row.get("FGA", 0.0) - row.get("FGM", 0.0), "fg_miss", ""),
            ("Extra points", row.get("XP", 0.0), "xp", ""),
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
