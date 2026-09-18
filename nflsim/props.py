"""
nflsim/props.py — player prop lines, frozen model predictions, and grading.

The schedule feed carries game lines only; player props come from The Odds
API (the-odds-api.com), which the app calls ONLY when the user presses fetch.
Everything fetched goes into a ledger (`props/ledger.csv`) so the history
accumulates from the first fetch — the free tier has no historical props.

Honest evaluation needs two things this module enforces:

  * **Predictions are frozen at fetch time.** Three projections of every line
    (mean, median, P(over)) are computed when the line is recorded and never
    recomputed, so later injury news cannot leak into a "prediction": the
    drive engine's (`drive_*`) and the play engine's (`play_*`), each one
    simulated game per fixture with every player read off the box score —
    the thing the pages ship — and the single-stat Game view (`pred_*`, the
    player pages' own path). The engines are judged on identical lines;
    `ui.DEFAULT_ENGINE` says which one the page grades by default.
  * **The book is the benchmark.** Each line comes with its over/under prices;
    the implied (vig-free) probability is what the model's P(over) is scored
    against, and a side is only "the model's pick" when it disagrees with the
    price by a margin.

Credit control: the events list is free; each event-odds call costs
(markets x regions) credits, so two markets for one week's games is ~32 of the
500/month free tier. `fetch_week` only requests games kicking off within
`days_ahead`, skips (game, market) pairs already in the ledger unless
`refresh`, and reports the remaining quota the API returns with every call.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D

LEDGER = Path(__file__).resolve().parent.parent / "props" / "ledger.csv"

API = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"

# market key -> (label, the weekly-feed column that settles it, kind)
MARKETS = {
    "player_rush_yds": ("Rushing yards", "rushing_yards", "yards"),
    "player_reception_yds": ("Receiving yards", "receiving_yards", "yards"),
    "player_receptions": ("Receptions", "receptions", "count"),
    "player_anytime_td": ("Anytime TD", "tds", "anytime"),
}
DEFAULT_MARKETS = ("player_rush_yds", "player_reception_yds")

# The Odds API names teams in full; nflverse uses abbreviations.
TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

LEDGER_COLS = [
    "season", "week", "game_id", "home", "away", "commence", "event_id",
    "bookmaker", "market", "player", "player_id", "team", "line",
    "over_price", "under_price", "fetched_at",
    "pred_mean", "pred_median", "p_over", "predicted_at", "model_version",
    "drive_mean", "drive_median", "drive_p_over",
    "play_mean", "play_median", "play_p_over",
    "actual", "result",
]

# engine -> its frozen (mean, median, P(over)) columns
ENGINES = {
    "drive": ("drive_mean", "drive_median", "drive_p_over"),
    "play": ("play_mean", "play_median", "play_p_over"),
    "game": ("pred_mean", "pred_median", "p_over"),
}
ENGINE_LABELS = {"drive": "Drive engine", "play": "Play engine", "game": "Game view"}
# sims per fixture when freezing: the play engine runs ~700 games a second, so
# 10,000 is ~15 s (a 16-game week in ~4 minutes) and puts the Monte-Carlo
# error on P(over) at 0.5%; the drive engine is quick enough for 20,000
ENGINE_SIMS = {"drive": 20000, "play": 10000}


def model_version() -> str:
    """A short hash of the model's source. Stored with every prediction so
    that, after a code change, lines whose games have not started are
    re-projected automatically and lines already started or settled keep the
    prediction they were graded on."""
    root = Path(__file__).resolve().parent
    h = hashlib.sha1()
    for f in sorted(list(root.glob("*.py")) + [root.parent / "model.py"]):
        if f.name in ("props.py", "backtest.py", "calibrate.py", "ui.py"):
            continue
        h.update(f.read_bytes())
    return h.hexdigest()[:10]


# ---------------------------------------------------------------------------
# The Odds API
# ---------------------------------------------------------------------------

def _get(url: str, params: dict) -> tuple[object, dict]:
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{q}", headers={"User-Agent": "nflsim"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = json.loads(r.read().decode("utf-8"))
        quota = dict(remaining=r.headers.get("x-requests-remaining"),
                     used=r.headers.get("x-requests-used"),
                     last_cost=r.headers.get("x-requests-last"))
    return body, quota


def list_events(api_key: str, days_ahead: int = 7) -> tuple[pd.DataFrame, dict]:
    """Upcoming NFL events within `days_ahead` (this call is free)."""
    body, quota = _get(f"{API}/sports/{SPORT}/events", dict(apiKey=api_key))
    now = _dt.datetime.now(_dt.timezone.utc)
    rows = []
    for e in body:
        t = pd.to_datetime(e["commence_time"], utc=True)
        if now - _dt.timedelta(hours=6) <= t <= now + _dt.timedelta(days=days_ahead):
            rows.append(dict(event_id=e["id"], commence=t,
                             home=TEAM_ABBR.get(e["home_team"], e["home_team"]),
                             away=TEAM_ABBR.get(e["away_team"], e["away_team"])))
    return pd.DataFrame(rows), quota


def fetch_event_props(api_key: str, event_id: str, markets=DEFAULT_MARKETS,
                      regions: str = "us", bookmakers: str | None = None) -> tuple[pd.DataFrame, dict]:
    """One event's player-prop lines, every bookmaker returned. Costs
    len(markets) x regions credits."""
    params = dict(apiKey=api_key, markets=",".join(markets), oddsFormat="american")
    if bookmakers:
        params["bookmakers"] = bookmakers
    else:
        params["regions"] = regions
    body, quota = _get(f"{API}/sports/{SPORT}/events/{event_id}/odds", params)
    rows = []
    for bk in body.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m["key"] not in MARKETS:
                continue
            # pair Over/Under (or Yes for anytime TD) by player
            by_player = {}
            for o in m.get("outcomes", []):
                name = o.get("description") or o.get("name")
                d = by_player.setdefault(name, dict(line=o.get("point"), over=None, under=None))
                if o.get("name") in ("Over", "Yes"):
                    d["over"] = o.get("price"); d["line"] = o.get("point", d["line"])
                elif o.get("name") in ("Under", "No"):
                    d["under"] = o.get("price")
            for player, d in by_player.items():
                rows.append(dict(event_id=event_id, bookmaker=bk["key"], market=m["key"],
                                 player=player, line=d["line"] if d["line"] is not None else 0.5,
                                 over_price=d["over"], under_price=d["under"]))
    return pd.DataFrame(rows), quota


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def load_ledger() -> pd.DataFrame:
    if not LEDGER.exists():
        return pd.DataFrame(columns=LEDGER_COLS)
    d = pd.read_csv(LEDGER)
    for c in LEDGER_COLS:
        if c not in d.columns:
            d[c] = np.nan
    return d[LEDGER_COLS]


def save_ledger(d: pd.DataFrame) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    d[LEDGER_COLS].to_csv(LEDGER, index=False)


_KEY = ["season", "week", "game_id", "bookmaker", "market", "player"]
_PRED_COLS = ("pred_mean", "pred_median", "p_over", "predicted_at",
              "drive_mean", "drive_median", "drive_p_over",
              "play_mean", "play_median", "play_p_over")


# Book names that are not the nflverse display name (looked up after normalising).
_NICKNAMES = {
    "joshua palmer": "josh palmer", "hollywood brown": "marquise brown",
    "gabe davis": "gabriel davis", "chig okonkwo": "chigoziem okonkwo",
}

_FIRST_NAMES = {"joshua": "josh", "michael": "mike", "matthew": "matt", "nicholas": "nick",
                "christopher": "chris", "cameron": "cam", "zachary": "zach", "jonathan": "jon",
                "kenneth": "ken", "alexander": "alex", "daniel": "dan", "robert": "rob",
                "william": "will", "benjamin": "ben", "anthony": "tony", "samuel": "sam"}


def _norm(name: str) -> str:
    s = re.sub(r"[.\'’]", "", str(name).lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _NICKNAMES.get(s, s)


def _loose(name: str) -> str:
    """Second-chance key: hyphens dropped, long first names shortened."""
    parts = _norm(name).replace("-", " ").split(" ")
    if parts:
        parts[0] = _FIRST_NAMES.get(parts[0], parts[0])
    return " ".join(parts)


def match_players(lines: pd.DataFrame, rosters: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Attach player_id / team by name, preferring the two teams' rosters."""
    lines = lines.copy()
    both = rosters[rosters["team"].isin([home, away])]
    lookup = {_norm(n): (p, t) for n, p, t in zip(both["name"], both["player_id"], both["team"])}
    league = {_norm(n): (p, t) for n, p, t in zip(rosters["name"], rosters["player_id"], rosters["team"])}
    loose = {_loose(n): (p, t) for n, p, t in zip(both["name"], both["player_id"], both["team"])}
    ids, teams = [], []
    for nm in lines["player"]:
        hit = lookup.get(_norm(nm)) or league.get(_norm(nm)) or loose.get(_loose(nm))
        ids.append(hit[0] if hit else None); teams.append(hit[1] if hit else None)
    lines["player_id"] = ids
    lines["team"] = teams
    return lines


def fetch_week(api_key: str, ctx: dict, sched: pd.DataFrame, rosters: pd.DataFrame,
               markets=DEFAULT_MARKETS, days_ahead: int = 7, refresh: bool = False,
               regions: str = "us", progress=None) -> tuple[pd.DataFrame, dict, list[str]]:
    """Fetch this week's lines into the ledger, freeze predictions for new
    rows, return (ledger, quota, log)."""
    log = []
    events, quota = list_events(api_key, days_ahead)
    if events.empty:
        return load_ledger(), quota, ["No NFL events within the window."]
    led = load_ledger()
    season = int(ctx["depth_seasons"][-1])
    new_rows = []
    for k, ev in events.iterrows():
        g = sched[(sched["home_team"] == ev["home"]) & (sched["away_team"] == ev["away"])]
        g = g[~g["played"]] if "played" in g.columns else g
        if g.empty:
            log.append(f"{ev['away']} @ {ev['home']}: not in the schedule, skipped"); continue
        g = g.iloc[0]
        have = led[(led["game_id"] == g["game_id"])]["market"].unique().tolist() if not led.empty else []
        want = [m for m in markets if refresh or m not in have]
        if not want:
            log.append(f"{ev['away']} @ {ev['home']}: already in the ledger, skipped"); continue
        if progress:
            progress(k / max(len(events), 1), f"{ev['away']} @ {ev['home']}")
        try:
            lines, quota = fetch_event_props(api_key, ev["event_id"], want, regions)
        except Exception as e:
            log.append(f"{ev['away']} @ {ev['home']}: fetch failed ({e})"); continue
        if lines.empty:
            log.append(f"{ev['away']} @ {ev['home']}: no prop markets posted yet"); continue
        lines = match_players(lines, rosters, ev["home"], ev["away"])
        lines["season"] = season; lines["week"] = int(g["week"]); lines["game_id"] = g["game_id"]
        lines["home"] = ev["home"]; lines["away"] = ev["away"]; lines["commence"] = ev["commence"]
        lines["fetched_at"] = pd.Timestamp.now(tz="UTC")
        unmatched = int(lines["player_id"].isna().sum())
        log.append(f"{ev['away']} @ {ev['home']}: {len(lines)} lines across "
                   f"{lines['bookmaker'].nunique()} books" + (f", {unmatched} players unmatched" if unmatched else ""))
        new_rows.append(lines)
    if not new_rows:
        return led, quota, log
    new = pd.concat(new_rows, ignore_index=True)
    for c in LEDGER_COLS:
        if c not in new.columns:
            new[c] = np.nan
    # a refetch replaces the same key (closing line wins); keep frozen predictions
    # from earlier rows only if the line is unchanged
    if not led.empty:
        merged = new.merge(led[_KEY + ["line"] + list(_PRED_COLS)],
                           on=_KEY, how="left", suffixes=("", "_old"))
        same = merged["line"] == merged["line_old"]
        for c in _PRED_COLS:
            merged[c] = np.where(same, merged[c + "_old"], np.nan)
        new = merged[LEDGER_COLS]
        led = led.merge(new[_KEY], on=_KEY, how="left", indicator=True)
        led = led[led["_merge"] == "left_only"].drop(columns="_merge")
    led = pd.concat([led, new[LEDGER_COLS]], ignore_index=True)
    led = predict_missing(led, ctx)
    save_ledger(led)
    return led, quota, log


# ---------------------------------------------------------------------------
# Frozen predictions (Game view, exactly as the pages compute them)
# ---------------------------------------------------------------------------

def _samples(ctx: dict, market: str, row: pd.Series, n_sims: int = 20000,
             memo: dict | None = None):
    """Simulated samples for one ledger row. `memo` (per prediction pass) holds
    each team's roster so a slate of 100 lines builds two rosters, not 100."""
    from . import game as G, roster as RO, rushing as R, touchdowns as TDm, ui as UI
    import model as M
    team, opp = row["team"], (row["away"] if row["team"] == row["home"] else row["home"])
    is_home = row["team"] == row["home"]
    memo = memo if memo is not None else {}
    if ("roster", team) not in memo:
        memo[("roster", team)] = RO.roster_for(ctx, team, True)
    roster = memo[("roster", team)]
    rr = roster[roster["player_id"].astype(str) == str(row["player_id"])]
    if rr.empty:
        return None
    live = rr.iloc[0]
    f = G.script_factors(ctx, team, opp, home="a" if is_home else "b")
    wk = ctx["wk"]
    rng_seed = 7
    if market == "player_rush_yds":
        try:
            pri = R.player_rush_priors(wk, str(row["player_id"]), ctx["pfr_agg"], ctx["pfr_lg"])
        except ValueError:
            pri = RO.rushing_priors_from_role(live, ctx["pfr_lg"])
        pri["mu_share"] = UI.live_share(live, "carry_share")
        return R.simulate(pri, f["carries"], ctx["rush_def"].get(opp), n_sims=n_sims, seed=rng_seed)["yards"]
    if market in ("player_reception_yds", "player_receptions"):
        rec = wk[wk["position"].isin(M.RECEIVING_POSITIONS)]
        try:
            pri = M.player_priors(rec, str(row["player_id"]))
        except ValueError:
            pri = RO.receiving_priors_from_role(rec, live)
        pri["mu_ts"] = UI.live_share(live, "target_share")
        tv = M.team_pass_volume(rec).get(team, M.team_pass_volume(rec)["_LEAGUE_"])
        k = f["dropbacks"][0] / max(f["dropbacks_typical"], 1e-6)
        sim = M.simulate(pri, (tv[0] * k, tv[1] * k), M.defense_profiles(rec).get((opp, pri["position"])),
                         n_sims=n_sims, seed=rng_seed)
        return sim["yards"] if market == "player_reception_yds" else sim["receptions"]
    if market == "player_anytime_td":
        lg = TDm.league_td_rates(wk)
        gl = ctx.get("goal_line") or None
        try:
            pri = TDm.player_td_priors(wk, str(row["player_id"]), lg, gl=gl)
        except ValueError:
            pri = RO.td_priors_from_role(live, ctx["team_vol"], lg)
        tv = ctx["team_vol"].get(team, ctx["team_vol"]["_LEAGUE_"])
        pri = RO.scale_volume_priors(pri, "mu_rec", "var_rec",
                                     UI.live_share(live, "target_share") * tv["targets"] * float(live["catch_rate"]) * f["dropbacks"][0] / max(f["dropbacks_typical"], 1e-6))
        pri = RO.scale_volume_priors(pri, "mu_car", "var_car",
                                     UI.live_share(live, "carry_share") * f["carries"][0])
        pri["p_rec_td"] = float(np.clip(pri["p_rec_td"] * f["td_factor"], 0, 0.5))
        pri["p_rush_td"] = float(np.clip(pri["p_rush_td"] * f["td_factor"], 0, 0.4))
        return TDm.simulate(pri, TDm.td_defense_profiles(wk).get((opp, pri["position"])), n_sims=n_sims, seed=rng_seed)["total"]
    return None


def _engine_box(ctx: dict, row: pd.Series, memo: dict, engine: str) -> dict:
    """One engine's box arrays for one fixture, simulated once per prediction
    pass (memoised on game_id): {team: box} for both sides."""
    from . import game as G, roster as RO
    key = (engine, "box", row["game_id"])
    if key not in memo:
        home, away = row["home"], row["away"]
        for team in (home, away):
            if ("roster", team) not in memo:
                memo[("roster", team)] = RO.roster_for(ctx, team, True)
        # the engines take the active roster only, in the order they index
        # the box arrays
        ra = memo[("roster", home)]; ra = ra[ra["active"]].reset_index(drop=True)
        rb = memo[("roster", away)]; rb = rb[rb["active"]].reset_index(drop=True)
        sim = G.run_game(ctx, ra, rb, home, away, n_sims=ENGINE_SIMS[engine], seed=7, home="a",
                         avail=ctx.get("avail"), engine=engine)
        memo[key] = {home: sim["box_a"], away: sim["box_b"]}
    return memo[key]


def _box_samples(ctx: dict, market: str, row: pd.Series, memo: dict, engine: str):
    """Simulated samples for one ledger row from a game engine: the player's
    column of his team's box score across every simulated game. Weather is
    left out, as it is for the Game-view samples, so every projection sees
    identical inputs."""
    box = _engine_box(ctx, row, memo, engine).get(row["team"])
    if box is None:
        return None                     # matched to a roster outside this game
    r = box["roster"].reset_index(drop=True)
    j = np.flatnonzero(r["player_id"].astype(str).values == str(row["player_id"]))
    if not len(j):
        return None                     # ruled out: not on the active roster
    j = int(j[0])
    if market == "player_rush_yds":
        return box["rush_yards"][:, j]
    if market == "player_reception_yds":
        return box["rec_yards"][:, j]
    if market == "player_receptions":
        return box["receptions"][:, j]
    if market == "player_anytime_td":
        return box["rec_tds"][:, j] + box["rush_tds"][:, j]
    return None


_SAMPLERS = {
    "game": _samples,
    "drive": lambda ctx, market, row, memo: _box_samples(ctx, market, row, memo, "drive"),
    "play": lambda ctx, market, row, memo: _box_samples(ctx, market, row, memo, "play"),
}


def rematch(led: pd.DataFrame, rosters: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Second pass over rows the fetch could not attach to a player (book
    nicknames, hyphens): returns (ledger, rows newly matched). Only rows with
    no player_id are touched, so settled rows never move."""
    led = led.copy()
    miss = led["player_id"].isna()
    if not miss.any() or rosters is None or rosters.empty:
        return led, 0
    n = 0
    for (home, away), g in led[miss].groupby(["home", "away"]):
        m = match_players(g, rosters, home, away)
        got = m["player_id"].notna()
        led.loc[m.index[got], "player_id"] = m.loc[got, "player_id"]
        led.loc[m.index[got], "team"] = m.loc[got, "team"]
        n += int(got.sum())
    return led, n


def predict_missing(led: pd.DataFrame, ctx: dict, progress=None,
                    engines: tuple = tuple(ENGINES)) -> pd.DataFrame:
    """Freeze predictions for ledger rows that have none — both game engines
    (one simulated game per fixture each) and the Game view, in one pass."""
    led = led.copy()
    led["predicted_at"] = led["predicted_at"].astype(object)
    led["model_version"] = led["model_version"].astype(object)
    version = model_version()
    # never freeze a prediction for a game that has started: a late match (see
    # `rematch`) on a played game stays unpredicted rather than graded in hindsight
    commence = pd.to_datetime(led["commence"], utc=True, errors="coerce")
    open_ = led["result"].isna() & (commence.isna() | (commence > pd.Timestamp.now(tz="UTC")))
    missing = np.zeros(len(led), bool)
    for eng in engines:
        missing |= led[ENGINES[eng][0]].isna().values
    todo = led.index[missing & led["player_id"].notna() & open_]
    cache, memo = {}, {}
    for i, idx in enumerate(todo):
        row = led.loc[idx]
        if progress:
            progress(i / max(len(todo), 1), f"{row['player']} {MARKETS[row['market']][0]}")
        for eng in engines:
            c_mean, c_median, c_p = ENGINES[eng]
            if pd.notna(row[c_mean]):
                continue
            key = (eng, row["game_id"], row["market"], str(row["player_id"]))
            if key not in cache:
                try:
                    cache[key] = _SAMPLERS[eng](ctx, row["market"], row, memo=memo)
                except Exception:
                    cache[key] = None
            x = cache[key]
            if x is None:
                continue
            x = np.asarray(x, float)
            line = float(row["line"])
            led.loc[idx, c_mean] = float(x.mean())
            led.loc[idx, c_median] = float(np.median(x))
            led.loc[idx, c_p] = float((x >= 1).mean()) if row["market"] == "player_anytime_td" else float((x > line).mean())
            led.loc[idx, "predicted_at"] = pd.Timestamp.now(tz="UTC")
            led.loc[idx, "model_version"] = version
    return led


def reproject_unplayed(led: pd.DataFrame, ctx: dict, progress=None,
                       only_stale: bool = False) -> tuple[pd.DataFrame, int]:
    """Re-freeze predictions for lines whose game has NOT kicked off (a model
    fix should reach them; anything started or settled stays as it was).
    `only_stale` limits it to predictions made by an older model version.
    Returns (ledger, rows re-projected)."""
    led = led.copy()
    now = pd.Timestamp.now(tz="UTC")
    commence = pd.to_datetime(led["commence"], utc=True, errors="coerce")
    open_ = led["result"].isna() & led["player_id"].notna() & (commence.isna() | (commence > now))
    if only_stale:
        open_ &= led["model_version"].astype(str) != model_version()
    if not open_.any():
        return led, 0
    for c in _PRED_COLS:
        led.loc[open_, c] = np.nan
    led = predict_missing(led, ctx, progress)
    return led, int(open_.sum())


# ---------------------------------------------------------------------------
# Actuals and grading
# ---------------------------------------------------------------------------

def fill_actuals(led: pd.DataFrame, wk: pd.DataFrame) -> pd.DataFrame:
    """Settle every row whose game is in the weekly feed."""
    led = led.copy()
    if wk is None or wk.empty or led.empty:
        return led
    led["result"] = led["result"].astype(object)
    led["actual"] = led["actual"].astype(float)
    w = wk.assign(tds=wk["receiving_tds"] + wk["rushing_tds"])
    key = w.set_index([w["player_id"].astype(str), w["season"].astype(int), w["week"].astype(int)])
    played = set(zip(w["recent_team"], w["season"].astype(int), w["week"].astype(int)))
    for idx, row in led[led["actual"].isna() & led["player_id"].notna()].iterrows():
        col = MARKETS[row["market"]][1]
        k = (str(row["player_id"]), int(row["season"]), int(row["week"]))
        if k in key.index:
            val = float(key.loc[k, col]) if not isinstance(key.loc[k, col], pd.Series) else float(key.loc[k, col].iloc[0])
        elif (row["team"], int(row["season"]), int(row["week"])) in played:
            led.loc[idx, "result"] = "void"     # his team played, he did not: no action
            continue
        else:
            continue
        led.loc[idx, "actual"] = val
        line = float(row["line"])
        if row["market"] == "player_anytime_td":
            led.loc[idx, "result"] = "over" if val >= 1 else "under"
        else:
            led.loc[idx, "result"] = "over" if val > line else "under" if val < line else "push"
    return led


def implied_prob(american) -> float:
    try:
        a = float(american)
    except (TypeError, ValueError):
        return np.nan
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def grade(led: pd.DataFrame, use: str = "median", edge: float = 0.03,
          engine: str = "drive") -> pd.DataFrame:
    """Per-row grading columns: the book's vig-free P(over), the model's edge,
    the model's pick (over / under / none) and whether it hit. `engine` picks
    which frozen projection is graded; its numbers are copied to `eng_mean`,
    `eng_median` and `eng_p` so the rest of the page needs no branching."""
    d = led.copy()
    c_mean, c_median, c_p = ENGINES[engine]
    d["eng_mean"], d["eng_median"], d["eng_p"] = d[c_mean], d[c_median], d[c_p]
    po, pu = d["over_price"].map(implied_prob), d["under_price"].map(implied_prob)
    tot = po + pu
    d["book_p_over"] = np.where(tot > 0, po / tot, np.nan)
    d["edge"] = d["eng_p"] - d["book_p_over"]
    centre = d["eng_median"] if use == "median" else d["eng_mean"]
    d["pick"] = np.where(d["edge"] > edge, "over", np.where(d["edge"] < -edge, "under", "none"))
    d["centre_pick"] = np.where(centre > d["line"], "over", np.where(centre < d["line"], "under", "none"))
    settled = d["result"].isin(["over", "under"]) & d["eng_p"].notna()
    d["hit"] = np.where(settled & (d["pick"] != "none"), d["pick"] == d["result"], np.nan)
    d["centre_hit"] = np.where(settled & (d["centre_pick"] != "none"), d["centre_pick"] == d["result"], np.nan)
    d["book_hit"] = np.where(settled, np.where(d["book_p_over"] > 0.5, "over", "under") == d["result"], np.nan)
    return d


def summary(g: pd.DataFrame) -> dict:
    """Headline numbers over settled rows of a graded frame."""
    s = g[g["result"].isin(["over", "under"]) & g["eng_p"].notna()]
    if s.empty:
        return dict(settled=0)
    y = (s["result"] == "over").astype(float)
    out = dict(settled=int(len(s)), lines=int(len(g)),
               model_brier=float(((s["eng_p"] - y) ** 2).mean()),
               book_brier=float(((s["book_p_over"] - y) ** 2).mean()),
               centre_hit=float(s["centre_hit"].dropna().astype(float).mean()) if s["centre_hit"].notna().any() else np.nan,
               centre_n=int(s["centre_hit"].notna().sum()),
               pick_hit=float(s["hit"].dropna().astype(float).mean()) if s["hit"].notna().any() else np.nan,
               pick_n=int(s["hit"].notna().sum()),
               book_hit=float(s["book_hit"].dropna().astype(float).mean()),
               over_rate=float(y.mean()), model_over=float((s["eng_p"] > 0.5).mean()))
    yards = s[s["market"].isin(["player_rush_yds", "player_reception_yds"])]
    if not yards.empty:
        out["mae_mean"] = float((yards["eng_mean"] - yards["actual"]).abs().mean())
        out["mae_median"] = float((yards["eng_median"] - yards["actual"]).abs().mean())
        out["mae_line"] = float((yards["line"] - yards["actual"]).abs().mean())
    return out


def compare_engines(d: pd.DataFrame, use: str = "median", edge: float = 0.03,
                    engines: tuple = ("drive", "play")) -> pd.DataFrame:
    """The engines' headline numbers on the SAME settled lines — only rows
    where every compared projection was frozen count, so none gets the easier
    subset. One row per engine, plus the book as the benchmark."""
    both = d
    for eng in engines:
        both = both[both[ENGINES[eng][2]].notna()]
    rows = []
    for eng in engines:
        s = summary(grade(both, use=use, edge=edge, engine=eng))
        if not s.get("settled"):
            continue
        rows.append(dict(engine=ENGINE_LABELS[eng], settled=s["settled"],
                         centre_hit=s["centre_hit"], centre_n=s["centre_n"],
                         pick_hit=s["pick_hit"], pick_n=s["pick_n"],
                         brier=s["model_brier"], mae_mean=s.get("mae_mean", np.nan),
                         mae_median=s.get("mae_median", np.nan), model_over=s["model_over"]))
    if rows:
        s = summary(grade(both, use=use, edge=edge, engine="drive"))
        rows.append(dict(engine="Book", settled=s["settled"], centre_hit=s["book_hit"], centre_n=s["settled"],
                         pick_hit=np.nan, pick_n=0, brier=s["book_brier"], mae_mean=s.get("mae_line", np.nan),
                         mae_median=s.get("mae_line", np.nan), model_over=np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Promotion: when does the play engine become the default?
# ---------------------------------------------------------------------------

PROMOTION_MIN_LINES = 600          # settled lines with both engines frozen (three to four weeks)
PROMOTION_CAL_BAND = (0.47, 0.53)  # share of actuals above the median, each yardage market
PROMOTION_TIER_SLACK = 0.02        # play may trail drive by this much in any line tier


def promotion_gate(d: pd.DataFrame, use: str = "median", edge: float = 0.03) -> pd.DataFrame:
    """The rule written down before the weeks came in (ROADMAP §17-18): the
    play engine holds the default while, on the same settled lines, it is at
    least as good as the drive engine on Brier and MAE, its median is
    calibrated on both yardage markets, and it is not worse in any line tier.
    One row per criterion with the numbers and whether it is met; a failed row
    is the tripwire to revisit `ui.DEFAULT_ENGINE`."""
    both = d[d["drive_p_over"].notna() & d["play_p_over"].notna() & d["result"].isin(["over", "under"])]
    rows = []
    n = len(both)
    rows.append(dict(criterion=f"settled lines with both engines frozen (≥ {PROMOTION_MIN_LINES})",
                     drive=np.nan, play=float(n), met=n >= PROMOTION_MIN_LINES))
    if n == 0:
        return pd.DataFrame(rows)
    gd, gp = grade(both, use, edge, "drive"), grade(both, use, edge, "play")
    sd, sp = summary(gd), summary(gp)
    rows.append(dict(criterion="Brier of P(over) (play ≤ drive)", drive=sd["model_brier"], play=sp["model_brier"],
                     met=sp["model_brier"] <= sd["model_brier"]))
    if "mae_median" in sd and "mae_median" in sp:
        rows.append(dict(criterion="MAE of the median, yards (play ≤ drive)", drive=sd["mae_median"], play=sp["mae_median"],
                         met=sp["mae_median"] <= sd["mae_median"]))
    lo, hi = PROMOTION_CAL_BAND
    for m, lab in (("player_reception_yds", "receiving"), ("player_rush_yds", "rushing")):
        q = both[both["market"] == m]
        if len(q) >= 50:
            cd = float((q["actual"] > q["drive_median"]).mean()); cp = float((q["actual"] > q["play_median"]).mean())
            rows.append(dict(criterion=f"actual above the median, {lab} ({lo:.2f}–{hi:.2f})", drive=cd, play=cp, met=lo <= cp <= hi))
    y = both[both["market"].isin(["player_reception_yds", "player_rush_yds"])].copy()
    if len(y) >= 100:
        y["tier"] = pd.cut(y["line"], [0, 30, 60, 500], labels=["under 30", "30–60", "60+"])
        for t, q in y.groupby("tier", observed=True):
            if len(q) < 30:
                continue
            hd = float(gd.loc[q.index, "centre_hit"].dropna().astype(float).mean())
            hp = float(gp.loc[q.index, "centre_hit"].dropna().astype(float).mean())
            rows.append(dict(criterion=f"{use} vs line hit rate, lines {t} (play ≥ drive − {PROMOTION_TIER_SLACK:.0%})",
                             drive=hd, play=hp, met=hp >= hd - PROMOTION_TIER_SLACK))
    return pd.DataFrame(rows)
