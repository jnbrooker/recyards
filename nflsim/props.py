"""
nflsim/props.py — player prop lines, frozen model predictions, and grading.

The schedule feed carries game lines only; player props come from The Odds
API (the-odds-api.com), which the app calls ONLY when the user presses fetch.
Everything fetched goes into a ledger (`props/ledger.csv`) so the history
accumulates from the first fetch — the free tier has no historical props.

Honest evaluation needs two things this module enforces:

  * **Predictions are frozen at fetch time.** The Game-view projection (mean,
    median, P(over) at the line) is computed when the line is recorded and
    never recomputed, so later injury news cannot leak into a "prediction".
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
    "pred_mean", "pred_median", "p_over", "predicted_at",
    "actual", "result",
]


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


def _norm(name: str) -> str:
    s = re.sub(r"[.\'’]", "", str(name).lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def match_players(lines: pd.DataFrame, rosters: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Attach player_id / team by name, preferring the two teams' rosters."""
    lines = lines.copy()
    both = rosters[rosters["team"].isin([home, away])]
    lookup = {_norm(n): (p, t) for n, p, t in zip(both["name"], both["player_id"], both["team"])}
    league = {_norm(n): (p, t) for n, p, t in zip(rosters["name"], rosters["player_id"], rosters["team"])}
    ids, teams = [], []
    for nm in lines["player"]:
        hit = lookup.get(_norm(nm)) or league.get(_norm(nm))
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
        merged = new.merge(led[_KEY + ["line", "pred_mean", "pred_median", "p_over", "predicted_at"]],
                           on=_KEY, how="left", suffixes=("", "_old"))
        same = merged["line"] == merged["line_old"]
        for c in ("pred_mean", "pred_median", "p_over", "predicted_at"):
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

def _samples(ctx: dict, market: str, row: pd.Series, n_sims: int = 20000):
    from . import game as G, roster as RO, rushing as R, touchdowns as TDm, ui as UI
    import model as M
    team, opp = row["team"], (row["away"] if row["team"] == row["home"] else row["home"])
    is_home = row["team"] == row["home"]
    roster = RO.roster_for(ctx, team, True)
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


def predict_missing(led: pd.DataFrame, ctx: dict, progress=None) -> pd.DataFrame:
    """Freeze Game-view predictions for ledger rows that have none."""
    led = led.copy()
    led["predicted_at"] = led["predicted_at"].astype(object)
    todo = led.index[led["pred_mean"].isna() & led["player_id"].notna()]
    cache = {}
    for i, idx in enumerate(todo):
        row = led.loc[idx]
        if progress:
            progress(i / max(len(todo), 1), f"{row['player']} {MARKETS[row['market']][0]}")
        key = (row["game_id"], row["market"], str(row["player_id"]))
        if key not in cache:
            try:
                cache[key] = _samples(ctx, row["market"], row)
            except Exception:
                cache[key] = None
        x = cache[key]
        if x is None:
            continue
        x = np.asarray(x, float)
        line = float(row["line"])
        led.loc[idx, "pred_mean"] = float(x.mean())
        led.loc[idx, "pred_median"] = float(np.median(x))
        led.loc[idx, "p_over"] = float((x >= 1).mean()) if row["market"] == "player_anytime_td" else float((x > line).mean())
        led.loc[idx, "predicted_at"] = pd.Timestamp.now(tz="UTC")
    return led


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


def grade(led: pd.DataFrame, use: str = "median", edge: float = 0.03) -> pd.DataFrame:
    """Per-row grading columns: the book's vig-free P(over), the model's edge,
    the model's pick (over / under / none) and whether it hit."""
    d = led.copy()
    po, pu = d["over_price"].map(implied_prob), d["under_price"].map(implied_prob)
    tot = po + pu
    d["book_p_over"] = np.where(tot > 0, po / tot, np.nan)
    d["edge"] = d["p_over"] - d["book_p_over"]
    centre = d["pred_median"] if use == "median" else d["pred_mean"]
    d["pick"] = np.where(d["edge"] > edge, "over", np.where(d["edge"] < -edge, "under", "none"))
    d["centre_pick"] = np.where(centre > d["line"], "over", np.where(centre < d["line"], "under", "none"))
    settled = d["result"].isin(["over", "under"])
    d["hit"] = np.where(settled & (d["pick"] != "none"), d["pick"] == d["result"], np.nan)
    d["centre_hit"] = np.where(settled & (d["centre_pick"] != "none"), d["centre_pick"] == d["result"], np.nan)
    d["book_hit"] = np.where(settled, np.where(d["book_p_over"] > 0.5, "over", "under") == d["result"], np.nan)
    return d


def summary(g: pd.DataFrame) -> dict:
    """Headline numbers over settled rows."""
    s = g[g["result"].isin(["over", "under"])]
    if s.empty:
        return dict(settled=0)
    y = (s["result"] == "over").astype(float)
    out = dict(settled=int(len(s)), lines=int(len(g)),
               model_brier=float(((s["p_over"] - y) ** 2).mean()),
               book_brier=float(((s["book_p_over"] - y) ** 2).mean()),
               centre_hit=float(s["centre_hit"].dropna().astype(float).mean()) if s["centre_hit"].notna().any() else np.nan,
               centre_n=int(s["centre_hit"].notna().sum()),
               pick_hit=float(s["hit"].dropna().astype(float).mean()) if s["hit"].notna().any() else np.nan,
               pick_n=int(s["hit"].notna().sum()),
               book_hit=float(s["book_hit"].dropna().astype(float).mean()),
               over_rate=float(y.mean()), model_over=float((s["p_over"] > 0.5).mean()))
    yards = s[s["market"].isin(["player_rush_yds", "player_reception_yds"])]
    if not yards.empty:
        out["mae_mean"] = float((yards["pred_mean"] - yards["actual"]).abs().mean())
        out["mae_median"] = float((yards["pred_median"] - yards["actual"]).abs().mean())
        out["mae_line"] = float((yards["line"] - yards["actual"]).abs().mean())
    return out
