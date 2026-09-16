"""
nflsim/learn.py — a LEARNED mean for player stats, from model-only features.

The hand-built priors (share x volume x efficiency, each regressed) sit at the
level of a trailing average. Every projection system that does better does it
the same way: fit a model on years of player-weeks with features that are all
knowable before kickoff, and let the fitter decide what a rookie's route share,
a role change, an opponent's man-coverage rate or the quarterback's efficiency
is worth. The simulation engine then keeps its job — the SHAPE around the mean
(calibrated spread, correlations, box scores that reconcile) — centred on the
learned mean instead of the hand-built one.

No market inputs. Every feature is the model's own: the player's history at
three horizons, routes run and targets per route (participation feed), snap
share, the team's pass volume and rate, the opponent's pass defence (yards per
target allowed to the position, EPA per pass, man rate, pressure rate), the
quarterback's EPA per dropback, home/dome, and the week.

Validation is walk-forward by season: fit on every season before the test
season, score the test season. `experiment()` prints the comparison against
the trailing average and, where a harness run is supplied, the current model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D

BASE = "https://github.com/nflverse/nflverse-data/releases/download/"
HALF_LIVES = (3, 6, 12)
MIN_PRIOR_GAMES = 5


# ---------------------------------------------------------------------------
# Feeds
# ---------------------------------------------------------------------------

@D.ttl_cache(maxsize=2)
def play_features(seasons: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """From pbp + participation, per season-week:
      routes  — (season, week, posteam, pid, routes, tgt)
      defense — (season, week, defteam, man_rate, pressure_rate, epa_pass_allowed, dropbacks_faced)
      qb      — (season, week, posteam, passer, dropbacks, epa_db)
    """
    R, DF, QB = [], [], []
    for yr in seasons:
        try:
            pb = pd.read_parquet(BASE + f"pbp/play_by_play_{yr}.parquet",
                                 columns=["game_id", "play_id", "season", "week", "season_type",
                                          "play_type", "qb_dropback", "receiver_player_id",
                                          "passer_player_id", "posteam", "defteam", "epa"])
        except Exception:
            continue
        pb = pb[(pb["season_type"] == "REG") & (pb["qb_dropback"] == 1) & pb["posteam"].notna()]
        try:
            pt = pd.read_parquet(BASE + f"pbp_participation/pbp_participation_{yr}.parquet",
                                 columns=["nflverse_game_id", "play_id", "offense_players",
                                          "defense_man_zone_type", "was_pressure"])
            pt = pt.rename(columns={"nflverse_game_id": "game_id"})
            j = pb.merge(pt, on=["game_id", "play_id"], how="left")
        except Exception:
            j = pb.assign(offense_players=None, defense_man_zone_type=None, was_pressure=None)
        # routes: every offensive player on the field for a dropback
        have = j[j["offense_players"].notna() & (j["offense_players"] != "")]
        if len(have):
            ex = have.assign(pid=have["offense_players"].str.split(";")).explode("pid")
            routes = ex.groupby(["season", "week", "posteam", "pid"]).size().rename("routes").reset_index()
            R.append(routes)
        # defence: man rate, pressure rate, EPA per dropback allowed
        d = j.assign(man=(j["defense_man_zone_type"] == "MAN_COVERAGE").astype(float),
                     labelled=j["defense_man_zone_type"].isin(["MAN_COVERAGE", "ZONE_COVERAGE"]).astype(float),
                     pres=pd.to_numeric(j["was_pressure"], errors="coerce"))
        g = d.groupby(["season", "week", "defteam"]).agg(
            dropbacks_faced=("play_id", "size"), man_n=("man", "sum"), lab_n=("labelled", "sum"),
            pressure_rate=("pres", "mean"), epa_pass_allowed=("epa", "mean")).reset_index()
        g["man_rate"] = np.where(g["lab_n"] > 0, g["man_n"] / g["lab_n"].clip(lower=1), np.nan)
        DF.append(g[["season", "week", "defteam", "dropbacks_faced", "man_rate", "pressure_rate", "epa_pass_allowed"]])
        # quarterback: EPA per dropback by passer
        q = (j.dropna(subset=["passer_player_id"])
              .groupby(["season", "week", "posteam", "passer_player_id"])
              .agg(dropbacks=("play_id", "size"), epa_db=("epa", "mean")).reset_index()
              .rename(columns={"passer_player_id": "passer"}))
        QB.append(q)
    empty = pd.DataFrame()
    return (pd.concat(R, ignore_index=True) if R else empty,
            pd.concat(DF, ignore_index=True) if DF else empty,
            pd.concat(QB, ignore_index=True) if QB else empty)


@D.ttl_cache(maxsize=2)
def snap_shares(seasons: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for yr in seasons:
        try:
            s = pd.read_parquet(BASE + f"snap_counts/snap_counts_{yr}.parquet",
                                columns=["season", "week", "game_type", "pfr_player_id", "team",
                                         "offense_pct"])
            frames.append(s[s["game_type"] == "REG"])
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    s = pd.concat(frames, ignore_index=True)
    xw = D._pfr_id_crosswalk()
    if xw.empty:
        return pd.DataFrame()
    s = s.merge(xw.rename(columns={"pfr_id": "pfr_player_id", "gsis_id": "player_id"}),
                on="pfr_player_id", how="inner")
    return s[["season", "week", "player_id", "offense_pct"]]


# ---------------------------------------------------------------------------
# Feature table
# ---------------------------------------------------------------------------

def _ew_prev(s: pd.Series, h: float) -> pd.Series:
    """Exponentially weighted mean of PRIOR games (shifted so the current game
    is excluded), half-life in games."""
    return s.shift(1).ewm(halflife=h, ignore_na=True, min_periods=1).mean()


def build_table(seasons: tuple[int, ...], positions=("WR", "TE", "RB")) -> pd.DataFrame:
    """One row per player-game with pre-game features and the outcome."""
    wk = D.load_weekly(seasons)
    wk = wk[wk["position"].isin(positions)].copy()
    routes, dfn, qb = play_features(seasons)
    snaps = snap_shares(seasons)
    sched = D.load_schedule(seasons)

    # team-week context
    allk = D.load_weekly(seasons)
    team = (allk.groupby(["recent_team", "season", "week"])
                 .agg(team_tgt=("targets", "sum"), team_att=("attempts", "sum"),
                      team_car=("carries", "sum")).reset_index())
    team["pass_rate"] = team["team_att"] / (team["team_att"] + team["team_car"]).clip(lower=1)
    wk = wk.merge(team, on=["recent_team", "season", "week"], how="left")

    # routes / TPRR
    if not routes.empty:
        wk = wk.merge(routes.rename(columns={"posteam": "recent_team", "pid": "player_id"}),
                      on=["season", "week", "recent_team", "player_id"], how="left")
    else:
        wk["routes"] = np.nan
    db = (routes.groupby(["season", "week", "posteam"])["routes"].max().rename("team_db").reset_index()
          .rename(columns={"posteam": "recent_team"})) if not routes.empty else None
    if db is not None:
        wk = wk.merge(db, on=["season", "week", "recent_team"], how="left")
    else:
        wk["team_db"] = np.nan
    wk["route_pct"] = wk["routes"] / wk["team_db"].clip(lower=1)
    wk["tprr"] = wk["targets"] / wk["routes"].clip(lower=1)
    wk["yprr"] = wk["receiving_yards"] / wk["routes"].clip(lower=1)
    wk["tshare"] = wk["targets"] / wk["team_tgt"].clip(lower=1)
    wk["ypt"] = wk["receiving_yards"] / wk["targets"].clip(lower=1)
    wk["adot"] = wk["receiving_air_yards"] / wk["targets"].clip(lower=1)
    wk["catch"] = wk["receptions"] / wk["targets"].clip(lower=1)
    wk["cshare"] = wk["carries"] / wk["team_car"].clip(lower=1)

    if not snaps.empty:
        wk = wk.merge(snaps, on=["season", "week", "player_id"], how="left")
    else:
        wk["offense_pct"] = np.nan

    # the team's quarterback that week (most dropbacks) and his PRIOR efficiency
    if not qb.empty:
        qb = qb.sort_values("dropbacks", ascending=False).drop_duplicates(["season", "week", "posteam"])
        qb = qb.sort_values(["passer", "season", "week"])
        qb["qb_epa_prev"] = qb.groupby("passer")["epa_db"].transform(lambda s: _ew_prev(s, 8))
        qb["qb_games_prev"] = qb.groupby("passer").cumcount()
        wk = wk.merge(qb[["season", "week", "posteam", "qb_epa_prev", "qb_games_prev"]]
                      .rename(columns={"posteam": "recent_team"}),
                      on=["season", "week", "recent_team"], how="left")

    # opponent defence, as of prior weeks (EW of its weekly numbers)
    opp = (wk.groupby(["opponent_team", "season", "week", "position"])
             .agg(d_tgt=("targets", "sum"), d_yds=("receiving_yards", "sum")).reset_index())
    opp["d_ypt"] = opp["d_yds"] / opp["d_tgt"].clip(lower=1)
    opp = opp.sort_values(["opponent_team", "position", "season", "week"])
    opp["def_ypt_prev"] = opp.groupby(["opponent_team", "position"])["d_ypt"].transform(lambda s: _ew_prev(s, 8))
    wk = wk.merge(opp[["opponent_team", "season", "week", "position", "def_ypt_prev"]],
                  on=["opponent_team", "season", "week", "position"], how="left")
    if not dfn.empty:
        dfn = dfn.sort_values(["defteam", "season", "week"])
        for c in ("man_rate", "pressure_rate", "epa_pass_allowed"):
            dfn[f"def_{c}_prev"] = dfn.groupby("defteam")[c].transform(lambda s: _ew_prev(s, 8))
        wk = wk.merge(dfn[["defteam", "season", "week", "def_man_rate_prev", "def_pressure_rate_prev",
                           "def_epa_pass_allowed_prev"]].rename(columns={"defteam": "opponent_team"}),
                      on=["opponent_team", "season", "week"], how="left")

    # schedule: home, dome
    if not sched.empty:
        h = sched[["game_id", "season", "week", "home_team", "away_team", "roof"]]
        home = h.melt(id_vars=["season", "week", "roof"], value_vars=["home_team", "away_team"],
                      var_name="side", value_name="recent_team")
        home["is_home"] = (home["side"] == "home_team").astype(int)
        home["dome"] = home["roof"].isin(["dome", "closed"]).astype(int)
        wk = wk.merge(home[["season", "week", "recent_team", "is_home", "dome"]],
                      on=["season", "week", "recent_team"], how="left")

    # player history at several horizons (all prior games only)
    wk = wk.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    g = wk.groupby("player_id")
    wk["games_prev"] = g.cumcount()
    wk["first_of_season"] = (g["season"].shift(1) != wk["season"]).astype(int)
    for col in ("targets", "receptions", "receiving_yards", "tshare", "ypt", "adot", "catch",
                "route_pct", "tprr", "yprr", "offense_pct", "carries", "cshare", "rushing_yards"):
        for hl in HALF_LIVES:
            wk[f"{col}_ew{hl}"] = g[col].transform(lambda s, h=hl: _ew_prev(s, h))
    # team pass volume as of prior weeks (the team's, not the player's)
    tt = team.sort_values(["recent_team", "season", "week"])
    tt["team_tgt_prev"] = tt.groupby("recent_team")["team_tgt"].transform(lambda s: _ew_prev(s, 6))
    tt["pass_rate_prev"] = tt.groupby("recent_team")["pass_rate"].transform(lambda s: _ew_prev(s, 6))
    wk = wk.merge(tt[["recent_team", "season", "week", "team_tgt_prev", "pass_rate_prev"]],
                  on=["recent_team", "season", "week"], how="left")
    # role rank within the team last week: 1 = most targets by EW share
    wk["tshare_rank"] = wk.groupby(["recent_team", "season", "week"])["tshare_ew6"].rank(ascending=False)
    wk["pos_code"] = wk["position"].map({"WR": 0, "TE": 1, "RB": 2})
    return wk


FEATURES = [
    "pos_code", "week", "games_prev", "first_of_season", "is_home", "dome",
    "team_tgt_prev", "pass_rate_prev", "tshare_rank",
    "qb_epa_prev", "qb_games_prev",
    "def_ypt_prev", "def_man_rate_prev", "def_pressure_rate_prev", "def_epa_pass_allowed_prev",
] + [f"{c}_ew{h}" for c in ("targets", "receptions", "receiving_yards", "tshare", "ypt", "adot",
                             "catch", "route_pct", "tprr", "yprr", "offense_pct", "carries",
                             "cshare", "rushing_yards") for h in HALF_LIVES]


def candidates(t: pd.DataFrame) -> pd.Series:
    """The scored set: enough history and prop-worthy expected volume."""
    exp_tgt = t["tshare_ew6"].fillna(0) * t["team_tgt_prev"].fillna(30)
    return (t["games_prev"] >= MIN_PRIOR_GAMES) & (exp_tgt >= 3.0)


def fit(train: pd.DataFrame, target: str = "receiving_yards", seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(loss="squared_error", max_iter=600, learning_rate=0.04,
                                      max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
                                      random_state=seed)
    m.fit(train[FEATURES], train[target])
    return m


def experiment(seasons=tuple(range(2016, 2026)), test_seasons=(2024, 2025),
               target: str = "receiving_yards", harness: pd.DataFrame | None = None) -> pd.DataFrame:
    """Walk-forward: for each test season fit on everything before it."""
    t = build_table(seasons)
    t = t[candidates(t)].copy()
    rows = []
    for S in test_seasons:
        tr, te = t[t["season"] < S], t[t["season"] == S]
        m = fit(tr, target)
        pred = m.predict(te[FEATURES])
        naive = te[f"{target}_ew6"].fillna(te[f"{target}_ew12"]).fillna(0).to_numpy()
        y = te[target].to_numpy(float)
        row = dict(season=S, n=len(te), train_n=len(tr),
                   mae=float(np.abs(pred - y).mean()), rmse=float(np.sqrt(((pred - y) ** 2).mean())),
                   bias=float((pred - y).mean()), corr=float(np.corrcoef(pred, y)[0, 1]),
                   naive_mae=float(np.abs(naive - y).mean()), naive_corr=float(np.corrcoef(naive, y)[0, 1]))
        if harness is not None and not harness.empty:
            h = harness[(harness["season"] == S)][["player_id", "week", "pred_mean"]]
            mm = te[["player_id", "week"]].assign(pred=pred, y=y).merge(h, on=["player_id", "week"], how="inner")
            if len(mm):
                row.update(matched=len(mm), model_mae=float(np.abs(mm["pred_mean"] - mm["y"]).mean()),
                           learned_mae_matched=float(np.abs(mm["pred"] - mm["y"]).mean()))
        rows.append(row)
        # feature importance by permutation on the test set (quick, top 12)
        try:
            from sklearn.inspection import permutation_importance
            pi = permutation_importance(m, te[FEATURES], y, n_repeats=3, random_state=0, scoring="neg_mean_absolute_error")
            top = sorted(zip(FEATURES, pi.importances_mean), key=lambda x: -x[1])[:12]
            print(f"  {S} top features:", ", ".join(f"{f} {v:.2f}" for f, v in top))
        except Exception:
            pass
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(experiment().to_string(index=False, float_format=lambda x: f"{x:.3f}"))
