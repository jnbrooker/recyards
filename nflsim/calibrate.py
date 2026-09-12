"""
nflsim/calibrate.py — the off-season refit, in one command.

Several constants in this codebase are FITTED on out-of-sample residuals
rather than chosen, and they should be refitted as each season is added:

  * the availability coefficients (`availability.QB_MARGIN_COEF`,
    `QB_SWING_COEF`, `DEF_MARGIN_COEF`, and the display-only `OL_MARGIN_COEF`),
  * the team ratings' shrinkage (`teams.RATING_PRIOR_N`), read off the
    calibration slope of actual on predicted margin (target 1.0),
  * the margin spread the harness and the drive engine reproduce
    (`backtest.MARGIN_SD`; `game.LEAD_BETA` is tuned to it),
  * the wind coefficient (`teams.WIND_COEF`).

    python -m nflsim.calibrate 2024 2025            # report only
    python -m nflsim.calibrate 2024 2025 --write    # also update the constants

The report shows the current value beside the refitted one with its t-stat,
so a change is a decision, not a side effect. `--write` edits the module
constants in place (the assignment lines only).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import backtest as B
from . import availability as AV
from . import teams as T


def _ols(X, y):
    X = np.column_stack([np.ones(len(y))] + list(X))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = float((resid ** 2).sum() / max(len(y) - X.shape[1], 1))
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return beta, beta / se


def _slope(x, y) -> float:
    return float(_ols([np.asarray(x, float)], np.asarray(y, float))[0][1])


def refit(score_seasons, n_prior: int = 2, progress=None) -> dict:
    """Run the harness on `score_seasons` (each fitted on the seasons before
    it) and refit every fitted constant. Returns a report dict."""
    bt = B.team_backtest(score_seasons, n_prior, weeks=None, progress=progress,
                         availability=False, weather=False)
    idx = AV.historical_indices(score_seasons, n_prior, progress=progress)
    key = idx.set_index(["game_id", "team"])
    d = bt.copy()
    for c in ("qb_idx", "qb_swing", "def_idx", "ol_idx"):
        d[c + "_h"] = [key.loc[(g, t), c] for g, t in zip(d["game_id"], d["home"])]
        d[c + "_a"] = [key.loc[(g, t), c] for g, t in zip(d["game_id"], d["away"])]
    d["dq"] = d["qb_idx_h"] - d["qb_idx_a"]
    d["ds"] = d["qb_swing_h"] - d["qb_swing_a"]
    d["dd"] = d["def_idx_a"] - d["def_idx_h"]
    d["dl"] = d["ol_idx_h"] - d["ol_idx_a"]
    rm = (d["actual_margin"] - d["pred_margin"]).values

    out = dict(games=int(len(d)), seasons=sorted(int(s) for s in score_seasons))

    # 1. calibration of the ratings (before availability)
    out["slope_base"] = _slope(d["pred_margin"], d["actual_margin"])
    out["rating_prior_n"] = T.RATING_PRIOR_N

    # 2. availability, margin form
    beta, t = _ols([d["dq"], d["ds"], d["dd"], d["dl"]], rm)
    out["availability"] = dict(
        qb=(AV.QB_MARGIN_COEF, float(beta[1]), float(t[1])),
        swing=(AV.QB_SWING_COEF, float(beta[2]), float(t[2])),
        defense=(AV.DEF_MARGIN_COEF, float(beta[3]), float(t[3])),
        oline=(AV.OL_MARGIN_COEF, float(beta[4]), float(t[4])),
    )
    shift = 0.5 * (beta[1] * d["dq"] + beta[2] * d["ds"] + beta[3] * d["dd"])
    d["pred_home_av"] = d["pred_home"] + shift
    d["pred_away_av"] = d["pred_away"] - shift
    d["pred_margin_av"] = d["pred_home_av"] - d["pred_away_av"]
    out["slope_with_availability"] = _slope(d["pred_margin_av"], d["actual_margin"])
    resid_m = d["actual_margin"] - d["pred_margin_av"]
    out["margin_sd"] = (B.MARGIN_SD, float(resid_m.std(ddof=1)))
    out["total_sd"] = float((d["actual_total"] - d["pred_total"]).std(ddof=1))
    ra = d["actual_home"] - d["pred_home_av"]
    rb = d["actual_away"] - d["pred_away_av"]
    out["team_sd"] = float(0.5 * (ra.std(ddof=1) + rb.std(ddof=1)))
    out["team_corr"] = float(np.corrcoef(ra, rb)[0, 1])
    out["rmse_base"] = float(np.sqrt(((d["actual_margin"] - d["pred_margin"]) ** 2).mean()))
    out["rmse_with_availability"] = float(np.sqrt((resid_m ** 2).mean()))
    out["rmse_line"] = float(np.sqrt(((d["actual_margin"] - d["line_margin"]) ** 2).mean()))

    # 3. wind
    sched = D.load_schedule(tuple(int(s) for s in score_seasons))
    w = d.merge(sched[["game_id", "roof", "wind"]], on="game_id", how="left")
    indoor = w["roof"].astype(str).str.lower().isin(T.INDOOR_ROOFS)
    wind = np.where(indoor, 0.0, np.clip(w["wind"].fillna(0.0) - T.WIND_FREE_MPH, 0, None))
    rt = (w["actual_total"] - w["pred_total"]).values
    bw, tw = _ols([wind], rt)
    out["wind"] = (T.WIND_COEF, float(bw[1]), float(tw[1]), int((wind > 0).sum()))
    return out


def report(r: dict) -> str:
    L = [f"Refit on {r['seasons']} — {r['games']} games out of sample", ""]
    L.append(f"Team ratings: calibration slope {r['slope_base']:.3f} before availability, "
             f"{r['slope_with_availability']:.3f} after (target 1.0; RATING_PRIOR_N = {r['rating_prior_n']:g}).")
    L.append("  slope > 1 means the ratings are under-dispersed — lower RATING_PRIOR_N; < 1, raise it.")
    L.append(f"Margin RMSE: {r['rmse_base']:.3f} base → {r['rmse_with_availability']:.3f} with availability "
             f"(closing line {r['rmse_line']:.3f}).")
    L.append("")
    L.append("Availability (points of margin per unit of index difference):")
    for name, (cur, new, t) in r["availability"].items():
        flag = "" if abs(t) >= 2 else "   (not clearly different from zero)"
        L.append(f"  {name:8s} current {cur:+7.2f}   refit {new:+7.2f}  (t {t:+.1f}){flag}")
    L.append("")
    cur, new = r["margin_sd"]
    L.append(f"Spread targets: margin sd {new:.2f} (MARGIN_SD is {cur:g}), total sd {r['total_sd']:.2f}, "
             f"team sd {r['team_sd']:.2f}, corr {r['team_corr']:+.3f}")
    L.append("  game.LEAD_BETA should reproduce margin / team sd and corr — check with a few simulated matchups.")
    cur, new, t, n = r["wind"]
    L.append(f"Wind: current {cur:+.2f}/mph over 10, refit {new:+.2f} (t {t:+.1f}, {n} windy games)")
    return "\n".join(L)


def _write_constant(path: Path, name: str, value: float) -> bool:
    src = path.read_text()
    pat = re.compile(rf"^({name}\s*=\s*)([-+]?\d+(?:\.\d+)?)", re.M)
    if not pat.search(src):
        return False
    path.write_text(pat.sub(lambda m: f"{m.group(1)}{value:.2f}", src, count=1))
    return True


def write(r: dict) -> list[str]:
    """Update the fitted constants in place; returns what changed."""
    root = Path(__file__).parent
    changes = []
    av = root / "availability.py"
    for name, const in (("qb", "QB_MARGIN_COEF"), ("swing", "QB_SWING_COEF"),
                        ("defense", "DEF_MARGIN_COEF")):
        cur, new, t = r["availability"][name]
        if abs(t) >= 2 and _write_constant(av, const, new):
            changes.append(f"{const}: {cur:+.2f} → {new:+.2f}")
    cur, new = r["margin_sd"]
    if _write_constant(root / "backtest.py", "MARGIN_SD", round(new, 1)):
        changes.append(f"MARGIN_SD: {cur:g} → {round(new, 1):g}")
    return changes


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    seasons = [int(a) for a in args] or [2024, 2025]
    r = refit(seasons, progress=lambda f, t: print(f"  {f:5.0%} {t}", end="\r"))
    print()
    print(report(r))
    if "--write" in sys.argv:
        ch = write(r)
        print("\nWritten:" if ch else "\nNothing written (no coefficient met |t| >= 2).")
        for c in ch:
            print("  " + c)
        print("Review the diff, re-run the harness (page 10 or `python -m nflsim.backtest`), then commit.")
