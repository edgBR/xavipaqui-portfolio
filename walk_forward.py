"""walk_forward.py — out-of-sample test of the claim HRP actually makes.

López de Prado's evidence for HRP is not about returns: it is that HRP delivers lower
OUT-OF-SAMPLE variance than a quadratic optimiser (CLA/min-variance) and than the
inverse-variance portfolio, despite min-variance being CLA's own objective. Nothing
else in this project tests that -- `hrp_nco.py` reports in-sample figures and
`monte_carlo.py` is a forward projection.

Design: estimate Sigma on a rolling WINDOW of months, compute each method's weights,
hold them for HOLD months (rebalancing monthly to those weights, so the test isolates
the weight choice from drift), then roll forward. Every method sees exactly the same
dates, so comparisons are paired.

    python walk_forward.py            # -> walk_forward.json, walk_forward.csv

Reported per method: realised out-of-sample volatility (the headline), CAGR, Sharpe,
max drawdown, annual turnover, and the ratio of predicted to realised volatility --
the risk-forecast check, which is answerable with this much data whereas a return
ranking is not.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

import hrp_nco as H

WINDOW = 60      # months of history used to estimate Sigma
HOLD = 12        # months the weights are held before re-estimating
RF = 0.0         # excess-return convention: raw returns


def weight_fns(cols):
    """label -> callable(cov_df) -> weight Series. Fixed-weight portfolios ignore cov."""
    def hand(label):
        w = pd.Series(H.HAND[label]).reindex(cols).fillna(0.0)
        return lambda cov: w
    return {
        "HRP":            lambda cov: H.hrp(cov).reindex(cols).fillna(0.0),
        "NCO min-var":    lambda cov: H.nco(cov).reindex(cols).fillna(0.0),
        "IVP":            lambda cov: pd.Series(H.ivp(cov.values), index=cols),
        "MinVar (CLA)":   lambda cov: pd.Series(H.min_var(cov.values), index=cols),
        "Equal weight":   lambda cov: pd.Series(1.0 / len(cols), index=cols),
        "Hand 60/40":     hand("58yo 60/40 (hand)"),
        "Hand 25/75":     hand("64yo 25/75 (hand)"),
    }


def run(ret: pd.DataFrame, denoise: bool) -> dict:
    cols = list(ret.columns)
    fns = weight_fns(cols)
    oos = {k: [] for k in fns}          # realised monthly OOS returns
    pred_vol = {k: [] for k in fns}     # predicted annual vol at each refit
    turn = {k: [] for k in fns}         # turnover per rebalance
    prev = {k: None for k in fns}
    dates = []

    for start in range(WINDOW, len(ret), HOLD):
        train = ret.iloc[start - WINDOW:start]
        test = ret.iloc[start:start + HOLD]
        if len(test) == 0:
            break
        cov = train.cov() * 12
        if denoise:
            cov = H.denoise_cov(cov, len(train))
        for k, fn in fns.items():
            w = fn(cov)
            wv = w.reindex(cols).fillna(0.0).values
            pred_vol[k].append(float(np.sqrt(wv @ cov.values @ wv)))
            if prev[k] is not None:
                turn[k].append(float(np.abs(wv - prev[k]).sum() / 2.0))
            prev[k] = wv
            oos[k].append(pd.Series((test.values * wv).sum(axis=1), index=test.index))
        dates.append(f"{test.index[0]:%Y-%m}")

    out = {}
    for k in fns:
        r = pd.concat(oos[k])
        nav = (1 + r).cumprod()
        yrs = len(r) / 12
        vol = float(r.std() * np.sqrt(12))
        cagr = float(nav.iloc[-1] ** (1 / yrs) - 1)
        dd = float((nav / nav.cummax() - 1).min())
        pv = float(np.mean(pred_vol[k]))
        out[k] = {"oos_vol_pct": round(vol * 100, 2), "cagr_pct": round(cagr * 100, 2),
                  "sharpe": round((cagr - RF) / vol, 3) if vol else None,
                  "maxdd_pct": round(dd * 100, 2),
                  "pred_vol_pct": round(pv * 100, 2),
                  "pred_over_realised": round(pv / vol, 3) if vol else None,
                  "turnover_pct_per_rebal": round(float(np.mean(turn[k])) * 100, 1) if turn[k] else 0.0,
                  "_ret": r}
    return {"methods": out, "n_oos_months": len(pd.concat(oos["HRP"])),
            "refits": len(dates), "first_oos": dates[0], "last_oos": dates[-1]}


def paired_variance_test(a: pd.Series, b: pd.Series) -> dict:
    """H0: the two return series have equal variance. Paired on dates, so the shared
    market component cancels -- far more powerful than comparing two vols independently."""
    d = (a.values ** 2) - (b.values ** 2)
    t, p = stats.ttest_1samp(d, 0.0)
    return {"t": round(float(t), 2), "p": round(float(p), 4),
            "vol_ratio": round(float(a.std() / b.std()), 3)}


def main():
    navs = pd.read_csv("navs_monthly.csv", index_col=0, parse_dates=True)
    cols = [c for c in H.EQUITY + H.BONDS if c in navs]
    ret = navs[cols].dropna().pct_change().dropna()

    payload = {"config": {"window_months": WINDOW, "hold_months": HOLD,
                          "assets": cols, "sample_months": len(ret)},
               "spectrum_full_sample": H.spectrum_report(ret.cov() * 12, len(ret)),
               "spectrum_window": H.spectrum_report(ret.iloc[:WINDOW].cov() * 12, WINDOW),
               "runs": {}}

    for tag, dn in (("raw", False), ("denoised", True)):
        res = run(ret, dn)
        rets = {k: v.pop("_ret") for k, v in res["methods"].items()}
        # LdP's actual comparison: HRP against min-variance and against IVP
        res["tests_vs_HRP"] = {k: paired_variance_test(rets["HRP"], rets[k])
                               for k in ("MinVar (CLA)", "IVP", "Equal weight", "NCO min-var")}
        payload["runs"][tag] = res
        print(f"\n=== {tag.upper()}  ({res['n_oos_months']} OOS months, "
              f"{res['refits']} refits, {res['first_oos']} -> {res['last_oos']})")
        tbl = pd.DataFrame(res["methods"]).T[
            ["oos_vol_pct", "pred_vol_pct", "pred_over_realised", "cagr_pct",
             "sharpe", "maxdd_pct", "turnover_pct_per_rebal"]]
        print(tbl.to_string())
        print("  paired variance tests, HRP vs:")
        for k, v in res["tests_vs_HRP"].items():
            verdict = "HRP lower" if v["vol_ratio"] < 1 else "HRP higher"
            sig = "significant" if v["p"] < 0.05 else "not significant"
            print(f"    {k:14s} vol ratio {v['vol_ratio']:.3f} ({verdict}), "
                  f"t={v['t']:+.2f} p={v['p']:.4f} -> {sig}")

    rows = [{"run": t, "method": k, **v} for t, r in payload["runs"].items()
            for k, v in r["methods"].items()]
    pd.DataFrame(rows).to_csv("walk_forward.csv", index=False)
    json.dump(payload, open("walk_forward.json", "w"), indent=1)
    print("\nSaved walk_forward.json, walk_forward.csv")


if __name__ == "__main__":
    main()
