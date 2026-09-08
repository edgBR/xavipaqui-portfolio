"""build_robustness.py — payload for the report's denoising + walk-forward sections."""
import json

import numpy as np
import pandas as pd

import hrp_nco as H


def main():
    navs = pd.read_csv("navs_monthly.csv", index_col=0, parse_dates=True)
    cols = [c for c in H.EQUITY + H.BONDS if c in navs]
    ret = navs[cols].dropna().pct_change().dropna()
    cov = ret.cov() * 12
    dn = H.denoise_cov(cov, len(ret))

    methods = {
        "HRP unconstrained": lambda c: H.hrp(c),
        "NCO min-var unconstrained": lambda c: H.nco(c),
    }
    for label, eq in H.EQ_SHARE.items():
        methods[f"HRP-in-sleeves ({label})"] = (
            lambda c, e=eq: H.sleeve_hrp(c, e))
        methods[f"NCO-in-sleeves ({label})"] = (
            lambda c, e=eq: pd.concat([H.nco(c.loc[H.EQUITY, H.EQUITY]) * e,
                                       H.nco(c.loc[H.BONDS, H.BONDS]) * (1 - e)]))

    weights = {}
    for name, fn in methods.items():
        a = fn(cov).reindex(cols).fillna(0.0) * 100
        b = fn(dn).reindex(cols).fillna(0.0) * 100
        weights[name] = {
            "raw": {k: round(float(a[k]), 2) for k in cols},
            "denoised": {k: round(float(b[k]), 2) for k in cols},
            "delta": {k: round(float(b[k] - a[k]), 2) for k in cols},
            "max_abs_delta": round(float((b - a).abs().max()), 2),
        }

    wf = json.load(open("walk_forward.json"))
    payload = {
        "spectrum": H.spectrum_report(cov, len(ret)),
        "spectrum_window": wf["spectrum_window"],
        "weights": weights,
        "order": cols,
        "walk_forward": {"config": wf["config"],
                         "runs": {t: {"methods": r["methods"], "tests_vs_HRP": r["tests_vs_HRP"],
                                      "n_oos_months": r["n_oos_months"], "refits": r["refits"],
                                      "first_oos": r["first_oos"], "last_oos": r["last_oos"]}
                                  for t, r in wf["runs"].items()}},
    }
    json.dump(payload, open("robustness.json", "w"), indent=1)
    print("wrote robustness.json")
    for n, v in weights.items():
        print(f"  {n:34s} max |delta| from denoising: {v['max_abs_delta']:5.2f}pp")


if __name__ == "__main__":
    main()
