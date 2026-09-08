"""build_estim.py — estimator precision of mu vs sigma, for the report's explainer."""
import json
import numpy as np
import pandas as pd
import hrp_nco as H


def build(path="navs_monthly.csv"):
    navs = pd.read_csv(path, index_col=0, parse_dates=True)
    cols = [c for c in H.EQUITY + H.BONDS if c in navs]
    ret = navs[cols].dropna().pct_change().dropna()
    n = len(ret); yrs = n / 12
    mu, sd = ret.mean() * 12, ret.std() * np.sqrt(12)
    se_mu, se_sd = sd / np.sqrt(yrs), sd / np.sqrt(2 * (n - 1))
    rows = [dict(asset=c, mu=round(mu[c] * 100, 2), se_mu=round(se_mu[c] * 100, 2),
                 rel_mu=round(se_mu[c] / abs(mu[c]) * 100), sd=round(sd[c] * 100, 2),
                 se_sd=round(se_sd[c] * 100, 2), rel_sd=round(se_sd[c] / sd[c] * 100),
                 ci_lo=round((mu[c] - 1.96 * se_mu[c]) * 100, 2),
                 ci_hi=round((mu[c] + 1.96 * se_mu[c]) * 100, 2)) for c in cols]
    return {"n_months": n, "years": round(yrs, 1), "rows": rows,
            "years_for_1pp_world": int((sd["World"] / 0.01) ** 2)}


if __name__ == "__main__":
    json.dump(build(), open("estim.json", "w"), indent=1)
    print("wrote estim.json")
