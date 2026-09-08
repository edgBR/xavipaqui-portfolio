"""
monte_carlo_dca.py — same three engines as monte_carlo.py, but projecting an actual
savings plan rather than a lump sum:

    initial investment   EUR 20,000
    monthly contribution EUR    600   (start of each month, invested immediately)

With cash flows going in, CAGR is no longer the right return measure — it ignores
WHEN the money arrived. This reports money-weighted return (IRR) instead, plus the
figures a saver actually cares about: euros in, euros out, and how far the balance
can fall along the way.

    python monte_carlo_dca.py       # -> mc_dca_results.json, mc_dca_summary.csv

NOTE ON THE CONTRIBUTION PERIOD: CONTRIB_YEARS defaults to the full horizon, i.e.
paying in until age 90. That is what was asked for, but it is unusual for people
already at/near retirement — set CONTRIB_YEARS to stop earlier to model
"contribute until 67, then leave it alone".
"""
import json
import sys

import numpy as np
import pandas as pd

import hrp_nco as H
import monte_carlo as MC

W0 = 20_000.0          # initial investment, EUR
CONTRIB = 600.0        # monthly contribution, EUR
# contributions stop at MC.CONTRIB_UNTIL_AGE (85); withdrawals from MC.WITHDRAW
N_SIM = MC.N_SIM
SEED = 20260908
PCTS = MC.PCTS


def project_dca(monthly: np.ndarray, contrib_months: int) -> dict:
    """Savings plan with periodic withdrawals. Percentiles are taken across paths."""
    infl = MC.INFLATION
    wp = MC.wealth_paths(monthly, W0, CONTRIB, contrib_months)
    horizon = wp["horizon"]
    years = horizon / 12.0
    W = wp["terminal"]
    wd, wd_m = wp["wd_amounts"], wp["wd_months"]
    withdrawn = wd.sum(axis=0) if len(wd) else np.zeros_like(W)
    received = W + withdrawn                      # everything the investor ends up holding
    real_terminal = W / (1.0 + infl) ** years
    # each withdrawal deflated to today at the moment it is actually taken
    real_received = real_terminal + sum(
        wd[j] / (1.0 + infl) ** (wd_m[j] / 12.0) for j in range(len(wd)))
    irr = MC.irr_annual(W0, CONTRIB, contrib_months, wp)
    paid_nom, paid_real = wp["paid_nom"], wp["paid_real"]
    fan_arr = wp["yr_vals"]

    pc = lambda a, p: round(float(np.percentile(a, p)), 2)
    return {
        "paid_in_eur": round(paid_nom, 2),
        "paid_in_real_eur": round(paid_real, 2),
        "contrib_years": contrib_months // 12,
        "terminal_eur": {str(p): pc(W, p) for p in PCTS},
        "real_terminal_eur": {str(p): pc(real_terminal, p) for p in PCTS},
        "withdrawn_eur": {str(p): pc(withdrawn, p) for p in PCTS},
        "received_eur": {str(p): pc(received, p) for p in PCTS},
        "real_received_eur": {str(p): pc(real_received, p) for p in PCTS},
        "withdrawals": [{"year": wd_m[j] // 12,
                         "eur": {str(p): pc(wd[j], p) for p in PCTS}}
                        for j in range(len(wd))],
        "multiple_of_paid_in": {str(p): round(float(np.percentile(received, p)) / paid_nom, 3)
                                for p in PCTS},
        "irr_pct": {str(p): round(float(np.percentile(irr, p)) * 100, 2) for p in PCTS},
        "p_below_paid_in": round(float((received < paid_nom).mean()) * 100, 2),
        "p_real_below_paid_in": round(float((real_received < paid_real).mean()) * 100, 2),
        "median_maxdd_pct": round(float(np.median(wp["maxdd"])) * 100, 2),
        "worst_maxdd_pct": round(float(np.percentile(wp["maxdd"], 1)) * 100, 2),
        "fan": {"years": wp["yr_years"],
                **{str(p): np.percentile(fan_arr, p, axis=1).round(2).tolist() for p in PCTS}},
    }


def main(path="navs_monthly.csv"):
    navs = pd.read_csv(path, index_col=0, parse_dates=True)
    cols = [c for c in H.EQUITY + H.BONDS if c in navs]
    ret = navs[cols].dropna().pct_change().dropna()
    R = ret.values
    cov = ret.cov() * 12
    mu_hist = ret.mean() * 12
    mu_fwd = pd.Series([MC.FWD["equity"] if c in H.EQUITY else MC.FWD["bond"] for c in cols],
                       index=cols)

    print(f"Sample {ret.index[0]:%Y-%m} -> {ret.index[-1]:%Y-%m} ({len(ret)} months)")
    print(f"Plan: EUR {W0:,.0f} initial + EUR {CONTRIB:,.0f}/month until age "
          f"{MC.CONTRIB_UNTIL_AGE}, then held to 90")
    print("Withdrawals: " + ", ".join(f"year {y} ({f:.0%})" for y, f in MC.WITHDRAW) + "\n")

    wsets = MC.weight_sets(cov, cols)
    rng = np.random.default_rng(SEED)
    results, rows = {}, []

    for label, methods in wsets.items():
        horizon = MC.HORIZON_YEARS[label] * 12
        cmonths = max(0, min((MC.CONTRIB_UNTIL_AGE - MC.AGES[label]) * 12, horizon))
        boot_idx = MC.stationary_bootstrap_idx(len(R), horizon, N_SIM, rng)
        z = rng.standard_normal((N_SIM, horizon))
        results[label] = {"horizon_years": MC.HORIZON_YEARS[label],
                          "contrib_years": cmonths // 12, "methods": {}}

        for mname, w in methods.items():
            wv = w.values
            pr = R @ wv
            sd = float(np.sqrt(wv @ cov.values @ wv))
            mu_h = float(mu_hist @ wv)
            fee = MC.weighted_ter(w) + MC.PLATFORM_FEE
            mu_f = float(mu_fwd @ wv) - fee   # gross assumption -> net of charges
            engines = {
                "bootstrap": project_dca(pr[boot_idx], cmonths),
                "hist-mvn": project_dca(MC.gauss_paths(mu_h, sd, z), cmonths),
                "fwd-mvn": project_dca(MC.gauss_paths(mu_f, sd, z), cmonths),
            }
            results[label]["methods"][mname] = {
                "weights_pct": (w * 100).round(2).to_dict(),
                "equity_pct": round(float(w[H.EQUITY].sum()) * 100, 1),
                "ann_vol_pct": round(sd * 100, 2),
                "mu_fwd_pct": round(mu_f * 100, 2), "mu_hist_pct": round(mu_h * 100, 2),
                "fee_pct": round(fee * 100, 3),
                "engines": engines,
            }
            f = engines["fwd-mvn"]
            rows.append([label, mname, round(float(w[H.EQUITY].sum()) * 100, 1),
                         f["paid_in_eur"], f["withdrawn_eur"]["50"],
                         f["terminal_eur"]["50"], f["received_eur"]["50"],
                         f["real_received_eur"]["50"], f["irr_pct"]["50"],
                         engines["bootstrap"]["received_eur"]["50"],
                         f["p_real_below_paid_in"], f["median_maxdd_pct"]])

    tbl = pd.DataFrame(rows, columns=["Portfolio", "Method", "Eq%", "PaidIn", "TakenOut",
                                      "EndValue", "Received", "RealRecvd", "IRR%",
                                      "bootRecvd", "P(real<paid)%", "medDD%"])
    pd.set_option("display.width", 200)
    print(tbl.to_string(index=False))

    payload = {"meta": {"w0_eur": W0, "contrib_eur": CONTRIB, "n_sim": N_SIM,
                        "inflation_pct": MC.INFLATION * 100,
                        "sample_start": f"{ret.index[0]:%Y-%m}",
                        "sample_end": f"{ret.index[-1]:%Y-%m}", "sample_months": len(ret),
                        "fwd_equity_pct": MC.FWD["equity"] * 100,
                        "fwd_bond_pct": MC.FWD["bond"] * 100,
                        "assets": cols, "equity": H.EQUITY, "bonds": H.BONDS},
               "portfolios": results}
    json.dump(payload, open("mc_dca_results.json", "w"), indent=1)
    tbl.to_csv("mc_dca_summary.csv", index=False)
    print("\nSaved mc_dca_results.json, mc_dca_summary.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "navs_monthly.csv")
