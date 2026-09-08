"""
monte_carlo.py — Monte Carlo projection of the two portfolios under each weighting
method, on the NAV history from fetch_navs.py and the weights from hrp_nco.py.

    python monte_carlo.py            # -> mc_results.json + console tables

Three engines, because Monte Carlo cannot avoid a return assumption the way HRP can:

  bootstrap  stationary block bootstrap of the ACTUAL historical monthly rows
             (no return assumption; keeps fat tails, cross-correlation, some
             autocorrelation) — but inherits whatever the sample window happened to pay
  hist-mvn   Gaussian, historical covariance + historical sample means
  fwd-mvn    Gaussian, historical covariance + forward means (equity 6%, bonds 3.3%)

Weight sets are drawn on COMMON RANDOM NUMBERS, so differences between methods are
the methods, not simulation noise.
"""
import json
import sys

import numpy as np
import pandas as pd

import build_funds_meta as BM
import hrp_nco as H

N_SIM = 20_000
SEED = 20260908
INFLATION = 0.02          # for the real-terms figures
FWD = {"equity": 0.06, "bond": 0.033}   # the forward assumption from the conversation
PLATFORM_FEE = 0.0   # annual custody/platform charge. MyInvestor charges none on funds;
                     # set e.g. 0.0025 to test a 0.25 %/yr platform that does.
HORIZON_YEARS = {"58yo 60/40": 32, "64yo 25/75": 26}   # both to age 90
AGES = {"58yo 60/40": 58, "64yo 25/75": 64}
CONTRIB_UNTIL_AGE = 85        # contributions stop here; the projection still runs to 90
WITHDRAW = [(10, 0.20), (20, 0.20)]   # (year, fraction of value) -- taken at that year-end
PCTS = [5, 25, 50, 75, 95]


def weighted_ter(w: pd.Series) -> float:
    """Annual ongoing charge of a weight vector, from the verified per-fund TERs.

    IMPORTANT, and the reason this exists: published fund NAVs are already NET of the
    fund's ongoing charge, so `mu_hist`/`Sigma` -- and therefore the `bootstrap` and
    `hist-mvn` engines -- have fees baked in. The `fwd-mvn` engine does NOT: 6 %/3.3 %
    are gross index-level assumptions, so the charge has to be subtracted explicitly or
    the headline engine silently overstates returns.
    """
    return float(sum(w[k] * BM.FUNDS[k]["ter"] / 100.0 for k in w.index if k in BM.FUNDS))


def weight_sets(cov: pd.DataFrame, cols: list[str]) -> dict:
    """Per portfolio: the hand-set weights plus each data-driven method at the same
    equity share, then the two unconstrained methods as a shared reference."""
    out = {}
    for label, eq_share in H.EQ_SHARE.items():
        hand = pd.Series(H.HAND[f"{label} (hand)"]).reindex(cols).fillna(0)
        eq, bd = H.EQUITY, H.BONDS
        nco_sleeves = pd.concat([
            H.nco(cov.loc[eq, eq]) * eq_share,
            H.nco(cov.loc[bd, bd]) * (1 - eq_share),
        ]).reindex(cols).fillna(0)
        out[label] = {
            "Hand-set": hand,
            "HRP-in-sleeves": H.sleeve_hrp(cov, eq_share).reindex(cols).fillna(0),
            "NCO-in-sleeves": nco_sleeves,
            "HRP unconstrained": H.hrp(cov).reindex(cols).fillna(0),
            "NCO min-var unconstrained": H.nco(cov).reindex(cols).fillna(0),
        }
    return out


def stationary_bootstrap_idx(n_obs: int, horizon: int, n_sim: int, rng,
                             mean_block: int = 12) -> np.ndarray:
    """Politis-Romano stationary bootstrap: geometric block lengths, wrap-around.
    Resamples whole rows, so cross-asset correlation survives intact."""
    p = 1.0 / mean_block
    idx = np.empty((n_sim, horizon), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n_obs, n_sim)
    for t in range(1, horizon):
        new_block = rng.random(n_sim) < p
        idx[:, t] = np.where(new_block,
                             rng.integers(0, n_obs, n_sim),
                             (idx[:, t - 1] + 1) % n_obs)
    return idx


def gauss_paths(mu_a: float, sd: float, z: np.ndarray) -> np.ndarray:
    """Lognormal monthly simple returns whose ANNUAL arithmetic mean is mu_a and
    annual volatility is sd. Matching both moments of G = exp(X), X ~ N(m, s2):
        s2 = log(1 + sd^2/(1+mu_a)^2)        m = log(1+mu_a) - s2/2
    Both the drift AND the shock must use s2 -- using sd/sqrt(12) as the monthly
    log sd overshoots realised volatility by ~5%."""
    s2 = np.log1p(sd ** 2 / (1.0 + mu_a) ** 2)
    drift = (np.log1p(mu_a) - 0.5 * s2) / 12.0
    return np.expm1(drift + np.sqrt(s2 / 12.0) * z)


def wealth_paths(monthly: np.ndarray, w0: float, contrib: float, contrib_months: int,
                 withdrawals=WITHDRAW) -> dict:
    """Walk the balance month by month. One engine for both the lump-sum case
    (w0=1, contrib=0) and the savings-plan case (w0=20000, contrib=600).

    Contributions are paid at the START of a month, so they earn that month's return.
    Withdrawals are taken at the END of the named year, so a year-end value is
    reported AFTER any withdrawal that year -- what the account would actually show.

    Drawdown deliberately does NOT count a withdrawal as a loss: the running peak is
    scaled by the same factor, so `maxdd` measures market declines only.
    """
    n_sim, horizon = monthly.shape
    wd_at = {y * 12 - 1: f for y, f in withdrawals if y * 12 - 1 < horizon}
    W = np.full(n_sim, float(w0))
    peak = W.copy()
    maxdd = np.zeros(n_sim)
    paid_nom = paid_real = float(w0)
    yr_years, yr_vals, wd_amounts, wd_months = [], [], [], []
    for t in range(horizon):
        if contrib and t < contrib_months:
            W = W + contrib
            paid_nom += contrib
            paid_real += contrib / (1.0 + INFLATION) ** (t / 12.0)
        W = W * (1.0 + monthly[:, t])
        peak = np.maximum(peak, W)
        maxdd = np.minimum(maxdd, W / peak - 1.0)
        if t in wd_at:
            f = wd_at[t]
            wd_amounts.append(W * f)
            wd_months.append(t + 1)
            W = W * (1.0 - f)
            peak = peak * (1.0 - f)      # a planned distribution is not a drawdown
        if (t + 1) % 12 == 0:
            yr_years.append((t + 1) // 12)
            yr_vals.append(W.copy())
    return {"terminal": W, "maxdd": maxdd, "paid_nom": paid_nom, "paid_real": paid_real,
            "yr_years": yr_years, "yr_vals": np.array(yr_vals),
            "wd_amounts": np.array(wd_amounts) if wd_amounts else np.zeros((0, n_sim)),
            "wd_months": wd_months, "horizon": horizon}


def irr_annual(w0: float, contrib: float, contrib_months: int, wp: dict,
               iters: int = 90) -> np.ndarray:
    """Money-weighted return. Outflows: w0 + contributions. Inflows: each withdrawal
    when it is taken, plus the terminal balance. NPV is monotone decreasing in the
    rate, so bisect elementwise across paths."""
    terminal, horizon = wp["terminal"], wp["horizon"]

    def npv(i):
        v = 1.0 / (1.0 + i)
        out = -w0
        if contrib and contrib_months:
            n = contrib_months
            # np.where evaluates BOTH branches, so guard the denominator or the
            # near-zero rate raises "invalid value encountered in divide" and the
            # discarded branch fills with NaN
            near0 = np.abs(i) < 1e-12
            i_safe = np.where(near0, 1.0, i)
            a = np.where(near0, float(n), (1.0 - v ** n) * (1.0 + i) / i_safe)
            out = out - contrib * a
        for amt, m in zip(wp["wd_amounts"], wp["wd_months"]):
            out = out + amt * v ** m
        return out + terminal * v ** horizon

    lo = np.full(terminal.shape, -0.05)
    hi = np.full(terminal.shape, 0.05)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pos = npv(mid) > 0.0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    return (1.0 + 0.5 * (lo + hi)) ** 12 - 1.0


def project(monthly_paths: np.ndarray, contrib_months: int = 0) -> dict:
    """Lump-sum view: w0 = 1, no contributions, withdrawals applied. All values are
    multiples of the starting capital."""
    horizon = monthly_paths.shape[1]
    wp = wealth_paths(monthly_paths, 1.0, 0.0, 0)
    years = horizon / 12.0
    terminal = wp["terminal"]
    withdrawn = wp["wd_amounts"].sum(axis=0) if len(wp["wd_amounts"]) else np.zeros_like(terminal)
    received = terminal + withdrawn
    cagr = irr_annual(1.0, 0.0, 0, wp)     # money-weighted: withdrawals are cash out
    real = terminal / (1.0 + INFLATION) ** years
    # deflate each withdrawal at the moment it is taken, not at the horizon --
    # a year-10 payout is worth far more in today's money than a year-32 one
    real_recv = real + sum(
        wp["wd_amounts"][j] / (1.0 + INFLATION) ** (wp["wd_months"][j] / 12.0)
        for j in range(len(wp["wd_amounts"])))
    dd = wp["maxdd"]
    nav = wp["yr_vals"]                    # (n_years, n_sim), post-withdrawal
    fan = {str(p): np.percentile(nav, p, axis=1).round(4).tolist() for p in PCTS}
    fan["years"] = wp["yr_years"]
    return {
        "withdrawn_x": {str(p): round(float(np.percentile(withdrawn, p)), 3) for p in PCTS},
        "received_x": {str(p): round(float(np.percentile(received, p)), 3) for p in PCTS},
        "real_received_x": {str(p): round(float(np.percentile(real_recv, p)), 3) for p in PCTS},
        "cagr_pct": {str(p): round(float(np.percentile(cagr, p)) * 100, 2) for p in PCTS},
        "cagr_mean_pct": round(float(cagr.mean()) * 100, 2),
        "terminal_x": {str(p): round(float(np.percentile(terminal, p)), 3) for p in PCTS},
        "real_terminal_x": {str(p): round(float(np.percentile(real, p)), 3) for p in PCTS},
        "p_nominal_loss": round(float((received < 1).mean()) * 100, 2),
        "p_real_loss": round(float((real_recv < 1).mean()) * 100, 2),
        "median_maxdd_pct": round(float(np.median(dd)) * 100, 2),
        "worst_maxdd_pct": round(float(np.percentile(dd, 1)) * 100, 2),
        "fan": fan,
    }


def main(path="navs_monthly.csv"):
    navs = pd.read_csv(path, index_col=0, parse_dates=True)
    cols = [c for c in H.EQUITY + H.BONDS if c in navs]
    ret = navs[cols].dropna().pct_change().dropna()
    R = ret.values
    cov = ret.cov() * 12
    mu_hist = ret.mean() * 12
    mu_fwd = pd.Series(
        [FWD["equity"] if c in H.EQUITY else FWD["bond"] for c in cols], index=cols)

    wd = ", ".join(f"yr {y}: {f:.0%}" for y, f in WITHDRAW)
    print(f"Sample: {ret.index[0]:%Y-%m} -> {ret.index[-1]:%Y-%m} ({len(ret)} months), "
          f"assets: {cols}")
    print(f"Withdrawals ({wd}) applied to every path, including the lump-sum cases.")
    print(f"Historical ann. means (%): {(mu_hist * 100).round(2).to_dict()}")
    print(f"Forward ann. means  (%): {(mu_fwd * 100).round(2).to_dict()}\n")

    wsets = weight_sets(cov, cols)
    rng = np.random.default_rng(SEED)
    results, rows = {}, []

    for label, methods in wsets.items():
        horizon = HORIZON_YEARS[label] * 12
        # common random numbers, shared by every method within this portfolio
        boot_idx = stationary_bootstrap_idx(len(R), horizon, N_SIM, rng)
        z = rng.standard_normal((N_SIM, horizon))
        results[label] = {"horizon_years": HORIZON_YEARS[label], "methods": {}}

        for mname, w in methods.items():
            wv = w.values
            pr = R @ wv                       # historical portfolio return series
            mu_h = float(mu_hist @ wv)
            fee = weighted_ter(w) + PLATFORM_FEE
            mu_f = float(mu_fwd @ wv) - fee   # gross assumption -> net of charges
            sd = float(np.sqrt(wv @ cov.values @ wv))
            engines = {
                "bootstrap": project(pr[boot_idx]),
                "hist-mvn": project(gauss_paths(mu_h, sd, z)),
                "fwd-mvn": project(gauss_paths(mu_f, sd, z)),
            }
            results[label]["methods"][mname] = {
                "weights_pct": (w * 100).round(2).to_dict(),
                "equity_pct": round(float(w[H.EQUITY].sum()) * 100, 1),
                "ann_vol_pct": round(sd * 100, 2),
                "mu_hist_pct": round(mu_h * 100, 2),
                "mu_fwd_pct": round(mu_f * 100, 2), "fee_pct": round(fee * 100, 3),
                "engines": engines,
            }
            rows.append([label, mname, round(w[H.EQUITY].sum() * 100, 1), round(sd * 100, 1),
                         engines["fwd-mvn"]["cagr_pct"]["50"],
                         engines["fwd-mvn"]["cagr_pct"]["5"],
                         engines["bootstrap"]["cagr_pct"]["50"],
                         engines["bootstrap"]["cagr_pct"]["5"],
                         engines["fwd-mvn"]["p_real_loss"]])

    tbl = pd.DataFrame(rows, columns=["Portfolio", "Method", "Eq%", "Vol%",
                                      "fwd med", "fwd p5", "boot med", "boot p5",
                                      "P(real loss)%"])
    print(tbl.to_string(index=False))

    payload = {
        "meta": {
            "sample_start": f"{ret.index[0]:%Y-%m}", "sample_end": f"{ret.index[-1]:%Y-%m}",
            "sample_months": len(ret), "n_sim": N_SIM, "inflation_pct": INFLATION * 100,
            "assets": cols, "equity": H.EQUITY, "bonds": H.BONDS,
            "withdrawals": [y for y, _ in WITHDRAW], "contrib_until_age": CONTRIB_UNTIL_AGE,
            "fwd_equity_pct": FWD["equity"] * 100, "fwd_bond_pct": FWD["bond"] * 100,
            "ann_vol_pct": {c: round(float(np.sqrt(cov.loc[c, c])) * 100, 2) for c in cols},
            "mu_hist_pct": {c: round(float(mu_hist[c]) * 100, 2) for c in cols},
            "corr": ret.corr().round(3).to_dict(),
        },
        "portfolios": results,
    }
    with open("mc_results.json", "w") as f:
        json.dump(payload, f, indent=1)
    tbl.to_csv("mc_summary.csv", index=False)
    print("\nSaved mc_results.json, mc_summary.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "navs_monthly.csv")
