"""
hrp_nco.py — Hierarchical Risk Parity and Nested Clustered Optimization
on the monthly NAV file produced by fetch_navs.py.

    pip install pandas numpy scipy scikit-learn
    python hrp_nco.py navs_monthly.csv

Outputs:
  1. Unconstrained HRP and NCO weights (what the pure methods say)
  2. Sleeve-constrained HRP: HRP inside equities and inside bonds,
     then fixed 60/40 (58yo) and 25/75 (64yo) across sleeves
  3. Realised annualised return / vol / max drawdown of each weighting
     vs the hand-set portfolios, on the same history
"""
import sys
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

EQUITY = ["World", "EM", "SmallCap"]
BONDS = ["GlobalBond", "ShortBond", "InflLinked"]
EQ_SHARE = {"58yo 60/40": 0.60, "64yo 25/75": 0.25}
HAND = {
    "58yo 60/40 (hand)": {"World": .45, "EM": .08, "SmallCap": .07,
                          "GlobalBond": .20, "ShortBond": .12, "InflLinked": .08},
    "64yo 25/75 (hand)": {"World": .20, "EM": .03, "SmallCap": .02,
                          "GlobalBond": .25, "ShortBond": .30, "InflLinked": .20},
}
for _k, _w in HAND.items():  # weights must sum to 1 and match the stated equity share
    assert abs(sum(_w.values()) - 1) < 1e-9, f"{_k} sums to {sum(_w.values())}"
    assert abs(sum(_w[a] for a in EQUITY) - EQ_SHARE[_k.split(" (")[0]]) < 1e-9, f"{_k} equity share"


# ------------------------------------------------------------------ denoising (López de Prado 2019/2020)
def mp_edge(n_assets: int, n_obs: int, sigma2: float = 1.0) -> float:
    """Upper edge of the Marchenko-Pastur bulk: eigenvalues below this are consistent
    with pure noise. q = T/N; sigma2 = 1 for a correlation matrix.

    NOTE ON VALIDITY: MP is an asymptotic result (N, T -> inf at fixed q). LdP fits
    sigma2 by matching a KDE of the empirical spectrum to the MP density, which needs a
    well-populated bulk. With N = 6 there is no bulk to fit, so we use the analytic
    sigma2 = 1 edge and treat the outcome as indicative, not authoritative.
    """
    q = n_obs / n_assets
    return sigma2 * (1.0 + np.sqrt(1.0 / q)) ** 2


def denoise_corr(corr: np.ndarray, n_obs: int, detone: int = 0) -> np.ndarray:
    """Constant-residual-eigenvalue denoising. Keep the eigenvalues above the MP edge;
    replace the rest by their mean so the trace is preserved; rebuild and renormalise.

    detone > 0 additionally zeroes that many leading eigenvalues (the market
    component), which LdP suggests so clustering is not dominated by it. At N = 6 the
    first eigenvector carries ~60 % of the variance, so detoning is drastic here --
    hence the default of 0.
    """
    n = corr.shape[0]
    w, v = np.linalg.eigh(corr)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    k = int((w >= mp_edge(n, n_obs)).sum())
    k = max(k, 1)                      # never shrink the whole spectrum away
    w2 = w.copy()
    if k < n:
        w2[k:] = w2[k:].mean()
    if detone:
        w2[:detone] = 0.0
    c2 = v @ np.diag(w2) @ v.T
    d = np.sqrt(np.clip(np.diag(c2), 1e-12, None))
    out = c2 / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return out


def denoise_cov(cov: pd.DataFrame, n_obs: int, detone: int = 0) -> pd.DataFrame:
    """Denoise a covariance matrix by denoising its correlation and re-applying the
    original standard deviations (which are estimated far more precisely than the
    correlation structure, so they are left alone)."""
    sd = np.sqrt(np.diag(cov.values))
    corr = cov.values / np.outer(sd, sd)
    dn = denoise_corr(corr, n_obs, detone)
    return pd.DataFrame(np.outer(sd, sd) * dn, index=cov.index, columns=cov.columns)


def spectrum_report(cov: pd.DataFrame, n_obs: int) -> dict:
    """Diagnostics for the report: where the MP edge falls and how much of the
    spectrum it classifies as noise."""
    n = cov.shape[0]
    sd = np.sqrt(np.diag(cov.values))
    corr = cov.values / np.outer(sd, sd)
    ev = np.sort(np.linalg.eigvalsh(corr))[::-1]
    edge = mp_edge(n, n_obs)
    below = ev < edge
    return {"n_assets": n, "n_obs": n_obs, "t_over_n": round(n_obs / n, 1),
            "mp_edge": round(float(edge), 3),
            "eigenvalues": [round(float(x), 3) for x in ev],
            "n_noise": int(below.sum()),
            "noise_share_of_variance_pct": round(float(ev[below].sum() / n) * 100, 1),
            "closest_to_edge_gap_pct": round(float(min(abs(ev - edge)) / edge) * 100, 2)}


# ------------------------------------------------------------------ HRP (López de Prado 2016)
def corr_dist(corr):
    return np.sqrt(np.clip(0.5 * (1 - corr), 0, None))


def quasi_diag(link):
    return dendrogram(link, no_plot=True)["leaves"]


def ivp(cov):
    w = 1 / np.diag(cov)
    return w / w.sum()


def cluster_var(cov, idx):
    c = cov.iloc[idx, idx].values
    w = ivp(c)
    return w @ c @ w


def hrp(cov: pd.DataFrame) -> pd.Series:
    corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    dist = corr_dist(corr.values)
    link = linkage(squareform(dist, checks=False), method="single")
    order = quasi_diag(link)
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [c[j:k] for c in clusters for j, k in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0, v1 = cluster_var(cov, c0), cluster_var(cov, c1)
            a = 1 - v0 / (v0 + v1)
            w[c0] *= a
            w[c1] *= 1 - a
    w.index = cov.index[w.index]
    return w.sort_index()


# ------------------------------------------------------------------ NCO (López de Prado 2019)
def min_var(cov, mu=None):
    inv = np.linalg.pinv(cov)
    ones = np.ones((cov.shape[0], 1))
    if mu is None:
        mu = ones
    w = inv @ mu
    return (w / (ones.T @ w)).flatten()


def cluster_kmeans(corr, max_k=None):
    dist = corr_dist(corr.values)
    n = corr.shape[0]
    max_k = max_k or max(2, n // 2)
    best, best_score = None, -np.inf
    for k in range(2, max_k + 1):
        for seed in range(10):
            km = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(dist)
            sil = silhouette_samples(dist, km.labels_)
            score = sil.mean() / sil.std() if sil.std() > 0 else sil.mean()
            if score > best_score:
                best, best_score = km, score
    return {c: list(np.where(best.labels_ == c)[0]) for c in np.unique(best.labels_)}


def nco(cov: pd.DataFrame, mu: pd.Series | None = None) -> pd.Series:
    """mu=None -> minimum-variance NCO; mu given -> max-Sharpe-style NCO."""
    corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    clusters = cluster_kmeans(corr)
    intra = pd.DataFrame(0.0, index=cov.index, columns=clusters.keys())
    for c, idx in clusters.items():
        sub_cov = cov.iloc[idx, idx].values
        sub_mu = None if mu is None else mu.iloc[idx].values.reshape(-1, 1)
        intra.iloc[idx, list(clusters).index(c)] = min_var(sub_cov, sub_mu)
    cov_c = intra.T @ cov @ intra
    mu_c = None if mu is None else (intra.T @ mu).values.reshape(-1, 1)
    inter = pd.Series(min_var(cov_c.values, mu_c), index=cov_c.index)
    return (intra @ inter).clip(lower=0).pipe(lambda w: w / w.sum())


# ------------------------------------------------------------------ constrained version
def sleeve_hrp(cov, eq_share):
    w_eq = hrp(cov.loc[EQUITY, EQUITY]) * eq_share
    w_bd = hrp(cov.loc[BONDS, BONDS]) * (1 - eq_share)
    return pd.concat([w_eq, w_bd]).reindex(cov.index).fillna(0)


# ------------------------------------------------------------------ evaluation
def stats(ret: pd.DataFrame, w: pd.Series):
    w = w.reindex(ret.columns).fillna(0)
    p = (ret * w).sum(axis=1)  # monthly rebalanced
    nav = (1 + p).cumprod()
    yrs = len(p) / 12
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    vol = p.std() * np.sqrt(12)
    dd = (nav / nav.cummax() - 1).min()
    return cagr, vol, dd


def main(path):
    navs = pd.read_csv(path, index_col=0, parse_dates=True)
    cols = [c for c in EQUITY + BONDS if c in navs]
    raw = navs[cols]
    navs = raw.dropna()                 # common history only
    ret = navs.pct_change().dropna()
    print(f"History used: {ret.index[0]:%Y-%m} -> {ret.index[-1]:%Y-%m}  ({len(ret)} months), assets: {cols}\n")
    binding = [c for c in cols if raw[c].first_valid_index() >= navs.index[0]]
    dropped = len(raw.loc[:navs.index[0]].index) - 1
    if dropped > 0:
        print(f"NOTE: common-window alignment discards {dropped} earlier months "
              f"(full data starts {raw.index[0]:%Y-%m}); binding sleeve(s): {', '.join(binding)}.")
        if raw.index[0].year <= 2008 <= navs.index[0].year - 1:
            print("      2008 is NOT in the sample, so MaxDD below understates a crisis.\n")
        else:
            print()

    cov = ret.cov() * 12
    mu_hist = ret.mean() * 12  # only used for the return-aware NCO variant

    print("Annualised vol:\n", (np.sqrt(np.diag(cov)) * 100).round(1), "\n")
    print("Correlation:\n", ret.corr().round(2), "\n")

    weights = {
        "HRP (unconstrained)": hrp(cov),
        "NCO min-var (unconstrained)": nco(cov),
        "NCO w/ hist. returns (unconstrained)": nco(cov, mu_hist),
        **{f"{k} HRP-in-sleeves": sleeve_hrp(cov, v) for k, v in EQ_SHARE.items()},
    }
    for k, v in HAND.items():
        weights[k] = pd.Series(v).reindex(cols).fillna(0)

    W = pd.DataFrame(weights).T[cols]
    print("Weights (%):\n", (W * 100).round(1), "\n")

    rows = []
    for name, w in W.iterrows():
        cagr, vol, dd = stats(ret, w)
        rows.append([name, cagr * 100, vol * 100, dd * 100, (w[EQUITY].sum()) * 100])
    res = pd.DataFrame(rows, columns=["Portfolio", "CAGR %", "Vol %", "MaxDD %", "Equity %"]).set_index("Portfolio")
    print("Realised on this history (monthly rebalanced):\n", res.round(1))
    res.to_csv("hrp_nco_results.csv")
    W.to_csv("hrp_nco_weights.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "navs_monthly.csv")
