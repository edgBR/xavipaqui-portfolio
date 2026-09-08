"""build_reports.py — assemble the published HTML reports from templates + simulation output.

    python build_reports.py            # -> reports/portfolio_mc.html

Templates live in reports/ (tech_head.html, tech_body.html, tech_js.html). Data is injected
into the <script type="application/json" id="data"> placeholder, so the report is a pure
function of the JSON the simulations write — rerun a sim, rebuild, republish.
"""
import json
import sys

TRIM = ("cagr_pct", "p_real_loss", "real_terminal_x", "terminal_x",
        "withdrawn_x", "received_x", "real_received_x",
        "median_maxdd_pct", "worst_maxdd_pct")


def slim_mc(path="mc_results.json"):
    d = json.load(open(path))
    out = {"meta": d["meta"], "portfolios": {}}
    for pl, pd in d["portfolios"].items():
        out["portfolios"][pl] = {"horizon_years": pd["horizon_years"], "methods": {}}
        for mn, md in pd["methods"].items():
            # every method x engine keeps its fan: the report lets the reader pick
            eng = {k: {**{kk: v[kk] for kk in TRIM}, "fan": v["fan"]}
                   for k, v in md["engines"].items()}
            out["portfolios"][pl]["methods"][mn] = {
                "weights_pct": md["weights_pct"], "equity_pct": md["equity_pct"],
                "ann_vol_pct": md["ann_vol_pct"], "mu_fwd_pct": md["mu_fwd_pct"],
                "mu_hist_pct": md["mu_hist_pct"], "fee_pct": md["fee_pct"], "engines": eng}
    return out


def build_technical():
    payload = {"mc": slim_mc(), "funds": json.load(open("funds_meta.json")),
               "estim": json.load(open("estim.json")),
               "rob": json.load(open("robustness.json"))}
    head = open("reports/tech_head.html").read()
    body = open("reports/tech_body.html").read()
    js = open("reports/tech_js.html").read()
    assert body.count("__DATA__") == 1, "body must carry exactly one __DATA__ placeholder"
    html = head + body.replace("__DATA__", json.dumps(payload, ensure_ascii=False,
                                                      separators=(",", ":"))) + js
    open("reports/portfolio_mc.html", "w").write(html)
    return html


ES_LABELS = {"58yo 60/40": dict(age=58, mix="60 % acciones / 40 % bonos", short="persona de 58 años"),
             "64yo 25/75": dict(age=64, mix="25 % acciones / 75 % bonos", short="persona de 64 años")}
KEEP = ("paid_in_eur", "paid_in_real_eur", "terminal_eur", "real_terminal_eur",
        "withdrawn_eur", "received_eur", "real_received_eur", "withdrawals",
        "multiple_of_paid_in", "irr_pct", "p_below_paid_in", "p_real_below_paid_in",
        "median_maxdd_pct", "worst_maxdd_pct", "fan")


def slim_dca(path="mc_dca_results.json"):
    d = json.load(open(path))
    W0, C = d["meta"]["w0_eur"], d["meta"]["contrib_eur"]
    out = {"meta": d["meta"], "people": {}}
    for pl, pd in d["portfolios"].items():
        h = pd["methods"]["Hand-set"]
        yrs = h["engines"]["fwd-mvn"]["fan"]["years"]
        out["people"][pl] = {
            **ES_LABELS[pl], "horizon_years": pd["horizon_years"],
            "contrib_years": pd["contrib_years"], "equity_pct": h["equity_pct"],
            "ann_vol_pct": h["ann_vol_pct"], "fee_pct": h["fee_pct"],
            "cautious": {k: h["engines"]["fwd-mvn"][k] for k in KEEP},
            "repeat": {k: h["engines"]["bootstrap"][k] for k in KEEP},
            "hist": {k: h["engines"]["hist-mvn"][k] for k in KEEP},
            "paid_by_year": [W0 + C * 12 * y for y in yrs]}
    return out


def calibration(path="walk_forward.json"):
    """How the two hand-set portfolios' predicted volatility compared with what
    actually happened out of sample. The plain-language report presents drawdown
    figures derived from in-sample covariance, so it has to say they came in mild."""
    m = json.load(open(path))["runs"]["raw"]["methods"]
    out = {}
    for key, who in (("Hand 60/40", "58yo 60/40"), ("Hand 25/75", "64yo 25/75")):
        v = m[key]
        out[who] = {"pred_vol_pct": v["pred_vol_pct"], "oos_vol_pct": v["oos_vol_pct"],
                    "understated_pct": round((v["oos_vol_pct"] / v["pred_vol_pct"] - 1) * 100, 1)}
    return out


def build_plain():
    payload = {"dca": slim_dca(), "funds": json.load(open("funds_meta.json")),
               "calib": calibration()}
    body = open("reports/plain_body.html").read()
    assert body.count("__DATA__") == 1
    html = (open("reports/plain_head.html").read()
            + body.replace("__DATA__", json.dumps(payload, ensure_ascii=False,
                                                  separators=(",", ":")))
            + open("reports/plain_js.html").read())
    open("reports/savings_plan.html", "w").write(html)
    return html


if __name__ == "__main__":
    for name, fn in (("portfolio_mc", build_technical), ("savings_plan", build_plain)):
        print(f"wrote reports/{name}.html ({len(fn()):,} bytes)")
