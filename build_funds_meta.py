"""build_funds_meta.py — fund metadata + per-plan euro splits for the reports.
Regenerates funds_meta.json from mc_dca_results.json. ISINs/TERs verified on FT 2026-09-08."""
import json

FUNDS = {
 "World":     dict(isin="IE000ZYRH0Q7", ter=0.06, sleeve="rv",
   name="iShares Developed World Index Fund (IE) S Acc EUR",
   en="Developed-world companies", corto="Empresas de países ricos",
   en_plain="Shares in about 1,400 large companies across the US, Europe and Japan. The engine of the portfolio.",
   llano="Acciones de unas 1.400 grandes empresas de EE. UU., Europa y Japón. Es el motor de la cartera."),
 "EM":        dict(isin="IE000QAZP7L2", ter=0.16, sleeve="rv",
   name="iShares Emerging Markets Index Fund (IE) S Acc EUR",
   en="Emerging-market companies", corto="Empresas de países emergentes",
   en_plain="Shares in companies from China, India, Brazil, Taiwan. They grow faster but fall harder.",
   llano="Acciones de empresas de China, India, Brasil, Taiwán… Crecen más rápido, pero caen más fuerte."),
 "SmallCap":  dict(isin="IE00B42W3S00", ter=0.29, sleeve="rv",
   name="Vanguard Global Small-Cap Index Fund Investor EUR Acc",
   en="Small companies", corto="Empresas pequeñas",
   en_plain="Shares in small companies in rich countries. More risk and more reward over very long periods.",
   llano="Acciones de empresas pequeñas de países ricos. Más riesgo y más recompensa a muy largo plazo."),
 "GlobalBond":dict(isin="IE00B18GC888", ter=0.15, sleeve="rf",
   name="Vanguard Global Bond Index Fund EUR Hedged Acc",
   en="Global bonds (intermediate)", corto="Bonos globales (plazo medio)",
   en_plain="Loans to governments and large companies worldwide, with the currency risk hedged away.",
   llano="Préstamos a gobiernos y grandes empresas de todo el mundo, con el riesgo de divisa cubierto."),
 "ShortBond": dict(isin="IE0004ZP1ND3", ter=0.08, sleeve="rf",
   name="iShares Global Aggregate 1-5 Year Bond Index (IE) S Acc EUR Hedged",
   en="Global bonds (short-dated)", corto="Bonos globales (plazo corto)",
   en_plain="The same, but repayable in 1-5 years. The calmest part: it barely moves.",
   llano="Lo mismo, pero a devolver en 1–5 años. Es la parte más tranquila: apenas se mueve."),
 "InflLinked":dict(isin="IE00B04GQR24", ter=0.12, sleeve="rf",
   name="Vanguard Eurozone Inflation-Linked Bond Index Fund EUR Acc",
   en="Inflation-linked bonds", corto="Bonos ligados a la inflación",
   en_plain="Bonds that pay more when prices rise. They protect against exactly what hurts a retiree most.",
   llano="Bonos que pagan más si los precios suben. Protegen justo de lo que más daña a un jubilado."),
}
ORDER = ["World", "EM", "SmallCap", "GlobalBond", "ShortBond", "InflLinked"]


def build(dca_path="mc_dca_results.json", lump_path="mc_results.json"):
    dca = json.load(open(dca_path))
    W0, C = dca["meta"]["w0_eur"], dca["meta"]["contrib_eur"]
    out = {"funds": {k: FUNDS[k] for k in ORDER}, "order": ORDER, "plans": {}}
    for pl, pd in dca["portfolios"].items():
        w = pd["methods"]["Hand-set"]["weights_pct"]
        ter = sum(w[k] / 100 * FUNDS[k]["ter"] for k in ORDER)
        out["plans"][pl] = {
            "weights_pct": {k: round(w[k], 1) for k in ORDER},
            "initial_eur": {k: round(W0 * w[k] / 100, 2) for k in ORDER},
            "monthly_eur": {k: round(C * w[k] / 100, 2) for k in ORDER},
            "ter_pct": round(ter, 3),
            "rv_pct": round(sum(w[k] for k in ORDER if FUNDS[k]["sleeve"] == "rv"), 1),
            "rf_pct": round(sum(w[k] for k in ORDER if FUNDS[k]["sleeve"] == "rf"), 1),
            "ter_cost_year1_eur": round(W0 * ter / 100, 2)}
    lump = json.load(open(lump_path))
    out["method_weights"] = {pl: {m: {k: round(md["weights_pct"].get(k, 0), 1) for k in ORDER}
                                  for m, md in pd["methods"].items()}
                             for pl, pd in lump["portfolios"].items()}
    return out


if __name__ == "__main__":
    json.dump(build(), open("funds_meta.json", "w"), ensure_ascii=False, indent=1)
    print("wrote funds_meta.json")
