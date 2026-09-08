"""
fetch_navs.py — download monthly EUR NAV history for the portfolio funds.

Sources, tried in order per fund:
  1. Morningstar (via `mstarpy`, ISIN lookup, REST under the hood)
  2. Yahoo Finance (via `yfinance`) on a long-history ETF proxy tracking the same index
  3. FT Markets tearsheet (BeautifulSoup scrape of the historical-prices page)

Output: navs_monthly.csv  (Date index, one column per fund, month-end NAV in EUR)

    pip install mstarpy yfinance beautifulsoup4 requests pandas
    python fetch_navs.py
"""
import datetime as dt
import os
import sys
import time

# mstarpy>=11 drives Selenium/Chrome. Do NOT add --headless here: Morningstar's bot
# detection answers headless Chrome with "403 Forbidden", which is why mstarpy itself
# ships with --headless=new commented out. These flags only stop the driver crashing
# when it is launched repeatedly. mstarpy reads extra flags from this env var.
os.environ.setdefault(
    "SELENIUM_CHROME_FLAGS",
    "--no-sandbox --disable-dev-shm-usage --disable-gpu",
)

import pandas as pd
import requests
from bs4 import BeautifulSoup

START = dt.date(2005, 1, 1)
END = dt.date.today()

# name -> (fund ISIN, long-history fund proxy ISIN or None, Yahoo ETF proxy ticker)
FUNDS = {
    "World":      ("IE000ZYRH0Q7", "IE00B03HCZ61", "IWDA.AS"),   # iShares Dev World S / Vanguard Global Stock / iShares Core MSCI World
    "EM":         ("IE000QAZP7L2", "IE0031786142", "IEMM.AS"),   # iShares EM S / Vanguard EM Stock / iShares MSCI EM (EUR-quoted, 2008-)
    "GlobalBond": ("IE00B18GC888", None,           "AGGH.MI"),   # Vanguard Global Bond EUR Hdg / iShares Global Agg EUR Hdg
    "ShortBond":  ("IE0004ZP1ND3", "IE00BH65QK91", "ERNE.L"),    # iShares Agg 1-5Y hdg / Vanguard Global ST Bond hdg / iShares EUR Ultrashort (approx)
    "SmallCap":   ("IE00B42W3S00", None,           "IUSN.DE"),   # Vanguard Global Small-Cap Idx / iShares MSCI World Small Cap (EUR)
    "InflLinked": ("IE00B04GQR24", None,           "IBCI.DE"),   # Vanguard Eurozone Infl-Linked / iShares EUR Infl Linked Govt (EUR)
    "Carmignac":  ("LU1623762843", None,           None),        # Carmignac Pf Credit A EUR Acc — not on FT, needs Morningstar
}
# Also fetchable by ISIN if you want them (verified live on FT, both EUR):
#   "EuroGovt":   ("IE0007472990", None, None)   Vanguard Euro Govt Bond Idx   — 260 months, from 2005-01
#   "ACWI_IMI":   ("IE0003PI5332", None, None)   L&G MSCI ACWI IMI I EUR Acc   —  20 months, from 2025-01 (too short)

MIN_MONTHS = 36  # a source must cover this many months to be accepted outright

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}


# ---------------------------------------------------------------- 1. Morningstar
def fetch_morningstar(isin: str) -> pd.Series | None:
    try:
        import mstarpy
    except ImportError:
        print("  mstarpy not installed (pip install mstarpy)"); return None
    for attempt in (1, 2):  # the Selenium launch fails transiently; one retry is enough
        try:
            f = mstarpy.Funds(term=isin)  # mstarpy >=11 has no `country` kwarg
            hist = f.nav(start_date=START, end_date=END, frequency="monthly")
            df = pd.DataFrame(hist)
            if df.empty:
                return None
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date")["nav"].astype(float).sort_index()
        except Exception as e:
            first_line = str(e).strip().splitlines()[0]
            print(f"  Morningstar attempt {attempt} failed for {isin}: {first_line}")
            if attempt == 2:
                return None
            time.sleep(3)


# ---------------------------------------------------------------- 2. Yahoo ETF proxy
def fetch_yahoo(ticker: str) -> pd.Series | None:
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed (pip install yfinance)"); return None
    try:
        px = yf.download(ticker, start=START.isoformat(), end=END.isoformat(),
                         interval="1mo", auto_adjust=True, progress=False)
        if px.empty:
            return None
        col = "Close" if "Close" in px else px.columns[0]
        s = px[col].squeeze().dropna()
        s.index = pd.to_datetime(s.index)
        return s
    except Exception as e:
        print(f"  Yahoo failed for {ticker}: {e}")
        return None


# ---------------------------------------------------------------- 3. FT Markets (BeautifulSoup)
def fetch_ft(isin: str) -> pd.Series | None:
    """
    FT tearsheet -> find internal symbol id -> hit the historical-prices endpoint (returns HTML table).
    """
    try:
        url = f"https://markets.ft.com/data/funds/tearsheet/historical?s={isin}:EUR"
        html = requests.get(url, headers=UA, timeout=20).text
        soup = BeautifulSoup(html, "html.parser")
        section = soup.find("section", class_="mod-tearsheet-add-to-watchlist")
        xid = section["data-mod-config"] if section else None
        if xid:
            import json
            xid = json.loads(xid).get("xid")
        if not xid:
            return None
        rows = []
        # endpoint pages in ~1 year chunks
        d0 = START
        while d0 < END:
            d1 = min(d0 + dt.timedelta(days=365), END)
            api = ("https://markets.ft.com/data/equities/ajax/get-historical-prices"
                   f"?startDate={d0:%Y/%m/%d}&endDate={d1:%Y/%m/%d}&symbol={xid}")
            js = requests.get(api, headers=UA, timeout=20).json()
            table = BeautifulSoup(js.get("html", ""), "html.parser")
            for tr in table.find_all("tr"):
                tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                if len(tds) >= 5:
                    date_txt = tr.find("span", class_="mod-ui-hide-small-below")
                    date_txt = date_txt.get_text(strip=True) if date_txt else tds[0]
                    rows.append((pd.to_datetime(date_txt, errors="coerce"),
                                 float(tds[4].replace(",", ""))))
            d0 = d1 + dt.timedelta(days=1)
            time.sleep(0.5)
        s = pd.Series(dict(r for r in rows if pd.notna(r[0]))).sort_index()
        return s if len(s) else None
    except Exception as e:
        print(f"  FT failed for {isin}: {e}")
        return None


# last fully completed month-end, so the tail is never a part-month NAV
LAST_FULL_MONTH = pd.Timestamp(END.replace(day=1) - dt.timedelta(days=1))


def month_end(s: pd.Series) -> pd.Series:
    s = s.resample("ME").last().dropna()
    return s[s.index <= LAST_FULL_MONTH]


def fetch_one(name, isin, proxy_isin, yahoo):
    print(f"\n{name}  ({isin})")
    best = None  # (label, series) — longest series that fell short of MIN_MONTHS
    for label, fn, arg in [
        ("FT fund", fetch_ft, isin),
        ("FT proxy", fetch_ft, proxy_isin),
        ("Yahoo ETF proxy", fetch_yahoo, yahoo),
        ("Morningstar fund", fetch_morningstar, isin),
        ("Morningstar long-history proxy", fetch_morningstar, proxy_isin),
    ]:
        if arg is None:
            continue
        s = fn(arg)
        if s is None or s.empty:
            continue
        s = month_end(s)  # resample FIRST: the gate below counts months, not daily rows
        if s.empty:
            continue
        print(f"  {label}: {s.index[0]:%Y-%m} -> {s.index[-1]:%Y-%m}, {len(s)} months")
        if len(s) >= MIN_MONTHS:
            print(f"  OK via {label}")
            return s
        if best is None or len(s) > len(best[1]):
            best = (label, s)
    if best is not None:
        print(f"  ~~ only short history available; using {best[0]} ({len(best[1])} months)")
        return best[1]
    print("  !! no source returned data")
    return None


def main():
    out = {}
    for name, (isin, proxy, yahoo) in FUNDS.items():
        s = fetch_one(name, isin, proxy, yahoo)
        if s is not None:
            out[name] = s
    if not out:
        sys.exit("Nothing fetched — check network / install mstarpy, yfinance.")
    df = pd.DataFrame(out).sort_index()
    df.to_csv("navs_monthly.csv", index_label="Date")
    print("\nSaved navs_monthly.csv")
    print(df.tail())
    print("\nCoverage per fund:")
    for c in df.columns:
        col = df[c].dropna()
        flag = "  <-- SHORT" if len(col) < MIN_MONTHS else ""
        print(f"  {c:12s} {col.index[0]:%Y-%m} -> {col.index[-1]:%Y-%m}  {len(col):4d} months{flag}")
    missing = [n for n in FUNDS if n not in df.columns]
    if missing:
        print(f"\n!! no data at all for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
