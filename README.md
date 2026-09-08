# Investments

This repository is a small, reproducible research pipeline for comparing two
long-term, EUR-denominated investment portfolios. It combines historical monthly
fund data, risk-based allocation methods, Monte Carlo projections, and static HTML
reports. It is a research aid, not personalised investment advice.

The analysis covers six equity and bond funds, two illustrative allocations (a
60/40 portfolio for a 58-year-old and a 25/75 portfolio for a 64-year-old), and
several allocation methods. The reports make the central limitation explicit:
simulated outcomes are substantially more sensitive to the return assumption than
to the choice among the weighting methods.

## What is in the repository

| Path | Purpose |
| --- | --- |
| `fetch_navs.py` | Fetches monthly EUR NAV histories, trying FT Markets, Yahoo Finance proxies, and Morningstar in that order. |
| `navs_monthly.csv` | Input NAV history used by the analysis. |
| `hrp_nco.py` | Calculates hand-set, Hierarchical Risk Parity (HRP), and Nested Clustered Optimization (NCO) allocations and historical statistics. |
| `monte_carlo.py` | Runs the lump-sum projections: stationary block bootstrap, historical Gaussian, and forward Gaussian return engines. |
| `monte_carlo_dca.py` | Runs the EUR 20,000 initial investment plus EUR 600/month savings-plan projections. |
| `build_funds_meta.py` / `build_estim.py` | Produce the fund/allocation metadata and return-estimation precision data used in the reports. |
| `build_reports.py` | Builds the English technical report and Spanish savings-plan report from templates and generated JSON. |
| `reports/` | HTML templates, KaTeX assets, and the generated reports. |
| `uv.lock` | Exact, locked dependency resolution for reproducible environments. |

The primary generated artifacts are:

| Artifact | Produced by |
| --- | --- |
| `hrp_nco_weights.csv`, `hrp_nco_results.csv` | `hrp_nco.py` |
| `mc_results.json`, `mc_summary.csv` | `monte_carlo.py` |
| `mc_dca_results.json`, `mc_dca_summary.csv` | `monte_carlo_dca.py` |
| `funds_meta.json`, `estim.json` | `build_funds_meta.py`, `build_estim.py` |
| `reports/portfolio_mc.html`, `reports/savings_plan.html` | `build_reports.py` |

## Prerequisites

- `uv`
- Internet access to refresh NAV data
- Python 3.13 or newer (managed automatically by `uv` in the commands below)
- A locally installed Chrome/Chromium browser if the Morningstar fallback is used;
  `mstarpy` drives it through Selenium and may open a browser window.

## Install uv

On macOS or Linux, install `uv` with its official standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the shell, then confirm it is available:

```bash
uv --version
```

Other common installation options are:

```bash
# macOS (Homebrew)
brew install uv

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Windows (WinGet)
winget install --id=astral-sh.uv -e
```

See the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
for package-manager and platform-specific alternatives.

## Recreate the Python environment

From the repository root, create `.venv`, install the Python version if necessary,
and install exactly the versions recorded in `uv.lock`:

```bash
uv python install 3.13
uv sync --locked
```

`--locked` makes the command fail if `pyproject.toml` and `uv.lock` disagree,
rather than silently changing the dependency resolution. No manual activation is
needed: use `uv run` for every command. If an activated shell is useful, run
`source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in
PowerShell.

## Rebuild the analysis

The committed data and reports can be inspected as-is. To refresh all data-derived
artifacts, run the pipeline in this order from the repository root:

```bash
uv run --locked python fetch_navs.py
uv run --locked python hrp_nco.py
uv run --locked python monte_carlo.py
uv run --locked python monte_carlo_dca.py
uv run --locked python build_funds_meta.py
uv run --locked python build_estim.py
uv run --locked python build_reports.py
```

`fetch_navs.py` calls external data providers and may take a while. Its historical
proxies are deliberate: when the selected fund lacks a long enough history, the
script uses a related, longer-lived proxy and records a monthly series through the
last completed month. Refreshing data therefore changes downstream results.

To rebuild only the static reports after changing a template or existing generated
JSON, run:

```bash
uv run --locked python build_reports.py
```

Open `reports/portfolio_mc.html` for the English technical report and
`reports/savings_plan.html` for the Spanish savings-plan report in a browser.

## Model assumptions

The Monte Carlo scripts use 20,000 common-random-number paths and a 2% annual
inflation assumption. They compare three return engines:

- `bootstrap`: stationary block bootstrap of historical monthly return rows.
- `hist-mvn`: Gaussian simulation using historical means and covariance.
- `fwd-mvn`: Gaussian simulation using a 6% equity and 3.3% bond forward arithmetic
  return assumption, net of fund charges.

Both portfolio scenarios run until age 90 and include two 20% portfolio withdrawals
at the ends of years 10 and 20. The savings-plan variant contributes until age 85.
These are scenario inputs rather than forecasts; adjust the constants near the top
of `monte_carlo.py` and `monte_carlo_dca.py` before treating the output as a
different case study.

## KaTeX CSS

`reports/katex/katex_inline.css` is a committed, self-contained stylesheet used by
the technical report. Only regenerate it when changing its source stylesheet or
font files:

```bash
uv run --locked python build_katex_css.py
```
