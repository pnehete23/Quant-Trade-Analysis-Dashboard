# Quant Dashboard — Deployment & Structure

## Live deployment (Railway)

The repo is configured to deploy as a Streamlit web service on Railway with no extra setup.

### Steps
1. Push this repo to GitHub.
2. In Railway: **New Project -> Deploy from GitHub repo** -> pick this repo.
3. Railway auto-detects `nixpacks.toml` + `requirements.txt` and builds Python 3.11.
4. Railway injects `$PORT` automatically; the start command in `Procfile` / `railway.json` binds Streamlit to it.
5. Once deployed, expose a public domain via **Settings -> Networking -> Generate Domain**.

### Health check
Streamlit's built-in `/_stcore/health` endpoint is used (configured in `railway.json`, 120s timeout).

### Local run
```bash
pip install -r requirements.txt
streamlit run "trading.quant mvp/src/visualization/dashboard.py"
```

## Repo structure (post-changes)

```
quantdashboard/
├── Procfile                       # Railway / Heroku-style start command
├── railway.json                   # Railway build/deploy config + healthcheck
├── nixpacks.toml                  # Nixpacks build (Python 3.11 + gcc)
├── runtime.txt                    # Python 3.11.9 pin
├── requirements.txt               # Single source of truth for deps (deduped)
├── .dockerignore                  # Excludes venv/data/secrets from build context
├── .streamlit/
│   └── config.toml                # Dark theme, headless server, no telemetry
├── info.md                        # This file
├── README.md
├── trading.quant mvp/
│   └── src/
│       ├── visualization/
│       │   └── dashboard.py       # Main Streamlit app (rewritten)
│       ├── models/
│       │   ├── momentum_strategy.py
│       │   ├── mean_reversion.py
│       │   └── risk_models.py
│       ├── backtesting/
│       │   ├── backtest_engine.py
│       │   └── performance_metrics.py
│       ├── risk_management/
│       │   ├── portfolio_risk.py
│       │   └── var_calculator.py
│       ├── data_collection/
│       │   ├── yfinance_collector.py
│       │   └── data_processor.py
│       ├── notebook/
│       └── tests/
└── clv-prediction/                # Unrelated sibling project, not deployed
```

## Changes log

### Dashboard (`trading.quant mvp/src/visualization/dashboard.py`)
- **Layout pivoted from sidebar -> top control bar.** Inputs (mode, ticker(s), dates, capital, run) live in a single horizontal row at the top. Strategy/risk knobs moved to a collapsible expander to declutter.
- **Removed the slider-heavy UX.** MA windows, RSI period, max position are now `number_input`s (typed values, no slider drag noise).
- **Dark theme + custom CSS.** Branded gradient header, card-style metrics, accent-colored tabs, dark Plotly template throughout.
- **Modular renderers.** `render_overview`, `render_performance`, `render_risk`, `render_trades`, `render_report`, `render_diagnostics` — replaces the monolithic `main()` body.
- **Data layer hardened:** `load_data` flattens yfinance MultiIndex columns, sets `auto_adjust=False`, swallows network errors and returns `None` instead of crashing.
- **Charts restyled** with green/red/violet accent palette and shared x-axes.

### Guardrails added
- `validate_ticker` regex: `^[A-Z0-9][A-Z0-9.\-]{0,9}$` (rejects garbage input before hitting yfinance).
- `validate_portfolio_tickers`: trims, dedupes, caps at `MAX_PORTFOLIO_TICKERS = 8`, separates valid/invalid.
- `validate_config`: enforces start < end, min 90-day window, short MA < long MA, min 2 valid tickers in portfolio mode.
- `MIN_BARS_FOR_BACKTEST = 60` floor — refuses to backtest on tiny data slices.
- Pipeline checks `len(data) >= max(long_w + 10, MIN_BARS_FOR_BACKTEST)` before running.
- Portfolio backtester aborts (with warning) if NaN share > 25% or overlap < 60 bars.
- All errors surface as `st.error` instead of stack traces.
- Module import failure now `st.stop()`s cleanly with a message.

### Deployment files (new)
- `Procfile` — start command for Railway/Heroku-style runners.
- `railway.json` — Nixpacks builder, start command, `/_stcore/health` healthcheck, restart-on-failure (5 retries).
- `nixpacks.toml` — explicit Python 3.11 + gcc, pip install from `requirements.txt`.
- `runtime.txt` — Python `3.11.9` pin.
- `.streamlit/config.toml` — headless mode, CORS/XSRF disabled (Railway proxy handles), dark theme matching dashboard CSS, telemetry off.
- `.dockerignore` — keeps venv, .git, raw data CSVs, claude config, secrets, sibling projects out of the build context.

### Requirements
- `requirements.txt` deduped (was listing pandas/numpy/streamlit/etc. twice with conflicting comments). Single coherent list with modern minimum versions.

### Backend trade-log surfacing
- `BacktestEngine.calculate_performance_metrics` now returns a `trades` DataFrame (entry/exit date, prices, qty, P&L, return %, duration). Was previously hidden in `self.trades` and never exposed.
- Dashboard headline trade count now uses the engine's actual count (e.g., 18-20 real round-trips), not the misleading "294" from `MomentumStrategy.calculate_performance_metrics` which counts every nonzero daily return.

### UX/visual upgrades (round 2)
- **How-to cards**: every page (Backtest / Watchlist / Compare / Optimize) now has a clear instructional banner at the top — numbered steps + a 💡 Tip line.
- **Visual KPI strips**: the Backtest Overview and Optimize Walk-forward both show colored KPI ribbons (best/worst trade, MA spread, RSI, train/test verdict).
- **Price chart**: best trade now annotated with ⭐ gold callout, worst trade with ⚠ red callout — instant identification of pivotal moments.
- **Equity curve**: max-drawdown period shaded in red, peak marked with gold star, trough with red X, current value with green dot — every key moment is now a visible KPI.
- **Walk-forward chart**: replaced 3 metric numbers with a real dual equity curve overlaying strategy + buy-and-hold for both train and test periods, with a gold split-marker dividing the regions.
- **Heatmap**: highlights the optimum cell with ⭐, fixed colorscale (zmin/zmax = ±2 Sharpe) for honest comparison, and a side-by-side **top-5 combos bar chart** so users see the best parameters explicitly.
- **Cost-sensitivity chart**: now shows the break-even commission rate as a gold vertical line ("Max viable: X%"), plus reference lines for retail (~0.1%) and high-touch (~0.3%) brokers.
- **Diagnostics promoted**: removed the "you must enable diagnostics" toggle. Diagnostics tab is now always populated and clearly labeled (🔧 emoji on tab). Sidebar advertises it. Inside, the tab now leads with a KPI strip (rows / missing % / signals computed / active signals) so health is visible at a glance.
- **Sidebar**: clearer "🚀 Navigate" label, structured "📅 Daily flow" guide, and pointer to the diagnostics tab.

### Daily-user feature pack (4 pages)
Sidebar now has a page navigator. Single backtest is still the default, plus three new pages designed for repeat daily use:

1. **Watchlist Scanner** — scan up to 30 tickers (default: 15 large-cap + sector ETFs) and surface BUY / SELL / HOLD signals with current price, day %, RSI, MA spread (conviction proxy), and signal age. 15-min cache. The "do I open this app daily?" answer.
2. **Strategy Comparison** — runs Momentum (MA + RSI), Mean Reversion (Z-score), and Buy-and-Hold side-by-side on the same data. Equity curves overlaid; metric table shows return / Sharpe / drawdown / volatility / trade count. Adds a benchmark overlay vs SPY / QQQ / IWM normalized.
3. **Optimize & Validate** — three credibility checks in one page:
   - **Walk-forward 70/30 split**: train on first 70%, evaluate on unseen 30%. Surfaces Sharpe decay → overfitting alarm.
   - **Parameter heatmap**: 5×5 grid of (short MA, long MA) combinations colored by Sharpe. Robust strategies show green *regions*; overfit ones show isolated peaks. Runs in <1s on Railway.
   - **Cost-sensitivity sweep**: re-runs at commission rates from 0% to 0.5%. Shows whether the edge survives realistic costs or is a transaction-cost mirage.
4. **Backtest** (existing) — unchanged single-symbol deep dive with action banner, real entry/exit pairs, trade log.

### Mean-reversion wired in
`models/mean_reversion.py` was sitting unused. Now imported and used in the comparison page. No new dependencies.

### Interactive chart upgrades
- **Action banner** above the price chart: declares BUY / SELL / HOLD for today based on the most recent signal, shows current price, and surfaces unrealized P&L on any open position.
- **Real entry/exit pairs** drawn on the price chart: green ▲ for entries, green ▼ for profitable exits, red ▼ for loss exits. Dotted connector lines colored by P&L make trade outcomes obvious at a glance.
- **Hover tooltips** on every marker show entry price, exit price, P&L $ + %, and holding period.
- **Range selector buttons** (1M / 3M / 6M / YTD / 1Y / All) on the price chart.
- **Unified hover mode** across price/RSI/volume panels.
- **Trade log table** in the Trades tab: sortable, formatted, with color-coded WIN/LOSS column.
- **Realized P&L histogram** in the Trades tab now uses actual trade outcomes (win/loss colored), not synthesized random samples.

## Environment variables

Railway injects automatically:
- `PORT` — bound to Streamlit's `--server.port`.

Optional:
- None required. App is fully read-only against public yfinance endpoints.

## Notes / known limits

- yfinance is rate-limited and occasionally flaky from cloud IPs. Dashboard handles empty/None responses gracefully but heavy concurrent use may degrade.
- No auth on the deployed app. If exposing publicly with sensitive params, front it with Railway's HTTP basic auth or a reverse proxy.
- Folder name `trading.quant mvp` contains a space — start commands quote it correctly. Don't rename without updating `Procfile`, `railway.json`, and `nixpacks.toml`.
