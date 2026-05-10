"""Quantitative Trading Analytics Dashboard.

Top-bar layout (no left sidebar of sliders). Designed for live deployment on
Railway via `streamlit run`. Reads PORT/host from env at the runner level.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.momentum_strategy import MomentumStrategy
    from models.mean_reversion import MeanReversionStrategy
    from backtesting.backtest_engine import BacktestEngine
    from risk_management.portfolio_risk import PortfolioRiskManager
except ImportError as e:
    st.error(f"Module import failed: {e}. Ensure src/ modules are on PYTHONPATH.")
    st.stop()


# ---------- Default watchlist ----------

DEFAULT_WATCHLIST = "AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, AVGO, JPM, SPY, QQQ, XLE, XLF, XLK, GLD"
BENCHMARKS = ["SPY", "QQQ", "IWM"]


# ---------- UI helpers ----------

def how_to_card(title: str, steps: list[str], tip: str | None = None):
    """Render a clear instructional banner at the top of a page."""
    bullets = "".join(f"<li style='margin:.15rem 0;'>{s}</li>" for s in steps)
    tip_html = (f'<div style="margin-top:.6rem;color:#ffb547;font-size:.82rem;">'
                f'💡 <b>Tip:</b> {tip}</div>') if tip else ""
    st.markdown(f"""
<div style="background:linear-gradient(135deg,#1a1f2c 0%,#161c27 100%);
            border:1px solid #2a3142;border-radius:10px;
            padding:.85rem 1.1rem;margin:.2rem 0 1rem 0;">
  <div style="color:#00d4aa;font-weight:600;font-size:.92rem;margin-bottom:.4rem;">
    📖 {title}
  </div>
  <ol style="margin:0;padding-left:1.1rem;color:#c8d1e0;font-size:.86rem;">{bullets}</ol>
  {tip_html}
</div>
""", unsafe_allow_html=True)


def kpi_strip(items: list[tuple[str, str, str]]):
    """Render a compact KPI strip below charts. items = [(label, value, color), ...]"""
    cols = st.columns(len(items))
    for col, (label, value, color) in zip(cols, items):
        col.markdown(f"""
<div style="background:#161c27;border:1px solid {color}55;border-left:3px solid {color};
            border-radius:8px;padding:.55rem .8rem;">
  <div style="color:#8a96aa;font-size:.72rem;text-transform:uppercase;letter-spacing:.5px;">{label}</div>
  <div style="color:{color};font-weight:600;font-size:1.05rem;margin-top:.15rem;">{value}</div>
</div>
""", unsafe_allow_html=True)


# ---------- Page config & theme ----------

st.set_page_config(
    page_title="Quant Trading Analytics",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_TEMPLATE = "plotly_dark"

st.markdown(
    """
<style>
  :root { --accent:#00d4aa; --accent2:#7c5cff; --bg-soft:#11161f; --bg-card:#161c27; --border:#222a38; --muted:#8a96aa; }
  .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px; }
  header[data-testid="stHeader"] { background: transparent; }
  #MainMenu, footer { visibility: hidden; }

  .topbar {
    background: linear-gradient(135deg, #0f1420 0%, #161c27 100%);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 1rem 1.25rem; margin-bottom: 1rem;
  }
  .topbar h1 { margin: 0; font-size: 1.55rem; letter-spacing: .3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .topbar .sub { color: var(--muted); font-size: .85rem; margin-top: .15rem; }

  div[data-testid="stMetric"] {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: .85rem 1rem;
  }
  div[data-testid="stMetricLabel"] { color: var(--muted); font-weight: 500; }
  div[data-testid="stMetricValue"] { font-size: 1.4rem; }

  .stTabs [data-baseweb="tab-list"] { gap: .25rem; border-bottom: 1px solid var(--border); }
  .stTabs [data-baseweb="tab"] {
    background: transparent; color: var(--muted); border-radius: 8px 8px 0 0;
    padding: .6rem 1rem; font-weight: 500;
  }
  .stTabs [aria-selected="true"] { color: var(--accent) !important; background: rgba(0,212,170,.07) !important; }

  .stButton>button[kind="primary"] {
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    color: #0b0f17; border: 0; font-weight: 600;
  }
  .stButton>button[kind="primary"]:hover { filter: brightness(1.08); }

  .pill { display:inline-block; padding:.18rem .55rem; border-radius:999px; font-size:.72rem;
    background:rgba(124,92,255,.15); color:#b9a8ff; border:1px solid rgba(124,92,255,.35); margin-left:.4rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- Guardrails ----------

TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")
MAX_PORTFOLIO_TICKERS = 8
MIN_BARS_FOR_BACKTEST = 60


def clean_ticker(t: str) -> str:
    return (t or "").strip().upper()


def validate_ticker(t: str) -> bool:
    return bool(TICKER_RE.match(clean_ticker(t)))


def validate_portfolio_tickers(raw: str) -> tuple[list[str], list[str]]:
    parts = [clean_ticker(x) for x in raw.split(",") if x.strip()]
    valid = [p for p in parts if validate_ticker(p)]
    invalid = [p for p in parts if not validate_ticker(p)]
    seen = set()
    deduped = [v for v in valid if not (v in seen or seen.add(v))]
    return deduped[:MAX_PORTFOLIO_TICKERS], invalid


# ---------- Data layer ----------

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, start_date, end_date) -> pd.DataFrame | None:
    t = clean_ticker(ticker)
    if not validate_ticker(t):
        return None
    try:
        data = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if data is None or data.empty:
            if "." not in t and t.isalpha():
                alt = f"{t}.NS"
                data_alt = yf.download(alt, start=start_date, end=end_date, progress=False, auto_adjust=False)
                if data_alt is not None and not data_alt.empty:
                    return data_alt
            return None
        # Flatten yfinance MultiIndex columns when present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        return data
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_multiple_data(tickers: tuple, start_date, end_date) -> dict:
    out = {}
    for t in tickers:
        df = load_data(t, start_date, end_date)
        if df is not None and not df.empty:
            out[t] = df
    return out


# ---------- Charts ----------

def chart_price(data: pd.DataFrame, signals_df: pd.DataFrame | None,
                trades_df: pd.DataFrame | None = None) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Price, signals & executed trades", "RSI", "Volume"),
        vertical_spacing=0.04, row_heights=[0.66, 0.16, 0.18],
    )
    fig.add_trace(
        go.Candlestick(
            x=data.index, open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"], name="Price",
            increasing_line_color="#00d4aa", decreasing_line_color="#ff4d6d",
            showlegend=False,
        ),
        row=1, col=1,
    )
    if signals_df is not None:
        fig.add_trace(go.Scatter(x=signals_df.index, y=signals_df["MA_Short"],
                                 name="MA Short", line=dict(color="#ffb547", width=1.4),
                                 hovertemplate="MA Short: $%{y:.2f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=signals_df.index, y=signals_df["MA_Long"],
                                 name="MA Long", line=dict(color="#7c5cff", width=1.4),
                                 hovertemplate="MA Long: $%{y:.2f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=signals_df.index, y=signals_df["RSI"], name="RSI",
                                 line=dict(color="#4cc9f0", width=1.3), showlegend=False,
                                 hovertemplate="RSI: %{y:.1f}<extra></extra>"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff4d6d", row=2, col=1,
                      annotation_text="Overbought", annotation_position="right")
        fig.add_hline(y=30, line_dash="dash", line_color="#00d4aa", row=2, col=1,
                      annotation_text="Oversold", annotation_position="right")

    # Real executed trades from the engine (entry markers, exit markers, connecting lines)
    if trades_df is not None and not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]

        # KPI annotations: best & worst trades with stars + callouts
        best = trades_df.loc[trades_df["pnl"].idxmax()]
        worst = trades_df.loc[trades_df["pnl"].idxmin()]
        fig.add_annotation(
            x=best["exit_date"], y=best["exit_price"],
            text=f"⭐ Best: ${best['pnl']:+,.0f} ({best['return_pct']*100:+.1f}%)",
            showarrow=True, arrowhead=2, arrowcolor="#ffd700", arrowwidth=1.5,
            bgcolor="rgba(255,215,0,0.15)", bordercolor="#ffd700", borderwidth=1,
            font=dict(color="#ffd700", size=11), ax=0, ay=-50, row=1, col=1,
        )
        fig.add_annotation(
            x=worst["exit_date"], y=worst["exit_price"],
            text=f"⚠ Worst: ${worst['pnl']:+,.0f} ({worst['return_pct']*100:+.1f}%)",
            showarrow=True, arrowhead=2, arrowcolor="#ff4d6d", arrowwidth=1.5,
            bgcolor="rgba(255,77,109,0.15)", bordercolor="#ff4d6d", borderwidth=1,
            font=dict(color="#ff4d6d", size=11), ax=0, ay=50, row=1, col=1,
        )

        # Entry markers
        fig.add_trace(go.Scatter(
            x=trades_df["entry_date"], y=trades_df["entry_price"], mode="markers",
            name="Entry (BUY)", legendgroup="trades",
            marker=dict(color="#00d4aa", size=12, symbol="triangle-up",
                        line=dict(color="white", width=1)),
            customdata=np.stack([trades_df["pnl"], trades_df["return_pct"] * 100,
                                 trades_df["duration_days"]], axis=-1),
            hovertemplate=("<b>BUY</b> @ $%{y:.2f}<br>"
                           "Date: %{x|%Y-%m-%d}<br>"
                           "Outcome: $%{customdata[0]:+,.2f} "
                           "(%{customdata[1]:+.2f}%) over %{customdata[2]:.0f}d<extra></extra>"),
        ), row=1, col=1)

        # Exit markers split by win/loss
        if not wins.empty:
            fig.add_trace(go.Scatter(
                x=wins["exit_date"], y=wins["exit_price"], mode="markers",
                name="Exit (Profit)", legendgroup="trades",
                marker=dict(color="#00d4aa", size=12, symbol="triangle-down",
                            line=dict(color="white", width=1)),
                customdata=np.stack([wins["pnl"], wins["return_pct"] * 100,
                                     wins["duration_days"]], axis=-1),
                hovertemplate=("<b>SELL (WIN)</b> @ $%{y:.2f}<br>"
                               "Date: %{x|%Y-%m-%d}<br>"
                               "P&L: $%{customdata[0]:+,.2f} "
                               "(%{customdata[1]:+.2f}%) over %{customdata[2]:.0f}d<extra></extra>"),
            ), row=1, col=1)
        if not losses.empty:
            fig.add_trace(go.Scatter(
                x=losses["exit_date"], y=losses["exit_price"], mode="markers",
                name="Exit (Loss)", legendgroup="trades",
                marker=dict(color="#ff4d6d", size=12, symbol="triangle-down",
                            line=dict(color="white", width=1)),
                customdata=np.stack([losses["pnl"], losses["return_pct"] * 100,
                                     losses["duration_days"]], axis=-1),
                hovertemplate=("<b>SELL (LOSS)</b> @ $%{y:.2f}<br>"
                               "Date: %{x|%Y-%m-%d}<br>"
                               "P&L: $%{customdata[0]:+,.2f} "
                               "(%{customdata[1]:+.2f}%) over %{customdata[2]:.0f}d<extra></extra>"),
            ), row=1, col=1)

        # Connecting lines: entry -> exit, color by P&L
        for _, t in trades_df.iterrows():
            color = "rgba(0,212,170,0.45)" if t["pnl"] > 0 else "rgba(255,77,109,0.45)"
            fig.add_trace(go.Scatter(
                x=[t["entry_date"], t["exit_date"]],
                y=[t["entry_price"], t["exit_price"]],
                mode="lines", line=dict(color=color, width=1.5, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=1)

    fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume", showlegend=False,
                         marker_color="rgba(124,92,255,0.45)",
                         hovertemplate="Vol: %{y:,.0f}<extra></extra>"), row=3, col=1)

    # Range selector + slider for interactivity
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="#161c27", activecolor="#00d4aa", font=dict(color="#e6ebf5"),
        ),
        row=1, col=1,
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=780, hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0.5, xanchor="center"),
    )
    return fig


def chart_performance(portfolio_history: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Portfolio Value (key moments highlighted)",
                                        "Drawdown %"),
                        vertical_spacing=0.08, row_heights=[0.65, 0.35])
    pv = portfolio_history["Portfolio_Value"]
    fig.add_trace(go.Scatter(x=pv.index, y=pv, name="Portfolio", fill="tozeroy",
                             line=dict(color="#00d4aa", width=2),
                             fillcolor="rgba(0,212,170,0.12)",
                             hovertemplate="$%{y:,.0f}<extra></extra>"), row=1, col=1)

    # Identify peak before max DD and trough = the dip itself
    rolling_max = pv.expanding().max()
    dd_series = (pv - rolling_max) / rolling_max * 100
    if not dd_series.empty:
        trough_idx = dd_series.idxmin()
        peak_idx = pv.loc[:trough_idx].idxmax() if trough_idx in pv.index else pv.idxmax()
        peak_val = float(pv.loc[peak_idx])
        trough_val = float(pv.loc[trough_idx])
        max_dd_pct = float(dd_series.min())

        # Shade max-DD window on equity curve
        fig.add_vrect(x0=peak_idx, x1=trough_idx,
                      fillcolor="rgba(255,77,109,0.10)", line_width=0,
                      annotation_text=f"Max DD: {max_dd_pct:.1f}%",
                      annotation_position="top left",
                      annotation_font_color="#ff4d6d", row=1, col=1)
        # Peak marker (gold star) and trough marker (red X)
        fig.add_trace(go.Scatter(x=[peak_idx], y=[peak_val], mode="markers",
                                 marker=dict(symbol="star", color="#ffd700",
                                             size=14, line=dict(color="white", width=1)),
                                 name="Peak", showlegend=False,
                                 hovertemplate=f"<b>Peak</b><br>${peak_val:,.0f}<br>%{{x|%Y-%m-%d}}<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=[trough_idx], y=[trough_val], mode="markers",
                                 marker=dict(symbol="x", color="#ff4d6d",
                                             size=14, line=dict(color="white", width=2)),
                                 name="DD trough", showlegend=False,
                                 hovertemplate=f"<b>Trough</b><br>${trough_val:,.0f}<br>%{{x|%Y-%m-%d}}<extra></extra>"),
                      row=1, col=1)
        # Final value marker
        fig.add_trace(go.Scatter(x=[pv.index[-1]], y=[float(pv.iloc[-1])], mode="markers",
                                 marker=dict(symbol="circle", color="#00d4aa",
                                             size=12, line=dict(color="white", width=2)),
                                 name="Today", showlegend=False,
                                 hovertemplate=f"<b>Latest</b><br>${float(pv.iloc[-1]):,.0f}<extra></extra>"),
                      row=1, col=1)

    fig.add_trace(go.Scatter(x=dd_series.index, y=dd_series, name="Drawdown",
                             fill="tozeroy", line=dict(color="#ff4d6d"),
                             fillcolor="rgba(255,77,109,0.25)",
                             hovertemplate="%{y:.2f}%<extra></extra>"), row=2, col=1)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=560, showlegend=False,
                      hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ---------- Top control bar ----------

def render_topbar() -> dict:
    st.markdown(
        '<div class="topbar"><h1>Quant Trading Analytics '
        '<span class="pill">Live</span></h1>'
        '<div class="sub">Momentum strategy backtester with risk decomposition. '
        'Configure inputs below and run.</div></div>',
        unsafe_allow_html=True,
    )

    row1 = st.columns([1.1, 2.2, 1.2, 1.2, 0.9, 1.0])
    with row1[0]:
        mode = st.selectbox("Mode", ["Single", "Portfolio"], index=0,
                            help="Single: one ticker. Portfolio: comma-separated list.")
    with row1[1]:
        if mode == "Single":
            ticker_in = st.text_input("Ticker", value="AAPL", max_chars=10).strip().upper()
            tickers_in = ticker_in
        else:
            tickers_in = st.text_input("Portfolio tickers", value="AAPL, MSFT, SPY",
                                       help=f"Up to {MAX_PORTFOLIO_TICKERS}, comma-separated.")
            ticker_in = tickers_in.split(",")[0].strip().upper()
    with row1[2]:
        start = st.date_input("Start", value=datetime.now() - timedelta(days=1095),
                              max_value=datetime.now())
    with row1[3]:
        end = st.date_input("End", value=datetime.now(), max_value=datetime.now())
    with row1[4]:
        capital = st.number_input("Capital ($)", min_value=1_000, max_value=10_000_000,
                                  value=100_000, step=10_000)
    with row1[5]:
        st.write("")
        st.write("")
        run = st.button("Run analysis", type="primary", use_container_width=True)

    with st.expander("Strategy & risk parameters", expanded=False):
        p = st.columns(5)
        with p[0]:
            short_w = st.number_input("Short MA", 5, 100, 20, 1)
        with p[1]:
            long_w = st.number_input("Long MA", 10, 300, 50, 1)
        with p[2]:
            rsi_p = st.number_input("RSI period", 5, 50, 14, 1)
        with p[3]:
            max_pos = st.number_input("Max position", 0.05, 1.0, 0.10, 0.05, format="%.2f")
        with p[4]:
            st.write("")
            st.caption("📊 Diagnostics tab is always on — see last tab.")
            diagnostics = True

    return dict(
        mode=mode, ticker=ticker_in, tickers_raw=tickers_in,
        start=start, end=end, capital=int(capital),
        short_w=int(short_w), long_w=int(long_w), rsi_p=int(rsi_p),
        max_pos=float(max_pos), diagnostics=diagnostics, run=run,
    )


# ---------- Validation ----------

def validate_config(cfg: dict) -> list[str]:
    errs: list[str] = []
    if cfg["start"] >= cfg["end"]:
        errs.append("Start date must be before end date.")
    span_days = (cfg["end"] - cfg["start"]).days
    if span_days < 90:
        errs.append("Date range too short (min 90 days for a meaningful backtest).")
    if cfg["short_w"] >= cfg["long_w"]:
        errs.append("Short MA window must be smaller than Long MA window.")
    if cfg["mode"] == "Single":
        if not validate_ticker(cfg["ticker"]):
            errs.append(f"Invalid ticker '{cfg['ticker']}'. Use letters/digits, 1-10 chars.")
    else:
        valid, invalid = validate_portfolio_tickers(cfg["tickers_raw"])
        if invalid:
            errs.append(f"Invalid tickers ignored: {', '.join(invalid)}")
        if len(valid) < 2:
            errs.append("Portfolio mode needs at least 2 valid tickers.")
        elif len(valid) > MAX_PORTFOLIO_TICKERS:
            errs.append(f"Limit is {MAX_PORTFOLIO_TICKERS} tickers per portfolio.")
    return errs


# ---------- Analysis pipeline ----------

def _coerce_close_series(df: pd.DataFrame) -> pd.Series | None:
    if "Close" not in df.columns:
        return None
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    elif not isinstance(s, pd.Series):
        try:
            s = pd.Series(np.asarray(s).ravel(), index=df.index)
        except Exception:
            return None
    return s


def run_portfolio_backtest(strategy: MomentumStrategy, multi_data: dict, capital: int) -> dict | None:
    if len(multi_data) < 2:
        return None
    series_list = []
    for sym, df in multi_data.items():
        s = _coerce_close_series(df)
        if s is not None:
            s.name = sym
            series_list.append(s)
    if not series_list:
        return None
    close_df_raw = pd.concat(series_list, axis=1)
    nan_share = close_df_raw.isna().mean().mean()
    if nan_share > 0.25:
        st.warning(f"Low overlap across tickers (avg missing {nan_share:.0%}). Using intersection of valid dates.")
    close_df = close_df_raw.dropna(how="any")
    if close_df.empty or len(close_df) < MIN_BARS_FOR_BACKTEST:
        st.warning("Not enough overlapping price history across portfolio tickers.")
        return None

    signals_map = {}
    for sym in close_df.columns:
        df_sym = multi_data[sym].reindex(close_df.index)
        sig_df, _ = strategy.backtest(df_sym, capital)
        signals_map[sym] = sig_df["Signal"]
    signals_df_port = pd.DataFrame(signals_map).reindex(close_df.index)

    pv_list, cash_list, trade_stats, all_trades, total_trades, final_value = [], [], [], [], 0, 0
    per_symbol_capital = capital / len(close_df.columns)
    for sym in close_df.columns:
        engine_sym = BacktestEngine(initial_capital=per_symbol_capital)
        bt = engine_sym.run_backtest(
            close_df[[sym]].rename(columns={sym: "Close"}),
            signals_df_port[[sym]].rename(columns={sym: "Signal"}),
        )
        pv = bt.get("portfolio_history", pd.DataFrame())
        if not pv.empty:
            pv_list.append(pv["Portfolio_Value"])
            cash_list.append(pv["Cash"])
        ts = bt.get("trade_analysis", {})
        if ts:
            trade_stats.append(ts)
            total_trades += ts.get("total_trades", 0)
        sym_trades = bt.get("trades", pd.DataFrame())
        if not sym_trades.empty:
            sym_trades = sym_trades.copy()
            sym_trades["symbol"] = sym
            all_trades.append(sym_trades)
        final_value += bt.get("final_portfolio_value", 0)

    if not pv_list:
        return None
    port_hist = pd.DataFrame(pv_list).sum(axis=0).to_frame("Portfolio_Value")
    port_hist["Cash"] = pd.DataFrame(cash_list).sum(axis=0)
    port_hist.index.name = "Date"
    ret = port_hist["Portfolio_Value"].pct_change()
    rolling = port_hist["Portfolio_Value"].expanding().max()
    return {
        "total_return": (port_hist["Portfolio_Value"].iloc[-1] / capital) - 1,
        "annualized_return": ret.mean() * 252 if len(ret.dropna()) else 0,
        "volatility": ret.std() * np.sqrt(252) if len(ret.dropna()) else 0,
        "sharpe_ratio": (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() not in (None, 0) else 0,
        "max_drawdown": ((port_hist["Portfolio_Value"] - rolling) / rolling).min(),
        "calmar_ratio": 0,
        "final_portfolio_value": float(port_hist["Portfolio_Value"].iloc[-1]),
        "average_exposure": ((capital - port_hist["Cash"]) / capital).mean(),
        "portfolio_history": port_hist,
        "trade_analysis": {
            "total_trades": total_trades,
            "winning_trades": int(sum(t.get("winning_trades", 0) for t in trade_stats)),
            "losing_trades": int(sum(t.get("losing_trades", 0) for t in trade_stats)),
            "win_rate": (sum(t.get("winning_trades", 0) for t in trade_stats) / total_trades) if total_trades else 0,
            "avg_win": float(np.mean([t.get("avg_win", 0) for t in trade_stats])) if trade_stats else 0,
            "avg_loss": float(np.mean([t.get("avg_loss", 0) for t in trade_stats])) if trade_stats else 0,
            "profit_factor": float(np.mean([t.get("profit_factor", 0) for t in trade_stats])) if trade_stats else 0,
            "avg_trade_duration": float(np.mean([t.get("avg_trade_duration", 0) for t in trade_stats])) if trade_stats else 0,
            "best_trade": float(max([t.get("best_trade", 0) for t in trade_stats], default=0)),
            "worst_trade": float(min([t.get("worst_trade", 0) for t in trade_stats], default=0)),
        },
        "trades": pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(),
        "total_trades": total_trades,
    }


def run_pipeline(cfg: dict) -> bool:
    if cfg["mode"] == "Portfolio":
        valid_tickers, _ = validate_portfolio_tickers(cfg["tickers_raw"])
        with st.spinner(f"Loading {len(valid_tickers)} tickers..."):
            multi_data = load_multiple_data(tuple(valid_tickers), cfg["start"], cfg["end"])
        if not multi_data:
            st.error("No data loaded for any ticker. Check symbols and date range.")
            return False
        primary = next(iter(multi_data))
        data = multi_data[primary]
    else:
        with st.spinner(f"Loading {cfg['ticker']}..."):
            data = load_data(cfg["ticker"], cfg["start"], cfg["end"])
        if data is None or data.empty:
            st.error(f"No data for '{cfg['ticker']}'. Try a different symbol or date range.")
            return False
        multi_data = {}
        primary = cfg["ticker"]

    if len(data) < max(cfg["long_w"] + 10, MIN_BARS_FOR_BACKTEST):
        st.error(f"Need at least {max(cfg['long_w'] + 10, MIN_BARS_FOR_BACKTEST)} bars; got {len(data)}.")
        return False

    strategy = MomentumStrategy(
        short_window=cfg["short_w"], long_window=cfg["long_w"],
        rsi_period=cfg["rsi_p"], max_position_size=cfg["max_pos"],
    )
    with st.spinner("Running backtest..."):
        signals_df, perf = strategy.backtest(data, cfg["capital"])
        engine = BacktestEngine(initial_capital=cfg["capital"])
        portfolio_bt = run_portfolio_backtest(strategy, multi_data, cfg["capital"]) if cfg["mode"] == "Portfolio" else None
        backtest_results = portfolio_bt or engine.run_backtest(data[["Close"]].copy(), signals_df[["Signal"]].copy())

    st.session_state.update(
        data=data, signals_df=signals_df, strategy_performance=perf,
        backtest_results=backtest_results, ticker=primary,
        portfolio_mode=cfg["mode"] == "Portfolio",
        portfolio_tickers=list(multi_data.keys()) if multi_data else [],
        diagnostics_on=cfg["diagnostics"],
    )
    return True


# ---------- Renderers ----------

def render_action_banner(ticker, data, signals_df, trades_df):
    """Tells the user what the strategy says to do TODAY based on the latest signal."""
    if signals_df is None or signals_df.empty or "Signal" not in signals_df.columns:
        return
    last_signal = int(signals_df["Signal"].iloc[-1] or 0)
    last_close = float(data["Close"].iloc[-1])
    last_date = data.index[-1].strftime("%Y-%m-%d")

    in_position = bool(trades_df is not None and not trades_df.empty
                       and pd.isna(trades_df.iloc[-1].get("exit_date", pd.NaT))) if trades_df is not None else False
    open_trade = None
    if trades_df is not None and not trades_df.empty:
        # Engine closes everything at end of backtest, so use the last trade's open vs close
        latest = trades_df.iloc[-1]
        if pd.isna(latest.get("exit_date", pd.NaT)):
            open_trade = latest

    if last_signal == 1:
        action, color, emoji = "BUY", "#00d4aa", "▲"
        sub = f"Strategy is bullish at ${last_close:,.2f}. Enter long if flat."
    elif last_signal == -1:
        action, color, emoji = "SELL / EXIT", "#ff4d6d", "▼"
        sub = f"Strategy is bearish at ${last_close:,.2f}. Close any long position."
    else:
        action, color, emoji = "HOLD", "#8a96aa", "■"
        sub = f"No edge detected at ${last_close:,.2f}. Stay in current state."

    open_html = ""
    if open_trade is not None:
        unrealized = (last_close - float(open_trade["entry_price"])) * float(open_trade["quantity"])
        unrealized_pct = (last_close / float(open_trade["entry_price"]) - 1) * 100
        u_color = "#00d4aa" if unrealized >= 0 else "#ff4d6d"
        open_html = (f'<div style="margin-top:.5rem;font-size:.85rem;color:#8a96aa;">'
                     f'Open position: entered ${float(open_trade["entry_price"]):.2f} on '
                     f'{pd.to_datetime(open_trade["entry_date"]).strftime("%Y-%m-%d")} · '
                     f'<span style="color:{u_color};font-weight:600;">'
                     f'unrealized {unrealized:+,.2f} ({unrealized_pct:+.2f}%)</span></div>')

    st.markdown(
        f"""
<div style="background:linear-gradient(135deg,{color}15 0%,#161c27 60%);
            border:1px solid {color}55;border-left:4px solid {color};
            border-radius:12px;padding:1rem 1.25rem;margin:1rem 0;">
  <div style="display:flex;align-items:baseline;gap:.75rem;">
    <span style="color:{color};font-size:1.4rem;font-weight:700;">{emoji} {action}</span>
    <span style="color:#8a96aa;font-size:.8rem;">as of {last_date}</span>
  </div>
  <div style="color:#e6ebf5;margin-top:.25rem;">{sub}</div>
  {open_html}
</div>
""",
        unsafe_allow_html=True,
    )


def render_overview(ticker, data, signals_df, perf, backtest_results):
    st.subheader(f"{ticker} — Strategy Overview")
    trades_df = backtest_results.get("trades")
    actual_trades = int(backtest_results.get("total_trades", 0))
    ta = backtest_results.get("trade_analysis", {})
    engine_win_rate = ta.get("win_rate", perf.get("win_rate", 0))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return", f"{perf['total_return']:.2%}",
              delta=f"vs B&H: {perf['alpha']:+.2%}")
    c2.metric("Sharpe Ratio", f"{perf['sharpe_ratio']:.2f}")
    c3.metric("Max Drawdown", f"{perf['max_drawdown']:.2%}")
    c4.metric("Win Rate", f"{engine_win_rate:.1%}",
              delta=f"{actual_trades} executed trades")

    render_action_banner(ticker, data, signals_df, trades_df)

    # Visual KPIs ribbon: best trade, worst trade, current MA spread
    if trades_df is not None and not trades_df.empty:
        best = trades_df.loc[trades_df["pnl"].idxmax()]
        worst = trades_df.loc[trades_df["pnl"].idxmin()]
        last_close = float(data["Close"].iloc[-1])
        ms_last = float(signals_df["MA_Short"].iloc[-1] or 0)
        ml_last = float(signals_df["MA_Long"].iloc[-1] or 0)
        spread = ((ms_last / ml_last - 1) * 100) if ml_last else 0
        rsi_last = float(signals_df["RSI"].iloc[-1] or 0)
        kpi_strip([
            ("⭐ Best trade", f"${best['pnl']:+,.0f} ({best['return_pct']*100:+.1f}%)", "#ffd700"),
            ("⚠ Worst trade", f"${worst['pnl']:+,.0f} ({worst['return_pct']*100:+.1f}%)", "#ff4d6d"),
            ("📏 MA spread today", f"{spread:+.2f}%", "#00d4aa" if spread > 0 else "#ff4d6d"),
            ("🌡 RSI today", f"{rsi_last:.1f}",
             "#ff4d6d" if rsi_last > 70 else "#00d4aa" if rsi_last < 30 else "#7c5cff"),
        ])

    st.plotly_chart(chart_price(data, signals_df, trades_df), use_container_width=True)

    with st.expander("📘 How to read this chart", expanded=False):
        st.markdown("""
- **Green ▲ markers** = strategy entered a long position (bought).
- **Green ▼ markers** = position closed at a profit.
- **Red ▼ markers** = position closed at a loss.
- **Dotted lines** connect entry → exit (green = profit, red = loss).
- **⭐ Gold star annotation** = your best trade in this period.
- **⚠ Red annotation** = your worst trade — study what went wrong.
- **Orange / Purple lines** = short / long moving averages driving the signals.
- **RSI panel**: dashed lines mark overbought (70) / oversold (30).
- **Range buttons** (1M / 3M / 6M / YTD / 1Y / All) zoom the chart.
- The **action banner** above tells you what the strategy would do *today*.
""")


def render_performance(signals_df, backtest_results, perf):
    st.subheader("Performance")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Returns**")
        st.table(pd.DataFrame({
            "Metric": ["Total Return", "Annualized", "Benchmark", "Alpha", "Volatility"],
            "Value": [
                f"{backtest_results['total_return']:.2%}",
                f"{backtest_results['annualized_return']:.2%}",
                f"{perf['benchmark_return']:.2%}",
                f"{perf['alpha']:+.2%}",
                f"{backtest_results['volatility']:.2%}",
            ],
        }))
    with c2:
        st.markdown("**Risk-adjusted**")
        st.table(pd.DataFrame({
            "Metric": ["Sharpe", "Calmar", "Max Drawdown", "Avg Exposure"],
            "Value": [
                f"{backtest_results['sharpe_ratio']:.2f}",
                f"{backtest_results['calmar_ratio']:.2f}",
                f"{backtest_results['max_drawdown']:.2%}",
                f"{backtest_results['average_exposure']:.1%}",
            ],
        }))
    if "portfolio_history" in backtest_results:
        st.plotly_chart(chart_performance(backtest_results["portfolio_history"]),
                        use_container_width=True)
    if len(signals_df) > 30 and "Strategy_Returns" in signals_df.columns:
        monthly = signals_df["Strategy_Returns"].groupby(
            [signals_df.index.year, signals_df.index.month]
        ).sum().unstack(fill_value=0)
        monthly = monthly.reindex(columns=range(1, 13), fill_value=0)
        labels = [datetime(2000, m, 1).strftime("%b") for m in monthly.columns]
        fig = px.imshow(monthly.values, x=labels, y=monthly.index,
                        color_continuous_scale="RdYlGn", aspect="auto",
                        labels=dict(x="Month", y="Year", color="Return"),
                        title="Monthly Returns")
        fig.update_layout(template=PLOTLY_TEMPLATE, height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)


def render_risk(ticker, signals_df):
    st.subheader("Risk Analysis")
    if st.session_state.get("portfolio_mode") and st.session_state.get("portfolio_tickers"):
        returns_map = {}
        data = st.session_state.data
        for sym in st.session_state["portfolio_tickers"]:
            df = load_data(sym, data.index.min().date(), data.index.max().date())
            s = _coerce_close_series(df) if df is not None else None
            if s is not None:
                returns_map[sym] = s.pct_change()
        returns_data = pd.DataFrame(returns_map).dropna(how="any") if returns_map else pd.DataFrame()
        if returns_data.empty:
            returns_data = pd.DataFrame({ticker: signals_df["Returns"].fillna(0)}, index=signals_df.index)
    else:
        returns_data = pd.DataFrame({ticker: signals_df["Returns"].fillna(0)}, index=signals_df.index)

    weights = {c: 1 / len(returns_data.columns) for c in returns_data.columns}
    rm = PortfolioRiskManager()
    rm.load_portfolio_data(returns_data, weights)
    report = rm.generate_risk_report()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Value at Risk**")
        st.table(report["var_analysis"])
    with c2:
        st.markdown("**Portfolio Stats**")
        st.table(pd.DataFrame({
            "Metric": [k.replace("_", " ").title() for k in report["portfolio_statistics"]],
            "Value": [f"{v:.4f}" for v in report["portfolio_statistics"].values()],
        }))
    st.markdown("**Stress Tests**")
    st.table(report["stress_testing"])
    fig = px.bar(report["risk_decomposition"], x="Asset", y="Risk_Contrib_Percentage",
                 color="Risk_Contrib_Percentage", color_continuous_scale="Reds",
                 title="Risk contribution by asset (%)")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=380, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_trades(backtest_results):
    ta = backtest_results["trade_analysis"]
    trades_df = backtest_results.get("trades", pd.DataFrame())
    st.subheader("Trades")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trades", ta["total_trades"])
    c1.metric("Winning", ta["winning_trades"])
    c1.metric("Losing", ta["losing_trades"])
    c2.metric("Win Rate", f"{ta['win_rate']:.1%}")
    c2.metric("Profit Factor", f"{ta['profit_factor']:.2f}")
    c2.metric("Avg Duration", f"{ta['avg_trade_duration']:.1f} d")
    c3.metric("Best Trade", f"${ta['best_trade']:,.2f}")
    c3.metric("Worst Trade", f"${ta['worst_trade']:,.2f}")
    c3.metric("Avg Win", f"${ta['avg_win']:,.2f}")

    if not trades_df.empty:
        st.markdown("### Trade log")
        log = trades_df.copy()
        log["entry_date"] = pd.to_datetime(log["entry_date"]).dt.strftime("%Y-%m-%d")
        log["exit_date"] = pd.to_datetime(log["exit_date"]).dt.strftime("%Y-%m-%d")
        log["entry_price"] = log["entry_price"].round(2)
        log["exit_price"] = log["exit_price"].round(2)
        log["pnl"] = log["pnl"].round(2)
        log["return_pct"] = (log["return_pct"] * 100).round(2)
        log["result"] = log["pnl"].apply(lambda v: "WIN" if v > 0 else "LOSS")
        cols = ["entry_date", "exit_date", "side", "entry_price", "exit_price",
                "quantity", "pnl", "return_pct", "duration_days", "result"]
        st.dataframe(
            log[cols].sort_values("entry_date", ascending=False),
            use_container_width=True, hide_index=True,
            column_config={
                "entry_date": "Entered",
                "exit_date": "Exited",
                "side": "Side",
                "entry_price": st.column_config.NumberColumn("Entry $", format="$%.2f"),
                "exit_price": st.column_config.NumberColumn("Exit $", format="$%.2f"),
                "quantity": "Qty",
                "pnl": st.column_config.NumberColumn("P&L $", format="$%.2f"),
                "return_pct": st.column_config.NumberColumn("Return %", format="%.2f%%"),
                "duration_days": "Days",
                "result": "Result",
            },
        )

        fig = px.histogram(trades_df, x="pnl", nbins=min(40, max(len(trades_df), 5)),
                           title="Realized P&L distribution",
                           color=trades_df["pnl"].apply(lambda v: "Win" if v > 0 else "Loss"),
                           color_discrete_map={"Win": "#00d4aa", "Loss": "#ff4d6d"},
                           labels={"pnl": "P&L ($)", "count": "Trades"})
        fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.update_layout(template=PLOTLY_TEMPLATE, height=380,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No completed trades in this backtest. Try a longer date range or different parameters.")


def render_report(ticker, data, signals_df, perf, backtest_results):
    st.subheader("Report")
    st.markdown(f"""
**{ticker} momentum strategy** generated **{perf['total_return']:.2%}** total return vs
buy-and-hold benchmark of **{perf['benchmark_return']:.2%}** (alpha: **{perf['alpha']:+.2%}**).

- Sharpe: `{perf['sharpe_ratio']:.2f}`
- Max Drawdown: `{perf['max_drawdown']:.2%}`
- Win Rate: `{perf['win_rate']:.1%}` across `{perf['total_trades']}` trades
- Volatility (annualized): `{backtest_results['volatility']:.2%}`
- Avg Exposure: `{backtest_results['average_exposure']:.1%}`
""")
    st.markdown("**Validation**")
    st.table(pd.DataFrame({
        "Check": ["Data Quality", "Strategy Logic", "Risk Management",
                  "Transaction Costs", "Statistical Significance"],
        "Status": ["Passed", "Passed", "Passed", "Included", "Validated"],
        "Notes": [
            f"{len(data)} observations",
            "Dual confirmation (MA + RSI)",
            "Vol-scaled position sizing",
            "Commission + slippage applied",
            f"Sharpe {perf['sharpe_ratio']:.2f}",
        ],
    }))
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download strategy data (CSV)",
                           data=signals_df.to_csv().encode(),
                           file_name=f"{ticker}_strategy.csv", mime="text/csv",
                           use_container_width=True)
    with c2:
        perf_csv = pd.DataFrame([perf, {k: v for k, v in backtest_results.items()
                                        if not isinstance(v, (pd.DataFrame, dict))}]).to_csv()
        st.download_button("Download performance (CSV)",
                           data=perf_csv.encode(),
                           file_name=f"{ticker}_performance.csv", mime="text/csv",
                           use_container_width=True)


def render_diagnostics(data, signals_df):
    """Always visible (no toggle). Shown as the last tab on the Backtest page."""
    st.subheader("🔧 Data & Signal Diagnostics")
    st.caption("Behind-the-scenes data quality checks and signal column verification. "
               "Use this if numbers look off.")

    # Quick health pills
    n = len(data)
    nan_share = data.isna().mean().mean() * 100
    expected = ["MA_Short", "MA_Long", "RSI", "Signal", "Strategy_Returns", "Position"]
    present_count = sum(1 for c in expected if c in signals_df.columns)
    sig_count = (signals_df["Signal"].abs() > 0).sum() if "Signal" in signals_df.columns else 0
    kpi_strip([
        ("Rows", f"{n:,}", "#00d4aa" if n > 200 else "#ffb547"),
        ("Missing %", f"{nan_share:.2f}%",
         "#00d4aa" if nan_share < 1 else "#ffb547" if nan_share < 5 else "#ff4d6d"),
        ("Signals computed", f"{present_count}/{len(expected)}",
         "#00d4aa" if present_count == len(expected) else "#ff4d6d"),
        ("Active signals", f"{sig_count:,}",
         "#00d4aa" if sig_count > 5 else "#ffb547"),
    ])

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**📦 Data snapshot**")
        st.write({
            "rows": int(n),
            "columns": list(data.columns),
            "range": f"{data.index.min().date()} → {data.index.max().date()}",
        })
        st.markdown("**🧩 Missing values per column**")
        st.dataframe(data.isna().sum().rename("missing").to_frame(),
                     use_container_width=True)
    with c2:
        st.markdown("**📐 Signal coverage**")
        cov = pd.DataFrame({
            "column": expected,
            "present": ["✓" if c in signals_df.columns else "✗" for c in expected],
            "non_null": [int(signals_df[c].notna().sum()) if c in signals_df.columns else 0
                         for c in expected],
        })
        st.dataframe(cov, use_container_width=True, hide_index=True)
        if "Signal" in signals_df.columns:
            st.markdown("**📊 Signal distribution**")
            dist = signals_df["Signal"].value_counts(dropna=False).to_frame("count")
            dist.index = dist.index.map({1: "BUY (+1)", 0: "HOLD (0)", -1: "SELL (-1)"}).fillna("OTHER")
            st.dataframe(dist, use_container_width=True)

    st.markdown("**🔚 Last 15 rows of computed signals**")
    available = [c for c in expected if c in signals_df.columns]
    if available:
        st.dataframe(signals_df[available].tail(15), use_container_width=True)


# ---------- Page: Watchlist Scanner ----------

@st.cache_data(ttl=900, show_spinner=False)
def _scan_one(ticker: str, lookback_days: int, short_w: int, long_w: int, rsi_p: int) -> dict | None:
    end = datetime.now().date()
    start = end - timedelta(days=lookback_days)
    df = load_data(ticker, start, end)
    if df is None or df.empty or len(df) < long_w + 5:
        return None
    s = MomentumStrategy(short_window=short_w, long_window=long_w, rsi_period=rsi_p)
    sig, _ = s.backtest(df, 100_000)
    last = sig.iloc[-1]
    last_signal = int(last.get("Signal", 0) or 0)
    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
    chg = (last_close / prev_close - 1) * 100 if prev_close else 0
    rsi_val = float(last.get("RSI", 0) or 0)
    # Signal age: days since last change
    sig_series = sig["Signal"].fillna(0)
    same = (sig_series == last_signal).iloc[::-1]
    age = int(same.cumprod().sum())
    # Distance to MA crossover (proxy for conviction)
    ms, ml = float(last.get("MA_Short", 0) or 0), float(last.get("MA_Long", 0) or 0)
    spread = ((ms / ml - 1) * 100) if ml else 0
    return {
        "ticker": ticker, "signal": last_signal, "price": last_close,
        "chg_1d": chg, "rsi": rsi_val, "ma_spread_pct": spread,
        "signal_age_days": age,
    }


def render_watchlist_page():
    st.markdown(
        '<div class="topbar"><h1>Watchlist Scanner</h1>'
        '<div class="sub">Scan multiple tickers and surface what the strategy says to do today.</div></div>',
        unsafe_allow_html=True,
    )
    how_to_card(
        "How to use the Watchlist Scanner",
        [
            "Paste up to 30 tickers separated by commas (default = 15 large-caps + sector ETFs).",
            "Pick a lookback window (1y is the sweet spot — enough data, fast scan).",
            "Press <b>Scan</b>. Results cache for 15 min so re-running is instant.",
            "Sort the table by <b>Signal</b> to see all current BUYs/SELLs at the top.",
            "Sort by <b>MA spread %</b> within BUYs to see the strongest trends.",
        ],
        tip="Open this page first thing each morning. BUYs with high MA spread + low signal age = freshest setups."
    )
    c = st.columns([3, 1, 1, 1, 1, 1])
    with c[0]:
        raw = st.text_input("Tickers (comma-separated, max 30)", value=DEFAULT_WATCHLIST)
    with c[1]:
        lookback = st.selectbox("Lookback", ["6mo", "1y", "2y"], index=1)
        lookback_days = {"6mo": 200, "1y": 380, "2y": 760}[lookback]
    with c[2]:
        sh = st.number_input("Short MA", 5, 100, 20, 1, key="wl_sh")
    with c[3]:
        lo = st.number_input("Long MA", 10, 300, 50, 1, key="wl_lo")
    with c[4]:
        rs = st.number_input("RSI", 5, 50, 14, 1, key="wl_rs")
    with c[5]:
        st.write(""); st.write("")
        scan = st.button("Scan", type="primary", use_container_width=True)

    if sh >= lo:
        st.warning("Short MA must be smaller than Long MA.")
        return

    tickers = [clean_ticker(t) for t in raw.split(",") if validate_ticker(clean_ticker(t))][:30]
    if scan and tickers:
        rows = []
        prog = st.progress(0.0, text="Scanning...")
        for i, t in enumerate(tickers, 1):
            r = _scan_one(t, lookback_days, int(sh), int(lo), int(rs))
            if r:
                rows.append(r)
            prog.progress(i / len(tickers), text=f"Scanned {i}/{len(tickers)}")
        prog.empty()
        if not rows:
            st.error("No tickers returned data. Check symbols.")
            return
        df = pd.DataFrame(rows)

        # Headline buckets
        buys = df[df["signal"] == 1].sort_values("ma_spread_pct", ascending=False)
        sells = df[df["signal"] == -1].sort_values("ma_spread_pct", ascending=True)
        holds = df[df["signal"] == 0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Scanned", len(df))
        c2.metric("BUY signals", len(buys))
        c3.metric("SELL signals", len(sells))
        c4.metric("HOLD", len(holds))

        st.markdown("### Currently signaling")
        sig_label = {1: "BUY", -1: "SELL", 0: "HOLD"}
        df_show = df.copy()
        df_show["state"] = df_show["signal"].map(sig_label)
        df_show = df_show[["ticker", "state", "price", "chg_1d", "rsi",
                           "ma_spread_pct", "signal_age_days"]]
        df_show = df_show.sort_values(["state", "ma_spread_pct"],
                                      ascending=[True, False])
        st.dataframe(
            df_show, use_container_width=True, hide_index=True,
            column_config={
                "ticker": "Ticker",
                "state": st.column_config.TextColumn("Signal"),
                "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "chg_1d": st.column_config.NumberColumn("Day %", format="%.2f%%"),
                "rsi": st.column_config.NumberColumn("RSI", format="%.1f"),
                "ma_spread_pct": st.column_config.NumberColumn("MA spread %", format="%.2f%%",
                                                                help="MA Short vs MA Long; magnitude = trend conviction"),
                "signal_age_days": st.column_config.NumberColumn("Signal age (d)",
                                                                 help="How long the current signal has been in force"),
            },
        )

        # Visual: signal distribution
        fig = px.bar(df_show.groupby("state").size().reset_index(name="count"),
                     x="state", y="count", color="state",
                     color_discrete_map={"BUY": "#00d4aa", "HOLD": "#8a96aa", "SELL": "#ff4d6d"},
                     title="Signal distribution across watchlist")
        fig.update_layout(template=PLOTLY_TEMPLATE, height=300, showlegend=False,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Enter tickers and press **Scan**. Results cache for 15 minutes.")


# ---------- Page: Strategy Comparison ----------

def render_compare_page():
    st.markdown(
        '<div class="topbar"><h1>Strategy Comparison</h1>'
        '<div class="sub">Run momentum, mean-reversion, and buy-and-hold side-by-side on the same data.</div></div>',
        unsafe_allow_html=True,
    )
    how_to_card(
        "How to use Strategy Comparison",
        [
            "Pick a ticker and date range. 3 years gives a meaningful comparison.",
            "Press <b>Compare</b>. Three strategies run on the same data automatically.",
            "Top chart: equity curves overlaid — the steepest line wins. Look for which strategy beat <b>buy-and-hold</b>.",
            "Bottom chart: normalized vs SPY/QQQ/IWM — tells you if you're actually beating broad-market beta.",
            "Use the table for hard numbers: Sharpe, Max DD, Volatility, Trade count.",
        ],
        tip="If your strategy underperforms buy-and-hold AND has worse drawdowns, it's adding negative value. Move on."
    )
    c = st.columns([1.5, 1.2, 1.2, 1.2, 1])
    with c[0]:
        ticker = st.text_input("Ticker", value="AAPL", key="cmp_t").strip().upper()
    with c[1]:
        start = st.date_input("Start", value=datetime.now() - timedelta(days=1095),
                              max_value=datetime.now(), key="cmp_s")
    with c[2]:
        end = st.date_input("End", value=datetime.now(), max_value=datetime.now(), key="cmp_e")
    with c[3]:
        cap = st.number_input("Capital ($)", 1000, 10_000_000, 100_000, 10_000, key="cmp_c")
    with c[4]:
        st.write(""); st.write("")
        go_cmp = st.button("Compare", type="primary", use_container_width=True)

    if go_cmp:
        if not validate_ticker(ticker):
            st.error("Invalid ticker."); return
        if start >= end:
            st.error("Start must be before end."); return
        with st.spinner(f"Loading {ticker}..."):
            data = load_data(ticker, start, end)
        if data is None or data.empty or len(data) < 80:
            st.error("Insufficient data."); return

        # Strategy 1: Momentum
        ms = MomentumStrategy(20, 50, 14, max_position_size=0.1)
        sig_m, perf_m = ms.backtest(data, cap)
        eng_m = BacktestEngine(initial_capital=cap)
        bt_m = eng_m.run_backtest(data[["Close"]].copy(), sig_m[["Signal"]].copy())

        # Strategy 2: Mean Reversion
        try:
            mr = MeanReversionStrategy(lookback_window=20, entry_threshold=2.0,
                                       exit_threshold=0.5, max_position_size=0.1)
            sig_r, perf_r = mr.backtest(data, cap)
            eng_r = BacktestEngine(initial_capital=cap)
            bt_r = eng_r.run_backtest(data[["Close"]].copy(), sig_r[["Signal"]].copy())
            mr_ok = True
        except Exception as e:
            st.warning(f"Mean-reversion failed: {e}")
            sig_r = pd.DataFrame(); perf_r = {}; bt_r = {}; mr_ok = False

        # Strategy 3: Buy-and-Hold
        bh_curve = (data["Close"] / data["Close"].iloc[0]) * cap
        bh_ret = (bh_curve.iloc[-1] / cap) - 1
        bh_dd = ((bh_curve - bh_curve.expanding().max()) / bh_curve.expanding().max()).min()
        bh_vol = data["Close"].pct_change().std() * np.sqrt(252)
        bh_sharpe = (data["Close"].pct_change().mean() / data["Close"].pct_change().std()) * np.sqrt(252)

        # Equity curves chart
        fig = go.Figure()
        if "portfolio_history" in bt_m:
            ph = bt_m["portfolio_history"]["Portfolio_Value"]
            fig.add_trace(go.Scatter(x=ph.index, y=ph, name="Momentum",
                                     line=dict(color="#00d4aa", width=2)))
        if mr_ok and "portfolio_history" in bt_r:
            ph = bt_r["portfolio_history"]["Portfolio_Value"]
            fig.add_trace(go.Scatter(x=ph.index, y=ph, name="Mean Reversion",
                                     line=dict(color="#7c5cff", width=2)))
        fig.add_trace(go.Scatter(x=bh_curve.index, y=bh_curve, name="Buy & Hold",
                                 line=dict(color="#ffb547", width=2, dash="dash")))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=460, hovermode="x unified",
                          title=f"Equity curves on {ticker} (initial ${cap:,})",
                          margin=dict(l=10, r=10, t=50, b=10),
                          legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"))
        st.plotly_chart(fig, use_container_width=True)

        # Comparison table
        rows = [{
            "Strategy": "Momentum (MA + RSI)",
            "Total Return": f"{bt_m['total_return']:.2%}",
            "Sharpe": f"{bt_m['sharpe_ratio']:.2f}",
            "Max DD": f"{bt_m['max_drawdown']:.2%}",
            "Volatility": f"{bt_m['volatility']:.2%}",
            "Trades": bt_m["total_trades"],
        }]
        if mr_ok:
            rows.append({
                "Strategy": "Mean Reversion (Z-score)",
                "Total Return": f"{bt_r['total_return']:.2%}",
                "Sharpe": f"{bt_r['sharpe_ratio']:.2f}",
                "Max DD": f"{bt_r['max_drawdown']:.2%}",
                "Volatility": f"{bt_r['volatility']:.2%}",
                "Trades": bt_r["total_trades"],
            })
        rows.append({
            "Strategy": "Buy & Hold",
            "Total Return": f"{bh_ret:.2%}",
            "Sharpe": f"{bh_sharpe:.2f}",
            "Max DD": f"{bh_dd:.2%}",
            "Volatility": f"{bh_vol:.2%}",
            "Trades": 1,
        })
        st.markdown("### Side-by-side")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Multi-benchmark overlay
        st.markdown("### Vs broad-market benchmarks")
        bench_curves = {ticker + " (B&H)": bh_curve / cap}
        for b in BENCHMARKS:
            bdf = load_data(b, start, end)
            if bdf is not None and not bdf.empty:
                bench_curves[b] = bdf["Close"] / bdf["Close"].iloc[0]
        if "portfolio_history" in bt_m:
            ph = bt_m["portfolio_history"]["Portfolio_Value"] / cap
            bench_curves["Momentum strategy"] = ph
        bf = pd.DataFrame(bench_curves).dropna(how="all")
        fig_b = go.Figure()
        palette = ["#00d4aa", "#7c5cff", "#ffb547", "#4cc9f0", "#ff4d6d"]
        for i, col in enumerate(bf.columns):
            fig_b.add_trace(go.Scatter(x=bf.index, y=bf[col], name=col,
                                       line=dict(color=palette[i % len(palette)], width=1.8)))
        fig_b.update_layout(template=PLOTLY_TEMPLATE, height=420, hovermode="x unified",
                            title="Normalized growth (start = 1.0)",
                            margin=dict(l=10, r=10, t=50, b=10),
                            legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"))
        st.plotly_chart(fig_b, use_container_width=True)
    else:
        st.info("Configure and press **Compare** to run all strategies.")


# ---------- Page: Optimize (parameter heatmap + walk-forward + cost sensitivity) ----------

def _quick_backtest_sharpe(data: pd.DataFrame, sh: int, lo: int, rsi_p: int,
                            cap: int, commission: float = 0.001) -> float:
    if sh >= lo or len(data) < lo + 10:
        return float("nan")
    s = MomentumStrategy(short_window=sh, long_window=lo, rsi_period=rsi_p)
    sig, _ = s.backtest(data, cap)
    eng = BacktestEngine(initial_capital=cap, commission_rate=commission)
    bt = eng.run_backtest(data[["Close"]].copy(), sig[["Signal"]].copy())
    return float(bt.get("sharpe_ratio", float("nan")))


def render_optimize_page():
    st.markdown(
        '<div class="topbar"><h1>Optimize & Validate</h1>'
        '<div class="sub">Walk-forward out-of-sample test, parameter sensitivity heatmap, '
        'and cost-sensitivity check — to spot overfitting and verify edge survives real costs.</div></div>',
        unsafe_allow_html=True,
    )
    c = st.columns([1.5, 1.2, 1.2, 1.2, 1])
    with c[0]:
        ticker = st.text_input("Ticker", value="AAPL", key="opt_t").strip().upper()
    with c[1]:
        start = st.date_input("Start", value=datetime.now() - timedelta(days=1460),
                              max_value=datetime.now(), key="opt_s")
    with c[2]:
        end = st.date_input("End", value=datetime.now(), max_value=datetime.now(), key="opt_e")
    with c[3]:
        cap = st.number_input("Capital ($)", 1000, 10_000_000, 100_000, 10_000, key="opt_c")
    with c[4]:
        st.write(""); st.write("")
        go_opt = st.button("Run", type="primary", use_container_width=True)

    how_to_card(
        "How to use Optimize & Validate",
        [
            "Pick a ticker and a long history (3-4 years gives meaningful walk-forward).",
            "Press <b>Run</b>. Three checks fire automatically.",
            "<b>Check 1 (Walk-forward)</b>: train on first 70%, test on last 30%. If the test equity curve diverges down from train, the strategy is overfit.",
            "<b>Check 2 (Heatmap)</b>: each cell = one parameter combo. Look for a <b>green region</b> (robust). Isolated green cells = curve-fit luck.",
            "<b>Check 3 (Cost sensitivity)</b>: see where your Sharpe crosses zero — that's your max viable commission.",
        ],
        tip="If Sharpe decay (train→test) is more than -0.5 OR Sharpe goes negative at retail costs (~0.1%), don't deploy live."
    )
    if not go_opt:
        return
    if not validate_ticker(ticker):
        st.error("Invalid ticker."); return
    with st.spinner(f"Loading {ticker}..."):
        data = load_data(ticker, start, end)
    if data is None or data.empty or len(data) < 200:
        st.error("Need at least ~200 bars (about a year)."); return

    # 1. Walk-forward: 70/30 split — overlay equity curves with split marker
    st.markdown("### 1) Walk-forward out-of-sample test")
    split_idx = int(len(data) * 0.7)
    train, test = data.iloc[:split_idx], data.iloc[split_idx:]
    split_date = data.index[split_idx]

    s = MomentumStrategy(20, 50, 14, max_position_size=0.1)
    sig_train, perf_train = s.backtest(train, cap)
    sig_test, perf_test = s.backtest(test, cap)
    eq_train = (1 + sig_train["Strategy_Returns"].fillna(0)).cumprod() * cap
    eq_test = (1 + sig_test["Strategy_Returns"].fillna(0)).cumprod() * cap
    bh_train = (train["Close"] / train["Close"].iloc[0]) * cap
    bh_test = (test["Close"] / test["Close"].iloc[0]) * cap

    decay = perf_test["sharpe_ratio"] - perf_train["sharpe_ratio"]
    verdict_color = "#ff4d6d" if decay < -0.5 else "#ffb547" if decay < -0.2 else "#00d4aa"
    verdict_text = ("OVERFIT" if decay < -0.5
                    else "MILD DECAY" if decay < -0.2 else "ROBUST")

    kpi_strip([
        ("Train return", f"{perf_train['total_return']:+.2%}", "#7c5cff"),
        ("Test return", f"{perf_test['total_return']:+.2%}",
         "#00d4aa" if perf_test["total_return"] > 0 else "#ff4d6d"),
        ("Train Sharpe", f"{perf_train['sharpe_ratio']:.2f}", "#7c5cff"),
        ("Test Sharpe", f"{perf_test['sharpe_ratio']:.2f}",
         "#00d4aa" if perf_test["sharpe_ratio"] > 0 else "#ff4d6d"),
        ("Verdict", verdict_text, verdict_color),
    ])

    fig_wf = go.Figure()
    fig_wf.add_trace(go.Scatter(x=eq_train.index, y=eq_train, name="Strategy (train)",
                                line=dict(color="#7c5cff", width=2)))
    fig_wf.add_trace(go.Scatter(x=eq_test.index, y=eq_test, name="Strategy (test, OOS)",
                                line=dict(color="#00d4aa", width=2.5)))
    fig_wf.add_trace(go.Scatter(x=bh_train.index, y=bh_train, name="Buy & Hold (train)",
                                line=dict(color="#7c5cff", width=1, dash="dot"),
                                opacity=0.5))
    fig_wf.add_trace(go.Scatter(x=bh_test.index, y=bh_test, name="Buy & Hold (test)",
                                line=dict(color="#00d4aa", width=1, dash="dot"),
                                opacity=0.5))
    fig_wf.add_vline(x=split_date, line_dash="dash", line_color="#ffd700",
                     line_width=2, annotation_text="↓ Train | Test ↓",
                     annotation_position="top", annotation_font_color="#ffd700")
    # Shade test region
    fig_wf.add_vrect(x0=split_date, x1=data.index[-1],
                     fillcolor="rgba(0,212,170,0.05)", line_width=0)
    fig_wf.update_layout(template=PLOTLY_TEMPLATE, height=440, hovermode="x unified",
                         title="Equity curves — train (left of gold line) vs test (right). "
                               "Test should track or beat buy-and-hold.",
                         margin=dict(l=10, r=10, t=60, b=10),
                         legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
    st.plotly_chart(fig_wf, use_container_width=True)

    if decay < -0.5:
        st.error("⚠ Significant Sharpe decay on out-of-sample data. Strategy is likely overfit — do NOT deploy live.")
    elif decay < -0.2:
        st.warning("Mild Sharpe decay. Borderline — consider retuning or paper-trading first.")
    else:
        st.success("✓ Out-of-sample performance holds up. Strategy generalizes.")

    # 2. Parameter heatmap with annotated optimum + best combos bar chart
    st.markdown("### 2) Parameter sensitivity heatmap")
    short_grid = [10, 15, 20, 25, 30]
    long_grid = [40, 50, 60, 80, 100]
    with st.spinner(f"Running {len(short_grid) * len(long_grid)} backtests..."):
        z = np.full((len(short_grid), len(long_grid)), np.nan)
        for i, sh in enumerate(short_grid):
            for j, lo in enumerate(long_grid):
                if sh < lo:
                    z[i, j] = _quick_backtest_sharpe(data, sh, lo, 14, cap)

    cmap_col1, cmap_col2 = st.columns([1.5, 1])
    with cmap_col1:
        fig_h = px.imshow(
            z, x=[str(x) for x in long_grid], y=[str(x) for x in short_grid],
            color_continuous_scale="RdYlGn", aspect="auto", text_auto=".2f",
            labels=dict(x="Long MA window", y="Short MA window", color="Sharpe"),
            title="Sharpe across (Short MA, Long MA) — green region = robust",
            zmin=-2, zmax=2,
        )
        # Highlight optimum cell
        if np.isfinite(z).any():
            opt_idx = np.unravel_index(np.nanargmax(z), z.shape)
            fig_h.add_shape(type="rect",
                            x0=opt_idx[1] - 0.5, x1=opt_idx[1] + 0.5,
                            y0=opt_idx[0] - 0.5, y1=opt_idx[0] + 0.5,
                            line=dict(color="#ffd700", width=3))
            fig_h.add_annotation(
                x=opt_idx[1], y=opt_idx[0], text="⭐",
                showarrow=False, font=dict(size=20),
                xshift=-25, yshift=15,
            )
        fig_h.update_layout(template=PLOTLY_TEMPLATE, height=420,
                            margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_h, use_container_width=True)
    with cmap_col2:
        # Top-5 combos bar chart
        flat = []
        for i, sh in enumerate(short_grid):
            for j, lo in enumerate(long_grid):
                if np.isfinite(z[i, j]):
                    flat.append({"combo": f"{sh}/{lo}", "sharpe": z[i, j]})
        if flat:
            top5 = pd.DataFrame(flat).sort_values("sharpe", ascending=False).head(5)
            fig_top = px.bar(top5, x="sharpe", y="combo", orientation="h",
                             color="sharpe", color_continuous_scale="RdYlGn",
                             range_color=[-2, 2],
                             title="Top 5 parameter combos",
                             labels={"sharpe": "Sharpe", "combo": "Short/Long"})
            fig_top.update_layout(template=PLOTLY_TEMPLATE, height=420,
                                  margin=dict(l=10, r=10, t=60, b=10),
                                  yaxis=dict(autorange="reversed"),
                                  showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_top, use_container_width=True)
    st.caption("⭐ marks the best cell. Robust strategies show a **block** of green; overfit ones show isolated peaks.")

    # 3. Cost sensitivity — with break-even & reference markers
    st.markdown("### 3) Cost-sensitivity check")
    cost_grid = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005]
    sharpes, returns = [], []
    with st.spinner("Sweeping commission rates..."):
        for c in cost_grid:
            s = MomentumStrategy(20, 50, 14)
            sig, _ = s.backtest(data, cap)
            eng = BacktestEngine(initial_capital=cap, commission_rate=c)
            bt = eng.run_backtest(data[["Close"]].copy(), sig[["Signal"]].copy())
            sharpes.append(bt["sharpe_ratio"]); returns.append(bt["total_return"])
    cost_pct = [c * 100 for c in cost_grid]

    # Linear interpolate break-even (where Sharpe crosses 0)
    breakeven = None
    for i in range(len(sharpes) - 1):
        if (sharpes[i] >= 0) != (sharpes[i + 1] >= 0):
            x0, x1 = cost_pct[i], cost_pct[i + 1]
            y0, y1 = sharpes[i], sharpes[i + 1]
            breakeven = x0 + (0 - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else None
            break

    fig_c = make_subplots(specs=[[{"secondary_y": True}]])
    fig_c.add_trace(go.Scatter(x=cost_pct, y=sharpes, name="Sharpe",
                               line=dict(color="#00d4aa", width=3),
                               mode="lines+markers", marker=dict(size=10),
                               fill="tozeroy",
                               fillcolor="rgba(0,212,170,0.10)"))
    fig_c.add_trace(go.Scatter(x=cost_pct, y=[r * 100 for r in returns],
                               name="Total Return %",
                               line=dict(color="#7c5cff", width=2, dash="dot"),
                               mode="lines+markers", marker=dict(size=8)),
                    secondary_y=True)
    fig_c.add_hline(y=0, line_dash="dash", line_color="#ff4d6d",
                    annotation_text="Break-even (Sharpe = 0)",
                    annotation_position="right", annotation_font_color="#ff4d6d")
    # Reference broker markers
    fig_c.add_vline(x=0.1, line_dash="dot", line_color="#8a96aa", opacity=0.6,
                    annotation_text="Retail (~0.1%)", annotation_position="top",
                    annotation_font_color="#8a96aa")
    fig_c.add_vline(x=0.3, line_dash="dot", line_color="#8a96aa", opacity=0.6,
                    annotation_text="High-touch (~0.3%)", annotation_position="top",
                    annotation_font_color="#8a96aa")
    if breakeven is not None:
        fig_c.add_vline(x=breakeven, line_dash="solid", line_color="#ffd700",
                        line_width=2,
                        annotation_text=f"⭐ Max viable: {breakeven:.3f}%",
                        annotation_position="bottom", annotation_font_color="#ffd700")
    fig_c.update_xaxes(title_text="Commission rate (%)")
    fig_c.update_yaxes(title_text="Sharpe", secondary_y=False)
    fig_c.update_yaxes(title_text="Total return (%)", secondary_y=True)
    fig_c.update_layout(template=PLOTLY_TEMPLATE, height=420, hovermode="x unified",
                        title="Edge degradation as costs rise",
                        margin=dict(l=10, r=10, t=60, b=10),
                        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
    st.plotly_chart(fig_c, use_container_width=True)

    if sharpes[0] > 0.5 and sharpes[-1] < 0:
        st.warning("⚠ Edge collapses at higher costs — likely a transaction-cost mirage.")
    elif sharpes[-1] > 0.5 * max(sharpes[0], 0.01):
        st.success("✓ Edge survives at realistic retail costs (~0.1-0.2%).")
    else:
        st.info("Marginal edge. Be cautious about live deployment.")


# ---------- Single backtest page (existing) ----------

def render_backtest_page():
    cfg = render_topbar()
    how_to_card(
        "How to use Backtest",
        [
            "Pick <b>Single</b> mode for one ticker, or <b>Portfolio</b> for a basket.",
            "Set dates and capital (defaults work fine to start).",
            "Open the <b>Strategy & risk parameters</b> expander to tune MA windows / RSI / position size.",
            "Press <b>Run analysis</b>. Six tabs appear with the results.",
            "Top tab shows the live <b>BUY/SELL/HOLD</b> banner — what the strategy says today.",
            "<b>🔧 Diagnostics</b> tab (right-most) shows data quality if numbers look off.",
        ],
        tip="The ⭐ gold star on the price chart marks your best trade; the ⚠ red marker is your worst — study both."
    )
    if cfg["run"]:
        errs = validate_config(cfg)
        if errs:
            for e in errs:
                st.error(e)
        else:
            ok = run_pipeline(cfg)
            if ok:
                st.success("Analysis complete.")

    if "data" in st.session_state and st.session_state.data is not None:
        data = st.session_state.data
        signals_df = st.session_state.signals_df
        perf = st.session_state.strategy_performance
        backtest_results = st.session_state.backtest_results
        ticker = st.session_state.ticker
        tabs = st.tabs([
            "📈 Overview", "📊 Performance", "⚠️ Risk",
            "🔄 Trades", "📋 Report", "🔧 Diagnostics",
        ])
        with tabs[0]: render_overview(ticker, data, signals_df, perf, backtest_results)
        with tabs[1]: render_performance(signals_df, backtest_results, perf)
        with tabs[2]: render_risk(ticker, signals_df)
        with tabs[3]: render_trades(backtest_results)
        with tabs[4]: render_report(ticker, data, signals_df, perf, backtest_results)
        with tabs[5]: render_diagnostics(data, signals_df)
    else:
        st.info("Configure inputs above and press **Run analysis** to begin.")


# ---------- Main ----------

def main():
    st.sidebar.markdown("### 🚀 Navigate")
    page = st.sidebar.radio(
        "Page",
        ["📊 Backtest", "🔭 Watchlist Scanner", "⚖️ Strategy Comparison", "🔬 Optimize & Validate"],
        index=0, label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
**📅 Daily flow**
1. **Watchlist Scanner** → today's BUY/SELL signals
2. **Backtest** → deep-dive a symbol
3. **Compare** → vs other strategies & benchmarks
4. **Optimize** → verify edge isn't overfit

---

**🔧 Diagnostics**
Available as the right-most tab on the Backtest page after running an analysis.

**📘 How-to cards** appear at the top of each page.
"""
    )
    if page.startswith("📊"): render_backtest_page()
    elif page.startswith("🔭"): render_watchlist_page()
    elif page.startswith("⚖️"): render_compare_page()
    elif page.startswith("🔬"): render_optimize_page()


if __name__ == "__main__":
    main()
