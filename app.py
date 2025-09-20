import io
import math
from typing import Optional, Tuple, Dict, List
from html import escape
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import ta
import plotly.graph_objects as go

# =========================
# KONFIG / WYGLĄD
# =========================
st.set_page_config(page_title="RocketStock – NASDAQ Scanner", page_icon="💎", layout="wide")
st.markdown(
    """
    <style>
    :root { --rocket-purple:#7c3aed; }
    .stButton>button,.stDownloadButton>button{
      background:var(--rocket-purple)!important;border-color:var(--rocket-purple)!important;
      color:#fff!important;border-radius:10px!important;font-weight:600!important;padding:6px 10px!important;
    }
    .stButton>button:hover,.stDownloadButton>button:hover{ filter:brightness(0.92); }
    input[type="checkbox"],input[type="radio"]{ accent-color:var(--rocket-purple)!important; }
    div[data-baseweb="slider"] .rc-slider-track{ background:var(--rocket-purple)!important; }
    div[data-baseweb="slider"] .rc-slider-handle{ border-color:var(--rocket-purple)!important; }
    div[data-baseweb="slider"] .rc-slider-handle:active{ box-shadow:0 0 0 4px rgba(124,58,237,.2)!important; }

    .pill{ padding:2px 8px; border-radius:999px; background:#f5f3ff; color:#4c1d95; margin-right:6px; }
    .small{ font-size:12px; color:#6b7280; }

    /* Mobile dopasowania */
    @media (max-width: 820px){
      .block-container{ padding-left:0.6rem; padding-right:0.6rem; }
      [data-testid="column"]{ width:100% !important; flex: 1 0 100% !important; display:block !important; }
      .stPlotlyChart, .stMetric, .stButton{ margin-left:auto; margin-right:auto; width:100%; }
      .stButton>button{ width:100%; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# STAŁE / ŹRÓDŁA
# =========================
NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"

@st.cache_data(show_spinner=False, ttl=60*30)
def fetch_nasdaq_online() -> pd.DataFrame:
    resp = requests.get(NASDAQ_URL, timeout=15)
    resp.raise_for_status()
    content = resp.content.decode("utf-8", errors="ignore")
    df = pd.read_csv(io.StringIO(content), sep="|")
    if "Symbol" not in df.columns:
        raise ValueError("Brak kolumny 'Symbol' w pliku NASDAQ.")
    df = df[df["Symbol"] != "File Creation Time"]
    if "Test Issue" in df.columns:
        df = df[df["Test Issue"] == "N"]
    tickers = df[["Symbol"]].rename(columns={"Symbol": "Ticker"}).dropna()
    tickers["Ticker"] = tickers["Ticker"].str.strip()
    tickers = tickers[tickers["Ticker"] != ""].drop_duplicates()
    return tickers

@st.cache_data(show_spinner=False)
def load_tickers_from_csv(path: str = "nasdaq_tickers_full.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Ticker" not in df.columns:
        raise ValueError("Plik CSV musi mieć kolumnę 'Ticker'.")
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df = df[df["Ticker"] != ""].drop_duplicates()
    return df

def get_tickers(source: str) -> pd.DataFrame:
    if source == "Auto (online, fallback do CSV)":
        try:
            t = fetch_nasdaq_online()
            st.caption(f"Źródło tickerów: NASDAQTrader (online). Liczba: **{len(t)}**")
            return t
        except Exception as e:
            st.caption(f"⚠️ Nie udało się pobrać online: {e}. Wczytuję CSV…")
            return load_tickers_from_csv()
    else:
        t = load_tickers_from_csv()
        st.caption(f"Źródło tickerów: CSV z repo. Liczba: **{len(t)}**")
        return t

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# =========================
# INDIKATORY
# =========================
def compute_indicators(df: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    if "Close" not in df.columns or "Volume" not in df.columns:
        raise ValueError("Dane muszą zawierać 'Close' i 'Volume'.")
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["EMA50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
    df["EMA200"] = ta.trend.EMAIndicator(df["Close"], window=200).ema_indicator()
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    df["AvgVolume"] = df["Volume"].rolling(vol_window).mean()
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14)
    df["ATR"] = atr.average_true_range()
    df["PrevClose"] = df["Close"].shift(1)
    df["GapUpPct"] = (df["Open"] - df["PrevClose"]) / df["PrevClose"] * 100.0
    df["EMA200_Slope5"] = df["EMA200"].diff().rolling(5).mean()
    df["DistEMA200Pct"] = (df["Close"] / df["EMA200"] - 1.0) * 100.0
    df["RSI_Up"] = df["RSI"] >= df["RSI"].shift(1)
    df["High_3m"] = df["High"].rolling(63, min_periods=1).max()
    df["RoomToHighPct"] = (df["High_3m"] - df["Close"]) / df["Close"] * 100.0
    df["HH3"] = (df["High"] > df["High"].shift(1)) & (df["High"].shift(1) > df["High"].shift(2))
    df["HL3"] = (df["Low"]  > df["Low"].shift(1))  & (df["Low"].shift(1)  > df["Low"].shift(2))
    return df

def macd_bullish_cross_recent(df: pd.DataFrame, lookback: int) -> bool:
    macd = df["MACD"]; sig  = df["MACD_signal"]
    cross_up = (macd.shift(1) <= sig.shift(1)) & (macd > sig)
    return bool(cross_up.tail(lookback).any()))

@st.cache_data(show_spinner=False)
def get_stock_df(ticker: str, period: str, vol_window: int) -> Optional[pd.DataFrame]:
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    df = flatten_columns(df)
    try:
        return compute_indicators(df, vol_window)
    except Exception:
        return None

# =========================
# POMOCNICZE: Market Cap i Short % Float
# =========================
@st.cache_data(show_spinner=False, ttl=60*30)
def get_market_cap_fast(ticker: str) -> Optional[float]:
    try:
        tk = yf.Ticker(ticker)
        fi = tk.fast_info or {}
        mc = fi.get("market_cap")
        if mc is None:
            try:
                info = tk.get_info()
            except Exception:
                info = getattr(tk, "info", {}) or {}
            mc = info.get("marketCap")
        return float(mc) if mc is not None else None
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=60*60)
def get_short_percent_float(ticker: str) -> Optional[float]:
    """
    Zwraca short float jako ułamek 0–1.
    Fallback: jeśli brak shortPercentOfFloat, liczymy sharesShort / floatShares (lub sharesFloat).
    """
    try:
        tk = yf.Ticker(ticker)
        try:
            info = tk.get_info()
        except Exception:
            info = getattr(tk, "info", {}) or {}
        v = info.get("shortPercentOfFloat")
        if v is None:
            shares_short = info.get("sharesShort")
            float_shares = info.get("floatShares") or info.get("sharesFloat")
            if shares_short is not None and float_shares:
                try:
                    v = float(shares_short) / float(float_shares)
                except Exception:
                    v = None
        return float(v) if v is not None else None
    except Exception:
        return None

def nz(x, default=None):
    return default if (x is None or (isinstance(x, float) and pd.isna(x))) else x

# =========================
# DIAMENTY
# =========================
def score_diamonds(price, ema200, rsi, macd_cross, vol_ok, mode: str, rsi_min: int, rsi_max: int) -> str:
    if pd.isna(rsi) or rsi < rsi_min or rsi > rsi_max:
        return "–"
    pts = 0
    if mode == "Konserwatywny":
        if pd.notna(price) and pd.notna(ema200) and price > ema200: pts += 1
        if pd.notna(rsi): pts += 1
        if macd_cross: pts += 1
        if vol_ok: pts += 1
        if pts >= 4: return "💎💎💎"
        if pts == 3: return "💎💎"
        if pts == 2: return "💎"
        return "–"
    elif mode == "Umiarkowany":
        if pd.notna(price) and pd.notna(ema200) and (price >= ema200*0.995): pts += 1
        if pd.notna(rsi): pts += 1
        if macd_cross: pts += 1
        if vol_ok: pts += 1
        if pts >= 3: return "💎💎💎"
        if pts == 2: return "💎💎"
        if pts == 1: return "💎"
        return "–"
    else:
        if pd.notna(price) and pd.notna(ema200) and (price >= ema200*0.98): pts += 1
        if pd.notna(rsi): pts += 1
        if macd_cross: pts += 1
        if vol_ok: pts += 1
        if pts >= 3: return "💎💎💎"
        if pts == 2: return "💎💎"
        if pts == 1: return "💎"
        return "–"

def vol_confirmation(volume, avg_volume) -> bool:
    if pd.isna(volume) or pd.isna(avg_volume): return False
    return volume > avg_volume

def volume_label_from_ratio_simple(vr: Optional[float]) -> str:
    if vr is None or pd.isna(vr):
        return "Średni"
    try:
        vr = float(vr)
    except Exception:
        return "Średni"
    if vr >= 1.2:
        return "Wysoki"
    if vr >= 0.8:
        return "Średni"
    return "Niski"

# =========================
# WYKRESY
# =========================
def plot_candles_with_ema(df: pd.DataFrame, ticker: str, bars: int = 180):
    d = df.tail(bars)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                                 name=ticker, showlegend=False))
    fig.add_trace(go.Scatter(x=d.index, y=d["EMA200"], name="EMA200", mode="lines"))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10),
                      title=f"{ticker} — Świece + EMA200", xaxis_rangeslider_visible=False)
    return fig

def plot_rsi(df: pd.DataFrame, ticker: str, bars: int = 180):
    d = df.tail(bars)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], name="RSI(14)", mode="lines"))
    fig.add_hline(y=30, line_dash="dash"); fig.add_hline(y=70, line_dash="dash")
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=40, b=10),
                      title=f"{ticker} — RSI(14)", yaxis=dict(range=[0, 100]))
    return fig

def plot_macd(df: pd.DataFrame, ticker: str, bars: int = 180):
    d = df.tail(bars)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=df.tail(bars)["MACD_hist"], name="Histogram", opacity=0.3))
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], name="MACD", mode="lines"))
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD_signal"], name="Signal", mode="lines"))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=40, b=10),
                      title=f"{ticker} — MACD", showlegend=True)
    return fig

# =========================
# RENDER TABELI HTML (ESCAPE → fix na InvalidCharacterError)
# =========================
def render_table_left(df: pd.DataFrame, cols: list, max_h: int = 600):
    df_tbl = df[cols].copy()
    thead = "<tr>" + "".join(f"<th>{escape(str(c))}</th>" for c in cols) + "</tr>"
    rows_html = []
    for _, r in df_tbl.iterrows():
        tds = []
        for c in cols:
            v = r[c]
            if v is None or (isinstance(v, float) and pd.isna(v)):
                v = ""
            tds.append(f"<td data-label='{escape(str(c))}'>{escape(str(v))}</td>")
        rows_html.append("<tr>" + "".join(tds) + "</tr>")
    tbody = "\n".join(rows_html)
    html = f"""
    <div class="rs-table-wrap" style="max-height:{max_h}px; overflow:auto; border:1px solid #eee; border-radius:8px;">
      <table class="rs-table">
        <thead>
          {thead}
        </thead>
        <tbody>
          {tbody}
        </tbody>
      </table>
    </div>
    <style>
      .rs-table {{
        width:100%; border-collapse:collapse; table-layout:auto;
        min-width: 920px;
      }}
      .rs-table th, .rs-table td {{
        text-align:left; padding:10px 12px; border-bottom:1px solid #f1f1f1;
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
        font-size:14px;
      }}
      .rs-table thead th {{
        position: sticky; top: 0; background:#fafafa; z-index:1;
        font-weight:600; color:#374151;
      }}
      .rs-table tr:hover td {{ background:#fcfcff; }}
      .rs-table-wrap {{ overflow:auto; }}
      @media (max-width: 820px) {{
        .rs-table th, .rs-table td {{ padding:10px 10px; font-size:13px; }}
      }}
    </style>
    """
    st.markdown(html, unsafe_allow_html=True)

# =========================
# SIDEBAR — USTAWIENIA
# =========================
with st.sidebar:
    with st.expander("Skaner", expanded=True):
        signal_mode = st.radio("Tryb sygnału", ["Konserwatywny", "Umiarkowany", "Agresywny"], index=1, horizontal=True)
        rsi_min, rsi_max = st.slider("Przedział RSI (twardy)", 10, 80, (30, 50))
        vol_window = st.selectbox("Średni wolumen (okno)", ["MA20", "MA50"], index=0)
        vol_window = 20 if vol_window == "MA20" else 50

        require_price_above_ema200 = st.checkbox("Wymagaj Close > EMA200 (prescan)", value=True)
        ema_dist_cap = st.slider("Max % nad EMA200 (prescan)", 0, 15, 15)

        only_three = st.checkbox("Pokaż tylko 💎💎💎", value=False)
        vol_filter = st.selectbox("Filtr wolumenu", ["Wszystkie", "Wysoki", "Średni", "Niski"], index=0)
        scan_limit = st.slider("Limit skanowania (dla bezpieczeństwa)", 50, 5000, 300, step=50)

    with st.expander("Dodatkowe Filtry", expanded=False):
        f_minavg_on = st.checkbox("Min. średni wolumen (AvgVolume)", value=False)
        f_minavg_val = st.number_input("— Min AvgVolume", 0, 50_000_000, 1_000_000, step=100_000)

        f_vr_on = st.checkbox("Widełki VolRatio", value=False)
        colV1, colV2 = st.columns(2)
        with colV1:
            f_vr_min = st.number_input("— VR min", 0.0, 10.0, 1.2, step=0.1, format="%.1f")
        with colV2:
            f_vr_max = st.number_input("— VR max (cap)", 0.5, 10.0, 3.0, step=0.1, format="%.1f")

        f_mcap_on = st.checkbox("Filtr kapitalizacji (USD)", value=False)
        colM1, colM2 = st.columns(2)
        with colM1:
            f_mcap_min = st.number_input("— MC min (USD)", 0.0, 5_000_000_000_000.0, 300_000_000.0, step=50_000_000.0, format="%.0f")
        with colM2:
            f_mcap_max = st.number_input("— MC max (USD)", 0.0, 5_000_000_000_000.0, 2_000_000_000_000.0, step=50_000_000.0, format="%.0f")

        f_short_on = st.checkbox("Short float ≥ %", value=False)
        f_short_min = st.slider("— Próg Short float (≥ %)", 0, 100, 20, step=1)

        # OPCJONALNE TWARDZE filtry
        f_macd_on = st.checkbox("Wymagaj MACD cross (twardo)", value=False)
        macd_lookback = st.slider("— MACD: przecięcie (ostatnie N dni)", 1, 10, 3)
        f_volconfirm_on = st.checkbox("Wymagaj potwierdzenia wolumenem (twardo)", value=False)

        st.markdown("---")
        f_gap_on = st.checkbox("Max GAP UP %", value=False)
        f_gap_max = st.number_input("— GAP UP ≤ %", 0.0, 30.0, 8.0, step=0.5, format="%.1f")
        f_minprice_on = st.checkbox("Min cena ($)", value=False)
        f_minprice_val = st.number_input("— Cena ≥ $", 0.0, 2000.0, 5.0, step=0.5, format="%.1f")
        f_atr_on = st.checkbox("Max ATR% (ATR14/Close)", value=False)
        f_atr_max = st.number_input("— ATR% ≤", 0.0, 30.0, 8.0, step=0.5, format="%.1f")
        f_hhhl_on = st.checkbox("Struktura: HH & HL (ostatnie 3 świece)", value=False)
        f_resist_on = st.checkbox("Bliskość oporu: min 3% do 3-mies. high", value=False)
        f_resist_min = st.number_input("— Min odległość do 3m high (%)", 0.0, 20.0, 3.0, step=0.5, format="%.1f")

        st.markdown("---")
        source = st.selectbox("Źródło listy NASDAQ", ["Auto (online, fallback do CSV)", "Tylko CSV w repo"], index=0)
        period = st.selectbox("Okres danych", ["6mo", "1y", "2y"], index=1)

    with st.expander("Ranking", expanded=True):
        enable_rank = st.checkbox("Ranking (bez AI)", value=True)
        top_n = st.selectbox("Ile pozycji w TOP", [5, 10], index=1)
        rank_layout = st.selectbox("Układ rankingu", ["Kompakt (6/wiersz)", "Średni (4/wiersz)", "Wąski (3/wiersz)"], index=0)

    run_scan = st.button("🚀 Uruchom skaner", use_container_width=True, type="primary")

# ===== STAN
st.session_state.setdefault("scan_results_raw", pd.DataFrame())
st.session_state.setdefault("selected_symbol", None)
st.session_state.setdefault("selection_source", None)
st.session_state.setdefault("selectbox_symbol", "—")
st.session_state.setdefault("scan_done", False)
st.session_state.setdefault("scan_last_count", 0)
st.session_state["period"] = locals().get("period", "1y")
st.session_state["vol_window"] = locals().get("vol_window", 20)

# =========================
# PRO podsumowanie — formattery
# =========================
def _fmt_money(x, cur="USD"):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)): return "N/A"
        absx = abs(float(x))
        if absx >= 1e12: s = f"{x/1e12:.2f}T"
        elif absx >= 1e9: s = f"{x/1e9:.2f}B"
        elif absx >= 1e6: s = f"{x/1e6:.2f}M"
        elif absx >= 1e3: s = f"{x/1e3:.2f}K"
        else: s = f"{x:.2f}"
        return f"{s} {cur}"
    except Exception:
        return "N/A"

def _fmt_pct(x):
    try:
        return f"{float(x)*100:.1f}%" if abs(float(x))<2 else f"{float(x):.1f}%"
    except Exception:
        return "N/A"

def _pct_from(a, b) -> Optional[float]:
    try:
        if a is None or b in (None, 0): return None
        return (float(a)/float(b)-1.0)*100.0
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=60*15)
def fetch_fundamentals(ticker: str) -> dict:
    tk = yf.Ticker(ticker)
    data = {"ticker": ticker}
    try:
        fi = tk.fast_info
    except Exception:
        fi = {}
    data["currency"] = fi.get("currency")
    data["last_price"] = fi.get("last_price") or fi.get("lastPrice")
    data["market_cap"] = fi.get("market_cap")
    data["year_high"] = fi.get("year_high")
    data["year_low"]  = fi.get("year_low")
    data["shares"] = fi.get("shares")

    info = {}
    try:
        info = tk.get_info()
    except Exception:
        try:
            info = tk.info
        except Exception:
            info = {}
    def g(k): return info.get(k)

    data.update({
        "long_name": g("longName") or g("shortName"),
        "sector": g("sector"),
        "industry": g("industry"),
        "country": g("country"),
        "trailing_pe": g("trailingPE"),
        "forward_pe": g("forwardPE"),
        "peg_ratio": g("pegRatio"),
        "price_to_sales": g("priceToSalesTrailing12Months"),
        "price_to_book": g("priceToBook"),
        "enterprise_value": g("enterpriseValue"),
        "ebitda": g("ebitda"),
        "gross_margin": g("grossMargins"),
        "oper_margin": g("operatingMargins"),
        "profit_margin": g("profitMargins"),
        "free_cashflow": g("freeCashflow"),
        "total_debt": g("totalDebt"),
        "total_cash": g("totalCash"),
        "current_ratio": g("currentRatio"),
        "quick_ratio": g("quickRatio"),
        "shares_short": g("sharesShort"),
        "short_ratio": g("shortRatio"),
        "short_percent_float": g("shortPercentOfFloat"),
        "shares_float": g("floatShares") or g("sharesFloat"),
        "long_business_summary": g("longBusinessSummary"),
    })
    try:
        et = tk.earnings_trend
        if isinstance(et, pd.DataFrame) and not et.empty and "growth" in et:
            data["forward_eps_growth"] = et["growth"].astype(float).dropna().iloc[-1]
    except Exception:
        pass
    try:
        rs = tk.recommendations_summary
        if isinstance(rs, pd.DataFrame) and not rs.empty:
            data["rec_strong_buy"] = int(rs.get("strongBuy", pd.Series()).fillna(0).astype(int).sum())
            data["rec_buy"] = int(rs.get("buy", pd.Series()).fillna(0).astype(int).sum())
            data["rec_hold"] = int(rs.get("hold", pd.Series()).fillna(0).astype(int).sum())
            data["rec_sell"] = int(rs.get("sell", pd.Series()).fillna(0).astype(int).sum())
    except Exception:
        pass
    try:
        pt = tk.analyst_price_target
        if isinstance(pt, pd.DataFrame) and not pt.empty:
            data["price_target_mean"] = float(pt["targetMean"].dropna().iloc[-1])
            data["price_target_high"] = float(pt["targetHigh"].dropna().iloc[-1])
            data["price_target_low"]  = float(pt["targetLow"].dropna().iloc[-1])
    except Exception:
        pass
    try:
        divs = tk.dividends
        if isinstance(divs, pd.Series) and not divs.empty:
            ttm = float(divs.tail(4).sum())
            lp = data.get("last_price")
            data["div_ttm"] = ttm
            data["div_yield"] = (ttm/lp) if (lp and lp>0) else None
    except Exception:
        pass
    try:
        hist = tk.history(period="1y", interval="1d", auto_adjust=False)
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            ser = hist["Close"].dropna()
            def ret(days):
                if len(ser) < days+1: return None
                return (ser.iloc[-1]/ser.iloc[-1-days]-1.0)*100.0
            data["ret_1m"] = ret(21); data["ret_3m"] = ret(63)
            data["ret_6m"] = ret(126); data["ret_1y"] = ret(252 if len(ser)>252 else len(ser)-1)
            roll_max = ser.cummax(); drawdown = ser/roll_max - 1.0
            data["max_dd_1y"] = float(drawdown.min()*100.0)
    except Exception:
        pass
    return data

def compute_entries(df_full: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    if df_full is None or df_full.empty: return (None, None)
    last = df_full.iloc[-1]
    close = last.get("Close"); atr = last.get("ATR")
    ema50 = last.get("EMA50"); ema200 = last.get("EMA200")
    h20 = df_full["High"].tail(20).max() if "High" in df_full.columns else None
    if pd.isna(atr) or pd.isna(close): return (None, None)
    entry_breakout = None
    if pd.notna(h20):
        entry_breakout = max(float(close), float(h20)) + 0.10*float(atr)
    base_ema = ema50 if (pd.notna(ema50) and pd.notna(ema200) and pd.notna(close) and close>ema50>ema200) else ema200
    entry_pullback = (float(base_ema) + 0.10*float(atr)) if pd.notna(base_ema) else None
    return (entry_breakout, entry_pullback)

# ====== LOKALIZACJA PL (prosty słowniczek + tłumaczenie opisów) ======
SECTOR_MAP_PL = {
    "Technology": "Technologie",
    "Healthcare": "Ochrona zdrowia",
    "Consumer Cyclical": "Dobra konsumpcyjne cykliczne",
    "Consumer Defensive": "Dobra konsumpcyjne defensywne",
    "Industrials": "Przemysł",
    "Basic Materials": "Surowce",
    "Energy": "Energia",
    "Utilities": "Usługi użyteczności publicznej",
    "Financial Services": "Usługi finansowe",
    "Real Estate": "Nieruchomości",
    "Communication Services": "Usługi komunikacyjne",
}
INDUSTRY_MAP_PL = {
    "Semiconductors": "Półprzewodniki",
    "Software—Application": "Oprogramowanie — aplikacje",
    "Biotechnology": "Biotechnologia",
    "Medical Devices": "Urządzenia medyczne",
    "Diagnostics & Research": "Diagnostyka i badania",
    "Electronic Components": "Komponenty elektroniczne",
    "Electronics & Computer Distribution": "Dystrybucja elektroniki i komputerów",
    "Information Technology Services": "Usługi IT",
    "Internet Content & Information": "Treści i informacje internetowe",
}
COUNTRY_MAP_PL = {
    "United States": "Stany Zjednoczone",
    "Canada": "Kanada",
    "United Kingdom": "Wielka Brytania",
    "Germany": "Niemcy",
    "France": "Francja",
    "Japan": "Japonia",
    "China": "Chiny",
    "Israel": "Izrael",
    "Netherlands": "Niderlandy",
}

def _to_pl_label(en: Optional[str], mapping: Dict[str, str]) -> Optional[str]:
    if not en: return en
    return mapping.get(en, en)

def _translate_summary_simple(text: Optional[str]) -> Optional[str]:
    """Lekki, lokalny 'przekład': podstawienia prostych fraz + pozostawienie reszty po EN."""
    if not text: return text
    repl = {
        "company": "spółka",
        "provides": "świadczy",
        "manufactures": "produkuje",
        "develops": "rozwija",
        "designs": "projektuje",
        "and": "i",
        "services": "usługi",
        "products": "produkty",
        "customers": "klienci",
        "including": "w tym",
        "solutions": "rozwiązania",
        "software": "oprogramowanie",
        "hardware": "sprzęt",
        "cloud": "chmura",
        "medical": "medyczne",
        "devices": "urządzenia",
        "distribution": "dystrybucja",
        "electronics": "elektronika",
    }
    out = text
    for k, v in repl.items():
        out = out.replace(f" {k} ", f" {v} ")
        out = out.replace(f" {k}.", f" {v}.")
        out = out.replace(f" {k},", f" {v},")
        out = out.replace(f" {k};", f" {v};")
        out = out.replace(f" {k}:", f" {v}:")
        # na początku zdania
        if out.startswith(k + " "): out = out.replace(k + " ", v + " ", 1)
    return out

# ====== PODSUMOWANIE „JAK WCZEŚNIEJ” (+ dane z tabeli + opis PL) ======
def render_summary_pro(sym: str, df_src: pd.DataFrame, rsi_min: int, rsi_max: int):
    base_row = df_src[df_src["Ticker"] == sym]
    if base_row.empty:
        st.info("Brak danych do podsumowania."); return
    row = base_row.iloc[0]
    close = row.get("Close"); rsi=row.get("RSI"); ema=row.get("EMA200"); ema50=row.get("EMA50")
    vr=row.get("VolRatio")
    dist_pct = _pct_from(close, ema)
    df_full = get_stock_df(sym, period=st.session_state.get("period","1y"), vol_window=st.session_state.get("vol_window",20))
    entry_break, entry_pull = compute_entries(df_full)
    fn = fetch_fundamentals(sym)
    cur = fn.get("currency") or "USD"

    # Lokalne tłumaczenia podstawowych atrybutów
    sector_pl   = _to_pl_label(fn.get("sector"), SECTOR_MAP_PL)
    industry_pl = _to_pl_label(fn.get("industry"), INDUSTRY_MAP_PL)
    country_pl  = _to_pl_label(fn.get("country"), COUNTRY_MAP_PL)

    cap_txt = _fmt_money(fn.get("market_cap"), cur)
    title_bits = [
        sym,
        fn.get("long_name") or "",
        f"• {industry_pl or fn.get('industry') or '—'}",
        f"• {country_pl or fn.get('country') or '—'}",
        f"• MC: {cap_txt}"
    ]
    st.markdown("**" + "  ".join([x for x in title_bits if x]) + f"  •  waluta: {cur}**")

    # === DANE Z TABELI ===
    st.markdown("##### Dane z tabeli")
    # przygotowanie pełnego zestawu tak jak w tabeli listy
    short_pct = row.get("ShortPctFloat")
    mc_val = row.get("MarketCap")
    mc_b = (round(mc_val/1e9, 2) if (mc_val is not None) else None)
    grid = [
        ("Ticker", sym),
        ("Sygnał", row.get("Sygnał", "–")),
        ("Close", f"{close:.2f}" if close is not None else "—"),
        ("RSI", f"{rsi:.2f}" if rsi is not None else "—"),
        ("EMA200", f"{ema:.2f}" if ema is not None else "—"),
        ("VolRatio", f"{vr:.2f}" if vr is not None else "—"),
        ("AvgVolume", f"{int(row.get('AvgVolume')):,}".replace(",", " ") if row.get("AvgVolume") is not None else "—"),
        ("Short%", f"{short_pct:.2f}%" if short_pct is not None else "—"),
        ("MC (B USD)", f"{mc_b:.2f}" if mc_b is not None else "—"),
    ]
    # render w 3 kolumnach
    c1, c2, c3 = st.columns(3)
    cols = [c1, c2, c3]
    for i, (label, val) in enumerate(grid):
        with cols[i % 3]:
            st.write(f"**{label}:** {val}")

    # snapshot podstawowe
    snap = []
    if pd.notna(close): snap.append(f"Close **${close:.2f}**")
    if pd.notna(rsi):   snap.append(f"RSI **{rsi:.1f}** (zakres {rsi_min}–{rsi_max})")
    if dist_pct is not None: snap.append(f"vs EMA200 **{dist_pct:.2f}%**")
    if pd.notna(vr):    snap.append(f"VR **{vr:.2f}**")
    if snap:
        st.write(" · ".join(snap))

    # opis spółki — PL (z fallbackiem do EN)
    if fn.get("long_business_summary"):
        with st.expander("Czym zajmuje się spółka (po polsku)", expanded=False):
            pl_txt = _translate_summary_simple(fn["long_business_summary"])
            st.write(pl_txt or fn["long_business_summary"])
            with st.expander("Pokaż oryginał (EN)", expanded=False):
                st.write(fn["long_business_summary"])

    # proponowane wejścia
    reco = None
    if dist_pct is not None and pd.notna(rsi):
        if dist_pct > 8 or rsi >= (rsi_max - 1): reco = "Preferuj **pullback** (mniejszy pościg, lepszy RR)."
        else: reco = "Możliwy **breakout** nad lokalnym oporem."
    entry_lines = []
    if entry_break is not None: entry_lines.append(f"**Wejście (breakout):** ${entry_break:.2f}  _(max(H20, Close) + 0.10×ATR)_")
    if entry_pull  is not None: entry_lines.append(f"**Wejście (pullback):** ${entry_pull:.2f}  _(EMA bazowa + 0.10×ATR)_")
    if entry_lines or reco:
        st.markdown("**Proponowane wejścia**")
        for ln in entry_lines: st.write(ln)
        if reco: st.write(reco)

    # === Sekcje „jak wcześniej” ===
    with st.expander("Wycena i jakość", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"P/E (TTM): **{nz(fn.get('trailing_pe'),'N/A')}**")
            st.write(f"Forward P/E: **{nz(fn.get('forward_pe'),'N/A')}**")
            st.write(f"PEG: **{nz(fn.get('peg_ratio'),'N/A')}**")
        with col2:
            st.write(f"P/S: **{nz(fn.get('price_to_sales'),'N/A')}**")
            st.write(f"P/B: **{nz(fn.get('price_to_book'),'N/A')}**")
            ev = nz(fn.get('enterprise_value')); ebitda = nz(fn.get('ebitda'))
            try:
                ev_ebitda = (float(ev)/float(ebitda)) if (ev and ebitda and float(ebitda)!=0) else None
            except Exception:
                ev_ebitda = None
            st.write(f"EV/EBITDA: **{ev_ebitda:.2f}**" if ev_ebitda is not None else "EV/EBITDA: **N/A**")
        with col3:
            gm = fn.get("gross_margin"); om = fn.get("oper_margin"); pm = fn.get("profit_margin")
            st.write(f"Gross Margin: **{_fmt_pct(gm)}**")
            st.write(f"Oper. Margin: **{_fmt_pct(om)}**")
            st.write(f"Net Margin: **{_fmt_pct(pm)}**")

    with st.expander("Zwroty i ryzyko", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"1M: **{fn.get('ret_1m'):.1f}%**" if fn.get("ret_1m") is not None else "1M: **N/A**")
            st.write(f"3M: **{fn.get('ret_3m'):.1f}%**" if fn.get("ret_3m") is not None else "3M: **N/A**")
        with col2:
            st.write(f"6M: **{fn.get('ret_6m'):.1f}%**" if fn.get("ret_6m") is not None else "6M: **N/A**")
            st.write(f"1Y: **{fn.get('ret_1y'):.1f}%**" if fn.get("ret_1y") is not None else "1Y: **N/A**")
        with col3:
            st.write(f"Max DD (1Y): **{fn.get('max_dd_1y'):.1f}%**" if fn.get("max_dd_1y") is not None else "Max DD (1Y): **N/A**")

    with st.expander("Wydarzenia i dywidendy", expanded=False):
        st.write("Earnings: **N/A**")
        div_y = fn.get("div_yield")
        if div_y is not None:
            st.write(f"Dividend (TTM): **{_fmt_money(fn.get('div_ttm'), cur)}**  •  Yield: **{div_y*100:.2f}%**")
        else:
            st.write("Dividend: **N/A**")

    with st.expander("Short interest (Yahoo)", expanded=False):
        ss = fn.get("shares_short")
        spf = fn.get("short_percent_float")  # 0..1
        sr  = fn.get("short_ratio")
        flt = fn.get("shares_float")
        row1 = f"Shares short: **{int(ss):,}**" if ss not in (None, float('nan')) else "Shares short: **N/A**"
        row1 = row1.replace(",", " ")
        st.write(row1)
        st.write(f"Short % float: **{spf*100:.2f}%**" if spf not in (None, float('nan')) else "Short % float: **N/A**")
        st.write(f"Short ratio (days to cover): **{sr:.2f}**" if sr not in (None, float('nan')) else "Short ratio: **N/A**")
        if flt not in (None, float('nan')):
            try:
                st.write(f"Float shares: **{int(flt):,}**".replace(",", " "))
            except Exception:
                st.write("Float shares: **N/A**")

    # „Ryzyka / wady / uwagi” — proste reguły
    with st.expander("Ryzyka / wady / uwagi (automatyczne)", expanded=False):
        notes = []
        # dystans do EMA200
        if dist_pct is not None and dist_pct > 12:
            notes.append("• **Daleko od EMA200** (>12%) → ryzyko pościgu.")
        # ATR%
        atr_pct = None
        if df_full is not None and not df_full.empty:
            last = df_full.iloc[-1]
            if pd.notna(last.get("ATR")) and pd.notna(last.get("Close")) and float(last.get("Close"))>0:
                atr_pct = float(last["ATR"])/float(last["Close"])*100.0
        if atr_pct is not None and atr_pct > 10:
            notes.append("• **Wysokie ATR%** (>10%) → podwyższona zmienność.")
        # GAP
        if df_full is not None and not df_full.empty:
            last = df_full.iloc[-1]
            if pd.notna(last.get("GapUpPct")) and float(last["GapUpPct"]) > 8:
                notes.append("• **Duży GAP UP** (>8%) → ryzyko domknięcia luki.")
        # Płynność
        if row.get("AvgVolume") is not None and row["AvgVolume"] < 1_000_000:
            notes.append("• **Niska płynność** (AvgVolume < 1M) → gorsze wykonanie zleceń.")
        # Short wysoki
        if row.get("ShortPctFloat") is not None and row["ShortPctFloat"] >= 20:
            notes.append("• **Wysoki Short float** (≥20%) → większa zmienność (możliwy squeeze).")
        if notes:
            for n in notes: st.write(n)
        else:
            st.write("Brak szczególnych ostrzeżeń wg prostych progów.")

# =========================
# FAST SCAN: batch cen + prescan + równoległe meta
# =========================
def batch_download_prices(tickers: List[str], period: str) -> Dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    data = yf.download(tickers, period=period, interval="1d", auto_adjust=False,
                       group_by="ticker", threads=True, progress=False)
    out: Dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.get_level_values(0):
                df_t = data[t].dropna(how="all")
                if not df_t.empty:
                    out[t] = df_t
    else:
        t = tickers[0]
        out[t] = data.dropna(how="all")
    return out

def fetch_meta_parallel(tickers: List[str], need_mcap: bool, need_short: bool) -> Dict[str, dict]:
    results: Dict[str, dict] = {t: {} for t in tickers}
    if not tickers: return results

    def worker(t):
        d = {}
        try:
            if need_mcap:
                d["MarketCap"] = get_market_cap_fast(t)
        except Exception:
            d["MarketCap"] = None
        try:
            if need_short:
                spf = get_short_percent_float(t)
                d["ShortPctFloat"] = (spf * 100.0) if spf is not None else None
        except Exception:
            d["ShortPctFloat"] = None
        return t, d

    with ThreadPoolExecutor(max_workers=min(16, max(4, len(tickers)))) as ex:
        futs = [ex.submit(worker, t) for t in tickers]
        for f in as_completed(futs):
            t, d = f.result()
            results[t].update(d)
    return results

# =========================
# SKAN — PRESCAN (twarde) + ZAPIS
# =========================
if run_scan:
    st.session_state["scan_done"] = False
    raw_results = []
    tickers_df = get_tickers(source)
    if tickers_df is None or tickers_df.empty:
        st.error("Brak tickerów do skanowania.")
    else:
        tickers_list = tickers_df["Ticker"].tolist()[:scan_limit]
        if not tickers_list:
            st.error("Lista tickerów pusta po limitach.")
        else:
            status = st.empty()
            status.write("⏳ Pobieram notowania (batch)…")
            price_map = batch_download_prices(tickers_list, period)
            if not price_map:
                st.error("Nie udało się pobrać notowań (batch).")
            else:
                progress = st.progress(0)
                passed_first: List[str] = []
                for i, t in enumerate(tickers_list, start=1):
                    df_raw = price_map.get(t)
                    if df_raw is None or df_raw.empty:
                        progress.progress(i/len(tickers_list)); continue
                    df = flatten_columns(df_raw.copy())
                    try:
                        df = compute_indicators(df, st.session_state.get("vol_window",20))
                    except Exception:
                        progress.progress(i/len(tickers_list)); continue

                    last = df.iloc[-1]
                    # RSI twardo
                    if not (pd.notna(last.get("RSI")) and (rsi_min <= float(last.get("RSI")) <= rsi_max)):
                        progress.progress(i/len(tickers_list)); continue
                    # Close > EMA200 + cap (opcjonalny)
                    if require_price_above_ema200:
                        if not (pd.notna(last.get("Close")) and pd.notna(last.get("EMA200")) and float(last.get("EMA200"))>0):
                            progress.progress(i/len(tickers_list)); continue
                        dist_pct_now = (float(last.get("Close"))/float(last.get("EMA200")) - 1.0)*100.0
                        if not (0.0 <= dist_pct_now <= float(ema_dist_cap)):
                            progress.progress(i/len(tickers_list)); continue
                    # Min AvgVolume
                    if f_minavg_on:
                        if not (pd.notna(last.get("AvgVolume")) and float(last.get("AvgVolume")) >= float(f_minavg_val)):
                            progress.progress(i/len(tickers_list)); continue
                    # VolRatio widełki
                    vr_val = None
                    if pd.notna(last.get("Volume")) and pd.notna(last.get("AvgVolume")) and float(last.get("AvgVolume"))>0:
                        vr_val = float(last.get("Volume"))/float(last.get("AvgVolume"))
                    if f_vr_on:
                        if not (vr_val is not None and (float(f_vr_min) <= vr_val <= float(f_vr_max))):
                            progress.progress(i/len(tickers_list)); continue
                    # GAP
                    if f_gap_on:
                        if not (pd.notna(last.get("GapUpPct")) and float(last.get("GapUpPct")) <= float(f_gap_max)):
                            progress.progress(i/len(tickers_list)); continue
                    # Min cena
                    if f_minprice_on:
                        if not (pd.notna(last.get("Close")) and float(last.get("Close")) >= float(f_minprice_val)):
                            progress.progress(i/len(tickers_list)); continue
                    # ATR%
                    if f_atr_on:
                        if not (pd.notna(last.get("ATR")) and pd.notna(last.get("Close")) and float(last.get("Close"))>0):
                            progress.progress(i/len(tickers_list)); continue
                        atr_pct = float(last.get("ATR"))/float(last.get("Close"))*100.0
                        if atr_pct > float(f_atr_max):
                            progress.progress(i/len(tickers_list)); continue
                    # HHHL
                    if f_hhhl_on and len(df) >= 3:
                        hh_ok = bool(pd.notna(df["HH3"].iloc[-1]) and df["HH3"].iloc[-1])
                        hl_ok = bool(pd.notna(df["HL3"].iloc[-1]) and df["HL3"].iloc[-1])
                        if not (hh_ok and hl_ok):
                            progress.progress(i/len(tickers_list)); continue
                    # Dystans do 3m high
                    if f_resist_on and pd.notna(last.get("RoomToHighPct")):
                        if float(last.get("RoomToHighPct")) < float(f_resist_min):
                            progress.progress(i/len(tickers_list)); continue
                    # Twardo: MACD cross (jeśli żądane)
                    macd_cross = macd_bullish_cross_recent(df, macd_lookback)
                    if f_macd_on and not macd_cross:
                        progress.progress(i/len(tickers_list)); continue
                    # Twardo: potwierdzenie wolumenem (jeśli żądane)
                    vol_ok_simple = vol_confirmation(last.get("Volume"), last.get("AvgVolume"))
                    if f_volconfirm_on and not vol_ok_simple:
                        progress.progress(i/len(tickers_list)); continue

                    passed_first.append(t)
                    progress.progress(i/len(tickers_list))

                status.write(f"✅ Wstępny prescan: {len(passed_first)} kandydatów. Pobieram metadane…")

                # Meta tylko dla przefiltrowanych
                need_mcap = True
                need_short = True
                meta = fetch_meta_parallel(passed_first, need_mcap=need_mcap, need_short=need_short)

                # Drugi etap: MC/Short + zapis
                for t in passed_first:
                    df = compute_indicators(price_map[t].copy(), st.session_state.get("vol_window",20))
                    last = df.iloc[-1]
                    mc_val = meta.get(t, {}).get("MarketCap")
                    spf_pct = meta.get(t, {}).get("ShortPctFloat")  # 0..100

                    if f_mcap_on:
                        if not (mc_val is not None and float(f_mcap_min) <= mc_val <= float(f_mcap_max)):
                            continue
                    if f_short_on:
                        if not (spf_pct is not None and spf_pct >= float(f_short_min)):
                            continue

                    macd_cross = macd_bullish_cross_recent(df, macd_lookback)
                    vol_ok_simple = vol_confirmation(last.get("Volume"), last.get("AvgVolume"))
                    di = score_diamonds(
                        last.get("Close"), last.get("EMA200"), last.get("RSI"),
                        macd_cross, vol_ok_simple, signal_mode, rsi_min, rsi_max
                    )

                    vr_val = None
                    if pd.notna(last.get("Volume")) and pd.notna(last.get("AvgVolume")) and float(last.get("AvgVolume"))>0:
                        vr_val = float(last.get("Volume"))/float(last.get("AvgVolume"))

                    raw_results.append({
                        "Ticker": t,
                        "Sygnał": di,
                        "Close": round(float(last.get("Close")), 2) if pd.notna(last.get("Close")) else None,
                        "RSI": round(float(last.get("RSI")), 2) if pd.notna(last.get("RSI")) else None,
                        "EMA200": round(float(last.get("EMA200")), 2) if pd.notna(last.get("EMA200")) else None,
                        "VolRatio": round(vr_val, 2) if vr_val is not None else None,
                        "AvgVolume": int(last.get("AvgVolume")) if pd.notna(last.get("AvgVolume")) else None,
                        "MarketCap": float(mc_val) if mc_val is not None else None,
                        "ShortPctFloat": float(spf_pct) if spf_pct is not None else None,
                    })

                st.session_state.scan_results_raw = pd.DataFrame(raw_results)
                st.session_state["scan_done"] = True
                st.session_state["scan_last_count"] = len(raw_results)
                status.write("✅ Zakończono skan.")
                st.success(f"Wyników po twardych filtrach: **{len(raw_results)}**.")

# =========================
# RANKING
# =========================
def _clip01(x):
    try: return max(0.0, min(1.0, float(x)))
    except Exception: return 0.0

def _trapezoid(x, a, b, c, d):
    try: x = float(x)
    except Exception: return 0.0
    if x <= a or x >= d: return 0.0
    if b <= x <= c: return 1.0
    if a < x < b: return (x - a) / (b - a)
    return (d - x) / (d - c)

def _bell(x, a, b, c, d): return _trapezoid(x, a, b, c, d)

def _liquidity_score(avgv):
    try: v = float(avgv or 0)
    except Exception: v = 0
    if v >= 5_000_000: return 1.0
    if v >= 2_000_000: return 0.7
    if v >= 1_000_000: return 0.5
    if v >=   500_000: return 0.2
    return 0.0

def rank_score_row_v2(row, rsi_min: int, rsi_max: int) -> float:
    e200pos = _bell(row.get("Close")/row.get("EMA200")*100-100 if (row.get("Close") and row.get("EMA200")) else 0, 0.0,2.0,6.0,12.0)
    slope_score = 0.0
    stack_score = 1.0 if (row.get("Close") and row.get("EMA200") and row.get("Close")>row.get("EMA200")) else 0.0
    rsi = row.get("RSI"); rsi_score = 0.0
    if rsi is not None:
        if 45 <= rsi <= 55: rsi_score = 1.0
        elif (40 <= rsi < 45) or (55 < rsi <= 60): rsi_score = 0.5
    macd_score = 0.5
    vr = row.get("VolRatio"); volq = _bell(vr, 0.9, 1.1, 1.8, 3.0)
    atr_score = 0.5
    liq = _liquidity_score(row.get("AvgVolume"))
    score = 15*e200pos + 10*slope_score + 10*stack_score + 15*rsi_score + 10*macd_score + 15*volq + 10*atr_score + 15*liq
    return round(score, 1)

def build_ranking(df: pd.DataFrame, rsi_min: int, rsi_max: int, top_n: int) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame(columns=["Ticker","Score"])
    base = df.copy()
    base = base[base["Sygnał"]=="💎💎💎"]
    if base.empty: return pd.DataFrame(columns=["Ticker","Score"])
    base["Score"] = base.apply(lambda r: rank_score_row_v2(r, rsi_min, rsi_max), axis=1)
    base = base.sort_values(["Score","AvgVolume","Ticker"], ascending=[False, False, True])
    return base[["Ticker","Score"]].head(top_n).reset_index(drop=True)

# =========================
# ZAKŁADKI
# =========================
PRZEWODNIK_MD = r"""
# Przewodnik użytkownika – RocketStock

(…)
"""

tab_scan, tab_guide = st.tabs(["Skaner", "Przewodnik"])

with tab_scan:
    raw = st.session_state.get("scan_results_raw", pd.DataFrame())

    if not raw.empty:
        df_view = raw.copy()
        df_view["Wolumen"] = df_view["VolRatio"].apply(volume_label_from_ratio_simple)
        if only_three:
            df_view = df_view[df_view["Sygnał"] == "💎💎💎"]
        if 'vol_filter' in locals() and vol_filter != "Wszystkie":
            df_view = df_view[df_view["Wolumen"] == vol_filter]

        # ===== RANKING =====
        if enable_rank:
            rank_df = build_ranking(raw, rsi_min, rsi_max, top_n)
            st.markdown(f"### Proponowane (ranking 1–{len(rank_df) if not rank_df.empty else top_n})")
            if rank_df.empty:
                st.info("Brak kandydatów (💎💎💎). Zmień parametry.")
            else:
                per_row = 6 if "Kompakt" in rank_layout else (4 if "Średni" in rank_layout else 3)
                for start in range(0, len(rank_df), per_row):
                    row_slice = rank_df.iloc[start:start+per_row]
                    cols = st.columns(len(row_slice))
                    for pos, ((_, rr), col) in enumerate(zip(row_slice.iterrows(), cols)):
                        with col:
                            rank_no = start + pos + 1
                            label = f"{rank_no}. {rr['Ticker']} · {rr['Score']:.1f}"
                            if st.button(label, key=f"rank_{rr['Ticker']}", use_container_width=True):
                                st.session_state["selected_symbol"] = rr["Ticker"]
                                st.session_state["selection_source"] = "rank"
                                st.session_state["selectbox_symbol"] = "—"

        # ===== WYBÓR + SORTOWANIE =====
        st.subheader("Wybierz spółkę i sortowanie")
        sort_cols = ["Ticker","Sygnał","Close","RSI","EMA200","Wolumen","Short%","MC (B USD)"]
        col_left, col_right = st.columns([2, 1])
        with col_left:
            tickers_list = df_view["Ticker"].dropna().astype(str).sort_values().unique().tolist()
            sel = st.selectbox("Spółka", ["—"] + tickers_list, key="selectbox_symbol")
            if sel != "—":
                st.session_state["selected_symbol"] = sel
                st.session_state["selection_source"] = "selectbox"
        with col_right:
            sort_by = st.selectbox("Sortowanie", sort_cols, index=0, key="sort_by")
            sort_dir = st.radio("Kierunek", ["Rosnąco","Malejąco"], index=0, horizontal=True, key="sort_dir")

        # ===== TABELA =====
        st.markdown("---")
        st.subheader("Wyniki skanera (lista)")
        pills = (
            f"<span class='pill'>Wyników: <b>{len(df_view)}</b></span>"
            f"<span class='pill'>RSI (twardo): <b>{rsi_min}–{rsi_max}</b></span>"
            f"<span class='pill'>Tryb: <b>{signal_mode}</b></span>"
            f"<span class='pill'>Okres: <b>{period}</b></span>"
            f"<span class='pill'>Close>EMA200 cap: <b>0–{ema_dist_cap}%</b></span>"
        )
        if f_short_on: pills += f"<span class='pill'>Short float: <b>≥ {f_short_min}%</b></span>"
        if f_macd_on: pills += "<span class='pill'>MACD cross: <b>wymagany</b></span>"
        if f_volconfirm_on: pills += "<span class='pill'>Potw. wolumenem: <b>wymagane</b></span>"
        st.write(pills, unsafe_allow_html=True)

        df_show = df_view[["Ticker","Sygnał","Close","RSI","EMA200","Wolumen","MarketCap","ShortPctFloat"]].copy()
        df_show.rename(columns={"ShortPctFloat": "Short%", "MarketCap": "MC (B USD)"}, inplace=True)
        df_show["MC (B USD)"] = df_show["MC (B USD)"].apply(lambda x: round(x/1e9, 2) if pd.notna(x) else None)
        for c in ["Close","RSI","EMA200","Short%"]:
            if c in df_show.columns:
                df_show[c] = df_show[c].apply(lambda x: round(float(x), 2) if pd.notna(x) else None)

        ascending = (sort_dir == "Rosnąco")
        if sort_by in df_show.columns:
            df_show = df_show.sort_values(by=sort_by, ascending=ascending, na_position="last", kind="mergesort")

        rows = len(df_show)
        row_h = 35
        header_h = 46
        target_h = min(700, max(240, header_h + rows*row_h))
        cols_tbl = ["Ticker","Sygnał","Close","RSI","EMA200","Wolumen","Short%","MC (B USD)"]
        try:
            render_table_left(df_show, cols_tbl, max_h=target_h)
        except Exception as e:
            st.warning(f"Problem z rendererem tabeli: {e}. Pokazuję widok awaryjny.")
            st.dataframe(df_show, use_container_width=True)

        # ===== PODSUMOWANIE + WYKRESY =====
        sym = st.session_state.get("selected_symbol")
        if sym:
            st.markdown("---")
            st.subheader(f"{sym} — podgląd wykresów")
            with st.spinner(f"Ładuję wykresy dla {sym}…"):
                df_sel = get_stock_df(sym, period=st.session_state.get("period","1y"), vol_window=st.session_state.get("vol_window",20))
            if df_sel is None or df_sel.empty:
                st.error("Nie udało się pobrać danych wykresu.")
            else:
                last = df_sel.iloc[-1]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Kurs (Close)", f"{last.get('Close'):.2f}" if pd.notna(last.get("Close")) else "—")
                m2.metric("RSI", f"{last.get('RSI'):.2f}" if pd.notna(last.get("RSI")) else "—")
                distv = (last.get("Close")/last.get("EMA200")-1)*100 if pd.notna(last.get("Close")) and pd.notna(last.get("EMA200")) else None
                m3.metric("Dystans do EMA200", f"{distv:.2f}%" if distv is not None else "—")
                macd_cross_here = macd_bullish_cross_recent(df_sel, locals().get("macd_lookback",3))
                vol_ok_here = vol_confirmation(last.get("Volume"), last.get("AvgVolume"))
                di_here = score_diamonds(last.get("Close"), last.get("EMA200"), last.get("RSI"),
                                         macd_cross_here, vol_ok_here, locals().get("signal_mode","Umiarkowany"), rsi_min, rsi_max)
                m4.metric("Sygnał", di_here)

                st.plotly_chart(plot_candles_with_ema(df_sel, sym), use_container_width=True)
                st.plotly_chart(plot_rsi(df_sel, sym), use_container_width=True)
                st.plotly_chart(plot_macd(df_sel, sym), use_container_width=True)

                st.markdown("### Podsumowanie")
                try:
                    render_summary_pro(sym, st.session_state.get("scan_results_raw", pd.DataFrame()), rsi_min, rsi_max)
                except Exception as e:
                    st.warning(f"Nie udało się zbudować Podsumowania: {e}")

    else:
        if st.session_state.get("scan_done"):
            st.info(
                "Skan zakończony, ale **brak wyników** przy aktualnych twardych filtrach. "
                "Poluzuj przynajmniej jeden warunek (np. wyłącz `Close>EMA200`, obniż `Short float ≥ %`, "
                "lub wyłącz `Wymagaj MACD`/`potwierdzenie wolumenem`) i uruchom ponownie."
            )
            cnt = st.session_state.get("scan_last_count", 0)
            st.caption(f"Wyników po filtrach: **{cnt}**")
        else:
            st.info("Otwórz panel **Skaner** po lewej i kliknij **Uruchom skaner**.")

with tab_guide:
    st.markdown(PRZEWODNIK_MD, unsafe_allow_html=True)
    st.caption("© RocketStock — materiały edukacyjne. Brak rekomendacji inwestycyjnych.")
