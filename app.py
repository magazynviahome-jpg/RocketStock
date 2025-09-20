import io
import math
from typing import Optional, Tuple
from html import escape

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
    return bool(cross_up.tail(lookback).any())

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
# SCORING DIAMENTÓW
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
    """Proste potwierdzenie: dzisiaj > średnia."""
    if pd.isna(volume) or pd.isna(avg_volume): return False
    return volume > avg_volume

# Kategoryzacja VolRatio → Wolumen: Wysoki / Średni / Niski
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
        # (PRZENOŚNIK) suwak MACD usunięty z tej sekcji
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

        # ---- Short float %: twardy próg "≥ X%"
        f_short_on = st.checkbox("Short float ≥ %", value=False)
        f_short_min = st.slider("— Próg Short float (≥ %)", 0, 100, 20, step=1)

        # ---- OPCJONALNE TWARDYE filtry
        f_macd_on = st.checkbox("Wymagaj MACD cross (twardo)", value=False)
        # suwak przeniesiony TUTAJ:
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
st.session_state["period"] = locals().get("period", "1y")
st.session_state["vol_window"] = locals().get("vol_window", 20)

# =========================
# PRO podsumowanie
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

def render_summary_pro(sym: str, df_src: pd.DataFrame, rsi_min: int, rsi_max: int):
    base_row = df_src[df_src["Ticker"] == sym]
    if base_row.empty:
        st.info("Brak danych do podsumowania."); return
    row = base_row.iloc[0]
    close = row.get("Close"); rsi=row.get("RSI"); ema=row.get("EMA200"); vr=row.get("VolRatio")
    dist_pct = _pct_from(close, ema)
    df_full = get_stock_df(sym, period=st.session_state.get("period","1y"), vol_window=st.session_state.get("vol_window",20))
    entry_break, entry_pull = compute_entries(df_full)
    fn = fetch_fundamentals(sym)
    cur = fn.get("currency") or "USD"

    cap_txt = _fmt_money(fn.get("market_cap"), cur)
    title_bits = [sym, fn.get("long_name") or "", f"• {fn.get('industry') or '—'}", f"• {fn.get('country') or '—'}", f"• MC: {cap_txt}"]
    st.markdown("**" + "  ".join([x for x in title_bits if x]) + f"  •  waluta: {cur}**")

    wolumen_cat = volume_label_from_ratio_simple(vr)
    mc_b = row.get("MarketCap")
    mc_b_disp = f"{round(mc_b/1e9, 2)}" if (mc_b is not None and not pd.isna(mc_b)) else "—"
    short_pct = row.get("ShortPctFloat")

    st.markdown("**Dane z tabeli**")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.write(f"Sygnał: **{row.get('Sygnał') or '—'}**")
        st.write(f"Close: **{close:.2f}**" if pd.notna(close) else "Close: **—**")
    with colB:
        st.write(f"RSI: **{rsi:.2f}**" if pd.notna(rsi) else "RSI: **—**")
        st.write(f"EMA200: **{ema:.2f}**" if pd.notna(ema) else "EMA200: **—**")
    with colC:
        st.write(f"Dist EMA200 %: **{dist_pct:.2f}%**" if dist_pct is not None else "Dist EMA200 %: **—**")
        st.write(f"Wolumen: **{wolumen_cat}**")
    with colD:
        st.write(f"Short %: **{short_pct:.2f}%**" if short_pct is not None else "Short %: **—**")
        st.write(f"MC (B USD): **{mc_b_disp}**")

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

    with st.expander("Wycena i jakość", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"P/E (TTM): **{nz(fn.get('trailing_pe'),'N/A')}**")
            st.write(f"Forward P/E: **{nz(fn.get('forward_pe'),'N/A')}**")
            st.write(f"PEG: **{nz(fn.get('peg_ratio'),'N/A')}**")
    ...  # (pozostała część fetch_fundamentals / expanderów bez zmian — skrócone tu tylko dla czytelności)
