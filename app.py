import io
import math
from typing import Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import ta
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

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
    .ag-theme-alpine .ag-header,.ag-theme-alpine .ag-root-wrapper{ border-radius:8px; }
    .ag-theme-alpine .ag-row.ag-row-selected{ background-color:rgba(124,58,237,.12)!important; }
    .ag-theme-alpine .ag-row-hover{ background-color:rgba(124,58,237,.08)!important; }
    .pill{ padding:2px 8px; border-radius:999px; background:#f5f3ff; color:#4c1d95; margin-right:6px; }
    .small{ font-size:12px; color:#6b7280; }
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

def vol_confirmation(volume, avg_volume, require: bool) -> bool:
    if not require: return True
    if pd.isna(volume) or pd.isna(avg_volume): return False
    return volume > avg_volume

def volume_label_from_ratio_qtile(q) -> str:
    if pd.isna(q): return "—"
    if q >= 0.80:  return "Bardzo wysoki"
    if q >= 0.60:  return "Wysoki"
    if q >= 0.40:  return "Normalny"
    if q >= 0.20:  return "Niski"
    return "Bardzo niski"

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
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], name="MACD", mode="lines"))
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD_signal"], name="Signal", mode="lines"))
    fig.add_trace(go.Bar(x=d.index, y=df.tail(bars)["MACD_hist"], name="Histogram", opacity=0.3))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=40, b=10),
                      title=f"{ticker} — MACD", showlegend=True)
    return fig

# =========================
# SIDEBAR — USTAWIENIA + WYGLĄD
# =========================
with st.sidebar:
    with st.expander("Skaner", expanded=True):
        signal_mode = st.radio("Tryb sygnału", ["Konserwatywny", "Umiarkowany", "Agresywny"], index=1, horizontal=True)
        rsi_min, rsi_max = st.slider("Przedział RSI (twardy)", 10, 80, (30, 50))
        macd_lookback = st.slider("MACD: przecięcie (ostatnie N dni)", 1, 10, 3)
        use_volume = st.checkbox("Wymagaj potwierdzenia wolumenem", value=True)
        vol_window = st.selectbox("Średni wolumen (okno)", ["MA20", "MA50"], index=0)
        vol_window = 20 if vol_window == "MA20" else 50

        only_three = st.checkbox("Pokaż tylko 💎💎💎", value=False)
        require_price_above_ema200 = st.checkbox("Wymagaj Close > EMA200", value=True)

        vol_filter = st.selectbox("Filtr wolumenu", ["Wszystkie", "Bardzo wysoki", "Wysoki", "Normalny", "Niski", "Bardzo niski"], index=0)
        scan_limit = st.slider("Limit skanowania (dla bezpieczeństwa)", 50, 5000, 300, step=50)

        st.markdown("---")
        source = st.selectbox("Źródło listy NASDAQ", ["Auto (online, fallback do CSV)", "Tylko CSV w repo"], index=0)
        period = st.selectbox("Okres danych", ["6mo", "1y", "2y"], index=1)

    with st.expander("Dodatkowe filtry (opcjonalne)", expanded=False):
        # wszystkie odznaczone domyślnie
        f_maxdist_on = st.checkbox("Max dystans do EMA200", value=False)
        f_maxdist_pct = st.slider("— Maks. % nad EMA200", 5, 30, 15) if f_maxdist_on else 15
        f_slope_on = st.checkbox("EMA200 rośnie (nachylenie > 0)", value=False)
        f_align_on = st.checkbox("Zgranie średnich: Close > EMA50 > EMA200", value=False)
        f_macd_fresh_on = st.checkbox("MACD świeży: cross w N dniach + histogram rośnie", value=False)
        f_macd_fresh_look = 3
        f_macd_hist_up_days = 1
        f_rsi_up_on = st.checkbox("RSI dziś ≥ RSI wczoraj", value=False)

        st.markdown("---")
        f_minavg_on = st.checkbox("Min. średni wolumen (AvgVolume)", value=False)
        f_minavg_val = st.number_input("— Min AvgVolume", 0, 50_000_000, 1_000_000, step=100_000)
        f_vr_on = st.checkbox("Widełki VolRatio", value=False)
        colV1, colV2 = st.columns(2)
        with colV1:
            f_vr_min = st.number_input("— VR min", 0.0, 10.0, 1.2, step=0.1, format="%.1f")
        with colV2:
            f_vr_max = st.number_input("— VR max (cap)", 0.5, 10.0, 3.0, step=0.1, format="%.1f")

        st.markdown("---")
        f_mcap_on = st.checkbox("Filtr kapitalizacji (USD)", value=False)
        colM1, colM2 = st.columns(2)
        with colM1:
            f_mcap_min = st.number_input("— MC min (USD)", 0.0, 5_000_000_000_000.0, 300_000_000.0, step=50_000_000.0, format="%.0f")
        with colM2:
            f_mcap_max = st.number_input("— MC max (USD)", 0.0, 5_000_000_000_000.0, 2_000_000_000_000.0, step=50_000_000.0, format="%.0f")

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

    with st.expander("Ranking i Tabela — wygląd", expanded=True):
        enable_rank = st.checkbox("Ranking (bez AI)", value=True)
        top_n = st.selectbox("Ile pozycji w TOP", [5, 10], index=1)
        rank_layout = st.selectbox("Układ rankingu", ["Kompakt (6/wiersz)", "Średni (4/wiersz)", "Wąski (3/wiersz)"], index=0)
        fit_cols = st.checkbox("Dopasuj kolumny do szerokości", value=True)
        table_height = st.slider("Wysokość tabeli (px)", 420, 900, 560, step=20)

    run_scan = st.button("🚀 Uruchom skaner", use_container_width=True, type="primary")

# pamiętaj ważne rzeczy w stanie
st.session_state["period"] = period
st.session_state["vol_window"] = vol_window
st.session_state.setdefault("selected_symbol", None)
st.session_state.setdefault("selection_source", None)          # "rank" | "table"
st.session_state.setdefault("last_table_selected", None)       # ostatni ticker wybrany w tabeli

# =========================
# FUNDAMENTY + SHORT (yfinance)
# =========================
@st.cache_data(show_spinner=False, ttl=60*30)
def get_market_cap_fast(ticker: str) -> Optional[float]:
    try:
        fi = yf.Ticker(ticker).fast_info
        mc = fi.get("market_cap")
        return float(mc) if mc is not None else None
    except Exception:
        return None

def nz(x, default=None):
    return default if (x is None or (isinstance(x, float) and pd.isna(x))) else x

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
        # --- SHORT (Yahoo) ---
        "shares_short": g("sharesShort"),
        "short_ratio": g("shortRatio"),
        "short_percent_float": g("shortPercentOfFloat"),
        "shares_float": g("floatShares") or g("sharesFloat"),
    })

    # trendy i rekomendacje (best effort)
    try:
        et = tk.earnings_trend
        if isinstance(et, pd.DataFrame) and not et.empty:
            col = "growth" if "growth" in et else None
            if col:
                data["forward_eps_growth"] = et[col].astype(float).dropna().iloc[-1]
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
        cal = tk.calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            if "Earnings Date" in cal.index:
                ed = cal.loc["Earnings Date"].iloc[0]
                data["earnings_date"] = pd.to_datetime(ed) if pd.notna(ed) else None
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

# =========================
# SKAN
# =========================
if run_scan:
    prev_symbol = st.session_state.get("selected_symbol")
    prev_source = st.session_state.get("selection_source")

    tickers_df = get_tickers(source)
    if tickers_df is None or tickers_df.empty:
        st.error("Brak tickerów do skanowania.")
    else:
        tickers_list = tickers_df["Ticker"].tolist()[:scan_limit]
        progress = st.progress(0); status = st.empty(); results = []
        for i, t in enumerate(tickers_list, start=1):
            status.write(f"⏳ {i}/{len(tickers_list)} – {t}")

            if f_mcap_on:
                mc = get_market_cap_fast(t)
                if mc is None or not (f_mcap_min <= mc <= f_mcap_max):
                    progress.progress(i/len(tickers_list)); continue
            else:
                mc = None

            df = get_stock_df(t, period=period, vol_window=vol_window)
            if df is not None and not df.empty:
                last = df.iloc[-1]

                rsi_ok = pd.notna(last.get("RSI")) and (rsi_min <= float(last.get("RSI")) <= rsi_max)
                price_ok = True
                if require_price_above_ema200:
                    price_ok = pd.notna(last.get("Close")) and pd.notna(last.get("EMA200")) and (float(last.get("Close")) > float(last.get("EMA200")))

                if not (rsi_ok and price_ok):
                    di = "–"; macd_cross = False; vol_ok = False
                else:
                    vol_ok = vol_confirmation(last.get("Volume"), last.get("AvgVolume"), use_volume)
                    macd_cross = macd_bullish_cross_recent(df, macd_lookback)
                    di = score_diamonds(last.get("Close"), last.get("EMA200"), last.get("RSI"),
                                        macd_cross, vol_ok, signal_mode, rsi_min, rsi_max)

                # dodatkowe filtry (opcjonalne)
                gap_ok = True
                if f_gap_on and pd.notna(last.get("GapUpPct")):
                    gap_ok = (float(last.get("GapUpPct")) <= f_gap_max)

                maxdist_ok = True
                if f_maxdist_on and pd.notna(last.get("DistEMA200Pct")):
                    maxdist_ok = (float(last.get("DistEMA200Pct")) <= f_maxdist_pct)

                slope_ok = True
                if f_slope_on and pd.notna(last.get("EMA200_Slope5")):
                    slope_ok = (float(last.get("EMA200_Slope5")) > 0)

                align_ok = True
                if f_align_on and pd.notna(last.get("EMA50")) and pd.notna(last.get("EMA200")) and pd.notna(last.get("Close")):
                    align_ok = (float(last.get("Close")) > float(last.get("EMA50")) > float(last.get("EMA200")))

                macd_fresh_ok = True
                if f_macd_fresh_on:
                    hist = df["MACD_hist"].tail(int(f_macd_hist_up_days)+1).dropna()
                    hist_up = (hist.diff() > 0).tail(int(f_macd_hist_up_days)).all() if len(hist) >= (f_macd_hist_up_days+1) else False
                    macd_recent = macd_bullish_cross_recent(df, int(f_macd_fresh_look))
                    macd_fresh_ok = macd_recent and hist_up

                rsi_up_ok = True
                if f_rsi_up_on and pd.notna(last.get("RSI")) and len(df) >= 2 and pd.notna(df["RSI"].iloc[-2]):
                    rsi_up_ok = bool(last.get("RSI") >= df["RSI"].iloc[-2])

                minavg_ok = True
                if f_minavg_on and pd.notna(last.get("AvgVolume")):
                    minavg_ok = (float(last.get("AvgVolume")) >= float(f_minavg_val))

                vr_ok = True
                vr_val = None
                if pd.notna(last.get("Volume")) and pd.notna(last.get("AvgVolume")) and float(last.get("AvgVolume")) > 0:
                    vr_val = float(last.get("Volume")) / float(last.get("AvgVolume"))
                if f_vr_on and vr_val is not None:
                    vr_ok = (vr_val >= float(f_vr_min)) and (vr_val <= float(f_vr_max))

                minprice_ok = True
                if f_minprice_on and pd.notna(last.get("Close")):
                    minprice_ok = (float(last.get("Close")) >= float(f_minprice_val))

                atr_ok = True
                if f_atr_on and pd.notna(last.get("ATR")) and pd.notna(last.get("Close")) and float(last.get("Close"))>0:
                    atr_pct = float(last.get("ATR")) / float(last.get("Close")) * 100.0
                    atr_ok = (atr_pct <= float(f_atr_max))

                hhhl_ok = True
                if f_hhhl_on and len(df) >= 3 and pd.notna(df["HH3"].iloc[-1]) and pd.notna(df["HL3"].iloc[-1]):
                    hhhl_ok = bool(df["HH3"].iloc[-1] and df["HL3"].iloc[-1])

                resist_ok = True
                if f_resist_on and pd.notna(last.get("RoomToHighPct")):
                    resist_ok = (float(last.get("RoomToHighPct")) >= float(f_resist_min))

                mcap_ok = True if not f_mcap_on else (mc is not None and (f_mcap_min <= mc <= f_mcap_max))

                passed_all_filters = all([
                    gap_ok, maxdist_ok, slope_ok, align_ok, macd_fresh_ok, rsi_up_ok,
                    minavg_ok, vr_ok, minprice_ok, atr_ok, hhhl_ok, resist_ok, mcap_ok
                ])

                results.append({
                    "Ticker": t,
                    "Close": round(float(last.get("Close")), 2) if pd.notna(last.get("Close")) else None,
                    "RSI": round(float(last.get("RSI")), 2) if pd.notna(last.get("RSI")) else None,
                    "EMA50": round(float(last.get("EMA50")), 2) if pd.notna(last.get("EMA50")) else None,
                    "EMA200": round(float(last.get("EMA200")), 2) if pd.notna(last.get("EMA200")) else None,
                    "MACD": round(float(last.get("MACD")), 4) if pd.notna(last.get("MACD")) else None,
                    "MACD_signal": round(float(last.get("MACD_signal")), 4) if pd.notna(last.get("MACD_signal")) else None,
                    "MACD_hist": round(float(last.get("MACD_hist")), 4) if pd.notna(last.get("MACD_hist")) else None,
                    "Volume": int(last.get("Volume")) if pd.notna(last.get("Volume")) else None,
                    "AvgVolume": int(last.get("AvgVolume")) if pd.notna(last.get("AvgVolume")) else None,
                    "VolRatio": vr_val,
                    "GapUpPct": round(float(last.get("GapUpPct")), 2) if pd.notna(last.get("GapUpPct")) else None,
                    "DistEMA200Pct": round(float(last.get("DistEMA200Pct")), 2) if pd.notna(last.get("DistEMA200Pct")) else None,
                    "ATR": round(float(last.get("ATR")), 4) if pd.notna(last.get("ATR")) else None,
                    "RoomToHighPct": round(float(last.get("RoomToHighPct")), 2) if pd.notna(last.get("RoomToHighPct")) else None,
                    "MarketCap": float(mc) if mc is not None else None,
                    "FiltersOK": passed_all_filters,
                    "Sygnał": di
                })
            progress.progress(i/len(tickers_list))
        status.write("✅ Zakończono skan.")
        st.session_state.scan_results = pd.DataFrame(results)

        # przywróć poprzedni wybór, jeśli nadal istnieje
        if prev_symbol and prev_symbol in set(st.session_state.scan_results["Ticker"]):
            st.session_state["selected_symbol"] = prev_symbol
            st.session_state["selection_source"] = prev_source
        else:
            if not prev_symbol:
                st.session_state["selected_symbol"] = None
                st.session_state["selection_source"] = None
        # nie resetuj last_table_selected — to pomaga wykryć "nowy" klik

# =========================
# RANKING (bez AI)
# =========================
def _safe(val, default=None):
    return default if val is None or (isinstance(val, float) and math.isnan(val)) else val

def rank_score_row(row, rsi_min: int, rsi_max: int) -> float:
    close = _safe(row.get("Close")); ema200 = _safe(row.get("EMA200"))
    rsi = _safe(row.get("RSI")); macd = _safe(row.get("MACD")); macd_sig = _safe(row.get("MACD_signal"))
    volr = _safe(row.get("VolRatio")); avgv = _safe(row.get("AvgVolume"))
    dist_score = 0.0
    if close and ema200 and ema200>0:
        dist = close/ema200 - 1.0
        dist_score = max(0.0, min(dist, 0.10)) / 0.10
    rsi_score = 0.0
    if rsi is not None:
        mid = (rsi_min + rsi_max)/2.0
        half_range = max(1.0, (rsi_max - rsi_min)/2.0)
        rsi_score = 1.0 - min(abs(rsi-mid)/half_range, 1.0)
    macd_score = 0.0
    if macd is not None and macd_sig is not None:
        diff = macd - macd_sig
        macd_score = max(0.0, min(diff, 0.50))/0.50
    volr_score = 0.0
    if volr is not None:
        volr_score = max(0.0, min(volr, 2.0))/2.0
    liq_score = 0.0
    if isinstance(avgv,(int,float)) and avgv:
        if avgv >= 5_000_000: liq_score = 1.0
        elif avgv >= 2_000_000: liq_score = 0.7
        elif avgv >= 1_000_000: liq_score = 0.5
        elif avgv > 0: liq_score = 0.2
    return round((0.30*dist_score + 0.25*rsi_score + 0.25*macd_score + 0.15*volr_score + 0.05*liq_score)*100.0, 1)

def build_ranking(df: pd.DataFrame, rsi_min: int, rsi_max: int, top_n: int) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame(columns=["Ticker","Score"])
    base = df.copy()
    base = base[(base["Sygnał"]=="💎💎💎") & (base["FiltersOK"]==True)]
    if base.empty: return pd.DataFrame(columns=["Ticker","Score"])
    base["Score"] = base.apply(lambda r: rank_score_row(r, rsi_min, rsi_max), axis=1)
    def rsi_dev(r):
        rv = _safe(r.get("RSI"))
        if rv is None: return 999.0
        mid = (rsi_min+rsi_max)/2.0
        return abs(rv-mid)
    base["_dev"] = base.apply(rsi_dev, axis=1)
    base = base.sort_values(["Score","AvgVolume","_dev","Ticker"], ascending=[False, False, True, True]).drop(columns=["_dev"])
    return base[["Ticker","Score"]].head(top_n).reset_index(drop=True)

# =========================
# PRO: wejścia + podsumowania (w tym short z Yahoo)
# =========================
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
    close = row.get("Close"); rsi=row.get("RSI"); ema=row.get("EMA200"); ema50=row.get("EMA50")
    vr=row.get("VolRatio"); macd=row.get("MACD"); sig=row.get("MACD_signal")
    dist_pct = _pct_from(close, ema)
    macd_delta = (macd - sig) if pd.notna(macd) and pd.notna(sig) else None
    df_full = get_stock_df(sym, period=st.session_state.get("period","1y"), vol_window=st.session_state.get("vol_window",20))
    entry_break, entry_pull = compute_entries(df_full)
    fn = fetch_fundamentals(sym)
    cur = fn.get("currency") or "USD"

    cap_txt = _fmt_money(fn.get("market_cap"), cur)
    title_bits = [sym, fn.get("long_name") or "", f"• {fn.get('industry') or '—'}", f"• {fn.get('country') or '—'}", f"• MC: {cap_txt}"]
    st.markdown("**" + "  ".join([x for x in title_bits if x]) + f"  •  waluta: {cur}**")

    snap = []
    if pd.notna(close): snap.append(f"Close **${close:.2f}**")
    if pd.notna(rsi):   snap.append(f"RSI **{rsi:.1f}** (zakres {rsi_min}–{rsi_max})")
    if dist_pct is not None: snap.append(f"vs EMA200 **{dist_pct:.2f}%**")
    if pd.notna(vr):    snap.append(f"VR **{vr:.2f}**")
    if macd_delta is not None: snap.append(f"ΔMACD **{macd_delta:.3f}**")
    st.write(" · ".join(snap))

    reco = None
    if dist_pct is not None and pd.notna(rsi):
        if dist_pct > 8 or rsi >= (rsi_max - 1): reco = "Preferuj **pullback** (mniejszy pościg, lepszy RR)."
        else: reco = "Możliwy **breakout** nad lokalnym oporem."
    entry_lines = []
    if entry_break is not None: entry_lines.append(f"**Wejście (breakout):** ${entry_break:.2f}  _(max(H20, Close) + 0.10×ATR)_")
    if entry_pull  is not None: entry_lines.append(f"**Wejście (pullback):** ${entry_pull:.2f}  _(EMA bazowa + 0.10×ATR)_")
    if entry_lines or reco:
        st.markdown("**🎯 Proponowane wejścia**")
        for ln in entry_lines: st.write(ln)
        if reco: st.write(reco)

    with st.expander("📊 Wycena i jakość", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"P/E (TTM): **{nz(fn.get('trailing_pe'),'N/A')}**")
            st.write(f"Forward P/E: **{nz(fn.get('forward_pe'),'N/A')}**")
            st.write(f"PEG: **{nz(fn.get('peg_ratio'),'N/A')}**")
        with col2:
            st.write(f"P/S: **{nz(fn.get('price_to_sales'),'N/A')}**")
            st.write(f"P/B: **{nz(fn.get('price_to_book'),'N/A')}**")
            ev = nz(fn.get("enterprise_value")); ebitda = nz(fn.get("ebitda"))
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

    with st.expander("📈 Zwroty i ryzyko", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"1M: **{fn.get('ret_1m'):.1f}%**" if fn.get("ret_1m") is not None else "1M: **N/A**")
            st.write(f"3M: **{fn.get('ret_3m'):.1f}%**" if fn.get("ret_3m") is not None else "3M: **N/A**")
        with col2:
            st.write(f"6M: **{fn.get('ret_6m'):.1f}%**" if fn.get("ret_6m") is not None else "6M: **N/A**")
            st.write(f"1Y: **{fn.get('ret_1y'):.1f}%**" if fn.get("ret_1y") is not None else "1Y: **N/A**")
        with col3:
            st.write(f"Max DD (1Y): **{fn.get('max_dd_1y'):.1f}%**" if fn.get("max_dd_1y") is not None else "Max DD (1Y): **N/A**")

    with st.expander("🗓️ Wydarzenia i dywidendy", expanded=False):
        earn = fn.get("earnings_date")
        if earn is not None:
            try:
                st.write(f"Earnings: **{pd.to_datetime(earn).date()}**")
            except Exception:
                st.write("Earnings: **N/A**")
        else:
            st.write("Earnings: **N/A**")
        div_y = fn.get("div_yield")
        if div_y is not None:
            st.write(f"Dividend (TTM): **{_fmt_money(fn.get('div_ttm'), cur)}**  •  Yield: **{div_y*100:.2f}%**")
        else:
            st.write("Dividend: **N/A**")

    # --- NEW: Short interest (Yahoo) ---
    with st.expander("📉 Short interest (Yahoo)", expanded=False):
        ss = fn.get("shares_short")
        spf = fn.get("short_percent_float")
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

    reasons = []
    if pd.notna(close) and pd.notna(ema) and close>ema: reasons.append("✅ **Trend D1:** cena > EMA200.")
    if macd_delta is not None and macd_delta>=0: reasons.append("✅ **Momentum:** MACD > signal.")
    if pd.notna(rsi) and rsi_min <= rsi <= rsi_max: reasons.append(f"✅ **RSI** w zakresie ({rsi_min}–{rsi_max}).")
    risks = []
    if dist_pct is not None and dist_pct>15: risks.append("⚠️ Spory dystans nad EMA200 (>15%).")

    if reasons:
        st.markdown("**Dlaczego na liście:**")
        for r in reasons: st.write("- " + r)
    if risks:
        st.markdown("**Ryzyka / na co uważać:**")
        for r in risks: st.write("- " + r)

# =========================
# WIDOK + TABELA + RANKING + WYKRESY
# =========================
if "scan_results" in st.session_state and not st.session_state.scan_results.empty:
    df_res = st.session_state.scan_results.copy()

    # Klasy wolumenu
    ratio_series = pd.to_numeric(df_res["VolRatio"], errors="coerce")
    if ratio_series.notna().sum() >= 5:
        qtiles = ratio_series.rank(pct=True)
        df_res["VolRankPct"] = qtiles
        df_res["Wolumen"] = df_res["VolRankPct"].apply(volume_label_from_ratio_qtile)
    else:
        def _fallback(row):
            v, a = row.get("Volume"), row.get("AvgVolume")
            if pd.isna(v) or pd.isna(a) or a <= 0: return "—"
            r = float(v) / float(a)
            if r > 2.0: return "Bardzo wysoki"
            if r > 1.5: return "Wysoki"
            if r > 1.0: return "Normalny"
            if r > 0.5: return "Niski"
            return "Bardzo niski"
        df_res["Wolumen"] = df_res.apply(_fallback, axis=1)

    # Filtr wolumenu (widok)
    if vol_filter != "Wszystkie":
        df_res = df_res[df_res["Wolumen"] == vol_filter]

    # ===== RANKING =====
    if enable_rank:
        rank_df = build_ranking(st.session_state.scan_results, rsi_min, rsi_max, top_n)
        st.session_state.rank_df = rank_df
        st.markdown(f"### 🔝 Proponowane (ranking 1–{len(rank_df) if not rank_df.empty else top_n})")
        if rank_df.empty:
            st.info("Brak kandydatów (💎💎💎 + aktywne filtry). Zmień parametry.")
        else:
            per_row = 6 if "Kompakt (6/wiersz)" in rank_layout else (4 if "Średni (4/wiersz)" in rank_layout else 3)
            for start in range(0, len(rank_df), per_row):
                row_slice = rank_df.iloc[start:start+per_row]
                cols = st.columns(len(row_slice))
                for col, (_, rr) in zip(cols, row_slice.iterrows()):
                    with col:
                        label = f"{start + rr.name + 1}. {rr['Ticker']} · {rr['Score']:.1f}"
                        if st.button(label, key=f"chip_{rr['Ticker']}", use_container_width=True):
                            st.session_state["selected_symbol"] = rr["Ticker"]
                            st.session_state["selection_source"] = "rank"

    # ===== TABELA =====
    view_cols = ["Ticker", "Sygnał", "Close", "RSI", "EMA200", "Wolumen", "DistEMA200Pct", "VolRatio", "MarketCap"]

    # „Pokaż tylko 💎💎💎” – stosujemy TUŻ przed tabelą (widok)
    if only_three:
        df_res = df_res[df_res["Sygnał"] == "💎💎💎"]

    # porządek i sort
    df_res = df_res[df_res["Sygnał"].isin(["💎💎", "💎💎💎", "–"])].reset_index(drop=True)
    def _rank(di: str) -> int: return 2 if di == "💎💎💎" else (1 if di == "💎💎" else 0)
    df_res["Rank"] = df_res["Sygnał"].apply(_rank)
    df_res = df_res.sort_values(["Rank","Ticker"], ascending=[False, True]).drop(columns=["Rank"]).reset_index(drop=True)

    st.subheader("📋 Wyniki skanera")
    st.write(
        f"<span class='pill'>Wyników: <b>{len(df_res)}</b></span>"
        f"<span class='pill'>RSI (twardo): <b>{rsi_min}–{rsi_max}</b></span>"
        f"<span class='pill'>Tryb: <b>{signal_mode}</b></span>"
        f"<span class='pill'>Okres: <b>{period}</b></span>"
        f"<span class='pill'>Close>EMA200: <b>{'ON' if require_price_above_ema200 else 'OFF'}</b></span>",
        unsafe_allow_html=True
    )

    grid_key = "scan_table_aggrid"

    gb = GridOptionsBuilder.from_dataframe(df_res[view_cols])
    gb.configure_selection('single', use_checkbox=False)
    gb.configure_grid_options(rowHeight=36, suppressPaginationPanel=True, domLayout='normal')
    grid_options = gb.build()

    grid_response = AgGrid(
        df_res[view_cols],
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        theme='alpine',
        height=table_height,
        fit_columns_on_grid_load=bool(fit_cols),
        key=grid_key,
    )

    # Odczyt selekcji z tabeli: PRZEKAŻEMY GŁOS TYLKO PRZY NOWEJ SELEKCJI
    current_table_select = None
    if isinstance(grid_response, dict):
        sel = grid_response.get("selected_rows") or grid_response.get("selectedRows") or []
        if sel:
            current_table_select = sel[0]["Ticker"]
    elif hasattr(grid_response, "selected_rows"):
        sel = getattr(grid_response, "selected_rows", []) or []
        if sel:
            current_table_select = sel[0]["Ticker"]

    # przełącz na tabelę TYLKO, gdy wybór się zmienił
    if current_table_select and current_table_select != st.session_state.get("last_table_selected"):
        st.session_state["last_table_selected"] = current_table_select
        st.session_state["selected_symbol"] = current_table_select
        st.session_state["selection_source"] = "table"
    # w innym wypadku – nie nadpisuj wyboru z rankingu

    # -------- WYKRESY + PODSUMOWANIE PRO --------
    sym = st.session_state.get("selected_symbol")
    if sym:
        st.markdown("---")
        st.subheader(f"📈 {sym} — podgląd wykresów")

        with st.spinner(f"Ładuję wykresy dla {sym}…"):
            df_sel = get_stock_df(sym, period=st.session_state.get("period","1y"), vol_window=st.session_state.get("vol_window",20))

        if df_sel is None or df_sel.empty:
            st.error("Nie udało się pobrać danych wykresu.")
        else:
            last = df_sel.iloc[-1]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Kurs (Close)", f"{last.get('Close'):.2f}" if pd.notna(last.get("Close")) else "—")
            m2.metric("RSI", f"{last.get('RSI'):.2f}" if pd.notna(last.get("RSI")) else "—")
            dist = (last.get("Close")/last.get("EMA200")-1)*100 if pd.notna(last.get("Close")) and pd.notna(last.get("EMA200")) else None
            m3.metric("Dystans do EMA200", f"{dist:.2f}%" if dist is not None else "—")
            macd_cross_here = macd_bullish_cross_recent(df_sel, macd_lookback)
            vol_ok_here = vol_confirmation(last.get("Volume"), last.get("AvgVolume"), use_volume)
            di_here = score_diamonds(last.get("Close"), last.get("EMA200"), last.get("RSI"),
                                     macd_cross_here, vol_ok_here, signal_mode, rsi_min, rsi_max)
            m4.metric("Sygnał", di_here)

            st.plotly_chart(plot_candles_with_ema(df_sel, sym), use_container_width=True)
            st.plotly_chart(plot_rsi(df_sel, sym), use_container_width=True)
            st.plotly_chart(plot_macd(df_sel, sym), use_container_width=True)

            st.markdown("### 🧭 Podsumowanie PRO")
            try:
                render_summary_pro(sym, st.session_state.scan_results, rsi_min, rsi_max)
            except Exception as e:
                st.warning(f"Nie udało się zbudować Podsumowania PRO: {e}")

else:
    st.info("Otwórz panel **Skaner** po lewej i kliknij **🚀 Uruchom skaner**.")
