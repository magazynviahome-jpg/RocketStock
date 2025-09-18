import io
import math
from typing import Optional

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import ta
import plotly.graph_objects as go

# Awaryjnie importujemy AgGrid – jeśli brak/st-aggrid się wywala, dalej zadziała fallback
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    AGGRID_AVAILABLE = True
except Exception:
    AGGRID_AVAILABLE = False

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
    .chip{ background:#f5f3ff;border:1px solid #e9d5ff;color:#4c1d95; padding:6px 10px; border-radius:999px;
           font-weight:600; font-size:13px; white-space:nowrap; display:inline-block; width:100%; text-align:center; }
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
    fig.add_trace(go.Bar(x=d.index, y=d["MACD_hist"].tail(bars), name="Histogram", opacity=0.3))
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
        vol_window_opt = st.selectbox("Średni wolumen (okno)", ["MA20", "MA50"], index=0)
        vol_window = 20 if vol_window_opt == "MA20" else 50

        only_three = st.checkbox("Pokaż tylko 💎💎💎", value=False)
        require_price_above_ema200 = st.checkbox("Wymagaj Close > EMA200", value=True)

        vol_filter_opt = st.selectbox("Filtr wolumenu", ["Wszystkie", "Bardzo wysoki", "Wysoki", "Normalny", "Niski", "Bardzo niski"], index=0)
        scan_limit = st.slider("Limit skanowania (dla bezpieczeństwa)", 50, 3500, 300, step=50)

        st.markdown("---")
        source = st.selectbox("Źródło listy NASDAQ", ["Auto (online, fallback do CSV)", "Tylko CSV w repo"], index=0)
        period = st.selectbox("Okres danych", ["6mo", "1y", "2y"], index=1)

    with st.expander("Dodatkowe filtry (opcjonalne)", expanded=False):
        f_maxdist_on = st.checkbox("Max dystans do EMA200", value=True)
        f_maxdist_pct = st.slider("— Maks. % nad EMA200", 5, 30, 15) if f_maxdist_on else 15
        f_slope_on = st.checkbox("EMA200 rośnie (nachylenie > 0)", value=True)
        f_align_on = st.checkbox("Zgranie średnich: Close > EMA50 > EMA200", value=True)
        f_macd_fresh_on = st.checkbox("MACD świeży: cross w N dniach + histogram rośnie", value=False)
        colA, colB = st.columns(2)
        with colA:
            f_macd_fresh_look = st.number_input("— N dni (cross)", 1, 10, 3)
        with colB:
            f_macd_hist_up_days = st.number_input("— Min. dni wzrostu histogramu", 1, 5, 1)
        f_rsi_up_on = st.checkbox("RSI dziś ≥ RSI wczoraj", value=False)

        st.markdown("---")
        f_minavg_on = st.checkbox("Min. średni wolumen (AvgVolume)", value=True)
        f_minavg_val = st.number_input("— Min AvgVolume", 0, 50_000_000, 1_000_000, step=100_000)
        f_vr_on = st.checkbox("Widełki VolRatio", value=True)
        colV1, colV2 = st.columns(2)
        with colV1:
            f_vr_min = st.number_input("— VR min", 0.0, 10.0, 1.2, step=0.1, format="%.1f")
        with colV2:
            f_vr_max = st.number_input("— VR max (cap)", 0.5, 10.0, 3.0, step=0.1, format="%.1f")

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
        fit_cols = st.checkbox("Dopasuj kolumny do szerokości (AgGrid)", value=True)
        table_height = st.slider("Wysokość tabeli (px)", 420, 900, 560, step=20)
        simple_table_mode = st.checkbox("Tryb awaryjny tabeli (bez AgGrid)", value=not AGGRID_AVAILABLE)

    run_scan = st.button("🚀 Uruchom skaner", use_container_width=True, type="primary")

# =========================
# RANKING + PODSUMOWANIE
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

def summarize_row_rich(row, rsi_min: int, rsi_max: int) -> str:
    tck = row.get("Ticker","?")
    di  = row.get("Sygnał","–")
    close = row.get("Close"); rsi=row.get("RSI"); ema=row.get("EMA200")
    vr=row.get("VolRatio"); av=row.get("AvgVolume"); macd=row.get("MACD"); sig=row.get("MACD_signal")
    dist_txt = "—"
    if pd.notna(close) and pd.notna(ema) and ema:
        dist = (close/ema - 1.0)*100
        dist_txt = f"{dist:.2f}%"
    vr_txt = f"{vr:.2f}" if pd.notna(vr) else "—"
    rsi_txt = f"{rsi:.1f}" if pd.notna(rsi) else "—"
    macd_delta = (macd - sig) if pd.notna(macd) and pd.notna(sig) else None
    macd_txt = f"{macd_delta:.3f}" if macd_delta is not None else "—"

    reasons = []
    if pd.notna(close) and pd.notna(ema) and close>ema: reasons.append("✅ **Trend:** cena > EMA200.")
    elif pd.notna(close) and pd.notna(ema): reasons.append("⚠️ **Trend:** cena ≤ EMA200.")
    if macd_delta is not None and macd_delta>=0: reasons.append("✅ **Momentum:** MACD > signal.")
    elif macd_delta is not None: reasons.append("⚠️ **Momentum:** MACD < signal.")
    if pd.notna(rsi): reasons.append(f"✅ **RSI:** {rsi:.1f} w Twoim zakresie ({rsi_min}–{rsi_max}).")
    if pd.notna(vr):
        lab = "bardzo wysoki" if vr>=2.0 else ("wysoki" if vr>=1.5 else ("normalny" if vr>=1.0 else "niski"))
        reasons.append(f"✅ **Wolumen:** {lab} (VR={vr:.2f}).")
    if pd.notna(row.get("DistEMA200Pct")) and row["DistEMA200Pct"]>0:
        reasons.append(f"📏 **Odległość od EMA200:** {row['DistEMA200Pct']:.2f}%.")

    risks = []
    if pd.notna(rsi) and rsi >= (rsi_max - 1):
        risks.append("⚠️ RSI blisko górnej krawędzi zakresu → możliwy oddech.")
    if pd.notna(row.get("DistEMA200Pct")) and row["DistEMA200Pct"]>15:
        risks.append("⚠️ Spory dystans nad EMA200 (>15%) — ryzyko wyciszenia.")

    levels = []
    ema50 = row.get("EMA50")
    if pd.notna(ema50): levels.append(f"EMA50: **{ema50:.2f}**")
    if pd.notna(ema):   levels.append(f"EMA200: **{ema:.2f}**")
    if pd.notna(row.get("RoomToHighPct")):
        levels.append(f"Od 3m High: **{row['RoomToHighPct']:.2f}%**")

    plan = "🎯 Wejście po kontynuacji powyżej ostatniego high; inwalidacja: MACD < signal lub powrót < EMA50."

    md = []
    md.append(f"**{tck} — {di}**  |  Close **{close:.2f}**  |  RSI **{rsi_txt}**  |  +**{dist_txt}** vs EMA200  |  VR **{vr_txt}**  |  ΔMACD **{macd_txt}**")
    if reasons:
        md.append("\n**Dlaczego na liście:**")
        md += [f"- {x}" for x in reasons]
    if risks:
        md.append("\n**Ryzyka / na co uważać:**")
        md += [f"- {x}" for x in risks]
    if levels:
        md.append("\n**Kluczowe poziomy:** " + "  ·  ".join(levels))
    md.append("\n**Plan:** " + plan)
    return "\n".join(md)

# =========================
# SKAN
# =========================
if run_scan:
    st.session_state.selected_symbol = None
    tickers_df = get_tickers(source)
    if tickers_df is None or tickers_df.empty:
        st.error("Brak tickerów do skanowania.")
    else:
        tickers_list = tickers_df["Ticker"].tolist()[:scan_limit]
        progress = st.progress(0); status = st.empty(); results = []
        for i, t in enumerate(tickers_list, start=1):
            status.write(f"⏳ {i}/{len(tickers_list)} – {t}")
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

                # opcjonalne filtry
                gap_ok = not f_gap_on or (pd.notna(last.get("GapUpPct")) and float(last.get("GapUpPct")) <= f_gap_max)
                maxdist_ok = (not f_maxdist_on) or (pd.notna(last.get("DistEMA200Pct")) and float(last.get("DistEMA200Pct")) <= f_maxdist_pct)
                slope_ok = (not f_slope_on) or (pd.notna(last.get("EMA200_Slope5")) and float(last.get("EMA200_Slope5")) > 0)
                align_ok = (not f_align_on) or (pd.notna(last.get("EMA50")) and pd.notna(last.get("EMA200")) and pd.notna(last.get("Close")) and (float(last.get("Close")) > float(last.get("EMA50")) > float(last.get("EMA200"))))
                macd_fresh_ok = True
                if f_macd_fresh_on:
                    hist = df["MACD_hist"].tail(int(f_macd_hist_up_days)+1).dropna()
                    hist_up = (hist.diff() > 0).tail(int(f_macd_hist_up_days)).all() if len(hist) >= (f_macd_hist_up_days+1) else False
                    macd_recent = macd_bullish_cross_recent(df, int(f_macd_fresh_look))
                    macd_fresh_ok = macd_recent and hist_up
                rsi_up_ok = (not f_rsi_up_on) or (pd.notna(last.get("RSI")) and len(df)>=2 and pd.notna(df["RSI"].iloc[-2]) and last.get("RSI")>=df["RSI"].iloc[-2])
                minavg_ok = (not f_minavg_on) or (pd.notna(last.get("AvgVolume")) and float(last.get("AvgVolume")) >= float(f_minavg_val))
                vr_val = None
                if pd.notna(last.get("Volume")) and pd.notna(last.get("AvgVolume")) and float(last.get("AvgVolume"))>0:
                    vr_val = float(last.get("Volume"))/float(last.get("AvgVolume"))
                vr_ok = (not f_vr_on) or (vr_val is not None and (vr_val >= float(f_vr_min)) and (vr_val <= float(f_vr_max)))
                minprice_ok = (not f_minprice_on) or (pd.notna(last.get("Close")) and float(last.get("Close")) >= float(f_minprice_val))
                atr_ok = True
                if f_atr_on and pd.notna(last.get("ATR")) and pd.notna(last.get("Close")) and float(last.get("Close"))>0:
                    atr_ok = (float(last.get("ATR"))/float(last.get("Close"))*100.0 <= float(f_atr_max))
                hhhl_ok = (not f_hhhl_on) or (len(df)>=3 and pd.notna(df["HH3"].iloc[-1]) and pd.notna(df["HL3"].iloc[-1]) and bool(df["HH3"].iloc[-1] and df["HL3"].iloc[-1]))
                resist_ok = (not f_resist_on) or (pd.notna(last.get("RoomToHighPct")) and float(last.get("RoomToHighPct")) >= float(f_resist_min))

                passed_all_filters = all([gap_ok, maxdist_ok, slope_ok, align_ok, macd_fresh_ok, rsi_up_ok,
                                          minavg_ok, vr_ok, minprice_ok, atr_ok, hhhl_ok, resist_ok])

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
                    "FiltersOK": passed_all_filters,
                    "Sygnał": di
                })
            progress.progress(i/len(tickers_list))
        status.write("✅ Zakończono skan.")
        st.session_state.scan_results = pd.DataFrame(results)

# =========================
# WIDOK + TABELA + RANKING + WYKRESY
# =========================
view_cols = ["Ticker", "Sygnał", "Close", "RSI", "EMA200", "Wolumen", "DistEMA200Pct", "VolRatio"]
df_src = st.session_state.get("scan_results", pd.DataFrame())

# Przygotuj DF widoku
if df_src is None or df_src.empty:
    df_view = pd.DataFrame(columns=view_cols)
else:
    df_res = df_src.copy()

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

    # Filtry widoku
    if vol_filter_opt != "Wszystkie":
        df_res = df_res[df_res["Wolumen"] == vol_filter_opt]
    if only_three:
        df_res = df_res[df_res["Sygnał"] == "💎💎💎"]

    # Sort
    def _rank(di: str) -> int: return 2 if di == "💎💎💎" else (1 if di == "💎💎" else 0)
    if not df_res.empty:
        df_res["Rank"] = df_res["Sygnał"].apply(_rank)
        df_res = df_res.sort_values(["Rank","Ticker"], ascending=[False, True]).drop(columns=["Rank"]).reset_index(drop=True)

    for c in view_cols:
        if c not in df_res.columns: df_res[c] = None
    df_view = df_res[view_cols]

# Nagłówek
st.subheader("📋 Wyniki skanera")
st.write(
    f"<span class='pill'>Wyników: <b>{len(df_view)}</b></span>"
    f"<span class='pill'>RSI (twardo): <b>{rsi_min}–{rsi_max}</b></span>"
    f"<span class='pill'>Tryb: <b>{signal_mode}</b></span>"
    f"<span class='pill'>Okres: <b>{period}</b></span>"
    f"<span class='pill'>Close>EMA200: <b>{'ON' if require_price_above_ema200 else 'OFF'}</b></span>",
    unsafe_allow_html=True
)

# RANKING (poziomo)
if enable_rank and not df_src.empty:
    rank_df = build_ranking(df_src, rsi_min, rsi_max, top_n)
    st.markdown(f"### 🔝 Proponowane (ranking 1–{len(rank_df) if not rank_df.empty else top_n})")
    if rank_df.empty:
        st.info("Brak kandydatów (💎💎💎 + aktywne filtry). Zmień parametry.")
    else:
        per_row = 6 if "Kompakt" in rank_layout else (4 if "Średni" in rank_layout else 3)
        for start in range(0, len(rank_df), per_row):
            row_slice = rank_df.iloc[start:start+per_row]
            cols = st.columns(len(row_slice))
            for col, (_, rr) in zip(cols, row_slice.iterrows()):
                with col:
                    label = f"{start + rr.name + 1}. {rr['Ticker']} · {rr['Score']:.1f}"
                    if st.button(label, key=f"chip_{start}_{rr['Ticker']}", use_container_width=True):
                        st.session_state.selected_symbol = rr["Ticker"]

# ====== TABELA: AgGrid -> fallback ======
def render_table(df_view: pd.DataFrame):
    if simple_table_mode or not AGGRID_AVAILABLE:
        st.dataframe(df_view, height=table_height, use_container_width=True)
        return

    try:
        from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
        gb = GridOptionsBuilder.from_dataframe(df_view)
        gb.configure_selection('single', use_checkbox=False)
        gb.configure_grid_options(rowHeight=36, suppressPaginationPanel=True, domLayout='normal')
        grid_options = gb.build()
        # usuwamy potencjalny śmieć z opcji
        grid_options.pop("preSelectAllRows", None)

        grid_response = AgGrid(
            df_view,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            theme='alpine',
            height=int(table_height),
            fit_columns_on_grid_load=bool(fit_cols),
            key="scan_table",
            enable_enterprise_modules=False,   # wymuś community (nie każdy build to ma, ale try/except nas ochroni)
        )

        # wybór
        selected_from_table = None
        if isinstance(grid_response, dict):
            sel = grid_response.get("selected_rows") or grid_response.get("selectedRows") or []
            if sel: selected_from_table = sel[0].get("Ticker")
        elif hasattr(grid_response, "selected_rows"):
            sel = getattr(grid_response, "selected_rows", []) or []
            if sel: selected_from_table = sel[0].get("Ticker")
        if selected_from_table:
            st.session_state.selected_symbol = selected_from_table

    except Exception as e:
        st.warning(f"AgGrid ma problem („{e}”). Przełączam na tryb awaryjny.")
        st.dataframe(df_view, height=table_height, use_container_width=True)

render_table(df_view)

# -------- WYKRESY + PODSUMOWANIE --------
sym = st.session_state.get("selected_symbol")
if sym:
    st.markdown("---")
    st.subheader(f"📈 {sym} — podgląd wykresów")

    df_sel = get_stock_df(sym, period=period, vol_window=vol_window)
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

        st.markdown("### 🧭 Podsumowanie")
        base_row = df_src[df_src["Ticker"] == sym]
        if not base_row.empty:
            st.markdown(summarize_row_rich(base_row.iloc[0], rsi_min, rsi_max))
        else:
            st.info("Brak danych do podsumowania dla wybranej spółki.")
else:
    st.info("Otwórz panel **Skaner** po lewej i kliknij **🚀 Uruchom skaner**.")
