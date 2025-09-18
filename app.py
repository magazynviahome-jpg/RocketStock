import io
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import ta
import plotly.graph_objects as go

# =========================
# USTAWIENIA I WYGLĄD
# =========================
st.set_page_config(
    page_title="🚀 RocketStock – NASDAQ Scanner",
    page_icon="🚀",
    layout="wide"
)

st.markdown(
    """
    <style>
    .muted {color:#6b7280;}
    .pill {padding:2px 8px;border-radius:999px;background:#f3f4f6;margin-right:6px;}
    .grid-buttons .stButton>button{height:54px;font-size:14px;line-height:1.1;padding:8px 10px;margin-bottom:6px}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚀 RocketStock")
st.caption("Skaner NASDAQ z RSI 30–50, EMA200, MACD i wolumenem — diamentowy scoring 💎. (Zmiany tylko w UX)")

# =========================
# STAŁE / CACHE
# =========================
NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"

@st.cache_data(show_spinner=False, ttl=60*30)
def fetch_nasdaq_online() -> pd.DataFrame:
    """Pobiera listę NASDAQ (txt z separatorami |) i zwraca DataFrame z kolumną 'Ticker'."""
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
    """Wczytuje tickery z pliku CSV w repo (kolumna 'Ticker')."""
    df = pd.read_csv(path)
    if "Ticker" not in df.columns:
        raise ValueError("Plik CSV musi mieć kolumnę 'Ticker'.")
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df = df[df["Ticker"] != ""].drop_duplicates()
    return df

def get_tickers(source: str) -> pd.DataFrame:
    """Źródło: Online (fallback do CSV) lub tylko CSV."""
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

def compute_indicators(df: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    """LOGIKA BEZ ZMIAN: RSI, EMA200, MACD, średni wolumen."""
    if "Close" not in df.columns or "Volume" not in df.columns:
        raise ValueError("Dane muszą zawierać 'Close' i 'Volume'.")
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["EMA200"] = ta.trend.EMAIndicator(df["Close"], window=200).ema_indicator()
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["AvgVolume"] = df["Volume"].rolling(vol_window).mean()
    return df

def macd_bullish_cross_recent(df: pd.DataFrame, lookback: int) -> bool:
    macd = df["MACD"]
    sig  = df["MACD_signal"]
    cross_up = (macd.shift(1) <= sig.shift(1)) & (macd > sig)
    return bool(cross_up.tail(lookback).any())

@st.cache_data(show_spinner=False)
def get_stock_df(ticker: str, period: str, vol_window: int) -> pd.DataFrame | None:
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    df = flatten_columns(df)
    try:
        df = compute_indicators(df, vol_window)
        return df
    except Exception:
        return None

# --------- SCORING (bez zmian) ---------
def score_diamonds(price, ema200, rsi, macd_cross, vol_ok, mode: str, rsi_min: int, rsi_max: int) -> str:
    pts = 0
    if mode == "Konserwatywny":
        if pd.notna(price) and pd.notna(ema200) and price > ema200: pts += 1
        if pd.notna(rsi) and rsi_min <= rsi <= rsi_max: pts += 1
        if macd_cross: pts += 1
        if vol_ok: pts += 1
        if pts >= 4: return "💎💎💎"
        if pts == 3: return "💎💎"
        if pts == 2: return "💎"
        return "–"
    elif mode == "Umiarkowany":
        if pd.notna(price) and pd.notna(ema200) and (price >= ema200*0.995): pts += 1
        if pd.notna(rsi) and (rsi_min-2) <= rsi <= (rsi_max+2): pts += 1
        if macd_cross: pts += 1
        if vol_ok: pts += 1
        if pts >= 3: return "💎💎💎"
        if pts == 2: return "💎💎"
        if pts == 1: return "💎"
        return "–"
    else:  # Agresywny
        if pd.notna(price) and pd.notna(ema200) and (price >= ema200*0.98): pts += 1
        if pd.notna(rsi) and (rsi_min-5) <= rsi <= (rsi_max+5): pts += 1
        if macd_cross: pts += 1
        if vol_ok: pts += 1
        if pts >= 3: return "💎💎💎"
        if pts == 2: return "💎💎"
        if pts == 1: return "💎"
        return "–"

def vol_confirmation(volume, avg_volume, require: bool) -> bool:
    if not require:
        return True
    if pd.isna(volume) or pd.isna(avg_volume):
        return False
    return volume > avg_volume

def diamond_rank(di: str) -> int:
    return 0 if di == "–" else len(di)

# =========================
# WYKRESY (Plotly) — tylko prezentacja, logika bez zmian
# =========================
def plot_candles_with_ema(df: pd.DataFrame, ticker: str, bars: int = 180):
    d = df.tail(bars)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name=ticker, showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=d.index, y=d["EMA200"], name="EMA200", mode="lines"
    ))
    fig.update_layout(
        height=460, margin=dict(l=10, r=10, t=40, b=10),
        title=f"📊 {ticker} — Świece + EMA200", xaxis_rangeslider_visible=False
    )
    return fig

def plot_rsi(df: pd.DataFrame, ticker: str, bars: int = 180):
    d = df.tail(bars)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], name="RSI(14)", mode="lines"))
    # poziomy referencyjne
    fig.add_hline(y=30, line_dash="dash")
    fig.add_hline(y=70, line_dash="dash")
    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=40, b=10),
        title=f"📈 {ticker} — RSI(14)", yaxis=dict(range=[0, 100])
    )
    return fig

# =========================
# SIDEBAR — „SKANER” + URUCHOM
# =========================
with st.sidebar:
    with st.expander("Skaner", expanded=True):
        # parametry sygnału (bez zmian w logice)
        signal_mode = st.radio("Tryb sygnału", ["Konserwatywny", "Umiarkowany", "Agresywny"], index=1, horizontal=True)
        rsi_min, rsi_max = st.slider("Przedział RSI", 10, 80, (30, 50))
        macd_lookback = st.slider("MACD: przecięcie (ostatnie N dni)", 1, 10, 3)
        use_volume = st.checkbox("Wymagaj potwierdzenia wolumenem", value=True)
        vol_window = st.selectbox("Średni wolumen (okno)", ["MA20", "MA50"], index=0)
        vol_window = 20 if vol_window == "MA20" else 50
        show_only_signals = st.checkbox("Pokaż tylko sygnały (min. 💎)", value=True)
        scan_limit = st.slider("Limit skanowania (dla bezpieczeństwa)", 50, 3500, 300, step=50)

        st.markdown("---")
        # przeniesione na dół
        source = st.selectbox("Źródło listy NASDAQ", ["Auto (online, fallback do CSV)", "Tylko CSV w repo"], index=0)
        period = st.selectbox("Okres danych", ["6mo", "1y", "2y"], index=1)  # domyślnie 1y

        # przycisk URUCHOM SKANER
        run_scan = st.button("🚀 Uruchom skaner", use_container_width=True, type="primary")

# =========================
# URUCHOMIENIE SKANU → WYNIKI → WYBÓR SPÓŁKI → WYKRESY
# =========================
if run_scan:
    tickers_df = get_tickers(source)
    if tickers_df is None or tickers_df.empty:
        st.error("Brak tickerów do skanowania.")
    else:
        tickers_list = tickers_df["Ticker"].tolist()[:scan_limit]
        progress = st.progress(0)
        status = st.empty()
        results = []

        for i, t in enumerate(tickers_list, start=1):
            status.write(f"⏳ {i}/{len(tickers_list)} – {t}")
            df = get_stock_df(t, period=period, vol_window=vol_window)
            if df is not None and not df.empty:
                last = df.iloc[-1]
                vol_ok = vol_confirmation(last.get("Volume"), last.get("AvgVolume"), use_volume)
                macd_cross = macd_bullish_cross_recent(df, macd_lookback)
                di = score_diamonds(
                    last.get("Close"), last.get("EMA200"), last.get("RSI"),
                    macd_cross, vol_ok, signal_mode, rsi_min, rsi_max
                )
                results.append({
                    "Ticker": t,
                    "Close": round(float(last.get("Close")), 2) if pd.notna(last.get("Close")) else None,
                    "RSI": round(float(last.get("RSI")), 2) if pd.notna(last.get("RSI")) else None,
                    "EMA200": round(float(last.get("EMA200")), 2) if pd.notna(last.get("EMA200")) else None,
                    "MACD": round(float(last.get("MACD")), 4) if pd.notna(last.get("MACD")) else None,
                    "MACD_signal": round(float(last.get("MACD_signal")), 4) if pd.notna(last.get("MACD_signal")) else None,
                    "Volume": int(last.get("Volume")) if pd.notna(last.get("Volume")) else None,
                    "Sygnał": di
                })
            progress.progress(i/len(tickers_list))

        status.write("✅ Zakończono skan.")
        # zapisz w sesji
        st.session_state.scan_results = pd.DataFrame(results)

# Sekcja wyników (jeśli są w sesji)
if "scan_results" in st.session_state:
    df_res = st.session_state.scan_results.copy()

    # filtr widoku — tylko sygnały czy wszystko
    if show_only_signals:
        df_res = df_res[df_res["Sygnał"] != "–"]

    # sortowanie wg siły sygnału
    df_res["Rank"] = df_res["Sygnał"].apply(diamond_rank)
    df_res = df_res.sort_values(["Rank","Ticker"], ascending=[False, True]).drop(columns=["Rank"])

    st.subheader("📋 Wyniki skanera")
    st.write(
        f"<span class='pill'>Wyników: <b>{len(df_res)}</b></span>"
        f"<span class='pill'>Tryb: <b>{signal_mode}</b></span>"
        f"<span class='pill'>Okres: <b>{period}</b></span>"
        f"<span class='pill'>RSI: <b>{rsi_min}–{rsi_max}</b></span>",
        unsafe_allow_html=True
    )
    st.dataframe(df_res, use_container_width=True)

    # Szybki wybór spółki → wykresy
    st.markdown("### 💎 Wybierz spółkę do podglądu wykresów")
    symbols = df_res["Ticker"].tolist()
    if symbols:
        # siatka przycisków
        cols = st.columns(6)
        selected_symbol = None
        for i, sym in enumerate(symbols):
            if cols[i % 6].button(sym):
                selected_symbol = sym

        # jeśli kliknięto, pokaż wykresy (Plotly)
        if selected_symbol:
            with st.spinner(f"Ładuję wykresy dla {selected_symbol}…"):
                df_sel = get_stock_df(selected_symbol, period=period, vol_window=vol_window)
            if df_sel is None or df_sel.empty:
                st.error("Nie udało się pobrać danych wykresu.")
            else:
                last = df_sel.iloc[-1]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Kurs (Close)", f"{last.get('Close'):.2f}" if pd.notna(last.get("Close")) else "—")
                m2.metric("RSI", f"{last.get('RSI'):.2f}" if pd.notna(last.get("RSI")) else "—")
                dist = (last.get("Close")/last.get("EMA200")-1)*100 if pd.notna(last.get("Close")) and pd.notna(last.get("EMA200")) else None
                m3.metric("Dystans do EMA200", f"{dist:.2f}%" if dist is not None else "—")
                # policz diamenty dla wybranej (ta sama logika)
                macd_cross = macd_bullish_cross_recent(df_sel, macd_lookback)
                vol_ok = vol_confirmation(last.get("Volume"), last.get("AvgVolume"), use_volume)
                di = score_diamonds(last.get("Close"), last.get("EMA200"), last.get("RSI"),
                                    macd_cross, vol_ok, signal_mode, rsi_min, rsi_max)
                m4.metric("Sygnał", di)

                st.plotly_chart(plot_candles_with_ema(df_sel, selected_symbol), use_container_width=True)
                st.plotly_chart(plot_rsi(df_sel, selected_symbol), use_container_width=True)

                # mała tabela pod wykresami
                with st.expander("Pokaż tabelę wskaźników (ostatnie 15 d.):", expanded=False):
                    st.dataframe(
                        df_sel[["Close","RSI","EMA200","MACD","MACD_signal","Volume","AvgVolume"]].tail(15),
                        use_container_width=True
                    )
    else:
        st.info("Brak spółek do wyboru przy obecnych filtrach.")
else:
    st.info("Kliknij **🚀 Uruchom skaner** w panelu bocznym, aby rozpocząć.")
