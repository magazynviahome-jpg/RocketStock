import io
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import ta

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
    .small-note {opacity: 0.7; font-size: 0.9rem;}
    .ok-badge {background:#e8f7ee; color:#1e7e34; padding:2px 8px; border-radius:12px; font-weight:600;}
    .warn-badge {background:#fff4e5; color:#8a5a00; padding:2px 8px; border-radius:12px; font-weight:600;}
    .error-badge {background:#fdecea; color:#b00020; padding:2px 8px; border-radius:12px; font-weight:600;}
    .muted {color:#6b7280;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚀 RocketStock")
st.caption("Prosty skaner NASDAQ oparty o RSI 30–50, EMA200, MACD i wolumen — z diamentowym scoringiem 💎.")

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
            st.caption(f"<span class='warn-badge'>Uwaga</span> Nie udało się pobrać online: {e}. Wczytuję CSV…", unsafe_allow_html=True)
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

def score_diamonds(price, ema200, rsi, macd_cross, vol_ok, mode: str, rsi_min: int, rsi_max: int) -> str:
    """Konwersja warunków → diamenty, z 3 czułościami."""
    # bazowe punkty
    pts = 0
    # TRYBY
    if mode == "Konserwatywny":
        # wymagające progi (4 kryteria → 3 diamenty)
        if pd.notna(price) and pd.notna(ema200) and price > ema200: pts += 1
        if pd.notna(rsi) and rsi_min <= rsi <= rsi_max: pts += 1
        if macd_cross: pts += 1
        if vol_ok: pts += 1
        if pts >= 4: return "💎💎💎"
        if pts == 3: return "💎💎"
        if pts == 2: return "💎"
        return "–"

    elif mode == "Umiarkowany":
        # 3 kryteria → 3 diamenty
        if pd.notna(price) and pd.notna(ema200) and (price >= ema200*0.995): pts += 1
        if pd.notna(rsi) and (rsi_min-2) <= rsi <= (rsi_max+2): pts += 1
        if macd_cross: pts += 1
        if vol_ok: pts += 1
        if pts >= 3: return "💎💎💎"
        if pts == 2: return "💎💎"
        if pts == 1: return "💎"
        return "–"

    else:  # Agresywny
        # 2–3 kryteria → 3 diamenty, miękkie progi
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
# SIDEBAR – PROSTE USTAWIENIA
# =========================
with st.sidebar:
    st.header("⚙️ Ustawienia")
    source = st.selectbox("Źródło listy NASDAQ", ["Auto (online, fallback do CSV)", "Tylko CSV w repo"], index=0)
    period = st.selectbox("Okres danych", ["6mo", "1y", "2y"], index=1)

    st.markdown("---")
    with st.expander("Ustawienia zaawansowane", expanded=False):
        signal_mode = st.radio("Tryb sygnału", ["Konserwatywny", "Umiarkowany", "Agresywny"], index=1, horizontal=True)
        rsi_min, rsi_max = st.slider("Przedział RSI", 10, 80, (30, 50))
        macd_lookback = st.slider("MACD: przecięcie (ostatnie N dni)", 1, 10, 3)
        use_volume = st.checkbox("Wymagaj potwierdzenia wolumenem", value=True)
        vol_window = st.selectbox("Średni wolumen (okno)", ["MA20", "MA50"], index=0)
        vol_window = 20 if vol_window == "MA20" else 50
        show_only_signals = st.checkbox("Pokaż tylko sygnały (min. 💎)", value=True)
        scan_limit = st.slider("Limit skanowania (dla bezpieczeństwa)", 50, 3500, 300, step=50)

# =========================
# KARTY: SPÓŁKA / SKANER
# =========================
tab1, tab2 = st.tabs(["🔎 Spółka", "📈 Skaner"])

# ---------- TAB 1: POJEDYNCZA SPÓŁKA ----------
with tab1:
    with st.form("single_form"):
        c1, c2 = st.columns([2,1])
        with c1:
            single_ticker = st.text_input("Ticker", "AAPL")
        with c2:
            run_single = st.form_submit_button("Uruchom analizę")
    if run_single:
        with st.spinner(f"Pobieram dane dla {single_ticker}…"):
            df = get_stock_df(single_ticker, period=period, vol_window=vol_window)
        if df is None or df.empty:
            st.error("Nie udało się pobrać danych lub policzyć wskaźników.")
        else:
            last = df.iloc[-1]
            vol_ok = vol_confirmation(last.get("Volume"), last.get("AvgVolume"), use_volume)
            macd_cross = macd_bullish_cross_recent(df, macd_lookback)
            diamonds = score_diamonds(
                last.get("Close"), last.get("EMA200"), last.get("RSI"),
                macd_cross, vol_ok, signal_mode, rsi_min, rsi_max
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Kurs (Close)", f"{last.get('Close'):.2f}" if pd.notna(last.get("Close")) else "—")
            m2.metric("RSI", f"{last.get('RSI'):.2f}" if pd.notna(last.get("RSI")) else "—")
            dist = (last.get("Close")/last.get("EMA200")-1)*100 if pd.notna(last.get("Close")) and pd.notna(last.get("EMA200")) else None
            m3.metric("Dystans do EMA200", f"{dist:.2f}%" if dist is not None else "—")
            m4.metric("Sygnał", diamonds)

            st.line_chart(df["Close"], height=260)
            with st.expander("Pokaż tabelę wskaźników (ostatnie 15 d.):", expanded=False):
                st.dataframe(
                    df[["Close","RSI","EMA200","MACD","MACD_signal","Volume","AvgVolume"]].tail(15),
                    use_container_width=True
                )

# ---------- TAB 2: SKANER NASDAQ ----------
with tab2:
    with st.form("scan_form"):
        st.write("Przeskanuj listę NASDAQ według wybranych warunków i nadaj diamentowy scoring 💎.")
        run_scan = st.form_submit_button("Uruchom skan teraz")

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
            if results:
                df_res = pd.DataFrame(results)
                if show_only_signals:
                    df_res = df_res[df_res["Sygnał"] != "–"]

                df_res["Rank"] = df_res["Sygnał"].apply(diamond_rank)
                df_res = df_res.sort_values(["Rank","Ticker"], ascending=[False, True]).drop(columns=["Rank"])

                st.success(f"Wyniki: {len(df_res)} spółek")
                st.dataframe(df_res, use_container_width=True)

                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                csv_bytes = df_res.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Pobierz wyniki CSV",
                    data=csv_bytes,
                    file_name=f"rocketstock_scan_{ts}.csv",
                    mime="text/csv"
                )
            else:
                st.info("Brak spółek spełniających kryteria.")
