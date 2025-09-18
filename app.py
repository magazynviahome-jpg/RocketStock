import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import io
import requests
from datetime import datetime

# =========================
# USTAWIENIA APLIKACJI
# =========================
st.set_page_config(page_title="🚀 RocketStock", layout="wide")
st.title("🚀 RocketStock – NASDAQ Scanner (RSI 30–50, EMA200, MACD, Wolumen)")

# =========================
# FUNKCJE POMOCNICZE
# =========================
NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"

@st.cache_data(show_spinner=False, ttl=60*30)
def fetch_nasdaq_online() -> pd.DataFrame:
    """
    Pobiera oficjalną listę NASDAQ (txt z separatorami |) i zwraca DataFrame z kolumną 'Ticker'.
    Filtrowanie:
      - Test Issue == 'N' (wycina tickery testowe)
      - Usuwa wiersz 'File Creation Time'
    """
    resp = requests.get(NASDAQ_URL, timeout=15)
    resp.raise_for_status()
    content = resp.content.decode("utf-8", errors="ignore")
    df = pd.read_csv(io.StringIO(content), sep="|")
    if "Symbol" not in df.columns:
        raise ValueError("Nie znaleziono kolumny 'Symbol' w pliku NASDAQ.")
    # Usuń stopkę
    df = df[df["Symbol"] != "File Creation Time"]
    # Wytnij testowe tickery
    if "Test Issue" in df.columns:
        df = df[df["Test Issue"] == "N"]
    tickers = df[["Symbol"]].rename(columns={"Symbol": "Ticker"}).dropna()
    tickers["Ticker"] = tickers["Ticker"].str.strip()
    tickers = tickers[tickers["Ticker"] != ""].drop_duplicates()
    return tickers

@st.cache_data(show_spinner=False)
def load_tickers_from_csv(path: str = "nasdaq_tickers_full.csv") -> pd.DataFrame:
    """
    Wczytuje tickery z pliku CSV w repo (jedna kolumna 'Ticker').
    """
    df = pd.read_csv(path)
    if "Ticker" not in df.columns:
        raise ValueError("Plik CSV musi zawierać kolumnę 'Ticker'.")
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df = df[df["Ticker"] != ""].drop_duplicates()
    return df

def get_final_ticker_list(source: str) -> pd.DataFrame:
    """
    Zwraca listę tickerów wg wybranego źródła:
      - 'Auto (online, fallback do CSV)'
      - 'Tylko CSV w repo'
    """
    if source == "Auto (online, fallback do CSV)":
        try:
            tickers = fetch_nasdaq_online()
            st.caption(f"Źródło: NASDAQTrader (online). Liczba tickerów: {len(tickers)}")
            return tickers
        except Exception as e:
            st.warning(f"Nie udało się pobrać listy online: {e}. Próba wczytania z CSV…")
            return load_tickers_from_csv()
    else:
        tickers = load_tickers_from_csv()
        st.caption(f"Źródło: CSV w repo. Liczba tickerów: {len(tickers)}")
        return tickers

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spłaszcza MultiIndex kolumn z yfinance, aby było np. 'Close' zamiast ('Close', 'AAPL').
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Liczy RSI, EMA200, MACD oraz średni wolumen dla przekazanego DataFrame z kolumnami:
    ['Close', 'Volume'] (wymagane), opcjonalnie inne.
    Zwraca DataFrame z kolumnami: RSI, EMA200, MACD, MACD_signal, AvgVolume20.
    """
    if "Close" not in df.columns or "Volume" not in df.columns:
        raise ValueError("Dane muszą zawierać kolumny 'Close' i 'Volume'.")
    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    # EMA200
    df["EMA200"] = ta.trend.EMAIndicator(df["Close"], window=200).ema_indicator()
    # MACD
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    # Średni wolumen (20)
    df["AvgVolume20"] = df["Volume"].rolling(20).mean()
    return df

def signal_scoring(price, ema200, rsi, macd_line, macd_signal, volume, avg_volume, adx=None) -> str:
    """
    Skala diamentowa: – / 💎 / 💎💎 / 💎💎💎
    Kryteria:
      +1: cena > EMA200
      +1: 30 <= RSI <= 50
      +1: MACD linia > sygnał (przecięcie w górę)
      +1: wolumen > średnia wolumenu (20)
      +1: (opcjonalnie) ADX > 20
    """
    score = 0
    try:
        if pd.notna(price) and pd.notna(ema200) and price > ema200:
            score += 1
        if pd.notna(rsi) and 30 <= rsi <= 50:
            score += 1
        if pd.notna(macd_line) and pd.notna(macd_signal) and macd_line > macd_signal:
            score += 1
        if pd.notna(volume) and pd.notna(avg_volume) and volume > avg_volume:
            score += 1
        if adx is not None and pd.notna(adx) and adx > 20:
            score += 1
    except Exception:
        pass

    if score >= 4:
        return "💎💎💎"
    elif score == 3:
        return "💎💎"
    elif score == 2:
        return "💎"
    else:
        return "–"

@st.cache_data(show_spinner=False)
def get_stock_history(ticker: str, period: str = "6mo") -> pd.DataFrame | None:
    """
    Pobiera dane 1D z yfinance dla jednego tickera i liczy wskaźniki.
    Zwraca DataFrame lub None jeśli brak danych.
    """
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    df = flatten_columns(df)
    try:
        df = compute_indicators(df)
        return df
    except Exception:
        return None

# =========================
# UI – SIDEBAR
# =========================
st.sidebar.header("⚙️ Ustawienia")
source = st.sidebar.selectbox(
    "Źródło listy NASDAQ",
    ["Auto (online, fallback do CSV)", "Tylko CSV w repo"],
    index=0
)
limit = st.sidebar.slider("Maks. liczba spółek do skanowania (dla bezpieczeństwa)", 50, 3000, 200, step=50)
period = st.sidebar.selectbox("Okres danych", ["6mo", "1y", "2y"], index=0)
show_only_signals = st.sidebar.checkbox("Pokaż tylko spółki z sygnałem (min. 💎)", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Wskazówka: pełny skan 3000+ spółek może długo trwać. Zacznij od 200–500.")

# =========================
# TRYB 1: POJEDYNCZY TICKER
# =========================
st.subheader("🔍 Analiza pojedynczej spółki")
col1, col2 = st.columns([2,1])
with col1:
    ticker = st.text_input("Podaj ticker (np. AAPL, MSFT, TSLA):", "AAPL")
with col2:
    run_single = st.button("Szukaj spółki")

if run_single:
    with st.spinner(f"Pobieram i liczę wskaźniki dla {ticker}…"):
        data = get_stock_history(ticker, period=period)
    if data is None:
        st.warning("Brak danych lub nie udało się policzyć wskaźników.")
    else:
        st.line_chart(data["Close"])
        last = data.iloc[-1]
        diamonds = signal_scoring(
            last.get("Close"), last.get("EMA200"), last.get("RSI"),
            last.get("MACD"), last.get("MACD_signal"),
            last.get("Volume"), last.get("AvgVolume20")
        )
        st.success(f"Sygnał dla {ticker}: {diamonds}")
        st.dataframe(
            data[["Close", "RSI", "EMA200", "MACD", "MACD_signal", "Volume", "AvgVolume20"]].tail(10)
        )

# =========================
# TRYB 2: SKANER NASDAQ
# =========================
st.subheader("📈 Skaner NASDAQ (na żądanie)")

run_scan = st.button("Przeskanuj listę NASDAQ teraz")
if run_scan:
    tickers_df = get_final_ticker_list(source)
    if tickers_df is None or tickers_df.empty:
        st.error("Nie mam żadnych tickerów do skanowania.")
    else:
        tickers_list = tickers_df["Ticker"].tolist()[:limit]
        progress = st.progress(0)
        results = []

        for i, t in enumerate(tickers_list, start=1):
            df = get_stock_history(t, period=period)
            if df is None:
                progress.progress(i / len(tickers_list))
                continue
            last = df.iloc[-1]
            diamonds = signal_scoring(
                last.get("Close"), last.get("EMA200"), last.get("RSI"),
                last.get("MACD"), last.get("MACD_signal"),
                last.get("Volume"), last.get("AvgVolume20")
            )
            row = {
                "Ticker": t,
                "Close": round(float(last.get("Close", float("nan"))), 2) if pd.notna(last.get("Close")) else None,
                "RSI": round(float(last.get("RSI", float("nan"))), 2) if pd.notna(last.get("RSI")) else None,
                "EMA200": round(float(last.get("EMA200", float("nan"))), 2) if pd.notna(last.get("EMA200")) else None,
                "MACD": round(float(last.get("MACD", float("nan"))), 4) if pd.notna(last.get("MACD")) else None,
                "MACD_signal": round(float(last.get("MACD_signal", float("nan"))), 4) if pd.notna(last.get("MACD_signal")) else None,
                "Volume": int(last.get("Volume")) if pd.notna(last.get("Volume")) else None,
                "Sygnał": diamonds
            }
            results.append(row)
            progress.progress(i / len(tickers_list))

        if results:
            df_results = pd.DataFrame(results)
            # Opcjonalne filtrowanie tylko sygnałów
            if show_only_signals:
                df_results = df_results[df_results["Sygnał"] != "–"]

            # Ranking wg liczby diamentów
            def diamond_rank(x: str) -> int:
                return 0 if x == "–" else len(x)
            df_results["Rank"] = df_results["Sygnał"].apply(diamond_rank)
            df_results = df_results.sort_values(["Rank", "Ticker"], ascending=[False, True]).drop(columns=["Rank"])

            st.write(f"Wyniki: {len(df_results)} spółek")
            st.dataframe(df_results, use_container_width=True)

            # Przyciski do pobrania CSV
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            csv_bytes = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Pobierz wyniki CSV",
                data=csv_bytes,
                file_name=f"rocketstock_scan_{ts}.csv",
                mime="text/csv"
            )
        else:
            st.warning("Brak wyników do wyświetlenia.")
