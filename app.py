import streamlit as st
import yfinance as yf
import pandas as pd
import ta

st.title("📊 NASDAQ Scanner MVP")

# Wpisywanie tickera
ticker = st.text_input("Podaj ticker (np. AAPL, MSFT, TSLA):", "AAPL")

if st.button("🔍 Szukaj spółki"):
    data = yf.download(ticker, period="6mo", interval="1d")

    if not data.empty:
        st.write(f"Dane dla: {ticker}")
        st.line_chart(data["Close"])

        # RSI
        data["RSI"] = ta.momentum.RSIIndicator(data["Close"]).rsi()

        # EMA200
        data["EMA200"] = ta.trend.EMAIndicator(data["Close"], window=200).ema_indicator()

        # MACD
        macd = ta.trend.MACD(data["Close"])
        data["MACD"] = macd.macd()
        data["MACD_signal"] = macd.macd_signal()

        st.write(data.tail())

    else:
        st.warning("Brak danych dla tego tickera.")
