RocketStock – jak działa skaner, wskaźniki i scoring

RocketStock skanuje spółki z NASDAQ, oblicza zestaw wskaźników technicznych, filtruje je według Twoich krylaży, a następnie ocenia sygnał i tworzy ranking najlepszych kandydatów. Ten dokument wyjaśnia co liczymy i jak podejmowane są decyzje.

Dane i okresy

Notowania dzienne z Yahoo Finance (yfinance).

Okres: 6mo, 1y, 2y.

Wolumen średni: krocząca średnia z okna MA20 lub MA50 (wybierasz w panelu).

Wskaźniki techniczne (co liczymy)
1) RSI(14)

Klasyczny RSI z oknem 14 sesji.

W aplikacji ustawiasz twarde widełki RSI (np. 30–50). Jeśli RSI spółki jest poza tym zakresem, jest odrzucana na etapie scoringu (patrz „Diamenty”).

2) EMA50 i EMA200

EMA50 i EMA200 z ceny zamknięcia.

Dodatkowo liczymy:

DistEMA200Pct = (Close / EMA200 − 1) × 100% – procentowe oddalenie ceny od EMA200.

EMA200_Slope5 – uśrednione 5-dniowe tempo zmiany EMA200 (trend długoterminowy „pod górę”/„w dół”).

3) MACD (12,26,9)

Linia MACD, Signal i Histogram = MACD − Signal.

Skaner sprawdza bycze przecięcie (MACD przecina Signal w górę) w ostatnich N dniach (ustawiasz suwak „MACD: przecięcie (ostatnie N dni)”).

4) ATR(14) i ATR%

ATR(14) jako miara zmienności.

W filtrach możesz ograniczyć ATR% = ATR / Close × 100% (np. ≤ 8%), by eliminować zbyt „rozchwiane” walory.

5) Średni wolumen i relacja wolumenu (VolRatio)

AvgVolume = średnia krocząca wolumenu (MA20/MA50).

VolRatio = Volume / AvgVolume.

Kategoryzacja używana w tabeli i filtrze:

Wysoki: VolRatio ≥ 1.2

Średni: 0.8 ≤ VolRatio < 1.2

Niski: VolRatio < 0.8

6) Luki i cena minimalna

GapUpPct = (Open − wczorajszy Close) / wczorajszy Close × 100%.

Filtr „Max GAP UP %” odcina walory z nadmierną luką (np. > 8%).

Filtr „Min cena ($)” eliminuje groszówki poniżej progu.

7) Struktura szczytów/dołków i bliskość oporu

HH3: 3 ostatnie High rosnące sekwencyjnie.

HL3: 3 ostatnie Low rosnące sekwencyjnie.

Filtr „Struktura: HH & HL (ostatnie 3 świece)” wymaga jednoczesnego HH3 i HL3.

High_3m: maksimum z 63 sesji (ok. 3 miesiące).

RoomToHighPct = (High_3m − Close)/Close × 100%.

Filtr „Bliskość oporu” wymaga, by do szczytu 3M było np. ≥ 3% „oddechu”.

Preskan (twarde kryteria zanim policzymy sygnał)

RSI w widełkach: jeśli RSI < RSI_min lub > RSI_max → odpad.

(Opcjonalnie) Close > EMA200 oraz cap: jeśli zaznaczysz „Wymagaj Close > EMA200 (prescan)”, to dodatkowo:

DistEMA200Pct musi być ≥ 0% i ≤ ema_dist_cap (np. max 15% powyżej EMA200).

(Opcjonalnie) dodatkowe filtry:

Min. AvgVolume, widełki VolRatio (min–max),

Kapitalizacja rynkowa w przedziale (min–max),

Max GAP UP %, Min cena ($),

Max ATR%, HH & HL, Min odległość do 3m high.

Jeśli spółka przejdzie preskan, dopiero wtedy liczymy sygnał i „diamenty”.

Sygnał i „Diamenty” (logika scoringu)
Wymagania wejściowe do punktacji

RSI musi być w widełkach ([RSI_min, RSI_max]) – inaczej od razu „–”.

Vol. potwierdzenie (jeśli zaznaczone): Volume > AvgVolume.

Punkty składowe (maks. 4)

Cena vs EMA200

Tryb Konserwatywny: Close > EMA200

Umiarkowany: Close ≥ 0.995 × EMA200 (do 0.5% poniżej tolerowane)

Agresywny: Close ≥ 0.98 × EMA200 (do 2% poniżej tolerowane)

RSI – jeśli jest, to +1 pkt (już przeszło widełki).

MACD bullish cross w ostatnich N dniach – +1 pkt.

Potwierdzenie wolumenem (jeśli wymagane) – +1 pkt.

Mapa punktów → „diamenty”

Konserwatywny

4 pkt → 💎💎💎

3 pkt → 💎💎

2 pkt → 💎

mniej → –

Umiarkowany / Agresywny

≥3 pkt → 💎💎💎

2 pkt → 💎💎

1 pkt → 💎

0 pkt → –

Uwaga: jeśli RSI jest poza widełkami – niezależnie od pozostałych warunków – wynik to „–”.

Ranking (TOP propozycje bez AI)

Do rankingu trafiają tylko spółki z wynikiem 💎💎💎. Każda dostaje Score 0–100:

dist_score (35%) – im bliżej, ale nie dalej niż +10% nad EMA200, tym lepiej:
dist = clamp((Close/EMA200 − 1), 0, 0.10) / 0.10

rsi_score (35%) – im bliżej środka widełek RSI, tym lepiej:
mid = (RSI_min + RSI_max)/2
half = max(1, (RSI_max − RSI_min)/2)
rsi_score = 1 − clamp(|RSI − mid| / half, 0, 1)

volr_score (20%) – VolRatio obcięty do 2.0 i przeskalowany do 0–1:
volr_score = clamp(VolRatio, 0, 2.0)/2.0

liq_score (10%) – wg AvgVolume:

≥ 5M → 1.0

≥ 2M → 0.7

≥ 1M → 0.5

0 → 0.2

brak → 0.0

Score = 0.35·dist + 0.35·rsi + 0.20·volr + 0.10·liq, w skali 0–100 (1 miejsce = najwyższy score).
W rankingu przycisk z etykietą 1. AMD · 97.3 oznacza: pozycja oraz wynik rankingowy.

Opis filtrów i ich wpływ

Tryb sygnału: Konserwatywny/Umiarkowany/Agresywny – wpływa na warunek ceny względem EMA200 przy liczeniu diamentów.

Przedział RSI (twardy): spółki z RSI poza widełkami nie dostają sygnału (z automatu „–”).

MACD: przecięcie (ostatnie N dni): sprawdzamy, czy w oknie N dni był bullish cross MACD.

Wymagaj potwierdzenia wolumenem: w scoringu punkt za Volume > AvgVolume.

Średni wolumen (okno): wybór MA20/MA50 wpływa na AvgVolume i VolRatio.

Wymagaj Close > EMA200 (prescan) + Max % nad EMA200: twardy filtr przed scoringiem.

Pokaż tylko 💎💎💎: ogranicza widok do najlepszych sygnałów.

Filtr wolumenu: po kategoryzacji VolRatio – Wysoki / Średni / Niski.

Min. średni wolumen (AvgVolume): odcina mało płynne walory.

Widełki VolRatio: dopuszczalny zakres np. od 1.2 do 3.0.

Kapitalizacja (USD): zakres MC minimalny–maksymalny.

Max GAP UP %: eliminuje zbyt „wystrzelone” otwarcia.

Min cena ($): odcina bardzo tanie akcje.

Max ATR%: ogranicza wysoką zmienność.

Struktura HH & HL: wymaga wzrostowej struktury 3 ostatnich świec.

Bliskość oporu (min % do 3M high): wymaga min. „oddechu” do 3-miesięcznego szczytu.

Podsumowanie PRO (co pokazujemy)

Nie wpływa na scoring/filtry – to opis spółki w oparciu o Yahoo:

Metryki wyceny: P/E (TTM), Forward P/E, PEG, P/S, P/B, EV/EBITDA (liczone z EV i EBITDA).

Marże: brutto, operacyjna, netto.

Zwroty: 1M, 3M, 6M, 1Y (na podstawie cen z ostatniego roku) + Max drawdown 1Y.

Dywidendy: TTM i stopa.

Rekomendacje: sumy Strong Buy/Buy/Hold/Sell (o ile dostępne).

Price Target: mean/high/low (o ile dostępne).

Short interest: shares short, short % float, short ratio (days to cover), float shares.

Wejścia techniczne (pomocnicze):

Breakout ≈ max(20-dniowe High, Close) + 0.10×ATR

Pullback ≈ EMA bazowa (EMA50 jeśli > EMA200, inaczej EMA200) + 0.10×ATR

Kolejność sekcji na stronie

Ranking (jeśli włączony): guziki 1. TICKER · Score.

Wybierz spółkę do podsumowania (selectbox) – zawsze nad wynikami.

Wyniki skanera (tabela) – wyrównanie do lewej, na telefonie przewijana w poziomie.

Podsumowanie PRO + wykresy wybranej spółki.

Uwaga: kliknięcie w przycisk rankingu resetuje selectbox, żeby wybór z rankingu nie był nadpisywany.

Dobre praktyki doboru parametrów

Zacznij od Umiarkowanego trybu, RSI 30–50, MACD N=3, Wymagaj wolumenu.

Ustaw Max % nad EMA200 na 10–15% – unikasz kupowania „za wysoko”.

Volumen: zacznij od Wysoki + Średni (VolRatio ≥ 0.8) lub filtr Average Volume ≥ 1M.

Dodaj Max ATR% (np. 8%), by ograniczyć skrajnie zmienne walory.

W rankingu patrz na Score, ale też na AvgVolume i RoomToHighPct (w Podsumowaniu).

Zastrzeżenie

Wszystkie wskaźniki i filtry mają charakter informacyjny. To nie jest rekomendacja inwestycyjna. Zmienność rynku może powodować nagłe zmiany we wskaźnikach i wynikach.

Słownik wskaźników
RSI (Relative Strength Index)

Co to jest: oscylator impetu (momentum) 0–100 pokazujący „siłę” ruchu ceny.

Wzór (idea): średnia zysków vs. średnia strat z ostatnich 14 świec → przeskalowane do 0–100.

Jak czytać:

30–50: umiarkowana siła po spadkach / faza budowy trendu.

70: wykupienie (ryzyko korekty rośnie), <30: wyprzedanie (możliwy odbicie).

W Twoim skanerze RSI ma twarde widełki — poza nimi spółka odpada w scoringu.

EMA / SMA (średnie kroczące)

Co to jest: wygładzone linie trendu ceny.

EMA50, EMA200: wykładnicze średnie z 50 i 200 dni (EMA mocniej „waży” najświeższe świece).

Jak czytać:

Close > EMA200: długoterminowo „po byczej stronie”.

EMA200 Slope5: średnia 5-dniowych zmian EMA200 — pokazuje, czy długoterminowy trend rośnie/spada.

MACD (Moving Average Convergence/Divergence)

Co to jest: wskaźnik momentum oparty na różnicy dwóch EMA.

Składniki:

MACD line = EMA12 − EMA26

Signal = EMA(MACD, 9)

Histogram = MACD − Signal

Sygnał byczy (u Ciebie): bullish cross — MACD przecina od dołu linię Signal w ostatnich N dniach.

Jak czytać: przecięcia i kierunek histogramu wskazują zmianę tempa trendu.

ATR (Average True Range)

Co to jest: średnia rzeczywistego zasięgu ruchu (zmienność) z okna, u Ciebie 14.

ATR%: ATR / Close × 100% — „jaka część ceny to dzienny szum”.

Jak czytać: duży ATR% = wysoka zmienność (ryzyko). W filtrach możesz limitować (np. ≤ 8%).

AvgVolume (średni wolumen)

Co to jest: średnia krocząca wolumenu (MA20 lub MA50).

Do czego: baza do porównania bieżącego wolumenu i kategoryzacji płynności.

VolRatio (relacja wolumenu)

Definicja: VolRatio = dzisiejszy Volume / AvgVolume.

Kategoryzacja (w aplikacji):

Wysoki: ≥ 1.2

Średni: 0.8–1.2

Niski: < 0.8

Jak czytać: aktywność obrotu vs. typowa — potwierdza/negowa siłę ruchu.

GapUpPct (luka wzrostowa)

Definicja: (Open − wczorajszy Close) / wczorajszy Close × 100%.

Jak czytać: duże luki = news/impuls. Filtr „Max GAP UP %” odcina „przepalone” starty.

DistEMA200Pct (dystans do EMA200)

Definicja: (Close / EMA200 − 1) × 100%.

Jak czytać: jak „daleko” od długoterminowej średniej (ryzyko pościgu). U Ciebie może być limit (cap).

High_3m i RoomToHighPct

High_3m: najwyższy kurs z ~63 sesji (ok. 3 miesiące).

RoomToHighPct: (High_3m − Close) / Close × 100% — „oddech” do lokalnego oporu.

Jak czytać: zbyt mały „oddech” = ryzyko wybicia prosto w opór (można filtrować).

HH3 / HL3 (struktura szczytów i dołków)

HH3: trzy ostatnie High rosną (Higher High).

HL3: trzy ostatnie Low rosną (Higher Low).

Jak czytać: razem HH3&HL3 = sekwencja wzrostowa na świecach.

Potwierdzenie wolumenem

Warunek w scoringu: Volume > AvgVolume (jeśli włączone).

Jak czytać: ruch „na wolumenie” ma większą wiarygodność.

Pojęcia fundamentalne (w Podsumowaniu PRO)

MarketCap (kapitalizacja): wartość rynkowa spółki = cena × liczba akcji.

P/E (TTM / Forward): ile płacisz za 1 USD zysku (z przeszłości / oczekiwany).

PEG: P/E skorygowane o wzrost zysków (≈ P/E ÷ growth). Δ<1 bywa postrzegane jako tańsze.

P/S, P/B: cena do sprzedaży / wartość księgowa — wycena względem przychodów/aktywa netto.

EV/EBITDA: Enterprise Value do EBITDA — porównywalna miara wyceny (mniej zależna od struktury długu).

Margins:

Gross (brutto), Operating (operacyjna), Net (netto) — jakość zysków.

Dividends (TTM / yield): wypłaty z 12 mies. i stopa dywidendy.

Short interest:

Short % Float: odsetek akcji w krótkiej sprzedaży vs. free float.

Short ratio (days to cover): dni potrzebne, by obkupić krótkie pozycje przy średnim wolumenie.

Zwroty (1M/3M/6M/1Y) i Max Drawdown 1Y: momentum i ryzyko historyczne.

Logika „Diamentów” (skrót)

Najpierw preskan: RSI w widełkach + (opcjonalnie) Close > EMA200 i cap + inne filtry.

Potem punkty (0–4):

Cena względem EMA200 (zależnie od trybu),

RSI (już w widełkach),

MACD bullish cross w ostatnich N dniach,

Potwierdzenie wolumenem (jeśli wymagane).

Mapowanie punktów → diamenty:

Konserwatywny: 4=💎💎💎, 3=💎💎, 2=💎, <2=–

Umiarkowany/Agresywny: ≥3=💎💎💎, 2=💎💎, 1=💎, 0=–

Jak używać w praktyce (tipy)

Zacznij od RSI 30–50, MACD okno 3, wymagaj wolumenu.

Close > EMA200 z cap 10–15% ogranicza „kupowanie szczytów”.

Filtrowanie po Wysoki/Średni wolumen i ATR% ≤ 8% poprawia jakość kandydatów.

W rankingu patrz na Score i płynność (AvgVolume), a w Podsumowaniu na RoomToHighPct i marże.
