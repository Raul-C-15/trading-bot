# 📈 Algorithmic Swing Trading Bot (Acțiuni SUA)

Un bot de tranzacționare algoritmică pentru piața de acțiuni din SUA, proiectat cu focus pe **managementul strict al riscului (1-2% risc per tranzacție)**, identificarea trendurilor și execuție automatizată.

---

## 🎯 Strategia de Tranzacționare Instituțională (Pullback EMA 20 & VCP Breakout)

1. **Filtrul de Regim al Pieței pe 3 Niveluri (Multi-Tier Market Regime — `SPY` + `QQQ` + `^VIX`):**
   - **POWER TREND (Risc 100% -> 1.5% / trade):** `SPY > SMA 50 > SMA 200`, `QQQ > SMA 200` și `VIX < 25`.
   - **CAUTIOUS UPTREND (Risc Redus 50% -> 0.75% / trade):** `SPY > SMA 200`, dar sub `SMA 50` sau cu volatilitate ridicată (`VIX >= 25`).
   - **BEARISH (Cash 100%):** `SPY < SMA 200` — botul blochează automat orice poziție nouă Long pentru protejarea capitalului.
2. **Putere Relativă Instituțională (`RS_vs_SPY`) & Trend Săptămânal (`1W`):**
   - Calculează scorul ponderat **IBD Relative Strength** (performanța pe 20, 60 și 120 de zile față de S&P 500).
   - Verifică alinierea pe graficul săptămânal (`Close > 1W EMA 10 > 1W SMA 30`) și panta pozitivă a `SMA 50`.
3. **Două Setup-uri Complementare cu Contracție de Volatilitate (VCP):**
   - **Setup 1 — Pullback la EMA 20:** Retragere controlată către `EMA 20` cu `Volume Dry-Up` (volum în scădere pe corecție), `RSI (14)` între `38–56` și lumânare de reversare bullish.
   - **Setup 2 — Volume Breakout:** Spargerea maximului ultimelor 20 de zile (`Swing_High_20`) cu volum `>= 1.3x SMA 20`, contracție VCP (`ATR_5 / ATR_14`) și filtru anti-extensie (`<= 12%` față de `EMA 20`).
4. **Filtru Machine Learning (`XGBoost` cu 11 Caracteristici & Validare Out-of-Sample):**
   - Evaluează probabilitatea de câștig a fiecărui semnal pe baza a 11 caracteristici tehnice și de forță relativă (`rs_vs_spy`, `atr_contraction`, `weekly_rsi`, etc.) fără *Data Leakage* în backtest.
5. **Managementul Riscului & Execuție Split-Order (Lot 1 + Runner):**
   - **Risc per tranzacție:** `1.5%` în `POWER_TREND` (`0.75%` în `CAUTIOUS_UPTREND`), cu limită strictă de **Portfolio Heat** (`MAX_OPEN_POSITIONS = 4` poziții cu risc activ).
   - **Stop-Loss Structural + ATR:** Plasat sub minimul local (`Swing_Low_10` / `EMA 20`) minus buffer `ATR`, plafonat la maximum `8%`.
   - **Execuție Split-Order pe Alpaca:** Plasează automat **Lot 1 (50%)** cu țintă la **TP1 (`1.5R`)** și **Lot 2 Runner (50%)** cu țintă la **TP2 (`3.0R`)**.
   - **Free-Roll Breakeven & Trailing Stop Hibrid:** Imediat ce `TP1` este atins, `position_manager.py` mută automat Stop-Loss-ul Lotului 2 la **Breakeven (`+0.15%`)**, eliberând bugetul de risc, și activează un **Trailing Stop dinamic** (`Chandelier 3.8%` + suport `EMA 20` / `Low_3D`).
   - **Protecție Earnings:** Blochează intrările noi cu `7` zile înainte de raportul financiar și securizează automat pozițiile deschise cu `<= 2` zile înainte de Earnings.

---

## 🚀 Cum deschizi și rulezi proiectul în Visual Studio Code

### 1. Deschide folderul în VS Code
1. Deschide **Visual Studio Code**.
2. Mergi la meniul de sus: `File` -> `Open Folder...`.
3. Selectează folderul: `/Users/raulcetean15/trading-bot`.

### 2. Selectează mediul virtual Python (.venv)
1. În VS Code, apasă combinația de taste: `Cmd + Shift + P` (pe Mac).
2. Tastează și selectează: **`Python: Select Interpreter`**.
3. Alege interpretorul din folderul creat:
   `./.venv/bin/python` (Python 3.9).

### 3. Rulează Botul din Terminal
Deschide terminalul integrat în VS Code (`Terminal` -> `New Terminal`) și rulează:
```bash
python3 main.py
```

---

## 🛠️ Conectarea la Alpaca Paper Trading (Gratuit, Fără Bani Reali)

Botul poate rula fie în mod **simulat local** (fără cont), fie conectat la contul tău demo **Alpaca**:

1. Creează un cont gratuit pe [Alpaca Markets](https://alpaca.markets).
2. În panoul de control, comută pe **Paper Trading** (în stânga sus, butonul mov).
3. Generează o pereche de chei API: **API Key** și **Secret Key**.
4. Creează un fișier numit `.env` în folderul `trading-bot/` (poți copia `.env.example`):
   ```env
   ALPACA_API_KEY=PKXXXXXXXXXXXXXXXXXX
   ALPACA_SECRET_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
   ALPACA_PAPER=True
   ```
5. La următoarea rulare a lui `python main.py`, botul se va conecta automat și va citi balanța contului tău demo!

---

## 📱 Notificări Instant pe Telegram (Opțional)

Botul îți poate trimite notificări directe pe telefon:
1. Creează un bot gratuit pe Telegram vorbind cu `@BotFather` (comanda `/newbot`).
2. Copiază **HTTP API Token-ul**.
3. Află ID-ul tău de utilizator Telegram dând start la `@userinfobot`.
4. Adaugă datele în fișierul `.env`:
   ```env
   TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
   TELEGRAM_CHAT_ID=123456789
   ```

---

## 🤖 Modul Autonom (Scheduler 24/7)

Poți lăsa botul să ruleze continuu în fundal:
```bash
python3 scheduler.py
```
*(sau alegi **Opțiunea 4** din `python3 main.py`)*

### Ce face Scheduler-ul:
1. **Verifică programul Bursei SUA:** Știe automat când bursa este deschisă (16:30 - 23:00 ora României) și când sunt sărbători legale prin API-ul Alpaca.
2. **Scanează automat piața:** La fiecare 30 de minute cât timp bursa este deschisă.
3. **Trimite alerte Telegram:** De fiecare dată când apare un setup cu R:R optim.
4. **Auto-Trade (Opțional):** Dacă setezi în `.env` `AUTO_TRADE=True`, botul va plasa automat ordinele Bracket pe Alpaca fără intervenție manuală!

---

## 📁 Structura Fișierelor

* `config.py` — Parametrii strategiei, universul extins de lideri (`UNIVERSE_POOL`), pragurile RS/VCP/ML și regulile de risc.
* `indicators.py` — Indicatori tehnici zilnici/săptămânali, Puterea Relativă (`RS_vs_SPY`), contracția VCP (`ATR_Contraction`) și panta `SMA_50_Slope`.
* `data_loader.py` — Descărcarea datelor cu cache SPY, regimul pieței pe 3 niveluri (`SPY + QQQ + ^VIX`), calendarul Earnings și Screener-ul dinamic de lideri Stage 2.
* `strategy.py` — Motorul unificat de analiză multi-timeframe, filtrare RS/VCP/ML, scor compozit (`0-100`) și dimensionare adaptivă.
* `backtester.py` — Simulator istoric aliniat 100% cu strategia live (slippage, Trailing Stop pe Runner, Sharpe/Sortino/Expectancy R și simulare realistă de portofoliu la comun).
* `ml_model.py` & `ml_predictor.py` — Antrenare și predicție `XGBoost` pe 11 caracteristici cu stocarea predicțiilor Out-of-Sample (`TimeSeriesSplit`).
* `bot_trader.py` — Execuție pe Alpaca Paper/Live cu suport **Split-Order Bracket** (`50% TP1 + 50% TP2 Runner`) și verificare **Portfolio Heat**.
* `position_manager.py` — Gestionare autonomă a pozițiilor deschise (Free-Roll Breakeven, Trailing Stop hibrid și protecție înainte de Earnings).
* `risk_learner.py` — Jurnal automat de tranzacții sincronizat cu Alpaca și sistem de carantină (`Cooldown`) după 2 poziții pierzătoare consecutive.
* `optimizer.py` — Optimizare robustă Walk-Forward (Anti-Overfitting) pe ferestre In-Sample / Out-of-Sample.
* `notifier.py` & `telegram_listener.py` — Alerte Telegram cu deduplicare zilnică și butoane interactive `[✅ CUMPĂRĂ]` / `[❌ IGNORĂ]`.
* `scheduler.py` — Execuție autonomă în timpul orelor de bursă cu prioritizarea semnalelor după Scorul Compozit.
* `main.py` — Meniul principal interactiv (9 module).
