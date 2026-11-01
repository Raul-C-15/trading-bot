"""
Configurari generale pentru Trading Bot (Institutional Swing Trading Edition).
Toate setarile strategiei, filtrelor multi-timeframe, Relative Strength si ale riscului sunt definite aici.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Incarcare variabile de mediu din .env (daca exista)
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# --- LISTA DE ACTIUNI PENTRU SCANARE (CORE WATCHLIST DIVERSIFICAT MULTI-SECTOR) ---
# Portofoliu institutional (30 lideri de piata cu lichiditate si Win-Rate ridicat pe Swing Trading):
# Tech & AI, Semiconductori, Cybersecurity, Finante/Plati, Industriale/Energie, Sanatate & Retail.
WATCHLIST = [
    # 1. Mega-Cap Tech, Cloud & AI Software
    "NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "ORCL", "NOW", "PLTR",
    # 2. Semiconductori & Infrastructura Hardware AI
    "AVGO", "AMD", "TSM", "MU", "ANET", "MRVL",
    # 3. Cybersecurity & E-Commerce Leaders
    "CRWD", "PANW", "SHOP",
    # 4. Finante, Banci de Investitii & Retele Globale de Plati
    "JPM", "GS", "MA", "V",
    # 5. Industriale, Aerospace & Energie pentru Data-Centers
    "CAT", "GE", "CEG",
    # 6. Sanatate, Pharma & Robotica Medicala (MedTech)
    "LLY", "ISRG", "BSX",
    # 7. Retail Defensiv & Energie Clasica
    "WMT", "COST", "XOM"
]

# Numarul de lideri selectati de Screener-ul Dinamic (Optiunea 9 din meniu)
TOP_LEADERS_COUNT = 25

# Univers extins (~75 companii de elita S&P 500 / Nasdaq 100) pentru Screener-ul Dinamic de Forta Relativa
UNIVERSE_POOL = [
    # Tech, Cloud & AI Software
    "NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "PLTR", "CRM", "NOW", "ORCL", "ADBE", "INTU", "IBM", "SNOW", "DELL", "CSCO",
    # Semiconductors & AI Hardware / Networking
    "AVGO", "AMD", "TSM", "ASML", "MU", "ANET", "MRVL", "ARM", "KLAC", "LRCX", "AMAT", "QCOM", "TXN", "CRDO", "CLS", "FN", "SMCI",
    # Cybersecurity & Consumer Digital
    "PANW", "CRWD", "FTNT", "ZS", "NFLX", "UBER", "SHOP", "SPOT", "BKNG", "ABNB", "DKNG",
    # Financials, FinTech & Capital Markets
    "JPM", "GS", "MS", "V", "MA", "AXP", "BLK", "SPGI", "KKR", "APO", "BX", "SCHW", "BAC", "WFC", "HOOD", "NU",
    # Industrials, Aerospace, Defense & Power Grid
    "GE", "CAT", "ETN", "VRT", "GEV", "CEG", "VST", "RTX", "LMT", "URI", "PH", "DAL", "UAL", "RCL",
    # Healthcare, Biotech & MedTech
    "LLY", "ISRG", "ABBV", "VRTX", "REGN", "BSX", "SYK", "UNH",
    # Consumer, Retail, Auto & Energy/Materials
    "COST", "WMT", "HD", "TJX", "TSLA", "XOM", "CVX", "SLB", "FCX", "NKE", "KO", "DIS"
]

# Benchmark-urile pietei pentru filtrul de regim multidimensional (Market Regime Filter)
BENCHMARK_TICKER = "SPY"
SECONDARY_BENCHMARK = "QQQ"
VIX_TICKER = "^VIX"

# --- REGULI DE MANAGEMENT AL RISCULUI (NON-NEGOCIABILE) ---
# Risc maxim intre 1% si 2% per tranzactie (1.5% in Power Trend)
RISK_PER_TRADE_PERCENT = 0.015  # 1.5% ($15.00 risc la un cont de $1,000)

# Capital implicit si plafon de buget pentru testarea realista pe contul de $1,000
DEFAULT_SIMULATED_CAPITAL = 1000.0  # 1,000 USD
ACCOUNT_BUDGET_CAP = 1000.0         # Plafoneaza capitalul de lucru la $1,000 (+ PnL) chiar daca Alpaca Paper are $100,000
ALLOW_FRACTIONAL_SHARES = True      # Permite cumpararea de actiuni fractionare (ex: 0.28x LLY, 0.43x META, 1.75x NVDA)

# Numar maxim de pozitii cu RISC ACTIV deschise simultan (Portfolio Heat maxim: ~6%, max $250 / pozitie la $1,000)
# Nota: Pozitiile deja asigurate la Breakeven (Risc 0) devin 'Free-Roll'
MAX_OPEN_POSITIONS = 4

# Raport Risc / Recompensa minim acceptat pentru TP1 (1:1.5 -> mutare SL la Breakeven)
MIN_RISK_REWARD_RATIO = 1.5
TP2_RISK_REWARD_RATIO = 3.0     # Tinta extinsa pentru Runner (Lotul 2)

# Procent din pozitie inchis la Take Profit 1 (TP1)
TP1_EXIT_PERCENT = 0.50  # 50%

# Activare Split-Order pe Alpaca (50% Lot 1 la TP1 + 50% Lot 2 la TP2 / Trailing Stop)
USE_SPLIT_ORDERS = True

# Cost de alunecare (Slippage) per executie in Backtest pentru realism institutional (0.05%)
BACKTEST_SLIPPAGE_PCT = 0.0005

# --- PARAMETRI TEHNICI (1D DAILY) ---
TIMEFRAME = "1d"         # Grafic zilnic
LOOKBACK_DAYS = 365 * 2  # 2 ani de date istorice pentru calcul si backtest

SMA_MARKET_PERIOD = 200  # SPY > SMA 200 pentru Market Bullish
SMA_STOCK_PERIOD = 50    # Stock > SMA 50 pentru Trend Ascendent
EMA_PULLBACK_PERIOD = 20 # EMA 20 - suportul de pullback
RSI_PERIOD = 14          # RSI pe 14 lumanari
RSI_PULLBACK_MIN = 38.0  # Zona inferioara sanatoasa pentru retragere
RSI_PULLBACK_MAX = 56.0  # Zona superioara sanatoasa pentru retragere
VOLUME_AVG_PERIOD = 20   # Media mobila a volumului
VOLUME_SURGE_MULTIPLIER = 1.25  # Factor de volum exploziv la spargere (125%)

# --- FILTRE PRO: RELATIVE STRENGTH (RS vs SPY) & VCP (VOLATILITY CONTRACTION) ---
USE_RELATIVE_STRENGTH = True
# Scor RS minim fata de SPY (evita actiunile care subperformeaza sever piata)
MIN_RS_VS_SPY_PULLBACK = -5.0   # Permite pullback-uri usoare pe lideri
MIN_RS_VS_SPY_BREAKOUT = 0.0    # La Breakout, actiunea TREBUIE sa bata SPY (RS >= 0)
MAX_BREAKOUT_EXTENSION_EMA20 = 0.12  # Nu cumparam breakout daca pretul e deja la >12% peste EMA 20 (Climax Top)

# --- FILTRU MULTI-TIMEFRAME (1W + 1D CONFLUENCE) ---
USE_WEEKLY_FILTER = True
WEEKLY_EMA_PERIOD = 10     # Weekly EMA 10 (Trend intermediar saptamanal ~50 zile)
WEEKLY_SMA_PERIOD = 30     # Weekly SMA 30 (Trend primar Weinstein Stage 2)
WEEKLY_RSI_MIN = 40.0      # Evita actiunile blocate in downtrend structural saptamanal
WEEKLY_RSI_MAX = 72.0      # Evita cumpararea la supra-extindere saptamanala

# --- CONFIGURARE BROKER (ALPACA) ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "True").lower() in ("true", "1", "yes")

# --- CONFIGURARE NOTIFICARI TELEGRAM ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# --- AUTOMATIZARE & SCHEDULER ---
AUTO_TRADE = os.getenv("AUTO_TRADE", "False").lower() in ("true", "1", "yes")
SCAN_INTERVAL_MINUTES = 15

# --- FILTRU EARNINGS (ANTI-BINARY RISK) ---
EARNINGS_BUFFER_DAYS = 7        # Blocheaza intrari noi cu <= 7 zile inainte de Earnings
EARNINGS_OPEN_POS_WARN_DAYS = 2 # Avertizeaza / strânge SL pentru pozitii deschise cu <= 2 zile inainte de Earnings

# --- MACHINE LEARNING (SIGNAL CONFIDENCE FILTER) ---
ML_ENABLED = True
ML_MIN_CONFIDENCE = 0.52
ML_MIN_TRAINING_SAMPLES = 80
