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

# --- LISTA DE ACTIUNI PENTRU SCANARE (HIGH-ALPHA AI, TECH & MOMENTUM LEADERS) ---
# Portofoliu institutional concentrat pe sectoarele unde curg banii institutionali (Money Flow):
# 1. Semiconductori & Infrastructura Hardware AI
# 2. Mega-Cap Tech, Cloud & AI Software
# 3. Cybersecurity & Cloud Zero-Trust
# 4. Infrastructura Data-Centers, Power Grid AI & Electrificare
# 5. High-Beta Digital Growth, E-Commerce & FinTech Leaders
# (Sectoarele lente/defensive precum Healthcare/Pharma, Consumer Staples, Airlines au fost eliminate)
WATCHLIST = [
    # 1. AI Semiconductors, Chip Design & High-Speed Networking (Inima Revolutiei AI)
    "NVDA", "AVGO", "AMD", "TSM", "MU", "ARM", "ANET", "MRVL", "ASML", "KLAC", "CRDO",
    # 2. Mega-Cap Tech, Cloud & Enterprise AI Software
    "PLTR", "META", "MSFT", "AMZN", "GOOGL", "AAPL", "ORCL", "NOW", "APP", "SNOW", "CRM", "RDDT",
    # 3. Cybersecurity & Cloud Protection
    "CRWD", "PANW", "NET", "ZS",
    # 4. AI Data-Center Power, Cooling & Grid Infrastructure
    "VRT", "VST", "CEG", "GEV", "ETN",
    # 5. High-Beta Digital Growth, E-Commerce & FinTech Attackers
    "TSLA", "NFLX", "SHOP", "SPOT", "HOOD", "COIN"
]

# Numarul de lideri selectati de Screener-ul Dinamic (Optiunea 3 din Unelte)
TOP_LEADERS_COUNT = 25

# Univers extins (~75 companii High-Alpha S&P 500 / Nasdaq 100) exclusiv din sectoare de Momentum & Crestere
UNIVERSE_POOL = [
    # 1. AI Semiconductors, Hardware, Storage & Networking
    "NVDA", "AVGO", "AMD", "TSM", "MU", "ARM", "ANET", "MRVL", "ASML", "KLAC",
    "LRCX", "AMAT", "CRDO", "CLS", "MPWR", "QCOM", "ON", "SNPS", "CDNS", "SMCI", "FN", "DELL",
    # 2. AI Software, Cloud, Big Tech & Data Analytics
    "PLTR", "META", "MSFT", "AMZN", "GOOGL", "AAPL", "ORCL", "NOW", "APP", "SNOW",
    "CRM", "DDOG", "MDB", "RDDT", "IOT", "ADBE", "INTU", "IBM", "TEAM", "HUBS", "MSTR",
    # 3. Cybersecurity
    "CRWD", "PANW", "NET", "ZS", "FTNT", "CYBR", "S",
    # 4. AI Power Grid, Nuclear Energy for Data Centers, Cooling & Electrification
    "VRT", "VST", "CEG", "GEV", "ETN", "PWR", "GE", "CAT", "URI", "AXON",
    # 5. High-Beta Consumer Tech, E-Commerce & FinTech Momentum
    "TSLA", "NFLX", "SHOP", "SPOT", "UBER", "MELI", "DASH", "DKNG", "ABNB", "BKNG",
    "HOOD", "COIN", "SOFI", "NU", "AFRM", "XYZ", "GS", "KKR", "APO", "AXP", "V", "MA"
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
AUTO_TRADE = os.getenv("AUTO_TRADE", "True").lower() in ("true", "1", "yes")
SCAN_INTERVAL_MINUTES = 15

# --- FILTRU EARNINGS (ANTI-BINARY RISK) ---
EARNINGS_BUFFER_DAYS = 7        # Blocheaza intrari noi cu <= 7 zile inainte de Earnings
EARNINGS_OPEN_POS_WARN_DAYS = 2 # Avertizeaza / strânge SL pentru pozitii deschise cu <= 2 zile inainte de Earnings

# --- MACHINE LEARNING (SIGNAL CONFIDENCE FILTER) ---
ML_ENABLED = True
ML_MIN_CONFIDENCE = 0.52
ML_MIN_TRAINING_SAMPLES = 80

# --- FUNCTII QUANT PRO 10/10: SECTOR ROTATION (ATAC PE AI & TECH), DATE LIVE ALPACA & FILTRU STIRI ---
MAX_POSITIONS_PER_SECTOR = 2    # Limita implicita pe sectoare secundare
USE_ALPACA_LIVE_DATA = True     # Preia pretul intraday in timp real (IEX/SIP) direct din Alpaca API
USE_NEWS_FILTER = True          # Analizeaza stirile din ultimele 48h prin Alpaca News API
USE_SECTOR_ROTATION = True      # Prioritizeaza sectoarele Fierbinti (Hot Sectors) cu cel mai puternic Money Flow

# Sectoare Defensive / Lente interzise la deschiderea de pozitii noi (nu blocam capitalul in Pharma/Utilitati/Oil)
BLOCKED_SECTORS = {"HEALTHCARE", "ENERGY_MATERIALS", "DEFENSIVE_STAPLES"}

# Bonus de Prioritate Alpha in Scorul Compozit (0-100) pentru concentrarea focurilor unde se fac banii:
SECTOR_PRIORITY_BONUS = {
    "SEMICONDUCTORS": 14.0,     # Tier 1 Alpha: Cipuri AI, Hardware & Networking
    "SOFTWARE_CLOUD": 12.0,     # Tier 1 Alpha: Cloud AI, Enterprise Software & Big Tech
    "CYBERSECURITY": 10.0,      # Tier 1 Alpha: Securitate Cibernetica
    "AI_POWER_INFRA": 9.0,      # Tier 1 Alpha: Data-Center Power Grid & Cooling
    "HIGH_BETA_GROWTH": 6.0,    # Tier 2: Consumer Tech, E-Commerce & FinTech Momentum
    "INDUSTRIALS_POWER": 5.0,   # Alias compatibilitate
    "CONSUMER_RETAIL": 3.0,     # Alias compatibilitate
    "FINANCIALS": 2.0,          # Tier 3: Doar cand au RS exceptional
    "HEALTHCARE": -25.0,        # Blocat / Penalizat sever
    "ENERGY_MATERIALS": -20.0,  # Blocat / Penalizat sever
    "DEFENSIVE_STAPLES": -25.0, # Blocat / Penalizat sever
    "OTHER": 0.0,
}

# Limita maxima de pozitii simultane per sector (permitem pana la 3 pozitii pe liderii AI/Tech!):
SECTOR_MAX_POSITIONS = {
    "SEMICONDUCTORS": 3,
    "SOFTWARE_CLOUD": 3,
    "CYBERSECURITY": 2,
    "AI_POWER_INFRA": 2,
    "HIGH_BETA_GROWTH": 2,
    "FINANCIALS": 1,
    "HEALTHCARE": 0,
    "ENERGY_MATERIALS": 0,
    "DEFENSIVE_STAPLES": 0,
}

# Cota maxima de simboluri in Watchlist-ul Zilnic (35 total) — pondere masiva pe Tech & AI:
SECTOR_WATCHLIST_QUOTA = {
    "SEMICONDUCTORS": 12,
    "SOFTWARE_CLOUD": 11,
    "CYBERSECURITY": 4,
    "AI_POWER_INFRA": 5,
    "HIGH_BETA_GROWTH": 6,
    "FINANCIALS": 2,
    "HEALTHCARE": 0,
    "ENERGY_MATERIALS": 0,
    "DEFENSIVE_STAPLES": 0,
}

SECTOR_MAP = {
    # 1. Semiconductori, Cipuri AI, Stocare & High-Speed Networking (Tier 1 Alpha)
    "NVDA": "SEMICONDUCTORS", "AVGO": "SEMICONDUCTORS", "AMD": "SEMICONDUCTORS",
    "TSM": "SEMICONDUCTORS", "ASML": "SEMICONDUCTORS", "MU": "SEMICONDUCTORS",
    "ANET": "SEMICONDUCTORS", "MRVL": "SEMICONDUCTORS", "ARM": "SEMICONDUCTORS",
    "KLAC": "SEMICONDUCTORS", "LRCX": "SEMICONDUCTORS", "AMAT": "SEMICONDUCTORS",
    "QCOM": "SEMICONDUCTORS", "TXN": "SEMICONDUCTORS", "CRDO": "SEMICONDUCTORS",
    "CLS": "SEMICONDUCTORS", "FN": "SEMICONDUCTORS", "SMCI": "SEMICONDUCTORS",
    "MPWR": "SEMICONDUCTORS", "ON": "SEMICONDUCTORS", "SNPS": "SEMICONDUCTORS",
    "CDNS": "SEMICONDUCTORS",
    # 2. Software, Cloud, Big Tech AI & Data (Tier 1 Alpha)
    "MSFT": "SOFTWARE_CLOUD", "AAPL": "SOFTWARE_CLOUD", "GOOGL": "SOFTWARE_CLOUD",
    "AMZN": "SOFTWARE_CLOUD", "META": "SOFTWARE_CLOUD", "PLTR": "SOFTWARE_CLOUD",
    "CRM": "SOFTWARE_CLOUD", "NOW": "SOFTWARE_CLOUD", "ORCL": "SOFTWARE_CLOUD",
    "APP": "SOFTWARE_CLOUD", "ADBE": "SOFTWARE_CLOUD", "INTU": "SOFTWARE_CLOUD",
    "IBM": "SOFTWARE_CLOUD", "SNOW": "SOFTWARE_CLOUD", "DELL": "SOFTWARE_CLOUD",
    "CSCO": "SOFTWARE_CLOUD", "DDOG": "SOFTWARE_CLOUD", "MDB": "SOFTWARE_CLOUD",
    "RDDT": "SOFTWARE_CLOUD", "IOT": "SOFTWARE_CLOUD", "TEAM": "SOFTWARE_CLOUD",
    "HUBS": "SOFTWARE_CLOUD", "MSTR": "SOFTWARE_CLOUD",
    # 3. Cybersecurity (Tier 1 Alpha)
    "PANW": "CYBERSECURITY", "CRWD": "CYBERSECURITY", "NET": "CYBERSECURITY",
    "ZS": "CYBERSECURITY", "FTNT": "CYBERSECURITY", "CYBR": "CYBERSECURITY", "S": "CYBERSECURITY",
    # 4. Infrastructura AI Data-Centers, Energie Nucleara & Electrificare (Tier 1 Alpha)
    "VRT": "AI_POWER_INFRA", "VST": "AI_POWER_INFRA", "CEG": "AI_POWER_INFRA",
    "GEV": "AI_POWER_INFRA", "ETN": "AI_POWER_INFRA", "PWR": "AI_POWER_INFRA",
    "GE": "AI_POWER_INFRA", "CAT": "AI_POWER_INFRA", "URI": "AI_POWER_INFRA", "AXON": "AI_POWER_INFRA",
    # 5. High-Beta Consumer Tech, E-Commerce & FinTech Attackers (Tier 2 Momentum)
    "TSLA": "HIGH_BETA_GROWTH", "NFLX": "HIGH_BETA_GROWTH", "UBER": "HIGH_BETA_GROWTH",
    "SHOP": "HIGH_BETA_GROWTH", "SPOT": "HIGH_BETA_GROWTH", "MELI": "HIGH_BETA_GROWTH",
    "DASH": "HIGH_BETA_GROWTH", "BKNG": "HIGH_BETA_GROWTH", "ABNB": "HIGH_BETA_GROWTH",
    "DKNG": "HIGH_BETA_GROWTH", "HOOD": "HIGH_BETA_GROWTH", "COIN": "HIGH_BETA_GROWTH",
    "SOFI": "HIGH_BETA_GROWTH", "NU": "HIGH_BETA_GROWTH", "AFRM": "HIGH_BETA_GROWTH",
    "XYZ": "HIGH_BETA_GROWTH",
    # 6. Finante & Capital Markets (Tier 3 - Selectiv)
    "GS": "FINANCIALS", "KKR": "FINANCIALS", "APO": "FINANCIALS", "AXP": "FINANCIALS",
    "V": "FINANCIALS", "MA": "FINANCIALS", "JPM": "FINANCIALS", "MS": "FINANCIALS",
    "BLK": "FINANCIALS", "SPGI": "FINANCIALS", "BX": "FINANCIALS", "SCHW": "FINANCIALS",
    # 7. Sectoare Defensive / Vechi (Mapate pentru blocare automata daca apar in istoric)
    "LLY": "HEALTHCARE", "ISRG": "HEALTHCARE", "ABBV": "HEALTHCARE", "VRTX": "HEALTHCARE",
    "REGN": "HEALTHCARE", "BSX": "HEALTHCARE", "SYK": "HEALTHCARE", "UNH": "HEALTHCARE",
    "XOM": "ENERGY_MATERIALS", "CVX": "ENERGY_MATERIALS", "SLB": "ENERGY_MATERIALS", "FCX": "ENERGY_MATERIALS",
    "WMT": "DEFENSIVE_STAPLES", "COST": "DEFENSIVE_STAPLES", "KO": "DEFENSIVE_STAPLES",
    "DIS": "DEFENSIVE_STAPLES", "NKE": "DEFENSIVE_STAPLES", "HD": "DEFENSIVE_STAPLES",
    "TJX": "DEFENSIVE_STAPLES", "LMT": "DEFENSIVE_STAPLES", "RTX": "DEFENSIVE_STAPLES",
    "DAL": "DEFENSIVE_STAPLES", "UAL": "DEFENSIVE_STAPLES", "RCL": "DEFENSIVE_STAPLES",
}


def get_ticker_sector(ticker: str) -> str:
    """Returneaza sectorul institutional al unui simbol pentru controlul rotatiei si corelatiei."""
    return SECTOR_MAP.get(ticker.upper(), "OTHER")


def is_sector_blocked(sector: str) -> bool:
    """Returneaza True daca sectorul este blocat (ex: HEALTHCARE, DEFENSIVE_STAPLES)."""
    return sector.upper() in BLOCKED_SECTORS or SECTOR_MAX_POSITIONS.get(sector.upper(), 2) <= 0


def get_sector_max_positions(sector: str) -> int:
    """Returneaza numarul maxim de pozitii simultane permise intr-un anumit sector."""
    return SECTOR_MAX_POSITIONS.get(sector.upper(), MAX_POSITIONS_PER_SECTOR)


def get_sector_priority_bonus(sector: str) -> float:
    """Returneaza bonusul de prioritate Alpha al sectorului (pentru concentrarea pe AI & Tech)."""
    return SECTOR_PRIORITY_BONUS.get(sector.upper(), 0.0)


