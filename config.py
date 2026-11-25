"""
Configurări centrale pentru Trading Bot (Institutional Swing Trading V2 — Validated Out-of-Sample).
Toate constantele strategiei, costurile reale de tranzacționare, limitele de risc,
Kill-Switch-ul și reziliența API sunt izolate strict aici (fără valori hardcodate în module).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Încărcare variabile de mediu din .env (dacă există)
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ==============================================================================
# 1. UNIVERSUL DE ACȚIUNI LICHIDE (MULTI-SECTOR STAGE 2 INSTITUTIONAL LEADERS)
# ==============================================================================
# Diversificare reală pe sectoare S&P 500 / Nasdaq 100 (fără blocarea artificială
# a sectoarelor defensive/industriale când rotația capitalului instituțional se mută acolo,
# și fără acțiuni meme/cripto hiper-volatile care distrug Win Rate-ul).
WATCHLIST = [
    # 1. Semiconductori & Infrastructură AI
    "NVDA", "AVGO", "AMD", "TSM", "ANET", "KLAC", "LRCX", "AMAT", "QCOM", "ASML",
    # 2. Cloud, Big Tech & Enterprise Software
    "MSFT", "AAPL", "GOOGL", "AMZN", "META", "ORCL", "NOW", "CRM", "ADBE", "INTU",
    # 3. Cybersecurity
    "PANW", "CRWD", "FTNT",
    # 4. Industriale, Electrificare & Apărare
    "ETN", "CAT", "GE", "PWR", "URI", "LMT", "RTX",
    # 5. Finanțe & Plăți Globale
    "JPM", "GS", "V", "MA", "AXP", "BLK", "SPGI",
    # 6. Healthcare & MedTech Lideri
    "LLY", "ISRG", "BSX", "VRTX", "ABBV",
    # 7. Consumer & Retail Compounders
    "COST", "WMT", "TJX", "BKNG", "UBER", "NFLX"
]

TOP_LEADERS_COUNT = 25
DAILY_WATCHLIST_MAX_COUNT = 35

UNIVERSE_POOL = [
    # 1. Semiconductori & Hardware
    "NVDA", "AVGO", "AMD", "TSM", "MU", "ARM", "ANET", "MRVL", "ASML", "KLAC",
    "LRCX", "AMAT", "CRDO", "CLS", "FN", "MPWR", "QCOM", "ON", "SNPS", "CDNS", "TXN", "DELL",
    # 2. Software, Cloud, Cybersecurity & Big Tech
    "MSFT", "AAPL", "GOOGL", "AMZN", "META", "PLTR", "ORCL", "NOW", "APP", "SNOW",
    "CRM", "ADBE", "INTU", "IBM", "CSCO", "DDOG", "MDB", "IOT", "TEAM", "HUBS",
    "PANW", "CRWD", "FTNT", "NET", "ZS", "CYBR", "S",
    # 3. Industriale, Power Grid & Aerospace
    "ETN", "PWR", "GE", "CAT", "URI", "AXON", "GEV", "VST", "CEG", "VRT", "LMT", "RTX",
    # 4. Finanțe & Capital Markets
    "JPM", "GS", "MS", "KKR", "APO", "AXP", "V", "MA", "BLK", "SPGI", "BX", "SCHW",
    # 5. Healthcare & Biotech Lideri
    "LLY", "ISRG", "ABBV", "VRTX", "REGN", "BSX", "SYK", "UNH",
    # 6. Consumer, Retail, Travel & Digital Leaders
    "COST", "WMT", "TJX", "HD", "KO", "DIS", "NKE", "DAL", "UAL", "RCL",
    "NFLX", "UBER", "BKNG", "ABNB", "SHOP", "SPOT", "MELI", "DASH", "DKNG", "TSLA",
    # 7. Energie & Materiale Lideri
    "XOM", "CVX", "SLB", "FCX"
]

BENCHMARK_TICKER = "SPY"
SECONDARY_BENCHMARK = "QQQ"
VIX_TICKER = "^VIX"

# ==============================================================================
# 2. MANAGEMENTUL RISCULUI, POSITION SIZING & PORTFOLIO HEAT (NON-NEGOCIABIL)
# ==============================================================================
# Risc dinamic per tranzacție (între 1% și 2% din capitalul curent):
#   position_size = (total_capital * effective_risk_pct) / abs(entry_price - stop_loss)
RISK_PER_TRADE_PERCENT = 0.0175      # 1.75% risc pe tranzacție în Power Trend (strict în intervalul 1% - 2%)
CAUTIOUS_RISK_MULTIPLIER = 0.70      # ~1.22% risc pe tranzacție în Cautious Uptrend

DEFAULT_SIMULATED_CAPITAL = 1000.0   # Capital inițial implicit pentru simulare ($1,000)
ACCOUNT_BUDGET_CAP = 1000.0          # Plafon buget activ (+ PnL real) dacă contul Paper are $100k
ALLOW_FRACTIONAL_SHARES = True

# Limite stricte de expunere la nivel de portofoliu și sector
MAX_OPEN_POSITIONS = 4               # Max 4 poziții cu RISC ACTIV simultan (Portfolio Heat <= 6%)
MAX_TOTAL_POSITIONS = 5              # Plafon absolut poziții deschise (inclusiv cele la Breakeven)
MAX_POSITIONS_PER_SECTOR = 2         # Previne concentrarea pe acțiuni puternic corelate
MAX_SINGLE_POSITION_PCT = 0.33       # Max 33% din capitalul total într-o singură acțiune
MIN_POSITION_ALLOC_PCT = 0.12        # Nu deschide poziții fărâmițate sub 12% din capital (~$120)
MIN_POSITION_COST_USD = 50.0         # Prag minim absolut în USD per ordin

# Praguri obligatorii de Risk / Reward & Scale-Out
MIN_RISK_REWARD_RATIO = 2.0          # Prag minim obligatoriu R:R până la ținta principală (1:2.0)
TP1_RISK_REWARD_RATIO = 1.5          # TP1 la 1.5R: închide 50% + mută Stop-Loss la Breakeven
TP2_RISK_REWARD_RATIO = 2.5          # TP2 la 2.5R: ținta Runner-ului (urmărit cu Trailing pe EMA 20)
TP1_EXIT_PERCENT = 0.50              # 50% din poziție marcată la TP1
BREAKEVEN_BUFFER_PCT = 0.0020        # +0.20% peste prețul de intrare (acoperă spread/comision)
EMA20_TRAIL_BUFFER_PCT = 0.015       # Trail Runner la 1.5% sub EMA 20 zilnic după TP1
USE_SPLIT_ORDERS = True

# Limite structurale Stop-Loss (fără tăiere artificială în aer!)
MIN_STOP_LOSS_PCT = 0.022            # Minim 2.2% distanță SL (evită zgomotul intraday)
MAX_STOP_LOSS_PCT = 0.075            # Maxim 7.5% distanță SL (dacă suportul e mai jos, respinge semnalul!)
ATR_STOP_MULTIPLIER = 1.8            # Buffer minim de volatilitate (1.8x ATR_14)

# ==============================================================================
# 3. KILL-SWITCH AUTOMAT & SIGURANȚĂ LA EXECUȚIE (API RESILIENCE)
# ==============================================================================
KILL_SWITCH_ENABLED = True
MAX_DAILY_LOSS_PCT = 0.03            # Oprește intrările noi dacă pierderea zilei >= 3.0%
MAX_PORTFOLIO_DRAWDOWN_PCT = 0.10    # Oprește intrările noi dacă drawdown-ul față de HWM >= 10.0%

# Protecție la deschiderea bursei & alunecare la intrare
MARKET_OPEN_BUFFER_MINUTES = 15      # Nu scanează intrări noi în primele 15 min (09:30 - 09:45 ET)
POSITION_MGR_OPEN_GUARD_MINUTES = 10 # Evită vânzări Market în spread-ul haotic din primele 10 min (09:30 - 09:40 ET)
PULLBACK_LIMIT_DISCOUNT_PCT = 0.003  # Intrare Limit cu -0.3% sub Close pe Pullback (nu urmărește gap-up)
MAX_ENTRY_GAP_PCT = 0.012            # Anulează intrarea dacă prețul sare cu > +1.2% peste semnal

# Reziliență API (Timeout, Retry & Exponential Backoff)
API_TIMEOUT_SECONDS = 10.0
API_MAX_RETRIES = 3
API_BACKOFF_BASE_SECONDS = 1.0

# ==============================================================================
# 4. COSTURI REALE DE TRANZACȚIONARE ÎN BACKTEST (ANTI-ILLUSION)
# ==============================================================================
BACKTEST_SPREAD_PCT = 0.0008         # 0.08% spread bid/ask complet (0.04% half-spread pe sens)
BACKTEST_SLIPPAGE_PCT = 0.0006       # 0.06% slippage defavorabil pe sens (total cost variabil: 0.10%/sens)
BACKTEST_COMMISSION_USD = 0.02       # Taxe SEC/FINRA TAF / execuție per ordin

# ==============================================================================
# 5. PARAMETRI TEHNICI ZILNICI (1D) & MULTI-TIMEFRAME (1W) — LUMÂNĂRI ÎNCHISE
# ==============================================================================
TIMEFRAME = "1d"
LOOKBACK_DAYS = 365 * 2
CLOSED_CANDLES_ONLY = True           # ZERO Look-Ahead Bias: calculează semnalele exclusiv pe bare 1D închise

SMA_MARKET_PERIOD = 200
SMA_STOCK_PERIOD = 50
EMA_PULLBACK_PERIOD = 20
RSI_PERIOD = 14
RSI_FAST_PERIOD = 4                  # RSI(4) Wilder pentru detectarea retragerilor ordonate în Stage 2

RSI4_PULLBACK_MAX = 30.0             # Prag platou validat OOS (28-32): epuizarea vânzătorilor pe pullback
RSI_PULLBACK_MIN = 38.0              # Zona sănătoasă de suport în trend ascendent
RSI_PULLBACK_MAX = 54.0
PULLBACK_MAX_VOL_RATIO_3D = 1.05     # Volume Dry-Up: volumul pe corecție nu depășește media

VOLUME_AVG_PERIOD = 20
MIN_AVG_VOLUME = 700_000             # Lichiditate instituțională minimă (700k acțiuni/zi)
VOLUME_SURGE_MULTIPLIER = 1.50       # Confirmare Breakout: volum >= 150% din media pe 20 zile
MAX_ATR_PCT = 4.3                    # Exclude acțiunile haotice/parabolice cu ATR zilnic > 4.3%
MIN_ATR_PCT = 1.0

# Filtre Regim Piață (SPY + QQQ + VIX)
VIX_POWER_TREND_MAX = 21.5
VIX_BEARISH_THRESHOLD = 25.0         # Peste VIX 25 blocăm intrările noi Long

# Filtre Relative Strength (RS vs SPY) & VCP
USE_RELATIVE_STRENGTH = True
ENABLE_BREAKOUT_SETUP = False        # Dezactivat implicit: Pullback-ul Ordonat Stage 2 are Win Rate (56-58% poz / 72% lot) și Sharpe net superioare OOS!
MIN_RS_VS_SPY_PULLBACK = 2.0         # Cumpărăm doar lideri care bat S&P 500 (RS >= +2%)
MAX_RS_VS_SPY_PARABOLIC = 45.0       # Evită cumpărarea în blow-off tops parabolice
MIN_RS_VS_SPY_BREAKOUT = 3.0
MAX_BREAKOUT_EXTENSION_EMA20 = 0.07  # Nu cumpărăm breakout extins la >7% peste EMA 20
MAX_VCP_ATR_CONTRACTION = 0.85       # Contracție strictă a volatilității pentru Breakout
MAX_VCP_BASE_RANGE_PCT = 8.0         # Bază strânsă pe 3 săptămâni (<= 8% amplitudine)

# Filtru Multi-Timeframe Săptămânal (1W — shift(1) săptămână complet închisă)
USE_WEEKLY_FILTER = True
WEEKLY_EMA_PERIOD = 10
WEEKLY_SMA_PERIOD = 30
WEEKLY_SMA50_PERIOD = 50
WEEKLY_RSI_MIN = 45.0
WEEKLY_RSI_MAX = 72.0

# Filtru Earnings & Știri
EARNINGS_BUFFER_DAYS = 7
EARNINGS_OPEN_POS_WARN_DAYS = 2
USE_ALPACA_LIVE_DATA = True          # Folosit DOAR la execuție/cotație live, NICIODATĂ la poluarea lumânărilor 1D!
USE_NEWS_FILTER = True
USE_SECTOR_ROTATION = True

# Filtru Machine Learning (cu Poartă Automată de Calitate Out-of-Sample)
ML_ENABLED = True
ML_MIN_CONFIDENCE = 0.53
ML_MIN_TRAINING_SAMPLES = 80
ML_MIN_OOS_ROC_AUC = 0.55            # Dezactivează automat respingerea ML dacă ROC-AUC CV < 0.55 (zgomot)

# ==============================================================================
# 6. CREDENȚIALE, MOD DE EXECUȚIE & VALIDARE STARTUP (FAIL-FAST)
# ==============================================================================
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "").strip()
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "").strip()
_RAW_ALPACA_PAPER = os.getenv("ALPACA_PAPER", "True").strip().lower()
ALPACA_PAPER = _RAW_ALPACA_PAPER in ("true", "1", "yes")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

AUTO_TRADE = os.getenv("AUTO_TRADE", "True").strip().lower() in ("true", "1", "yes")
SCAN_INTERVAL_MINUTES = 15
CLOUD_STANDBY_MAX_HOURS = 17         # Mentine containerul Cloud in Standby pana la 17h inainte de deschiderea bursei SUA


def validate_runtime_environment(require_broker: bool = False) -> dict:
    """
    Validează explicit variabilele de mediu la pornire (Fail-Fast pentru execuția Live):
      - Verifică neechivoc modul de execuție ('paper' vs 'live').
      - Oprește imediat pornirea în mod 'live' (bani reali) dacă ALPACA_PAPER are o valoare
        ambiguă sau dacă lipsesc cheile API valide.
    """
    raw_mode = os.getenv("ALPACA_PAPER", "True").strip().lower()
    valid_true = ("true", "1", "yes")
    valid_false = ("false", "0", "no")

    if raw_mode not in valid_true + valid_false:
        raise RuntimeError(
            f"Configurație invalidă/ambiguă pentru ALPACA_PAPER='{raw_mode}'. "
            f"Setează explicit ALPACA_PAPER=True (Paper) sau ALPACA_PAPER=False (Live) în .env."
        )

    is_paper = raw_mode in valid_true
    mode_name = "paper" if is_paper else "live"
    has_keys = bool(ALPACA_API_KEY and ALPACA_SECRET_KEY and "XXXX" not in ALPACA_API_KEY)

    if not is_paper and not has_keys:
        raise RuntimeError(
            "EXECUȚIE LIVE BLOCATĂ (FAIL-FAST): ALPACA_PAPER=False (bani reali), "
            "dar cheile ALPACA_API_KEY / ALPACA_SECRET_KEY lipsesc sau sunt invalide!"
        )

    if not is_paper and ALPACA_API_KEY.upper().startswith("PK"):
        raise RuntimeError(
            "EXECUȚIE LIVE BLOCATĂ (FAIL-FAST): ALPACA_PAPER=False, dar cheia API începe cu 'PK' (cheie Paper)!"
        )

    if require_broker and not has_keys:
        raise RuntimeError(
            "Conexiunea la broker este obligatorie pentru acest mod, dar cheile Alpaca API lipsesc din .env."
        )

    return {
        "mode": mode_name,
        "is_paper": is_paper,
        "has_keys": has_keys,
    }


# ==============================================================================
# 7. MAPARE SECTOARE & LIMITE DE CORELAȚIE
# ==============================================================================
BLOCKED_SECTORS: set = set()  # Niciun sector S&P 500 nu este blocat orb; selecția se face pe Stage 2 + RS/ATR

SECTOR_PRIORITY_BONUS = {
    "SEMICONDUCTORS": 0.0,
    "SOFTWARE_CLOUD": 0.0,
    "CYBERSECURITY": 0.0,
    "AI_POWER_INFRA": 0.0,
    "INDUSTRIALS_POWER": 0.0,
    "FINANCIALS": 0.0,
    "HEALTHCARE": 0.0,
    "CONSUMER_RETAIL": 0.0,
    "HIGH_BETA_GROWTH": 0.0,
    "ENERGY_MATERIALS": 0.0,
    "DEFENSIVE_STAPLES": 0.0,
    "OTHER": 0.0,
}

SECTOR_MAX_POSITIONS = {
    "SEMICONDUCTORS": 2,
    "SOFTWARE_CLOUD": 2,
    "CYBERSECURITY": 2,
    "AI_POWER_INFRA": 2,
    "INDUSTRIALS_POWER": 2,
    "FINANCIALS": 2,
    "HEALTHCARE": 2,
    "CONSUMER_RETAIL": 2,
    "HIGH_BETA_GROWTH": 2,
    "ENERGY_MATERIALS": 1,
    "DEFENSIVE_STAPLES": 2,
}

SECTOR_WATCHLIST_QUOTA = {
    "SEMICONDUCTORS": 7,
    "SOFTWARE_CLOUD": 7,
    "CYBERSECURITY": 3,
    "AI_POWER_INFRA": 5,
    "FINANCIALS": 5,
    "HEALTHCARE": 4,
    "CONSUMER_RETAIL": 4,
    "HIGH_BETA_GROWTH": 3,
    "DEFENSIVE_STAPLES": 3,
    "ENERGY_MATERIALS": 2,
}

SECTOR_MAP = {
    # 1. Semiconductori & Networking
    "NVDA": "SEMICONDUCTORS", "AVGO": "SEMICONDUCTORS", "AMD": "SEMICONDUCTORS",
    "TSM": "SEMICONDUCTORS", "ASML": "SEMICONDUCTORS", "MU": "SEMICONDUCTORS",
    "ANET": "SEMICONDUCTORS", "MRVL": "SEMICONDUCTORS", "ARM": "SEMICONDUCTORS",
    "KLAC": "SEMICONDUCTORS", "LRCX": "SEMICONDUCTORS", "AMAT": "SEMICONDUCTORS",
    "QCOM": "SEMICONDUCTORS", "TXN": "SEMICONDUCTORS", "CRDO": "SEMICONDUCTORS",
    "CLS": "SEMICONDUCTORS", "FN": "SEMICONDUCTORS", "SMCI": "SEMICONDUCTORS",
    "MPWR": "SEMICONDUCTORS", "ON": "SEMICONDUCTORS", "SNPS": "SEMICONDUCTORS",
    "CDNS": "SEMICONDUCTORS",
    # 2. Software, Cloud & Big Tech
    "MSFT": "SOFTWARE_CLOUD", "AAPL": "SOFTWARE_CLOUD", "GOOGL": "SOFTWARE_CLOUD",
    "AMZN": "SOFTWARE_CLOUD", "META": "SOFTWARE_CLOUD", "PLTR": "SOFTWARE_CLOUD",
    "CRM": "SOFTWARE_CLOUD", "NOW": "SOFTWARE_CLOUD", "ORCL": "SOFTWARE_CLOUD",
    "APP": "SOFTWARE_CLOUD", "ADBE": "SOFTWARE_CLOUD", "INTU": "SOFTWARE_CLOUD",
    "IBM": "SOFTWARE_CLOUD", "SNOW": "SOFTWARE_CLOUD", "DELL": "SOFTWARE_CLOUD",
    "CSCO": "SOFTWARE_CLOUD", "DDOG": "SOFTWARE_CLOUD", "MDB": "SOFTWARE_CLOUD",
    "RDDT": "SOFTWARE_CLOUD", "IOT": "SOFTWARE_CLOUD", "TEAM": "SOFTWARE_CLOUD",
    "HUBS": "SOFTWARE_CLOUD", "MSTR": "SOFTWARE_CLOUD",
    # 3. Cybersecurity
    "PANW": "CYBERSECURITY", "CRWD": "CYBERSECURITY", "NET": "CYBERSECURITY",
    "ZS": "CYBERSECURITY", "FTNT": "CYBERSECURITY", "CYBR": "CYBERSECURITY", "S": "CYBERSECURITY",
    # 4. Industriale, Electrificare & Infrastructură
    "VRT": "AI_POWER_INFRA", "VST": "AI_POWER_INFRA", "CEG": "AI_POWER_INFRA",
    "GEV": "AI_POWER_INFRA", "ETN": "AI_POWER_INFRA", "PWR": "AI_POWER_INFRA",
    "GE": "AI_POWER_INFRA", "CAT": "AI_POWER_INFRA", "URI": "AI_POWER_INFRA", "AXON": "AI_POWER_INFRA",
    "LMT": "INDUSTRIALS_POWER", "RTX": "INDUSTRIALS_POWER",
    # 5. Growth & Consumer Tech
    "TSLA": "HIGH_BETA_GROWTH", "NFLX": "HIGH_BETA_GROWTH", "UBER": "HIGH_BETA_GROWTH",
    "SHOP": "HIGH_BETA_GROWTH", "SPOT": "HIGH_BETA_GROWTH", "MELI": "HIGH_BETA_GROWTH",
    "DASH": "HIGH_BETA_GROWTH", "BKNG": "CONSUMER_RETAIL", "ABNB": "CONSUMER_RETAIL",
    "DKNG": "HIGH_BETA_GROWTH", "HOOD": "HIGH_BETA_GROWTH", "COIN": "HIGH_BETA_GROWTH",
    "SOFI": "HIGH_BETA_GROWTH", "NU": "HIGH_BETA_GROWTH", "AFRM": "HIGH_BETA_GROWTH",
    "XYZ": "HIGH_BETA_GROWTH",
    # 6. Finanțe
    "GS": "FINANCIALS", "KKR": "FINANCIALS", "APO": "FINANCIALS", "AXP": "FINANCIALS",
    "V": "FINANCIALS", "MA": "FINANCIALS", "JPM": "FINANCIALS", "MS": "FINANCIALS",
    "BLK": "FINANCIALS", "SPGI": "FINANCIALS", "BX": "FINANCIALS", "SCHW": "FINANCIALS",
    # 7. Healthcare
    "LLY": "HEALTHCARE", "ISRG": "HEALTHCARE", "ABBV": "HEALTHCARE", "VRTX": "HEALTHCARE",
    "REGN": "HEALTHCARE", "BSX": "HEALTHCARE", "SYK": "HEALTHCARE", "UNH": "HEALTHCARE",
    # 8. Energie & Materiale
    "XOM": "ENERGY_MATERIALS", "CVX": "ENERGY_MATERIALS", "SLB": "ENERGY_MATERIALS", "FCX": "ENERGY_MATERIALS",
    # 9. Consumer & Retail Leaders
    "WMT": "DEFENSIVE_STAPLES", "COST": "DEFENSIVE_STAPLES", "KO": "DEFENSIVE_STAPLES",
    "DIS": "CONSUMER_RETAIL", "NKE": "CONSUMER_RETAIL", "HD": "CONSUMER_RETAIL",
    "TJX": "CONSUMER_RETAIL", "DAL": "CONSUMER_RETAIL", "UAL": "CONSUMER_RETAIL", "RCL": "CONSUMER_RETAIL",
}


def get_ticker_sector(ticker: str) -> str:
    """Returnează sectorul unui simbol pentru controlul rotației și corelației."""
    return SECTOR_MAP.get(ticker.upper(), "OTHER")


def is_sector_blocked(sector: str) -> bool:
    """Returnează True doar dacă sectorul este blocat explicit în configurație."""
    return sector.upper() in BLOCKED_SECTORS or SECTOR_MAX_POSITIONS.get(sector.upper(), MAX_POSITIONS_PER_SECTOR) <= 0


def get_sector_max_positions(sector: str) -> int:
    """Returnează numărul maxim de poziții simultane permise într-un sector."""
    return SECTOR_MAX_POSITIONS.get(sector.upper(), MAX_POSITIONS_PER_SECTOR)


def get_sector_priority_bonus(sector: str) -> float:
    """Returnează bonusul structural de sector."""
    return SECTOR_PRIORITY_BONUS.get(sector.upper(), 0.0)
