"""
Modul pentru descarcarea, curatarea datelor istorice si analiza Regimului de Piata.
Functionalitati Pro:
  - Cache inteligent pentru benchmark-ul SPY (calculeaza automat RS_vs_SPY pentru fiecare actiune).
  - Regim de Piata Multidimensional (SPY + QQQ + ^VIX) pe 3 paliere:
      1. POWER_TREND (Risc 100%, Pullbacks + Breakouts)
      2. CAUTIOUS_UPTREND (Risc redus, selectivitate ridicata)
      3. BEARISH (Cash / Protectie capital)
  - Filtru de Earnings Date (Anti-Binary Risk) pentru intrari noi si pozitii deja deschise.
  - Screener Dinamic de Forta Relativa (Top RS Stage 2 Leaders).
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
import requests
import pandas as pd
import yfinance as yf
import config
from indicators import add_all_indicators

# Cache in memorie pentru SPY si pentru Stirile Alpaca
_SPY_CACHE: Dict[str, Tuple[float, pd.DataFrame]] = {}
_SPY_CACHE_TTL_SECONDS = 900  # 15 minute
_NEWS_CACHE: Dict[str, Tuple[float, Tuple[bool, float, str, str]]] = {}
_NEWS_CACHE_TTL_SECONDS = 900


def get_alpaca_live_snapshot(ticker: str) -> Optional[dict]:
    """
    Preia in timp real (0-latency IEX/SIP) ultimul pret si bara zilnica curenta
    direct din API-ul Alpaca Market Data v2.
    """
    if not getattr(config, "USE_ALPACA_LIVE_DATA", True):
        return None
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY or ticker.startswith("^"):
        return None

    url = f"https://data.alpaca.markets/v2/stocks/{ticker.upper()}/snapshot"
    headers = {
        "APCA-API-KEY-ID": config.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
    }
    try:
        res = requests.get(url, headers=headers, timeout=4)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None


def _download_raw_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Descarca datele istorice OHLCV de pe Yahoo Finance si suprapune automat
    ultimul pret/bara in timp real din Alpaca Market Data API (IEX/SIP).
    """
    df = yf.download(
        ticker,
        period=period,
        progress=False,
        auto_adjust=True
    )

    if df.empty:
        raise ValueError(f"Nu s-au putut descarca date pentru simbolul: {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in expected_cols if c in df.columns]].dropna(subset=["Close"])

    # Suprapunere pret live din Alpaca (elimina intarzierile Yahoo Finance in timpul sesiunii)
    snap = get_alpaca_live_snapshot(ticker)
    if snap and not df.empty:
        try:
            latest_trade = snap.get("latestTrade") or {}
            daily_bar = snap.get("dailyBar") or {}
            live_price = float(latest_trade.get("p") or daily_bar.get("c") or 0.0)
            if live_price > 0:
                last_idx = df.index[-1]
                df.loc[last_idx, "Close"] = live_price
                if daily_bar.get("h"):
                    df.loc[last_idx, "High"] = max(float(df.loc[last_idx, "High"]), float(daily_bar["h"]), live_price)
                else:
                    df.loc[last_idx, "High"] = max(float(df.loc[last_idx, "High"]), live_price)
                if daily_bar.get("l"):
                    df.loc[last_idx, "Low"] = min(float(df.loc[last_idx, "Low"]), float(daily_bar["l"]), live_price)
                else:
                    df.loc[last_idx, "Low"] = min(float(df.loc[last_idx, "Low"]), live_price)
                if daily_bar.get("v") and float(daily_bar["v"]) > float(df.loc[last_idx, "Volume"]):
                    df.loc[last_idx, "Volume"] = float(daily_bar["v"])
        except Exception:
            pass

    return df


def _get_cached_spy(period: str = "2y") -> Optional[pd.DataFrame]:
    """Returneaza DataFrame-ul SPY din cache sau il descarca daca a expirat."""
    now = time.time()
    if period in _SPY_CACHE:
        ts, cached_df = _SPY_CACHE[period]
        if now - ts < _SPY_CACHE_TTL_SECONDS and not cached_df.empty:
            return cached_df

    try:
        raw_spy = _download_raw_ohlcv(config.BENCHMARK_TICKER, period=period)
        spy_df = add_all_indicators(raw_spy, spy_df=None)
        _SPY_CACHE[period] = (now, spy_df)
        return spy_df
    except Exception:
        return None


def fetch_stock_data(ticker: str, period: str = "2y", spy_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Descarca datele istorice OHLCV pentru un ticker de pe Yahoo Finance.
    Adauga automat toti indicatorii tehnici calculati, inclusiv RS_vs_SPY.
    """
    raw_df = _download_raw_ohlcv(ticker, period=period)

    if ticker.upper() == config.BENCHMARK_TICKER.upper():
        df = add_all_indicators(raw_df, spy_df=None)
        _SPY_CACHE[period] = (time.time(), df)
        return df

    if spy_df is None:
        spy_df = _get_cached_spy(period=period)

    df = add_all_indicators(raw_df, spy_df=spy_df)
    return df


def fetch_market_regime(benchmark: str = "SPY", period: str = "2y") -> dict:
    """
    Determina starea multidimensionala a pietei (SPY + QQQ + VIX):
      - POWER_TREND      : SPY > SMA 50 > SMA 200 si VIX < 24 (Risc 100%, 1.5%)
      - CAUTIOUS_UPTREND : SPY > SMA 200, dar sub SMA 50 sau VIX 24-30 (Risc 65%, ~1.0%)
      - BEARISH          : SPY <= SMA 200 sau VIX > 30 (Risc 0%, Fara pozitii Long)
    """
    df = fetch_stock_data(benchmark, period=period)
    last_row = df.iloc[-1]

    current_price = float(last_row["Close"])
    sma_200 = float(last_row["SMA_200"])
    sma_50 = float(last_row.get("SMA_50", sma_200))

    pct_from_sma200 = ((current_price - sma_200) / sma_200) * 100 if sma_200 > 0 else 0.0
    pct_from_sma50 = ((current_price - sma_50) / sma_50) * 100 if sma_50 > 0 else 0.0

    # Verificare secundara QQQ (Nasdaq 100 - apetitul pentru risc pe Tech/Growth)
    qqq_bullish = True
    qqq_price = 0.0
    try:
        qqq_Ticker = getattr(config, "SECONDARY_BENCHMARK", "QQQ")
        qqq_df = _download_raw_ohlcv(qqq_Ticker, period="1y")
        qqq_sma50 = float(qqq_df["Close"].rolling(50).mean().iloc[-1])
        qqq_price = float(qqq_df["Close"].iloc[-1])
        qqq_bullish = qqq_price >= qqq_sma50 * 0.99
    except Exception:
        qqq_bullish = current_price >= sma_50

    # Verificare indice de volatilitate ^VIX
    vix_level = 18.0
    try:
        vix_ticker = getattr(config, "VIX_TICKER", "^VIX")
        vix_df = _download_raw_ohlcv(vix_ticker, period="3mo")
        if not vix_df.empty:
            vix_level = round(float(vix_df["Close"].iloc[-1]), 2)
    except Exception:
        vix_level = 18.0

    # Clasificare Regim de Piata
    spy_above_200 = current_price > sma_200
    spy_above_50 = current_price >= sma_50 * 0.995

    if not spy_above_200 or vix_level > 31.0:
        regime = "BEARISH"
        is_bullish = False
        risk_multiplier = 0.0
        max_positions_allowed = 0
    elif spy_above_50 and qqq_bullish and vix_level < 24.0:
        regime = "POWER_TREND"
        is_bullish = True
        risk_multiplier = 1.0
        max_positions_allowed = config.MAX_OPEN_POSITIONS
    else:
        regime = "CAUTIOUS_UPTREND"
        is_bullish = True
        risk_multiplier = 0.65  # Reducem automat riscul la ~1.0% in piata mixta/volatila
        max_positions_allowed = max(2, config.MAX_OPEN_POSITIONS - 1)

    return {
        "benchmark": benchmark,
        "price": round(current_price, 2),
        "sma_200": round(sma_200, 2),
        "sma_50": round(sma_50, 2),
        "is_bullish": is_bullish,
        "regime": regime,
        "qqq_bullish": qqq_bullish,
        "vix": vix_level,
        "risk_multiplier": risk_multiplier,
        "max_positions_allowed": max_positions_allowed,
        "pct_from_sma": round(pct_from_sma200, 2),
        "pct_from_sma50": round(pct_from_sma50, 2),
        "as_of": df.index[-1].strftime("%Y-%m-%d")
    }


def get_next_earnings_date(ticker: str) -> Optional[datetime]:
    """
    Returneaza data urmatorului raport de castiguri (earnings) pentru un ticker,
    folosind Yahoo Finance via yfinance.
    """
    try:
        stock = yf.Ticker(ticker)
        cal = stock.calendar

        if isinstance(cal, dict):
            earnings_date = cal.get("Earnings Date")
            if earnings_date is None:
                return None
            if isinstance(earnings_date, list):
                if not earnings_date:
                    return None
                earnings_date = earnings_date[0]
            if hasattr(earnings_date, "to_pydatetime"):
                return earnings_date.to_pydatetime().replace(tzinfo=None)
            if isinstance(earnings_date, datetime):
                return earnings_date.replace(tzinfo=None)
            if hasattr(earnings_date, "year"):
                return datetime(earnings_date.year, earnings_date.month, earnings_date.day)
            return None

        elif isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.index:
                val = cal.loc["Earnings Date"].iloc[0]
                if pd.notna(val):
                    ts = pd.Timestamp(val).to_pydatetime().replace(tzinfo=None)
                    return ts
            return None

    except Exception:
        return None


def check_earnings_risk(
    ticker: str,
    buffer_days: int = 7
) -> Tuple[bool, str]:
    """
    Verifica daca un ticker are un raport de castiguri programat
    in urmatoarele `buffer_days` zile calendaristice.
    """
    next_earnings = get_next_earnings_date(ticker)

    if next_earnings is None:
        return False, ""

    days_until = (next_earnings - datetime.now()).days

    if days_until < 0:
        return False, ""

    if days_until <= buffer_days:
        return True, (
            f"⚠️ EARNINGS RISK: {ticker} raporteaza pe "
            f"{next_earnings.strftime('%d %b %Y')} "
            f"(peste {days_until} zile). Intrarea este BLOCATA (buffer {buffer_days} zile)."
        )

    return False, f"Urmator earnings: {next_earnings.strftime('%d %b %Y')} (peste {days_until} zile) — OK"


def check_news_sentiment(ticker: str, limit: int = 8) -> Tuple[bool, float, str, str]:
    """
    Analizeaza stirile si catalizatorii din ultimele 48 de ore prin Alpaca News API (gratuit).
    Returneaza:
      - is_blocked      : True daca exista un catalizator negativ sever (investigatie SEC, faliment, dilutie, taiere guidance)
      - sentiment_score : Scor agregat (ex: -3.0 .. +3.0)
      - sentiment_label : Eticheta lizibila ('🟢 POZITIV', '⚪ NEUTRU', '🔴 NEGATIV SEVER')
      - latest_headline : Titlul celei mai relevante stiri recente
    """
    if not getattr(config, "USE_NEWS_FILTER", True):
        return False, 0.0, "⚪ NEUTRU", ""
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        return False, 0.0, "⚪ NEUTRU", ""

    now = time.time()
    cache_key = ticker.upper()
    if cache_key in _NEWS_CACHE:
        ts, cached_val = _NEWS_CACHE[cache_key]
        if now - ts < _NEWS_CACHE_TTL_SECONDS:
            return cached_val

    url = "https://data.alpaca.markets/v1beta1/news"
    headers = {
        "APCA-API-KEY-ID": config.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
    }
    start_dt = (datetime.utcnow() - timedelta(hours=72)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "symbols": cache_key,
        "limit": limit,
        "start": start_dt,
    }

    severe_negative_terms = [
        "sec investigation", "doj investigation", "subpoena", "accounting irregularities",
        "fraud", "bankruptcy", "chapter 11", "delisting", "fda clinical hold",
        "fda complete response letter", "cuts guidance", "lowers guidance", "slashes guidance",
        "withdraws guidance", "secondary offering", "public offering of common stock",
        "share dilution", "auditor resigns", "restatement", "default on debt",
    ]
    moderate_negative_terms = [
        "downgrades", "downgraded to sell", "downgraded to underperform", "misses estimates",
        "revenue miss", "earnings miss", "class action lawsuit", "antitrust lawsuit",
        "product recall", "weak demand", "price target cut", "price target lowered",
    ]
    positive_terms = [
        "beats estimates", "raises guidance", "boosts guidance", "lifts guidance",
        "upgraded to buy", "upgraded to outperform", "price target raised", "record revenue",
        "fda approval", "share buyback", "stock repurchase", "new ai contract",
        "strategic partnership", "strong demand", "blowout quarter",
    ]

    score = 0.0
    latest_headline = ""
    severe_trigger = ""

    try:
        res = requests.get(url, headers=headers, params=params, timeout=5)
        if res.status_code == 200:
            news_items = res.json().get("news", [])
            for idx, item in enumerate(news_items):
                headline = (item.get("headline") or "").strip()
                summary = (item.get("summary") or "").strip()
                if idx == 0 and headline:
                    latest_headline = headline

                text_blob = f"{headline} {summary}".lower()

                for term in severe_negative_terms:
                    if term in text_blob:
                        score -= 2.5
                        if not severe_trigger:
                            severe_trigger = headline or term

                for term in moderate_negative_terms:
                    if term in text_blob:
                        score -= 1.0

                for term in positive_terms:
                    if term in text_blob:
                        score += 1.0
    except Exception:
        pass

    score = round(max(-5.0, min(5.0, score)), 2)
    is_blocked = score <= -2.0

    if is_blocked:
        label = f"🔴 RISC ȘTIRI ({score:+.1f})"
        if severe_trigger:
            latest_headline = severe_trigger
    elif score >= 1.0:
        label = f"🟢 CATALIZATOR POZITIV ({score:+.1f})"
    elif score <= -1.0:
        label = f"🟡 ATENȚIE ȘTIRI ({score:+.1f})"
    else:
        label = "⚪ NEUTRU"

    result = (is_blocked, score, label, latest_headline[:110])
    _NEWS_CACHE[cache_key] = (now, result)
    return result


def screen_top_rs_leaders(
    universe: Optional[List[str]] = None,
    top_n: int = 16
) -> List[Dict]:
    """
    Screener Dinamic Institutional (Stage 2 + Relative Strength):
    Scaneaza universul extins de actiuni lichide si selecteaza liderii de piata care:
      1. Se afla in Stage 2 Uptrend (Close > SMA 50 > SMA 200)
      2. Au lichiditate institutionala (Volum mediu > 800k actiuni/zi)
      3. Au Forta Relativa superioara fata de S&P 500 (RS_vs_SPY > 0)
    Returneaza lista sortata descrescator dupa RS_vs_SPY.
    """
    if universe is None:
        universe = getattr(config, "UNIVERSE_POOL", config.WATCHLIST)

    spy_df = _get_cached_spy(period="2y")
    candidates = []

    for ticker in universe:
        try:
            df = fetch_stock_data(ticker, period="1y", spy_df=spy_df)
            if len(df) < 150:
                continue

            last = df.iloc[-1]
            close = float(last["Close"])
            sma_50 = float(last["SMA_50"])
            sma_200 = float(last.get("SMA_200", sma_50 * 0.9))
            ema_20 = float(last["EMA_20"])
            rsi = float(last["RSI_14"])
            vol_sma = float(last.get("Volume_SMA_20", 0))
            rs_score = float(last.get("RS_vs_SPY", 0.0))
            stock_roc = float(last.get("ROC_Weighted", rs_score))
            atr_contraction = float(last.get("ATR_Contraction", 1.0))
            atr_pct = (float(last.get("ATR_14", close * 0.02)) / close) * 100
            dollar_vol_m = (close * vol_sma) / 1_000_000.0

            # Criterii de calificare Stage 2 Institutional
            in_stage2 = (close > sma_50) and ((pd.isna(sma_200)) or (close > sma_200))
            liquid = vol_sma >= 500_000

            if in_stage2 and liquid:
                candidates.append({
                    "ticker": ticker,
                    "close": round(close, 2),
                    "ema_20": round(ema_20, 2),
                    "sma_50": round(sma_50, 2),
                    "rsi": round(rsi, 1),
                    "rs_vs_spy": round(rs_score, 2),
                    "stock_roc": round(stock_roc, 2),
                    "stage2": bool(in_stage2 and sma_50 >= sma_200 * 0.98),
                    "atr_contraction": round(atr_contraction, 2),
                    "dollar_vol_m": round(dollar_vol_m, 1),
                    "atr_pct": round(atr_pct, 2),
                    "dist_to_ema20_pct": round(((close - ema_20) / ema_20) * 100, 2)
                })
        except Exception:
            continue

    candidates.sort(key=lambda x: x["rs_vs_spy"], reverse=True)
    return candidates[:top_n]


DAILY_WATCHLIST_CACHE_FILE = config.BASE_DIR / "data" / "daily_watchlist.json"
_CORE_DEFAULT_WATCHLIST = list(config.WATCHLIST)


def get_smart_hybrid_watchlist(force_refresh: bool = False) -> Tuple[List[str], Dict]:
    """
    Construiește și returnează Watchlist-ul Dinamic 'Smart Hybrid' cu cache zilnic:
      1. Păstrează liderii din lista de bază (Core Watchlist) care sunt în trend sănătos
         (Close >= SMA 50 * 0.985), pentru a nu rata intrările pe Pullback la EMA 20.
      2. Adaugă automat Top 18 Super-Lideri de Forță Relativă (RS vs SPY) în Stage 2
         din întregul UNIVERSE_POOL (S&P 500 / Nasdaq 100), pentru Breakout-uri.
      3. Salvează rezultatul în data/daily_watchlist.json pentru ziua curentă.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")

    if not force_refresh and DAILY_WATCHLIST_CACHE_FILE.exists():
        try:
            data = json.loads(DAILY_WATCHLIST_CACHE_FILE.read_text(encoding="utf-8"))
            if data.get("date") == today_str and data.get("watchlist"):
                config.WATCHLIST = data["watchlist"]
                data["from_cache"] = True
                return data["watchlist"], data
        except Exception:
            pass

    print("⚡ Auto-Screener Smart Hybrid: Se actualizează lista de lideri pentru astăzi...")
    spy_df = _get_cached_spy(period="2y")
    universe = list(dict.fromkeys(_CORE_DEFAULT_WATCHLIST + getattr(config, "UNIVERSE_POOL", [])))

    core_kept = []
    rs_candidates = []

    for ticker in universe:
        try:
            df = fetch_stock_data(ticker, period="1y", spy_df=spy_df)
            if len(df) < 120:
                continue
            last = df.iloc[-1]
            close = float(last["Close"])
            sma_50 = float(last["SMA_50"])
            sma_200 = float(last.get("SMA_200", sma_50 * 0.9))
            vol_sma = float(last.get("Volume_SMA_20", 0))
            rs_score = float(last.get("RS_vs_SPY", 0.0))

            if ticker in _CORE_DEFAULT_WATCHLIST and close >= sma_50 * 0.985:
                core_kept.append(ticker)

            in_stage2 = close > sma_50 and (pd.isna(sma_200) or close > sma_200)
            if in_stage2 and vol_sma >= 500_000 and rs_score > 0:
                rs_candidates.append((ticker, rs_score))
        except Exception:
            continue

    rs_candidates.sort(key=lambda x: x[1], reverse=True)
    top_rs_tickers = [t for t, _ in rs_candidates[:18]]

    # Combinare fără duplicate și cu diversificare pe sectoare (max 7 acțiuni per sector în Watchlist):
    raw_candidates = list(dict.fromkeys(top_rs_tickers + core_kept + _CORE_DEFAULT_WATCHLIST))
    hybrid_list = []
    sector_counts: Dict[str, int] = {}
    for t in raw_candidates:
        sec = config.get_ticker_sector(t)
        if sector_counts.get(sec, 0) < 7:
            hybrid_list.append(t)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
        if len(hybrid_list) >= 35:
            break

    config.WATCHLIST = hybrid_list

    meta = {
        "date": today_str,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "count": len(hybrid_list),
        "top_rs_added": [t for t in top_rs_tickers if t not in _CORE_DEFAULT_WATCHLIST][:10],
        "top_5_rs": [f"{t} ({rs:+.1f}%)" for t, rs in rs_candidates[:5]],
        "sectors": sector_counts,
        "watchlist": hybrid_list,
        "from_cache": False
    }

    try:
        DAILY_WATCHLIST_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        DAILY_WATCHLIST_CACHE_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass

    return hybrid_list, meta
