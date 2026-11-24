"""
Modul pentru descărcarea curată a datelor istorice (exclusiv lumânări ÎNCHISE - Zero Look-Ahead Bias),
interogarea rezilientă a cotațiilor live pentru execuție și analiza Regimului de Piață:
  - Separare strictă între lumânările zilnice închise ('closed') și cotațiile intraday de execuție.
  - Retry cu Exponential Backoff + Timeout explicit pe apelurile HTTP către API-ul Alpaca.
  - Regim de Piață Multidimensional (SPY + QQQ + ^VIX) cu protecție rapidă la corecții.
  - Screener Dinamic Multi-Sector bazat pe Momentum Ajustat la Risc (RS / ATR_Pct + VCP).
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List, Dict
import requests
import pandas as pd
import yfinance as yf
import config
from indicators import add_all_indicators

try:
    from zoneinfo import ZoneInfo
    _ET_TZ = ZoneInfo("America/New_York")
except Exception:
    _ET_TZ = timezone(timedelta(hours=-5))

_SPY_CACHE: Dict[str, Tuple[float, pd.DataFrame]] = {}
_SPY_CACHE_TTL_SECONDS = 900
_NEWS_CACHE: Dict[str, Tuple[float, Tuple[bool, float, str, str]]] = {}
_NEWS_CACHE_TTL_SECONDS = 900


def _http_get_with_backoff(url: str, headers: dict = None, params: dict = None) -> Optional[requests.Response]:
    """
    Execută un apel HTTP GET cu timeout explicit și retry cu exponential backoff
    pentru erori tranzitorii de rețea, rate-limiting (429) sau erori de server (5xx).
    """
    timeout_sec = float(getattr(config, "API_TIMEOUT_SECONDS", 10.0))
    max_retries = int(getattr(config, "API_MAX_RETRIES", 3))
    base_delay = float(getattr(config, "API_BACKOFF_BASE_SECONDS", 1.0))

    for attempt in range(max_retries):
        try:
            res = requests.get(url, headers=headers, params=params, timeout=timeout_sec)
            if res.status_code == 200:
                return res
            if res.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            return res
        except requests.RequestException:
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            return None
    return None


def _is_us_session_unclosed_today() -> Tuple[bool, datetime.date]:
    """
    Returnează (True, today_et_date) dacă sesiunea regulată SUA de azi este încă în desfășurare
    (sau înainte de închiderea oficială de la 16:05 ET), astfel încât lumânarea zilnică de azi
    NU este încă închisă ('closed').
    """
    now_et = datetime.now(_ET_TZ)
    today_et = now_et.date()
    if now_et.weekday() >= 5:
        return False, today_et
    # Lumânarea zilnică 1D devine 'closed' abia după 16:02 ET
    before_close = (now_et.hour < 16) or (now_et.hour == 16 and now_et.minute < 2)
    return before_close, today_et


def get_alpaca_live_snapshot(ticker: str) -> Optional[dict]:
    """
    Preia snapshot-ul live din Alpaca Market Data v2 (DOAR pentru verificări de preț la execuție,
    niciodată pentru modificarea lumânărilor zilnice folosite în calculul semnalelor!).
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
    res = _http_get_with_backoff(url, headers=headers)
    if res is not None and res.status_code == 200:
        try:
            return res.json()
        except Exception:
            return None
    return None


def get_realtime_price(ticker: str) -> Optional[float]:
    """
    Returnează ultimul preț de piață în timp real (din Alpaca Snapshot sau fallback Yahoo)
    exclusiv pentru validarea gap-ului și calibrarea ordinelor Limit la execuție.
    """
    snap = get_alpaca_live_snapshot(ticker)
    if snap:
        try:
            latest_trade = snap.get("latestTrade") or {}
            latest_quote = snap.get("latestQuote") or {}
            daily_bar = snap.get("dailyBar") or {}
            ask_p = float(latest_quote.get("ap") or 0.0)
            trade_p = float(latest_trade.get("p") or daily_bar.get("c") or 0.0)
            if trade_p > 0:
                return trade_p
            if ask_p > 0:
                return ask_p
        except Exception:
            pass
    return None


def _download_raw_ohlcv(ticker: str, period: str = "2y", closed_only: bool = True) -> pd.DataFrame:
    """
    Descarcă datele istorice OHLCV de pe Yahoo Finance.
    Când `closed_only=True` (implicit și obligatoriu pentru semnale/backtest):
      - NU suprapune niciodată prețuri intraday peste lumânarea zilnică.
      - Elimină automat bara zilei curente dacă sesiunea SUA nu s-a închis încă (< 16:02 ET),
        garantând 100% Zero Look-Ahead Bias (exclusiv lumânări 'closed').
    """
    df = yf.download(
        ticker,
        period=period,
        progress=False,
        auto_adjust=True
    )

    if df.empty:
        raise ValueError(f"Nu s-au putut descărca date pentru simbolul: {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in expected_cols if c in df.columns]].dropna(subset=["Open", "High", "Low", "Close"])
    df = df.sort_index()

    if closed_only and getattr(config, "CLOSED_CANDLES_ONLY", True) and not df.empty:
        unclosed_today, today_et = _is_us_session_unclosed_today()
        if unclosed_today:
            last_bar_date = pd.Timestamp(df.index[-1]).date()
            if last_bar_date >= today_et and len(df) > 1:
                df = df.iloc[:-1]

    return df


def _get_cached_spy(period: str = "2y", closed_only: bool = True) -> Optional[pd.DataFrame]:
    """Returnează DataFrame-ul SPY din cache sau îl descarcă dacă a expirat."""
    cache_key = f"{period}_{closed_only}"
    now = time.time()
    if cache_key in _SPY_CACHE:
        ts, cached_df = _SPY_CACHE[cache_key]
        if now - ts < _SPY_CACHE_TTL_SECONDS and not cached_df.empty:
            return cached_df

    try:
        raw_spy = _download_raw_ohlcv(config.BENCHMARK_TICKER, period=period, closed_only=closed_only)
        spy_df = add_all_indicators(raw_spy, spy_df=None)
        _SPY_CACHE[cache_key] = (now, spy_df)
        return spy_df
    except Exception:
        return None


def fetch_stock_data(
    ticker: str,
    period: str = "2y",
    spy_df: Optional[pd.DataFrame] = None,
    closed_only: bool = True
) -> pd.DataFrame:
    """
    Descarcă datele istorice OHLCV (exclusiv lumânări închise când `closed_only=True`)
    și calculează toți indicatorii tehnici cauzali + RS_vs_SPY.
    """
    raw_df = _download_raw_ohlcv(ticker, period=period, closed_only=closed_only)

    if ticker.upper() == config.BENCHMARK_TICKER.upper():
        df = add_all_indicators(raw_df, spy_df=None)
        _SPY_CACHE[f"{period}_{closed_only}"] = (time.time(), df)
        return df

    if spy_df is None:
        spy_df = _get_cached_spy(period=period, closed_only=closed_only)

    df = add_all_indicators(raw_df, spy_df=spy_df)
    return df


def fetch_market_regime(benchmark: str = "SPY", period: str = "2y") -> dict:
    """
    Determină starea multidimensională a pieței pe lumânări zilnice închise (SPY + QQQ + ^VIX):
      - POWER_TREND      : SPY >= SMA 50 > SMA 200, QQQ >= SMA 50 și VIX < 21.5 (Risc 100%)
      - CAUTIOUS_UPTREND : SPY > SMA 200, cel puțin un indice peste SMA 50 și VIX < 25.0 (Risc 70%)
      - BEARISH          : SPY <= SMA 200 SAU VIX >= 25.0 SAU (SPY < SMA 50 și QQQ < SMA 50) -> Cash 100%
    """
    df = fetch_stock_data(benchmark, period=period, closed_only=True)
    last_row = df.iloc[-1]

    current_price = float(last_row["Close"])
    sma_200 = float(last_row["SMA_200"])
    sma_50 = float(last_row.get("SMA_50", sma_200))

    pct_from_sma200 = ((current_price - sma_200) / sma_200) * 100.0 if sma_200 > 0 else 0.0
    pct_from_sma50 = ((current_price - sma_50) / sma_50) * 100.0 if sma_50 > 0 else 0.0

    qqq_bullish = True
    try:
        qqq_ticker = getattr(config, "SECONDARY_BENCHMARK", "QQQ")
        qqq_df = _download_raw_ohlcv(qqq_ticker, period="1y", closed_only=True)
        qqq_sma50 = float(qqq_df["Close"].rolling(50).mean().iloc[-1])
        qqq_price = float(qqq_df["Close"].iloc[-1])
        qqq_bullish = qqq_price >= qqq_sma50 * 0.995
    except Exception:
        qqq_bullish = current_price >= sma_50

    vix_level = 18.0
    try:
        vix_ticker = getattr(config, "VIX_TICKER", "^VIX")
        vix_df = _download_raw_ohlcv(vix_ticker, period="3mo", closed_only=True)
        if not vix_df.empty:
            vix_level = round(float(vix_df["Close"].iloc[-1]), 2)
    except Exception:
        vix_level = 18.0

    spy_above_200 = current_price > sma_200
    spy_above_50 = current_price >= sma_50
    vix_bearish = float(getattr(config, "VIX_BEARISH_THRESHOLD", 25.0))
    vix_power = float(getattr(config, "VIX_POWER_TREND_MAX", 21.5))

    if (not spy_above_200) or (vix_level >= vix_bearish) or ((not spy_above_50) and (not qqq_bullish)):
        regime = "BEARISH"
        is_bullish = False
        risk_multiplier = 0.0
        max_positions_allowed = 0
    elif spy_above_50 and qqq_bullish and vix_level < vix_power:
        regime = "POWER_TREND"
        is_bullish = True
        risk_multiplier = 1.0
        max_positions_allowed = config.MAX_OPEN_POSITIONS
    else:
        regime = "CAUTIOUS_UPTREND"
        is_bullish = True
        risk_multiplier = float(getattr(config, "CAUTIOUS_RISK_MULTIPLIER", 0.70))
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
        "as_of": pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d")
    }


def get_next_earnings_date(ticker: str) -> Optional[datetime]:
    """Returnează data următorului raport financiar (Earnings) pentru un ticker."""
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
                    return pd.Timestamp(val).to_pydatetime().replace(tzinfo=None)
            return None

    except Exception:
        return None


def check_earnings_risk(ticker: str, buffer_days: int = 7) -> Tuple[bool, str]:
    """Verifică dacă un ticker raportează Earnings în următoarele `buffer_days` zile."""
    next_earnings = get_next_earnings_date(ticker)
    if next_earnings is None:
        return False, ""

    days_until = (next_earnings - datetime.now()).days
    if days_until < 0:
        return False, ""

    if days_until <= buffer_days:
        return True, (
            f"⚠️ EARNINGS RISK: {ticker} raportează pe "
            f"{next_earnings.strftime('%d %b %Y')} "
            f"(peste {days_until} zile). Intrarea este BLOCATĂ (buffer {buffer_days} zile)."
        )

    return False, f"Următor earnings: {next_earnings.strftime('%d %b %Y')} (peste {days_until} zile) — OK"


def check_news_sentiment(ticker: str, limit: int = 8) -> Tuple[bool, float, str, str]:
    """Analizează știrile din ultimele 72h prin Alpaca News API cu retry și timeout explicit."""
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
    start_dt = (datetime.now(timezone.utc) - timedelta(hours=72)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {"symbols": cache_key, "limit": limit, "start": start_dt}

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

    res = _http_get_with_backoff(url, headers=headers, params=params)
    if res is not None and res.status_code == 200:
        try:
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


def screen_top_rs_leaders(universe: Optional[List[str]] = None, top_n: int = 25) -> List[Dict]:
    """
    Screener Dinamic Multi-Sector (Stage 2 + Momentum Ajustat la Volatilitate):
    Selectează liderii în Stage 2 (Close > SMA 50 > SMA 200, ATR_Pct <= MAX_ATR_PCT)
    ordonați după calitatea trendului (RS_vs_SPY / ATR_Pct + VCP Contraction).
    """
    if universe is None:
        universe = getattr(config, "UNIVERSE_POOL", config.WATCHLIST)

    spy_df = _get_cached_spy(period="2y", closed_only=True)
    candidates = []
    max_atr = float(getattr(config, "MAX_ATR_PCT", 4.3))
    min_vol = float(getattr(config, "MIN_AVG_VOLUME", 700_000))

    for ticker in universe:
        sec = config.get_ticker_sector(ticker)
        if config.is_sector_blocked(sec):
            continue

        try:
            df = fetch_stock_data(ticker, period="2y", spy_df=spy_df, closed_only=True)
            if len(df) < 205:
                continue

            last = df.iloc[-1]
            close = float(last["Close"])
            sma_50 = float(last["SMA_50"])
            sma_200 = float(last["SMA_200"])
            ema_20 = float(last["EMA_20"])
            rsi = float(last["RSI_14"])
            vol_sma = float(last.get("Volume_SMA_20", 0.0))
            rs_score = float(last.get("RS_vs_SPY", 0.0))
            stock_roc = float(last.get("ROC_Weighted", rs_score))
            atr_contraction = float(last.get("ATR_Contraction", 1.0))
            atr_pct = float(last.get("ATR_Pct", 2.5))
            dollar_vol_m = (close * vol_sma) / 1_000_000.0

            in_stage2 = (close > sma_50 > sma_200) and (atr_pct <= max_atr)
            liquid = vol_sma >= min_vol

            if in_stage2 and liquid and rs_score >= 0.0:
                quality_score = (rs_score / max(1.5, atr_pct)) * 10.0 + (1.0 - min(1.2, atr_contraction)) * 12.0
                candidates.append({
                    "ticker": ticker,
                    "sector": sec,
                    "close": round(close, 2),
                    "ema_20": round(ema_20, 2),
                    "sma_50": round(sma_50, 2),
                    "rsi": round(rsi, 1),
                    "rs_vs_spy": round(rs_score, 2),
                    "attack_score": round(quality_score, 2),
                    "stock_roc": round(stock_roc, 2),
                    "stage2": True,
                    "atr_contraction": round(atr_contraction, 2),
                    "dollar_vol_m": round(dollar_vol_m, 1),
                    "atr_pct": round(atr_pct, 2),
                    "dist_to_ema20_pct": round(((close - ema_20) / ema_20) * 100.0, 2)
                })
        except Exception:
            continue

    candidates.sort(key=lambda x: x["attack_score"], reverse=True)
    return candidates[:top_n]


DAILY_WATCHLIST_CACHE_FILE = config.BASE_DIR / "data" / "daily_watchlist.json"
_CORE_DEFAULT_WATCHLIST = list(config.WATCHLIST)
_SECTOR_STRENGTH_CACHE: Dict[str, float] = {}


def get_live_sector_bonus(sector: str) -> float:
    """Returnează bonusul moderat al sectorului pe baza Forței Relative live."""
    base_bonus = config.get_sector_priority_bonus(sector)
    if not _SECTOR_STRENGTH_CACHE and DAILY_WATCHLIST_CACHE_FILE.exists():
        try:
            data = json.loads(DAILY_WATCHLIST_CACHE_FILE.read_text(encoding="utf-8"))
            for sec_name, avg_rs in (data.get("sector_avg_rs") or {}).items():
                _SECTOR_STRENGTH_CACHE[sec_name] = float(avg_rs)
        except Exception:
            pass

    live_rs = _SECTOR_STRENGTH_CACHE.get(sector.upper(), 0.0)
    dynamic_boost = max(-2.0, min(5.0, live_rs * 0.25))
    return round(base_bonus + dynamic_boost, 2)


def get_smart_hybrid_watchlist(force_refresh: bool = False) -> Tuple[List[str], Dict]:
    """
    Construiește Watchlist-ul Zilnic Multi-Sector pe lumânări zilnice închise:
      - Selectează liderii autentici Stage 2 cu volatilitate controlată (ATR_Pct <= MAX_ATR_PCT)
        din toate sectoarele S&P 500 unde curge capitalul (fără concentrare oarbă pe un singur sector).
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    max_wl_count = int(getattr(config, "DAILY_WATCHLIST_MAX_COUNT", 35))

    if not force_refresh and DAILY_WATCHLIST_CACHE_FILE.exists():
        try:
            data = json.loads(DAILY_WATCHLIST_CACHE_FILE.read_text(encoding="utf-8"))
            cached_wl = data.get("watchlist", [])
            if data.get("date") == today_str and cached_wl and data.get("version") == "v2_multisector":
                config.WATCHLIST = cached_wl
                for sec_name, avg_rs in (data.get("sector_avg_rs") or {}).items():
                    _SECTOR_STRENGTH_CACHE[sec_name] = float(avg_rs)
                data["from_cache"] = True
                return cached_wl, data
        except Exception:
            pass

    print("⚡ Auto-Screener Multi-Sector Stage 2: Se selectează liderii cu cel mai bun raport Momentum/Volatilitate...")
    spy_df = _get_cached_spy(period="2y", closed_only=True)
    universe = list(dict.fromkeys(_CORE_DEFAULT_WATCHLIST + getattr(config, "UNIVERSE_POOL", [])))
    max_atr = float(getattr(config, "MAX_ATR_PCT", 4.3))
    min_vol = float(getattr(config, "MIN_AVG_VOLUME", 700_000))

    sector_rs_buckets: Dict[str, List[float]] = {}
    scored_stocks: List[Tuple[str, float, float, str, float]] = []

    for ticker in universe:
        sec = config.get_ticker_sector(ticker)
        if config.is_sector_blocked(sec):
            continue

        try:
            df = fetch_stock_data(ticker, period="2y", spy_df=spy_df, closed_only=True)
            if len(df) < 205:
                continue
            last = df.iloc[-1]
            close = float(last["Close"])
            sma_50 = float(last["SMA_50"])
            sma_200 = float(last["SMA_200"])
            vol_sma = float(last.get("Volume_SMA_20", 0.0))
            rs_score = float(last.get("RS_vs_SPY", 0.0))
            atr_pct = float(last.get("ATR_Pct", 2.5))
            atr_c = float(last.get("ATR_Contraction", 1.0))

            in_stage2 = (close > sma_50 > sma_200) and (vol_sma >= min_vol) and (atr_pct <= max_atr)
            if in_stage2:
                sector_rs_buckets.setdefault(sec, []).append(rs_score)
                if rs_score >= 1.0:
                    q_score = (rs_score / max(1.5, atr_pct)) * 10.0 + (1.0 - min(1.2, atr_c)) * 12.0
                    scored_stocks.append((ticker, rs_score, q_score, sec, atr_pct))
        except Exception:
            continue

    sector_avg_rs: Dict[str, float] = {}
    for sec, rs_vals in sector_rs_buckets.items():
        top_vals = sorted(rs_vals, reverse=True)[:5]
        sector_avg_rs[sec] = round(sum(top_vals) / len(top_vals), 2) if top_vals else 0.0
        _SECTOR_STRENGTH_CACHE[sec] = sector_avg_rs[sec]

    sector_power_ranking = sorted(sector_avg_rs.keys(), key=lambda s: sector_avg_rs[s], reverse=True)
    hot_sectors_summary = [f"{s} (RS: {sector_avg_rs[s]:+.1f}%)" for s in sector_power_ranking[:3]]

    ranked_candidates = []
    for ticker, rs_score, q_score, sec, _atr_pct in scored_stocks:
        sec_boost = get_live_sector_bonus(sec) * 0.5
        ranked_candidates.append((ticker, rs_score, q_score + sec_boost, sec))

    ranked_candidates.sort(key=lambda x: x[2], reverse=True)
    top_rs_tickers = [t for t, _, _, _ in ranked_candidates[:22]]

    quotas = getattr(config, "SECTOR_WATCHLIST_QUOTA", {})
    raw_ordered = list(dict.fromkeys([t for t, _, _, _ in ranked_candidates] + _CORE_DEFAULT_WATCHLIST))

    hybrid_list = []
    sector_counts: Dict[str, int] = {}
    for t in raw_ordered:
        sec = config.get_ticker_sector(t)
        if config.is_sector_blocked(sec):
            continue
        max_quota = quotas.get(sec, 5)
        if sector_counts.get(sec, 0) < max_quota:
            hybrid_list.append(t)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
        if len(hybrid_list) >= max_wl_count:
            break

    config.WATCHLIST = hybrid_list
    top_5_display = [f"{t} ({rs:+.1f}%)" for t, rs, _, _ in ranked_candidates[:5]]

    meta = {
        "version": "v2_multisector",
        "date": today_str,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "count": len(hybrid_list),
        "hot_sectors": hot_sectors_summary,
        "sector_avg_rs": sector_avg_rs,
        "top_rs_added": [t for t in top_rs_tickers if t not in _CORE_DEFAULT_WATCHLIST][:10],
        "top_5_rs": top_5_display,
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
