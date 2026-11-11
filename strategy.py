"""
Logica strategiei Swing Trading Institutional si a Managementului de Risc.
Sursa unica de adevar (Single Source of Truth) pentru Scanarea Live si Backtester:
  1. Filtru Multi-Timeframe 1W (Stan Weinstein Stage 2: Weekly EMA 10 & SMA 30).
  2. Filtru de Forta Relativa (RS_vs_SPY): Tranzactioneaza liderii care bat S&P 500.
  3. SCENARIUL 1 — PULLBACK EMA 20: Retragere controlata pe suport cu confirmare
     Anti-Falling Knife (lumanare verde / Hammer) si panta SMA 50 pozitiva.
  4. SCENARIUL 2 — VOLUME BREAKOUT: Spargere maxim 20 zile cu volum exploziv,
     inchidere in top 40% din range, RS pozitiv si fara supra-extindere (Climax Top).
  5. Filtru ML XGBoost + Scor Compozit de Prioritizare a Semnalelor.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import config
from risk_learner import is_in_cooldown


def check_weekly_confluence(df: pd.DataFrame) -> Tuple[bool, str, dict]:
    """
    Verifica daca actiunea se afla intr-un trend saptamanal favorabil (Multi-Timeframe 1W):
    1. Close >= Weekly_SMA_30 * 0.99 (peste trendul primar — Weinstein Stage 2).
    2. Close >= Weekly_EMA_10 * 0.98 (mentine suportul intermediar saptamanal).
    3. Weekly_EMA_10 >= Weekly_SMA_30 * 0.97 (mediile saptamanale sunt aliniate ascendent).
    4. Weekly_RSI_14 intre WEEKLY_RSI_MIN si WEEKLY_RSI_MAX.
    """
    if not getattr(config, "USE_WEEKLY_FILTER", True):
        return True, "Filtru saptamanal dezactivat din config.", {}

    if len(df) < 30:
        return True, "Date insuficiente pentru calculul saptamanal.", {}

    last = df.iloc[-1]
    if isinstance(last, pd.DataFrame):
        last = last.iloc[0]

    close = float(last["Close"])
    weekly_ema = float(last.get("Weekly_EMA_10", np.nan))
    weekly_sma = float(last.get("Weekly_SMA_30", np.nan))
    weekly_rsi = float(last.get("Weekly_RSI_14", np.nan))

    if np.isnan(weekly_ema) or np.isnan(weekly_sma) or weekly_ema <= 0 or weekly_sma <= 0:
        return True, "Indicatori saptamanali indisponibili.", {}

    metrics = {
        "weekly_close": close,
        "weekly_ema10": weekly_ema,
        "weekly_sma30": weekly_sma,
        "weekly_rsi": weekly_rsi,
    }

    if close < weekly_sma * 0.99:
        pct_below = ((weekly_sma - close) / weekly_sma) * 100
        return False, f"Sub trendul primar saptamanal (Close ${close:.2f} < 1W SMA 30 ${weekly_sma:.2f}, -{pct_below:.1f}%)", metrics

    if close < weekly_ema * 0.98:
        pct_below = ((weekly_ema - close) / weekly_ema) * 100
        return False, f"Sub EMA 10 saptamanal (Close ${close:.2f} < 1W EMA 10 ${weekly_ema:.2f}, -{pct_below:.1f}%)", metrics

    if weekly_ema < weekly_sma * 0.97:
        return False, f"Medii saptamanale inversate (1W EMA 10 ${weekly_ema:.2f} < 1W SMA 30 ${weekly_sma:.2f} — Stage 4 Downtrend)", metrics

    rsi_max = getattr(config, "WEEKLY_RSI_MAX", 72.0)
    rsi_min = getattr(config, "WEEKLY_RSI_MIN", 40.0)

    if not np.isnan(weekly_rsi):
        if weekly_rsi > rsi_max:
            return False, f"RSI saptamanal supracumparat (1W RSI: {weekly_rsi:.1f} > {rsi_max}) — Risc de epuizare", metrics
        if weekly_rsi < rsi_min:
            return False, f"RSI saptamanal in declin structural (1W RSI: {weekly_rsi:.1f} < {rsi_min})", metrics

    return True, f"1W Trend Bullish: Close (${close:.2f}) > EMA 10 (${weekly_ema:.2f}) & SMA 30 (${weekly_sma:.2f})", metrics


@dataclass
class TradeSignal:
    ticker: str
    action: str            # 'BUY' sau 'NO_TRADE'
    setup_type: str        # 'PULLBACK_EMA20', 'VOLUME_BREAKOUT' sau 'NONE'
    entry_price: float
    stop_loss: float
    tp1: float             # Take Profit 1 (50% scale-out + SL la Breakeven)
    tp2: float             # Take Profit 2 (50% Runner)
    risk_per_share: float
    risk_reward_ratio: float
    shares: float          # Numarul optim de actiuni
    position_cost: float   # Valoarea totala a pozitiei
    reason: str
    ml_confidence: float = 0.0
    rs_score: float = 0.0          # Forta Relativa vs SPY (%)
    composite_score: float = 0.0   # Scor global de prioritizare a semnalului
    sector: str = "OTHER"          # Sectorul industrial (ex: SEMICONDUCTORS)
    news_sentiment: str = "⚪ NEUTRU"
    news_headline: str = ""
    news_score: float = 0.0


def calculate_position_size(
    entry_price: float,
    stop_loss: float,
    total_capital: float = config.DEFAULT_SIMULATED_CAPITAL,
    risk_percent: float = config.RISK_PER_TRADE_PERCENT,
    max_positions: int = config.MAX_OPEN_POSITIONS,
    risk_multiplier: float = 1.0
) -> dict:
    """
    Calculeaza marimea pozitiei cu suport complet pentru ACTIUNI FRACTIONARE (ex: 0.28x LLY, 1.75x NVDA),
    conform formulei stricte de 1-2% risc din capital si plafonului de alocare per pozitie (25% = $250 la $1,000):
      Numar Actiuni = (Capital Total * Risc% * Multiplier) / (Pret Intrare - Pret Stop Loss)
    Astfel, orice companie (chiar si de $600-$900/actiune) poate fi cumparata proportional pe un cont de $1,000.
    """
    risk_per_share = entry_price - stop_loss
    if risk_per_share <= 0 or entry_price <= 0:
        return {"shares": 0, "position_cost": 0, "cash_at_risk": 0, "risk_per_share": 0}

    effective_risk_pct = risk_percent * max(0.0, min(1.25, risk_multiplier))
    max_allowed_dollar_risk = total_capital * effective_risk_pct

    shares = max_allowed_dollar_risk / risk_per_share

    # Plafonare: O singura pozitie nu depaseste alocarea maxima pe portofoliu (ex: 25% = $250 pe cont de $1,000)
    max_capital_per_position = total_capital / max(1, max_positions)
    if (shares * entry_price) > max_capital_per_position:
        shares = max_capital_per_position / entry_price

    if getattr(config, "ALLOW_FRACTIONAL_SHARES", True):
        shares = round(shares, 4)
    else:
        shares = float(int(shares))

    position_cost = round(shares * entry_price, 2)
    actual_dollar_risk = round(shares * risk_per_share, 2)

    return {
        "shares": shares,
        "position_cost": position_cost,
        "cash_at_risk": actual_dollar_risk,
        "risk_per_share": round(risk_per_share, 2)
    }


def check_pullback_setup(
    ticker: str,
    df: pd.DataFrame,
    total_capital: float,
    rsi_min: Optional[float] = None,
    rsi_max: Optional[float] = None,
    min_rr: Optional[float] = None,
    ema_period: int = 20,
    vol_mult: Optional[float] = None,
    risk_multiplier: float = 1.0
) -> Optional[TradeSignal]:
    """
    SCENARIUL 1: PULLBACK LA EMA 20 (CU FILTRU DE RICOȘEU, RS & VCP)
    1. Trend Bullish: Close > SMA 50 si SMA 50 nu se prabuseste.
    2. Suport EMA 20: Pretul testeaza zona EMA 20 (-1.5% .. +2.5%).
    3. RSI in zona sanatoasa de resetare (38 - 56).
    4. Forta Relativa (RS_vs_SPY) peste pragul minim acceptat.
    5. Confirmare Anti-Falling Knife: Lumanare Verde SAU Hammer (fitil inferior >= 35%).
    6. Fara distributie institutionala (fara lumanare rosie pe volum ridicat).
    """
    if len(df) < 50:
        return None

    rsi_min = rsi_min if rsi_min is not None else config.RSI_PULLBACK_MIN
    rsi_max = rsi_max if rsi_max is not None else config.RSI_PULLBACK_MAX
    min_rr = min_rr if min_rr is not None else config.MIN_RISK_REWARD_RATIO
    vol_mult = vol_mult if vol_mult is not None else getattr(config, "VOLUME_SURGE_MULTIPLIER", 1.25)
    tp2_rr = getattr(config, "TP2_RISK_REWARD_RATIO", max(3.0, min_rr + 1.0))

    last = df.iloc[-1]
    open_p = float(last["Open"])
    high_p = float(last["High"])
    low_p = float(last["Low"])
    close_p = float(last["Close"])

    sma_50 = float(last["SMA_50"])
    ema_col = f"EMA_{ema_period}"
    ema_val = float(last[ema_col]) if ema_col in last else float(last["EMA_20"])
    rsi = float(last["RSI_14"])
    atr = float(last["ATR_14"]) if "ATR_14" in df.columns and not pd.isna(last["ATR_14"]) else (close_p * 0.02)
    volume = float(last["Volume"])
    vol_sma_20 = float(last.get("Volume_SMA_20", 0))
    rs_vs_spy = float(last.get("RS_vs_SPY", 0.0))
    sma50_slope = float(last.get("SMA_50_Slope", 0.0))

    # 1. Filtru Trend & Panta SMA 50
    if close_p < sma_50 or sma50_slope < -0.50:
        return None

    # 2. Filtru Forta Relativa vs SPY
    if getattr(config, "USE_RELATIVE_STRENGTH", True):
        min_rs = getattr(config, "MIN_RS_VS_SPY_PULLBACK", -5.0)
        if not np.isnan(rs_vs_spy) and rs_vs_spy < min_rs:
            return None

    # 3. Testare suport EMA (pretul trebuie sa inchida pe sau peste suportul EMA 20)
    distance_to_ema = abs(close_p - ema_val) / ema_val
    touched_ema = (low_p <= ema_val * 1.01) and (close_p >= ema_val * 0.99)
    valid_ema_zone = (distance_to_ema <= 0.025 and close_p >= ema_val * 0.99) or touched_ema
    if not valid_ema_zone:
        return None

    # 4. Filtru RSI
    if not (rsi_min <= rsi <= rsi_max):
        return None

    # 5. Filtru Anti-Falling Knife (Confirmare stricta cu lumanare VERDE de reversare)
    if close_p <= open_p:
        return None

    # 6. Filtru Volum: Refuzam daca exista distributie anormala
    if vol_sma_20 > 0 and volume > vol_mult * vol_sma_20 and close_p < open_p:
        return None

    # 7. Stop-Loss Structural + ATR (calibrat intre 2.5% si 5.5% pentru R:R si Win-Rate optim)
    recent_low = float(df["Low"].iloc[-4:].min())
    stop_loss = round(min(recent_low, low_p - (0.35 * atr)), 2)

    stop_dist_pct = (close_p - stop_loss) / close_p
    if stop_dist_pct < 0.025:
        stop_loss = round(close_p * 0.972, 2)
    elif stop_dist_pct > 0.055:
        stop_loss = round(close_p * 0.945, 2)

    risk_per_share = round(close_p - stop_loss, 2)
    if risk_per_share <= 0:
        return None

    tp1 = round(close_p + (min_rr * risk_per_share), 2)
    tp2 = round(close_p + (tp2_rr * risk_per_share), 2)

    sizing = calculate_position_size(
        close_p, stop_loss, total_capital,
        risk_percent=config.RISK_PER_TRADE_PERCENT,
        risk_multiplier=risk_multiplier
    )
    if sizing["shares"] <= 0:
        return None

    rs_tag = f" | RS vs SPY: {rs_vs_spy:+.1f}%" if not np.isnan(rs_vs_spy) else ""
    return TradeSignal(
        ticker=ticker,
        action="BUY",
        setup_type="PULLBACK_EMA20",
        entry_price=round(close_p, 2),
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        risk_per_share=risk_per_share,
        risk_reward_ratio=min_rr,
        shares=sizing["shares"],
        position_cost=sizing["position_cost"],
        reason=f"Setup PULLBACK CONFIRMAT: Ricoșeu validat la EMA {ema_period}{rs_tag}.",
        rs_score=round(rs_vs_spy, 2) if not np.isnan(rs_vs_spy) else 0.0
    )


def check_breakout_setup(
    ticker: str,
    df: pd.DataFrame,
    total_capital: float,
    min_rr: Optional[float] = None,
    vol_mult: Optional[float] = None,
    risk_multiplier: float = 1.0
) -> Optional[TradeSignal]:
    """
    SCENARIUL 2: BREAKOUT DE VOLUM CONFIRMAT (PE LIDERI DE PIATA)
    - Pretul peste SMA 50
    - Forta Relativa pozitiva fata de SPY (RS_vs_SPY >= 0)
    - Inchidere peste maximul ultimelor 20 de zile
    - Volum exploziv (>= VOLUME_SURGE_MULTIPLIER * Volume_SMA_20)
    - Lumanare VERDE cu inchidere in top 40% din amplitudinea zilei
    - Fara supra-extindere exagerata fata de EMA 20 (evita Climax Tops)
    """
    if len(df) < 30:
        return None

    min_rr = min_rr if min_rr is not None else config.MIN_RISK_REWARD_RATIO
    vol_mult = vol_mult if vol_mult is not None else getattr(config, "VOLUME_SURGE_MULTIPLIER", 1.25)
    tp2_rr = getattr(config, "TP2_RISK_REWARD_RATIO", max(3.0, min_rr + 1.0))

    last = df.iloc[-1]
    open_p = float(last["Open"])
    high_p = float(last["High"])
    low_p = float(last["Low"])
    close_p = float(last["Close"])

    sma_50 = float(last["SMA_50"])
    ema_20 = float(last["EMA_20"])
    rsi = float(last["RSI_14"])
    volume = float(last["Volume"])
    vol_sma_20 = float(last.get("Volume_SMA_20", 0))
    rs_vs_spy = float(last.get("RS_vs_SPY", 0.0))

    if close_p < sma_50 or close_p < open_p:
        return None

    # Filtru Forta Relativa: Breakout-urile sunt permise doar pe Lideri (RS >= 0)
    if getattr(config, "USE_RELATIVE_STRENGTH", True):
        min_rs_bo = getattr(config, "MIN_RS_VS_SPY_BREAKOUT", 0.0)
        if not np.isnan(rs_vs_spy) and rs_vs_spy < min_rs_bo:
            return None

    # Filtru Anti-Exhaustion: Nu cumparam spargeri deja extinse la >12% peste EMA 20
    max_ext = getattr(config, "MAX_BREAKOUT_EXTENSION_EMA20", 0.12)
    if ema_20 > 0 and ((close_p - ema_20) / ema_20) > max_ext:
        return None

    # Inchiderea trebuie sa fie in jumatatea/treimea superioara a lumanarii (forta reala)
    day_range = high_p - low_p
    if day_range > 0 and ((close_p - low_p) / day_range) < 0.60:
        return None

    prior_20_high = float(df["High"].iloc[-21:-1].max())
    is_breakout = close_p > prior_20_high
    has_volume_surge = (vol_sma_20 > 0) and (volume >= vol_mult * vol_sma_20)
    rsi_valid = 52.0 <= rsi <= 76.0

    if not (is_breakout and has_volume_surge and rsi_valid):
        return None

    recent_support = float(df["Low"].iloc[-3:].min())
    stop_loss = round(max(prior_20_high * 0.98, recent_support), 2)
    stop_dist_pct = (close_p - stop_loss) / close_p
    if stop_loss >= close_p or stop_dist_pct < 0.025:
        stop_loss = round(close_p * 0.972, 2)
    elif stop_dist_pct > 0.055:
        stop_loss = round(close_p * 0.945, 2)

    risk_per_share = round(close_p - stop_loss, 2)
    if risk_per_share <= 0:
        return None

    tp1 = round(close_p + (min_rr * risk_per_share), 2)
    tp2 = round(close_p + (tp2_rr * risk_per_share), 2)

    sizing = calculate_position_size(
        close_p, stop_loss, total_capital,
        risk_percent=config.RISK_PER_TRADE_PERCENT,
        risk_multiplier=risk_multiplier
    )
    if sizing["shares"] <= 0:
        return None

    vol_pct = int((volume / vol_sma_20) * 100) if vol_sma_20 > 0 else 0
    rs_tag = f" | RS: {rs_vs_spy:+.1f}%" if not np.isnan(rs_vs_spy) else ""

    return TradeSignal(
        ticker=ticker,
        action="BUY",
        setup_type="VOLUME_BREAKOUT",
        entry_price=round(close_p, 2),
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        risk_per_share=risk_per_share,
        risk_reward_ratio=min_rr,
        shares=sizing["shares"],
        position_cost=sizing["position_cost"],
        reason=f"Setup BREAKOUT CONFIRMAT: Spargere maxim 20z (${prior_20_high:.2f}) cu volum ({vol_pct}%){rs_tag}.",
        rs_score=round(rs_vs_spy, 2) if not np.isnan(rs_vs_spy) else 0.0
    )


def _compute_composite_score(signal: TradeSignal) -> float:
    """
    Calculeaza un scor compozit de atac (0 - 100) pentru prioritizarea semnalelor cand
    apar mai multe oportunitati in aceeasi zi decat numarul de locuri in portofoliu:
      - 42% pondere din increderea modelului ML (daca este activ)
      - 38% pondere din Forta Relativa fata de SPY (RS_vs_SPY normalizat)
      - Pana la +18 puncte Bonus de Rotatie Sectoriala & Alpha (prioritate maxima pe AI, Semiconductori, Cloud & Cyber)
      - Bonus/Penalizare Catalizator Știri (-5 la +5 puncte)
    """
    ml_part = (signal.ml_confidence if signal.ml_confidence > 0 else 0.55) * 42.0
    rs_clamped = max(-10.0, min(40.0, signal.rs_score))
    rs_part = ((rs_clamped + 10.0) / 50.0) * 38.0

    sec = getattr(signal, "sector", None) or config.get_ticker_sector(signal.ticker)
    try:
        from data_loader import get_live_sector_bonus
        sec_bonus = get_live_sector_bonus(sec)
    except Exception:
        sec_bonus = config.get_sector_priority_bonus(sec) if hasattr(config, "get_sector_priority_bonus") else 0.0

    raw_news_score = getattr(signal, "news_score", 0.0)
    news_val = float(raw_news_score) if isinstance(raw_news_score, (int, float)) else 0.0
    news_bonus = max(-5.0, min(5.0, news_val * 2.5))
    return round(max(0.0, min(100.0, ml_part + rs_part + sec_bonus + news_bonus)), 2)


def _apply_ml_filter(ticker: str, df: pd.DataFrame, signal: "TradeSignal") -> "TradeSignal":
    """
    Filtru secundar ML: evalueaza calitatea semnalului cu modelul XGBoost.
    """
    ml_enabled = getattr(config, "ML_ENABLED", True)
    if not ml_enabled:
        signal.composite_score = _compute_composite_score(signal)
        return signal

    try:
        from ml_predictor import get_predictor
        predictor = get_predictor()
        if not predictor.is_ready:
            signal.composite_score = _compute_composite_score(signal)
            return signal

        confidence, is_confident = predictor.predict(df)
        threshold_pct = int(getattr(config, "ML_MIN_CONFIDENCE", 0.52) * 100)

        if not is_confident:
            return TradeSignal(
                ticker=ticker,
                action="NO_TRADE",
                setup_type="NONE",
                entry_price=signal.entry_price,
                stop_loss=0.0,
                tp1=0.0,
                tp2=0.0,
                risk_per_share=0.0,
                risk_reward_ratio=0.0,
                shares=0.0,
                position_cost=0.0,
                reason=(
                    f"🤖 ML FILTER: Setup {signal.setup_type} detectat, dar încrederea "
                    f"modelului este insuficientă ({confidence*100:.1f}% < {threshold_pct}% prag). "
                    f"Semnal de calitate slabă — ignorat."
                ),
                ml_confidence=confidence,
                rs_score=signal.rs_score
            )

        signal.ml_confidence = confidence
        signal.composite_score = _compute_composite_score(signal)
        signal.reason = (
            f"{signal.reason} "
            f"[🧠 ML: {confidence*100:.1f}% WIN | Scor: {signal.composite_score:.0f}/100]"
        )
        return signal

    except Exception:
        signal.composite_score = _compute_composite_score(signal)
        return signal


def analyze_ticker(
    ticker: str,
    df: pd.DataFrame,
    is_market_bullish: bool,
    total_capital: float = config.DEFAULT_SIMULATED_CAPITAL,
    risk_multiplier: float = 1.0,
    market_regime: str = "POWER_TREND"
) -> TradeSignal:
    """
    Evalueaza un ticker testand toate scenariile cu filtre de confirmare multi-timeframe,
    Sector Rotation (focus pe AI & Tech), Forta Relativa (RS vs SPY), protectie Earnings,
    Cooldown si Machine Learning.
    """
    last_rs = float(df["RS_vs_SPY"].iloc[-1]) if len(df) > 0 and "RS_vs_SPY" in df.columns and not pd.isna(df["RS_vs_SPY"].iloc[-1]) else 0.0
    sector = config.get_ticker_sector(ticker)

    # 0. Filtru Sector Rotation: Blocam sectoarele defensive/lente (Healthcare, Staples, Oil)
    if getattr(config, "USE_SECTOR_ROTATION", True) and hasattr(config, "is_sector_blocked") and config.is_sector_blocked(sector):
        return TradeSignal(
            ticker=ticker,
            action="NO_TRADE",
            setup_type="NONE",
            entry_price=0.0,
            stop_loss=0.0,
            tp1=0.0,
            tp2=0.0,
            risk_per_share=0.0,
            risk_reward_ratio=0.0,
            shares=0.0,
            position_cost=0.0,
            reason=f"🚫 SECTOR DEFENSIV EXCLUS ({sector}): Capitalul este rezervat exclusiv pentru sectoarele de Momentum (AI, Semiconductori, Cloud & Tech).",
            rs_score=last_rs,
            sector=sector
        )

    if not is_market_bullish:
        return TradeSignal(
            ticker=ticker,
            action="NO_TRADE",
            setup_type="NONE",
            entry_price=0.0,
            stop_loss=0.0,
            tp1=0.0,
            tp2=0.0,
            risk_per_share=0.0,
            risk_reward_ratio=0.0,
            shares=0.0,
            position_cost=0.0,
            reason="Piața generală (SPY) este în Regim Bearish (sub SMA 200 sau VIX extrem). Nu deschidem poziții Long.",
            rs_score=last_rs,
            sector=sector
        )

    # Verificare Cooldown
    in_cooldown, cooldown_msg = is_in_cooldown(ticker)
    if in_cooldown:
        return TradeSignal(
            ticker=ticker,
            action="NO_TRADE",
            setup_type="NONE",
            entry_price=0.0,
            stop_loss=0.0,
            tp1=0.0,
            tp2=0.0,
            risk_per_share=0.0,
            risk_reward_ratio=0.0,
            shares=0.0,
            position_cost=0.0,
            reason=f"🔒 BLOCAT ÎN CARANTINĂ: {cooldown_msg}",
            rs_score=last_rs,
            sector=sector
        )

    # Verificare Earnings Risk (Anti-Binary Risk)
    try:
        from data_loader import check_earnings_risk
        earnings_risky, earnings_msg = check_earnings_risk(
            ticker,
            buffer_days=getattr(config, "EARNINGS_BUFFER_DAYS", 7)
        )
        if earnings_risky:
            return TradeSignal(
                ticker=ticker,
                action="NO_TRADE",
                setup_type="NONE",
                entry_price=0.0,
                stop_loss=0.0,
                tp1=0.0,
                tp2=0.0,
                risk_per_share=0.0,
                risk_reward_ratio=0.0,
                shares=0.0,
                position_cost=0.0,
                reason=earnings_msg,
                rs_score=last_rs,
                sector=sector
            )
    except Exception:
        pass

    if len(df) < 60:
        return TradeSignal(
            ticker=ticker,
            action="NO_TRADE",
            setup_type="NONE",
            entry_price=0.0,
            stop_loss=0.0,
            tp1=0.0,
            tp2=0.0,
            risk_per_share=0.0,
            risk_reward_ratio=0.0,
            shares=0.0,
            position_cost=0.0,
            reason="Date istorice insuficiente pentru calcul.",
            rs_score=last_rs,
            sector=sector
        )

    # Verificare Confluență Multi-Timeframe (1W - Grafic Săptămânal)
    weekly_ok, weekly_reason, _ = check_weekly_confluence(df)
    if not weekly_ok:
        last_close = float(df["Close"].iloc[-1])
        return TradeSignal(
            ticker=ticker,
            action="NO_TRADE",
            setup_type="NONE",
            entry_price=last_close,
            stop_loss=0.0,
            tp1=0.0,
            tp2=0.0,
            risk_per_share=0.0,
            risk_reward_ratio=0.0,
            shares=0.0,
            position_cost=0.0,
            reason=f"📉 MULTI-TIMEFRAME (1W): {weekly_reason}",
            rs_score=last_rs,
            sector=sector
        )

    # Filtru de Selectivitate Sectoriala: Sectoarele secundare (ex: FINANCIALS) au voie sa intre
    # DOAR daca sunt Super-Lideri cu Forta Relativa ridicata (RS >= +5%), ca sa nu ocupe locurile AI/Tech!
    tier1_sectors = {"SEMICONDUCTORS", "SOFTWARE_CLOUD", "CYBERSECURITY", "AI_POWER_INFRA"}
    if getattr(config, "USE_SECTOR_ROTATION", True) and sector not in tier1_sectors and last_rs < 3.5:
        last_close = float(df["Close"].iloc[-1])
        return TradeSignal(
            ticker=ticker,
            action="NO_TRADE",
            setup_type="NONE",
            entry_price=last_close,
            stop_loss=0.0,
            tp1=0.0,
            tp2=0.0,
            risk_per_share=0.0,
            risk_reward_ratio=0.0,
            shares=0.0,
            position_cost=0.0,
            reason=f"🎯 FOCUS AI/TECH: Simbolul din {sector} are RS prea mic ({last_rs:+.1f}% < +3.5%) pentru a concura cu liderii Tech/AI.",
            rs_score=last_rs,
            sector=sector
        )

    # 1. Testare Scenariul 1: Pullback Confirmat la EMA 20
    candidate = check_pullback_setup(ticker, df, total_capital, risk_multiplier=risk_multiplier)

    # 2. Testare Scenariul 2: Breakout de Volum (in Cautious Uptrend cerem RS >= +5% pentru extra siguranta)
    if not candidate:
        breakout = check_breakout_setup(ticker, df, total_capital, risk_multiplier=risk_multiplier)
        if breakout:
            if market_regime == "CAUTIOUS_UPTREND" and breakout.rs_score < 5.0:
                pass  # In piata precauta permitem breakout doar pe super-lideri (RS >= +5%)
            else:
                candidate = breakout

    if candidate:
        candidate.sector = sector
        candidate.reason = f"{candidate.reason} [📅 1W Stage 2]"

        # 3. Verificare Catalizatori Știri & Sentiment Live (Alpaca News API)
        try:
            from data_loader import check_news_sentiment
            is_blocked, news_score, news_label, news_headline = check_news_sentiment(ticker)
            candidate.news_sentiment = news_label
            candidate.news_headline = news_headline
            candidate.news_score = news_score

            if is_blocked:
                return TradeSignal(
                    ticker=ticker,
                    action="NO_TRADE",
                    setup_type="NONE",
                    entry_price=candidate.entry_price,
                    stop_loss=0.0,
                    tp1=0.0,
                    tp2=0.0,
                    risk_per_share=0.0,
                    risk_reward_ratio=0.0,
                    shares=0.0,
                    position_cost=0.0,
                    reason=f"📰 NEWS RISK FILTER: Catalizator negativ sever ({news_label}) — {news_headline}",
                    rs_score=last_rs,
                    sector=candidate.sector,
                    news_sentiment=news_label,
                    news_headline=news_headline,
                    news_score=news_score,
                )
        except Exception:
            pass

        return _apply_ml_filter(ticker, df, candidate)

    last = df.iloc[-1]
    close = float(last["Close"])
    open_p = float(last["Open"])
    sma_50 = float(last["SMA_50"])
    rsi = float(last["RSI_14"])

    if close < sma_50:
        reason = f"Sub trendul principal (Close {close:.2f} < SMA 50 {sma_50:.2f})"
    elif getattr(config, "USE_RELATIVE_STRENGTH", True) and last_rs < getattr(config, "MIN_RS_VS_SPY_PULLBACK", -5.0):
        reason = f"Forță relativă slabă față de S&P 500 (RS: {last_rs:+.1f}%)"
    elif close < open_p:
        reason = "Lumânare roșie în scădere (fără ricoșeu sau confirmare de cumpărători)"
    else:
        reason = f"În consolidare fără semnal (RSI: {rsi:.1f} | RS: {last_rs:+.1f}%)"

    return TradeSignal(
        ticker=ticker,
        action="NO_TRADE",
        setup_type="NONE",
        entry_price=close,
        stop_loss=0.0,
        tp1=0.0,
        tp2=0.0,
        risk_per_share=0.0,
        risk_reward_ratio=0.0,
        shares=0.0,
        position_cost=0.0,
        reason=reason,
        rs_score=last_rs
    )
