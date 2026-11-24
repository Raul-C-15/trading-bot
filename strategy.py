"""
Logica Strategiei Institutional Swing Trading V2 (Validată In-Sample & Out-of-Sample, Zero Look-Ahead Bias).
Sursa unică de adevăr pentru Scanarea Live și Backtester:
  1. Evaluare EXCLUSIV pe lumânări zilnice complet închise ('closed') și indicatori săptămânali cu shift(1).
  2. Filtru Multi-Timeframe (1D + 1W Stage 2):
       - 1D: Close > SMA 50 > SMA 200, EMA 20 > SMA 50, pante pozitive pe SMA 50 & SMA 200, volatilitate controlată.
       - 1W: Close > 1W SMA 50 / SMA 30, 1W EMA 10 >= 1W SMA 30, 1W RSI în zonă sănătoasă.
  3. SCENARIUL 1 — PULLBACK_EMA20 (Stage 2 Orderly Multi-Day Pullback):
       - Retragere ordonată pe mai multe zile (RSI_4 <= 30, RSI_14 38-54) pe volum scăzut (Volume Dry-Up),
         cu ordin Limit sub închidere și Stop-Loss sub suportul structural + ATR (fără tăiere în aer!).
  4. SCENARIUL 2 — VOLUME_BREAKOUT (3+ Week Tight VCP Breakout în Power Trend):
       - Contracție VCP (ATR_5/ATR_14 <= 0.85, bază 15D <= 8%), spargere peste maximul a 20 zile cu volum >= 150% SMA 20.
  5. Dimensionare dinamică din Stop-Loss, verificare obligatorie R:R >= 1:2.0 și Poartă de Calitate ML.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import config
from risk_learner import is_in_cooldown


def check_weekly_confluence(df: pd.DataFrame) -> Tuple[bool, str, dict]:
    """
    Verifică alinierea trendului pe graficul săptămânal (1W, cu shift(1) anti-lookahead):
      1. Close >= Weekly_SMA_30 (și peste Weekly_SMA_50 când există istoric suficient).
      2. Weekly_EMA_10 >= Weekly_SMA_30 (mediile săptămânale aliniate ascendent).
      3. Weekly_RSI_14 între WEEKLY_RSI_MIN și WEEKLY_RSI_MAX.
    """
    if not getattr(config, "USE_WEEKLY_FILTER", True):
        return True, "Filtru săptămânal dezactivat din config.", {}

    if len(df) < 30:
        return True, "Date insuficiente pentru calculul săptămânal.", {}

    last = df.iloc[-1]
    if isinstance(last, pd.DataFrame):
        last = last.iloc[0]

    close = float(last["Close"])
    weekly_ema = float(last.get("Weekly_EMA_10", np.nan))
    weekly_sma30 = float(last.get("Weekly_SMA_30", np.nan))
    weekly_sma50 = float(last.get("Weekly_SMA_50", np.nan))
    weekly_rsi = float(last.get("Weekly_RSI_14", np.nan))

    if np.isnan(weekly_ema) or np.isnan(weekly_sma30) or weekly_ema <= 0 or weekly_sma30 <= 0:
        return True, "Indicatori săptămânali indisponibili.", {}

    metrics = {
        "weekly_close": close,
        "weekly_ema10": weekly_ema,
        "weekly_sma30": weekly_sma30,
        "weekly_sma50": weekly_sma50,
        "weekly_rsi": weekly_rsi,
    }

    if close < weekly_sma30:
        pct_below = ((weekly_sma30 - close) / weekly_sma30) * 100.0
        return False, f"Sub trendul primar săptămânal (Close ${close:.2f} < 1W SMA 30 ${weekly_sma30:.2f}, -{pct_below:.1f}%)", metrics

    if weekly_ema < weekly_sma30:
        return False, f"Medii săptămânale inversate (1W EMA 10 ${weekly_ema:.2f} < 1W SMA 30 ${weekly_sma30:.2f})", metrics

    rsi_max = float(getattr(config, "WEEKLY_RSI_MAX", 72.0))
    rsi_min = float(getattr(config, "WEEKLY_RSI_MIN", 45.0))

    if not np.isnan(weekly_rsi):
        if weekly_rsi > rsi_max:
            return False, f"RSI săptămânal supracumpărat (1W RSI: {weekly_rsi:.1f} > {rsi_max})", metrics
        if weekly_rsi < rsi_min:
            return False, f"RSI săptămânal slab (1W RSI: {weekly_rsi:.1f} < {rsi_min})", metrics

    return True, f"1W Stage 2 Bullish: Close (${close:.2f}) > 1W EMA 10 (${weekly_ema:.2f}) >= SMA 30 (${weekly_sma30:.2f})", metrics


@dataclass
class TradeSignal:
    ticker: str
    action: str                  # 'BUY' sau 'NO_TRADE'
    setup_type: str              # 'PULLBACK_EMA20', 'VOLUME_BREAKOUT' sau 'NONE'
    entry_price: float           # Preț Limit propus pentru execuție
    stop_loss: float
    tp1: float                   # Take Profit 1 (50% scale-out + mutare SL la Breakeven)
    tp2: float                   # Take Profit 2 (50% Runner + Trailing pe EMA 20, R:R >= 2.0)
    risk_per_share: float
    risk_reward_ratio: float     # Raport R:R până la ținta principală (>= MIN_RISK_REWARD_RATIO)
    shares: float
    position_cost: float
    reason: str
    ml_confidence: float = 0.0
    rs_score: float = 0.0
    composite_score: float = 0.0
    sector: str = "OTHER"
    news_sentiment: str = "⚪ NEUTRU"
    news_headline: str = ""
    news_score: float = 0.0
    signal_date: str = ""        # Data lumânării închise pe care s-a confirmat semnalul (pt idempotență)
    signal_close: float = 0.0    # Prețul Close al lumânării închise
    limit_entry: bool = True     # True = ordin Limit de pullback


def calculate_position_size(
    entry_price: float,
    stop_loss: float,
    total_capital: float = config.DEFAULT_SIMULATED_CAPITAL,
    risk_percent: float = config.RISK_PER_TRADE_PERCENT,
    max_positions: int = config.MAX_OPEN_POSITIONS,
    risk_multiplier: float = 1.0
) -> dict:
    """
    Calculează dinamic mărimea poziției exclusiv din distanța până la Stop-Loss și riscul procentual:
      position_size = (total_capital * effective_risk_pct) / abs(entry_price - stop_loss)
    Respectă plafonul maxim pe o singură poziție (MAX_SINGLE_POSITION_PCT) și pragul minim anti-fragmentare.
    """
    if entry_price <= 0 or stop_loss <= 0 or entry_price <= stop_loss or total_capital <= 0:
        return {"shares": 0.0, "position_cost": 0.0, "cash_at_risk": 0.0, "risk_per_share": 0.0}

    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share <= 1e-6:
        return {"shares": 0.0, "position_cost": 0.0, "cash_at_risk": 0.0, "risk_per_share": 0.0}

    effective_risk_pct = risk_percent * max(0.0, min(1.25, risk_multiplier))
    max_allowed_dollar_risk = total_capital * effective_risk_pct

    shares = max_allowed_dollar_risk / risk_per_share

    max_single_pct = float(getattr(config, "MAX_SINGLE_POSITION_PCT", 1.0 / max(1, max_positions)))
    max_capital_per_position = total_capital * max_single_pct
    if (shares * entry_price) > max_capital_per_position:
        shares = max_capital_per_position / entry_price

    if getattr(config, "ALLOW_FRACTIONAL_SHARES", True):
        shares = round(shares, 4)
    else:
        shares = float(int(shares))

    position_cost = round(shares * entry_price, 2)
    min_alloc = max(
        float(getattr(config, "MIN_POSITION_COST_USD", 50.0)),
        total_capital * float(getattr(config, "MIN_POSITION_ALLOC_PCT", 0.12))
    )
    if shares <= 0 or (total_capital >= min_alloc and position_cost < min_alloc):
        return {"shares": 0.0, "position_cost": 0.0, "cash_at_risk": 0.0, "risk_per_share": round(risk_per_share, 2)}

    actual_dollar_risk = round(shares * risk_per_share, 2)
    return {
        "shares": shares,
        "position_cost": position_cost,
        "cash_at_risk": actual_dollar_risk,
        "risk_per_share": round(risk_per_share, 2)
    }


def _passes_common_stage2_quality(last: pd.Series) -> Tuple[bool, str]:
    """
    Verifică filtrele instituționale comune de calitate a trendului Stage 2 (1D):
      - Lichiditate >= MIN_AVG_VOLUME
      - Volatilitate ordonată: MIN_ATR_PCT <= ATR_Pct <= MAX_ATR_PCT
      - Aliniere Stage 2: Close > SMA 50 > SMA 200, EMA 20 > SMA 50 * 0.995
      - Pante pozitive atât pe SMA 50, cât și pe SMA 200
      - Fără zi de distribuție instituțională severă (lumânare roșie pe volum > 130% SMA 20)
    """
    close_p = float(last["Close"])
    open_p = float(last["Open"])
    sma_50 = float(last.get("SMA_50", np.nan))
    sma_200 = float(last.get("SMA_200", np.nan))
    ema_20 = float(last.get("EMA_20", np.nan))
    atr_14 = float(last.get("ATR_14", np.nan))
    vol_sma = float(last.get("Volume_SMA_20", 0.0))
    volume = float(last.get("Volume", 0.0))
    atr_pct = float(last.get("ATR_Pct", (atr_14 / close_p * 100.0) if close_p > 0 else 99.0))
    slope_50 = float(last.get("SMA_50_Slope", 0.0))
    slope_200 = float(last.get("SMA_200_Slope", 0.0))

    if any(np.isnan(x) for x in [sma_50, sma_200, ema_20, atr_14]):
        return False, "Indicatori zilnici incompleți (necesar > 200 lumânări)"

    min_vol = float(getattr(config, "MIN_AVG_VOLUME", 700_000))
    if vol_sma < min_vol:
        return False, f"Lichiditate sub pragul instituțional ({vol_sma:,.0f} < {min_vol:,.0f})"

    max_atr = float(getattr(config, "MAX_ATR_PCT", 4.3))
    min_atr = float(getattr(config, "MIN_ATR_PCT", 1.0))
    if atr_pct > max_atr:
        return False, f"Volatilitate prea ridicată / haotică (ATR {atr_pct:.2f}% > {max_atr:.1f}%)"
    if atr_pct < min_atr:
        return False, f"Amplitudine prea mică (ATR {atr_pct:.2f}%)"

    if not (close_p > sma_50 > sma_200):
        return False, f"Nu este în Stage 2 Uptrend (Close ${close_p:.2f}, SMA50 ${sma_50:.2f}, SMA200 ${sma_200:.2f})"

    if ema_20 <= sma_50 * 0.995:
        return False, f"Trendul scurt s-a deteriorat (EMA 20 ${ema_20:.2f} <= SMA 50 ${sma_50:.2f})"

    if slope_50 <= 0.0 or slope_200 <= 0.0:
        return False, f"Mediile mobile nu au pantă ascendentă (Slope50: {slope_50:+.2f}%, Slope200: {slope_200:+.2f}%)"

    if close_p < open_p and vol_sma > 0 and volume > 1.30 * vol_sma:
        return False, "Distribuție instituțională detectată (vânzare pe volum ridicat)"

    return True, "OK"


def check_pullback_setup(
    ticker: str,
    df: pd.DataFrame,
    total_capital: float,
    rsi_min: Optional[float] = None,
    rsi_max: Optional[float] = None,
    min_rr: Optional[float] = None,
    ema_period: int = 20,
    vol_mult: Optional[float] = None,
    risk_multiplier: float = 1.0,
    rsi4_max: Optional[float] = None
) -> Optional[TradeSignal]:
    """
    SCENARIUL 1: PULLBACK ORDONAT ÎN STAGE 2 UPTREND (Validat OOS cu ~56% Pos WR / ~72% Lot WR)
      - Acțiune lider (RS_vs_SPY >= MIN_RS_VS_SPY_PULLBACK, fără extindere parabolică)
      - Trend Stage 2 intact (Close > SMA 50 * 1.01 > SMA 200, EMA > SMA 50, pante pozitive)
      - Retragere ordonată pe mai multe zile (RSI_4 <= 30, RSI_14 între 38 și 54) cu Volume Dry-Up
      - Intrare prin ordin LIMIT (-0.3% sub Close), Stop-Loss sub suportul structural + 1.8x ATR
      - Respinge tranzacția dacă Stop-Loss-ul depășește MAX_STOP_LOSS_PCT sau dacă R:R < MIN_RISK_REWARD_RATIO
    """
    if len(df) < 205:
        return None

    rsi_min = float(rsi_min if rsi_min is not None else config.RSI_PULLBACK_MIN)
    rsi_max = float(rsi_max if rsi_max is not None else config.RSI_PULLBACK_MAX)
    rsi4_cap = float(rsi4_max if rsi4_max is not None else getattr(config, "RSI4_PULLBACK_MAX", 30.0))
    required_min_rr = float(min_rr if min_rr is not None else config.MIN_RISK_REWARD_RATIO)
    tp1_rr = float(getattr(config, "TP1_RISK_REWARD_RATIO", 1.5))
    tp2_rr = max(required_min_rr, float(getattr(config, "TP2_RISK_REWARD_RATIO", 2.5)))

    last = df.iloc[-1]
    ok_q, _ = _passes_common_stage2_quality(last)
    if not ok_q:
        return None

    close_p = float(last["Close"])
    low_p = float(last["Low"])
    sma_50 = float(last["SMA_50"])
    ema_col = f"EMA_{ema_period}"
    ema_val = float(last[ema_col]) if ema_col in last and not pd.isna(last[ema_col]) else float(last["EMA_20"])
    if ema_val <= sma_50 * 0.995:
        return None

    rsi_14 = float(last["RSI_14"])
    rsi_4 = float(last.get("RSI_4", 50.0))
    atr_14 = float(last["ATR_14"])
    vol_ratio_3d = float(last.get("Volume_Ratio_3D", 1.0))
    rs_vs_spy = float(last.get("RS_vs_SPY", 0.0))

    if getattr(config, "USE_RELATIVE_STRENGTH", True):
        min_rs = float(getattr(config, "MIN_RS_VS_SPY_PULLBACK", 2.0))
        max_rs = float(getattr(config, "MAX_RS_VS_SPY_PARABOLIC", 45.0))
        if np.isnan(rs_vs_spy) or rs_vs_spy < min_rs or rs_vs_spy > max_rs:
            return None

    max_vol_r3 = float(getattr(config, "PULLBACK_MAX_VOL_RATIO_3D", 1.05))
    if not (rsi_4 <= rsi4_cap and (rsi_min <= rsi_14 <= rsi_max) and close_p >= sma_50 * 1.01 and vol_ratio_3d <= max_vol_r3):
        return None

    lim_disc = float(getattr(config, "PULLBACK_LIMIT_DISCOUNT_PCT", 0.003))
    entry_limit = round(close_p * (1.0 - lim_disc), 2)

    atr_mult = float(getattr(config, "ATR_STOP_MULTIPLIER", 1.8))
    stop_loss = round(min(sma_50 * 0.985, low_p - (0.5 * atr_14), close_p - (atr_mult * atr_14)), 2)

    risk_per_share = round(entry_limit - stop_loss, 2)
    if risk_per_share <= 0:
        return None

    stop_dist_pct = risk_per_share / entry_limit
    min_sl_pct = float(getattr(config, "MIN_STOP_LOSS_PCT", 0.022))
    max_sl_pct = float(getattr(config, "MAX_STOP_LOSS_PCT", 0.075))
    if stop_dist_pct < min_sl_pct or stop_dist_pct > max_sl_pct:
        return None

    tp1 = round(entry_limit + (tp1_rr * risk_per_share), 2)
    tp2 = round(entry_limit + (tp2_rr * risk_per_share), 2)
    actual_rr = round((tp2 - entry_limit) / risk_per_share, 2)
    if actual_rr < required_min_rr:
        return None

    sizing = calculate_position_size(
        entry_limit, stop_loss, total_capital,
        risk_percent=config.RISK_PER_TRADE_PERCENT,
        risk_multiplier=risk_multiplier
    )
    if sizing["shares"] <= 0:
        return None

    sig_date = pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d")
    rs_tag = f" | RS: {rs_vs_spy:+.1f}%"
    return TradeSignal(
        ticker=ticker,
        action="BUY",
        setup_type="PULLBACK_EMA20",
        entry_price=entry_limit,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        risk_per_share=sizing["risk_per_share"],
        risk_reward_ratio=actual_rr,
        shares=sizing["shares"],
        position_cost=sizing["position_cost"],
        reason=f"PULLBACK STAGE 2 CONFIRMAT: RSI(4)={rsi_4:.1f}, VolDry={vol_ratio_3d:.2f}x, R:R 1:{actual_rr:.1f}{rs_tag}.",
        rs_score=round(rs_vs_spy, 2),
        signal_date=sig_date,
        signal_close=round(close_p, 2),
        limit_entry=True
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
    SCENARIUL 2: VCP VOLUME BREAKOUT (Bază strânsă >= 3 săptămâni + Volum >= 150% SMA 20)
      - Contracție de volatilitate (ATR_Contraction <= 0.85 și Range_15D_Pct <= 8%)
      - Spargere peste Swing_High_20 cu închidere în top 30% din lumânare și volum >= 1.5x media
      - Fără supra-extindere (<= 7% față de EMA 20) și R:R >= MIN_RISK_REWARD_RATIO (1:2.0+)
    """
    if len(df) < 205:
        return None

    required_min_rr = float(min_rr if min_rr is not None else config.MIN_RISK_REWARD_RATIO)
    vol_mult = float(vol_mult if vol_mult is not None else getattr(config, "VOLUME_SURGE_MULTIPLIER", 1.50))
    tp1_rr = float(getattr(config, "TP1_RISK_REWARD_RATIO", 1.5))
    tp2_rr = max(required_min_rr, float(getattr(config, "TP2_RISK_REWARD_RATIO", 2.5)))

    last = df.iloc[-1]
    ok_q, _ = _passes_common_stage2_quality(last)
    if not ok_q:
        return None

    open_p = float(last["Open"])
    high_p = float(last["High"])
    low_p = float(last["Low"])
    close_p = float(last["Close"])
    ema_20 = float(last["EMA_20"])
    rsi_14 = float(last["RSI_14"])
    atr_14 = float(last["ATR_14"])
    volume = float(last["Volume"])
    vol_sma_20 = float(last.get("Volume_SMA_20", 0.0))
    rs_vs_spy = float(last.get("RS_vs_SPY", 0.0))
    atr_c = float(last.get("ATR_Contraction", 1.0))
    range_15d = float(last.get("Range_15D_Pct", 12.0))
    prior_20_high = float(last.get("Swing_High_20", np.nan))
    swing_low_10 = float(last.get("Swing_Low_10_All", low_p))

    if np.isnan(prior_20_high) or close_p <= open_p:
        return None

    if getattr(config, "USE_RELATIVE_STRENGTH", True):
        min_rs_bo = float(getattr(config, "MIN_RS_VS_SPY_BREAKOUT", 3.0))
        max_rs = float(getattr(config, "MAX_RS_VS_SPY_PARABOLIC", 45.0))
        if np.isnan(rs_vs_spy) or rs_vs_spy < min_rs_bo or rs_vs_spy > max_rs:
            return None

    max_ext = float(getattr(config, "MAX_BREAKOUT_EXTENSION_EMA20", 0.07))
    if ema_20 > 0 and ((close_p - ema_20) / ema_20) > max_ext:
        return None

    max_vcp_c = float(getattr(config, "MAX_VCP_ATR_CONTRACTION", 0.85))
    max_base_rng = float(getattr(config, "MAX_VCP_BASE_RANGE_PCT", 8.0))
    if atr_c > max_vcp_c or range_15d > max_base_rng:
        return None

    day_range = high_p - low_p
    if day_range <= 0 or ((close_p - low_p) / day_range) < 0.70:
        return None

    is_breakout = close_p > prior_20_high
    has_volume_surge = (vol_sma_20 > 0) and (volume >= vol_mult * vol_sma_20)
    rsi_valid = 54.0 <= rsi_14 <= 73.0
    if not (is_breakout and has_volume_surge and rsi_valid):
        return None

    atr_mult = float(getattr(config, "ATR_STOP_MULTIPLIER", 1.8))
    stop_loss = round(min(swing_low_10, close_p - (atr_mult * atr_14)), 2)
    risk_per_share = round(close_p - stop_loss, 2)
    if risk_per_share <= 0:
        return None

    stop_dist_pct = risk_per_share / close_p
    min_sl_pct = float(getattr(config, "MIN_STOP_LOSS_PCT", 0.022))
    max_sl_pct = float(getattr(config, "MAX_STOP_LOSS_PCT", 0.075))
    if stop_dist_pct < min_sl_pct or stop_dist_pct > max_sl_pct:
        return None

    tp1 = round(close_p + (tp1_rr * risk_per_share), 2)
    tp2 = round(close_p + (tp2_rr * risk_per_share), 2)
    actual_rr = round((tp2 - close_p) / risk_per_share, 2)
    if actual_rr < required_min_rr:
        return None

    sizing = calculate_position_size(
        close_p, stop_loss, total_capital,
        risk_percent=config.RISK_PER_TRADE_PERCENT,
        risk_multiplier=risk_multiplier
    )
    if sizing["shares"] <= 0:
        return None

    vol_pct = int((volume / vol_sma_20) * 100) if vol_sma_20 > 0 else 0
    sig_date = pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d")
    return TradeSignal(
        ticker=ticker,
        action="BUY",
        setup_type="VOLUME_BREAKOUT",
        entry_price=round(close_p, 2),
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        risk_per_share=sizing["risk_per_share"],
        risk_reward_ratio=actual_rr,
        shares=sizing["shares"],
        position_cost=sizing["position_cost"],
        reason=f"VCP BREAKOUT CONFIRMAT: Spargere ${prior_20_high:.2f} pe volum {vol_pct}% (VCP {atr_c:.2f}, R:R 1:{actual_rr:.1f}).",
        rs_score=round(rs_vs_spy, 2),
        signal_date=sig_date,
        signal_close=round(close_p, 2),
        limit_entry=False
    )


def compute_quality_composite_score(
    rs_score: float,
    atr_pct: float,
    atr_contraction: float,
    sector: str,
    ml_confidence: float = 0.0,
    ml_has_edge: bool = False,
    news_score: float = 0.0,
    use_live_sector_cache: bool = True
) -> float:
    """
    Calculează Scorul Compozit de Calitate (0 - 100) bazat pe Momentum Ajustat la Risc (RS / ATR_Pct)
    și contracția VCP. Modelul ML contribuie DOAR dacă a demonstrat putere predictivă reală
    Out-of-Sample (ROC-AUC >= ML_MIN_OOS_ROC_AUC).
    """
    risk_adj_rs = (max(0.0, min(40.0, rs_score)) / max(1.5, atr_pct)) * 3.8  # ~0..65 puncte
    vcp_points = max(0.0, min(18.0, (1.15 - min(1.15, atr_contraction)) * 36.0))

    if use_live_sector_cache:
        try:
            from data_loader import get_live_sector_bonus
            sec_bonus = get_live_sector_bonus(sector)
        except Exception:
            sec_bonus = config.get_sector_priority_bonus(sector)
    else:
        sec_bonus = config.get_sector_priority_bonus(sector)

    news_bonus = max(-4.0, min(4.0, float(news_score) * 2.0))
    base_score = risk_adj_rs + vcp_points + sec_bonus + news_bonus

    if ml_has_edge and ml_confidence > 0:
        ml_boost = (ml_confidence - 0.50) * 35.0
        base_score = 0.75 * base_score + 0.25 * (50.0 + ml_boost)

    return round(max(0.0, min(100.0, base_score)), 2)


def _apply_ml_filter(ticker: str, df: pd.DataFrame, signal: TradeSignal) -> TradeSignal:
    """
    Aplică filtrul ML DOAR dacă modelul are putere predictivă validată Out-of-Sample
    (ROC-AUC >= config.ML_MIN_OOS_ROC_AUC). În caz contrar, nu introduce zgomot aleatoriu.
    """
    last = df.iloc[-1]
    atr_pct = float(last.get("ATR_Pct", 2.5))
    atr_c = float(last.get("ATR_Contraction", 1.0))

    if not getattr(config, "ML_ENABLED", True):
        signal.composite_score = compute_quality_composite_score(
            signal.rs_score, atr_pct, atr_c, signal.sector, news_score=signal.news_score
        )
        return signal

    try:
        from ml_predictor import get_predictor
        predictor = get_predictor()
        if not predictor.is_ready or not predictor.has_predictive_edge:
            signal.composite_score = compute_quality_composite_score(
                signal.rs_score, atr_pct, atr_c, signal.sector, news_score=signal.news_score
            )
            return signal

        confidence, is_confident = predictor.predict(df)
        threshold_pct = int(getattr(config, "ML_MIN_CONFIDENCE", 0.53) * 100)

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
                    f"🤖 ML FILTER: Setup {signal.setup_type} respins de modelul validat OOS "
                    f"({confidence*100:.1f}% < {threshold_pct}%)."
                ),
                ml_confidence=confidence,
                rs_score=signal.rs_score,
                sector=signal.sector,
                signal_date=signal.signal_date
            )

        signal.ml_confidence = confidence
        signal.composite_score = compute_quality_composite_score(
            signal.rs_score, atr_pct, atr_c, signal.sector,
            ml_confidence=confidence, ml_has_edge=True, news_score=signal.news_score
        )
        signal.reason = f"{signal.reason} [🧠 ML OOS: {confidence*100:.1f}% | Scor: {signal.composite_score:.0f}/100]"
        return signal

    except Exception:
        signal.composite_score = compute_quality_composite_score(
            signal.rs_score, atr_pct, atr_c, signal.sector, news_score=signal.news_score
        )
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
    Evaluează un simbol strict pe lumânări zilnice închise ('closed'), aplicând:
      - Filtrul de Regim de Piață, Cooldown și Protecție Earnings (7 zile)
      - Confluența Multi-Timeframe 1W + 1D Stage 2
      - Setup 1 (Pullback Ordonat Stage 2) și Setup 2 (VCP Breakout în Power Trend)
      - Filtrul de Știri negative severe și Poarta de Calitate ML
    """
    last_rs = float(df["RS_vs_SPY"].iloc[-1]) if len(df) > 0 and "RS_vs_SPY" in df.columns and not pd.isna(df["RS_vs_SPY"].iloc[-1]) else 0.0
    sector = config.get_ticker_sector(ticker)

    if getattr(config, "USE_SECTOR_ROTATION", True) and config.is_sector_blocked(sector):
        return TradeSignal(
            ticker=ticker, action="NO_TRADE", setup_type="NONE",
            entry_price=0.0, stop_loss=0.0, tp1=0.0, tp2=0.0,
            risk_per_share=0.0, risk_reward_ratio=0.0, shares=0.0, position_cost=0.0,
            reason=f"🚫 Sector exclus din configurație ({sector}).",
            rs_score=last_rs, sector=sector
        )

    if not is_market_bullish or risk_multiplier <= 0:
        return TradeSignal(
            ticker=ticker, action="NO_TRADE", setup_type="NONE",
            entry_price=0.0, stop_loss=0.0, tp1=0.0, tp2=0.0,
            risk_per_share=0.0, risk_reward_ratio=0.0, shares=0.0, position_cost=0.0,
            reason="Piața (SPY/QQQ/VIX) este în regim defensiv/corecție — Capital 100% protejat în Cash.",
            rs_score=last_rs, sector=sector
        )

    in_cooldown, cooldown_msg = is_in_cooldown(ticker)
    if in_cooldown:
        return TradeSignal(
            ticker=ticker, action="NO_TRADE", setup_type="NONE",
            entry_price=0.0, stop_loss=0.0, tp1=0.0, tp2=0.0,
            risk_per_share=0.0, risk_reward_ratio=0.0, shares=0.0, position_cost=0.0,
            reason=f"🔒 BLOCAT ÎN CARANTINĂ: {cooldown_msg}",
            rs_score=last_rs, sector=sector
        )

    if len(df) < 205:
        return TradeSignal(
            ticker=ticker, action="NO_TRADE", setup_type="NONE",
            entry_price=0.0, stop_loss=0.0, tp1=0.0, tp2=0.0,
            risk_per_share=0.0, risk_reward_ratio=0.0, shares=0.0, position_cost=0.0,
            reason="Date istorice insuficiente (< 205 zile închise pentru SMA 200).",
            rs_score=last_rs, sector=sector
        )

    weekly_ok, weekly_reason, _ = check_weekly_confluence(df)
    if not weekly_ok:
        last_close = float(df["Close"].iloc[-1])
        return TradeSignal(
            ticker=ticker, action="NO_TRADE", setup_type="NONE",
            entry_price=last_close, stop_loss=0.0, tp1=0.0, tp2=0.0,
            risk_per_share=0.0, risk_reward_ratio=0.0, shares=0.0, position_cost=0.0,
            reason=f"📉 MULTI-TIMEFRAME (1W): {weekly_reason}",
            rs_score=last_rs, sector=sector
        )

    ok_stage2, stage2_reason = _passes_common_stage2_quality(df.iloc[-1])
    if not ok_stage2:
        last_close = float(df["Close"].iloc[-1])
        return TradeSignal(
            ticker=ticker, action="NO_TRADE", setup_type="NONE",
            entry_price=last_close, stop_loss=0.0, tp1=0.0, tp2=0.0,
            risk_per_share=0.0, risk_reward_ratio=0.0, shares=0.0, position_cost=0.0,
            reason=stage2_reason,
            rs_score=last_rs, sector=sector
        )

    # 1. Testare Scenariul 1: Pullback Ordonat în Stage 2
    candidate = check_pullback_setup(ticker, df, total_capital, risk_multiplier=risk_multiplier)

    # 2. Testare Scenariul 2: VCP Breakout (opțional din config, doar în POWER_TREND)
    if not candidate and market_regime == "POWER_TREND" and getattr(config, "ENABLE_BREAKOUT_SETUP", False):
        candidate = check_breakout_setup(ticker, df, total_capital, risk_multiplier=risk_multiplier)

    if candidate:
        # Verificăm riscul de Earnings doar pentru candidații tehnici valizi (evită apeluri API lente inutile)
        try:
            from data_loader import check_earnings_risk
            earnings_risky, earnings_msg = check_earnings_risk(
                ticker, buffer_days=getattr(config, "EARNINGS_BUFFER_DAYS", 7)
            )
            if earnings_risky:
                return TradeSignal(
                    ticker=ticker, action="NO_TRADE", setup_type="NONE",
                    entry_price=candidate.entry_price, stop_loss=0.0, tp1=0.0, tp2=0.0,
                    risk_per_share=0.0, risk_reward_ratio=0.0, shares=0.0, position_cost=0.0,
                    reason=earnings_msg, rs_score=last_rs, sector=sector
                )
        except Exception:
            pass

        candidate.sector = sector
        candidate.reason = f"{candidate.reason} [📅 1W+1D Stage 2]"

        try:
            from data_loader import check_news_sentiment
            is_blocked, news_score, news_label, news_headline = check_news_sentiment(ticker)
            candidate.news_sentiment = news_label
            candidate.news_headline = news_headline
            candidate.news_score = news_score

            if is_blocked:
                return TradeSignal(
                    ticker=ticker, action="NO_TRADE", setup_type="NONE",
                    entry_price=candidate.entry_price, stop_loss=0.0, tp1=0.0, tp2=0.0,
                    risk_per_share=0.0, risk_reward_ratio=0.0, shares=0.0, position_cost=0.0,
                    reason=f"📰 NEWS RISK FILTER: Catalizator negativ sever ({news_label}) — {news_headline}",
                    rs_score=last_rs, sector=sector,
                    news_sentiment=news_label, news_headline=news_headline, news_score=news_score
                )
        except Exception:
            pass

        return _apply_ml_filter(ticker, df, candidate)

    last = df.iloc[-1]
    close_p = float(last["Close"])
    rsi_14 = float(last["RSI_14"])
    rsi_4 = float(last.get("RSI_4", 50.0))

    if last_rs < float(getattr(config, "MIN_RS_VS_SPY_PULLBACK", 2.0)):
        reason = f"Forță relativă sub pragul liderilor (RS: {last_rs:+.1f}%)"
    else:
        reason = f"În trend Stage 2, așteptăm pullback ordonat (RSI4: {rsi_4:.0f} | RSI14: {rsi_14:.1f} | RS: {last_rs:+.1f}%)"

    return TradeSignal(
        ticker=ticker, action="NO_TRADE", setup_type="NONE",
        entry_price=close_p, stop_loss=0.0, tp1=0.0, tp2=0.0,
        risk_per_share=0.0, risk_reward_ratio=0.0, shares=0.0, position_cost=0.0,
        reason=reason, rs_score=last_rs, sector=sector
    )
