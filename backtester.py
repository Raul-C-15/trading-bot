"""
Motor de Backtesting Instituțional V2 (Zero Look-Ahead Bias, Costuri Reale & Validare Portofoliu):
  - ZERO Look-Ahead Bias: Semnalele sunt confirmate exclusiv la închiderea barei `t` ('closed')
    și se execută cel mai devreme pe bara `t + 1` (la Open / Limit, cu filtru de Gap).
  - Costuri reale deduse din PnL-ul net: Comision broker + Half-Spread (Bid/Ask) + Slippage defavorabil
    aplicat atât la intrare, cât și la ieșire / Gap-Down la Stop-Loss.
  - Aliniere 100% cu `strategy.py` și `position_manager.py`:
      * Verificare conservatoare Stop-Loss ÎNAINTE de Take-Profit pe aceeași bară.
      * TP1 la 1.5R (50% scale-out + mutare Stop-Loss la Breakeven).
      * Trailing Stop pe EMA 20 zilnic EXCLUSIV după atingerea TP1, până la TP2 (>= 2.0R - 2.5R).
  - Limite stricte de portofoliu: Portfolio Heat (`MAX_OPEN_POSITIONS`), `MAX_TOTAL_POSITIONS`,
    expunere maximă pe sector (`MAX_POSITIONS_PER_SECTOR`) și raportare completă a metricilor de risc
    (`Max Drawdown %`, `Profit Factor`, `Sharpe Ratio`, `Sortino Ratio`, `Total Trades`, `Expectancy R`).
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import config
from strategy import compute_quality_composite_score


class Backtester:
    def __init__(
        self,
        initial_capital: float = config.DEFAULT_SIMULATED_CAPITAL,
        risk_per_trade: float = config.RISK_PER_TRADE_PERCENT,
        rsi_pullback_min: float = config.RSI_PULLBACK_MIN,
        rsi_pullback_max: float = config.RSI_PULLBACK_MAX,
        rsi4_pullback_max: float = config.RSI4_PULLBACK_MAX,
        min_risk_reward: float = config.MIN_RISK_REWARD_RATIO,
        ema_period: int = config.EMA_PULLBACK_PERIOD,
        volume_surge_mult: float = config.VOLUME_SURGE_MULTIPLIER,
        use_ml: bool = False,
        use_weekly_filter: bool = getattr(config, "USE_WEEKLY_FILTER", True),
        use_rs_filter: bool = getattr(config, "USE_RELATIVE_STRENGTH", True),
        slippage_pct: float = getattr(config, "BACKTEST_SLIPPAGE_PCT", 0.0006),
        spread_pct: float = getattr(config, "BACKTEST_SPREAD_PCT", 0.0008),
        commission_usd: float = getattr(config, "BACKTEST_COMMISSION_USD", 0.02),
    ):
        self.initial_capital = float(initial_capital)
        self.capital = float(initial_capital)
        self.risk_per_trade = float(risk_per_trade)
        self.rsi_pullback_min = float(rsi_pullback_min)
        self.rsi_pullback_max = float(rsi_pullback_max)
        self.rsi4_pullback_max = float(rsi4_pullback_max)
        self.min_risk_reward = float(min_risk_reward)
        self.ema_period = int(ema_period)
        self.volume_surge_mult = float(volume_surge_mult)
        self.use_ml = bool(use_ml)
        self.use_weekly_filter = bool(use_weekly_filter)
        self.use_rs_filter = bool(use_rs_filter)
        self.slippage_pct = float(slippage_pct)
        self.spread_pct = float(spread_pct)
        self.one_way_cost_pct = (self.spread_pct * 0.5) + self.slippage_pct
        self.commission_usd = float(commission_usd)
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [self.initial_capital]

    def _classify_market_bar(
        self,
        date,
        spy_df: Optional[pd.DataFrame],
        qqq_df: Optional[pd.DataFrame] = None,
        vix_df: Optional[pd.DataFrame] = None,
    ) -> tuple:
        """Determină cauzal regimul pieței la închiderea barei `date`."""
        if spy_df is None or date not in spy_df.index:
            return "POWER_TREND", 1.0, 16.0

        sb = spy_df.loc[date]
        if isinstance(sb, pd.DataFrame):
            sb = sb.iloc[0]

        sc = float(sb["Close"])
        ss50 = float(sb.get("SMA_50", np.nan))
        ss200 = float(sb.get("SMA_200", np.nan))
        spy_atr_pct = float(sb.get("ATR_Pct", 1.0))

        if np.isnan(ss200) or sc <= ss200:
            return "BEARISH", 0.0, 28.0

        if qqq_df is not None and date in qqq_df.index:
            qb = qqq_df.loc[date]
            if isinstance(qb, pd.DataFrame):
                qb = qb.iloc[0]
            qqq_ok = float(qb["Close"]) >= float(qb.get("SMA_50", qb["Close"])) * 0.995
        else:
            qqq_ok = not np.isnan(ss50) and sc >= ss50 * 0.995

        if vix_df is not None and date in vix_df.index:
            vb = vix_df.loc[date]
            vix_val = float(vb["Close"].iloc[0] if isinstance(vb, pd.DataFrame) else vb["Close"])
        else:
            # Proxy cauzal din volatilitatea SPY dacă ^VIX nu este furnizat separat
            vix_val = spy_atr_pct * 14.5

        vix_bear = float(getattr(config, "VIX_BEARISH_THRESHOLD", 25.0))
        vix_pow = float(getattr(config, "VIX_POWER_TREND_MAX", 21.5))

        if vix_val >= vix_bear or ((not np.isnan(ss50) and sc < ss50) and not qqq_ok):
            return "BEARISH", 0.0, vix_val
        if (not np.isnan(ss50) and sc >= ss50) and qqq_ok and vix_val < vix_pow:
            return "POWER_TREND", 1.0, vix_val
        return "CAUTIOUS_UPTREND", float(getattr(config, "CAUTIOUS_RISK_MULTIPLIER", 0.70)), vix_val

    def _check_entry_on_bar(
        self,
        ticker: str,
        df: pd.DataFrame,
        i: int,
        regime: str,
        risk_mult: float,
        vix_val: float = 18.0,
    ) -> Optional[Dict]:
        """
        Evaluează la închiderea barei `i` (lumânare 100% închisă) dacă există un semnal valid
        pentru execuție la bara `i + 1` (fără acces la viitor).
        """
        if regime == "BEARISH" or risk_mult <= 0.0:
            return None

        bar = df.iloc[i]
        open_p = float(bar["Open"])
        high_p = float(bar["High"])
        low_p = float(bar["Low"])
        close_p = float(bar["Close"])

        ema_col = f"EMA_{self.ema_period}"
        ema_val = float(bar[ema_col]) if ema_col in bar and not pd.isna(bar[ema_col]) else float(bar.get("EMA_20", np.nan))
        sma_50 = float(bar.get("SMA_50", np.nan))
        sma_200 = float(bar.get("SMA_200", np.nan))
        slope_50 = float(bar.get("SMA_50_Slope", 0.0))
        slope_200 = float(bar.get("SMA_200_Slope", 0.0))
        rsi_14 = float(bar.get("RSI_14", 50.0))
        rsi_4 = float(bar.get("RSI_4", 50.0))
        atr_14 = float(bar.get("ATR_14", np.nan))
        atr_pct = float(bar.get("ATR_Pct", 2.5))
        atr_c = float(bar.get("ATR_Contraction", 1.0))
        volume = float(bar.get("Volume", 0.0))
        vol_sma = float(bar.get("Volume_SMA_20", 0.0))
        vol_r3 = float(bar.get("Volume_Ratio_3D", 1.0))
        rs_vs_spy = float(bar.get("RS_vs_SPY", 0.0))
        range_15d = float(bar.get("Range_15D_Pct", 12.0))
        prior_20_high = float(bar.get("Swing_High_20", np.nan))
        swing_low_10 = float(bar.get("Swing_Low_10_All", low_p))

        if any(np.isnan(x) for x in [sma_50, sma_200, ema_val, atr_14]) or vol_sma < float(getattr(config, "MIN_AVG_VOLUME", 700_000)):
            return None

        max_atr = float(getattr(config, "MAX_ATR_PCT", 4.3))
        min_atr = float(getattr(config, "MIN_ATR_PCT", 1.0))
        if atr_pct > max_atr or atr_pct < min_atr:
            return None

        # 1. Stage 2 Trend Zilnic
        if not (close_p > sma_50 > sma_200 and ema_val > sma_50 * 0.995 and slope_50 > 0.0 and slope_200 > 0.0):
            return None
        if close_p < open_p and volume > 1.30 * vol_sma:
            return None

        # 2. Confluență Săptămânală (1W cu shift(1))
        if self.use_weekly_filter:
            w_ema = float(bar.get("Weekly_EMA_10", np.nan))
            w_sma30 = float(bar.get("Weekly_SMA_30", np.nan))
            w_rsi = float(bar.get("Weekly_RSI_14", np.nan))
            if not np.isnan(w_ema) and not np.isnan(w_sma30) and w_ema > 0 and w_sma30 > 0:
                if close_p < w_sma30 or w_ema < w_sma30:
                    return None
                if not np.isnan(w_rsi):
                    if w_rsi > float(getattr(config, "WEEKLY_RSI_MAX", 72.0)) or w_rsi < float(getattr(config, "WEEKLY_RSI_MIN", 45.0)):
                        return None

        # 3. Forță Relativă (RS vs SPY)
        if self.use_rs_filter:
            min_rs = float(getattr(config, "MIN_RS_VS_SPY_PULLBACK", 2.0))
            max_rs = float(getattr(config, "MAX_RS_VS_SPY_PARABOLIC", 45.0))
            if np.isnan(rs_vs_spy) or rs_vs_spy < min_rs or rs_vs_spy > max_rs:
                return None

        setup_type = None
        stop_loss = 0.0
        limit_entry = True
        limit_price = close_p
        atr_mult = float(getattr(config, "ATR_STOP_MULTIPLIER", 1.8))

        # SCENARIUL 1: Pullback Ordonat în Stage 2
        max_vol_r3 = float(getattr(config, "PULLBACK_MAX_VOL_RATIO_3D", 1.05))
        if (
            rsi_4 <= self.rsi4_pullback_max
            and (self.rsi_pullback_min <= rsi_14 <= self.rsi_pullback_max)
            and close_p >= sma_50 * 1.01
            and vol_r3 <= max_vol_r3
        ):
            setup_type = "PULLBACK_EMA20"
            lim_disc = float(getattr(config, "PULLBACK_LIMIT_DISCOUNT_PCT", 0.003))
            limit_price = round(close_p * (1.0 - lim_disc), 2)
            stop_loss = round(min(sma_50 * 0.985, low_p - 0.5 * atr_14, close_p - atr_mult * atr_14), 2)
            limit_entry = True

        # SCENARIUL 2: VCP Breakout (opțional din config, doar în POWER_TREND calm, VIX < 19.5)
        elif getattr(config, "ENABLE_BREAKOUT_SETUP", False) and regime == "POWER_TREND" and vix_val < 19.5 and not np.isnan(prior_20_high):
            day_rng = high_p - low_p
            strong_close = day_rng > 0 and ((close_p - low_p) / day_rng) >= 0.70
            not_ext = ((close_p - ema_val) / ema_val) <= float(getattr(config, "MAX_BREAKOUT_EXTENSION_EMA20", 0.07))
            vcp_ok = (
                atr_c <= float(getattr(config, "MAX_VCP_ATR_CONTRACTION", 0.85))
                and range_15d <= float(getattr(config, "MAX_VCP_BASE_RANGE_PCT", 8.0))
            )
            rs_bo_ok = (not self.use_rs_filter) or (rs_vs_spy >= float(getattr(config, "MIN_RS_VS_SPY_BREAKOUT", 3.0)))
            if (
                close_p > prior_20_high
                and close_p > open_p
                and volume >= self.volume_surge_mult * vol_sma
                and (54.0 <= rsi_14 <= 73.0)
                and strong_close
                and not_ext
                and vcp_ok
                and rs_bo_ok
            ):
                setup_type = "VOLUME_BREAKOUT"
                limit_price = round(close_p, 2)
                stop_loss = round(min(swing_low_10, close_p - atr_mult * atr_14), 2)
                limit_entry = False

        if setup_type is None:
            return None

        ref_risk = limit_price - stop_loss
        min_sl_pct = float(getattr(config, "MIN_STOP_LOSS_PCT", 0.022))
        max_sl_pct = float(getattr(config, "MAX_STOP_LOSS_PCT", 0.075))
        if ref_risk <= 0 or (ref_risk / limit_price) < min_sl_pct or (ref_risk / limit_price) > max_sl_pct:
            return None

        # Filtru opțional ML (doar dacă modelul are putere predictivă Out-of-Sample reală, fără Data Leakage)
        ml_conf = 0.0
        ml_edge = False
        if self.use_ml:
            try:
                from ml_predictor import get_predictor
                pred = get_predictor()
                if pred.is_ready and pred.has_predictive_edge:
                    ml_edge = True
                    bar_date_str = pd.Timestamp(df.index[i]).strftime("%Y-%m-%d")
                    prob_win, is_conf = pred.predict_for_backtest(bar, vol_sma, bar_date_str)
                    ml_conf = prob_win
                    if not is_conf:
                        return None
            except Exception:
                pass

        sec = config.get_ticker_sector(ticker)
        comp_score = compute_quality_composite_score(
            rs_vs_spy, atr_pct, atr_c, sec, ml_confidence=ml_conf, ml_has_edge=ml_edge, use_live_sector_cache=False
        )

        return {
            "ticker": ticker,
            "setup_type": setup_type,
            "signal_date": pd.Timestamp(df.index[i]),
            "signal_close": close_p,
            "limit_entry": limit_entry,
            "limit_price": limit_price,
            "stop_loss": stop_loss,
            "risk_mult": risk_mult,
            "rs_score": rs_vs_spy,
            "ml_confidence": ml_conf,
            "composite_score": comp_score,
        }

    def _try_fill_pending_signal(self, pending: Dict, bar_next: pd.Series) -> Optional[Dict]:
        """
        Execută semnalul generat pe bara `t` la deschiderea / minimul barei `t + 1` (Zero Look-Ahead),
        aplicând half-spread + slippage defavorabil și verificând R:R >= min_risk_reward.
        """
        open_next = float(bar_next["Open"])
        low_next = float(bar_next["Low"])

        if pending["limit_entry"]:
            lim_p = pending["limit_price"]
            if open_next <= lim_p:
                raw_in = open_next
            elif low_next <= lim_p:
                raw_in = lim_p
            else:
                return None  # Prețul a plecat în sus fără să atingă ordinul Limit
        else:
            max_gap = float(getattr(config, "MAX_ENTRY_GAP_PCT", 0.012))
            if open_next > pending["signal_close"] * (1.0 + max_gap):
                return None
            raw_in = open_next

        stop_loss = pending["stop_loss"]
        if raw_in <= stop_loss * 1.005:
            return None

        entry_fill = round(raw_in * (1.0 + self.one_way_cost_pct), 4)
        risk_per_share = entry_fill - stop_loss
        max_sl_pct = float(getattr(config, "MAX_STOP_LOSS_PCT", 0.075))
        if risk_per_share <= 0 or (risk_per_share / entry_fill) > max_sl_pct + 0.005:
            return None

        tp1_rr = float(getattr(config, "TP1_RISK_REWARD_RATIO", 1.5))
        tp2_rr = max(self.min_risk_reward, float(getattr(config, "TP2_RISK_REWARD_RATIO", 2.5)))
        tp1 = round(entry_fill + tp1_rr * risk_per_share, 2)
        tp2 = round(entry_fill + tp2_rr * risk_per_share, 2)

        if ((tp2 - entry_fill) / risk_per_share) < self.min_risk_reward - 1e-6:
            return None

        return {
            "entry_price": entry_fill,
            "stop_loss": stop_loss,
            "initial_risk": risk_per_share,
            "tp1": tp1,
            "tp2": tp2,
        }

    def run(
        self,
        ticker: str,
        df: pd.DataFrame,
        spy_df: Optional[pd.DataFrame] = None,
        qqq_df: Optional[pd.DataFrame] = None,
        vix_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Rulează simularea Single-Asset cu execuție strictă `t -> t + 1` (fără look-ahead)
        și deducerea tuturor costurilor reale de tranzacționare.
        """
        if len(df) < 220:
            return {"error": "Date insuficiente pentru backtest (necesar >= 220 zile închise)."}

        if spy_df is not None and "Close" in spy_df.columns and "RS_vs_SPY" not in df.columns:
            from indicators import add_relative_strength
            df = add_relative_strength(df.copy(), spy_df=spy_df)

        in_position = False
        pending_sig: Optional[Dict] = None
        pos: Dict = {}

        tp1_pct = float(getattr(config, "TP1_EXIT_PERCENT", 0.50))
        be_buf = float(getattr(config, "BREAKEVEN_BUFFER_PCT", 0.0020))
        ema_trail_buf = float(getattr(config, "EMA20_TRAIL_BUFFER_PCT", 0.015))
        max_single_pct = float(getattr(config, "MAX_SINGLE_POSITION_PCT", 0.33))
        min_alloc_pct = float(getattr(config, "MIN_POSITION_ALLOC_PCT", 0.12))

        for i in range(205, len(df)):
            bar = df.iloc[i]
            date = df.index[i]
            open_p = float(bar["Open"])
            high_p = float(bar["High"])
            low_p = float(bar["Low"])
            close_p = float(bar["Close"])
            ema_20 = float(bar.get("EMA_20", close_p))

            # 1. Execuție semnal din bara anterioară `i - 1` pe bara curentă `i`
            if (not in_position) and (pending_sig is not None):
                fill = self._try_fill_pending_signal(pending_sig, bar)
                if fill is not None:
                    dollar_risk = self.capital * self.risk_per_trade * pending_sig["risk_mult"]
                    shares = dollar_risk / fill["initial_risk"]
                    max_cap = self.capital * max_single_pct
                    if shares * fill["entry_price"] > max_cap:
                        shares = max_cap / fill["entry_price"]
                    shares = round(shares, 4)
                    cost = shares * fill["entry_price"] + self.commission_usd
                    if shares > 0 and cost <= self.capital and cost >= self.capital * min_alloc_pct:
                        self.capital -= self.commission_usd
                        in_position = True
                        pos = {
                            "setup_type": pending_sig["setup_type"],
                            "signal_date": pending_sig["signal_date"],
                            "entry_date": date,
                            "entry_price": fill["entry_price"],
                            "stop_loss": fill["stop_loss"],
                            "initial_risk": fill["initial_risk"],
                            "tp1": fill["tp1"],
                            "tp2": fill["tp2"],
                            "shares_tot": shares,
                            "shares_rem": shares,
                            "tp1_hit": False,
                            "peak_price": fill["entry_price"],
                        }
                pending_sig = None

            # 2. Gestionare poziție activă pe bara `i`
            if in_position:
                # 2a. Verificare conservatoare Stop-Loss PRIMUL (inclusiv Gap-Down la Open)
                if low_p <= pos["stop_loss"]:
                    raw_exit = min(pos["stop_loss"], open_p) if date > pos["entry_date"] else pos["stop_loss"]
                    exit_price = round(raw_exit * (1.0 - self.one_way_cost_pct), 4)
                    pnl = (exit_price - pos["entry_price"]) * pos["shares_rem"] - self.commission_usd
                    self.capital += pnl
                    r_mult = ((exit_price - pos["entry_price"]) / pos["initial_risk"]) * (pos["shares_rem"] / pos["shares_tot"])
                    exit_label = "STOP_LOSS" if not pos["tp1_hit"] else ("TRAILING_STOP" if exit_price > pos["entry_price"] * 1.005 else "BREAKEVEN_EXIT")

                    self.trades.append({
                        "ticker": ticker,
                        "signal_date": pos["signal_date"],
                        "entry_date": pos["entry_date"],
                        "exit_date": date,
                        "setup_type": pos["setup_type"],
                        "type": exit_label,
                        "entry_price": round(pos["entry_price"], 2),
                        "exit_price": round(exit_price, 2),
                        "shares": pos["shares_rem"],
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(((exit_price - pos["entry_price"]) / pos["entry_price"]) * 100.0, 2),
                        "r_multiple": round(r_mult, 2),
                    })
                    in_position = False
                    pos = {}
                    self.equity_curve.append(self.capital)
                    continue

                pos["peak_price"] = max(pos["peak_price"], high_p)

                # 2b. Verificare TP1 (50% Scale-Out + Mutare Stop la Breakeven)
                if (not pos["tp1_hit"]) and high_p >= pos["tp1"]:
                    pos["tp1_hit"] = True
                    exit_sh = round(pos["shares_tot"] * tp1_pct, 4)
                    exit_fill = round(pos["tp1"] * (1.0 - self.one_way_cost_pct * 0.5), 4)
                    pnl_tp1 = (exit_fill - pos["entry_price"]) * exit_sh - self.commission_usd
                    self.capital += pnl_tp1
                    pos["shares_rem"] = round(pos["shares_tot"] - exit_sh, 4)
                    pos["stop_loss"] = round(pos["entry_price"] * (1.0 + be_buf), 2)
                    r_mult = ((exit_fill - pos["entry_price"]) / pos["initial_risk"]) * tp1_pct

                    self.trades.append({
                        "ticker": ticker,
                        "signal_date": pos["signal_date"],
                        "entry_date": pos["entry_date"],
                        "exit_date": date,
                        "setup_type": pos["setup_type"],
                        "type": "TP1_HIT (50%)",
                        "entry_price": round(pos["entry_price"], 2),
                        "exit_price": round(exit_fill, 2),
                        "shares": exit_sh,
                        "pnl": round(pnl_tp1, 2),
                        "pnl_pct": round(((exit_fill - pos["entry_price"]) / pos["entry_price"]) * 100.0, 2),
                        "r_multiple": round(r_mult, 2),
                    })

                # 2c. Verificare TP2 (pe barele ulterioare lui TP1)
                elif pos["tp1_hit"] and high_p >= pos["tp2"]:
                    exit_fill = round(pos["tp2"] * (1.0 - self.one_way_cost_pct * 0.5), 4)
                    pnl_tp2 = (exit_fill - pos["entry_price"]) * pos["shares_rem"] - self.commission_usd
                    self.capital += pnl_tp2
                    r_mult = ((exit_fill - pos["entry_price"]) / pos["initial_risk"]) * (pos["shares_rem"] / pos["shares_tot"])

                    self.trades.append({
                        "ticker": ticker,
                        "signal_date": pos["signal_date"],
                        "entry_date": pos["entry_date"],
                        "exit_date": date,
                        "setup_type": pos["setup_type"],
                        "type": "TP2_HIT (100%)",
                        "entry_price": round(pos["entry_price"], 2),
                        "exit_price": round(exit_fill, 2),
                        "shares": pos["shares_rem"],
                        "pnl": round(pnl_tp2, 2),
                        "pnl_pct": round(((exit_fill - pos["entry_price"]) / pos["entry_price"]) * 100.0, 2),
                        "r_multiple": round(r_mult, 2),
                    })
                    in_position = False
                    pos = {}
                    self.equity_curve.append(self.capital)
                    continue

                # 2d. Trailing Stop pe EMA 20 zilnic EXCLUSIV după ce TP1 a fost atins
                if pos["tp1_hit"]:
                    trail_ema = round(ema_20 * (1.0 - ema_trail_buf), 2)
                    if trail_ema > pos["stop_loss"] and trail_ema < close_p * 0.995:
                        pos["stop_loss"] = trail_ema

                self.equity_curve.append(self.capital + ((close_p - pos["entry_price"]) * pos["shares_rem"]))

            else:
                # 3. Evaluare semnal la închiderea barei `i` pentru execuție la `i + 1`
                regime, r_mult, vix_val = self._classify_market_bar(date, spy_df, qqq_df, vix_df)
                pending_sig = self._check_entry_on_bar(ticker, df, i, regime, r_mult, vix_val=vix_val)
                self.equity_curve.append(self.capital)

        return self._compute_metrics(ticker)

    def _compute_metrics(self, ticker: str) -> Dict:
        """Calculează obligatoriu toate metricile instituționale de risc și performanță."""
        if not self.trades:
            return {
                "ticker": ticker,
                "total_trades": 0,
                "total_positions": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "position_win_rate": 0.0,
                "total_pnl": 0.0,
                "total_return_pct": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "expectancy_r": 0.0,
                "max_drawdown_pct": 0.0,
                "initial_capital": self.initial_capital,
                "final_capital": round(self.capital, 2),
            }

        df_trades = pd.DataFrame(self.trades)
        wins = df_trades[df_trades["pnl"] > 0]
        losses = df_trades[df_trades["pnl"] <= 0]

        total_trades = len(df_trades)
        win_rate = (len(wins) / total_trades) * 100.0 if total_trades > 0 else 0.0

        pos_groups = df_trades.groupby("entry_date").agg({"pnl": "sum", "r_multiple": "sum"})
        n_positions = len(pos_groups)
        pos_wins = int((pos_groups["pnl"] > 0).sum())
        position_win_rate = (pos_wins / n_positions * 100.0) if n_positions > 0 else win_rate
        expectancy_r = float(pos_groups["r_multiple"].mean()) if n_positions > 0 else 0.0

        gp = float(pos_groups[pos_groups["pnl"] > 0]["pnl"].sum())
        gl = float(abs(pos_groups[pos_groups["pnl"] <= 0]["pnl"].sum()))
        profit_factor = (gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0)

        final_eq = float(self.equity_curve[-1]) if self.equity_curve else self.capital
        total_pnl = final_eq - self.initial_capital
        total_return_pct = (total_pnl / self.initial_capital) * 100.0

        equity_series = pd.Series(self.equity_curve, dtype=float)
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak.replace(0, np.nan)
        max_drawdown = abs(float(drawdown.min())) * 100.0 if len(drawdown) > 0 else 0.0

        daily_returns = equity_series.pct_change().dropna()
        if len(daily_returns) > 10 and daily_returns.std() > 0:
            sharpe_ratio = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))
            downside = daily_returns[daily_returns < 0]
            downside_std = float(downside.std()) if len(downside) > 2 else float(daily_returns.std())
            sortino_ratio = float((daily_returns.mean() / downside_std) * np.sqrt(252)) if downside_std > 0 else sharpe_ratio
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0

        return {
            "ticker": ticker,
            "total_trades": total_trades,
            "total_positions": n_positions,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate, 1),
            "position_win_rate": round(position_win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "expectancy_r": round(expectancy_r, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "initial_capital": self.initial_capital,
            "final_capital": round(final_eq, 2),
        }

    def run_portfolio_backtest(
        self,
        data_cache: Dict[str, pd.DataFrame],
        spy_df: Optional[pd.DataFrame] = None,
        qqq_df: Optional[pd.DataFrame] = None,
        vix_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """Rulează simularea realistă de portofoliu la comun (Zero Look-Ahead + Costuri Reale)."""
        return run_portfolio_backtest(
            data_cache=data_cache,
            spy_df=spy_df,
            qqq_df=qqq_df,
            vix_df=vix_df,
            initial_capital=self.initial_capital,
            risk_per_trade=self.risk_per_trade,
            use_ml=self.use_ml,
        )


def run_portfolio_backtest(
    data_cache: Dict[str, pd.DataFrame],
    spy_df: Optional[pd.DataFrame] = None,
    qqq_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    initial_capital: float = config.DEFAULT_SIMULATED_CAPITAL,
    risk_per_trade: float = config.RISK_PER_TRADE_PERCENT,
    use_ml: bool = False,
) -> Dict:
    """
    Simulează întregul portofoliu pe un SINGUR CONT COMUN cu execuție strictă `t -> t + 1`:
      - Semnalele confirmate la închiderea zilei `t` se execută pe ziua `t + 1` (Limit/Open).
      - Respectă strict `MAX_OPEN_POSITIONS` (Portfolio Heat), `MAX_TOTAL_POSITIONS` și `MAX_POSITIONS_PER_SECTOR`.
      - Deduce comision + spread + slippage la fiecare execuție.
    """
    helper_bt = Backtester(initial_capital=initial_capital, risk_per_trade=risk_per_trade, use_ml=use_ml)
    from indicators import add_relative_strength

    aligned_dfs: Dict[str, pd.DataFrame] = {}
    all_dates_set = set()
    for ticker, df in data_cache.items():
        if len(df) < 220:
            continue
        df_c = add_relative_strength(df.copy(), spy_df=spy_df) if (spy_df is not None and "RS_vs_SPY" not in df.columns) else df.copy()
        aligned_dfs[ticker] = df_c
        all_dates_set.update(df_c.index[205:])

    sorted_dates = sorted(all_dates_set)
    if not sorted_dates:
        return {"error": "Date insuficiente pentru simularea portofoliului."}

    cash = float(initial_capital)
    open_positions: Dict[str, Dict] = {}
    pending_signals: List[Dict] = []
    closed_trades: List[Dict] = []
    equity_curve: List[float] = [float(initial_capital)]

    max_open_risk = int(config.MAX_OPEN_POSITIONS)
    max_total_pos = int(getattr(config, "MAX_TOTAL_POSITIONS", max_open_risk + 1))
    tp1_pct = float(getattr(config, "TP1_EXIT_PERCENT", 0.50))
    be_buf = float(getattr(config, "BREAKEVEN_BUFFER_PCT", 0.0020))
    ema_trail_buf = float(getattr(config, "EMA20_TRAIL_BUFFER_PCT", 0.015))
    max_single_pct = float(getattr(config, "MAX_SINGLE_POSITION_PCT", 0.33))
    min_alloc_pct = float(getattr(config, "MIN_POSITION_ALLOC_PCT", 0.12))

    for date in sorted_dates:
        # 1. Executăm la deschiderea/sesiunea `date` semnalele confirmate pe bara închisă anterioară
        if pending_signals:
            active_risk_count = sum(1 for p in open_positions.values() if not p["tp1_hit"])
            total_eq = cash + sum(p["shares_remaining"] * p["last_close"] for p in open_positions.values())

            for sig in pending_signals:
                if active_risk_count >= max_open_risk or len(open_positions) >= max_total_pos:
                    break
                tk = sig["ticker"]
                if tk in open_positions or date not in aligned_dfs[tk].index:
                    continue

                sec = config.get_ticker_sector(tk)
                max_sec = config.get_sector_max_positions(sec)
                sec_count = sum(1 for k in open_positions if config.get_ticker_sector(k) == sec)
                if sec_count >= max_sec:
                    continue

                bar_today = aligned_dfs[tk].loc[date]
                if isinstance(bar_today, pd.DataFrame):
                    bar_today = bar_today.iloc[0]

                fill = helper_bt._try_fill_pending_signal(sig, bar_today)
                if fill is None:
                    continue

                dollar_risk = total_eq * risk_per_trade * sig["risk_mult"]
                shares = dollar_risk / fill["initial_risk"]
                max_alloc = total_eq * max_single_pct
                if shares * fill["entry_price"] > max_alloc:
                    shares = max_alloc / fill["entry_price"]
                shares = round(shares, 4)

                cost = shares * fill["entry_price"] + helper_bt.commission_usd
                min_cost = max(float(getattr(config, "MIN_POSITION_COST_USD", 50.0)), total_eq * min_alloc_pct)
                if shares <= 0 or cost > cash or cost < min_cost:
                    continue

                cash -= cost
                open_positions[tk] = {
                    "ticker": tk,
                    "setup_type": sig["setup_type"],
                    "entry_date": date,
                    "entry_price": fill["entry_price"],
                    "last_close": fill["entry_price"],
                    "peak_price": fill["entry_price"],
                    "stop_loss": fill["stop_loss"],
                    "initial_risk": fill["initial_risk"],
                    "tp1": fill["tp1"],
                    "tp2": fill["tp2"],
                    "shares_total": shares,
                    "shares_remaining": shares,
                    "tp1_hit": False,
                }
                active_risk_count += 1

            pending_signals = []

        # 2. Gestionăm pozițiile deschise în ziua `date`
        for ticker in list(open_positions.keys()):
            pos = open_positions[ticker]
            df = aligned_dfs[ticker]
            if date not in df.index:
                continue

            bar = df.loc[date]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[0]

            open_p = float(bar["Open"])
            high_p = float(bar["High"])
            low_p = float(bar["Low"])
            close_p = float(bar["Close"])
            ema_20 = float(bar.get("EMA_20", close_p))
            pos["last_close"] = close_p

            # Stop-Loss PRIMUL (conservator)
            if low_p <= pos["stop_loss"]:
                raw_exit = min(pos["stop_loss"], open_p) if date > pos["entry_date"] else pos["stop_loss"]
                exit_price = round(raw_exit * (1.0 - helper_bt.one_way_cost_pct), 4)
                proceeds = exit_price * pos["shares_remaining"] - helper_bt.commission_usd
                pnl = proceeds - (pos["entry_price"] * pos["shares_remaining"])
                cash += proceeds
                r_mult = ((exit_price - pos["entry_price"]) / pos["initial_risk"]) * (pos["shares_remaining"] / pos["shares_total"])
                closed_trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "setup_type": pos["setup_type"],
                    "type": "STOP_LOSS" if not pos["tp1_hit"] else "TRAILING_STOP",
                    "pnl": round(pnl, 2),
                    "r_multiple": round(r_mult, 2),
                })
                del open_positions[ticker]
                continue

            pos["peak_price"] = max(pos["peak_price"], high_p)

            # TP1 (50% Scale-Out + Mutare Stop la Breakeven)
            if (not pos["tp1_hit"]) and high_p >= pos["tp1"]:
                pos["tp1_hit"] = True
                exit_sh = round(pos["shares_total"] * tp1_pct, 4)
                exit_fill = round(pos["tp1"] * (1.0 - helper_bt.one_way_cost_pct * 0.5), 4)
                proceeds = exit_fill * exit_sh - helper_bt.commission_usd
                pnl_tp1 = proceeds - (pos["entry_price"] * exit_sh)
                cash += proceeds
                pos["shares_remaining"] = round(pos["shares_total"] - exit_sh, 4)
                pos["stop_loss"] = round(pos["entry_price"] * (1.0 + be_buf), 2)
                r_mult = ((exit_fill - pos["entry_price"]) / pos["initial_risk"]) * tp1_pct
                closed_trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "setup_type": pos["setup_type"],
                    "type": "TP1_HIT (50%)",
                    "pnl": round(pnl_tp1, 2),
                    "r_multiple": round(r_mult, 2),
                })

            elif pos["tp1_hit"] and high_p >= pos["tp2"]:
                exit_fill = round(pos["tp2"] * (1.0 - helper_bt.one_way_cost_pct * 0.5), 4)
                proceeds = exit_fill * pos["shares_remaining"] - helper_bt.commission_usd
                pnl_tp2 = proceeds - (pos["entry_price"] * pos["shares_remaining"])
                cash += proceeds
                r_mult = ((exit_fill - pos["entry_price"]) / pos["initial_risk"]) * (pos["shares_remaining"] / pos["shares_total"])
                closed_trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "setup_type": pos["setup_type"],
                    "type": "TP2_HIT (100%)",
                    "pnl": round(pnl_tp2, 2),
                    "r_multiple": round(r_mult, 2),
                })
                del open_positions[ticker]
                continue

            if pos["tp1_hit"]:
                trail_ema = round(ema_20 * (1.0 - ema_trail_buf), 2)
                if trail_ema > pos["stop_loss"] and trail_ema < close_p * 0.995:
                    pos["stop_loss"] = trail_ema

        # 3. Generăm semnale la închiderea zilei `date` pentru sesiunea următoare
        regime, r_mult, vix_val = helper_bt._classify_market_bar(date, spy_df, qqq_df, vix_df)
        active_risk_count = sum(1 for p in open_positions.values() if not p["tp1_hit"])
        max_allowed_regime = max_open_risk if regime == "POWER_TREND" else max(2, max_open_risk - 1)

        if regime != "BEARISH" and r_mult > 0 and active_risk_count < max_allowed_regime and len(open_positions) < max_total_pos:
            day_candidates = []
            for ticker, df in aligned_dfs.items():
                if ticker in open_positions or date not in df.index:
                    continue
                if config.is_sector_blocked(config.get_ticker_sector(ticker)):
                    continue
                idx_loc = df.index.get_loc(date)
                if isinstance(idx_loc, (slice, np.ndarray)) or idx_loc < 205:
                    continue
                sig = helper_bt._check_entry_on_bar(ticker, df, int(idx_loc), regime, r_mult, vix_val=vix_val)
                if sig is not None:
                    day_candidates.append(sig)

            day_candidates.sort(key=lambda x: x["composite_score"], reverse=True)
            pending_signals = day_candidates[:max_allowed_regime]

        open_market_val = sum(p["shares_remaining"] * p["last_close"] for p in open_positions.values())
        equity_curve.append(cash + open_market_val)

    final_equity = equity_curve[-1]
    total_pnl = final_equity - initial_capital
    total_return_pct = (total_pnl / initial_capital) * 100.0

    df_tr = pd.DataFrame(closed_trades)
    if not df_tr.empty:
        wins_legs = df_tr[df_tr["pnl"] > 0]
        win_rate = (len(wins_legs) / len(df_tr)) * 100.0
        pos_grp = df_tr.groupby(["ticker", "entry_date"]).agg({"pnl": "sum", "r_multiple": "sum"})
        n_pos = len(pos_grp)
        pos_wr = float((pos_grp["pnl"] > 0).mean() * 100.0) if n_pos > 0 else win_rate
        gp = float(pos_grp[pos_grp["pnl"] > 0]["pnl"].sum())
        gl = float(abs(pos_grp[pos_grp["pnl"] <= 0]["pnl"].sum()))
        pf = (gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0)
        exp_r = float(pos_grp["r_multiple"].mean()) if n_pos > 0 else 0.0
    else:
        n_pos = 0
        win_rate, pos_wr, pf, exp_r = 0.0, 0.0, 0.0, 0.0

    eq_s = pd.Series(equity_curve, dtype=float)
    peak = eq_s.cummax()
    dd = (eq_s - peak) / peak.replace(0, np.nan)
    max_dd = abs(float(dd.min())) * 100.0 if len(dd) > 0 else 0.0

    rets = eq_s.pct_change().dropna()
    sharpe = float((rets.mean() / rets.std()) * np.sqrt(252)) if len(rets) > 10 and rets.std() > 0 else 0.0
    downside = rets[rets < 0]
    sortino = float((rets.mean() / downside.std()) * np.sqrt(252)) if len(downside) > 2 and downside.std() > 0 else sharpe

    return {
        "initial_capital": initial_capital,
        "final_capital": round(final_equity, 2),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "total_positions": n_pos,
        "total_trades": len(closed_trades),
        "total_executions": len(closed_trades),
        "win_rate": round(win_rate, 1),
        "position_win_rate": round(pos_wr, 1),
        "profit_factor": round(pf, 2),
        "expectancy_r": round(exp_r, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown_pct": round(max_dd, 2),
    }
