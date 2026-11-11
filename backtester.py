"""
Motor de Backtesting Institutional (Single-Asset & Portfolio-Level):
Simuleaza strategia pas cu pas pe date istorice, 100% aliniat cu strategy.py si position_manager.py:
  - Filtru Regim Piata (SPY > SMA 200) + Confluenta Saptamanala (1W Stage 2) + Relative Strength (RS_vs_SPY)
  - Slippage realist la executie (config.BACKTEST_SLIPPAGE_PCT)
  - Stop-Loss structural (cu modelare realista de Gap-Down la Open)
  - TP1 la 1:1.5 / 1:2 R:R (inchidere 50% + mutare Stop la Breakeven)
  - Trailing Stop Dinamic pe restul de 50% ("Runner") pana la TP2 (1:3 R:R)
  - Metrici institutionale: Sharpe Ratio, Sortino Ratio, Expectancy (R), Profit Factor, Max Drawdown
  - Simulator de Portofoliu Comun (PortfolioBacktester): Respecta MAX_OPEN_POSITIONS si aloca capitalul
    prioritar catre semnalele cu cel mai mare scor compozit (RS + ML).
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import config


class Backtester:
    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.015,
        rsi_pullback_min: float = config.RSI_PULLBACK_MIN,
        rsi_pullback_max: float = config.RSI_PULLBACK_MAX,
        min_risk_reward: float = config.MIN_RISK_REWARD_RATIO,
        ema_period: int = 20,
        volume_surge_mult: float = config.VOLUME_SURGE_MULTIPLIER,
        use_ml: bool = False,
        use_weekly_filter: bool = getattr(config, "USE_WEEKLY_FILTER", True),
        use_rs_filter: bool = getattr(config, "USE_RELATIVE_STRENGTH", True),
        slippage_pct: float = getattr(config, "BACKTEST_SLIPPAGE_PCT", 0.0005)
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.rsi_pullback_min = rsi_pullback_min
        self.rsi_pullback_max = rsi_pullback_max
        self.min_risk_reward = min_risk_reward
        self.ema_period = ema_period
        self.volume_surge_mult = volume_surge_mult
        self.use_ml = use_ml
        self.use_weekly_filter = use_weekly_filter
        self.use_rs_filter = use_rs_filter
        self.slippage_pct = slippage_pct
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [initial_capital]

    def _check_entry_on_bar(self, df: pd.DataFrame, i: int, market_ok: bool) -> Optional[Dict]:
        """
        Verifica daca la bara `i` sunt indeplinite conditiile de intrare
        (100% aliniat cu check_pullback_setup si check_breakout_setup din strategy.py).
        Returneaza un dict cu detaliile semnalului sau None.
        """
        if not market_ok:
            return None

        current_bar = df.iloc[i]
        open_p = float(current_bar["Open"])
        high = float(current_bar["High"])
        low = float(current_bar["Low"])
        close = float(current_bar["Close"])

        ema_col = f"EMA_{self.ema_period}"
        ema = float(current_bar[ema_col]) if ema_col in current_bar else float(current_bar.get("EMA_20", close))
        sma_50 = float(current_bar["SMA_50"])
        sma50_slope = float(current_bar.get("SMA_50_Slope", 0.0))
        rsi = float(current_bar["RSI_14"])
        atr = float(current_bar["ATR_14"]) if not pd.isna(current_bar.get("ATR_14", np.nan)) else close * 0.02
        volume = float(current_bar["Volume"])
        vol_sma = float(current_bar.get("Volume_SMA_20", 0))
        rs_vs_spy = float(current_bar.get("RS_vs_SPY", 0.0))

        # 1. Trend zilnic peste SMA 50 (si SMA 50 nu se prabuseste)
        if close <= sma_50 or sma50_slope < -0.50:
            return None

        # 2. Filtru Multi-Timeframe (1W Confluence)
        if self.use_weekly_filter:
            w_ema = float(current_bar.get("Weekly_EMA_10", 0.0))
            w_sma = float(current_bar.get("Weekly_SMA_30", 0.0))
            w_rsi = float(current_bar.get("Weekly_RSI_14", 50.0))
            if not np.isnan(w_ema) and not np.isnan(w_sma) and w_ema > 0 and w_sma > 0:
                if close < w_sma * 0.99 or close < w_ema * 0.98:
                    return None
                if w_ema < w_sma * 0.97:
                    return None
                if not np.isnan(w_rsi):
                    if w_rsi > getattr(config, "WEEKLY_RSI_MAX", 72.0) or w_rsi < getattr(config, "WEEKLY_RSI_MIN", 40.0):
                        return None

        # --- SCENARIUL 1: PULLBACK EMA ---
        distance_to_ema = abs(close - ema) / ema if ema > 0 else 1.0
        touched_ema = (low <= ema * 1.01) and (close >= ema * 0.99)
        valid_ema_touch = (distance_to_ema <= 0.025 and close >= ema * 0.99) or touched_ema
        rsi_pullback = self.rsi_pullback_min <= rsi <= self.rsi_pullback_max

        candle_range = high - low
        is_green = close > open_p

        rs_ok_pullback = True
        if self.use_rs_filter and not np.isnan(rs_vs_spy):
            rs_ok_pullback = rs_vs_spy >= getattr(config, "MIN_RS_VS_SPY_PULLBACK", -5.0)

        is_pullback = valid_ema_touch and rsi_pullback and is_green and rs_ok_pullback

        # --- SCENARIUL 2: VOLUME BREAKOUT ---
        prior_20_high = float(df["High"].iloc[max(0, i - 20):i].max())
        vol_surge = (vol_sma > 0) and (volume >= self.volume_surge_mult * vol_sma)
        rsi_breakout = 52.0 <= rsi <= 76.0
        strong_close = (candle_range > 0) and (((close - low) / candle_range) >= 0.60)
        max_ext = getattr(config, "MAX_BREAKOUT_EXTENSION_EMA20", 0.12)
        not_overextended = (ema <= 0) or (((close - ema) / ema) <= max_ext)

        rs_ok_breakout = True
        if self.use_rs_filter and not np.isnan(rs_vs_spy):
            rs_ok_breakout = rs_vs_spy >= getattr(config, "MIN_RS_VS_SPY_BREAKOUT", 0.0)

        is_breakout = (
            (close > prior_20_high)
            and vol_surge
            and rsi_breakout
            and is_green
            and strong_close
            and not_overextended
            and rs_ok_breakout
        )

        if not (is_pullback or is_breakout):
            return None

        # Filtru optional de Machine Learning (foloseste Out-of-Sample cache daca exista pentru a preveni Data Leakage)
        ml_conf = 0.55
        if self.use_ml:
            try:
                from ml_predictor import get_predictor
                pred = get_predictor()
                if pred.is_ready:
                    bar_date_str = pd.Timestamp(df.index[i]).strftime("%Y-%m-%d")
                    prob_win, is_conf = pred.predict_for_backtest(current_bar, vol_sma, bar_date_str)
                    ml_conf = prob_win
                    if not is_conf:
                        return None
            except Exception:
                pass

        setup_type = "VOLUME_BREAKOUT" if is_breakout and not is_pullback else "PULLBACK_EMA20"

        # Pret de intrare ajustat cu slippage realist la cumparare
        entry_price = round(close * (1.0 + self.slippage_pct), 2)

        # Calcul Stop-Loss 100% aliniat cu strategy.py (calibrat intre 2.5% si 5.5%)
        if setup_type == "VOLUME_BREAKOUT":
            recent_support = float(df["Low"].iloc[max(0, i - 2):i + 1].min())
            stop_loss = round(max(prior_20_high * 0.98, recent_support), 2)
        else:
            recent_low = float(df["Low"].iloc[max(0, i - 3):i + 1].min())
            stop_loss = round(min(recent_low, low - (0.35 * atr)), 2)

        stop_dist_pct = (entry_price - stop_loss) / entry_price
        if stop_loss >= entry_price or stop_dist_pct < 0.025:
            stop_loss = round(entry_price * 0.972, 2)
        elif stop_dist_pct > 0.055:
            stop_loss = round(entry_price * 0.945, 2)

        risk_per_share = round(entry_price - stop_loss, 2)
        if risk_per_share <= 0:
            return None

        tp2_rr = getattr(config, "TP2_RISK_REWARD_RATIO", max(3.0, self.min_risk_reward + 1.0))
        tp1 = round(entry_price + (self.min_risk_reward * risk_per_share), 2)
        tp2 = round(entry_price + (tp2_rr * risk_per_share), 2)

        rs_clamped = max(-10.0, min(40.0, rs_vs_spy if not np.isnan(rs_vs_spy) else 0.0))
        comp_score = (ml_conf * 45.0) + (((rs_clamped + 10.0) / 50.0) * 45.0)

        return {
            "setup_type": setup_type,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "initial_risk": risk_per_share,
            "tp1": tp1,
            "tp2": tp2,
            "rs_score": rs_vs_spy if not np.isnan(rs_vs_spy) else 0.0,
            "ml_confidence": ml_conf,
            "composite_score": round(comp_score, 2)
        }

    def run(self, ticker: str, df: pd.DataFrame, spy_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Ruleaza simularea pe datele istorice ale unui ticker.
        Include Trailing Stop dinamic dupa TP1 si calculul metricilor institutionale.
        """
        if len(df) < 220:
            return {"error": "Date insuficiente pentru backtest (necesar > 220 zile)."}

        # Daca RS_vs_SPY nu a fost calculat fata de spy_df, il actualizam
        if spy_df is not None and "Close" in spy_df.columns:
            from indicators import add_relative_strength
            df = add_relative_strength(df.copy(), spy_df=spy_df)

        in_position = False
        entry_price = 0.0
        stop_loss = 0.0
        initial_risk = 0.0
        tp1 = 0.0
        tp2 = 0.0
        peak_price = 0.0
        shares_total = 0.0
        shares_remaining = 0.0
        tp1_hit = False
        entry_date = None
        setup_type = "PULLBACK_EMA20"

        spy_bullish_dates = set()
        if spy_df is not None and "SMA_200" in spy_df.columns:
            valid_spy = spy_df[spy_df["Close"] > spy_df["SMA_200"]]
            spy_bullish_dates = set(valid_spy.index)

        for i in range(200, len(df)):
            current_bar = df.iloc[i]
            date = df.index[i]

            open_p = float(current_bar["Open"])
            high = float(current_bar["High"])
            low = float(current_bar["Low"])
            close = float(current_bar["Close"])
            ema_20 = float(current_bar.get("EMA_20", close))
            low_3d = float(current_bar.get("Low_3D", low))

            if in_position:
                peak_price = max(peak_price, high)

                # 1. Verificare Stop-Loss / Trailing Stop (cu Gap-Down realism + slippage)
                if low <= stop_loss:
                    raw_exit = min(stop_loss, open_p)
                    exit_price = round(raw_exit * (1.0 - self.slippage_pct), 2)
                    pnl = (exit_price - entry_price) * shares_remaining
                    self.capital += pnl
                    r_mult = ((exit_price - entry_price) / initial_risk) * (shares_remaining / shares_total) if initial_risk > 0 else 0.0
                    exit_label = "STOP_LOSS" if not tp1_hit else ("TRAILING_STOP" if exit_price > entry_price * 1.005 else "BREAKEVEN_EXIT")

                    self.trades.append({
                        "ticker": ticker,
                        "entry_date": entry_date,
                        "exit_date": date,
                        "setup_type": setup_type,
                        "type": exit_label,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "shares": shares_remaining,
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(((exit_price - entry_price) / entry_price) * 100, 2),
                        "r_multiple": round(r_mult, 2)
                    })
                    in_position = False
                    shares_remaining = 0.0
                    self.equity_curve.append(self.capital)
                    continue

                # 2. Verificare TP1 (50% Scale-Out + Mutare Stop la Breakeven)
                if not tp1_hit and high >= tp1:
                    tp1_hit = True
                    exit_shares = round(shares_total * config.TP1_EXIT_PERCENT, 2)
                    pnl_tp1 = (tp1 - entry_price) * exit_shares
                    self.capital += pnl_tp1
                    shares_remaining = round(shares_total - exit_shares, 2)
                    r_mult = ((tp1 - entry_price) / initial_risk) * config.TP1_EXIT_PERCENT if initial_risk > 0 else 0.0

                    # Mutam Stop-Loss la Breakeven (+0.15% buffer)
                    stop_loss = round(entry_price * 1.0015, 2)

                    self.trades.append({
                        "ticker": ticker,
                        "entry_date": entry_date,
                        "exit_date": date,
                        "setup_type": setup_type,
                        "type": "TP1_HIT (50%)",
                        "entry_price": entry_price,
                        "exit_price": tp1,
                        "shares": exit_shares,
                        "pnl": round(pnl_tp1, 2),
                        "pnl_pct": round(((tp1 - entry_price) / entry_price) * 100, 2),
                        "r_multiple": round(r_mult, 2)
                    })

                # 3. Verificare TP2 (Inchidere Runner 50%)
                if tp1_hit and high >= tp2:
                    pnl_tp2 = (tp2 - entry_price) * shares_remaining
                    self.capital += pnl_tp2
                    r_mult = ((tp2 - entry_price) / initial_risk) * (shares_remaining / shares_total) if initial_risk > 0 else 0.0

                    self.trades.append({
                        "ticker": ticker,
                        "entry_date": entry_date,
                        "exit_date": date,
                        "setup_type": setup_type,
                        "type": "TP2_HIT (100%)",
                        "entry_price": entry_price,
                        "exit_price": tp2,
                        "shares": shares_remaining,
                        "pnl": round(pnl_tp2, 2),
                        "pnl_pct": round(((tp2 - entry_price) / entry_price) * 100, 2),
                        "r_multiple": round(r_mult, 2)
                    })
                    in_position = False
                    shares_remaining = 0.0
                    self.equity_curve.append(self.capital)
                    continue

                # 3.5. Protectie Auto-Breakeven la +3.0% sau +1.0R (100% aliniat cu position_manager.py!)
                breakeven_trigger = min(entry_price * 1.03, entry_price + initial_risk)
                if not tp1_hit and high >= breakeven_trigger:
                    be_stop = round(entry_price * 1.002, 2)
                    if be_stop > stop_loss:
                        stop_loss = be_stop

                # 4. Actualizare Trailing Stop Dinamic pentru Runner (dupa ce TP1 a fost atins)
                if tp1_hit:
                    chandelier_stop = round(peak_price * 0.962, 2)
                    structural_stop = round(max(ema_20 * 0.988, low_3d * 0.995), 2) if not np.isnan(low_3d) else chandelier_stop
                    candidate_trail = max(chandelier_stop, structural_stop)
                    if candidate_trail > stop_loss and candidate_trail < close:
                        stop_loss = candidate_trail

                self.equity_curve.append(self.capital + ((close - entry_price) * shares_remaining))

            else:
                market_ok = (len(spy_bullish_dates) == 0) or (date in spy_bullish_dates)
                sig = self._check_entry_on_bar(df, i, market_ok)
                if sig is None:
                    self.equity_curve.append(self.capital)
                    continue

                entry_price = sig["entry_price"]
                stop_loss = sig["stop_loss"]
                initial_risk = sig["initial_risk"]
                tp1 = sig["tp1"]
                tp2 = sig["tp2"]
                setup_type = sig["setup_type"]
                peak_price = entry_price

                dollar_risk = self.capital * self.risk_per_trade
                shares_total = round(dollar_risk / initial_risk, 2)

                max_pos_pct = max(1.0 / config.MAX_OPEN_POSITIONS, getattr(config, "MAX_SINGLE_POSITION_PCT", 0.35))
                max_cap = self.capital * max_pos_pct
                if shares_total * entry_price > max_cap:
                    shares_total = round(max_cap / entry_price, 2)

                if shares_total <= 0:
                    self.equity_curve.append(self.capital)
                    continue

                in_position = True
                tp1_hit = False
                shares_remaining = shares_total
                entry_date = date
                self.equity_curve.append(self.capital)

        return self._compute_metrics(ticker)

    def _compute_metrics(self, ticker: str) -> Dict:
        """Calculeaza metricile institutionale de performanta (inclusiv Sharpe, Sortino, Expectancy R)."""
        if not self.trades:
            return {
                "ticker": ticker,
                "total_trades": 0,
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
                "final_capital": self.initial_capital
            }

        df_trades = pd.DataFrame(self.trades)
        wins = df_trades[df_trades["pnl"] > 0]
        losses = df_trades[df_trades["pnl"] < 0]

        total_trades = len(df_trades)
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0.0

        # Calculam si rata de castig si Expectancy R la nivel de POZITIE (grupata dupa entry_date)
        pos_groups = df_trades.groupby("entry_date").agg({
            "pnl": "sum",
            "r_multiple": "sum"
        })
        n_positions = len(pos_groups)
        pos_wins = int((pos_groups["pnl"] > 0).sum())
        position_win_rate = (pos_wins / n_positions * 100) if n_positions > 0 else win_rate
        expectancy_r = float(pos_groups["r_multiple"].mean()) if n_positions > 0 and "r_multiple" in pos_groups else 0.0

        gross_profit = float(wins["pnl"].sum())
        gross_loss = float(abs(losses["pnl"].sum()))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

        total_pnl = self.capital - self.initial_capital
        total_return_pct = (total_pnl / self.initial_capital) * 100

        # Max Drawdown, Sharpe & Sortino din curba de capital zilnica
        equity_series = pd.Series(self.equity_curve, dtype=float)
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak.replace(0, np.nan)
        max_drawdown = abs(float(drawdown.min())) * 100 if len(drawdown) > 0 else 0.0

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
            "final_capital": round(self.capital, 2)
        }

    def run_portfolio_backtest(
        self,
        data_cache: Dict[str, pd.DataFrame],
        spy_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """Metoda pe instanta Backtester care apeleaza simularea realista de portofoliu."""
        return run_portfolio_backtest(
            data_cache=data_cache,
            spy_df=spy_df,
            initial_capital=self.initial_capital,
            use_ml=self.use_ml
        )


def run_portfolio_backtest(
    data_cache: Dict[str, pd.DataFrame],
    spy_df: Optional[pd.DataFrame] = None,
    initial_capital: float = config.DEFAULT_SIMULATED_CAPITAL,
    use_ml: bool = False
) -> Dict:
    """
    Simuleaza intregul portofoliu de actiuni pe un SINGUR CONT COMUN ($10,000):
      - Respecta strict limita MAX_OPEN_POSITIONS cu risc activ (Free-Roll permis dupa Breakeven).
      - Prioritizeaza semnalele din aceeasi zi descrescator dupa Scorul Compozit (RS + ML).
      - Ofera imaginea 100% realista a evolutiei contului (CAGR, Sharpe, Max Drawdown pe cont).
    """
    helper_bt = Backtester(initial_capital=initial_capital, use_ml=use_ml)

    # Pre-aliniem RS_vs_SPY si indexul de date
    aligned_dfs: Dict[str, pd.DataFrame] = {}
    all_dates_set = set()
    from indicators import add_relative_strength

    for ticker, df in data_cache.items():
        if len(df) < 220:
            continue
        df_c = add_relative_strength(df.copy(), spy_df=spy_df) if spy_df is not None else df.copy()
        aligned_dfs[ticker] = df_c
        all_dates_set.update(df_c.index[200:])

    sorted_dates = sorted(all_dates_set)
    if not sorted_dates:
        return {"error": "Date insuficiente pentru simularea portofoliului."}

    spy_bullish_dates = set()
    if spy_df is not None and "SMA_200" in spy_df.columns:
        spy_bullish_dates = set(spy_df[spy_df["Close"] > spy_df["SMA_200"]].index)

    cash = initial_capital
    open_positions: Dict[str, Dict] = {}
    closed_trades: List[Dict] = []
    equity_curve: List[float] = [initial_capital]

    for date in sorted_dates:
        # 1. Gestionam iesirile pentru pozitiile active in ziua `date`
        for ticker in list(open_positions.keys()):
            pos = open_positions[ticker]
            df = aligned_dfs[ticker]
            if date not in df.index:
                continue

            bar = df.loc[date]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[0]

            open_p = float(bar["Open"])
            high = float(bar["High"])
            low = float(bar["Low"])
            close = float(bar["Close"])
            ema_20 = float(bar.get("EMA_20", close))
            low_3d = float(bar.get("Low_3D", low))

            pos["peak_price"] = max(pos["peak_price"], high)
            pos["last_close"] = close

            # Stop-Loss / Trailing Stop
            if low <= pos["stop_loss"]:
                raw_exit = min(pos["stop_loss"], open_p)
                exit_price = round(raw_exit * (1.0 - helper_bt.slippage_pct), 2)
                proceeds = exit_price * pos["shares_remaining"]
                pnl = (exit_price - pos["entry_price"]) * pos["shares_remaining"]
                cash += proceeds
                r_mult = ((exit_price - pos["entry_price"]) / pos["initial_risk"]) * (pos["shares_remaining"] / pos["shares_total"])
                closed_trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "type": "STOP_LOSS" if not pos["tp1_hit"] else "TRAILING_STOP",
                    "pnl": round(pnl, 2),
                    "r_multiple": round(r_mult, 2)
                })
                del open_positions[ticker]
                continue

            # TP1 (50% Scale-Out + Free-Roll Breakeven)
            if not pos["tp1_hit"] and high >= pos["tp1"]:
                pos["tp1_hit"] = True
                exit_shares = round(pos["shares_total"] * config.TP1_EXIT_PERCENT, 2)
                proceeds = pos["tp1"] * exit_shares
                pnl_tp1 = (pos["tp1"] - pos["entry_price"]) * exit_shares
                cash += proceeds
                pos["shares_remaining"] = round(pos["shares_total"] - exit_shares, 2)
                pos["stop_loss"] = round(pos["entry_price"] * 1.0015, 2)
                r_mult = ((pos["tp1"] - pos["entry_price"]) / pos["initial_risk"]) * config.TP1_EXIT_PERCENT
                closed_trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "type": "TP1_HIT (50%)",
                    "pnl": round(pnl_tp1, 2),
                    "r_multiple": round(r_mult, 2)
                })

            # TP2 (Runner 50%)
            if pos["tp1_hit"] and high >= pos["tp2"]:
                proceeds = pos["tp2"] * pos["shares_remaining"]
                pnl_tp2 = (pos["tp2"] - pos["entry_price"]) * pos["shares_remaining"]
                cash += proceeds
                r_mult = ((pos["tp2"] - pos["entry_price"]) / pos["initial_risk"]) * (pos["shares_remaining"] / pos["shares_total"])
                closed_trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "type": "TP2_HIT (100%)",
                    "pnl": round(pnl_tp2, 2),
                    "r_multiple": round(r_mult, 2)
                })
                del open_positions[ticker]
                continue

            # Protectie Auto-Breakeven la +3.0% sau +1.0R (100% aliniat cu position_manager.py)
            be_trigger = min(pos["entry_price"] * 1.03, pos["entry_price"] + pos["initial_risk"])
            if not pos["tp1_hit"] and high >= be_trigger:
                be_stop = round(pos["entry_price"] * 1.002, 2)
                if be_stop > pos["stop_loss"]:
                    pos["stop_loss"] = be_stop

            # Actualizare Trailing Stop pe Runner
            if pos["tp1_hit"]:
                chandelier = round(pos["peak_price"] * 0.962, 2)
                structural = round(max(ema_20 * 0.988, low_3d * 0.995), 2) if not np.isnan(low_3d) else chandelier
                cand_stop = max(chandelier, structural)
                if cand_stop > pos["stop_loss"] and cand_stop < close:
                    pos["stop_loss"] = cand_stop

        # Calculam echitatea curenta a portofoliului
        open_market_val = sum(p["shares_remaining"] * p["last_close"] for p in open_positions.values())
        total_equity = cash + open_market_val

        # Numar de pozitii cu RISC ACTIV (cele care NU au atins inca Breakeven)
        active_risk_count = sum(1 for p in open_positions.values() if not p["tp1_hit"])
        market_ok = (len(spy_bullish_dates) == 0) or (date in spy_bullish_dates)

        # 2. Cautam intrari noi daca mai avem loc in bugetul de risc (Portfolio Heat)
        if market_ok and active_risk_count < config.MAX_OPEN_POSITIONS and len(open_positions) < config.MAX_OPEN_POSITIONS + 2:
            day_candidates = []
            for ticker, df in aligned_dfs.items():
                if ticker in open_positions or date not in df.index:
                    continue
                idx_loc = df.index.get_loc(date)
                if isinstance(idx_loc, slice) or isinstance(idx_loc, np.ndarray):
                    continue
                if idx_loc < 200:
                    continue
                sig = helper_bt._check_entry_on_bar(df, int(idx_loc), market_ok=True)
                if sig is not None:
                    sig["ticker"] = ticker
                    sec_bonus = config.get_sector_priority_bonus(config.get_ticker_sector(ticker)) if hasattr(config, "get_sector_priority_bonus") else 0.0
                    sig["composite_score"] = round(min(100.0, sig["composite_score"] + sec_bonus), 2)
                    day_candidates.append(sig)

            # Sortam candidatii zilei dupa Scorul Compozit (Liderii cu cel mai mare RS + ML primii!)
            day_candidates.sort(key=lambda x: x["composite_score"], reverse=True)

            for sig in day_candidates:
                if active_risk_count >= config.MAX_OPEN_POSITIONS:
                    break

                dollar_risk = total_equity * config.RISK_PER_TRADE_PERCENT
                shares = round(dollar_risk / sig["initial_risk"], 2)
                max_pos_pct = max(1.0 / config.MAX_OPEN_POSITIONS, getattr(config, "MAX_SINGLE_POSITION_PCT", 0.35))
                max_alloc = min(cash * 0.98, total_equity * max_pos_pct)
                if shares * sig["entry_price"] > max_alloc:
                    shares = round(max_alloc / sig["entry_price"], 2)

                cost = shares * sig["entry_price"]
                min_cost = min(50.0, total_equity * 0.05)
                if shares <= 0 or cost > cash or cost < min_cost:
                    continue

                cash -= cost
                open_positions[sig["ticker"]] = {
                    "ticker": sig["ticker"],
                    "entry_date": date,
                    "entry_price": sig["entry_price"],
                    "last_close": sig["entry_price"],
                    "peak_price": sig["entry_price"],
                    "stop_loss": sig["stop_loss"],
                    "initial_risk": sig["initial_risk"],
                    "tp1": sig["tp1"],
                    "tp2": sig["tp2"],
                    "shares_total": shares,
                    "shares_remaining": shares,
                    "tp1_hit": False
                }
                active_risk_count += 1

        open_market_val = sum(p["shares_remaining"] * p["last_close"] for p in open_positions.values())
        equity_curve.append(cash + open_market_val)

    final_equity = equity_curve[-1]
    total_pnl = final_equity - initial_capital
    total_return_pct = (total_pnl / initial_capital) * 100

    df_tr = pd.DataFrame(closed_trades)
    if not df_tr.empty:
        wins = df_tr[df_tr["pnl"] > 0]
        losses = df_tr[df_tr["pnl"] < 0]
        win_rate = (len(wins) / len(df_tr)) * 100
        gp = float(wins["pnl"].sum())
        gl = float(abs(losses["pnl"].sum()))
        pf = (gp / gl) if gl > 0 else 999.0
        pos_grp = df_tr.groupby(["ticker", "entry_date"]).agg({"pnl": "sum", "r_multiple": "sum"})
        n_pos = len(pos_grp)
        pos_wr = float((pos_grp["pnl"] > 0).mean() * 100) if n_pos > 0 else win_rate
        exp_r = float(pos_grp["r_multiple"].mean()) if n_pos > 0 else 0.0
    else:
        n_pos = 0
        win_rate, pos_wr, pf, exp_r = 0.0, 0.0, 0.0, 0.0

    eq_s = pd.Series(equity_curve, dtype=float)
    peak = eq_s.cummax()
    dd = (eq_s - peak) / peak.replace(0, np.nan)
    max_dd = abs(float(dd.min())) * 100 if len(dd) > 0 else 0.0

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
        "max_drawdown_pct": round(max_dd, 2)
    }
