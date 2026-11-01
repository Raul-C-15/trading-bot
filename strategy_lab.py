"""
Laborator de Strategii & Grafic Interactiv Stil TradingView (100% Gratuit, Local).
Permite:
  1. Compararea simultană a 5 strategii consacrate de Swing Trading pe orice acțiune sau pe tot portofoliul:
     - Strategia 1 (SWING_PRO): Pullback EMA 20 + Volume Breakout + RS vs SPY + 1W Stage 2 (Strategia curentă a botului)
     - Strategia 2 (RSI2_CONNORS): Larry Connors RSI(2) Mean Reversion în Uptrend (Win-Rate ridicat)
     - Strategia 3 (MINERVINI_VCP): Mark Minervini Volatility Contraction Pattern (VCP) + Pocket Pivot
     - Strategia 4 (DONCHIAN_TURTLE): Turtle Trading 20-Day Breakout cu Trailing pe 10-Day Low
     - Strategia 5 (MACD_MOMENTUM): Trend Following cu EMA 10/20 + Histogramă MACD + Filtru RSI
  2. Generarea unui Dashboard Web Interactiv (HTML cu librăria oficială TradingView Lightweight Charts):
     - Grafic japonez cu lumânări, EMA 20, SMA 50, SMA 200
     - Săgeți vizuale de BUY și EXIT (TP1 / TP2 / SL / Trail) direct pe grafic
     - Curba de capital (Equity Curve) comparativă pentru toate cele 5 strategii
     - Lista completă a tranzacțiilor (identică cu tab-ul "Strategy Tester" din TradingView Pro)
"""

import json
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from tabulate import tabulate

import config
from data_loader import fetch_stock_data
from backtester import Backtester


STRATEGIES_INFO = {
    "SWING_PRO": {
        "name": "1. Institutional Swing Pro (Actuală)",
        "desc": "Pullback EMA 20 + Volume Breakout 20D + RS vs SPY + 1W Stage 2 + Split TP1/TP2"
    },
    "RSI2_CONNORS": {
        "name": "2. Larry Connors RSI(2) Mean Reversion",
        "desc": "Cumpără supravânzarea extremă pe termen scurt (RSI 2 < 15) în trend major (Close > SMA 200 & SMA 50), ieșire la revenirea peste EMA 5"
    },
    "MINERVINI_VCP": {
        "name": "3. Mark Minervini VCP (Volatility Contraction)",
        "desc": "Stage 2 Trend (Close > SMA 50 > SMA 200) + Contracție ATR (ATR5 < 0.92*ATR14) + Spargere maxim 10 zile pe volum"
    },
    "DONCHIAN_TURTLE": {
        "name": "4. Donchian Turtle Trend (20D High / 10D Low)",
        "desc": "Spargere canal Donchian 20 zile în trend ascendent, fără țintă fixă — lasă câștigurile să curgă cu Trailing Stop pe minimul a 10 zile"
    },
    "MACD_MOMENTUM": {
        "name": "5. MACD + EMA 10/20 Momentum Swing",
        "desc": "Aliniere EMA 10 > EMA 20 > SMA 50 + Impuls pozitiv pe histograma MACD + RSI(14) în zona de forță (50-70)"
    }
}


def _add_extra_lab_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculează indicatorii suplimentari necesari pentru strategiile alternative din laborator."""
    df = df.copy()

    # RSI(2) pentru strategia Larry Connors
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).fillna(0.0)
    loss = (-delta.where(delta < 0, 0.0)).fillna(0.0)
    avg_g2 = gain.ewm(alpha=1 / 2, min_periods=2, adjust=False).mean()
    avg_l2 = loss.ewm(alpha=1 / 2, min_periods=2, adjust=False).mean()
    rs2 = avg_g2 / avg_l2.replace(0, np.nan)
    df["RSI_2"] = (100.0 - (100.0 / (1.0 + rs2))).fillna(50.0)

    # EMA 5 și EMA 10
    df["EMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    # MACD (12, 26, 9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_Line"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD_Line"] - df["MACD_Signal"]

    # High 10D anterior (pentru VCP Pivot)
    df["Swing_High_10"] = df["High"].shift(1).rolling(window=10, min_periods=5).max()

    return df


def run_alternative_strategy(
    strategy_key: str,
    ticker: str,
    df: pd.DataFrame,
    spy_df: Optional[pd.DataFrame] = None,
    initial_capital: float = config.DEFAULT_SIMULATED_CAPITAL,
    risk_per_trade: float = config.RISK_PER_TRADE_PERCENT
) -> Dict:
    """
    Rulează backtest-ul pentru oricare dintre cele 5 strategii pe un simbol.
    Dacă strategy_key == 'SWING_PRO', folosește motorul oficial Backtester din backtester.py.
    """
    if strategy_key == "SWING_PRO":
        bt = Backtester(
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
            use_ml=False
        )
        res = bt.run(ticker, df, spy_df=spy_df)
        res["trades_list"] = bt.trades
        res["equity_curve"] = bt.equity_curve
        return res

    df = _add_extra_lab_indicators(df)
    if len(df) < 220:
        return {"error": "Date insuficiente (< 220 zile)."}

    spy_bullish_dates = set()
    if spy_df is not None and "SMA_200" in spy_df.columns:
        valid_spy = spy_df[spy_df["Close"] > spy_df["SMA_200"]]
        spy_bullish_dates = set(valid_spy.index)

    capital = initial_capital
    equity_curve = [initial_capital]
    trades = []
    slippage = getattr(config, "BACKTEST_SLIPPAGE_PCT", 0.0005)

    in_position = False
    entry_price = 0.0
    stop_loss = 0.0
    initial_risk = 0.0
    tp_target = 0.0
    shares = 0.0
    entry_date = None
    peak_price = 0.0

    for i in range(200, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        date = df.index[i]

        open_p = float(row["Open"])
        high_p = float(row["High"])
        low_p = float(row["Low"])
        close_p = float(row["Close"])
        sma_50 = float(row["SMA_50"])
        sma_200 = float(row["SMA_200"])
        ema_20 = float(row["EMA_20"])
        ema_10 = float(row["EMA_10"])
        ema_5 = float(row["EMA_5"])
        rsi_14 = float(row["RSI_14"])
        rsi_2 = float(row["RSI_2"])
        atr_14 = float(row["ATR_14"]) if not pd.isna(row.get("ATR_14", np.nan)) else close_p * 0.02
        vol = float(row["Volume"])
        vol_sma = float(row.get("Volume_SMA_20", 0.0))
        swing_low_10 = float(row.get("Swing_Low_10", low_p * 0.96))
        swing_high_20 = float(row.get("Swing_High_20", high_p))
        swing_high_10 = float(row.get("Swing_High_10", high_p))
        atr_contraction = float(row.get("ATR_Contraction", 1.0))
        macd_hist = float(row.get("MACD_Hist", 0.0))
        prev_macd_hist = float(prev_row.get("MACD_Hist", 0.0))

        if in_position:
            peak_price = max(peak_price, high_p)
            exit_triggered = False
            exit_price = close_p
            exit_type = ""

            # 1. Verificare Stop-Loss
            if low_p <= stop_loss:
                raw_exit = min(stop_loss, open_p)
                exit_price = round(raw_exit * (1.0 - slippage), 2)
                exit_type = "STOP_LOSS" if exit_price < entry_price else "TRAILING_STOP"
                exit_triggered = True

            # 2. Reguli specifice de ieșire în profit pentru fiecare strategie
            elif strategy_key == "RSI2_CONNORS":
                # Connors RSI(2): Iese când prețul închide peste EMA 5 sau atinge TP 2.5R
                if close_p >= ema_5 or high_p >= tp_target:
                    exit_price = round((tp_target if high_p >= tp_target else close_p) * (1.0 - slippage), 2)
                    exit_type = "EMA5_REVERSION_EXIT"
                    exit_triggered = True

            elif strategy_key == "DONCHIAN_TURTLE":
                # Turtle: Trailing Stop pe minimul ultimelor 10 zile (sau Chandelier 5%)
                new_stop = round(max(swing_low_10 * 0.995, peak_price * 0.945), 2)
                if new_stop > stop_loss and new_stop < close_p:
                    stop_loss = new_stop
                if high_p >= tp_target:
                    exit_price = round(tp_target * (1.0 - slippage), 2)
                    exit_type = "TP_RUNNER_HIT"
                    exit_triggered = True

            else:
                # MINERVINI_VCP & MACD_MOMENTUM: Auto-Breakeven la +1R, TP la 2.5R, Trail sub EMA 20
                if high_p >= entry_price + initial_risk and stop_loss < entry_price:
                    stop_loss = round(entry_price * 1.002, 2)
                if high_p >= entry_price + (1.5 * initial_risk):
                    trail_ema = round(ema_20 * 0.985, 2)
                    if trail_ema > stop_loss and trail_ema < close_p:
                        stop_loss = trail_ema
                if high_p >= tp_target:
                    exit_price = round(tp_target * (1.0 - slippage), 2)
                    exit_type = "TAKE_PROFIT"
                    exit_triggered = True

            if exit_triggered:
                pnl = (exit_price - entry_price) * shares
                capital += pnl
                r_mult = (exit_price - entry_price) / initial_risk if initial_risk > 0 else 0.0
                trades.append({
                    "ticker": ticker,
                    "entry_date": entry_date,
                    "exit_date": date,
                    "setup_type": strategy_key,
                    "type": exit_type,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "shares": shares,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(((exit_price - entry_price) / entry_price) * 100.0, 2),
                    "r_multiple": round(r_mult, 2)
                })
                in_position = False
                shares = 0.0
                equity_curve.append(capital)
                continue

            equity_curve.append(capital + ((close_p - entry_price) * shares))

        else:
            market_ok = (len(spy_bullish_dates) == 0) or (date in spy_bullish_dates)
            if not market_ok:
                equity_curve.append(capital)
                continue

            signal_triggered = False
            sl_candidate = close_p * 0.96
            rr_target = 2.5

            if strategy_key == "RSI2_CONNORS":
                # Trend ascendent major + supravânzare severă pe 2 zile (RSI_2 < 15)
                if close_p > sma_200 and close_p > sma_50 * 0.97 and rsi_2 < 15.0:
                    signal_triggered = True
                    sl_candidate = close_p - (1.8 * atr_14)
                    rr_target = 2.2

            elif strategy_key == "MINERVINI_VCP":
                # Stage 2 Uptrend + Volatilitate comprimată + Spargere maxim 10 zile pe volum
                vcp_tight = atr_contraction <= 0.95 or float(prev_row.get("ATR_Contraction", 1.0)) <= 0.92
                vol_ok = vol_sma > 0 and vol >= 1.15 * vol_sma
                if (
                    close_p > sma_50 > sma_200
                    and vcp_tight
                    and close_p > swing_high_10
                    and close_p > open_p
                    and vol_ok
                    and 50.0 <= rsi_14 <= 75.0
                ):
                    signal_triggered = True
                    sl_candidate = min(float(df["Low"].iloc[max(0, i - 3):i + 1].min()), close_p - 1.2 * atr_14)
                    rr_target = 2.8

            elif strategy_key == "DONCHIAN_TURTLE":
                # Spargere maxim 20 zile în trend puternic
                if (
                    close_p > swing_high_20
                    and ema_20 > sma_50 > sma_200
                    and close_p > open_p
                    and vol_sma > 0 and vol >= 1.10 * vol_sma
                ):
                    signal_triggered = True
                    sl_candidate = max(swing_low_10, close_p - 2.0 * atr_14)
                    rr_target = 4.0

            elif strategy_key == "MACD_MOMENTUM":
                # Încrucișare / Impuls MACD + EMA 10 > EMA 20 > SMA 50
                macd_cross = (macd_hist > 0 and prev_macd_hist <= 0) or (
                    macd_hist > prev_macd_hist > 0 and float(prev_row["Close"]) <= float(prev_row["EMA_10"]) and close_p > ema_10
                )
                if (
                    close_p > sma_50 > sma_200
                    and ema_10 > ema_20
                    and macd_cross
                    and 48.0 <= rsi_14 <= 68.0
                    and close_p > open_p
                ):
                    signal_triggered = True
                    sl_candidate = min(ema_20 * 0.985, close_p - 1.5 * atr_14)
                    rr_target = 2.5

            if not signal_triggered:
                equity_curve.append(capital)
                continue

            entry_price = round(close_p * (1.0 + slippage), 2)
            stop_dist = (entry_price - sl_candidate) / entry_price
            if sl_candidate >= entry_price or stop_dist < 0.025:
                stop_loss = round(entry_price * 0.972, 2)
            elif stop_dist > 0.055:
                stop_loss = round(entry_price * 0.945, 2)
            else:
                stop_loss = round(sl_candidate, 2)

            initial_risk = round(entry_price - stop_loss, 2)
            if initial_risk <= 0:
                equity_curve.append(capital)
                continue

            tp_target = round(entry_price + (rr_target * initial_risk), 2)
            dollar_risk = capital * risk_per_trade
            shares = round(dollar_risk / initial_risk, 2)
            max_cap = capital / config.MAX_OPEN_POSITIONS
            if shares * entry_price > max_cap:
                shares = round(max_cap / entry_price, 2)

            if shares <= 0:
                equity_curve.append(capital)
                continue

            in_position = True
            peak_price = entry_price
            entry_date = date
            equity_curve.append(capital)

    # Calcul metrici finale
    total_trades = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = round((len(wins) / total_trades) * 100.0, 1) if total_trades > 0 else 0.0
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    total_pnl = round(capital - initial_capital, 2)
    total_return_pct = round((total_pnl / initial_capital) * 100.0, 2)
    expectancy_r = round(float(np.mean([t["r_multiple"] for t in trades])), 2) if trades else 0.0

    eq_series = pd.Series(equity_curve)
    rolling_max = eq_series.cummax()
    dd_series = (eq_series - rolling_max) / rolling_max * 100.0
    max_dd = round(abs(float(dd_series.min())), 2) if not dd_series.empty else 0.0

    daily_ret = eq_series.pct_change().dropna()
    sharpe = 0.0
    if len(daily_ret) > 10 and daily_ret.std() > 0:
        sharpe = round(float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252)), 2)

    return {
        "ticker": ticker,
        "strategy": strategy_key,
        "initial_capital": initial_capital,
        "final_capital": round(capital, 2),
        "total_pnl": total_pnl,
        "total_return_pct": total_return_pct,
        "total_trades": total_trades,
        "total_positions": total_trades,
        "win_rate": win_rate,
        "position_win_rate": win_rate,
        "profit_factor": pf,
        "expectancy_r": expectancy_r,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "trades_list": trades,
        "equity_curve": [round(x, 2) for x in equity_curve]
    }


def generate_interactive_html_report(
    tickers_data: Dict[str, pd.DataFrame],
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> Path:
    """
    Generează un aplicație web interactivă (HTML + TradingView Lightweight Charts)
    în care utilizatorul poate schimba simbolul și strategia cu un singur click,
    văzând lumânările, mediile mobile, săgețile de BUY/SELL și tabelul Strategy Tester.
    """
    payload = {"strategies": STRATEGIES_INFO, "tickers": {}}

    for ticker, df in tickers_data.items():
        df_clean = df.iloc[-320:].copy()  # Ultimele ~320 zile de tranzacționare pt viteză optimă
        candles = []
        ema20_line = []
        sma50_line = []
        sma200_line = []

        for dt, row in df_clean.iterrows():
            d_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
            candles.append({
                "time": d_str,
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
            })
            if not pd.isna(row.get("EMA_20", np.nan)):
                ema20_line.append({"time": d_str, "value": round(float(row["EMA_20"]), 2)})
            if not pd.isna(row.get("SMA_50", np.nan)):
                sma50_line.append({"time": d_str, "value": round(float(row["SMA_50"]), 2)})
            if not pd.isna(row.get("SMA_200", np.nan)):
                sma200_line.append({"time": d_str, "value": round(float(row["SMA_200"]), 2)})

        strat_payload = {}
        for strat_key, res in all_results.get(ticker, {}).items():
            if "error" in res:
                continue
            serialized_trades = []
            markers = []
            for t in res.get("trades_list", []):
                e_date = pd.Timestamp(t["entry_date"]).strftime("%Y-%m-%d")
                x_date = pd.Timestamp(t["exit_date"]).strftime("%Y-%m-%d")
                serialized_trades.append({
                    "entry_date": e_date,
                    "exit_date": x_date,
                    "setup": t.get("setup_type", strat_key),
                    "type": t.get("type", "EXIT"),
                    "entry_price": t["entry_price"],
                    "exit_price": t["exit_price"],
                    "shares": t["shares"],
                    "pnl": t["pnl"],
                    "pnl_pct": t["pnl_pct"],
                    "r_multiple": t.get("r_multiple", 0.0)
                })
                markers.append({
                    "time": e_date,
                    "position": "belowBar",
                    "color": "#00E676",
                    "shape": "arrowUp",
                    "text": f"BUY @ ${t['entry_price']}"
                })
                markers.append({
                    "time": x_date,
                    "position": "aboveBar",
                    "color": "#2979FF" if t["pnl"] >= 0 else "#FF1744",
                    "shape": "arrowDown",
                    "text": f"{t.get('type', 'EXIT')} ({t['pnl']:+.0f}$)"
                })

            # Sortare cronologică a markerelor pentru TradingView Lightweight Charts
            markers.sort(key=lambda m: m["time"])

            strat_payload[strat_key] = {
                "total_return_pct": res.get("total_return_pct", 0.0),
                "total_pnl": res.get("total_pnl", 0.0),
                "win_rate": res.get("position_win_rate", res.get("win_rate", 0.0)),
                "profit_factor": res.get("profit_factor", 0.0),
                "expectancy_r": res.get("expectancy_r", 0.0),
                "max_drawdown_pct": res.get("max_drawdown_pct", 0.0),
                "sharpe_ratio": res.get("sharpe_ratio", 0.0),
                "total_trades": res.get("total_positions", res.get("total_trades", 0)),
                "trades": serialized_trades,
                "markers": markers
            }

        payload["tickers"][ticker] = {
            "candles": candles,
            "ema20": ema20_line,
            "sma50": sma50_line,
            "sma200": sma200_line,
            "results": strat_payload
        }

    json_str = json.dumps(payload)

    html = f"""<!DOCTYPE html>
<html lang="ro">
<head>
<meta charset="UTF-8">
<title>TradingView Strategy Lab — Local Pro Tester</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root {{
    --bg: #0b0e14;
    --panel: #131722;
    --border: #2a2e39;
    --text: #d1d4dc;
    --muted: #787b86;
    --accent: #2962ff;
    --green: #00e676;
    --red: #ff1744;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  body {{ background: var(--bg); color: var(--text); padding: 16px; }}
  .header {{
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 14px 20px; margin-bottom: 14px;
  }}
  .header h1 {{ font-size: 18px; font-weight: 700; color: #fff; }}
  .controls {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  select {{
    background: #1e222d; color: #fff; border: 1px solid var(--border);
    padding: 8px 14px; border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer;
  }}
  .kpi-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px; margin-bottom: 14px;
  }}
  .kpi-card {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px;
  }}
  .kpi-label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi-val {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
  .pos {{ color: var(--green); }}
  .neg {{ color: var(--red); }}
  #chart-container {{
    width: 100%; height: 470px; background: var(--panel);
    border: 1px solid var(--border); border-radius: 8px; margin-bottom: 16px; position: relative;
  }}
  .legend {{
    position: absolute; top: 12px; left: 14px; z-index: 10;
    background: rgba(19, 23, 34, 0.85); padding: 6px 12px; border-radius: 6px;
    font-size: 12px; border: 1px solid var(--border); pointer-events: none;
  }}
  .section-box {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 16px;
  }}
  .section-box h2 {{ font-size: 15px; margin-bottom: 12px; color: #fff; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 9px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; }}
  tr:hover {{ background: rgba(41, 98, 255, 0.08); cursor: pointer; }}
  tr.active-strat {{ background: rgba(41, 98, 255, 0.18); font-weight: 600; }}
  .badge {{ padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
  .badge-buy {{ background: rgba(0, 230, 118, 0.15); color: var(--green); }}
  .badge-loss {{ background: rgba(255, 23, 68, 0.15); color: var(--red); }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>📈 TradingView Local Strategy Lab (Fără Limitări & 100% Gratuit)</h1>
    <div id="strat-desc" style="font-size:12px; color:var(--muted); margin-top:4px;"></div>
  </div>
  <div class="controls">
    <label style="font-size:13px; color:var(--muted);">Simbol:</label>
    <select id="ticker-select"></select>
    <label style="font-size:13px; color:var(--muted);">Strategie Activă:</label>
    <select id="strat-select"></select>
  </div>
</div>

<div class="kpi-grid" id="kpi-grid"></div>

<div id="chart-container">
  <div class="legend">
    <span style="color:#00e5ff; font-weight:600;">● EMA 20</span> &nbsp;|&nbsp;
    <span style="color:#ff9800; font-weight:600;">● SMA 50</span> &nbsp;|&nbsp;
    <span style="color:#ff1744; font-weight:600;">● SMA 200</span> &nbsp;|&nbsp;
    <span>🟢 Săgeată Sus = Intrare BUY &nbsp; 🔵/🔴 Săgeată Jos = Ieșire (TP / SL)</span>
  </div>
</div>

<div class="section-box">
  <h2>🏆 Comparație Directă între cele 5 Strategii pe Simbolul Selectat (Click pe un rând pentru a schimba graficul)</h2>
  <table>
    <thead>
      <tr>
        <th>Strategie</th>
        <th>Randament Net (%)</th>
        <th>Profit ($)</th>
        <th>Win Rate</th>
        <th>Profit Factor</th>
        <th>Expectancy (R)</th>
        <th>Sharpe Ratio</th>
        <th>Max Drawdown</th>
        <th>Tranzacții</th>
      </tr>
    </thead>
    <tbody id="comparison-tbody"></tbody>
  </table>
</div>

<div class="section-box">
  <h2>📋 Lista Tranzacțiilor (Strategy Tester — Istoric Detaliat)</h2>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Data Intrare</th>
        <th>Data Ieșire</th>
        <th>Setup</th>
        <th>Tip Ieșire</th>
        <th>Preț Intrare</th>
        <th>Preț Ieșire</th>
        <th>Acțiuni</th>
        <th>PnL ($)</th>
        <th>Randament (%)</th>
        <th>Multiplu R</th>
      </tr>
    </thead>
    <tbody id="trades-tbody"></tbody>
  </table>
</div>

<script>
const DATA = {json_str};

const tickerSelect = document.getElementById('ticker-select');
const stratSelect = document.getElementById('strat-select');

Object.keys(DATA.tickers).forEach(t => {{
  const opt = document.createElement('option');
  opt.value = t;
  opt.textContent = t;
  tickerSelect.appendChild(opt);
}});

Object.entries(DATA.strategies).forEach(([k, info]) => {{
  const opt = document.createElement('option');
  opt.value = k;
  opt.textContent = info.name;
  stratSelect.appendChild(opt);
}});

const chartEl = document.getElementById('chart-container');
const chart = LightweightCharts.createChart(chartEl, {{
  layout: {{ background: {{ type: 'solid', color: '#131722' }}, textColor: '#d1d4dc' }},
  grid: {{ vertLines: {{ color: '#1e222d' }}, horzLines: {{ color: '#1e222d' }} }},
  crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  rightPriceScale: {{ borderColor: '#2a2e39' }},
  timeScale: {{ borderColor: '#2a2e39', timeVisible: true }}
}});

const candleSeries = chart.addCandlestickSeries({{
  upColor: '#00e676', downColor: '#ff1744', borderVisible: false,
  wickUpColor: '#00e676', wickDownColor: '#ff1744'
}});
const ema20Series = chart.addLineSeries({{ color: '#00e5ff', lineWidth: 2, title: 'EMA 20' }});
const sma50Series = chart.addLineSeries({{ color: '#ff9800', lineWidth: 2, title: 'SMA 50' }});
const sma200Series = chart.addLineSeries({{ color: '#ff1744', lineWidth: 2, title: 'SMA 200' }});

window.addEventListener('resize', () => {{
  chart.applyOptions({{ width: chartEl.clientWidth, height: chartEl.clientHeight }});
}});

function renderDashboard() {{
  const ticker = tickerSelect.value;
  const stratKey = stratSelect.value;
  const tData = DATA.tickers[ticker];
  if (!tData) return;

  document.getElementById('strat-desc').textContent = DATA.strategies[stratKey].desc;

  candleSeries.setData(tData.candles);
  ema20Series.setData(tData.ema20);
  sma50Series.setData(tData.sma50);
  sma200Series.setData(tData.sma200);

  const res = tData.results[stratKey] || {{}};
  candleSeries.setMarkers(res.markers || []);
  chart.timeScale().fitContent();

  // KPI Cards
  const ret = res.total_return_pct || 0;
  const pnl = res.total_pnl || 0;
  const wr = res.win_rate || 0;
  const pf = res.profit_factor || 0;
  const expR = res.expectancy_r || 0;
  const dd = res.max_drawdown_pct || 0;

  document.getElementById('kpi-grid').innerHTML = `
    <div class="kpi-card"><div class="kpi-label">Randament Net</div><div class="kpi-val ${{ret >= 0 ? 'pos' : 'neg'}}">${{ret >= 0 ? '+' : ''}}${{ret.toFixed(2)}}%</div></div>
    <div class="kpi-card"><div class="kpi-label">Profit Net ($)</div><div class="kpi-val ${{pnl >= 0 ? 'pos' : 'neg'}}">$${{pnl.toLocaleString()}}</div></div>
    <div class="kpi-card"><div class="kpi-label">Win Rate</div><div class="kpi-val">${{wr.toFixed(1)}}%</div></div>
    <div class="kpi-card"><div class="kpi-label">Profit Factor</div><div class="kpi-val ${{pf >= 1.5 ? 'pos' : ''}}">${{pf.toFixed(2)}}</div></div>
    <div class="kpi-card"><div class="kpi-label">Expectancy</div><div class="kpi-val ${{expR >= 0 ? 'pos' : 'neg'}}">${{expR >= 0 ? '+' : ''}}${{expR.toFixed(2)}}R</div></div>
    <div class="kpi-card"><div class="kpi-label">Max Drawdown</div><div class="kpi-val neg">-${{dd.toFixed(2)}}%</div></div>
  `;

  // Comparison Table
  const compBody = document.getElementById('comparison-tbody');
  compBody.innerHTML = '';
  Object.entries(DATA.strategies).forEach(([k, info]) => {{
    const r = tData.results[k];
    if (!r) return;
    const tr = document.createElement('tr');
    if (k === stratKey) tr.className = 'active-strat';
    tr.onclick = () => {{ stratSelect.value = k; renderDashboard(); }};
    tr.innerHTML = `
      <td>${{info.name}}</td>
      <td class="${{r.total_return_pct >= 0 ? 'pos' : 'neg'}}">${{r.total_return_pct >= 0 ? '+' : ''}}${{r.total_return_pct.toFixed(2)}}%</td>
      <td class="${{r.total_pnl >= 0 ? 'pos' : 'neg'}}">$${{r.total_pnl.toFixed(2)}}</td>
      <td>${{r.win_rate.toFixed(1)}}%</td>
      <td>${{r.profit_factor.toFixed(2)}}</td>
      <td>${{r.expectancy_r >= 0 ? '+' : ''}}${{r.expectancy_r.toFixed(2)}}R</td>
      <td>${{r.sharpe_ratio.toFixed(2)}}</td>
      <td class="neg">-${{r.max_drawdown_pct.toFixed(2)}}%</td>
      <td>${{r.total_trades}}</td>
    `;
    compBody.appendChild(tr);
  }});

  // Trades Table
  const tradesBody = document.getElementById('trades-tbody');
  tradesBody.innerHTML = '';
  const trades = (res.trades || []).slice().reverse();
  trades.forEach((t, idx) => {{
    const tr = document.createElement('tr');
    const isWin = t.pnl >= 0;
    tr.innerHTML = `
      <td>${{trades.length - idx}}</td>
      <td>${{t.entry_date}}</td>
      <td>${{t.exit_date}}</td>
      <td>${{t.setup}}</td>
      <td><span class="badge ${{isWin ? 'badge-buy' : 'badge-loss'}}">${{t.type}}</span></td>
      <td>$${{t.entry_price.toFixed(2)}}</td>
      <td>$${{t.exit_price.toFixed(2)}}</td>
      <td>${{t.shares}}</td>
      <td class="${{isWin ? 'pos' : 'neg'}}">${{isWin ? '+' : ''}}$${{t.pnl.toFixed(2)}}</td>
      <td class="${{isWin ? 'pos' : 'neg'}}">${{isWin ? '+' : ''}}${{t.pnl_pct.toFixed(2)}}%</td>
      <td class="${{t.r_multiple >= 0 ? 'pos' : 'neg'}}">${{t.r_multiple >= 0 ? '+' : ''}}${{t.r_multiple.toFixed(2)}}R</td>
    `;
    tradesBody.appendChild(tr);
  }});
}}

tickerSelect.addEventListener('change', renderDashboard);
stratSelect.addEventListener('change', renderDashboard);
renderDashboard();
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def run_strategy_lab(custom_tickers: Optional[List[str]] = None, open_browser: bool = True):
    """
    Rulează comparatorul multi-strategie în terminal + generează graficul interactiv
    în stil TradingView în browser.
    """
    print("\n" + "=" * 78)
    print("   🧪 TRADINGVIEW LOCAL STRATEGY LAB — TESTARE & COMPARARE 5 STRATEGII   ")
    print("=" * 78)

    if custom_tickers is None:
        default_sample = ["NVDA", "META", "AVGO", "AAPL", "MSFT", "PLTR", "AMZN", "GOOGL"]
        print(f"\nIntrodu simbolurile pe care vrei să le testezi (ex: NVDA, META, TSLA)")
        print(f"sau apasă ENTER pentru selecția implicită ({', '.join(default_sample)}):")
        try:
            user_in = input("Simboluri: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            user_in = ""

        if user_in == "ALL":
            tickers = config.WATCHLIST
        elif user_in:
            tickers = [t.strip() for t in user_in.replace(";", ",").split(",") if t.strip()]
        else:
            tickers = default_sample
    else:
        tickers = custom_tickers

    print(f"\n📥 Se descarcă datele istorice (2 ani) pentru SPY și {len(tickers)} acțiuni...")
    try:
        spy_df = fetch_stock_data(config.BENCHMARK_TICKER)
    except Exception:
        spy_df = None

    tickers_data: Dict[str, pd.DataFrame] = {}
    all_results: Dict[str, Dict[str, Dict]] = {}

    # Agregare globală per strategie
    summary_agg = {
        k: {"pnl": 0.0, "ret_list": [], "wr_list": [], "pf_list": [], "exp_list": [], "dd_list": [], "trades": 0}
        for k in STRATEGIES_INFO
    }

    for ticker in tickers:
        try:
            df = fetch_stock_data(ticker, spy_df=spy_df)
            tickers_data[ticker] = df
            all_results[ticker] = {}

            for strat_key in STRATEGIES_INFO:
                res = run_alternative_strategy(strat_key, ticker, df, spy_df=spy_df)
                all_results[ticker][strat_key] = res
                if "error" not in res and res.get("total_trades", 0) > 0:
                    agg = summary_agg[strat_key]
                    agg["pnl"] += res["total_pnl"]
                    agg["ret_list"].append(res["total_return_pct"])
                    agg["wr_list"].append(res.get("position_win_rate", res["win_rate"]))
                    if res["profit_factor"] < 90:
                        agg["pf_list"].append(res["profit_factor"])
                    agg["exp_list"].append(res.get("expectancy_r", 0.0))
                    agg["dd_list"].append(res["max_drawdown_pct"])
                    agg["trades"] += res.get("total_positions", res["total_trades"])
        except Exception as e:
            print(f"  ⚠️ Eroare la procesarea {ticker}: {e}")

    # Afișare clasament comparativ în terminal
    table_rows = []
    for strat_key, info in STRATEGIES_INFO.items():
        agg = summary_agg[strat_key]
        avg_ret = np.mean(agg["ret_list"]) if agg["ret_list"] else 0.0
        avg_wr = np.mean(agg["wr_list"]) if agg["wr_list"] else 0.0
        avg_pf = np.mean(agg["pf_list"]) if agg["pf_list"] else 0.0
        avg_exp = np.mean(agg["exp_list"]) if agg["exp_list"] else 0.0
        avg_dd = np.mean(agg["dd_list"]) if agg["dd_list"] else 0.0

        table_rows.append([
            info["name"],
            f"${agg['pnl']:+,.2f}",
            f"{avg_ret:+.2f}%",
            f"{avg_wr:.1f}%",
            f"{avg_pf:.2f}",
            f"{avg_exp:+.2f}R",
            f"-{avg_dd:.2f}%",
            agg["trades"]
        ])

    headers = [
        "Strategie Testată", "Profit Cumulat ($)", "Randament Mediu/Simbol",
        "Win Rate Mediu", "Profit Factor", "Expectancy", "Max DD Mediu", "Total Poziții"
    ]
    print("\n🏆 CLASAMENT COMPARATIV AL CELOR 5 STRATEGII:")
    print(tabulate(table_rows, headers=headers, tablefmt="grid"))

    if tickers_data:
        report_file = config.BASE_DIR / "data" / "strategy_lab.html"
        generate_interactive_html_report(tickers_data, all_results, report_file)
        print(f"\n✨ Graficul interactiv TradingView Local a fost generat în:")
        print(f"   👉 {report_file}")
        if open_browser:
            try:
                webbrowser.open(f"file://{report_file.resolve()}")
                print("🌐 S-a deschis automat în browser! Poți schimba simbolul și strategia din meniul de sus.")
            except Exception:
                pass


if __name__ == "__main__":
    run_strategy_lab()
