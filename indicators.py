"""
Calculul indicatorilor tehnici folosind exclusiv pandas si numpy.
Include:
  - Medii Mobile Zilnice: SMA (50, 200), EMA (15, 20, 25), SMA_50_Slope
  - Momentum & Volatilitate: RSI (14), ATR (5, 14), ATR_Contraction (VCP)
  - Volum & Structura: Volume_SMA_20, Volume_Ratio_3D (Volume Dry-Up), Swing_High_20, Swing_Low_10
  - Forta Relativa Institutionala: ROC_Weighted (IBD Style) & RS_vs_SPY (Mansfield / Alpha vs S&P 500)
  - Multi-Timeframe (1W): Weekly_EMA_10, Weekly_SMA_30, Weekly_RSI_14 (cu protectie Anti-Lookahead shift(1))
"""

from typing import Optional
import pandas as pd
import numpy as np


def add_sma(df: pd.DataFrame, period: int, column: str = "Close") -> pd.DataFrame:
    """Adauga Simple Moving Average (SMA) la dataframe."""
    df[f"SMA_{period}"] = df[column].rolling(window=period).mean()
    return df


def add_ema(df: pd.DataFrame, period: int, column: str = "Close") -> pd.DataFrame:
    """Adauga Exponential Moving Average (EMA) la dataframe."""
    df[f"EMA_{period}"] = df[column].ewm(span=period, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.DataFrame:
    """
    Calculeaza Relative Strength Index (RSI) folosind metoda clasica Wilder / EWM.
    """
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"RSI_{period}"] = (100 - (100 / (1 + rs))).fillna(50.0)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculeaza Average True Range (ATR) pentru volatilitate.
    """
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[f"ATR_{period}"] = tr.rolling(window=period).mean()
    return df


def compute_weighted_roc(close_series: pd.Series) -> pd.Series:
    """
    Calculeaza Rate of Change ponderat (stil William O'Neil / IBD RS):
      - 40% pondere pe ultimele 20 zile (~1 luna)
      - 30% pondere pe ultimele 60 zile (~3 luni)
      - 30% pondere pe ultimele 120 zile (~6 luni)
    """
    roc_20 = close_series.pct_change(20) * 100
    roc_60 = close_series.pct_change(60) * 100
    roc_120 = close_series.pct_change(120) * 100

    # Daca 120 zile nu sunt disponibile inca, folosim media disponibila
    weighted = (0.40 * roc_20.fillna(0)) + (0.30 * roc_60.fillna(roc_20.fillna(0))) + (0.30 * roc_120.fillna(roc_60.fillna(0)))
    return weighted


def add_relative_strength(df: pd.DataFrame, spy_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculeaza Forta Relativa a actiunii fata de indicele de referinta SPY (RS_vs_SPY).
    O valoare pozitiva (> 0) inseamna ca actiunea este Lider de Piata (bate S&P 500).
    """
    df["ROC_Weighted"] = compute_weighted_roc(df["Close"])

    if spy_df is not None and not spy_df.empty and "Close" in spy_df.columns:
        try:
            spy_roc = compute_weighted_roc(spy_df["Close"])
            # Aliniere index fara timezone mismatch
            df_idx = pd.to_datetime(df.index)
            if getattr(df_idx, "tz", None) is not None:
                df_idx = df_idx.tz_localize(None)
            spy_idx = pd.to_datetime(spy_df.index)
            if getattr(spy_idx, "tz", None) is not None:
                spy_idx = spy_idx.tz_localize(None)

            spy_series = pd.Series(spy_roc.values, index=spy_idx)
            aligned_spy_roc = spy_series.reindex(df_idx).ffill().fillna(0.0)
            df["RS_vs_SPY"] = (df["ROC_Weighted"].values - aligned_spy_roc.values).round(2)
        except Exception:
            df["RS_vs_SPY"] = df["ROC_Weighted"].round(2)
    else:
        if "RS_vs_SPY" not in df.columns:
            df["RS_vs_SPY"] = df["ROC_Weighted"].round(2)

    return df


def add_weekly_indicators(
    df: pd.DataFrame,
    ema_period: int = 10,
    sma_period: int = 30,
    rsi_period: int = 14
) -> pd.DataFrame:
    """
    Calculeaza indicatorii pe interval saptamanal (1W) si ii aliniaza
    la DataFrame-ul zilnic cu un shift strict de 1 saptamana.
    
    Protectie Anti-Lookahead Bias:
      - O lumanare zilnica din cursul saptamanii are acces DOAR la indicatorii
        saptamanii anterioare complet inchise (shift(1)).
      - Indicatori: Weekly_EMA_10, Weekly_SMA_30, Weekly_RSI_14, Weekly_Close.
    """
    try:
        df_idx = df.copy()
        if not isinstance(df_idx.index, pd.DatetimeIndex):
            df_idx.index = pd.to_datetime(df_idx.index)

        weekly = df_idx.resample("W-FRI").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna(subset=["Close"])

        if len(weekly) < 5:
            df[f"Weekly_EMA_{ema_period}"] = np.nan
            df[f"Weekly_SMA_{sma_period}"] = np.nan
            df[f"Weekly_RSI_{rsi_period}"] = np.nan
            df["Weekly_Close"] = np.nan
            return df

        weekly[f"Weekly_EMA_{ema_period}"] = (
            weekly["Close"].ewm(span=ema_period, adjust=False).mean()
        )

        weekly[f"Weekly_SMA_{sma_period}"] = (
            weekly["Close"].rolling(window=sma_period, min_periods=max(5, sma_period // 3)).mean()
        )

        delta = weekly["Close"].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        weekly[f"Weekly_RSI_{rsi_period}"] = (100 - (100 / (1 + rs))).fillna(50.0)

        # Shift strict de 1 saptamana completa pentru a elimina lookahead bias
        weekly_shifted = weekly[[
            f"Weekly_EMA_{ema_period}",
            f"Weekly_SMA_{sma_period}",
            f"Weekly_RSI_{rsi_period}",
            "Close"
        ]].shift(1)
        weekly_shifted.rename(columns={"Close": "Weekly_Close"}, inplace=True)

        cols_to_drop = [c for c in weekly_shifted.columns if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        df = df.join(weekly_shifted, how="left").ffill()

    except Exception:
        for col in [f"Weekly_EMA_{ema_period}", f"Weekly_SMA_{sma_period}", f"Weekly_RSI_{rsi_period}", "Weekly_Close"]:
            if col not in df.columns:
                df[col] = np.nan

    return df


def add_all_indicators(df: pd.DataFrame, spy_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculeaza toti indicatorii necesari pentru strategia Swing Trading Pro:
    - Medii mobile zilnice (SMA 50/200, EMA 15/20/25) + Panta SMA 50
    - Momentum & Volatilitate (RSI 14, ATR 5/14, VCP Contraction)
    - Volum & Structura (Volume SMA 20, Volume Ratio 3D, Swing High 20 / Low 10)
    - Forta Relativa vs SPY (RS_vs_SPY)
    - Indicatori Multi-Timeframe (1W - Saptamanal)
    """
    df = df.copy()
    df = df.sort_index()

    # Medii mobile zilnice
    df = add_sma(df, period=200)
    df = add_sma(df, period=50)
    df = add_ema(df, period=15)
    df = add_ema(df, period=20)
    df = add_ema(df, period=25)

    # Panta SMA 50 pe ultimele 5 zile (%)
    df["SMA_50_Slope"] = ((df["SMA_50"] - df["SMA_50"].shift(5)) / df["SMA_50"].shift(5).replace(0, np.nan) * 100).fillna(0.0)

    # Momentum & Volatilitate zilnice
    df = add_rsi(df, period=14)
    df = add_atr(df, period=14)
    df = add_atr(df, period=5)

    # Raport de contractie a volatilitatii (VCP: < 1.0 indica comprimare de volatilitate)
    df["ATR_Contraction"] = (df["ATR_5"] / df["ATR_14"].replace(0, np.nan)).fillna(1.0)

    # Volum si Volume Dry-Up (media pe 3 zile raportata la media pe 20 zile)
    df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()
    vol_3d = df["Volume"].rolling(window=3).mean()
    df["Volume_Ratio_3D"] = (vol_3d / df["Volume_SMA_20"].replace(0, np.nan)).fillna(1.0)

    # Niveluri Structurale de Suport / Rezistenta (fara bara curenta - shift(1))
    df["Swing_High_20"] = df["High"].shift(1).rolling(window=20, min_periods=5).max()
    df["Swing_Low_10"] = df["Low"].shift(1).rolling(window=10, min_periods=3).min()
    df["Low_3D"] = df["Low"].shift(1).rolling(window=3, min_periods=1).min()

    # Forta Relativa (Relative Strength vs SPY)
    df = add_relative_strength(df, spy_df=spy_df)

    # Indicatori Multi-Timeframe (1W - Saptamanal)
    df = add_weekly_indicators(df, ema_period=10, sma_period=30, rsi_period=14)

    return df
