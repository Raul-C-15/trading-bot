"""
Calculul indicatorilor tehnici folosind exclusiv pandas și numpy (Zero Look-Ahead Bias).
Toți indicatorii sunt calculați strict cauzal pe lumânări complet închise ('closed'):
  - Medii Mobile Zilnice: SMA (50, 200), EMA (15, 20, 25), SMA_50_Slope, SMA_200_Slope
  - Momentum & Volatilitate: RSI (14), RSI (4) pentru pullback-uri ordonate, ATR (5, 14), ATR_Pct, ATR_Contraction (VCP)
  - Volum & Structură: Volume_SMA_20, Volume_Ratio_3D (Volume Dry-Up), Swing_High_20, Swing_Low_10, Range_15D_Pct
  - Forță Relativă Instituțională: ROC_Weighted (IBD Style) & RS_vs_SPY
  - Multi-Timeframe (1W): Weekly_EMA_10, Weekly_SMA_30, Weekly_SMA_50, Weekly_RSI_14 (cu shift(1) săptămână închisă)
"""

from typing import Optional
import pandas as pd
import numpy as np


def add_sma(df: pd.DataFrame, period: int, column: str = "Close") -> pd.DataFrame:
    """Adaugă Simple Moving Average (SMA) la dataframe."""
    df[f"SMA_{period}"] = df[column].rolling(window=period).mean()
    return df


def add_ema(df: pd.DataFrame, period: int, column: str = "Close") -> pd.DataFrame:
    """Adaugă Exponential Moving Average (EMA) la dataframe."""
    df[f"EMA_{period}"] = df[column].ewm(span=period, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.DataFrame:
    """Calculează Relative Strength Index (RSI) folosind metoda clasică Wilder."""
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0.0).fillna(0.0)
    loss = (-delta.where(delta < 0, 0.0)).fillna(0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"RSI_{period}"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculează Average True Range (ATR) pentru volatilitate."""
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
    Calculează Rate of Change ponderat (stil William O'Neil / IBD RS):
      - 40% pondere pe ultimele 20 zile (~1 lună)
      - 30% pondere pe ultimele 60 zile (~3 luni)
      - 30% pondere pe ultimele 120 zile (~6 luni)
    """
    roc_20 = close_series.pct_change(20) * 100.0
    roc_60 = close_series.pct_change(60) * 100.0
    roc_120 = close_series.pct_change(120) * 100.0

    weighted = (
        0.40 * roc_20.fillna(0.0)
        + 0.30 * roc_60.fillna(roc_20.fillna(0.0))
        + 0.30 * roc_120.fillna(roc_60.fillna(0.0))
    )
    return weighted


def add_relative_strength(df: pd.DataFrame, spy_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Calculează Forța Relativă a acțiunii față de indicele de referință SPY (RS_vs_SPY)."""
    df["ROC_Weighted"] = compute_weighted_roc(df["Close"])

    if spy_df is not None and not spy_df.empty and "Close" in spy_df.columns:
        try:
            spy_roc = compute_weighted_roc(spy_df["Close"])
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
    sma50_period: int = 50,
    rsi_period: int = 14
) -> pd.DataFrame:
    """
    Calculează indicatorii pe interval săptămânal (1W) și îi aliniază la DataFrame-ul zilnic
    cu un shift strict de 1 săptămână complet închisă (Zero Look-Ahead Bias).
    """
    w_cols = [
        f"Weekly_EMA_{ema_period}",
        f"Weekly_SMA_{sma_period}",
        f"Weekly_SMA_{sma50_period}",
        f"Weekly_RSI_{rsi_period}",
        "Weekly_Close",
    ]
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
            for col in w_cols:
                df[col] = np.nan
            return df

        weekly[f"Weekly_EMA_{ema_period}"] = weekly["Close"].ewm(span=ema_period, adjust=False).mean()
        weekly[f"Weekly_SMA_{sma_period}"] = weekly["Close"].rolling(
            window=sma_period, min_periods=max(5, sma_period // 3)
        ).mean()
        weekly[f"Weekly_SMA_{sma50_period}"] = weekly["Close"].rolling(
            window=sma50_period, min_periods=max(10, sma50_period // 2)
        ).mean()

        delta = weekly["Close"].diff()
        gain = delta.where(delta > 0, 0.0).fillna(0.0)
        loss = (-delta.where(delta < 0, 0.0)).fillna(0.0)
        avg_gain = gain.ewm(alpha=1.0 / rsi_period, min_periods=rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / rsi_period, min_periods=rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        weekly[f"Weekly_RSI_{rsi_period}"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

        # Shift strict de 1 săptămână completă pentru eliminarea totală a look-ahead bias-ului
        weekly_shifted = weekly[[
            f"Weekly_EMA_{ema_period}",
            f"Weekly_SMA_{sma_period}",
            f"Weekly_SMA_{sma50_period}",
            f"Weekly_RSI_{rsi_period}",
            "Close"
        ]].shift(1)
        weekly_shifted.rename(columns={"Close": "Weekly_Close"}, inplace=True)

        cols_to_drop = [c for c in weekly_shifted.columns if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        df = df.join(weekly_shifted, how="left").ffill()

    except Exception:
        for col in w_cols:
            if col not in df.columns:
                df[col] = np.nan

    return df


def add_all_indicators(df: pd.DataFrame, spy_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculează toți indicatorii necesari pentru strategia Swing Trading V2 pe lumânări închise:
    - Medii mobile zilnice (SMA 50/200, EMA 15/20/25) + Pantele SMA 50 și SMA 200
    - Momentum & Volatilitate (RSI 14, RSI 4, ATR 5/14, ATR_Pct, VCP Contraction)
    - Volum & Structură (Volume SMA 20, Volume Ratio 3D, Swing High 20 / Low 10, Range 15D)
    - Forța Relativă vs SPY (RS_vs_SPY)
    - Indicatori Multi-Timeframe (1W - Săptămânal cu shift(1))
    """
    df = df.copy()
    df = df.sort_index()

    # Medii mobile zilnice
    df = add_sma(df, period=200)
    df = add_sma(df, period=50)
    df = add_ema(df, period=15)
    df = add_ema(df, period=20)
    df = add_ema(df, period=25)

    # Pantele trendului (strict cauzale)
    df["SMA_50_Slope"] = (
        (df["SMA_50"] - df["SMA_50"].shift(5)) / df["SMA_50"].shift(5).replace(0, np.nan) * 100.0
    ).fillna(0.0)
    df["SMA_200_Slope"] = (
        (df["SMA_200"] - df["SMA_200"].shift(20)) / df["SMA_200"].shift(20).replace(0, np.nan) * 100.0
    ).fillna(0.0)

    # Momentum & Volatilitate zilnice
    df = add_rsi(df, period=14)
    df = add_rsi(df, period=4)
    df = add_atr(df, period=14)
    df = add_atr(df, period=5)

    df["ATR_Pct"] = ((df["ATR_14"] / df["Close"].replace(0, np.nan)) * 100.0).fillna(2.5)
    df["ATR_Contraction"] = (df["ATR_5"] / df["ATR_14"].replace(0, np.nan)).fillna(1.0)

    # Volum și Volume Dry-Up (media pe 3 zile raportată la media pe 20 zile)
    df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()
    vol_3d = df["Volume"].rolling(window=3).mean()
    df["Volume_Ratio_3D"] = (vol_3d / df["Volume_SMA_20"].replace(0, np.nan)).fillna(1.0)

    # Niveluri Structurale de Suport / Rezistență (strict fără look-ahead)
    df["Swing_High_20"] = df["High"].shift(1).rolling(window=20, min_periods=5).max()
    df["Swing_Low_10"] = df["Low"].shift(1).rolling(window=10, min_periods=3).min()
    df["Swing_Low_10_All"] = df["Low"].rolling(window=10, min_periods=3).min()
    df["Low_3D"] = df["Low"].shift(1).rolling(window=3, min_periods=1).min()
    df["Low_5D"] = df["Low"].shift(1).rolling(window=5, min_periods=2).min()

    high_15 = df["High"].shift(1).rolling(window=15, min_periods=8).max()
    low_15 = df["Low"].shift(1).rolling(window=15, min_periods=8).min()
    df["Range_15D_Pct"] = (((high_15 - low_15) / df["Close"].shift(1).replace(0, np.nan)) * 100.0).fillna(10.0)

    # Forța Relativă (Relative Strength vs SPY)
    df = add_relative_strength(df, spy_df=spy_df)

    # Indicatori Multi-Timeframe (1W - Săptămânal cu shift(1))
    df = add_weekly_indicators(df, ema_period=10, sma_period=30, sma50_period=50, rsi_period=14)

    return df
