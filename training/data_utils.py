import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL
from torch.utils.data import DataLoader, Dataset


class OrinocoDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


@dataclass
class SplitData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_columns: list[str]


def load_excel_dataset(excel_path: str, date_col: str, station_cols: Sequence[str]) -> pd.DataFrame:
    excel_file = Path(excel_path)
    if not excel_file.exists():
        cwd = Path.cwd()
        hint = (
            f"Archivo Excel no encontrado: '{excel_path}'. "
            f"Ruta absoluta esperada: '{excel_file.resolve()}'. "
            "Verifica --excel-path o coloca el archivo en data/raw/."
        )
        # Ayuda rapida para descubrir posibles archivos de datos existentes.
        candidates = sorted(cwd.glob("data/raw/*.xlsx"))
        if candidates:
            sample = ", ".join(str(p) for p in candidates[:5])
            hint += f" Candidatos en data/raw: {sample}"
        raise FileNotFoundError(hint)

    df = pd.read_excel(excel_file)
    expected_cols = [date_col, *station_cols]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en el Excel: {missing}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    df = df[list(station_cols)].copy()
    return df


def impute_missing_values(df: pd.DataFrame, short_gap_limit: int = 7) -> pd.DataFrame:
    # Interpolacion lineal para huecos cortos y ffill como respaldo.
    interpolated = df.interpolate(method="time", limit=short_gap_limit, limit_direction="both")
    return interpolated.ffill().bfill()


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    day_of_year = out.index.dayofyear
    month = out.index.month

    out["dia_sin"] = np.sin(2 * np.pi * day_of_year / 365.0)
    out["dia_cos"] = np.cos(2 * np.pi * day_of_year / 365.0)
    out["mes_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["mes_cos"] = np.cos(2 * np.pi * month / 12.0)
    out["temporada"] = out.index.month.map(_season_code).astype(float)
    return out


def _season_code(month: int) -> int:
    if month in (1, 2, 3):
        return 0
    if month in (4, 5, 6):
        return 1
    if month in (7, 8, 9, 10):
        return 2
    return 3


def detect_outliers_iqr_by_month(series: pd.Series, k: float = 1.5) -> pd.Series:
    flags = pd.Series(False, index=series.index)
    for m in range(1, 13):
        subset = series[series.index.month == m]
        if subset.empty:
            continue
        q1 = subset.quantile(0.25)
        q3 = subset.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr
        flags.loc[subset.index] = (subset < low) | (subset > high)
    return flags


def temporal_split(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> SplitData:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    feature_columns = list(df.columns)
    return SplitData(train_df=train_df, val_df=val_df, test_df=test_df, feature_columns=feature_columns)


def scale_splits(
    split_data: SplitData, scaler_path: str | Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    train_norm = scaler.fit_transform(split_data.train_df.values)
    val_norm = scaler.transform(split_data.val_df.values)
    test_norm = scaler.transform(split_data.test_df.values)
    joblib.dump(scaler, scaler_path)
    return train_norm, val_norm, test_norm, scaler


def create_windows(
    data: np.ndarray, target_idx: int, lookback: int = 90, horizon: int = 30, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    limit = len(data) - lookback - horizon + 1
    for i in range(0, max(limit, 0), stride):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + horizon, target_idx])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(OrinocoDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(OrinocoDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(OrinocoDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def save_feature_columns(feature_columns: Sequence[str], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"feature_columns": list(feature_columns)}, f, ensure_ascii=True, indent=2)


def run_fft_analysis(series: pd.Series, sampling_days: float = 1.0) -> pd.DataFrame:
    y = series.values.astype(float)
    n = len(y)
    yf = rfft(y - np.mean(y))
    xf = rfftfreq(n, d=sampling_days)
    power = np.abs(yf)
    out = pd.DataFrame({"frequency_per_day": xf, "power": power})
    out = out[out["frequency_per_day"] > 0].sort_values("power", ascending=False)
    return out.reset_index(drop=True)


def run_stl_decomposition(series: pd.Series, period: int = 365) -> STL:
    return STL(series, period=period, robust=True).fit()
