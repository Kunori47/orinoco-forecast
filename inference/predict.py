import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from models import LSTMForecaster, TransformerForecaster
from training.data_utils import add_cyclical_features, impute_missing_values, load_excel_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferencia autoregresiva Orinoco Forecast")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--scaler-path", type=str, default="")
    parser.add_argument("--feature-columns", type=str, default="")
    parser.add_argument("--excel-path", type=str, required=True)
    parser.add_argument("--date-col", type=str, default="fecha")
    parser.add_argument("--station-cols", nargs="+", default=["ayacucho", "caicara", "ciudad_bolivar", "palua"])
    parser.add_argument("--target-col", type=str, default="ciudad_bolivar")
    parser.add_argument("--model-type", type=str, choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--lookback", type=int, default=90)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument(
        "--recursive-step",
        type=int,
        default=1,
        help="Cuantos dias consumir por iteracion autoregresiva (1 reduce saltos/lag).",
    )
    parser.add_argument(
        "--enso-scenario",
        type=str,
        choices=["neutral", "nino", "nina"],
        default="neutral",
        help="Escenario explicito ENSO para ajustar la trayectoria futura.",
    )
    parser.add_argument(
        "--enso-strength",
        type=float,
        default=1.0,
        help="Intensidad del ajuste ENSO (0 desactiva, >1 mas intenso).",
    )
    parser.add_argument(
        "--enso-lag-days",
        type=int,
        default=0,
        help="Desfase en dias para el ajuste ENSO (positivo retrasa picos).",
    )
    parser.add_argument(
        "--enso-smooth-window",
        type=int,
        default=21,
        help="Ventana (dias) para suavizar el ajuste ENSO y evitar picos artificiales.",
    )
    parser.add_argument(
        "--variability-gain",
        type=float,
        default=1.0,
        help="Ganancia de variabilidad final (1.0 sin cambio, >1 mas picos, <1 mas plano).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--output-csv", type=str, default="")
    return parser.parse_args()


def build_model(model_type: str, input_size: int, horizon: int) -> torch.nn.Module:
    if model_type == "lstm":
        return LSTMForecaster(input_size=input_size, forecast_horizon=horizon)
    return TransformerForecaster(input_size=input_size, forecast_horizon=horizon)


def infer_horizon_from_state_dict(state_dict: dict) -> int | None:
    if "head.bias" in state_dict:
        return int(state_dict["head.bias"].shape[0])
    if "head.weight" in state_dict:
        return int(state_dict["head.weight"].shape[0])
    return None


def generate_future_features(last_date, n_days: int) -> np.ndarray:
    future_dates = [last_date + np.timedelta64(i + 1, "D") for i in range(n_days)]
    day_of_year = np.array([d.astype("datetime64[D]").astype(object).timetuple().tm_yday for d in future_dates])
    month = np.array([d.astype("datetime64[D]").astype(object).month for d in future_dates])
    dia_sin = np.sin(2 * np.pi * day_of_year / 365.0)
    dia_cos = np.cos(2 * np.pi * day_of_year / 365.0)
    mes_sin = np.sin(2 * np.pi * month / 12.0)
    mes_cos = np.cos(2 * np.pi * month / 12.0)
    temporada = np.array([season_code(m) for m in month], dtype=float)
    return np.column_stack([dia_sin, dia_cos, mes_sin, mes_cos, temporada]), future_dates


def season_code(month: int) -> int:
    if month in (1, 2, 3):
        return 0
    if month in (4, 5, 6):
        return 1
    if month in (7, 8, 9, 10):
        return 2
    return 3


def inverse_target(scaler, values: np.ndarray, target_idx: int, n_features: int) -> np.ndarray:
    helper = np.zeros((len(values), n_features), dtype=np.float32)
    helper[:, target_idx] = values
    inv = scaler.inverse_transform(helper)[:, target_idx]
    return inv


def scale_feature_values(scaler, values: np.ndarray, feature_idx: int) -> np.ndarray:
    # MinMaxScaler transforma por columna: x_scaled = x * scale_ + min_
    return values * scaler.scale_[feature_idx] + scaler.min_[feature_idx]


def resolve_artifact_path(explicit: str, new_path: Path, legacy_path: Path) -> Path:
    if explicit:
        return Path(explicit)
    if new_path.exists():
        return new_path
    return legacy_path


def get_enso_month_weights() -> dict[int, float]:
    # Mayor impacto en meses de crecida local.
    return {
        1: 0.4,
        2: 0.4,
        3: 0.5,
        4: 0.7,
        5: 0.9,
        6: 1.1,
        7: 1.2,
        8: 1.2,
        9: 1.0,
        10: 0.8,
        11: 0.6,
        12: 0.5,
    }


def apply_enso_adjustment(
    preds_real: np.ndarray,
    future_dates: list[np.datetime64],
    historical_target: pd.Series,
    scenario: str,
    strength: float,
    lag_days: int,
    smooth_window: int,
) -> np.ndarray:
    if scenario == "neutral" or strength <= 0:
        return preds_real

    monthly_std = historical_target.groupby(historical_target.index.month).std().fillna(0.0)
    month_weights = get_enso_month_weights()
    scenario_sign = -1.0 if scenario == "nino" else 1.0

    future_ts = pd.to_datetime(np.asarray(future_dates, dtype="datetime64[D]"))
    if lag_days != 0:
        future_ts = future_ts + pd.to_timedelta(lag_days, unit="D")
    months = future_ts.month.to_numpy()

    delta = np.asarray(
        [
            scenario_sign * strength * month_weights.get(int(m), 1.0) * float(monthly_std.get(int(m), 0.0))
            for m in months
        ],
        dtype=np.float32,
    )
    window = max(1, int(smooth_window))
    if window > 1 and len(delta) > 1:
        # Suavizado centrado para reducir saltos entre cambios mensuales.
        delta = pd.Series(delta).rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)

    adjusted = preds_real + delta
    return np.maximum(adjusted, 0.0)


def apply_variability_gain(preds_real: np.ndarray, gain: float, trend_window: int = 31) -> np.ndarray:
    gain = float(gain)
    if abs(gain - 1.0) < 1e-6:
        return preds_real
    if len(preds_real) <= 2:
        return np.maximum(preds_real, 0.0)

    # Amplifica/anula anomalias alrededor de la tendencia local.
    trend = pd.Series(preds_real).rolling(window=max(3, trend_window), center=True, min_periods=1).mean().to_numpy()
    adjusted = trend + gain * (preds_real - trend)
    return np.maximum(adjusted.astype(np.float32), 0.0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    target_tag = args.target_col.lower()
    model_target_tag = f"{args.model_type}_{target_tag}"
    artifacts_dir = Path(args.artifacts_dir)

    model_path = resolve_artifact_path(
        args.model_path,
        artifacts_dir / f"best_model_{model_target_tag}.pth",
        artifacts_dir / f"best_model_{args.model_type}.pth",
    )
    scaler_path = resolve_artifact_path(
        args.scaler_path,
        artifacts_dir / f"scaler_orinoco_{target_tag}.pkl",
        artifacts_dir / "scaler_orinoco.pkl",
    )
    feature_columns_path = resolve_artifact_path(
        args.feature_columns,
        artifacts_dir / f"feature_columns_{target_tag}.json",
        artifacts_dir / "feature_columns.json",
    )

    df = load_excel_dataset(args.excel_path, args.date_col, args.station_cols)
    df = impute_missing_values(df)
    df = add_cyclical_features(df)

    scaler = joblib.load(scaler_path)
    with open(feature_columns_path, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)["feature_columns"]

    if args.target_col not in feature_columns:
        raise ValueError(f"target-col {args.target_col} no esta en feature_columns")

    df = df[feature_columns]
    data_norm = scaler.transform(df.values)
    target_idx = feature_columns.index(args.target_col)
    n_features = data_norm.shape[1]

    state_dict = torch.load(model_path, map_location=device)
    checkpoint_horizon = infer_horizon_from_state_dict(state_dict)
    model_horizon = checkpoint_horizon if checkpoint_horizon is not None else args.horizon
    if checkpoint_horizon is not None and checkpoint_horizon != args.horizon:
        print(
            f"Aviso: checkpoint entrenado con horizon={checkpoint_horizon}. "
            f"Se usara ese valor en lugar de --horizon={args.horizon}."
        )

    model = build_model(args.model_type, input_size=n_features, horizon=model_horizon).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if len(data_norm) < args.lookback:
        raise ValueError("No hay suficientes filas para lookback")
    window = data_norm[-args.lookback :].copy()
    preds_norm_all = []
    last_date = df.index.values[-1]

    future_cyc, future_dates = generate_future_features(last_date=last_date, n_days=args.days)
    cyc_feature_names = ["dia_sin", "dia_cos", "mes_sin", "mes_cos", "temporada"]
    future_cyc_scaled = future_cyc.copy()
    for j, col_name in enumerate(cyc_feature_names):
        if col_name in feature_columns:
            col_idx = feature_columns.index(col_name)
            future_cyc_scaled[:, j] = scale_feature_values(scaler, future_cyc[:, j], col_idx)

    # Para exogenas (otras estaciones), evitamos congelarlas al ultimo valor:
    # usamos un perfil estacional por dia del anio para sostener el patron anual.
    future_exog_scaled: dict[str, np.ndarray] = {}
    future_doy = pd.to_datetime(np.asarray(future_dates, dtype="datetime64[D]")).dayofyear.to_numpy()
    exog_columns = [c for c in args.station_cols if c != args.target_col and c in feature_columns]
    for col_name in exog_columns:
        seasonal_profile = df[col_name].groupby(df.index.dayofyear).mean()
        seasonal_mean = float(seasonal_profile.mean())
        future_values = np.asarray(
            [float(seasonal_profile.get(int(doy), seasonal_mean)) for doy in future_doy], dtype=np.float32
        )
        col_idx = feature_columns.index(col_name)
        future_exog_scaled[col_name] = scale_feature_values(scaler, future_values, col_idx)

    generated = 0
    recursive_step = max(1, args.recursive_step)

    with torch.no_grad():
        while generated < args.days:
            x = torch.tensor(window[-args.lookback :], dtype=torch.float32).unsqueeze(0).to(device)
            pred_h = model(x).cpu().numpy().flatten()
            step = min(recursive_step, args.horizon, args.days - generated)
            take = pred_h[:step]
            preds_norm_all.extend(take.tolist())

            new_rows = np.tile(window[-1], (step, 1))
            new_rows[:, target_idx] = take

            cyc_slice = future_cyc_scaled[generated : generated + step]
            for col_name, values in zip(cyc_feature_names, cyc_slice.T):
                if col_name in feature_columns:
                    new_rows[:, feature_columns.index(col_name)] = values
            for col_name, values in future_exog_scaled.items():
                new_rows[:, feature_columns.index(col_name)] = values[generated : generated + step]

            window = np.vstack([window, new_rows])
            generated += step

    preds_norm = np.asarray(preds_norm_all, dtype=np.float32)
    preds_real = inverse_target(scaler, preds_norm, target_idx=target_idx, n_features=n_features)
    preds_real = apply_enso_adjustment(
        preds_real=preds_real,
        future_dates=future_dates,
        historical_target=df[args.target_col],
        scenario=args.enso_scenario,
        strength=float(args.enso_strength),
        lag_days=args.enso_lag_days,
        smooth_window=args.enso_smooth_window,
    )
    preds_real = apply_variability_gain(preds_real, gain=float(args.variability_gain))

    if args.output_csv:
        out = Path(args.output_csv)
    else:
        out = artifacts_dir / f"prediction_future_{model_target_tag}_{args.enso_scenario}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_data = np.column_stack([np.asarray(future_dates, dtype="datetime64[D]").astype(str), preds_real])
    np.savetxt(
        out,
        out_data,
        fmt="%s",
        delimiter=",",
        header=f"fecha,pred_{args.target_col}",
        comments="",
    )
    print(f"Prediccion guardada en {out}")


if __name__ == "__main__":
    main()
