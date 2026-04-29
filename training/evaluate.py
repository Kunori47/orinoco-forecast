import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models import LSTMForecaster, TransformerForecaster
from training.data_utils import (
    add_cyclical_features,
    create_windows,
    impute_missing_values,
    load_excel_dataset,
    temporal_split,
)
from training.metrics import summarize_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluacion Orinoco Forecast")
    parser.add_argument("--excel-path", type=str, required=True)
    parser.add_argument("--date-col", type=str, default="fecha")
    parser.add_argument("--station-cols", nargs="+", default=["ayacucho", "caicara", "ciudad_bolivar", "palua"])
    parser.add_argument("--target-col", type=str, default="ciudad_bolivar")
    parser.add_argument("--model", type=str, choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--lookback", type=int, default=90)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def resolve_artifact_path(new_path: Path, legacy_path: Path) -> Path:
    if new_path.exists():
        return new_path
    return legacy_path


def build_model(model_name: str, input_size: int, horizon: int) -> torch.nn.Module:
    if model_name == "lstm":
        return LSTMForecaster(input_size=input_size, forecast_horizon=horizon)
    return TransformerForecaster(input_size=input_size, forecast_horizon=horizon)


def infer_horizon_from_state_dict(state_dict: dict) -> int | None:
    if "head.bias" in state_dict:
        return int(state_dict["head.bias"].shape[0])
    if "head.weight" in state_dict:
        return int(state_dict["head.weight"].shape[0])
    return None


def inverse_target(scaler, arr_2d: np.ndarray, target_idx: int, n_features: int) -> np.ndarray:
    rows, horizon = arr_2d.shape
    flat = arr_2d.reshape(-1, 1)
    helper = np.zeros((flat.shape[0], n_features), dtype=np.float32)
    helper[:, target_idx] = flat[:, 0]
    inv = scaler.inverse_transform(helper)[:, target_idx]
    return inv.reshape(rows, horizon)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    artifacts = Path(args.artifacts_dir)
    figures = artifacts / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    target_tag = args.target_col.lower()
    model_target_tag = f"{args.model}_{target_tag}"

    scaler_path = resolve_artifact_path(
        artifacts / f"scaler_orinoco_{target_tag}.pkl",
        artifacts / "scaler_orinoco.pkl",
    )
    feature_columns_path = resolve_artifact_path(
        artifacts / f"feature_columns_{target_tag}.json",
        artifacts / "feature_columns.json",
    )
    model_path = resolve_artifact_path(
        artifacts / f"best_model_{model_target_tag}.pth",
        artifacts / f"best_model_{args.model}.pth",
    )

    scaler = __import__("joblib").load(scaler_path)
    with open(feature_columns_path, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)["feature_columns"]

    df = load_excel_dataset(args.excel_path, args.date_col, args.station_cols)
    df = impute_missing_values(df)
    df = add_cyclical_features(df)
    split_data = temporal_split(df, train_ratio=0.7, val_ratio=0.15)

    test_norm = scaler.transform(split_data.test_df[feature_columns].values)
    target_idx = feature_columns.index(args.target_col)
    X_test, y_test = create_windows(test_norm, target_idx, args.lookback, args.horizon, stride=1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    state_dict = torch.load(model_path, map_location=device)
    checkpoint_horizon = infer_horizon_from_state_dict(state_dict)
    model_horizon = checkpoint_horizon if checkpoint_horizon is not None else args.horizon
    if checkpoint_horizon is not None and checkpoint_horizon != args.horizon:
        print(
            f"Aviso: checkpoint entrenado con horizon={checkpoint_horizon}. "
            f"Se usara ese valor en lugar de --horizon={args.horizon}."
        )

    model = build_model(args.model, input_size=X_test.shape[-1], horizon=model_horizon).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        preds_norm = model(X_test_t).cpu().numpy()

    y_pred = inverse_target(scaler, preds_norm, target_idx, n_features=X_test.shape[-1])
    y_true = inverse_target(scaler, y_test, target_idx, n_features=X_test.shape[-1])
    metrics = summarize_metrics(y_true.flatten(), y_pred.flatten())

    with open(artifacts / f"metrics_{model_target_tag}_eval.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=True, indent=2)

    horizon_error = np.mean(np.abs(y_true - y_pred), axis=0)
    plt.figure(figsize=(8, 4))
    plt.plot(horizon_error)
    plt.title(f"Error absoluto por dia de horizonte ({args.model.upper()} - {args.target_col})")
    plt.xlabel("Dia futuro")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(figures / f"horizon_error_{model_target_tag}.png", dpi=150)
    plt.close()

    idx = np.arange(min(300, y_true.size))
    plt.figure(figsize=(10, 4))
    plt.plot(idx, y_true.flatten()[: len(idx)], label="Real")
    plt.plot(idx, y_pred.flatten()[: len(idx)], label="Predicho")
    plt.title(f"Prediccion vs Real ({args.model.upper()} - {args.target_col})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / f"pred_vs_real_{model_target_tag}.png", dpi=150)
    plt.close()

    print("Metricas de evaluacion:")
    for k, v in metrics.items():
        print(f"- {k}: {v:.6f}")


if __name__ == "__main__":
    main()
