import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models import LSTMForecaster, TransformerForecaster
from training.data_utils import (
    add_cyclical_features,
    create_windows,
    detect_outliers_iqr_by_month,
    impute_missing_values,
    load_excel_dataset,
    make_dataloaders,
    run_fft_analysis,
    run_stl_decomposition,
    save_feature_columns,
    scale_splits,
    temporal_split,
)
from training.metrics import summarize_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento Orinoco Forecast")
    parser.add_argument("--excel-path", type=str, required=True)
    parser.add_argument("--date-col", type=str, default="fecha")
    parser.add_argument(
        "--station-cols",
        nargs="+",
        default=["ayacucho", "caicara", "ciudad_bolivar", "palua"],
        help="Columnas de estaciones en el excel",
    )
    parser.add_argument("--target-col", type=str, default="ciudad_bolivar")
    parser.add_argument("--model", type=str, choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--lookback", type=int, default=90)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_model(args: argparse.Namespace, input_size: int) -> torch.nn.Module:
    if args.model == "lstm":
        return LSTMForecaster(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.lstm_layers,
            forecast_horizon=args.horizon,
            dropout=args.dropout,
        )
    return TransformerForecaster(
        input_size=input_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.transformer_layers,
        dim_feedforward=args.dim_feedforward,
        forecast_horizon=args.horizon,
        dropout=min(args.dropout, 0.1),
    )


def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    num_epochs: int,
    patience: int,
    lr: float,
    best_model_path: Path,
) -> tuple[torch.nn.Module, dict[str, list[float]]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_losses.append(criterion(model(X_batch), y_batch).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        print(f"[{epoch:03d}/{num_epochs}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping en epoch {epoch}")
                break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model, history


def predict_loader(model: torch.nn.Module, loader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            preds.append(model(X_batch).cpu().numpy())
    return np.concatenate(preds, axis=0)


def inverse_target(scaler, arr_2d: np.ndarray, target_idx: int, n_features: int) -> np.ndarray:
    rows, horizon = arr_2d.shape
    flat = arr_2d.reshape(-1, 1)
    helper = np.zeros((flat.shape[0], n_features), dtype=np.float32)
    helper[:, target_idx] = flat[:, 0]
    inv = scaler.inverse_transform(helper)[:, target_idx]
    return inv.reshape(rows, horizon)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    target_tag = args.target_col.lower()
    model_target_tag = f"{args.model}_{target_tag}"
    best_model_path = artifacts_dir / f"best_model_{model_target_tag}.pth"
    scaler_path = artifacts_dir / f"scaler_orinoco_{target_tag}.pkl"
    feature_columns_path = artifacts_dir / f"feature_columns_{target_tag}.json"

    df = load_excel_dataset(args.excel_path, args.date_col, args.station_cols)
    df = impute_missing_values(df)
    df = add_cyclical_features(df)

    fft_df = run_fft_analysis(df[args.target_col])
    fft_df.head(20).to_csv(artifacts_dir / f"fft_top_frequencies_{target_tag}.csv", index=False)

    stl = run_stl_decomposition(df[args.target_col], period=365)
    stl_df = np.column_stack([stl.trend, stl.seasonal, stl.resid])
    np.save(artifacts_dir / f"stl_components_{target_tag}.npy", stl_df)

    outlier_flags = detect_outliers_iqr_by_month(df[args.target_col])
    outlier_flags.to_frame(name="is_outlier").to_csv(artifacts_dir / f"outliers_target_{target_tag}.csv")

    split_data = temporal_split(df, train_ratio=0.7, val_ratio=0.15)
    train_norm, val_norm, test_norm, scaler = scale_splits(split_data, scaler_path)

    target_idx = split_data.feature_columns.index(args.target_col)
    X_train, y_train = create_windows(train_norm, target_idx, args.lookback, args.horizon, args.stride)
    X_val, y_val = create_windows(val_norm, target_idx, args.lookback, args.horizon, args.stride)
    X_test, y_test = create_windows(test_norm, target_idx, args.lookback, args.horizon, args.stride)

    if min(len(X_train), len(X_val), len(X_test)) == 0:
        raise ValueError(
            "No se pudieron construir ventanas suficientes. Prueba con menor lookback/horizon o mas datos."
        )

    train_loader, val_loader, test_loader = make_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=args.batch_size
    )

    model = pick_model(args, input_size=X_train.shape[-1]).to(device)
    start = time.perf_counter()
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        best_model_path=best_model_path,
    )
    elapsed = time.perf_counter() - start

    preds_norm = predict_loader(model, test_loader, device=device)
    y_test_true = y_test
    preds = inverse_target(scaler, preds_norm, target_idx=target_idx, n_features=X_train.shape[-1])
    y_true = inverse_target(scaler, y_test_true, target_idx=target_idx, n_features=X_train.shape[-1])

    metrics = summarize_metrics(y_true.flatten(), preds.flatten())
    metrics["training_seconds"] = float(elapsed)
    metrics["num_params"] = int(sum(p.numel() for p in model.parameters()))
    metrics["target_col"] = args.target_col
    metrics["model"] = args.model
    metrics["artifact_tag"] = model_target_tag

    save_feature_columns(split_data.feature_columns, feature_columns_path)
    np.save(artifacts_dir / f"y_true_{model_target_tag}.npy", y_true)
    np.save(artifacts_dir / f"y_pred_{model_target_tag}.npy", preds)

    with open(artifacts_dir / f"history_{model_target_tag}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=True, indent=2)

    with open(artifacts_dir / f"metrics_{model_target_tag}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=True, indent=2)

    print("\nMetricas test:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"- {k}: {v:.6f}")
        else:
            print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
