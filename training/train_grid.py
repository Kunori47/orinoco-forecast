import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena todas las combinaciones ciudad x modelo.")
    parser.add_argument("--excel-path", type=str, required=True)
    parser.add_argument("--date-col", type=str, default="fecha")
    parser.add_argument("--station-cols", nargs="+", default=["ayacucho", "caicara", "ciudad_bolivar", "palua"])
    parser.add_argument("--target-cols", nargs="*", default=[])
    parser.add_argument("--models", nargs="+", default=["lstm", "transformer", "arima", "random_forest"])
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
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=20)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument("--arima-p", type=int, default=3)
    parser.add_argument("--arima-d", type=int, default=1)
    parser.add_argument("--arima-q", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    return parser.parse_args()


def build_base_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "training.train",
        "--excel-path",
        args.excel_path,
        "--date-col",
        args.date_col,
        "--station-cols",
        *args.station_cols,
        "--lookback",
        str(args.lookback),
        "--horizon",
        str(args.horizon),
        "--stride",
        str(args.stride),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--lr",
        str(args.lr),
        "--hidden-size",
        str(args.hidden_size),
        "--lstm-layers",
        str(args.lstm_layers),
        "--dropout",
        str(args.dropout),
        "--d-model",
        str(args.d_model),
        "--nhead",
        str(args.nhead),
        "--transformer-layers",
        str(args.transformer_layers),
        "--dim-feedforward",
        str(args.dim_feedforward),
        "--rf-n-estimators",
        str(args.rf_n_estimators),
        "--rf-max-depth",
        str(args.rf_max_depth),
        "--rf-min-samples-leaf",
        str(args.rf_min_samples_leaf),
        "--arima-p",
        str(args.arima_p),
        "--arima-d",
        str(args.arima_d),
        "--arima-q",
        str(args.arima_q),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--artifacts-dir",
        args.artifacts_dir,
    ]


def main() -> None:
    args = parse_args()
    excel_path = Path(args.excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Excel no encontrado: '{args.excel_path}'. "
            "Pasa una ruta valida con --excel-path (ej: data/raw/tu_archivo.xlsx)."
        )
    targets = args.target_cols if args.target_cols else args.station_cols
    base_cmd = build_base_command(args)
    total_runs = len(targets) * len(args.models)
    run_idx = 0
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(project_root)

    for target in targets:
        for model in args.models:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] Entrenando target={target}, model={model}")
            cmd = base_cmd + ["--target-col", target, "--model", model]
            subprocess.run(cmd, check=True, cwd=project_root, env=env)

    print("\nEntrenamiento de grilla completado.")


if __name__ == "__main__":
    main()
