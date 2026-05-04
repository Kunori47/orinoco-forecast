import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye tabla comparativa de modelos por ciudad.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--output-csv", type=str, default="artifacts/model_comparison.csv")
    parser.add_argument("--output-markdown", type=str, default="artifacts/model_comparison.md")
    parser.add_argument("--models", nargs="+", default=["lstm", "transformer", "arima", "random_forest"])
    parser.add_argument(
        "--sort-by",
        type=str,
        choices=["best_rmse", "best_mae", "best_mape", "best_nse", "target_col"],
        default="best_rmse",
        help="Criterio de orden de la tabla final.",
    )
    return parser.parse_args()


def parse_metric_file_name(path: Path, models: list[str]) -> tuple[str, str] | None:
    # Espera nombres como:
    # - metrics_lstm_ciudad_bolivar_eval.json
    # - metrics_transformer_ciudad_bolivar_eval.json
    # - metrics_lstm_ciudad_bolivar.json (fallback entrenamiento)
    # Ignora legacy sin target explicito (ej: metrics_lstm.json).
    stem = path.stem  # metrics_...
    if not stem.startswith("metrics_"):
        return None
    tag = stem[len("metrics_") :]
    if tag.endswith("_eval"):
        tag = tag[: -len("_eval")]
    for model in models:
        prefix = f"{model}_"
        if tag.startswith(prefix):
            target = tag[len(prefix) :]
            if target:
                return model, target
    return None


def load_metric_rows(artifacts_dir: Path, models: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(artifacts_dir.glob("metrics_*.json")):
        parsed = parse_metric_file_name(path, models)
        if parsed is None:
            continue
        model, target = parsed
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not all(k in data for k in ("mae", "rmse", "mape", "nse")):
            continue
        target = str(data.get("target_col", target))
        rows.append(
            {
                "target_col": target,
                "model": model,
                "mae": float(data["mae"]),
                "rmse": float(data["rmse"]),
                "mape": float(data["mape"]),
                "nse": float(data["nse"]),
                "source_file": str(path),
            }
        )
    return rows


def _add_best_model_columns(out: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rmse_cols = [f"rmse_{m}" for m in models if f"rmse_{m}" in out.columns]
    mae_cols = [f"mae_{m}" for m in models if f"mae_{m}" in out.columns]
    mape_cols = [f"mape_{m}" for m in models if f"mape_{m}" in out.columns]
    nse_cols = [f"nse_{m}" for m in models if f"nse_{m}" in out.columns]

    if rmse_cols:
        out["best_rmse"] = out[rmse_cols].min(axis=1)
        out["best_model_rmse"] = out[rmse_cols].idxmin(axis=1).str.replace("rmse_", "", regex=False)
    if mae_cols:
        out["best_mae"] = out[mae_cols].min(axis=1)
        out["best_model_mae"] = out[mae_cols].idxmin(axis=1).str.replace("mae_", "", regex=False)
    if mape_cols:
        out["best_mape"] = out[mape_cols].min(axis=1)
        out["best_model_mape"] = out[mape_cols].idxmin(axis=1).str.replace("mape_", "", regex=False)
    if nse_cols:
        out["best_nse"] = out[nse_cols].max(axis=1)
        out["best_model_nse"] = out[nse_cols].idxmax(axis=1).str.replace("nse_", "", regex=False)
    return out


def build_comparison_table(df_long: pd.DataFrame, models: list[str], sort_by: str) -> pd.DataFrame:
    pivot = df_long.pivot_table(index="target_col", columns="model", values=["mae", "rmse", "mape", "nse"], aggfunc="first")
    pivot.columns = [f"{metric}_{model}" for metric, model in pivot.columns]
    out = pivot.reset_index()

    if "mae_transformer" in out.columns and "mae_lstm" in out.columns:
        out["delta_mae_transformer_minus_lstm"] = out["mae_transformer"] - out["mae_lstm"]
    if "rmse_transformer" in out.columns and "rmse_lstm" in out.columns:
        out["delta_rmse_transformer_minus_lstm"] = out["rmse_transformer"] - out["rmse_lstm"]
    if "mape_transformer" in out.columns and "mape_lstm" in out.columns:
        out["delta_mape_transformer_minus_lstm"] = out["mape_transformer"] - out["mape_lstm"]
    if "nse_transformer" in out.columns and "nse_lstm" in out.columns:
        out["delta_nse_transformer_minus_lstm"] = out["nse_transformer"] - out["nse_lstm"]
    out = _add_best_model_columns(out, models=models)

    if sort_by in out.columns:
        ascending = sort_by != "best_nse"
        out = out.sort_values(sort_by, ascending=ascending, na_position="last")
    else:
        out = out.sort_values("target_col")
    return out.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    rows = load_metric_rows(artifacts_dir, models=list(args.models))
    if not rows:
        raise ValueError(
            "No se encontraron metricas validas. Ejecuta training/evaluate.py para generar metrics_*_eval.json."
        )

    df_long = pd.DataFrame(rows).sort_values(["target_col", "model"]).reset_index(drop=True)
    df_comp = build_comparison_table(df_long, models=list(args.models), sort_by=args.sort_by)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(output_csv, index=False)

    output_md = Path(args.output_markdown)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# Comparativa de modelos\n\n")
        f.write("```csv\n")
        f.write(df_comp.to_csv(index=False))
        f.write("```\n")
        f.write("\n")

    print(f"Comparativa guardada en: {output_csv}")
    print(f"Tabla markdown guardada en: {output_md}")


if __name__ == "__main__":
    main()
