import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from training.data_utils import (
    add_cyclical_features,
    detect_outliers_iqr_by_month,
    impute_missing_values,
    load_excel_dataset,
    run_fft_analysis,
    run_stl_decomposition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exploracion de datos Orinoco")
    parser.add_argument("--excel-path", type=str, required=True)
    parser.add_argument("--date-col", type=str, default="fecha")
    parser.add_argument("--station-cols", nargs="+", default=["ayacucho", "caicara", "ciudad_bolivar", "palua"])
    parser.add_argument("--target-col", type=str, default="ciudad_bolivar")
    parser.add_argument("--output-dir", type=str, default="artifacts/eda")
    parser.add_argument("--acf-lags", type=int, default=365)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_excel_dataset(args.excel_path, args.date_col, args.station_cols)
    null_report = df.isna().sum().to_frame(name="null_count")
    null_report.to_csv(out / "null_report.csv")

    df = impute_missing_values(df)
    df = add_cyclical_features(df)

    # Boxplot mensual por estacion.
    monthly = df[args.station_cols].copy()
    monthly["month"] = monthly.index.month
    melted = monthly.melt(id_vars="month", var_name="station", value_name="level")
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=melted, x="month", y="level", hue="station")
    plt.title("Boxplot mensual por estacion")
    plt.tight_layout()
    plt.savefig(out / "boxplot_mensual_estaciones.png", dpi=150)
    plt.close()

    # Correlacion entre estaciones.
    plt.figure(figsize=(6, 5))
    sns.heatmap(df[args.station_cols].corr(), annot=True, cmap="viridis")
    plt.title("Heatmap de correlacion entre estaciones")
    plt.tight_layout()
    plt.savefig(out / "heatmap_correlacion_estaciones.png", dpi=150)
    plt.close()

    # Serie temporal completa y outliers.
    outliers = detect_outliers_iqr_by_month(df[args.target_col])
    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df[args.target_col], label=args.target_col)
    plt.scatter(df.index[outliers], df.loc[outliers, args.target_col], color="red", s=10, label="Outlier IQR")
    plt.legend()
    plt.title("Serie completa con marcadores de extremos")
    plt.tight_layout()
    plt.savefig(out / "serie_completa_con_extremos.png", dpi=150)
    plt.close()

    # ACF y PACF del target.
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(df[args.target_col], lags=args.acf_lags, ax=axs[0])
    axs[0].set_title("ACF")
    plot_pacf(df[args.target_col], lags=min(120, args.acf_lags), ax=axs[1], method="ywm")
    axs[1].set_title("PACF")
    plt.tight_layout()
    plt.savefig(out / "acf_pacf_target.png", dpi=150)
    plt.close()

    # FFT y STL.
    fft_df = run_fft_analysis(df[args.target_col])
    fft_df.to_csv(out / "fft_dominant_frequencies.csv", index=False)

    stl = run_stl_decomposition(df[args.target_col], period=365)
    stl_df = pd.DataFrame(
        {
            "trend": stl.trend,
            "seasonal": stl.seasonal,
            "resid": stl.resid,
        },
        index=df.index,
    )
    stl_df.to_csv(out / "stl_components.csv")

    print(f"EDA completado. Resultados en: {out}")


if __name__ == "__main__":
    main()
