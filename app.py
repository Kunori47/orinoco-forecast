import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Prediccion Rio Orinoco", layout="wide")
st.title("Prediccion Rio Orinoco")

st.markdown(
    "Visualiza predicciones futuras por ciudad, modelo y escenario ENSO (`neutral`, `nino`, `nina`)."
)

ARTIFACTS_DIR = "artifacts"
EXCEL_PATH = "data/raw/dataset-orinoco-imputado.xlsx"
DATE_COL = "fecha"
STATION_COLS = ["ayacucho", "caicara", "ciudad_bolivar", "palua"]


def run_python_module(app_dir: Path, module_name: str, module_args: list[str]) -> subprocess.CompletedProcess:
    env = dict(**os.environ)
    env["PYTHONPATH"] = f"{app_dir}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(app_dir)
    cmd = [sys.executable, "-m", module_name, *module_args]
    return subprocess.run(cmd, cwd=app_dir, env=env, capture_output=True, text=True)


def show_model_comparison_table(artifacts_dir: str) -> None:
    path = Path(artifacts_dir) / "model_comparison.csv"
    if not path.exists():
        st.info("Aun no existe `model_comparison.csv`. Ejecuta el boton de evaluacion.")
        return
    df = pd.read_csv(path)
    st.subheader("Tabla unica de comparativa (LSTM vs Transformer)")
    st.dataframe(df, use_container_width=True)


def build_model_scenarios_df(artifacts_dir: str, model_type: str, target_col: str, scenarios: list[str]):
    artifacts_path = Path(artifacts_dir)
    plot_df = None
    loaded_paths = []

    for scenario in scenarios:
        csv_path = artifacts_path / f"prediction_future_{model_type}_{target_col}_{scenario}.csv"
        if not csv_path.exists():
            st.warning(f"No existe: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if "fecha" not in df.columns:
            st.warning(f"CSV sin columna 'fecha': {csv_path}")
            continue

        pred_cols = [c for c in df.columns if c.startswith("pred_")]
        if not pred_cols:
            st.warning(f"CSV sin columna predicha: {csv_path}")
            continue

        pred_col = pred_cols[0]
        series_df = df[["fecha", pred_col]].copy()
        series_df["fecha"] = pd.to_datetime(series_df["fecha"], errors="coerce")
        series_df = series_df.dropna(subset=["fecha"]).set_index("fecha")
        series_df = series_df.rename(columns={pred_col: scenario})

        loaded_paths.append(str(csv_path))
        if plot_df is None:
            plot_df = series_df
        else:
            plot_df = plot_df.join(series_df, how="outer")
    return plot_df, loaded_paths


def load_scenarios_plot(artifacts_dir: str, model_type: str, target_col: str, scenarios: list[str]) -> None:
    if not scenarios:
        st.warning("Selecciona al menos un escenario.")
        return

    plot_df, loaded_paths = build_model_scenarios_df(artifacts_dir, model_type, target_col, scenarios)

    if plot_df is None or plot_df.empty:
        st.error("No se cargaron series validas para graficar.")
    else:
        plot_df = plot_df.sort_index()
        st.success("Series cargadas:\n- " + "\n- ".join(loaded_paths))
        st.line_chart(plot_df)
        st.dataframe(plot_df.tail(30))


def load_dual_model_comparison(artifacts_dir: str, target_col: str, scenarios: list[str]) -> None:
    if not scenarios:
        st.warning("Selecciona al menos un escenario.")
        return

    model_dfs: dict[str, pd.DataFrame] = {}
    missing_models = []
    for model in ["lstm", "transformer"]:
        df_model, _ = build_model_scenarios_df(artifacts_dir, model, target_col, scenarios)
        if df_model is None or df_model.empty:
            missing_models.append(model)
        else:
            model_dfs[model] = df_model.sort_index()

    if len(model_dfs) == 0:
        st.error("No hay datos para comparar entre modelos.")
        return

    st.subheader("Comparacion visual por modelo")
    c1, c2 = st.columns(2)
    if "lstm" in model_dfs:
        with c1:
            st.markdown("**LSTM**")
            st.line_chart(model_dfs["lstm"])
    else:
        with c1:
            st.warning("Sin datos LSTM.")

    if "transformer" in model_dfs:
        with c2:
            st.markdown("**Transformer**")
            st.line_chart(model_dfs["transformer"])
    else:
        with c2:
            st.warning("Sin datos Transformer.")

    if "lstm" in model_dfs and "transformer" in model_dfs:
        st.subheader("Comparacion superpuesta (LSTM vs Transformer)")
        overlay = None
        for scenario in scenarios:
            cols = []
            for model in ["lstm", "transformer"]:
                if scenario in model_dfs[model].columns:
                    cols.append(model_dfs[model][[scenario]].rename(columns={scenario: f"{model}_{scenario}"}))
            if cols:
                merged = cols[0]
                for d in cols[1:]:
                    merged = merged.join(d, how="outer")
                overlay = merged if overlay is None else overlay.join(merged, how="outer")

        if overlay is not None and not overlay.empty:
            st.line_chart(overlay.sort_index())
            st.dataframe(overlay.sort_index().tail(30))

    if missing_models:
        st.info("Modelos sin datos para esta combinacion: " + ", ".join(missing_models))


st.caption(
    f"Configuracion fija: artifacts={ARTIFACTS_DIR}, excel={EXCEL_PATH}, "
    f"fecha={DATE_COL}, estaciones={','.join(STATION_COLS)}"
)

view_mode = st.radio(
    "Modo de visualizacion",
    options=["Modelo unico", "Comparar LSTM vs Transformer"],
    horizontal=True,
)
model_type = st.selectbox("Modelo", options=["lstm", "transformer"], index=0)
target_col = st.selectbox(
    "Ciudad / serie objetivo",
    options=["ciudad_bolivar", "ayacucho", "caicara", "palua"],
    index=0,
)
scenarios = st.multiselect(
    "Escenarios ENSO a comparar",
    options=["neutral", "nino", "nina"],
    default=["neutral", "nino", "nina"],
)
days = st.number_input("Dias a predecir", min_value=1, max_value=3650, value=730, step=1)
lookback = st.number_input("Lookback", min_value=7, max_value=730, value=90, step=1)
horizon = st.number_input("Horizon", min_value=1, max_value=365, value=30, step=1)
recursive_step = st.number_input("Recursive step", min_value=1, max_value=60, value=1, step=1)
enso_smooth_window = st.number_input("Suavizado ENSO (dias)", min_value=1, max_value=120, value=21, step=1)
variability_gain = st.number_input("Ganancia de variabilidad", min_value=0.5, max_value=2.0, value=1.0, step=0.05)

col1, col2 = st.columns(2)
with col1:
    nino_strength = st.number_input("Intensidad El Nino", min_value=0.0, max_value=3.0, value=0.7, step=0.1)
    nino_lag = st.number_input("Lag El Nino (dias)", min_value=-365, max_value=365, value=12, step=1)
with col2:
    nina_strength = st.number_input("Intensidad La Nina", min_value=0.0, max_value=3.0, value=0.7, step=0.1)
    nina_lag = st.number_input("Lag La Nina (dias)", min_value=-365, max_value=365, value=-10, step=1)

if st.button("Generar escenarios ahora"):
    if not scenarios:
        st.warning("Selecciona al menos un escenario para generar.")
    else:
        app_dir = Path(__file__).resolve().parent
        generation_errors = False

        models_to_generate = ["lstm", "transformer"] if view_mode == "Comparar LSTM vs Transformer" else [model_type]
        for model_name in models_to_generate:
            for scenario in scenarios:
                module_args = [
                    "--excel-path",
                    EXCEL_PATH,
                    "--date-col",
                    DATE_COL,
                    "--target-col",
                    target_col,
                    "--station-cols",
                    *STATION_COLS,
                    "--model-type",
                    model_name,
                    "--lookback",
                    str(int(lookback)),
                    "--horizon",
                    str(int(horizon)),
                    "--days",
                    str(int(days)),
                    "--recursive-step",
                    str(int(recursive_step)),
                    "--enso-scenario",
                    scenario,
                    "--enso-smooth-window",
                    str(int(enso_smooth_window)),
                    "--variability-gain",
                    str(float(variability_gain)),
                    "--artifacts-dir",
                    ARTIFACTS_DIR,
                ]
                if scenario == "nino":
                    module_args.extend(
                        ["--enso-strength", str(float(nino_strength)), "--enso-lag-days", str(int(nino_lag))]
                    )
                elif scenario == "nina":
                    module_args.extend(
                        ["--enso-strength", str(float(nina_strength)), "--enso-lag-days", str(int(nina_lag))]
                    )

                run = run_python_module(app_dir, "inference.predict", module_args)
                if run.returncode == 0:
                    st.success(f"Escenario generado: modelo={model_name}, escenario={scenario}")
                else:
                    generation_errors = True
                    st.error(f"Fallo en modelo={model_name}, escenario={scenario}")
                    st.code(run.stderr or run.stdout)

        if not generation_errors:
            st.info("Generacion completada. Mostrando comparacion actualizada.")
            if view_mode == "Comparar LSTM vs Transformer":
                load_dual_model_comparison(ARTIFACTS_DIR, target_col, scenarios)
            else:
                load_scenarios_plot(ARTIFACTS_DIR, model_type, target_col, scenarios)

st.divider()
st.subheader("Evaluacion y tabla comparativa")
if st.button("Evaluar modelos y actualizar tabla unica"):
    app_dir = Path(__file__).resolve().parent
    eval_errors = False

    for model_name in ["lstm", "transformer"]:
        eval_args = [
            "--excel-path",
            EXCEL_PATH,
            "--date-col",
            DATE_COL,
            "--station-cols",
            *STATION_COLS,
            "--target-col",
            target_col,
            "--model",
            model_name,
            "--lookback",
            str(int(lookback)),
            "--horizon",
            str(int(horizon)),
            "--artifacts-dir",
            ARTIFACTS_DIR,
        ]
        run_eval = run_python_module(app_dir, "training.evaluate", eval_args)
        if run_eval.returncode == 0:
            st.success(f"Evaluacion completada: {model_name}")
        else:
            eval_errors = True
            st.error(f"Fallo evaluacion: {model_name}")
            st.code(run_eval.stderr or run_eval.stdout)

    if not eval_errors:
        compare_args = [
            "--artifacts-dir",
            ARTIFACTS_DIR,
            "--output-csv",
            f"{ARTIFACTS_DIR}/model_comparison.csv",
            "--output-markdown",
            f"{ARTIFACTS_DIR}/model_comparison.md",
        ]
        run_compare = run_python_module(app_dir, "training.compare_models", compare_args)
        if run_compare.returncode == 0:
            st.success("Tabla unica actualizada automaticamente.")
            show_model_comparison_table(ARTIFACTS_DIR)
        else:
            st.error("Fallo al generar tabla unica.")
            st.code(run_compare.stderr or run_compare.stdout)

if st.button("Ver tabla unica actual"):
    show_model_comparison_table(ARTIFACTS_DIR)

if st.button("Cargar comparacion"):
    if view_mode == "Comparar LSTM vs Transformer":
        load_dual_model_comparison(ARTIFACTS_DIR, target_col, scenarios)
    else:
        load_scenarios_plot(ARTIFACTS_DIR, model_type, target_col, scenarios)
