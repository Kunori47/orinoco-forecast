# Orinoco Forecast (PyTorch)

Proyecto de prediccion hidrologica del Rio Orinoco usando series temporales multivariadas y comparacion de modelos LSTM vs Transformer.

## Objetivo

Predecir el nivel del rio para un horizonte de 30 dias (ampliable), usando:

- Datos de estaciones: Ayacucho, Caicara, Ciudad Bolivar y Palua.
- Features ciclicas (`sin/cos`) para dia del ano y mes.
- Split temporal sin aleatorizacion.
- Entrenamiento y evaluacion comparativa en PyTorch.

## Estructura

```text
orinoco-forecast/
├── app.py
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
├── models/
│   ├── __init__.py
│   ├── lstm.py
│   └── transformer.py
├── training/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── evaluate.py
│   ├── metrics.py
│   └── train.py
├── inference/
│   ├── __init__.py
│   └── predict.py
└── notebooks/
```

## Instalacion

```bash
cd /home/kunori/Projects/orinoco-forecast
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Formato esperado del dataset

Archivo Excel en `data/raw/` con una columna de fecha y columnas de nivel por estacion. Por defecto:

- `fecha`
- `ayacucho`
- `caicara`
- `ciudad_bolivar`
- `palua`

Puedes sobrescribir los nombres por CLI en el entrenamiento.

## Entrenamiento

Ejemplo con LSTM:

```bash
python training/train.py \
  --excel-path data/raw/orinoco.xlsx \
  --date-col fecha \
  --target-col ciudad_bolivar \
  --station-cols ayacucho caicara ciudad_bolivar palua \
  --model lstm \
  --lookback 90 \
  --horizon 30 \
  --epochs 100
```

Ejemplo con Transformer:

```bash
python training/train.py \
  --excel-path data/raw/orinoco.xlsx \
  --date-col fecha \
  --target-col ciudad_bolivar \
  --station-cols ayacucho caicara ciudad_bolivar palua \
  --model transformer
```

Parametros destacados de entrenamiento:

- `--target-col`: ciudad/serie objetivo a predecir.
- `--model`: `lstm` o `transformer`.
- `--artifacts-dir`: carpeta de salida de artefactos (default `artifacts`).

Entrenamiento de todas las ciudades x todos los modelos:

```bash
python training/train_grid.py \
  --excel-path data/raw/orinoco.xlsx \
  --station-cols ayacucho caicara ciudad_bolivar palua \
  --models lstm transformer
```

Tabla unica de comparativa LSTM vs Transformer:

```bash
PYTHONPATH=. .venv/bin/python training/compare_models.py \
  --artifacts-dir artifacts \
  --output-csv artifacts/model_comparison.csv \
  --output-markdown artifacts/model_comparison.md
```

Salida:

- `artifacts/model_comparison.csv`
- `artifacts/model_comparison.md`

Los artefactos se guardan en `artifacts/`:

- `best_model_<modelo>_<ciudad>.pth` (ej. `best_model_lstm_ciudad_bolivar.pth`)
- `scaler_orinoco_<ciudad>.pkl`
- `feature_columns_<ciudad>.json`
- `metrics_<modelo>_<ciudad>.json`

Esto evita sobreescritura cuando entrenas varias ciudades y/o ambos modelos.

## Evaluacion

```bash
python training/evaluate.py \
  --excel-path data/raw/orinoco.xlsx \
  --date-col fecha \
  --target-col ciudad_bolivar \
  --station-cols ayacucho caicara ciudad_bolivar palua \
  --model lstm
```

Genera metricas (MAE, RMSE, MAPE, NSE) y graficas en `artifacts/figures/`.
Lee automaticamente artefactos por ciudad+modelo (si no existen, usa los legacy).

## Inferencia

```bash
python inference/predict.py \
  --excel-path data/raw/orinoco.xlsx \
  --date-col fecha \
  --target-col ciudad_bolivar \
  --station-cols ayacucho caicara ciudad_bolivar palua \
  --model-type lstm \
  --days 60 \
  --enso-scenario neutral
```

Escenarios ENSO explicitos:

- `--enso-scenario neutral` (sin ajuste)
- `--enso-scenario nino --enso-strength 0.7 --enso-lag-days 12`
- `--enso-scenario nina --enso-strength 0.7 --enso-lag-days -10`

Parametros nuevos de inferencia:

- `--recursive-step`: pasos consumidos por iteracion autoregresiva (recomendado `1` para menos lag/saltos).
- `--enso-scenario`: `neutral`, `nino`, `nina`.
- `--enso-strength`: intensidad del ajuste ENSO.
- `--enso-lag-days`: desfase del efecto ENSO en dias.
- `--enso-smooth-window`: suavizado (dias) del ajuste ENSO para evitar picos artificiales.
- `--variability-gain`: ganancia de variabilidad final (`1.0` sin cambio, `>1` mas picos, `<1` mas plano).
- `--artifacts-dir`: carpeta base para resolver artefactos por ciudad/modelo.
- `--model-path`, `--scaler-path`, `--feature-columns`: opcionales; si no se pasan, se resuelven automaticamente.
- `--output-csv`: opcional; si no se pasa, se genera nombre automatico.

Salida por defecto de inferencia:

- `prediction_future_<modelo>_<ciudad>_<escenario>.csv`

## Dashboard (opcional)

```bash
streamlit run app.py
```

La app permite:

- comparar en una sola grafica varios escenarios ENSO para una ciudad y modelo.
- generar escenarios desde la UI con el boton `Generar escenarios ahora` (sin ejecutar comandos manuales).
- ajustar desde selectores: modelo, ciudad, dias, lookback, horizon, recursive step, intensidades y lag ENSO.
- usar configuracion fija de datos base (excel, fecha, estaciones y artifacts), sin campos editables para esos valores.
- alternar `Modo de visualizacion` para comparar LSTM vs Transformer con graficas lado a lado y superpuestas.
- ejecutar `Evaluar modelos y actualizar tabla unica` para correr evaluacion de LSTM+Transformer y regenerar automaticamente `artifacts/model_comparison.csv`.
