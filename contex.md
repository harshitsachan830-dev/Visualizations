# Project Context

## Purpose

LightGBM Viz is a local-first visualization workspace for inspecting tabular datasets, preparing CSV data, and understanding the output of a real LightGBM model. Its visual direction follows the supplied reference: compact dark evergreen analytics panels, a persistent left navigation rail, bright data accents, and dense but responsive charts.

## Application structure

- `src/App.jsx` owns dashboard state, the built-in sample dataset, CSV ingestion, null-value cleaning, derived chart data, and model API requests.
- `src/App.css` and `src/index.css` define the responsive dashboard presentation and shared color/type tokens.
- `api/main.py` exposes `/api/health`, `/api/train`, `/api/compare`, and `/api/tune`. It fits `LGBMClassifier` or `LGBMRegressor` and returns measured metrics, training history, gain importance, a serialized tree, native TreeSHAP contributions, and a sampled response surface.
- `src/DecisionSurface.jsx` renders the model response surface and CSV feature observations with Three.js.
- `vite.config.js` proxies local `/api` requests to the Python service at port 8000.

## Data behavior

CSV files are parsed in-browser using their header row. Empty cells and common null strings are normalized to missing values. Data Explorer displays per-column type, missing count, unique count, and numeric mean. Mean and median are used for numeric values when selected; mode is used for categorical values. The cleaned rows drive all data charts and can be exported to a new CSV.

Model training sends the active rows, selected target, numeric feature names, and task type to the local API. Classification returns accuracy and log loss; regression returns R² and RMSE. Gain importance and recorded train/validation history are reported by LightGBM, not fabricated in the frontend. Before training, the feature-spread chart is descriptive variance rather than model importance.

After training, the row inspector uses the API's predictions indexed to the original CSV rows. Classification shows predicted class and confidence; regression shows the predicted numeric target. Rows excluded for incomplete model inputs show no prediction.

Tree visualization renders the first LightGBM tree. SHAP global summaries and per-row contributions come from `booster_.predict(pred_contrib=True)`; binary classification explains the positive class, while multiclass classification uses each row's predicted class. The 3D response surface fixes all other inputs to their medians and sweeps the selected two numeric features over their central 90% ranges. Comparisons run four fixed presets (shallow, balanced, deep, regularized) on the same 80/20 split; tuning selects the preset with the best held-out accuracy or R². These are local exploratory comparisons, not a full cross-validation search.

## Development

Run `npm run dev` for the frontend and `uvicorn api.main:app --reload --port 8000` for model training. Install backend packages with `python -m pip install -r requirements.txt`. See `README.md` for environment requirements, CSV constraints, and validation commands.

## Constraints

- Only numeric non-target columns are currently used as model inputs. Encode categorical predictors in the CSV first.
- The API has a 100,000-row request limit and performs an in-memory 80/20 split.
- TreeSHAP summaries sample up to 1,000 complete rows and row explanations up to 120; the API sends those values with the fitted model response.
- CSV inspection, imputation, charts, and export work without the API; real model training requires the Python dependencies and local API process.
