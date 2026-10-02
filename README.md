# LightGBM Viz

A local dashboard for exploring tabular data, preparing CSV files, and training real LightGBM classification or regression models. Its visual direction follows the supplied dark analytics-dashboard reference and updates data views from the active dataset.

## Features

- Import a header-based `.csv` file and inspect row count, inferred column type, unique values, and missing cells.
- Fill missing values with mean, median, or mode. Numeric columns use the selected numeric strategy; categorical columns use mode. Download the cleaned dataset as a new CSV.
- Explore numeric feature distributions, feature-space scatter plots, target balance, and individual rows with actual/predicted targets and classification confidence after training.
- Train a local LightGBM model and view held-out accuracy/log loss or R²/RMSE, training curves, runtime, tree count, and gain-based feature importance.
- Inspect the first trained tree, global and per-row TreeSHAP contributions, and a 3D response surface when two numeric features are available.
- Compare or tune four LightGBM parameter presets using the same reproducible data split.
- Use the built-in Iris-shaped sample data to explore the interface before loading a file.

## Run locally

Use Python 3.10 or newer and Node.js 20.19+ or 22.12+.

1. Install frontend packages:

   ```sh
   npm install
   ```

2. Create and activate a Python virtual environment, then install the model API dependencies:

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

3. Start the API in one terminal:

   ```sh
   uvicorn api.main:app --reload --port 8000
   ```

4. Start the dashboard in another terminal:

   ```sh
   npm run dev
   ```

5. Open the local URL printed by Vite (normally `http://localhost:5173`). The Vite development server proxies `/api` calls to the Python API on port 8000.

The dashboard and CSV parsing run in the browser. Training data is sent only to the local API at `127.0.0.1:8000`; no hosted model service is used. If the API is not running, CSV exploration and cleaning still work, while training reports that the API is offline.

## CSV notes

- The first row must contain column headers. Blank cells and common null markers (`null`, `NA`, `N/A`, `NaN`) are treated as missing.
- Select the target column in Data Explorer. Only numeric, non-target columns are passed as model features. Convert categorical predictors to numeric columns in the CSV before training.
- Classification expects a target with at least two classes. Regression expects numeric target values. At least four complete rows are needed.
- The current API trains with a reproducible 80/20 split and 100 estimators. Training is intended for local exploration, not production model serving.
- TreeSHAP summaries are computed on at most 1,000 complete rows to keep local responses compact. Binary classifiers explain the positive-class score; multiclass explanations use each row's predicted class. The decision surface covers the central 90% of the selected feature ranges.

## Checks

```sh
npm run lint
npm run build
python -m py_compile api/main.py
```
