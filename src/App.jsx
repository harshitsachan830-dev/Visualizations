import { lazy, Suspense, useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  BarChart3,
  Boxes,
  Braces,
  Check,
  ChevronDown,
  CircleHelp,
  Database,
  FileSpreadsheet,
  GitBranch,
  GitCompareArrows,
  Gauge,
  Layers3,
  Leaf,
  LoaderCircle,
  Menu,
  Play,
  Settings2,
  SlidersHorizontal,
  Sparkles,
  Upload,
  WandSparkles,
  X,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./App.css";

const DecisionSurface = lazy(() => import("./DecisionSurface.jsx"));
const palette = [
  "#4285F4",
  "#FBBC05",
  "#34A853",
  "#EA4335",
];
const navItems = [
  { label: "Overview", icon: Gauge, target: "overview" },
  { label: "Data explorer", icon: Database, target: "data-explorer" },
  { label: "Training process", icon: Activity, target: "training-process" },
  {
    label: "Tree visualization",
    icon: GitBranch,
    target: "tree-visualization",
  },
  {
    label: "Feature importance",
    icon: BarChart3,
    target: "feature-importance",
  },
  { label: "SHAP explanation", icon: Sparkles, target: "shap-explanation" },
  { label: "3D decision surface", icon: Layers3, target: "decision-surface" },
  {
    label: "Model comparison",
    icon: GitCompareArrows,
    target: "model-comparison",
  },
  {
    label: "Hyperparameter tuning",
    icon: SlidersHorizontal,
    target: "model-comparison",
  },
  { label: "Prediction explorer", icon: Sparkles, target: "prediction" },
];

function createDemoRows() {
  const classes = [
    { name: "Setosa", center: [5.0, 3.4, 1.5, 0.25] },
    { name: "Versicolor", center: [5.9, 2.8, 4.3, 1.3] },
    { name: "Virginica", center: [6.6, 3.0, 5.5, 2.0] },
  ];
  return classes.flatMap(({ name, center }, group) =>
    Array.from({ length: 50 }, (_, index) => {
      const wave = Math.sin((index + 1) * (group + 2));
      const row = {
        "Sepal Length": Number(
          (center[0] + wave * 0.44 + (index % 5) * 0.04).toFixed(2),
        ),
        "Sepal Width": Number(
          (center[1] + Math.cos(index * 1.7) * 0.28).toFixed(2),
        ),
        "Petal Length": Number(
          (center[2] + Math.sin(index * 0.82) * 0.55).toFixed(2),
        ),
        "Petal Width": Number(
          (center[3] + Math.cos(index * 0.73) * 0.2).toFixed(2),
        ),
        Species: name,
      };
      if ((index + group * 7) % 47 === 0) row["Sepal Width"] = null;
      return row;
    }),
  );
}

function numericColumns(rows, columns) {
  return columns.filter((column) =>
    rows.some(
      (row) => typeof row[column] === "number" && Number.isFinite(row[column]),
    ),
  );
}

function formatNumber(value, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : "--";
}

function getStatistics(rows, column) {
  const values = rows
    .map((row) => row[column])
    .filter((value) => typeof value === "number" && Number.isFinite(value))
    .sort((a, b) => a - b);
  if (!values.length) return null;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return {
    mean,
    median: values[Math.floor(values.length / 2)],
    spread: Math.sqrt(variance),
    min: values[0],
    max: values.at(-1),
  };
}

function summarize(rows, columns) {
  return columns.map((column) => {
    const values = rows
      .map((row) => row[column])
      .filter((value) => value !== null && value !== undefined && value !== "");
    const missing = rows.length - values.length;
    const numeric = values.filter(
      (value) => typeof value === "number" && Number.isFinite(value),
    );
    const unique = new Set(values.map(String)).size;
    return {
      column,
      missing,
      unique,
      type: numeric.length === values.length ? "Numeric" : "Category",
      mean: numeric.length
        ? numeric.reduce((a, b) => a + b, 0) / numeric.length
        : null,
    };
  });
}

function makeHistogram(rows, column) {
  const values = rows
    .map((row) => row[column])
    .filter((value) => typeof value === "number" && Number.isFinite(value));
  if (!values.length) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const width = (max - min) / 8 || 1;
  return Array.from({ length: 8 }, (_, index) => {
    const start = min + width * index;
    const end = index === 7 ? max + Number.EPSILON : start + width;
    return {
      range: `${start.toFixed(1)}`,
      count: values.filter((value) => value >= start && value < end).length,
    };
  });
}

function layoutTree(root) {
  if (!root) return null;
  let leafIndex = 0;
  let maxDepth = 0;
  let nextId = 0;
  const nodes = [];
  const edges = [];
  const visit = (node, depth) => {
    maxDepth = Math.max(maxDepth, depth);
    let left;
    let right;
    if (!node.leaf) {
      left = visit(node.left, depth + 1);
      right = visit(node.right, depth + 1);
    }
    const entry = {
      ...node,
      id: nextId++,
      depth,
      x: node.leaf ? leafIndex++ * 112 + 56 : (left.x + right.x) / 2,
      y: depth * 82 + 34,
    };
    nodes.push(entry);
    if (left) edges.push({ from: entry, to: left });
    if (right) edges.push({ from: entry, to: right });
    return entry;
  };
  visit(root, 0);
  return {
    nodes,
    edges,
    width: Math.max(leafIndex * 112, 336),
    height: (maxDepth + 1) * 82 + 34,
  };
}

function App() {
  const [rows, setRows] = useState(createDemoRows);
  const [columns, setColumns] = useState([
    "Sepal Length",
    "Sepal Width",
    "Petal Length",
    "Petal Width",
    "Species",
  ]);
  const [fileName, setFileName] = useState("Iris sample");
  const [activeNav, setActiveNav] = useState("Overview");
  const [strategy, setStrategy] = useState("mean");
  const [target, setTarget] = useState("Species");
  const [xFeature, setXFeature] = useState("Petal Length");
  const [yFeature, setYFeature] = useState("Petal Width");
  const [taskType, setTaskType] = useState("Classification");
  const [modelVariant, setModelVariant] = useState("balanced");
  const [model, setModel] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [training, setTraining] = useState(false);
  const [studyMode, setStudyMode] = useState("");
  const [notice, setNotice] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [mobileNav, setMobileNav] = useState(false);
  const [sampleIndex, setSampleIndex] = useState(0);
  const uploadRef = useRef(null);

  const stats = useMemo(() => summarize(rows, columns), [rows, columns]);
  const numeric = useMemo(
    () =>
      numericColumns(
        rows,
        columns.filter((column) => column !== target),
      ),
    [rows, columns, target],
  );
  const labels = useMemo(
    () => [
      ...new Set(
        rows
          .map((row) => row[target])
          .filter(
            (value) => value !== null && value !== undefined && value !== "",
          ),
      ),
    ],
    [rows, target],
  );
  const classCounts = useMemo(() => {
    if (taskType === "Regression") {
      const values = rows
        .map((row) => row[target])
        .filter((value) => typeof value === "number" && Number.isFinite(value));
      if (values.length) {
        const min = Math.min(...values);
        const max = Math.max(...values);
        const binWidth = (max - min) / 5 || 1;
        return Array.from({ length: 5 }, (_, index) => {
          const start = min + index * binWidth;
          const end = index === 4 ? max : start + binWidth;
          return {
            name: `${start.toFixed(1)} - ${end.toFixed(1)}`,
            count: values.filter(
              (value) =>
                value >= start && (index === 4 ? value <= end : value < end),
            ).length,
          };
        });
      }
    }
    return labels.map((label) => ({
      name: String(label),
      count: rows.filter((row) => row[target] === label).length,
    }));
  }, [labels, rows, target, taskType]);
  const scatterGroups =
    taskType === "Regression"
      ? [{ name: "Samples", count: rows.length }]
      : classCounts;
  const selectedX = numeric.includes(xFeature) ? xFeature : numeric[0];
  const selectedY = numeric.includes(yFeature)
    ? yFeature
    : numeric[1] || numeric[0];
  const missingTotal = stats.reduce((sum, column) => sum + column.missing, 0);
  const featureStats = numeric
    .map((column) => ({ column, ...getStatistics(rows, column) }))
    .filter((item) => item.mean !== null);
  const featureImportance = model?.feature_importance?.length
    ? model.feature_importance
    : featureStats
        .map(({ column, spread }) => ({ feature: column, value: spread }))
        .sort((a, b) => b.value - a.value);
  const maxImportance = Math.max(
    ...featureImportance.map((item) => item.value || 0),
    1,
  );
  const currentRow = rows.length ? rows[sampleIndex % rows.length] : {};
  const currentPrediction = model?.predictions?.find(
    (prediction) => prediction.row_index === sampleIndex,
  );
  const currentShap = model?.shap_samples?.find(
    (sample) => sample.row_index === sampleIndex,
  );
  const treeLayout = useMemo(() => layoutTree(model?.tree), [model]);
  const surfacePoints = useMemo(() => {
    if (!model?.surface) return [];
    return rows
      .filter(
        (row) =>
          Number.isFinite(row[model.surface.x_feature]) &&
          Number.isFinite(row[model.surface.y_feature]),
      )
      .slice(0, 400)
      .map((row) => ({
        x: row[model.surface.x_feature],
        y: row[model.surface.y_feature],
        label: String(row[target]),
      }));
  }, [model, rows, target]);

  function handleFile(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setNotice("Choose a .csv file to explore.");
      return;
    }
    Papa.parse(file, {
      header: true,
      skipEmptyLines: "greedy",
      dynamicTyping: true,
      transformHeader: (header) => header.trim(),
      complete: ({ data, meta, errors }) => {
        const nextColumns = meta.fields?.filter(Boolean) || [];
        if (errors.length || !nextColumns.length || !data.length) {
          setNotice(
            errors[0]?.message ||
              "This CSV has no readable rows or column headers.",
          );
          return;
        }
        const cleanedRows = data.map((source) =>
          Object.fromEntries(
            nextColumns.map((column) => {
              const value = source[column];
              const isEmpty =
                value === "" ||
                value === null ||
                value === undefined ||
                (typeof value === "string" &&
                  ["null", "na", "n/a", "nan"].includes(
                    value.trim().toLowerCase(),
                  ));
              return [column, isEmpty ? null : value];
            }),
          ),
        );
        setRows(cleanedRows);
        setColumns(nextColumns);
        setTarget(nextColumns.at(-1));
        setXFeature("");
        setYFeature("");
        setFileName(file.name);
        setModel(null);
        setComparison(null);
        setSampleIndex(0);
        setNotice(
          `${cleanedRows.length.toLocaleString()} rows loaded from ${file.name}.`,
        );
      },
    });
  }

  function applyImputation() {
    if (!missingTotal) {
      setNotice("No missing values found in this dataset.");
      return;
    }
    const filledRows = rows.map((row) => ({ ...row }));
    columns.forEach((column) => {
      const missingIndexes = filledRows
        .map((row, index) =>
          row[column] === null ||
          row[column] === undefined ||
          row[column] === ""
            ? index
            : -1,
        )
        .filter((index) => index >= 0);
      if (!missingIndexes.length) return;
      const values = filledRows
        .map((row) => row[column])
        .filter(
          (value) => value !== null && value !== undefined && value !== "",
        );
      const isNumeric =
        values.length > 0 &&
        values.every(
          (value) => typeof value === "number" && Number.isFinite(value),
        );
      let fillValue;
      if (strategy === "mean" && isNumeric)
        fillValue =
          values.reduce((sum, value) => sum + value, 0) / values.length;
      else if (strategy === "median" && isNumeric) {
        const sorted = [...values].sort((a, b) => a - b);
        const middle = Math.floor(sorted.length / 2);
        fillValue =
          sorted.length % 2
            ? sorted[middle]
            : (sorted[middle - 1] + sorted[middle]) / 2;
      } else {
        const frequencies = new Map();
        values.forEach((value) =>
          frequencies.set(value, (frequencies.get(value) || 0) + 1),
        );
        fillValue =
          [...frequencies.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ??
          null;
      }
      missingIndexes.forEach((index) => {
        filledRows[index][column] = fillValue;
      });
    });
    setRows(filledRows);
    setModel(null);
    setComparison(null);
    setNotice(`Missing values filled using ${strategy}.`);
  }

  function downloadCleanCsv() {
    const csv = Papa.unparse({
      fields: columns,
      data: rows.map((row) => columns.map((column) => row[column] ?? "")),
    });
    const link = document.createElement("a");
    const downloadUrl = URL.createObjectURL(
      new Blob([csv], { type: "text/csv;charset=utf-8" }),
    );
    link.href = downloadUrl;
    link.download = `${fileName.replace(/\.csv$/i, "")}-clean.csv`;
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 1000);
  }

  async function trainModel() {
    if (!rows.length || !numeric.length) {
      setNotice(
        "Upload a dataset with at least one numeric feature before training.",
      );
      return;
    }
    setTraining(true);
    setNotice("Training LightGBM with the local API...");
    try {
      const response = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rows,
          target,
          features: numeric,
          task_type: taskType.toLowerCase(),
          surface_features: [selectedX, selectedY],
          model_variant: modelVariant,
        }),
      });
      const result = await response.json();
      if (!response.ok)
        throw new Error(result.detail || "Training request failed.");
      setModel(result);
      setNotice(
        "Model trained successfully. Metrics reflect the held-out test split.",
      );
    } catch (error) {
      setNotice(
        error.message.includes("fetch") || error.message.includes("Failed")
          ? "Training API is offline. Start the Python API to fit a real LightGBM model."
          : error.message,
      );
    } finally {
      setTraining(false);
    }
  }

  async function runModelStudy(study) {
    if (!rows.length || !numeric.length) {
      setNotice(
        "Upload data with numeric features before comparing or tuning models.",
      );
      return;
    }
    setStudyMode(study);
    setNotice(
      study === "tune"
        ? "Evaluating LightGBM parameter presets..."
        : "Comparing LightGBM model presets...",
    );
    try {
      const response = await fetch(`/api/${study}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rows,
          target,
          features: numeric,
          task_type: taskType.toLowerCase(),
        }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Model study failed.");
      setComparison(result);
      setNotice(
        `${study === "tune" ? "Parameter search" : "Model comparison"} complete. Best preset: ${result.best_model}.`,
      );
    } catch (error) {
      setNotice(
        error.message.includes("fetch") || error.message.includes("Failed")
          ? "Training API is offline. Start the Python API to compare model presets."
          : error.message,
      );
    } finally {
      setStudyMode("");
    }
  }

  function jumpTo(item) {
    setActiveNav(item.label);
    setMobileNav(false);
    document
      .getElementById(item.target)
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}>
        <a
          className="brand"
          href="#overview"
          onClick={() => setActiveNav("Overview")}
        >
          <span className="brand-mark">
            <Leaf size={21} strokeWidth={2.2} />
          </span>
          <span>
            LightGBM <b>Viz</b>
          </span>
        </a>
        <button
          className="icon-button sidebar-close"
          aria-label="Close navigation"
          onClick={() => setMobileNav(false)}
        >
          <X size={17} />
        </button>
        <div className="workspace-tag">
          <span className="online-dot" /> MODEL WORKSPACE
        </div>
        <nav className="side-nav" aria-label="Dashboard sections">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                className={`nav-link ${activeNav === item.label ? "active" : ""}`}
                key={item.label}
                onClick={() => jumpTo(item)}
              >
                <Icon size={16} strokeWidth={1.9} />
                <span>{item.label}</span>
                {activeNav === item.label && (
                  <span className="nav-active-dot" />
                )}
              </button>
            );
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="side-note">
            <span className="note-icon">
              <Braces size={15} />
            </span>
            <div>
              <strong>LightGBM engine</strong>
              <small>Gradient boosting · leaf-wise</small>
            </div>
          </div>
          <button
            className="nav-link settings-link"
            onClick={() => setShowSettings((value) => !value)}
          >
            <Settings2 size={16} />
            <span>Settings</span>
          </button>
          <div className="sidebar-version">
            LOCAL WORKSPACE <span>v1.0</span>
          </div>
        </div>
      </aside>

      <main className="main-content" id="overview">
        <header className="topbar">
          <button
            className="icon-button mobile-menu"
            aria-label="Toggle navigation"
            onClick={() => setMobileNav((value) => !value)}
          >
            <Menu size={18} />
          </button>
          <div className="heading-block">
            <div className="eyebrow">
              <span className="online-dot" /> MODEL LAB{" "}
              <span className="eyebrow-divider">/</span> OVERVIEW
            </div>
            <h1>
              LightGBM <span>Visualization</span>
            </h1>
            <p>Explore how gradient boosting learns, one leaf at a time.</p>
          </div>
          <div className="header-actions">
            <label className="select-wrap dataset-picker">
              <FileSpreadsheet size={15} />
              <select
                aria-label="Dataset"
                value={fileName}
                onChange={(event) =>
                  event.target.value === "Iris sample" &&
                  (setRows(createDemoRows()),
                  setColumns([
                    "Sepal Length",
                    "Sepal Width",
                    "Petal Length",
                    "Petal Width",
                    "Species",
                  ]),
                  setTarget("Species"),
                  setTaskType("Classification"),
                  setFileName("Iris sample"),
                  setModel(null))
                }
              >
                <option value="Iris sample">Iris sample</option>
                <option value={fileName}>{fileName}</option>
              </select>
              <ChevronDown size={14} />
            </label>
            <label className="select-wrap task-picker">
              <select
                aria-label="Task type"
                value={taskType}
                onChange={(event) => {
                  setTaskType(event.target.value);
                  setModel(null);
                  setComparison(null);
                }}
              >
                <option>Classification</option>
                <option>Regression</option>
              </select>
              <ChevronDown size={14} />
            </label>
            <button
              className="icon-button settings-button"
              aria-label="Training settings"
              title="Training settings"
              onClick={() => setShowSettings((value) => !value)}
            >
              <Settings2 size={18} />
            </button>
            <button
              className="button-primary train-button"
              onClick={trainModel}
              disabled={training}
            >
              <span className="train-icon">
                {training ? (
                  <LoaderCircle size={15} className="spin" />
                ) : (
                  <Play size={15} fill="currentColor" />
                )}
              </span>
              {training ? "Training..." : "Train model"}
            </button>
          </div>
        </header>

        {showSettings && (
          <section className="settings-panel" aria-label="Training settings">
            <div>
              <strong>Training configuration</strong>
              <span>Reproducible 80/20 split · preset defaults</span>
            </div>
            <label className="compact-select">
              <span>PRESET</span>
              <select
                aria-label="Model preset"
                value={modelVariant}
                onChange={(event) => {
                  setModelVariant(event.target.value);
                  setModel(null);
                }}
              >
                <option value="balanced">Balanced</option>
                <option value="shallow">Shallow</option>
                <option value="deep">Deep</option>
                <option value="regularized">Regularized</option>
              </select>
              <ChevronDown size={13} />
            </label>
            <button
              className="icon-button"
              aria-label="Close settings"
              onClick={() => setShowSettings(false)}
            >
              <X size={16} />
            </button>
          </section>
        )}
        {notice && (
          <div className="notice-bar" role="status">
            <span>
              <Check size={15} />
              {notice}
            </span>
            <button aria-label="Dismiss message" onClick={() => setNotice("")}>
              <X size={15} />
            </button>
          </div>
        )}

        <section className="metric-grid" aria-label="Model summary">
          <Metric
            icon={Gauge}
            color="green"
            label={taskType === "Classification" ? "Test accuracy" : "Test R²"}
            value={
              model
                ? taskType === "Classification"
                  ? `${(model.accuracy * 100).toFixed(1)}%`
                  : formatNumber(model.r2, 3)
                : "--"
            }
            detail={model ? "Held-out split" : "Train to calculate"}
          />
          <Metric
            icon={Activity}
            color="pink"
            label={taskType === "Classification" ? "Log loss" : "RMSE"}
            value={
              model
                ? formatNumber(
                    taskType === "Classification" ? model.log_loss : model.rmse,
                    3,
                  )
                : "--"
            }
            detail={model ? "Lower is better" : "No model trained"}
          />
          <Metric
            icon={GitBranch}
            color="amber"
            label="Trees"
            value={model?.trees?.toLocaleString() || "--"}
            detail={model ? "Boosting rounds" : "100 estimators"}
          />
          <Metric
            icon={Gauge}
            color="blue"
            label="Training time"
            value={
              model?.training_time
                ? `${formatNumber(model.training_time, 2)}s`
                : "--"
            }
            detail="Local LightGBM"
          />
        </section>

        <section className="panel data-panel" id="data-explorer">
          <div className="panel-heading">
            <div>
              <div className="section-kicker">DATASET / {fileName}</div>
              <h2>
                Data explorer{" "}
                <span className="count-chip">
                  {rows.length.toLocaleString()} rows
                </span>
              </h2>
              <p>
                Inspect columns, missing values, and prepare a clean training
                set.
              </p>
            </div>
            <div className="panel-actions">
              <button
                className="button-secondary"
                onClick={() => uploadRef.current?.click()}
              >
                <Upload size={15} />
                Upload CSV
              </button>
              <button
                className="icon-button subtle-button"
                aria-label="Download cleaned CSV"
                title="Download cleaned CSV"
                onClick={downloadCleanCsv}
              >
                <ArrowDownToLine size={17} />
              </button>
              <input
                ref={uploadRef}
                type="file"
                accept=".csv,text/csv"
                hidden
                onChange={(event) => handleFile(event.target.files?.[0])}
              />
            </div>
          </div>
          <div className="data-summary-row">
            <div className="summary-stat">
              <span>FEATURES</span>
              <strong>{columns.length}</strong>
            </div>
            <div className="summary-stat">
              <span>MISSING VALUES</span>
              <strong className={missingTotal ? "warning-value" : ""}>
                {missingTotal}
              </strong>
            </div>
            <div className="summary-stat">
              <span>TARGET</span>
              <label className="compact-select">
                <select
                  aria-label="Target column"
                  value={target}
                  onChange={(event) => {
                    setTarget(event.target.value);
                    setModel(null);
                  }}
                >
                  {columns.map((column) => (
                    <option key={column}>{column}</option>
                  ))}
                </select>
                <ChevronDown size={13} />
              </label>
            </div>
            <div className="impute-control">
              <span>FILL MISSING WITH</span>
              <label className="compact-select">
                <select
                  aria-label="Missing value strategy"
                  value={strategy}
                  onChange={(event) => setStrategy(event.target.value)}
                >
                  <option value="mean">Mean</option>
                  <option value="median">Median</option>
                  <option value="mode">Mode</option>
                </select>
                <ChevronDown size={13} />
              </label>
              <button
                className="button-secondary apply-button"
                onClick={applyImputation}
                disabled={!missingTotal}
              >
                <WandSparkles size={14} />
                Apply
              </button>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Column</th>
                  <th>Type</th>
                  <th>Missing</th>
                  <th>Unique</th>
                  <th>Mean / example</th>
                  <th>Quality</th>
                </tr>
              </thead>
              <tbody>
                {stats.map((column) => (
                  <tr key={column.column}>
                    <td>
                      <span className="column-dot" />
                      {column.column}
                      {column.column === target && (
                        <span className="target-tag">TARGET</span>
                      )}
                    </td>
                    <td>
                      <span
                        className={`type-pill ${column.type === "Numeric" ? "numeric-pill" : ""}`}
                      >
                        {column.type}
                      </span>
                    </td>
                    <td>
                      {column.missing ? (
                        <span className="missing-count">
                          {column.missing}{" "}
                          <i>
                            {((column.missing / rows.length) * 100).toFixed(1)}%
                          </i>
                        </span>
                      ) : (
                        <span className="ok-text">0</span>
                      )}
                    </td>
                    <td>{column.unique}</td>
                    <td>
                      {column.mean == null
                        ? (rows.find((row) => row[column.column] != null)?.[
                            column.column
                          ] ?? "--")
                        : formatNumber(column.mean, 2)}
                    </td>
                    <td>
                      <span
                        className={`quality-bar ${column.missing ? "has-missing" : ""}`}
                      >
                        <i
                          style={{
                            width: `${Math.max(8, 100 - (column.missing / Math.max(rows.length, 1)) * 100)}%`,
                          }}
                        />
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <div className="chart-grid primary-charts">
          <section className="panel chart-panel" id="training-process">
            <ChartTitle
              icon={Activity}
              title="Training process"
              subtitle={
                model
                  ? "Held-out loss by boosting round"
                  : "Live model curves appear after training"
              }
            />
            <div className="chart-legend">
              <span>
                <i className="legend-dot green-dot" />
                Training
              </span>
              <span>
                <i className="legend-dot pink-dot" />
                Validation
              </span>
            </div>
            <div className="chart-area">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={model?.history || []}
                  margin={{ top: 12, right: 10, bottom: 3, left: -15 }}
                >
                  <CartesianGrid stroke="#292929" vertical={false} />
                  <XAxis
                    dataKey="iteration"
                    tick={{ fill: "#a3a3a3", fontSize: 10 }}
                    axisLine={{ stroke: "#444444" }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fill: "#a3a3a3", fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Line
                    type="monotone"
                    dataKey="train"
                    stroke="#34A853"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="valid"
                    stroke="#EA4335"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
              {!model && (
                <div className="chart-empty">
                  <span className="empty-pulse">
                    <Activity size={17} />
                  </span>
                  Awaiting a trained model
                </div>
              )}
            </div>
            <div className="chart-foot">
              <span>BOOSTING ROUND</span>
              <span>{model ? `${model.trees} rounds` : "Not started"}</span>
            </div>
          </section>
          <section className="panel chart-panel boundary-panel">
            <ChartTitle
              icon={Layers3}
              title="Feature space"
              subtitle={
                taskType === "Regression"
                  ? "Explore numeric samples across features"
                  : "Explore the selected features by target class"
              }
            />
            <div className="feature-selectors">
              <label className="compact-select">
                <select
                  aria-label="Horizontal feature"
                  value={selectedX || ""}
                  onChange={(event) => setXFeature(event.target.value)}
                >
                  {numeric.map((column) => (
                    <option key={column}>{column}</option>
                  ))}
                </select>
                <ChevronDown size={12} />
              </label>
              <ArrowRight size={13} />
              <label className="compact-select">
                <select
                  aria-label="Vertical feature"
                  value={selectedY || ""}
                  onChange={(event) => setYFeature(event.target.value)}
                >
                  {numeric.map((column) => (
                    <option key={column}>{column}</option>
                  ))}
                </select>
                <ChevronDown size={12} />
              </label>
            </div>
            <div className="chart-area scatter-area">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart
                  margin={{ top: 10, right: 10, bottom: 10, left: -12 }}
                >
                  <CartesianGrid stroke="#292929" />
                  <XAxis
                    type="number"
                    dataKey={selectedX}
                    name={selectedX}
                    tick={{ fill: "#a3a3a3", fontSize: 9 }}
                    axisLine={{ stroke: "#444444" }}
                    tickLine={false}
                  />
                  <YAxis
                    type="number"
                    dataKey={selectedY}
                    name={selectedY}
                    tick={{ fill: "#a3a3a3", fontSize: 9 }}
                    axisLine={{ stroke: "#444444" }}
                    tickLine={false}
                  />
                  <Tooltip
                    cursor={{ strokeDasharray: "3 3" }}
                    contentStyle={tooltipStyle}
                  />
                  <ReferenceLine
                    x={getStatistics(rows, selectedX)?.mean}
                    stroke="#707070"
                    strokeDasharray="4 4"
                  />
                  <ReferenceLine
                    y={getStatistics(rows, selectedY)?.mean}
                    stroke="#707070"
                    strokeDasharray="4 4"
                  />
                  {scatterGroups.map((group, index) => (
                    <Scatter
                      key={group.name}
                      name={group.name}
                      data={rows.filter(
                        (row) =>
                          (taskType === "Regression" ||
                            String(row[target]) === group.name) &&
                          Number.isFinite(row[selectedX]) &&
                          Number.isFinite(row[selectedY]),
                      )}
                      fill={palette[index % palette.length]}
                      fillOpacity={0.78}
                    />
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
            </div>
            <div className="legend-row">
              {scatterGroups.slice(0, 4).map((group, index) => (
                <span key={group.name}>
                  <i style={{ background: palette[index % palette.length] }} />
                  {group.name}
                </span>
              ))}
            </div>
          </section>
          <section className="panel chart-panel distribution-panel">
            <ChartTitle
              icon={BarChart3}
              title="Feature distribution"
              subtitle={
                selectedX
                  ? `Frequency across ${selectedX}`
                  : "Upload numeric columns to chart"
              }
            />
            <div className="chart-area">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={makeHistogram(rows, selectedX)}
                  margin={{ top: 18, right: 5, bottom: 0, left: -22 }}
                >
                  <CartesianGrid stroke="#292929" vertical={false} />
                  <XAxis
                    dataKey="range"
                    tick={{ fill: "#a3a3a3", fontSize: 9 }}
                    axisLine={{ stroke: "#444444" }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fill: "#a3a3a3", fontSize: 9 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar
                    dataKey="count"
                    fill="#FBBC05"
                    radius={[3, 3, 0, 0]}
                    maxBarSize={28}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-foot">
              <span>ROWS IN RANGE</span>
              <span>{rows.length.toLocaleString()} observations</span>
            </div>
          </section>
        </div>

        <div className="chart-grid lower-charts">
          <section className="panel feature-panel" id="feature-importance">
            <ChartTitle
              icon={BarChart3}
              title={model ? "Feature importance" : "Feature spread"}
              subtitle={
                model
                  ? "Gain from the trained LightGBM model"
                  : "Relative variance · train for gain importance"
              }
            />
            <div className="importance-list">
              {featureImportance.slice(0, 6).map((feature, index) => (
                <div className="importance-row" key={feature.feature}>
                  <span className="importance-label">{feature.feature}</span>
                  <span className="importance-track">
                    <i
                      style={{
                        width: `${(feature.value / maxImportance) * 100}%`,
                        background: palette[index % palette.length],
                      }}
                    />
                  </span>
                  <b>
                    {model
                      ? formatNumber(feature.value, 2)
                      : formatNumber(feature.value, 2)}
                  </b>
                </div>
              ))}
            </div>
            <div className="chart-foot">
              <span>{model ? "LIGHTGBM GAIN" : "STANDARD DEVIATION"}</span>
              <span>{featureImportance.length} numeric features</span>
            </div>
          </section>
          <section className="panel class-panel">
            <ChartTitle
              icon={Boxes}
              title={
                taskType === "Regression"
                  ? "Target range"
                  : "Target distribution"
              }
              subtitle={`${taskType === "Regression" ? "Value distribution" : "Class balance"} · ${target}`}
            />
            <div className="class-list">
              {classCounts.slice(0, 6).map((item, index) => (
                <div className="class-row" key={item.name}>
                  <span className="class-label">
                    <i
                      style={{ background: palette[index % palette.length] }}
                    />
                    {item.name}
                  </span>
                  <span className="class-track">
                    <i
                      style={{
                        width: `${(item.count / Math.max(rows.length, 1)) * 100}%`,
                        background: palette[index % palette.length],
                      }}
                    />
                  </span>
                  <b>{item.count}</b>
                </div>
              ))}
            </div>
            <div className="class-total">
              <span>TOTAL SAMPLES</span>
              <strong>{rows.length.toLocaleString()}</strong>
            </div>
          </section>
          <section className="panel prediction-panel" id="prediction">
            <div className="prediction-heading">
              <ChartTitle
                icon={Sparkles}
                title="Row inspector"
                subtitle="Review a record from the active dataset"
              />
              <div className="row-stepper">
                <button
                  aria-label="Previous row"
                  onClick={() =>
                    setSampleIndex(
                      (index) => (index - 1 + rows.length) % rows.length,
                    )
                  }
                >
                  ‹
                </button>
                <span>
                  {sampleIndex + 1}
                  <i> / {rows.length}</i>
                </span>
                <button
                  aria-label="Next row"
                  onClick={() =>
                    setSampleIndex((index) => (index + 1) % rows.length)
                  }
                >
                  ›
                </button>
              </div>
            </div>
            <div className="prediction-values">
              {columns.slice(0, 4).map((column) => (
                <div key={column}>
                  <span>{column}</span>
                  <b>
                    {currentRow[column] ?? (
                      <i className="null-value">missing</i>
                    )}
                  </b>
                </div>
              ))}
            </div>
            <div className="prediction-result">
              <div>
                <span>ACTUAL TARGET</span>
                <strong>{currentRow[target] ?? "Not set"}</strong>
              </div>
              <div>
                <span>
                  {taskType === "Classification"
                    ? "PREDICTED CLASS"
                    : "PREDICTED VALUE"}
                </span>
                <strong>
                  {currentPrediction
                    ? taskType === "Classification"
                      ? currentPrediction.prediction
                      : formatNumber(currentPrediction.prediction, 3)
                    : model
                      ? "Unavailable"
                      : "Train model"}
                </strong>
                {taskType === "Classification" && currentPrediction && (
                  <small>
                    {(currentPrediction.confidence * 100).toFixed(1)}%
                    confidence
                  </small>
                )}
              </div>
            </div>
          </section>
        </div>

        <section className="bottom-strip" id="tree-growth">
          <div className="bottom-title">
            <span className="tree-icon">
              <GitBranch size={18} />
            </span>
            <div>
              <strong>Leaf-wise growth</strong>
              <span>
                LightGBM expands the leaf with maximum gain at each step.
              </span>
            </div>
          </div>
          <div className="growth-steps">
            <div>
              <i className="step-node green-node" />
              <span>Find best split</span>
            </div>
            <ArrowRight size={14} />
            <div>
              <i className="step-node amber-node" />
              <span>Choose max gain</span>
            </div>
            <ArrowRight size={14} />
            <div>
              <i className="step-node pink-node" />
              <span>Grow one leaf</span>
            </div>
            <ArrowRight size={14} />
            <div>
              <i className="step-node blue-node" />
              <span>Repeat until done</span>
            </div>
          </div>
          <button
            className="learn-button"
            onClick={() =>
              setNotice(
                "LightGBM grows leaf-wise, choosing the leaf with the largest loss reduction at each iteration.",
              )
            }
          >
            <CircleHelp size={15} />
            How it works
          </button>
        </section>
        <AnalysisPanels
          model={model}
          treeLayout={treeLayout}
          currentShap={currentShap}
          sampleIndex={sampleIndex}
          taskType={taskType}
          comparison={comparison}
          studyMode={studyMode}
          runModelStudy={runModelStudy}
          surfacePoints={surfacePoints}
        />
        <footer className="page-footer">
          <span>
            <span className="online-dot" /> DATA STAYS IN YOUR BROWSER
          </span>
          <span>
            LIGHTGBM VIZ <i>·</i> LOCAL ANALYSIS
          </span>
        </footer>
      </main>
    </div>
  );
}

const tooltipStyle = {
  background: "#111111",
  border: "1px solid #303030",
  borderRadius: 6,
  color: "#f5f5f5",
  fontSize: 11,
};

function Metric({ icon: Icon, color, label, value, detail }) {
  return (
    <div className="metric-card">
      <span className={`metric-icon ${color}`}>
        <Icon size={18} />
      </span>
      <div className="metric-copy">
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
      <span className="metric-detail">{detail}</span>
    </div>
  );
}

function ChartTitle({ icon: Icon, title, subtitle }) {
  return (
    <div className="chart-title">
      <div className="chart-icon">
        <Icon size={15} />
      </div>
      <div>
        <h3>{title}</h3>
        <p>{subtitle}</p>
      </div>
    </div>
  );
}

function AnalysisPanels({
  model,
  treeLayout,
  currentShap,
  sampleIndex,
  taskType,
  comparison,
  studyMode,
  runModelStudy,
  surfacePoints,
}) {
  const largestImpact = Math.max(
    ...(currentShap?.values.map((value) => Math.abs(value.shap_value)) || [0]),
    0.0001,
  );
  return (
    <div className="analysis-grid">
      <section
        className="panel analysis-panel tree-panel"
        id="tree-visualization"
      >
        <ChartTitle
          icon={GitBranch}
          title="Tree visualization"
          subtitle={
            model
              ? `Tree 1 · ${model.trees} boosting rounds trained`
              : "Train a model to inspect its first tree"
          }
        />
        {treeLayout ? (
          <div className="tree-scroll">
            <svg
              className="tree-svg"
              width={treeLayout.width}
              height={treeLayout.height}
              viewBox={`0 0 ${treeLayout.width} ${treeLayout.height}`}
              role="img"
              aria-label="First LightGBM decision tree"
            >
              <g className="tree-edges">
                {treeLayout.edges.map((edge) => (
                  <line
                    key={`${edge.from.id}-${edge.to.id}`}
                    x1={edge.from.x}
                    y1={edge.from.y + 16}
                    x2={edge.to.x}
                    y2={edge.to.y - 16}
                  />
                ))}
              </g>
              {treeLayout.nodes.map((node) => (
                <g
                  key={node.id}
                  className={`tree-node ${node.leaf ? "leaf" : "split"}`}
                  transform={`translate(${node.x}, ${node.y})`}
                >
                  <rect x="-49" y="-15" width="98" height="30" rx="4" />
                  <text y="-1">
                    {node.leaf
                      ? `Leaf ${formatNumber(node.value, 2)}`
                      : `${node.feature} ≤ ${formatNumber(node.threshold, 2)}`}
                  </text>
                  <text className="tree-samples" y="25">
                    {node.samples} rows
                  </text>
                </g>
              ))}
            </svg>
          </div>
        ) : (
          <div className="analysis-empty">
            <GitBranch size={20} />
            <span>Train a model to render its LightGBM split tree.</span>
          </div>
        )}
      </section>

      <section
        className="panel analysis-panel shap-panel"
        id="shap-explanation"
      >
        <ChartTitle
          icon={Sparkles}
          title="SHAP explanation"
          subtitle={
            model
              ? `TreeSHAP · ${model.shap_class} · row ${sampleIndex + 1}`
              : "Native LightGBM feature contributions"
          }
        />
        {model?.shap_summary?.length ? (
          <div className="shap-content">
            <div className="shap-summary-list">
              <div className="analysis-label">GLOBAL · MEAN ABSOLUTE SHAP</div>
              {model.shap_summary.slice(0, 6).map((item) => (
                <div className="shap-summary-row" key={item.feature}>
                  <span>{item.feature}</span>
                  <i>
                    <b
                      style={{
                        width: `${(item.mean_abs_shap / model.shap_summary[0].mean_abs_shap) * 100}%`,
                      }}
                    />
                  </i>
                  <strong>{formatNumber(item.mean_abs_shap, 3)}</strong>
                </div>
              ))}
            </div>
            <div className="shap-row-impact">
              <div className="analysis-label">
                ROW CONTRIBUTIONS <span>lower score</span>
                <span>higher score</span>
              </div>
              {currentShap ? (
                currentShap.values
                  .slice()
                  .sort(
                    (a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value),
                  )
                  .slice(0, 6)
                  .map((item) => (
                    <div className="shap-impact-row" key={item.feature}>
                      <span>{item.feature}</span>
                      <div className="shap-impact-track">
                        <i
                          className={
                            item.shap_value < 0 ? "negative" : "positive"
                          }
                          style={{
                            width: `${Math.min((Math.abs(item.shap_value) / largestImpact) * 48, 48)}%`,
                          }}
                        />
                      </div>
                      <b>{formatNumber(item.shap_value, 3)}</b>
                    </div>
                  ))
              ) : (
                <span className="analysis-note">
                  This row has missing features and was omitted from SHAP.
                </span>
              )}
            </div>
          </div>
        ) : (
          <div className="analysis-empty">
            <Sparkles size={20} />
            <span>Train a model to calculate TreeSHAP values.</span>
          </div>
        )}
      </section>

      <section
        className="panel analysis-panel surface-panel"
        id="decision-surface"
      >
        <ChartTitle
          icon={Layers3}
          title="3D decision surface"
          subtitle={
            model?.surface
              ? `${model.surface.z_label} · model response`
              : "Train with at least two numeric features"
          }
        />
        {model?.surface ? (
          <>
            <Suspense
              fallback={
                <div className="surface-loading">Loading 3D surface...</div>
              }
            >
              <DecisionSurface surface={model.surface} points={surfacePoints} />
            </Suspense>
            <div className="surface-labels">
              <span>X · {model.surface.x_feature}</span>
              <span>Y · {model.surface.y_feature}</span>
              <span>Z · {model.surface.z_label}</span>
            </div>
          </>
        ) : (
          <div className="analysis-empty">
            <Layers3 size={20} />
            <span>A fitted model with two numeric features is required.</span>
          </div>
        )}
      </section>

      <section
        className="panel analysis-panel comparison-panel"
        id="model-comparison"
      >
        <div className="comparison-heading">
          <ChartTitle
            icon={GitCompareArrows}
            title="Model comparison"
            subtitle="Compare held-out results across LightGBM presets"
          />
          <div className="comparison-actions">
            <button
              className="button-secondary"
              onClick={() => runModelStudy("compare")}
              disabled={Boolean(studyMode)}
            >
              <GitCompareArrows size={14} />
              {studyMode === "compare" ? "Comparing" : "Compare"}
            </button>
            <button
              className="button-secondary"
              onClick={() => runModelStudy("tune")}
              disabled={Boolean(studyMode)}
            >
              <SlidersHorizontal size={14} />
              {studyMode === "tune" ? "Tuning" : "Tune"}
            </button>
          </div>
        </div>
        {comparison ? (
          <>
            <div className="comparison-best">
              <Check size={14} />
              <span>Best on held-out {comparison.selection_metric}:</span>
              <strong>{comparison.best_model}</strong>
              <i>{comparison.evaluated} presets</i>
            </div>
            <div className="comparison-table-wrap">
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th>Preset</th>
                    <th>{taskType === "Classification" ? "Accuracy" : "R²"}</th>
                    <th>
                      {taskType === "Classification" ? "Log loss" : "RMSE"}
                    </th>
                    <th>Trees</th>
                    <th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.models.map((item) => (
                    <tr
                      key={item.variant}
                      className={
                        item.variant === comparison.best_model ? "best-row" : ""
                      }
                    >
                      <td>{item.variant}</td>
                      <td>
                        {formatNumber(
                          taskType === "Classification"
                            ? item.accuracy
                            : item.r2,
                          3,
                        )}
                      </td>
                      <td>
                        {formatNumber(
                          taskType === "Classification"
                            ? item.log_loss
                            : item.rmse,
                          3,
                        )}
                      </td>
                      <td>{item.trees}</td>
                      <td>{formatNumber(item.training_time, 2)}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <div className="analysis-empty comparison-empty">
            <GitCompareArrows size={20} />
            <span>
              Run a comparison or tune search to evaluate four presets on the
              same split.
            </span>
          </div>
        )}
      </section>
    </div>
  );
}

export default App;
