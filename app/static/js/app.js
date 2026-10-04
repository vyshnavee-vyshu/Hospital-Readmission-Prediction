const MODEL_META = {
  logistic_regression: { name: "Logistic Regression", blurb: "Linear classifier with balanced class weights." },
  decision_tree: { name: "Decision Tree", blurb: "Axis-aligned splits with a limited tree depth." },
  random_forest: { name: "Random Forest", blurb: "Bagged trees using Gini impurity." },
  xgboost: { name: "XGBoost", blurb: "Gradient-boosted trees for tabular data." },
  svm: { name: "Support Vector Machine", blurb: "Calibrated linear SVM; large files use a training subsample." }
};

const SECTION_TITLES = {
  patient: "Patient Information",
  admission: "Admission",
  clinical: "Clinical",
  history: "History",
  treatment: "Treatment",
  other: "Other Features"
};

const charts = {};
let selectedModels = new Set(Object.keys(MODEL_META));
let lastResults = {};

function $(id) { return document.getElementById(id); }

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (typeof value === "number") return value.toFixed(digits);
  return String(value);
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function showAlert(message, kind = "error") {
  const region = $("alert-region");
  region.hidden = !message;
  region.innerHTML = message ? `<div class="alert ${kind}">${message}</div>` : "";
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let data = null;
  try { data = await response.json(); } catch { data = null; }
  if (!response.ok) {
    const detail = data && data.detail ? data.detail : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return data;
}

function destroyChart(key) {
  if (charts[key]) {
    charts[key].destroy();
    delete charts[key];
  }
}

function makeChart(key, canvas, config) {
  destroyChart(key);
  if (!canvas) return;
  charts[key] = new Chart(canvas, config);
}

function palette() {
  return {
    navy: "#102A43",
    blue: "#1976D2",
    teal: "#159A9C",
    muted: "#9FB3C8",
    warn: "#D99000"
  };
}

function setStatus(datasetLoaded, modelTrained) {
  $("dataset-status").innerHTML = datasetLoaded
    ? '<span class="dot ok"></span> Loaded'
    : '<span class="dot idle"></span> Not loaded';
  $("model-status").innerHTML = modelTrained
    ? '<span class="dot ok"></span> Trained'
    : '<span class="dot idle"></span> Not trained';
}

function tableHtml(columns, rows, emptyText) {
  if (!rows || !rows.length) return `<p class="empty-inline">${emptyText}</p>`;
  const head = columns.map((col) => `<th>${col}</th>`).join("");
  const body = rows.map((row) => `<tr>${columns.map((col) => `<td>${row[col] ?? ""}</td>`).join("")}</tr>`).join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

async function refreshStatus() {
  const status = await api("/api/models/status");
  setStatus(status.dataset_loaded, status.model_trained);
  return status;
}

function renderOverview(data) {
  const s = data.status;
  const cards = [
    ["Dataset Records", s.rows ?? "—"],
    ["Features", s.features ?? "—"],
    ["Readmission Rate", s.readmission_rate != null ? pct(s.readmission_rate) : "—"],
    ["Active Model", s.active_model_name ?? "—"],
    ["F1 Score", s.f1 != null ? fmt(s.f1) : "—"],
    ["ROC-AUC", s.roc_auc != null ? fmt(s.roc_auc) : "—"]
  ];
  $("overview-stats").innerHTML = cards.map(([label, value]) =>
    `<div class="stat-card"><span>${label}</span><strong>${value}</strong></div>`
  ).join("");

  const distEmpty = $("overview-dist-empty");
  if (data.distribution) {
    distEmpty.hidden = true;
    makeChart("ov-dist", $("chart-overview-dist"), barConfig(data.distribution.labels, data.distribution.values, palette().blue));
  } else {
    distEmpty.hidden = false;
    destroyChart("ov-dist");
  }

  const modelEmpty = $("overview-models-empty");
  if (data.model_comparison.length) {
    modelEmpty.hidden = true;
    makeChart("ov-models", $("chart-overview-models"), barConfig(
      data.model_comparison.map((m) => m.name),
      data.model_comparison.map((m) => m.f1),
      palette().teal
    ));
  } else {
    modelEmpty.hidden = false;
    destroyChart("ov-models");
  }

  const featEmpty = $("overview-features-empty");
  if (data.top_features.length) {
    featEmpty.hidden = true;
    makeChart("ov-feat", $("chart-overview-features"), horizontalBar(
      data.top_features.map((f) => f.feature),
      data.top_features.map((f) => f.importance),
      palette().navy
    ));
  } else {
    featEmpty.hidden = false;
    destroyChart("ov-feat");
  }

  const recent = data.recent_predictions.map((row) => ({
    Timestamp: row.created_at,
    Model: MODEL_META[row.model_name]?.name || row.model_name,
    Prediction: row.prediction === 1 ? "Readmitted" : "Not Readmitted",
    Probability: pct(row.probability)
  }));
  $("overview-recent").innerHTML = tableHtml(
    ["Timestamp", "Model", "Prediction", "Probability"],
    recent,
    "No predictions stored yet."
  );
}

function barConfig(labels, values, color) {
  return {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: color, borderWidth: 0 }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, ticks: { color: "#627D98" } }, x: { ticks: { color: "#627D98", maxRotation: 45 } } }
    }
  };
}

function groupedBar(labels, a, b) {
  return {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Not Readmitted", data: a, backgroundColor: "#9FB3C8" },
        { label: "Readmitted", data: b, backgroundColor: "#1976D2" }
      ]
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom" } } }
  };
}

function horizontalBar(labels, values, color) {
  return {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: color }] },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true } }
    }
  };
}

function unavailable(title) {
  return `<article class="panel"><h3>${title}</h3><p class="empty-inline">Not available for this dataset.</p></article>`;
}

function chartPanel(title, canvasId) {
  return `<article class="panel"><h3>${title}</h3><div class="chart-wrap"><canvas id="${canvasId}"></canvas></div></article>`;
}

async function loadOverview() {
  const data = await api("/api/overview");
  renderOverview(data);
}

async function uploadFile(file) {
  const body = new FormData();
  body.append("file", file);
  showAlert("Uploading dataset…", "ok");
  const result = await api("/api/dataset/upload", { method: "POST", body });
  showAlert(`Loaded ${result.info.original_filename}.`, "ok");
  await refreshStatus();
  await showDataset(result.info);
}

async function showDataset(info) {
  const summary = $("dataset-summary");
  const mapping = info.target_mapping
    ? `${info.target_column}: 0 = Not Readmitted, 1 = Readmitted`
    : "Target not detected";
  summary.hidden = false;
  summary.innerHTML = [
    ["Filename", info.original_filename],
    ["Rows", info.rows],
    ["Columns", info.columns],
    ["Missing Values", info.missing_values],
    ["Duplicate Records", info.duplicates],
    ["Target Column", info.target_column || "Not found"]
  ].map(([k, v]) => `<div class="stat-card"><span>${k}</span><strong>${v}</strong></div>`).join("");
  $("process-btn").disabled = !info.target_column;
  $("process-note").textContent = mapping;

  const preview = await api("/api/dataset/preview");
  const rows = preview.rows.map((row) => {
    const out = {};
    preview.columns.forEach((col) => { out[col] = row[col]; });
    return out;
  });
  $("preview-table").innerHTML = tableHtml(preview.columns, rows, "No preview available.");
}

async function loadDatasetView() {
  const info = await api("/api/dataset/info");
  if (!info.loaded) {
    $("dataset-summary").hidden = true;
    $("process-btn").disabled = true;
    $("preview-table").innerHTML = '<p class="empty-inline">No file loaded.</p>';
    return;
  }
  await showDataset(info);
}

async function loadAnalysis() {
  const host = $("eda-panels");
  try {
    const eda = await api("/api/analytics/eda");
    const blocks = [];
    if (eda.readmission_distribution) blocks.push(chartPanel("Readmission Distribution", "eda-dist"));
    if (eda.age_distribution) blocks.push(chartPanel("Age Distribution", "eda-age"));
    else blocks.push(unavailable("Age Distribution"));
    if (eda.gender_analysis) blocks.push(chartPanel("Gender Analysis", "eda-gender"));
    else blocks.push(unavailable("Gender Analysis"));
    if (eda.length_of_stay) blocks.push(chartPanel("Length of Stay", "eda-los"));
    else blocks.push(unavailable("Length of Stay"));
    if (eda.medication_analysis) blocks.push(chartPanel("Medication Analysis", "eda-meds"));
    else blocks.push(unavailable("Medication Analysis"));
    if (eda.missing_values && !eda.missing_values.empty) blocks.push(chartPanel("Missing Values", "eda-missing"));
    else blocks.push(unavailable("Missing Values"));
    if (eda.correlation) {
      blocks.push(`<article class="panel wide"><h3>Correlation Heatmap</h3><div id="corr-heat"></div></article>`);
    } else {
      blocks.push(unavailable("Correlation Heatmap"));
    }
    host.innerHTML = blocks.join("");
    if (eda.readmission_distribution) {
      makeChart("eda-dist", $("eda-dist"), barConfig(eda.readmission_distribution.labels, eda.readmission_distribution.values, palette().blue));
    }
    if (eda.age_distribution) {
      makeChart("eda-age", $("eda-age"), groupedBar(eda.age_distribution.labels, eda.age_distribution.not_readmitted, eda.age_distribution.readmitted));
    }
    if (eda.gender_analysis) {
      makeChart("eda-gender", $("eda-gender"), groupedBar(eda.gender_analysis.labels, eda.gender_analysis.not_readmitted, eda.gender_analysis.readmitted));
    }
    if (eda.length_of_stay) {
      makeChart("eda-los", $("eda-los"), barConfig(eda.length_of_stay.labels, eda.length_of_stay.values, palette().teal));
    }
    if (eda.medication_analysis) {
      makeChart("eda-meds", $("eda-meds"), barConfig(eda.medication_analysis.labels, eda.medication_analysis.values, palette().navy));
    }
    if (eda.missing_values && !eda.missing_values.empty) {
      makeChart("eda-missing", $("eda-missing"), barConfig(eda.missing_values.labels, eda.missing_values.values, palette().warn));
    }
    if (eda.correlation) renderHeatmap($("corr-heat"), eda.correlation);
  } catch (err) {
    host.innerHTML = `<article class="panel"><p class="empty-inline">${err.message}</p></article>`;
  }
}

function renderHeatmap(el, corr) {
  const size = corr.features.length;
  let html = `<div style="overflow:auto"><table>`;
  html += `<tr><th></th>${corr.features.map((f) => `<th>${f}</th>`).join("")}</tr>`;
  for (let i = 0; i < size; i += 1) {
    html += `<tr><th>${corr.features[i]}</th>`;
    for (let j = 0; j < size; j += 1) {
      const v = corr.matrix[i][j];
      const t = (v + 1) / 2;
      const bg = `rgba(25, 118, 210, ${0.08 + t * 0.55})`;
      html += `<td style="background:${bg}; text-align:center">${Number(v).toFixed(2)}</td>`;
    }
    html += "</tr>";
  }
  html += "</table></div>";
  el.innerHTML = html;
}

async function loadFeatures() {
  try {
    const data = await api("/api/analytics/features");
    $("feature-model-label").textContent = `Active model: ${data.display_name}`;
    const rows = data.features || [];
    if (!rows.length) {
      $("feature-table").innerHTML = '<p class="empty-inline">No feature importance is available for this estimator.</p>';
      destroyChart("feat");
      return;
    }
    makeChart("feat", $("chart-features"), horizontalBar(rows.map((r) => r.feature), rows.map((r) => r.importance), palette().blue));
    $("feature-table").innerHTML = tableHtml(
      ["Feature", "Importance", "Type"],
      rows.map((r) => ({ Feature: r.feature, Importance: fmt(r.importance, 4), Type: r.type }))
    );
  } catch (err) {
    $("feature-model-label").textContent = "";
    $("feature-table").innerHTML = `<p class="empty-inline">${err.message}</p>`;
    destroyChart("feat");
  }
}

function renderModelCards(results) {
  lastResults = results.results || {};
  const host = $("model-cards");
  host.innerHTML = Object.entries(MODEL_META).map(([key, meta]) => {
    const row = lastResults[key];
    const trained = row && row.trained;
    const score = trained ? `F1 ${fmt(row.f1)}` : "Not trained";
    const checked = selectedModels.has(key) ? "checked" : "";
    const active = results.active_model === key ? " • Active" : "";
    return `<article class="model-card">
      <header>
        <label><input type="checkbox" data-model="${key}" ${checked}> ${meta.name}</label>
        <span class="score">${score}${active}</span>
      </header>
      <p>${meta.blurb}</p>
      <button class="btn btn-secondary" data-activate="${key}" ${trained ? "" : "disabled"}>Use as active model</button>
    </article>`;
  }).join("");
  host.querySelectorAll("input[type=checkbox]").forEach((box) => {
    box.addEventListener("change", () => {
      if (box.checked) selectedModels.add(box.dataset.model);
      else selectedModels.delete(box.dataset.model);
    });
  });
  host.querySelectorAll("[data-activate]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      try {
        await api("/api/models/select", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: btn.dataset.activate })
        });
        await loadModels();
        await refreshStatus();
      } catch (err) { showAlert(err.message); }
    });
  });

  $("selection-note").textContent = results.criterion_note || "Highest F1-score in the current evaluation";
  const tableRows = Object.keys(MODEL_META).map((key) => {
    const row = lastResults[key];
    if (!row || !row.trained) {
      return { Model: MODEL_META[key].name, Accuracy: "—", Precision: "—", Recall: "—", F1: "—", "ROC-AUC": "—" };
    }
    return {
      Model: MODEL_META[key].name,
      Accuracy: fmt(row.accuracy),
      Precision: fmt(row.precision),
      Recall: fmt(row.recall),
      F1: fmt(row.f1),
      "ROC-AUC": fmt(row.roc_auc)
    };
  });
  $("results-table").innerHTML = tableHtml(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], tableRows, "Train models to populate this table.");

  const active = results.active_model;
  const activeRow = active && lastResults[active];
  const matrixHost = $("confusion-view");
  if (activeRow && activeRow.confusion) {
    const c = activeRow.confusion;
    matrixHost.innerHTML = `<div class="confusion">
      <div></div><div class="label">Pred. 0</div><div class="label">Pred. 1</div>
      <div class="label">Actual 0</div><div class="cell">${c.tn}</div><div class="cell">${c.fp}</div>
      <div class="label">Actual 1</div><div class="cell">${c.fn}</div><div class="cell">${c.tp}</div>
    </div>`;
    makeChart("roc", $("chart-roc"), {
      type: "line",
      data: {
        labels: activeRow.roc.fpr,
        datasets: [{
          data: activeRow.roc.fpr.map((x, i) => ({ x, y: activeRow.roc.tpr[i] })),
          borderColor: palette().blue,
          pointRadius: 0,
          tension: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { type: "linear", min: 0, max: 1, title: { display: true, text: "False positive rate" } },
          y: { min: 0, max: 1, title: { display: true, text: "True positive rate" } }
        }
      }
    });
  } else {
    matrixHost.innerHTML = '<p class="empty-inline">Train a model to view the confusion matrix.</p>';
    destroyChart("roc");
  }

  const trained = Object.entries(lastResults).filter(([, row]) => row && row.trained);
  if (trained.length) {
    makeChart("cmp", $("chart-compare"), {
      type: "bar",
      data: {
        labels: trained.map(([key]) => MODEL_META[key].name),
        datasets: [
          { label: "F1", data: trained.map(([, row]) => row.f1), backgroundColor: palette().blue },
          { label: "ROC-AUC", data: trained.map(([, row]) => row.roc_auc), backgroundColor: palette().teal }
        ]
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom" } }, scales: { y: { beginAtZero: true, max: 1 } } }
    });
  } else {
    destroyChart("cmp");
  }
}

async function loadModels() {
  const results = await api("/api/models/results");
  renderModelCards(results);
}

async function loadPredict() {
  const host = $("predict-fields");
  try {
    const schema = await api("/api/schema");
    const groups = {};
    schema.fields.forEach((field) => {
      const key = field.section || "other";
      groups[key] = groups[key] || [];
      groups[key].push(field);
    });
    const order = ["patient", "admission", "clinical", "history", "treatment", "other"];
    host.innerHTML = order.filter((key) => groups[key]).map((key) => {
      const fields = groups[key].map((field) => {
        if (field.type === "numeric") {
          return `<label class="field">${field.label}<input name="${field.name}" type="number" step="any" value="${field.median ?? 0}"></label>`;
        }
        const options = (field.options || []).map((opt) => `<option ${opt === field.default ? "selected" : ""}>${opt}</option>`).join("");
        return `<label class="field">${field.label}<select name="${field.name}">${options}</select></label>`;
      }).join("");
      return `<section class="form-section"><h3>${SECTION_TITLES[key]}</h3><div class="fields">${fields}</div></section>`;
    }).join("") + `<div class="toolbar"><button class="btn btn-primary" type="submit">Estimate readmission</button></div>`;
  } catch (err) {
    host.innerHTML = `<article class="panel"><p class="empty-inline">${err.message}</p></article>`;
  }
}

function renderPrediction(result) {
  $("predict-result").innerHTML = `
    <h3>Estimated readmission status</h3>
    <p class="status-word">${result.label.toUpperCase()}</p>
    <p class="muted">Model probability</p>
    <p class="prob">${pct(result.probability)}</p>
    <p>Model: ${result.model_name}</p>
    <p>Timestamp: ${result.timestamp}</p>
    <p class="disclaimer">${result.disclaimer}</p>
  `;
}

async function loadHistory() {
  const data = await api("/api/predictions/history");
  if (!data.items.length) {
    $("history-table").innerHTML = '<p class="empty-inline">No stored predictions.</p>';
    return;
  }
  const rows = data.items.map((item) => ({
    Timestamp: item.timestamp,
    Model: item.model_name,
    Prediction: item.label,
    Probability: pct(item.probability),
    Action: `<button class="btn btn-danger" data-del="${item.id}">Delete</button>`
  }));
  $("history-table").innerHTML = tableHtml(["Timestamp", "Model", "Prediction", "Probability", "Action"], rows);
  $("history-table").querySelectorAll("[data-del]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      try {
        await api(`/api/predictions/history/${btn.dataset.del}`, { method: "DELETE" });
        await loadHistory();
      } catch (err) { showAlert(err.message); }
    });
  });
}

const viewLoaders = {
  overview: loadOverview,
  dataset: loadDatasetView,
  analysis: loadAnalysis,
  features: loadFeatures,
  models: loadModels,
  predict: loadPredict,
  history: loadHistory,
  about: async () => {}
};

async function showView(name) {
  document.querySelectorAll(".view").forEach((el) => el.classList.toggle("active", el.id === `view-${name}`));
  document.querySelectorAll(".nav-item").forEach((el) => el.classList.toggle("active", el.dataset.view === name));
  showAlert("");
  try {
    await refreshStatus();
    await viewLoaders[name]();
  } catch (err) {
    showAlert(err.message);
  }
}

function bindEvents() {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => showView(btn.dataset.view));
  });

  const drop = $("drop-zone");
  const input = $("file-input");
  drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("drag"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
  drop.addEventListener("drop", async (event) => {
    event.preventDefault();
    drop.classList.remove("drag");
    const file = event.dataTransfer.files[0];
    if (file) {
      try { await uploadFile(file); } catch (err) { showAlert(err.message); }
    }
  });
  input.addEventListener("change", async () => {
    if (input.files[0]) {
      try { await uploadFile(input.files[0]); } catch (err) { showAlert(err.message); }
      input.value = "";
    }
  });

  $("process-btn").addEventListener("click", async () => {
    try {
      showAlert("Processing dataset…", "ok");
      await api("/api/dataset/process", { method: "POST" });
      showAlert("Dataset processed. You can train models next.", "ok");
      await refreshStatus();
      await loadDatasetView();
    } catch (err) { showAlert(err.message); }
  });

  $("train-btn").addEventListener("click", async () => {
    try {
      showAlert("Training selected models. This may take a minute…", "ok");
      const payload = { models: Array.from(selectedModels) };
      await api("/api/models/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      showAlert("Training complete.", "ok");
      await refreshStatus();
      await loadModels();
    } catch (err) { showAlert(err.message); }
  });

  $("predict-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.target);
    const features = {};
    form.forEach((value, key) => { features[key] = value; });
    try {
      const result = await api("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features })
      });
      renderPrediction(result);
      showAlert("");
    } catch (err) { showAlert(err.message); }
  });
}

bindEvents();
showView("overview");
