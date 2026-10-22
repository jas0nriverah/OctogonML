"use strict";
const $ = (id) => document.getElementById(id);
const escapeHtml = (value) =>
  String(value ?? "—").replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ],
  );
const number = (value, digits = 1) =>
  Number.isFinite(Number(value)) && value !== null && value !== ""
    ? Number(value).toLocaleString(undefined, {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
      })
    : "—";
const percent = (value) =>
  value === null || value === undefined
    ? "—"
    : `${number(Number(value) * 100, 1)}%`;
const date = (value) => (value ? String(value).slice(0, 10) : "—");
const state = { fighters: [], loaded: new Set(), recent: null };
async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(
      `The service returned an unreadable response (${response.status}).`,
    );
  }
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((x) => x.msg).join("; ")
          : data.message || `Request failed (${response.status}).`,
    );
  }
  return data;
}
function errorPanel(message) {
  return `<div class="panel data-panel"><h3>We couldn’t load this view.</h3><p class="error-message">${escapeHtml(message)}</p><p class="panel-description">Check the API service, then refresh to try again.</p></div>`;
}
function metric(label, value, note = "") {
  return `<div class="metric-card"><span class="stat-label">${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(note)}</small></div>`;
}
function metricValues(data) {
  const m = data?.test || data?.holdout || data?.metrics || data || {};
  return {
    n: m.n ?? m.sample_size ?? m.count ?? m.n_predictions ?? m.athlete_count,
    mae: m.mae ?? m.mean_absolute_error,
    rmse: m.rmse,
    coverage: m.interval_coverage ?? m.coverage ?? m.coverage_80,
    level: m.interval_level ?? 0.8,
  };
}
document.querySelectorAll("[data-view]").forEach((button) =>
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-view]").forEach((b) => {
      const selected = b === button;
      b.classList.toggle("active", selected);
      b.setAttribute("aria-selected", String(selected));
    });
    document
      .querySelectorAll(".view")
      .forEach(
        (view) => (view.hidden = view.id !== `view-${button.dataset.view}`),
      );
    document.querySelector(".note-number").textContent =
      { explorer: "01", recent: "02", evidence: "03" }[button.dataset.view] +
      " / 03";
    if (button.dataset.view === "recent") loadRecent();
    if (button.dataset.view === "evidence") loadEvidence();
  }),
);
async function initialize() {
  const results = await Promise.allSettled([
    api("/api/v1/overview"),
    api("/api/v1/fighters"),
  ]);
  if (results[0].status === "fulfilled") {
    const overview = results[0].value;
    const d = overview.dataset || {},
      m = overview.model || {};
    $("summary-bouts").textContent = number(d.bout_count, 0);
    $("summary-fighters").textContent = number(d.fighter_count, 0);
    $("summary-date").textContent = date(d.date_end || d.data_cutoff);
    $("summary-model").textContent =
      m.model_version || m.current_version || m.version || "Available";
    $("service-status").textContent = "MODEL READY";
    if (d.source_names?.length)
      $("source-note").textContent =
        `Sources: ${d.source_names.join(" · ")} · Independent historical analysis.`;
  } else {
    $("service-status").textContent = "SERVICE UNAVAILABLE";
  }
  if (results[1].status === "fulfilled") {
    state.fighters = results[1].value.fighters || [];
    if (state.fighters.length < 2) {
      showFormError(
        "At least two athletes are required. Run the data pipeline before forecasting.",
      );
      return;
    }
    const options = state.fighters
      .map(
        (f) =>
          `<option value="${escapeHtml(f.fighter_id)}">${escapeHtml(f.name)}${f.bout_count !== undefined ? ` · ${f.bout_count} bouts` : ""}</option>`,
      )
      .join("");
    $("fighter").innerHTML = options;
    $("opponent").innerHTML = options;
    $("opponent").selectedIndex = 1;
    const preferredFighter = state.fighters.find(
        (f) => f.name === "Max Holloway",
      ),
      preferredOpponent = state.fighters.find((f) => f.name === "Ilia Topuria");
    if (preferredFighter && preferredOpponent) {
      $("fighter").value = preferredFighter.fighter_id;
      $("opponent").value = preferredOpponent.fighter_id;
    }
    ["fighter", "opponent", "forecast-button"].forEach(
      (id) => ($(id).disabled = false),
    );
    enforceDistinct();
  } else {
    showFormError(results[1].reason.message);
  }
}
function showFormError(message) {
  $("form-error").textContent = message;
  $("form-error").hidden = false;
}
function enforceDistinct(event) {
  if ($("fighter").value === $("opponent").value) {
    const target =
      event?.target?.id === "opponent" ? $("fighter") : $("opponent");
    target.value = state.fighters.find(
      (f) =>
        String(f.fighter_id) !== String((event?.target || $("fighter")).value),
    ).fighter_id;
  }
  for (const id of ["fighter", "opponent"])
    for (const option of $(id).options)
      option.disabled =
        option.value === $(id === "fighter" ? "opponent" : "fighter").value;
}
$("fighter").addEventListener("change", enforceDistinct);
$("opponent").addEventListener("change", enforceDistinct);
$("forecast-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("form-error").hidden = true;
  const button = $("forecast-button");
  button.disabled = true;
  button.textContent = "Calculating forecast…";
  try {
    const prediction = await api("/api/v1/predictions", {
      method: "POST",
      body: JSON.stringify({
        fighter_id: $("fighter").value,
        opponent_id: $("opponent").value,
        scheduled_rounds: Number(
          document.querySelector("[name=scheduled_rounds]:checked").value,
        ),
        line: Number($("line").value),
      }),
    });
    renderPrediction(prediction);
  } catch (error) {
    showFormError(error.message);
  } finally {
    button.disabled = false;
    button.innerHTML = "Generate forecast <span>↗</span>";
  }
});
function distributionChart(prediction) {
  let points = (prediction.distribution || [])
    .map((p) => ({ x: Number(p.count), p: Number(p.probability) }))
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.p) && p.p >= 0)
    .sort((a, b) => a.x - b.x);
  if (!points.length)
    return '<p class="panel-description">Distribution detail is unavailable for this artifact.</p>';
  const width = 650,
    height = 205,
    left = 28,
    right = 12,
    top = 18,
    bottom = 34;
  const max = Math.max(...points.map((p) => p.p), 0.001);
  const upper = Math.max(
    points.at(-1).x,
    Number(prediction.line) * 1.05,
    Number(prediction.interval_high) * 1.05,
    10,
  );
  const x = (v) => left + (v / upper) * (width - left - right),
    y = (v) => height - bottom - (v / max) * (height - top - bottom);
  const groups = [];
  const stride = Math.max(1, Math.ceil(points.length / 130));
  for (let i = 0; i < points.length; i += stride) {
    const group = points.slice(i, i + stride);
    groups.push({
      x: group.reduce((sum, p) => sum + p.x, 0) / group.length,
      p: group.reduce((sum, p) => sum + p.p, 0) / group.length,
    });
  }
  const barWidth = Math.max(
    1,
    ((width - left - right) / Math.max(groups.length, 1)) * 0.83,
  );
  const bars = groups
    .map(
      (p) =>
        `<rect x="${x(p.x) - barWidth / 2}" y="${y(p.p)}" width="${barWidth}" height="${height - bottom - y(p.p)}" fill="${p.x > prediction.line ? "#d4ec89" : "#52616b"}" opacity=".88"/>`,
    )
    .join("");
  const ticks = Array.from({ length: 6 }, (_, i) => (upper * i) / 5);
  return `<svg class="distribution-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Probability distribution of significant strikes. Line ${escapeHtml(prediction.line)}; interval ${escapeHtml(prediction.interval_low)} to ${escapeHtml(prediction.interval_high)}."><rect x="${x(prediction.interval_low)}" y="${top}" width="${Math.max(0, x(prediction.interval_high) - x(prediction.interval_low))}" height="${height - top - bottom}" fill="#d4ec89" opacity=".04"/><line x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}" stroke="#46535a"/>${bars}<line x1="${x(prediction.line)}" y1="${top - 5}" x2="${x(prediction.line)}" y2="${height - bottom}" stroke="#eca771" stroke-dasharray="4 3"/><text x="${Math.min(width - 70, Math.max(left, x(prediction.line) + 6))}" y="12" fill="#eca771" font-size="9">LINE ${escapeHtml(prediction.line)}</text>${ticks.map((t) => `<text x="${x(t)}" y="${height - 15}" text-anchor="middle" fill="#a2adb3" font-size="9">${Math.round(t)}</text>`).join("")}<line x1="${x(prediction.interval_low)}" y1="${height - 3}" x2="${x(prediction.interval_high)}" y2="${height - 3}" stroke="#d4ec89" stroke-width="2"/></svg>`;
}
function renderPrediction(p) {
  const name = p.fighter?.name || "Selected athlete",
    opponent = p.opponent?.name || "Opponent";
  $("forecast-result").innerHTML =
    `<div class="panel result-panel"><div class="result-heading"><div><p class="eyebrow">SIGNIFICANT STRIKES LANDED</p><h2>${escapeHtml(name)}</h2><p class="result-subtitle">vs ${escapeHtml(opponent)} · ${escapeHtml(p.scheduled_rounds)} scheduled rounds</p></div><span class="pill">MODEL FORECAST</span></div><div class="forecast-numbers"><div><span class="stat-label">EXPECTED STRIKES</span><strong class="big-stat">${number(p.mean)}</strong><span class="stat-footnote">Median ${number(p.median, 0)} strikes</span></div><div><span class="stat-label">${number((p.interval_level ?? 0.8) * 100, 0)}% PREDICTION INTERVAL</span><strong class="big-stat interval">${number(p.interval_low, 0)}–${number(p.interval_high, 0)}</strong><span class="stat-footnote">Range of possible fight outcomes</span></div></div><div class="chart-wrap"><p class="chart-title">COUNT DISTRIBUTION / STRIKES LANDED</p>${distributionChart(p)}</div><div class="chart-legend"><span><i class="legend-dot"></i>Below line</span><span><i class="legend-dot over"></i>Above line</span><span><i class="legend-line"></i>Your line</span></div><div class="probability-grid"><div class="probability-card"><span>OVER ${escapeHtml(p.line)}</span><strong>${percent(p.p_over)}</strong></div><div class="probability-card under"><span>UNDER ${escapeHtml(p.line)}</span><strong>${percent(p.p_under)}</strong></div></div><p class="push-note">Exact-line probability: ${percent(p.p_push)}${Number.isInteger(p.line) ? " · Integer lines can push." : " · Half-point lines cannot push."}</p>${p.warnings?.length ? `<ul class="warnings">${p.warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}</ul>` : ""}<div class="provenance">MODEL ${escapeHtml(p.model_version)} · FEATURES THROUGH ${escapeHtml(date(p.feature_cutoff))}<br>Data through ${escapeHtml(date(p.data_cutoff))} · History: ${escapeHtml(p.history?.fighter_bouts ?? "—")} athlete / ${escapeHtml(p.history?.opponent_bouts ?? "—")} opponent bouts</div></div>`;
}
async function loadRecent() {
  if (state.loaded.has("recent")) return;
  try {
    const data = await api("/api/v1/recent");
    state.recent = data;
    state.loaded.add("recent");
    $("recent-period").textContent =
      `${date(data.from || "2026-10-08")} → ${date(data.to || "2026-10-22")}`;
    const rows = data.predictions || [],
      m = metricValues(data);
    if (!rows.length) {
      $("recent-content").innerHTML =
        '<div class="panel data-panel"><h3>No eligible fights in this window.</h3><p class="panel-description">The replay is empty because no eligible completed bouts were found. No results have been substituted from another period.</p></div>';
      return;
    }
    $("recent-content").innerHTML =
      `<div class="metrics-grid">${metric("ATHLETE FORECASTS", number(m.n ?? rows.length, 0), "Two athletes can share one bout")}${metric("MEAN ABSOLUTE ERROR", number(m.mae), "Strikes per athlete")}${metric("ROOT MEAN SQUARE ERROR", number(m.rmse), "Larger misses carry more weight")}${metric(`${number(m.level * 100, 0)}% INTERVAL COVERAGE`, percent(m.coverage), "Observed share inside the interval")}</div><div class="panel data-panel"><h3>Forecasts meet the scorecard</h3><p class="panel-description">Each dot is one athlete forecast. The diagonal marks a perfect count prediction.</p>${replayChart(rows)}</div><div class="panel data-panel"><div class="panel-heading"><h3>Fight-by-fight replay</h3><span class="small-label">SYNTHETIC LINES · NO ROI CLAIM</span></div><div class="table-scroll"><table><thead><tr><th>ATHLETE / OPPONENT</th><th>DATE</th><th>EXPECTED</th><th>ACTUAL</th><th>INTERVAL</th><th>ABS. ERROR</th><th>LINE</th><th>P(OVER)</th></tr></thead><tbody>${rows.map((r) => `<tr><td>${escapeHtml(r.fighter_name)}<small>vs ${escapeHtml(r.opponent_name)}</small></td><td>${escapeHtml(date(r.event_date))}</td><td>${number(r.mean)}</td><td class="positive">${number(r.actual, 0)}</td><td>${number(r.interval_low, 0)}–${number(r.interval_high, 0)}</td><td>${number(r.absolute_error ?? Math.abs(r.actual - r.mean))}</td><td>${number(r.line)}</td><td>${percent(r.p_over)}</td></tr>`).join("")}</tbody></table></div></div><p class="section-copy">${escapeHtml(data.description || data.warning || "Retrospective replay using pre-window training and pre-bout features. Lines are synthetic; outcomes are observed historical counts.")}</p>`;
  } catch (error) {
    $("recent-content").innerHTML = errorPanel(error.message);
  }
}
function replayChart(rows) {
  const points = rows.filter(
    (r) => Number.isFinite(Number(r.actual)) && Number.isFinite(Number(r.mean)),
  );
  if (!points.length) return "";
  const w = 850,
    h = 260,
    l = 48,
    r = 25,
    t = 15,
    b = 40,
    max =
      Math.ceil(
        Math.max(
          ...points.flatMap((p) => [Number(p.mean), Number(p.actual)]),
          20,
        ) / 20,
      ) * 20;
  const x = (v) => l + (v / max) * (w - l - r),
    y = (v) => h - b - (v / max) * (h - t - b);
  const ticks = Array.from({ length: 5 }, (_, i) => (max * i) / 4);
  return `<svg class="replay-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Scatter plot of predicted versus actual significant strikes for ${points.length} athletes">${ticks.map((v) => `<line x1="${l}" x2="${w - r}" y1="${y(v)}" y2="${y(v)}" stroke="#35424a"/><text x="${l - 10}" y="${y(v) + 3}" fill="#a2adb3" text-anchor="end" font-size="9">${Math.round(v)}</text><text x="${x(v)}" y="${h - 23}" fill="#a2adb3" text-anchor="middle" font-size="9">${Math.round(v)}</text>`).join("")}<line x1="${x(0)}" y1="${y(0)}" x2="${x(max)}" y2="${y(max)}" stroke="#a2adb3" stroke-dasharray="4 5"/>${points.map((p) => `<circle cx="${x(Number(p.mean))}" cy="${y(Number(p.actual))}" r="4" fill="#d4ec89" fill-opacity=".72"><title>${escapeHtml(p.fighter_name)}: expected ${number(p.mean)}, actual ${number(p.actual, 0)}</title></circle>`).join("")}<text x="${w / 2}" y="${h - 2}" fill="#a2adb3" font-size="9" text-anchor="middle">EXPECTED STRIKES</text><text transform="translate(10 ${h / 2}) rotate(-90)" fill="#a2adb3" font-size="9" text-anchor="middle">ACTUAL STRIKES</text></svg>`;
}
function metadataRow(label, value) {
  return `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value ?? "—")}</dd></div>`;
}
async function loadEvidence() {
  if (state.loaded.has("evidence")) return;
  const results = await Promise.allSettled([
    api("/api/v1/evaluation"),
    api("/api/v1/models/current"),
    api("/api/v1/benchmark"),
  ]);
  if (results[0].status === "rejected" && results[1].status === "rejected") {
    $("evidence-content").innerHTML = errorPanel(results[0].reason.message);
    return;
  }
  state.loaded.add("evidence");
  const report = results[0].status === "fulfilled" ? results[0].value : {},
    model = results[1].status === "fulfilled" ? results[1].value : {},
    benchmark = results[2].status === "fulfilled" ? results[2].value : null;
  const m = metricValues(report),
    baseline = report.baseline || report.baseline_metrics || {},
    features =
      model.feature_names || model.features || report.feature_names || [],
    limitations = model.limitations ||
      report.limitations || [
        "Historical performance does not guarantee future accuracy.",
        "Sparse athlete histories and changing styles can increase forecast uncertainty.",
        "Synthetic strike lines measure model behavior; they do not establish profitability.",
      ];
  const splits = report.splits || [];
  $("evidence-content").innerHTML =
    `<p class="section-copy">Historical report: ${escapeHtml((report.report_period || []).join(" → ") || "See evaluation splits below")}. These metrics evaluate time-separated historical artifacts; the current explorer artifact is refitted on all available history.</p><div class="metrics-grid">${metric("HOLDOUT FORECASTS", number(m.n, 0), "Chronological evaluation sample")}${metric("MODEL MAE", number(m.mae, 2), "Strikes per athlete")}${metric("BASELINE MAE", number(baseline.mae ?? baseline.mean_absolute_error, 2), "Compare on the same holdout")}${metric(`${number(m.level * 100, 0)}% INTERVAL COVERAGE`, percent(m.coverage), "Empirical holdout coverage")}</div><div class="evidence-columns"><div class="panel data-panel"><h3>Artifact & provenance</h3><dl class="metadata-list">${metadataRow("Model version", model.model_version || model.current_version || model.version || report.model_version)}${metadataRow("Model family", model.model_type || model.estimator || model.algorithm || model.champion)}${metadataRow("Training cutoff", date(model.training_cutoff || model.train_cutoff || model.data_through || report.training_cutoff))}${metadataRow("Data cutoff", date(model.data_cutoff || model.data_through))}${metadataRow("Built at", model.trained_at || model.created_at)}${metadataRow("Artifact SHA256", (typeof model.artifact_sha256 === "object" ? model.artifact_sha256?.current : model.artifact_sha256) || model.sha256 || model.artifact_hash)}${metadataRow("Feature schema", model.feature_schema_version || model.feature_version)}</dl></div><div class="panel data-panel"><h3>How to read these results</h3><ul class="limitation-list">${(Array.isArray(limitations) ? limitations : [limitations]).map((l) => `<li>${escapeHtml(l)}</li>`).join("")}</ul><p class="panel-description">MAE is the average absolute count error. Coverage is the fraction of observed outcomes inside the stated interval; compare it with the interval’s nominal level.</p></div></div>${evidenceDiagnostics(report)}${splits.length ? `<div class="panel data-panel"><h3>Time-based evaluation splits</h3><div class="table-scroll"><table><thead><tr><th>SPLIT</th><th>START</th><th>END</th><th>SAMPLES</th></tr></thead><tbody>${splits.map((s) => `<tr><td>${escapeHtml(s.name)}</td><td>${escapeHtml(date(s.start))}</td><td>${escapeHtml(date(s.end))}</td><td>${number(s.n ?? s.count, 0)}</td></tr>`).join("")}</tbody></table></div></div>` : ""}${features.length ? `<div class="panel data-panel"><h3>Inputs available before the fight</h3><p class="panel-description">Feature values are reconstructed using only eligible history before each forecast cutoff.</p><div class="feature-tags">${features.map((f) => `<span class="feature-tag">${escapeHtml(typeof f === "string" ? f : f.name)}</span>`).join("")}</div></div>` : ""}${benchmark && benchmark.status !== "not_measured" ? renderBenchmark(benchmark) : '<div class="panel data-panel"><h3>Runtime benchmark</h3><p class="panel-description">No benchmark artifact is available. Run the benchmark command to record measured latency and throughput.</p></div>'}<div class="panel data-panel"><h3>Inspect the service</h3><p class="panel-description">The same documented endpoints power this interface. Request schemas, validation rules, and response fields are available in the interactive OpenAPI reference.</p><a class="text-link" href="/docs">Open API documentation ↗</a></div>`;
}
function renderBenchmark(b) {
  return `<div class="panel data-panel"><h3>Measured serving performance</h3><dl class="metadata-list">${metadataRow("Requests", b.requests ?? b.n ?? b.sample_count)}${metadataRow("Concurrency", b.concurrency)}${metadataRow("p50 latency", b.p50_ms !== undefined ? `${number(b.p50_ms)} ms` : undefined)}${metadataRow("p95 latency", b.p95_ms !== undefined ? `${number(b.p95_ms)} ms` : undefined)}${metadataRow("Throughput", b.requests_per_second !== undefined ? `${number(b.requests_per_second)} requests/s` : undefined)}${metadataRow("Error rate", percent(b.error_rate))}${metadataRow("Measured at", b.measured_at)}${metadataRow("Environment", b.environment || b.hardware)}</dl><p class="panel-description">${escapeHtml(b.workload || "Performance is specific to the recorded environment and workload.")} ${escapeHtml(b.notes || "")}</p></div>`;
}
function evidenceDiagnostics(report) {
  const comparisons = report.metrics || {},
    groups = report.subgroups || {},
    calibration = report.calibration || [];
  let html = "";
  if (
    Object.values(comparisons).some(
      (m) => m && typeof m === "object" && m.mae !== undefined,
    )
  )
    html += `<div class="panel data-panel"><h3>Same sample. Different models.</h3><p class="panel-description">Held-out count errors and distribution quality. Lower MAE and negative log likelihood are better; coverage should approach 80%.</p><div class="table-scroll"><table><thead><tr><th>MODEL</th><th>SAMPLES</th><th>MAE</th><th>RMSE</th><th>NEG. LOG LIKELIHOOD</th><th>80% COVERAGE</th></tr></thead><tbody>${Object.entries(
      comparisons,
    )
      .filter(([, m]) => m && typeof m === "object" && m.mae !== undefined)
      .map(
        ([name, m]) =>
          `<tr><td>${escapeHtml(name)}${name === report.champion ? " · selected" : ""}</td><td>${number(m.n, 0)}</td><td>${number(m.mae, 2)}</td><td>${number(m.rmse, 2)}</td><td>${number(m.nll, 3)}</td><td>${percent(m.interval_coverage ?? m.coverage_80)}</td></tr>`,
      )
      .join("")}</tbody></table></div></div>`;
  if (calibration.length) {
    const w = 480,
      h = 260,
      l = 42,
      r = 20,
      t = 15,
      b = 35,
      x = (v) => l + v * (w - l - r),
      y = (v) => h - b - v * (h - t - b);
    html += `<div class="evidence-columns"><div class="panel data-panel"><h3>Probability calibration</h3><p class="panel-description">Synthetic-line over probabilities grouped into bins. Larger dots contain more forecasts. The diagonal marks agreement.</p><svg class="replay-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Predicted over probability versus observed frequency">${[0, 0.25, 0.5, 0.75, 1].map((v) => `<line x1="${l}" x2="${w - r}" y1="${y(v)}" y2="${y(v)}" stroke="#35424a"/><text x="${l - 8}" y="${y(v) + 3}" text-anchor="end" fill="#a2adb3" font-size="9">${Math.round(v * 100)}%</text><text x="${x(v)}" y="${h - 18}" text-anchor="middle" fill="#a2adb3" font-size="9">${Math.round(v * 100)}%</text>`).join("")}<line x1="${x(0)}" x2="${x(1)}" y1="${y(0)}" y2="${y(1)}" stroke="#a2adb3" stroke-dasharray="4 4"/>${calibration.map((c) => `<circle cx="${x(c.predicted)}" cy="${y(c.observed)}" r="${Math.min(12, 3 + Math.sqrt(c.n) / 3)}" fill="#d4ec89" fill-opacity=".75"><title>Predicted ${percent(c.predicted)}, observed ${percent(c.observed)}, n=${c.n}</title></circle>`).join("")}<text x="${w / 2}" y="${h - 1}" text-anchor="middle" fill="#a2adb3" font-size="9">PREDICTED PROBABILITY</text></svg></div><div class="panel data-panel"><h3>Where forecasts are harder</h3><p class="panel-description">Subgroup outcomes on the historical report sample. Smaller groups are less stable.</p><div class="table-scroll"><table><thead><tr><th>GROUP</th><th>N</th><th>MAE</th><th>COVERAGE</th></tr></thead><tbody>${Object.entries(
      groups,
    )
      .map(
        ([name, m]) =>
          `<tr><td>${escapeHtml(name.replaceAll("_", " "))}</td><td>${number(m.n, 0)}</td><td>${number(m.mae)}</td><td>${percent(m.coverage_80)}</td></tr>`,
      )
      .join("")}</tbody></table></div></div></div>`;
  }
  return html;
}
for (const id of ["fighter", "opponent"]) {
  $(`${id}-search`).addEventListener("input", (event) => {
    const selection = $(id).value,
      other = $(id === "fighter" ? "opponent" : "fighter").value,
      query = event.target.value.trim().toLowerCase();
    const filtered = state.fighters.filter(
      (f) =>
        f.name.toLowerCase().includes(query) ||
        String(f.fighter_id) === selection,
    );
    $(id).innerHTML = filtered
      .map(
        (f) =>
          `<option value="${escapeHtml(f.fighter_id)}" ${String(f.fighter_id) === other ? "disabled" : ""}>${escapeHtml(f.name)}${f.bout_count !== undefined ? ` · ${f.bout_count} bouts` : ""}</option>`,
      )
      .join("");
    $(id).value = selection;
  });
}
initialize();
