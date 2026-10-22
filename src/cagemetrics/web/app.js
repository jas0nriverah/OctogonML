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
const state = { fighters: [], loaded: new Set(), recent: null, prediction: null, photos: new Map(), upcoming: null };
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
          : detail?.message || data.message || `Request failed (${response.status}).`,
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
      { upcoming: "2026 / UFC", explorer: "MATCHUP / LAB", recent: "RECENT / RESULTS", evidence: "MODEL / EVIDENCE", forward: "FORWARD / TRACKER" }[button.dataset.view];
    document.querySelector('.research-menu').open = false;
    if (button.dataset.view === "upcoming") loadUpcoming();
    if (button.dataset.view === "recent") loadRecent();
    if (button.dataset.view === "evidence") loadEvidence();
    if (button.dataset.view === "forward") loadForward();
  }),
);
async function initialize() {
  const results = await Promise.allSettled([
    api("/api/v1/overview"),
    api("/api/v1/fighters"),
    api("/api/v1/fighter-photos"),
  ]);
  if(results[2].status==='fulfilled'){
    const photos=results[2].value.fighters||{};
    (Array.isArray(photos)?photos:Object.values(photos)).forEach(fighter=>rememberPhoto(fighter));
  }
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
    state.fighters.forEach(fighter => rememberPhoto(fighter));
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
    addExplorerPhotos(prediction);
    state.prediction = prediction;
    addMatchupDisclosure(prediction);
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
    `<p class="section-copy">Historical report: ${escapeHtml((report.report_period || []).join(" → ") || "See evaluation splits below")}. These metrics evaluate time-separated historical artifacts. The serving artifact is separately versioned; inspect its fitting and calibration cutoffs below.</p><div class="metrics-grid">${metric("HOLDOUT FORECASTS", number(m.n, 0), "Chronological evaluation sample")}${metric("MODEL MAE", number(m.mae, 2), "Strikes per athlete")}${metric("BASELINE MAE", number(baseline.mae ?? baseline.mean_absolute_error, 2), "Compare on the same holdout")}${metric(`${number(m.level * 100, 0)}% INTERVAL COVERAGE`, percent(m.coverage), "Empirical holdout coverage")}</div><div class="evidence-columns"><div class="panel data-panel"><h3>Artifact & provenance</h3><dl class="metadata-list">${metadataRow("Model version", model.model_version || model.current_version || model.version || report.model_version)}${metadataRow("Model family", model.model_type || model.estimator || model.algorithm || model.champion)}${metadataRow("Training cutoff", date(model.training_cutoff || model.train_cutoff || model.data_through || report.training_cutoff))}${metadataRow("Calibration cutoff", date(model.calibration_cutoff))}${metadataRow("Serving channel", model.serving_channel)}${metadataRow("Data cutoff", date(model.data_cutoff || model.data_through))}${metadataRow("Built at", model.trained_at || model.created_at)}${metadataRow("Artifact SHA256", (typeof model.artifact_sha256 === "object" ? model.artifact_sha256?.current : model.artifact_sha256) || model.sha256 || model.artifact_hash)}${metadataRow("Feature schema", model.feature_schema_version || model.feature_version)}</dl></div><div class="panel data-panel"><h3>How to read these results</h3><ul class="limitation-list">${(Array.isArray(limitations) ? limitations : [limitations]).map((l) => `<li>${escapeHtml(l)}</li>`).join("")}</ul><p class="panel-description">MAE is the average absolute count error. Coverage is the fraction of observed outcomes inside the stated interval; compare it with the interval’s nominal level.</p></div></div>${evidenceDiagnostics(report)}${splits.length ? `<div class="panel data-panel"><h3>Time-based evaluation splits</h3><div class="table-scroll"><table><thead><tr><th>SPLIT</th><th>START</th><th>END</th><th>SAMPLES</th></tr></thead><tbody>${splits.map((s) => `<tr><td>${escapeHtml(s.name)}</td><td>${escapeHtml(date(s.start))}</td><td>${escapeHtml(date(s.end))}</td><td>${number(s.n ?? s.count, 0)}</td></tr>`).join("")}</tbody></table></div></div>` : ""}${features.length ? `<div class="panel data-panel"><h3>Inputs available before the fight</h3><p class="panel-description">Feature values are reconstructed using only eligible history before each forecast cutoff.</p><div class="feature-tags">${features.map((f) => `<span class="feature-tag">${escapeHtml(typeof f === "string" ? f : f.name)}</span>`).join("")}</div></div>` : ""}${benchmark && benchmark.status !== "not_measured" ? renderBenchmark(benchmark) : '<div class="panel data-panel"><h3>Runtime benchmark</h3><p class="panel-description">No benchmark artifact is available. Run the benchmark command to record measured latency and throughput.</p></div>'}<div class="panel data-panel"><h3>Inspect the service</h3><p class="panel-description">The same documented endpoints power this interface. Request schemas, validation rules, and response fields are available in the interactive OpenAPI reference.</p><a class="text-link" href="/docs">Open API documentation ↗</a></div>`;
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
function safeUrl(value) {
  if (typeof value !== "string") return null;
  try {
    const parsed = new URL(value, location.origin);
    return ["http:", "https:"].includes(parsed.protocol) ? parsed.href : null;
  } catch { return null; }
}
function factText(value) {
  if (value === null || value === undefined) return "Unavailable";
  if (typeof value === "object") return value.description || value.message || value.status || JSON.stringify(value);
  return String(value);
}
function addMatchupDisclosure(prediction) {
  const details = document.createElement("details");
  details.className = "matchup-disclosure panel";
  details.innerHTML = '<summary>Why this projection? <span class="text-link">Explore history + sensitivity</span></summary><div class="matchup-context"><p class="panel-description">Open to load pre-fight history and model sensitivity. This separate request does not affect the forecast endpoint benchmark.</p></div>';
  $("forecast-result").append(details);
  let loaded = false;
  details.addEventListener("toggle", async () => {
    if (!details.open || loaded) return;
    const container = details.querySelector(".matchup-context");
    container.innerHTML = '<div class="loading-state">Loading matchup context…</div>';
    try {
      const params = new URLSearchParams({fighter_id:prediction.fighter.id, opponent_id:prediction.opponent.id, scheduled_rounds:String(prediction.scheduled_rounds)});
      const context = await api(`/api/v1/matchups?${params}`);
      container.innerHTML = renderMatchupContext(context, prediction);
      loaded = true;
    } catch (error) { container.innerHTML = `<p class="error-message">${escapeHtml(error.message)}</p><p class="panel-description">Close and reopen this panel to retry.</p>`; }
  });
}
function renderMatchupContext(context, prediction) {
  const rows = context.comparison || [];
  const explanations = context.explanation || [];
  const history = (items, name) => `<div><h3>${escapeHtml(name)} · last ${items.length}</h3>${items.length ? `<div class="history-spark">${items.map(item => `<div><span style="height:${Math.max(3,Math.min(100,Number(item.strikes)/Math.max(...items.map(i=>Number(i.strikes)),1)*100))}%" title="${escapeHtml(date(item.event_date))}: ${escapeHtml(item.strikes)} strikes"></span><small>${number(item.strikes,0)}</small></div>`).join("")}</div><table><thead><tr><th>DATE / OPPONENT</th><th>LANDED</th></tr></thead><tbody>${items.map(item=>`<tr><td>${escapeHtml(date(item.event_date))}<small>${escapeHtml(item.opponent||item.opponent_name)}</small></td><td>${number(item.strikes,0)}</td></tr>`).join("")}</tbody></table>` : '<p class="panel-description">No eligible prior bouts.</p>'}</div>`;
  return `<p class="panel-description">Historical summaries provide context; they do not make this hypothetical matchup a scheduled fixture.</p><div class="history-grid">${history(context.fighter_recent||[],prediction.fighter.name)}${history(context.opponent_recent||[],prediction.opponent.name)}</div>${rows.length?`<div class="table-scroll"><table><thead><tr><th>PRE-FIGHT COMPARISON</th><th>${escapeHtml(prediction.fighter.name)}</th><th>${escapeHtml(prediction.opponent.name)}</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${escapeHtml(r.label)}<small>${escapeHtml(r.unit||"")}</small></td><td>${number(r.fighter,2)}</td><td>${number(r.opponent,2)}</td></tr>`).join("")}</tbody></table></div>`:""}${explanations.length?`<h3 class="sensitivity-heading">Model sensitivity, not causal attribution</h3><p class="panel-description">${escapeHtml(context.explanation_note||"Changes in predicted mean under controlled feature substitutions. These are model responses, not causal effects or a decomposition that must sum to the forecast.")}</p><div class="sensitivity-list">${explanations.map(e=>`<div><span>${escapeHtml(e.group)}</span><strong>${Number(e.delta_mean)>0?"+":""}${number(e.delta_mean)} strikes</strong><small>${escapeHtml(e.description)}</small></div>`).join("")}</div>`:""}`;
}
async function loadForward(force=false) {
  if (state.loaded.has("forward") && !force) return;
  try {
    const report = await api("/api/v1/forward");
    state.loaded.add("forward");
    const summary=report.summary||{}, batches=report.batches||[], upcoming=report.upcoming||{};
    const pending=summary.pending_count??summary.n_pending??summary.pending, scored=summary.scored_count??summary.n_settled??summary.scored;
    $("forward-content").innerHTML=`<div class="metrics-grid">${metric("FROZEN BATCHES",number(summary.batch_count??batches.length,0),"Persisted before scheduled events")}${metric("PENDING OUTCOMES",number(pending,0),"Unscored predictions")}${metric("SCORED OUTCOMES",number(scored,0),"Observed results linked later")}${metric("FORWARD MAE",number(summary.mae,2),"Unavailable until outcomes are scored")}</div>${batches.length?`<div class="panel data-panel"><h3>Immutable forecast batches</h3><div class="table-scroll"><table><thead><tr><th>BATCH</th><th>FROZEN AT</th><th>FIXTURES</th><th>SCORED</th><th>STATUS</th></tr></thead><tbody>${batches.map(batch=>`<tr><td>${escapeHtml(batch.title||batch.batch_id)}<small>${escapeHtml(batch.batch_id||batch.id)}</small></td><td>${escapeHtml(batch.created_at||batch.recorded_at||batch.frozen_at)}</td><td>${number(batch.fixture_count??batch.n_entries??batch.prediction_count??batch.n,0)}</td><td>${number(batch.scored_count??batch.n_settled,0)}</td><td>${escapeHtml(batch.status||batch.mode||"Recorded")}</td></tr>`).join("")}</tbody></table></div></div>`:`<div class="panel forward-empty"><span class="empty-symbol">↗</span><div><p class="eyebrow">NO FORWARD EVIDENCE YET</p><h3>The clean ledger is the starting point.</h3><p>Verify a future fixture, choose a line and record its source, then freeze the batch below. Results appear only after an observed outcome is linked. The recent replay has not been inserted here.</p></div></div>`}<div class="forward-guide"><span class="outlined-label">RECORD → WAIT → SCORE</span><p>${escapeHtml(factText(report.guide?.description||report.guide?.message||"Scheduled timestamps must be in the future when recorded. Predictions and source labels are frozen; outcomes can be attached later without rewriting the forecast."))}</p></div>${upcoming.fixtures?.length?`<div class="panel data-panel"><h3>Upcoming fixture references</h3><p class="panel-description">Verify scheduled time and athlete identities before creating a batch. Availability does not imply a line feed.</p><div class="table-scroll"><table><thead><tr><th>FIXTURE</th><th>SCHEDULED</th><th>REFERENCE</th></tr></thead><tbody>${upcoming.fixtures.map(f=>`<tr><td>${escapeHtml(f.fighter_name||f.fighter_id)} vs ${escapeHtml(f.opponent_name||f.opponent_id)}</td><td>${escapeHtml(f.scheduled_at||f.event_date)}</td><td>${safeUrl(f.source_url)?`<a href="${escapeHtml(safeUrl(f.source_url))}" target="_blank" rel="noopener">Source ↗</a>`:"Verify source"}</td></tr>`).join("")}</tbody></table></div></div>`:""}`;
  } catch (error) { $("forward-content").innerHTML=errorPanel(error.message); }
}
$("forward-use-scenario").addEventListener("click",()=>{
  const message=$("forward-message"), scheduled=$("forward-scheduled").value;
  if(!scheduled || !$("forward-title").value.trim() || !$("forward-line-source").value.trim()) { message.textContent="Enter a batch title, future schedule, and a public schedule URL before building the request.";return; }
  if(!$("fighter").value || !$("opponent").value || !state.fighters.length){message.textContent="Load a valid matchup in Forecast Explorer first.";return;}
  const time=new Date(scheduled);
  if(!Number.isFinite(time.getTime())||time<=new Date()){message.textContent="The scheduled time must be in the future. Historical replays belong in Recent Cards.";return;}
  const source=$("forward-line-source").value.trim();
  if(!/^https?:\/\//i.test(source)||!safeUrl(source)){message.textContent="Provide an HTTP or HTTPS public schedule URL.";return;}
  const identity=crypto.randomUUID();
  const request={title:$("forward-title").value.trim(),idempotency_key:identity,fixtures:[{fixture_id:`manual-${identity}`,scheduled_at:time.toISOString(),event_date:scheduled.slice(0,10),schedule_kind:"user_attested",source_url:source,fighter_id:$("fighter").value,opponent_id:$("opponent").value,scheduled_rounds:Number(document.querySelector('[name=scheduled_rounds]:checked').value),line:Number($("line").value),line_kind:$("forward-line-kind").value}]};
  $("forward-json").value=JSON.stringify(request,null,2);
  message.textContent="JSON prepared from the current explorer scenario. Verify the fixture and line provenance before freezing this batch.";
});
$("forward-file").addEventListener("change",async event=>{
  const file=event.target.files?.[0];if(!file)return;
  if(file.size>1000000){$("forward-message").textContent="Use a JSON file smaller than 1 MB.";return;}
  try {const parsed=JSON.parse(await file.text());$("forward-json").value=JSON.stringify(parsed,null,2);$("forward-message").textContent="File imported locally. Review the request before submitting.";}
  catch {$("forward-message").textContent="This file is not valid JSON.";}
});
$("forward-submit").addEventListener("click",async()=>{
  const button=$("forward-submit"),message=$("forward-message");
  try {
    const request=JSON.parse($("forward-json").value);
    if(!request.title?.trim()||!request.idempotency_key?.trim()||!Array.isArray(request.fixtures)||!request.fixtures.length)throw new Error("Provide a title, idempotency key, and at least one verified future fixture.");
    button.disabled=true;message.textContent="Freezing batch and model provenance…";
    const response=await api('/api/v1/forward/batches',{method:'POST',body:JSON.stringify(request)});
    message.textContent=`Batch recorded: ${response.batch_id||response.id||request.title}. Keep the idempotency key for safe retries. No outcome has been inferred from this submission.`;
    await loadForward(true);
  } catch(error){message.textContent=error.message;}
  finally{button.disabled=false;}
});
async function loadExperiments() {
  const container=document.createElement('div');container.id='experiments-content';container.className='panel data-panel';container.innerHTML='<h3>Challenger experiments</h3><p class="panel-description">Loading the saved experiment report…</p>';
  $('evidence-content').append(container);
  try {
    const report=await api('/api/v1/experiments');
    const rows=report.development_aggregate||report.experiments||report.candidates||report.results||[];
    const list=Array.isArray(rows)?rows:Object.entries(rows).map(([name,result])=>({name,...result}));
    container.innerHTML=`<h3>Challenger experiments</h3><p class="panel-description">${escapeHtml(factText(report.selection_decision||report.decision||report.summary||'Saved research runs. A candidate is not promoted simply because one metric improves.'))}</p><p class="panel-description">Selection sample: development years ${escapeHtml((report.selection_period||[]).join(" → ")||"see report")}. Lower negative log likelihood is the declared selection rule.</p>${list.length?`<div class="table-scroll"><table><thead><tr><th>CANDIDATE</th><th>SAMPLES</th><th>MAE</th><th>NLL</th><th>80% COVERAGE</th><th>STATUS</th></tr></thead><tbody>${list.map(row=>{const m=row.metrics||row.test||row;return `<tr><td>${escapeHtml(row.name||row.model||row.candidate)}<small>${escapeHtml(row.notes||row.description||'')}</small></td><td>${number(m.n??((report.development||[]).filter(d=>d.candidate===row.name).reduce((sum,d)=>sum+d.n,0)||null),0)}</td><td>${number(m.mae,2)}</td><td>${number(m.nll,3)}</td><td>${percent(m.coverage_80??m.interval_coverage)}</td><td>${escapeHtml(row.status||(row.name===report.selected_candidate?'Selected family':'Not promoted'))}</td></tr>`;}).join('')}</tbody></table></div>`:'<p class="panel-description">No completed candidate rows are present. The current production artifact is unchanged.</p>'}${report.retrospective?`<p class="error-message">${escapeHtml(report.retrospective_comparison_label||"The additional historical comparison reuses already-seen outcomes and is not a fresh holdout.")}</p>`:""}${report.warnings?.length?`<ul class="limitation-list">${report.warnings.map(w=>`<li>${escapeHtml(w)}</li>`).join("")}</ul>`:""}<details class="report-details"><summary>Inspect exact experiment report</summary><pre>${escapeHtml(JSON.stringify(report,null,2))}</pre></details>`;
  } catch(error){container.innerHTML=`<h3>Challenger experiments</h3><p class="panel-description">No saved experiment report is available: ${escapeHtml(error.message)}. Historical evidence above describes the current evaluated model family.</p>`;}
}
const evidenceObserver=new MutationObserver(()=>{if(state.loaded.has('evidence')&&!$('experiments-content'))loadExperiments();});
evidenceObserver.observe($('evidence-content'),{childList:true});
function rememberPhoto(fighter) {
  const id = fighter?.fighter_id || fighter?.id;
  const url = fighter?.photo_url || fighter?.image_url || fighter?.photo?.url;
  if (id && url) state.photos.set(String(id), {url, source: fighter.photo_source_url || fighter.profile_url || fighter.photo?.source_url});
}
function addExplorerPhotos(prediction) {
  const panel=$('forecast-result').querySelector('.result-panel');
  const strip=document.createElement('div');
  strip.className='explorer-athletes';
  const selected=[prediction.fighter,prediction.opponent].map(f=>({...state.fighters.find(item=>String(item.fighter_id)===String(f.id)),...f}));
  strip.innerHTML=`${selected.map((fighter,index)=>`<div>${athletePortrait(fighter,true)}<span>${escapeHtml(fighter.name)}</span><small>${index===0?'FORECAST ATHLETE':'OPPONENT'}</small></div>`).join('<span class="explorer-versus">VS</span>')}`;
  panel.prepend(strip);
  bindPortraits(strip);
  const notes=document.createElement('details');notes.className='forecast-notes';notes.innerHTML='<summary>Model & data notes</summary>';
  for(const selector of ['.warnings','.provenance']){const node=panel.querySelector(selector);if(node)notes.append(node);}
  panel.append(notes);
}
function athletePortrait(fighter, compact=false) {
  rememberPhoto(fighter);
  const identity = String(fighter.id || fighter.fighter_id || '');
  const photo = state.photos.get(identity) || {url:fighter.photo_url||fighter.image_url,source:fighter.photo_source_url};
  const url = photo?.url && safeUrl(photo.url);
  const initials = (fighter.name || fighter.fighter_name || 'Athlete').split(/\s+/).filter(Boolean).slice(0,2).map(word=>word[0]).join('');
  return `<div class="athlete-portrait ${compact?'compact':''}"><span class="portrait-initials" aria-hidden="true">${escapeHtml(initials)}</span>${url?`<img src="${escapeHtml(url)}" alt="${escapeHtml(fighter.name||fighter.fighter_name)}" loading="lazy" decoding="async">`:''}</div>`;
}
function bindPortraits(container) {
  container.querySelectorAll('.athlete-portrait img').forEach(img=>{
    const failed=()=>{img.remove();};
    img.addEventListener('error',failed,{once:true});
    if(img.complete && img.naturalWidth===0)failed();
  });
}
function fixtureAthlete(fixture, side) {
  const prefix=side==='fighter'?'fighter':'opponent';
  const nested=fixture[prefix] || fixture[side==='fighter'?'fighter_a':'fighter_b'] || {};
  const athlete={id:nested.id||nested.fighter_id||fixture[`${prefix}_id`]||fixture[side==='fighter'?'fighter_a_id':'fighter_b_id'],name:nested.name||fixture[`${prefix}_name`]||fixture[side==='fighter'?'fighter_a_name':'fighter_b_name']||'Athlete TBA',photo_url:nested.photo_url||nested.image_url||fixture[`${prefix}_photo_url`],photo_source_url:nested.photo_source_url||nested.profile_url||fixture[`${prefix}_photo_source_url`]};
  rememberPhoto(athlete);
  return athlete;
}
function calendarDate(value, options={}) {
  if(!value)return 'Date TBA';
  const parsed=new Date(`${String(value).slice(0,10)}T12:00:00`);
  return Number.isNaN(parsed.getTime())?date(value):parsed.toLocaleDateString(undefined,{month:'long',day:'numeric',...options});
}
function projectionFor(fixture, athlete, index) {
  const predictions=fixture.projections||fixture.predictions||[];
  if(Array.isArray(predictions))return predictions.find(p=>String(p.fighter_id||p.fighter?.id)===String(athlete.id))||predictions[index];
  return predictions[athlete.id]||predictions[index===0?'fighter':'opponent'];
}
function fighterForecast(fixture, athlete, index) {
  const projection=projectionFor(fixture,athlete,index);
  if(!projection || fixture.projection_status==='unsupported')return '<div class="athlete-estimate unavailable">Forecast unavailable</div>';
  return `<div class="athlete-estimate"><strong>${number(projection.mean)}</strong><span>expected strikes</span><small>${number(projection.interval_low,0)}–${number(projection.interval_high,0)} <span>${number((projection.interval_level??.8)*100,0)}% range</span></small></div>`;
}
function upcomingFightCard(fixture) {
  const fighter=fixtureAthlete(fixture,'fighter'),opponent=fixtureAthlete(fixture,'opponent');
  const rounds=fixture.scheduled_rounds;
  const source=safeUrl(fixture.source_url||fixture.event_source_url);
  const confidence=fixture.time_confidence||fixture.schedule_confidence||fixture.schedule_kind;
  const exactTime=fixture.scheduled_at && !['date_only','unknown','tba'].includes(String(confidence).toLowerCase());
  const time=exactTime?new Date(fixture.scheduled_at).toLocaleTimeString(undefined,{hour:'numeric',minute:'2-digit',timeZoneName:'short'}):'Time TBA';
  return `<article class="fight-card"><div class="fight-card-top"><span>${escapeHtml(fixture.division||fixture.weight_class||'UFC MATCHUP')}</span><span>${rounds?`${escapeHtml(rounds)} ROUNDS`:'ROUNDS TBA'}</span></div><div class="fight-athletes"><div class="fight-athlete">${athletePortrait(fighter)}<h3>${escapeHtml(fighter.name)}</h3>${fighterForecast(fixture,fighter,0)}</div><span class="card-versus">VS</span><div class="fight-athlete">${athletePortrait(opponent)}<h3>${escapeHtml(opponent.name)}</h3>${fighterForecast(fixture,opponent,1)}</div></div>${fixture.projection_status==='unsupported'?`<p class="card-omission">${escapeHtml(fixture.omission_reason||'This announced matchup does not yet have enough verified information for a forecast.')}</p>`:''}<div class="fight-card-footer"><span title="${escapeHtml(confidence||'Published card or segment time')}">Card start · ${escapeHtml(time)}</span>${source?`<a href="${escapeHtml(source)}" target="_blank" rel="noopener">Fight source ↗</a>`:'<span>Source unavailable</span>'}</div></article>`;
}
function renderUpcoming(report) {
  let fixtures=report.fixtures||[];
  if(!fixtures.length && Array.isArray(report.events))fixtures=report.events.flatMap(event=>(event.fixtures||event.bouts||[]).map(fixture=>({...event,...fixture,event_name:event.name||event.event_name,event_date:event.event_date||fixture.event_date})));
  fixtures=fixtures.filter(f=>!f.event_date||String(f.event_date).slice(0,10)<='2026-12-31').sort((a,b)=>String(a.event_date||a.scheduled_at).localeCompare(String(b.event_date||b.scheduled_at)));
  const grouped=new Map();
  fixtures.forEach(fixture=>{const key=fixture.event_id||`${date(fixture.event_date||fixture.scheduled_at)}:${fixture.event_name||''}`;if(!grouped.has(key))grouped.set(key,[]);grouped.get(key).push(fixture);});
  const issued=report.schedule_observed_at||report.observed_at||fixtures[0]?.observed_at||report.source_as_of||report.issued_at||report.as_of||report.retrieved_at;
  const scheduleNote=report.notice||report.description||report.warning||'Only sourced, announced matchups are shown. Unannounced cards and unresolved forecasts are not filled in.';
  if(!fixtures.length){$('upcoming-content').innerHTML=`<div class="panel upcoming-empty"><p class="eyebrow">ANNOUNCEMENTS, NOT GUESSES</p><h3>No verified upcoming matchups are available.</h3><p>${escapeHtml(scheduleNote)}</p><p class="panel-description">Try Matchup Lab to explore a hypothetical pairing, or return after the announced schedule is refreshed.</p></div>`;return;}
  const events=Array.from(grouped.values()).map(group=>group.slice().reverse());
  const filters=events.map((group,index)=>`<button class="event-filter ${index===0?'active':''}" data-event-index="${index}" aria-pressed="${index===0}"><strong>${escapeHtml(calendarDate(group[0].event_date||group[0].scheduled_at,{month:'short'}))}</strong><span>${escapeHtml(group[0].event_name||'UFC event')}</span></button>`).join('');
  const sections=events.map((group,index)=>{
    const first=group[0], source=safeUrl(first.event_source_url||first.source_url);
    const when=new Date(`${date(first.event_date||first.scheduled_at)}T12:00:00`);
    return `<section class="event-group" data-calendar-event="${index}" ${index>0?'hidden':''}><div class="event-heading"><div class="event-date"><strong>${escapeHtml(when.toLocaleDateString(undefined,{month:'short'}))}</strong><span>${escapeHtml(when.getDate()||'—')}</span></div><div><p class="eyebrow">${escapeHtml(calendarDate(first.event_date||first.scheduled_at,{weekday:'long',year:'numeric'}))}</p><h3>${escapeHtml(first.event_name||first.name||'UFC event')}</h3>${first.location||first.venue?`<p>${escapeHtml(first.location||first.venue)}</p>`:''}</div>${source?`<a class="event-source" href="${escapeHtml(source)}" target="_blank" rel="noopener">Announcement ↗</a>`:''}</div><div class="fight-card-grid">${group.map(upcomingFightCard).join('')}</div></section>`;
  }).join('');
  $('upcoming-content').innerHTML=`<div class="calendar-summary"><span>${grouped.size} announced ${grouped.size===1?'event':'events'} · ${fixtures.length} matchups</span><span>Snapshot ${escapeHtml(date(issued))}</span></div><div class="event-filters" aria-label="Choose an announced event">${filters}<button class="event-filter" data-event-index="all" aria-pressed="false"><strong>All cards</strong><span>Through Dec 31</span></button></div>${sections}<div class="calendar-disclosure"><p>Expected significant strikes landed per athlete, not a winner prediction. The 80% range describes possible full-fight outcomes. No market line or betting recommendation is shown.</p><details><summary>Schedule, photos & forecast details</summary><p>${escapeHtml(scheduleNote)}</p><p>Schedule snapshot ${escapeHtml(date(issued))}. Model history through ${escapeHtml(date(report.data_cutoff||report.history_cutoff))}. ${escapeHtml(report.model_version||'')}.</p><p>Fighter photos are sourced from ESPN athlete profiles where available. Missing or unavailable photos use initials. Images identify athletes; this independent project is not affiliated with UFC or ESPN. Published times refer to a card or segment, not individual bout walkouts.</p></details></div>`;
  bindPortraits($('upcoming-content'));
  $('upcoming-content').querySelectorAll('[data-event-index]').forEach(button=>button.addEventListener('click',()=>{
    const selected=button.dataset.eventIndex;
    $('upcoming-content').querySelectorAll('[data-event-index]').forEach(item=>{item.classList.toggle('active',item===button);item.setAttribute('aria-pressed',String(item===button));});
    $('upcoming-content').querySelectorAll('[data-calendar-event]').forEach(group=>group.hidden=selected!=='all'&&group.dataset.calendarEvent!==selected);
  }));
}
async function loadUpcoming() {
  if(state.loaded.has('upcoming'))return;
  try {const report=await api('/api/v1/upcoming');state.upcoming=report;renderUpcoming(report);state.loaded.add('upcoming');}
  catch(error){$('upcoming-content').innerHTML=`<div class="panel upcoming-empty"><h3>The upcoming schedule couldn’t be loaded.</h3><p>${escapeHtml(error.message)}</p><p class="panel-description">Matchup Lab and Recent Results remain available. No placeholder fights have been added.</p><button class="secondary-button" id="retry-upcoming">Try again</button></div>`;$('retry-upcoming').addEventListener('click',loadUpcoming);}
}
initialize();
loadUpcoming();
