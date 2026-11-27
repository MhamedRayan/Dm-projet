const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const EXAMPLES = {
  fail: {
    code: `import os


def run_user_command():
    cmd = input("command: ")
    os.system(cmd)
`,
    metrics: {
      loc: 200, vg: 25, evg: 5, ivg: 8, n: 600,
      v: 2500.0, l: 30, d: 5, i: 10, e: 80000.0,
      b: 3, t: 20, locode: 150, locomment: 20,
      loccodeandcomment: 0, loblank: 30, uniq_op: 25,
      uniq_opnd: 30, total_op: 300, total_opnd: 120,
      branchcount: 50,
    },
  },
  pass: {
    code: `def add(a, b):
    return a + b
`,
    metrics: {
      loc: 8, vg: 1, evg: 1, ivg: 1, n: 20,
      v: 90.0, l: 2, d: 1, i: 1, e: 100.0,
      b: 0.01, t: 1.0, locode: 6, locomment: 1,
      loccodeandcomment: 0, loblank: 1, uniq_op: 3,
      uniq_opnd: 4, total_op: 8, total_opnd: 10,
      branchcount: 1,
    },
  },
};

function setupTabs(attr, paneAttr) {
  $$(`[${attr}]`).forEach((tab) => {
    tab.addEventListener("click", () => {
      const mode = tab.getAttribute(attr);
      $$(`[${attr}]`).forEach((item) => {
        const active = item === tab;
        item.classList.toggle("is-active", active);
        item.setAttribute("aria-selected", String(active));
      });
      $$(`[${paneAttr}]`).forEach((pane) => {
        pane.classList.toggle("is-active", pane.getAttribute(paneAttr) === mode);
      });
    });
  });
}

setupTabs("data-code-mode", "data-code-pane");
setupTabs("data-metrics-mode", "data-metrics-pane");

function showError(message) {
  const el = $("#formError");
  el.textContent = message;
  el.hidden = false;
}

function hideError() {
  $("#formError").hidden = true;
}

function updateCount() {
  const count = $("#codeInput").value.length;
  $("#codeCount").textContent =
    `${count.toLocaleString()} character${count === 1 ? "" : "s"}`;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function setChip(id, text) {
  const chip = $(`#${id}`);
  chip.textContent = text;
  chip.hidden = false;
}

async function handleFile(file, kind) {
  const limit = kind === "code" ? 2 * 1024 * 1024 : 512 * 1024;
  if (file.size > limit) {
    showError(`${file.name} is too large (max ${formatBytes(limit)}).`);
    return;
  }

  let text;
  try {
    text = await file.text();
  } catch {
    showError(`Could not read ${file.name}.`);
    return;
  }

  if (kind === "code") {
    $("#codeInput").value = text;
    updateCount();
    setChip("codeFileChip", `${file.name} · ${formatBytes(file.size)}`);
  } else {
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch {
      showError("Metrics file is not valid JSON.");
      return;
    }
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      showError("Metrics JSON must be an object.");
      return;
    }
    $("#metricsInput").value = JSON.stringify(parsed, null, 2);
    setChip("metricsFileChip", `${file.name} · ${formatBytes(file.size)}`);
  }
  hideError();
}

function setupDropzone(zoneId, inputId, kind) {
  const zone = $(`#${zoneId}`);
  const input = $(`#${inputId}`);

  input.addEventListener("change", () => {
    const file = input.files[0];
    if (file) handleFile(file, kind);
    input.value = "";
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.add("is-dragging");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.remove("is-dragging");
    });
  });

  zone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (file) handleFile(file, kind);
  });
}

setupDropzone("codeDropzone", "codeFile", "code");
setupDropzone("metricsDropzone", "metricsFile", "metrics");

["dragover", "drop"].forEach((eventName) => {
  window.addEventListener(eventName, (event) => event.preventDefault());
});

function readCode() {
  const code = $("#codeInput").value;
  if (!code.trim()) {
    throw new Error("Paste code or upload a source file first.");
  }
  return code;
}

function readMetrics() {
  const raw = $("#metricsInput").value.trim();
  if (!raw) return null;

  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error("Metrics must be valid JSON.");
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Metrics must be a JSON object.");
  }
  return parsed;
}

function setLoading(loading) {
  const button = $("#analyzeBtn");
  button.disabled = loading;
  $("#spinner").hidden = !loading;
  $("#analyzeLabel").textContent = loading ? "Analyzing…" : "Analyze code";
}

function formatScore(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "—";
}

function formatPercent(value) {
  return Number.isFinite(value) ? `${Math.round(value * 100)}%` : "—";
}

function renderAgent(id, data, options) {
  const card = $(`#${id}Card`);
  const stateEl = $(`#${id}State`);
  const scoreEl = $(`#${id}Score`);
  const thresholdEl = $(`#${id}Threshold`);
  const barEl = $(`#${id}Bar`);
  const noteEl = $(`#${id}Note`);

  const ok = data && data.status === "ok";
  const score = ok ? Number(data[options.scoreKey]) : NaN;
  const threshold = ok ? Number(data.threshold) : NaN;
  const valid = Number.isFinite(score);
  const flagged = ok && Boolean(data[options.flagKey]);

  card.dataset.state = !ok || !valid ? "error" : flagged ? "bad" : "ok";
  stateEl.textContent = !ok || !valid
    ? "Error"
    : options.labels[flagged ? 1 : 0];
  scoreEl.textContent = ok && valid ? formatScore(score) : "—";
  thresholdEl.textContent =
    ok && Number.isFinite(threshold) ? `threshold ${formatScore(threshold)}` : "";
  barEl.style.width =
    ok && valid ? `${Math.min(100, Math.max(2, score * 100))}%` : "0%";

  const note = ok ? "" : (data && data.reason) || "No result returned.";
  noteEl.textContent = note;
  noteEl.hidden = !note;
}

function render(data) {
  const agent1 = data.agent1 || {};
  const agent2 = data.agent2 || {};
  const agent3 = data.agent3 || {};
  const verdict = ["PASS", "REVIEW", "FAIL"].includes(agent3.verdict)
    ? agent3.verdict
    : "REVIEW";

  $("#emptyState").hidden = true;
  const results = $("#results");
  results.hidden = false;
  results.classList.remove("is-visible");
  void results.offsetWidth;
  results.classList.add("is-visible");

  const card = $("#verdictCard");
  card.className = `verdict ${verdict.toLowerCase()}`;
  $("#verdictBadge").textContent = verdict;

  const titles = {
    PASS: "No vulnerabilities found",
    REVIEW: "Needs human review",
    FAIL: "Vulnerabilities detected",
  };
  const fallbackReasons = {
    PASS: "Both agents agree the code looks safe.",
    REVIEW: "The agents disagree, or an input was missing.",
    FAIL: "Both agents flagged this code as risky.",
  };
  $("#verdictTitle").textContent = titles[verdict];
  $("#verdictReason").textContent =
    agent3.reason || fallbackReasons[verdict];
  $("#confidenceValue").textContent = formatPercent(agent3.confidence);

  $("#resultSub").textContent = agent3.status === "error"
    ? "Fail-safe verdict"
    : "All three agents completed";

  const elapsed = Number(data.elapsed_ms);
  $("#elapsed").hidden = false;
  $("#elapsed").textContent =
    Number.isFinite(elapsed) ? `${elapsed.toLocaleString()} ms` : "—";

  renderAgent("agent1", agent1, {
    scoreKey: "struct_risk_score",
    flagKey: "high_risk",
    labels: ["Low risk", "High risk"],
  });
  renderAgent("agent2", agent2, {
    scoreKey: "transformer_risk_score",
    flagKey: "vulnerable",
    labels: ["Clean", "Vulnerable"],
  });

  $("#rawJson").textContent = JSON.stringify(data, null, 2);
}

async function analyze() {
  hideError();

  let code;
  let metrics;
  try {
    code = readCode();
    metrics = readMetrics();
  } catch (error) {
    showError(error.message);
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code, metrics }),
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || `Request failed (${response.status}).`);
    }
    render(data);
  } catch (error) {
    showError(error.message || "Analysis failed.");
  } finally {
    setLoading(false);
  }
}

function selectMode(attr, mode) {
  const tab = document.querySelector(`[${attr}="${mode}"]`);
  if (tab) tab.click();
}

function loadExample(example) {
  selectMode("data-code-mode", "paste");
  selectMode("data-metrics-mode", "paste");
  $("#codeInput").value = example.code;
  $("#metricsInput").value = JSON.stringify(example.metrics, null, 2);
  $("#metricsDetails").open = true;
  $("#codeFileChip").hidden = true;
  $("#metricsFileChip").hidden = true;
  updateCount();
  hideError();
}

$("#analyzeBtn").addEventListener("click", analyze);

$("#codeInput").addEventListener("input", updateCount);
$("#codeInput").addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    analyze();
  }
});

$("#clearCode").addEventListener("click", () => {
  $("#codeInput").value = "";
  $("#codeFileChip").hidden = true;
  updateCount();
  hideError();
});

$("#loadFailExample").addEventListener("click", () => loadExample(EXAMPLES.fail));
$("#loadPassExample").addEventListener("click", () => loadExample(EXAMPLES.pass));

updateCount();
