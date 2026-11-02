/**
 * Concept Lab Frontend Application
 */

const AppState = {
  activeTab: "compare",
  selectedDataset: "cub",
  selectedBudget: 50,
  datasets: [],
  classesCache: {},
  currentImageFile: null,
};

// Initialize application on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  initRouter();
  initEventListeners();
  loadInitialData();
});

/* ==========================================================================
   Router
   ========================================================================== */

function initRouter() {
  function handleHash() {
    const rawHash = window.location.hash.replace("#", "").trim();
    const validTabs = ["compare", "try", "concepts", "about"];
    const targetTab = validTabs.includes(rawHash) ? rawHash : "compare";
    switchTab(targetTab);
  }

  window.addEventListener("hashchange", handleHash);
  handleHash();
}

function switchTab(tabName) {
  AppState.activeTab = tabName;

  // Update nav buttons
  document.querySelectorAll(".nav-tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.id === `tab-nav-${tabName}`);
  });

  // Update views
  document.querySelectorAll(".view-section").forEach((sec) => {
    sec.classList.toggle("active", sec.id === `view-${tabName}`);
  });

  // Tab-specific refreshes
  if (tabName === "compare") {
    loadCompareData();
  } else if (tabName === "try") {
    syncTryControls();
  } else if (tabName === "concepts") {
    loadConceptsView();
  } else if (tabName === "about") {
    loadAboutData();
  }
}

/* ==========================================================================
   Initial Data & Event Handlers
   ========================================================================== */

async function loadInitialData() {
  try {
    const res = await fetch("/api/datasets");
    if (!res.ok) throw new Error("Failed to load dataset registry");
    const data = await res.json();
    AppState.datasets = data.datasets || [];

    // Prefer CUB if it has a pair, else first with pair
    const cub = AppState.datasets.find((d) => d.id === "cub" && d.has_pair);
    if (cub) {
      AppState.selectedDataset = "cub";
    } else {
      const firstValid = AppState.datasets.find((d) => d.has_pair);
      if (firstValid) AppState.selectedDataset = firstValid.id;
    }

    populateDatasetDropdowns();
    loadCompareData();
  } catch (err) {
    console.error("Initialization error:", err);
  }
}

function initEventListeners() {
  // Compare Dataset Select
  const compareDsSelect = document.getElementById("compare-dataset-select");
  if (compareDsSelect) {
    compareDsSelect.addEventListener("change", (e) => {
      AppState.selectedDataset = e.target.value;
      syncDatasetSelects();
      loadCompareData();
    });
  }

  // Compare Budget Buttons
  const compareBudgetGroup = document.getElementById("compare-budget-controls");
  if (compareBudgetGroup) {
    compareBudgetGroup.addEventListener("click", (e) => {
      const btn = e.target.closest("button");
      if (!btn || btn.disabled) return;
      const b = parseInt(btn.dataset.budget, 10);
      AppState.selectedBudget = b;
      updateBudgetButtons();
      loadCompareData();
    });
  }

  // Try-it Dataset Select
  const tryDsSelect = document.getElementById("try-dataset-select");
  if (tryDsSelect) {
    tryDsSelect.addEventListener("change", (e) => {
      AppState.selectedDataset = e.target.value;
      syncDatasetSelects();
      syncTryControls();
      if (AppState.currentImageFile) {
        runPrediction(AppState.currentImageFile);
      }
    });
  }

  // Try-it Budget Buttons
  const tryBudgetGroup = document.getElementById("try-budget-controls");
  if (tryBudgetGroup) {
    tryBudgetGroup.addEventListener("click", (e) => {
      const btn = e.target.closest("button");
      if (!btn || btn.disabled) return;
      const b = parseInt(btn.dataset.budget, 10);
      AppState.selectedBudget = b;
      updateBudgetButtons();
      if (AppState.currentImageFile) {
        runPrediction(AppState.currentImageFile);
      }
    });
  }

  // Concepts Controls
  const concDsSelect = document.getElementById("concepts-dataset-select");
  if (concDsSelect) {
    concDsSelect.addEventListener("change", async (e) => {
      AppState.selectedDataset = e.target.value;
      syncDatasetSelects();
      await populateClassDropdown();
      loadConceptsView();
    });
  }

  const concClsSelect = document.getElementById("concepts-class-select");
  if (concClsSelect) {
    concClsSelect.addEventListener("change", () => {
      loadConceptsView();
    });
  }

  const concBudgetGroup = document.getElementById("concepts-budget-controls");
  if (concBudgetGroup) {
    concBudgetGroup.addEventListener("click", (e) => {
      const btn = e.target.closest("button");
      if (!btn || btn.disabled) return;
      AppState.selectedBudget = parseInt(btn.dataset.budget, 10);
      updateBudgetButtons();
      loadConceptsView();
    });
  }

  // File Upload & Drag-and-drop
  const dropZone = document.getElementById("image-drop-zone");
  const fileInput = document.getElementById("file-upload-input");

  if (dropZone && fileInput) {
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleImageFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleImageFile(e.target.files[0]);
      }
    });
  }

  // Toggle class list disclosure
  const btnToggleClasses = document.getElementById("btn-toggle-classes");
  const classesListBox = document.getElementById("classes-list-box");
  if (btnToggleClasses && classesListBox) {
    btnToggleClasses.addEventListener("click", async () => {
      if (classesListBox.style.display === "none") {
        classesListBox.style.display = "block";
        btnToggleClasses.textContent = "Hide supported classes";
        if (!classesListBox.hasChildNodes() || classesListBox.dataset.ds !== AppState.selectedDataset) {
          const classes = await fetchClasses(AppState.selectedDataset);
          classesListBox.dataset.ds = AppState.selectedDataset;
          classesListBox.innerHTML = classes.map((c, i) => `<div><strong>${i + 1}.</strong> ${c}</div>`).join("");
        }
      } else {
        classesListBox.style.display = "none";
        btnToggleClasses.textContent = "View supported classes";
      }
    });
  }
}

function populateDatasetDropdowns() {
  const selects = [
    document.getElementById("compare-dataset-select"),
    document.getElementById("try-dataset-select"),
    document.getElementById("concepts-dataset-select"),
  ];

  selects.forEach((sel) => {
    if (!sel) return;
    sel.innerHTML = AppState.datasets
      .map((d) => `<option value="${d.id}" ${d.id === AppState.selectedDataset ? "selected" : ""}>${d.name}${!d.has_pair ? " (Not run)" : ""}</option>`)
      .join("");
  });

  updateBudgetButtons();
}

function syncDatasetSelects() {
  ["compare-dataset-select", "try-dataset-select", "concepts-dataset-select"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.value = AppState.selectedDataset;
  });
  updateBudgetButtons();
}

function updateBudgetButtons() {
  const dsInfo = AppState.datasets.find((d) => d.id === AppState.selectedDataset);
  const availableBudgets = dsInfo ? dsInfo.available_budgets : [75, 50, 25];

  ["compare-budget-controls", "try-budget-controls", "concepts-budget-controls"].forEach((groupId) => {
    const group = document.getElementById(groupId);
    if (!group) return;

    group.querySelectorAll("button").forEach((btn) => {
      const b = parseInt(btn.dataset.budget, 10);
      const isAvailable = availableBudgets.includes(b);
      btn.disabled = !isAvailable;
      btn.classList.toggle("active", b === AppState.selectedBudget && isAvailable);
    });
  });
}

/* ==========================================================================
   Compare Page Logic
   ========================================================================== */

async function loadCompareData() {
  try {
    const url = `/api/compare?dataset=${AppState.selectedDataset}&budget=${AppState.selectedBudget}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error("Failed to load compare data");
    const data = await res.json();

    renderKPIs(data.kpis);
    Charts.renderAccuracyCurve("chart-accuracy-curve", data.curves);
    Charts.renderRedundancyBars("chart-redundancy-bars", data.redundancy_comparison);
    renderPaperVsReproTable(data.paper_vs_reproduced);
    renderRemovedConceptsTable(data.removed_concepts);
  } catch (err) {
    console.error("Error loading compare data:", err);
  }
}

function renderKPIs(kpis) {
  if (!kpis) return;

  function setCard(prefix, kpi, formatFn, isArrowUp) {
    const origEl = document.getElementById(`kpi-${prefix}-orig`);
    const filtEl = document.getElementById(`kpi-${prefix}-filt`);
    const badgeEl = document.getElementById(`kpi-${prefix}-badge`);

    if (!origEl || !filtEl || !badgeEl) return;

    if (!kpi || kpi.status === "not_run" || kpi.original === null) {
      origEl.textContent = "—";
      filtEl.textContent = "—";
      badgeEl.className = "kpi-badge grey";
      badgeEl.textContent = "Not run";
      return;
    }

    origEl.textContent = formatFn(kpi.original);
    filtEl.textContent = formatFn(kpi.filtered);

    badgeEl.className = `kpi-badge ${kpi.badge_color}`;
    const arrow = kpi.delta > 0 ? "▲ " : kpi.delta < 0 ? "▼ " : "";
    badgeEl.textContent = `${arrow}${kpi.badge_text}`;
    if (kpi.seed_note) {
      badgeEl.title = `${kpi.seed_note} (Std: Orig ±${(kpi.orig_std || 0).toFixed(3)}, Filt ±${(kpi.filt_std || 0).toFixed(3)})`;
    }
  }

  // Accuracy (e.g. 0.872)
  setCard("acc", kpis.accuracy, (v) => v.toFixed(3), true);
  // Number of Concepts (e.g. 20)
  setCard("con", kpis.n_concepts, (v) => Math.round(v).toString(), false);
  // Redundancy (e.g. 0.31)
  setCard("red", kpis.redundancy, (v) => v.toFixed(2), false);
  // Relevance (e.g. 0.68)
  setCard("rel", kpis.relevance, (v) => v.toFixed(2), true);
}

function renderPaperVsReproTable(rows) {
  const tbody = document.querySelector("#table-paper-vs-repro tbody");
  if (!tbody || !rows) return;

  tbody.innerHTML = rows
    .map((r) => {
      const isSel = r.is_selected;
      const paperStr = r.paper !== null ? r.paper.toFixed(3) : "—";
      const reproStr = r.reproduced !== null ? r.reproduced.toFixed(3) : `<span class="not-run-text">Not run</span>`;

      let diffStr = "—";
      if (r.diff !== null) {
        const sign = r.diff >= 0 ? "+" : "";
        diffStr = `${sign}${r.diff.toFixed(3)} (${sign}${r.diff_pct.toFixed(2)}%)`;
      }

      return `
        <tr class="${isSel ? "selected-row" : ""}">
          <td><strong>${r.display_name}</strong></td>
          <td>${paperStr}</td>
          <td>${reproStr}</td>
          <td>${diffStr}</td>
        </tr>
      `;
    })
    .join("");
}

function renderRemovedConceptsTable(concepts) {
  const tbody = document.querySelector("#table-removed-concepts tbody");
  if (!tbody) return;

  if (!concepts || concepts.length === 0) {
    tbody.innerHTML = `<tr><td colspan="4" class="not-run-text" style="text-align:center; padding:1.5rem;">No removed concepts for this budget or dataset not run.</td></tr>`;
    return;
  }

  tbody.innerHTML = concepts
    .map((item) => {
      const chipClass = item.reason_type === "duplicate" ? "amber" : "purple";
      return `
        <tr>
          <td style="color:var(--text-muted); font-size:0.75rem;">${item.index}</td>
          <td><strong>${item.concept}</strong></td>
          <td><span class="reason-chip ${chipClass}">${item.reason_chip}</span></td>
          <td style="color:var(--text-secondary);">${item.example}</td>
        </tr>
      `;
    })
    .join("");
}

/* ==========================================================================
   Try-it Page Logic
   ========================================================================== */

function syncTryControls() {
  syncDatasetSelects();
  updateBudgetButtons();

  const dsInfo = AppState.datasets.find((d) => d.id === AppState.selectedDataset);
  const dropZone = document.getElementById("image-drop-zone");
  if (dropZone) {
    if (!dsInfo || !dsInfo.has_pair) {
      dropZone.style.opacity = "0.5";
      dropZone.style.pointerEvents = "none";
    } else {
      dropZone.style.opacity = "1";
      dropZone.style.pointerEvents = "auto";
    }
  }
}

function handleImageFile(file) {
  if (!file) return;
  if (!file.type.match(/image\/(jpeg|png|webp)/) && !file.name.match(/\.(jpg|jpeg|png|webp)$/i)) {
    alert("Please upload a valid JPEG, PNG, or WEBP image.");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    alert("Image file size exceeds the 10 MB limit.");
    return;
  }

  AppState.currentImageFile = file;

  // Show thumbnail
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("drop-zone-default").style.display = "none";
    const previewContainer = document.getElementById("drop-zone-preview");
    previewContainer.style.display = "flex";
    document.getElementById("preview-img").src = e.target.result;
    document.getElementById("preview-filename").textContent = file.name;
  };
  reader.readAsDataURL(file);

  runPrediction(file);
}

async function runPrediction(file) {
  const agreementContainer = document.getElementById("agreement-container");
  const agreementPill = document.getElementById("agreement-pill");
  const agreementText = document.getElementById("agreement-text");

  // Show loading placeholders
  setPredictionLoadingState();

  try {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("dataset", AppState.selectedDataset);
    formData.append("budget", AppState.selectedBudget.toString());

    const res = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Prediction error (${res.status})`);
    }

    const data = await res.json();

    // Agreement Pill
    agreementContainer.style.display = "flex";
    if (data.models_agree) {
      agreementPill.className = "agreement-pill green";
      agreementText.textContent = "Both models agree";
    } else {
      agreementPill.className = "agreement-pill amber";
      agreementText.textContent = "Models disagree";
    }

    // Render original and filtered cards
    renderModelCard("orig", data.original);
    renderModelCard("filt", data.filtered);
  } catch (err) {
    console.error("Prediction failed:", err);
    alert(`Prediction failed: ${err.message}`);
    clearPredictionDisplay();
  }
}

function setPredictionLoadingState() {
  ["orig", "filt"].forEach((prefix) => {
    document.getElementById(`${prefix}-pred-class`).textContent = "Predicting...";
    document.getElementById(`${prefix}-conf-val`).textContent = "—";
    document.getElementById(`${prefix}-conf-bar`).style.width = "0%";
    document.getElementById(`${prefix}-concepts-list`).innerHTML = `
      <div style="text-align:center; padding:1.5rem; color:var(--text-muted);">
        <div class="spinner" style="border-top-color:var(--teal-600); margin-bottom:0.5rem;"></div>
        <div>Running inference & extracting concepts...</div>
      </div>
    `;
  });
}

function clearPredictionDisplay() {
  document.getElementById("agreement-container").style.display = "none";
  ["orig", "filt"].forEach((prefix) => {
    document.getElementById(`${prefix}-pred-class`).textContent = "—";
    document.getElementById(`${prefix}-conf-val`).textContent = "—";
    document.getElementById(`${prefix}-conf-bar`).style.width = "0%";
    document.getElementById(`${prefix}-concepts-list`).innerHTML = `
      <div style="color:var(--text-muted); font-size:0.85rem; padding:1rem 0; text-align:center;">
        Upload an image to see concept explanations.
      </div>
    `;
  });
}

function renderModelCard(prefix, pred) {
  if (!pred) return;

  const isOrig = prefix === "orig";
  const barColorClass = isOrig ? "purple" : "teal";

  // Predicted Class
  const classEl = document.getElementById(`${prefix}-pred-class`);
  classEl.textContent = pred.predicted_class;

  // Confidence
  const confEl = document.getElementById(`${prefix}-conf-val`);
  const confBar = document.getElementById(`${prefix}-conf-bar`);
  confEl.textContent = pred.confidence.toFixed(3);
  confBar.style.width = `${Math.min(100, Math.round(pred.confidence * 100))}%`;

  if (pred.is_low_confidence) {
    confEl.title = "Warning: confidence < 0.50 (low confidence)";
  }

  // Top Concepts
  const listEl = document.getElementById(`${prefix}-concepts-list`);
  if (!pred.top_concepts || pred.top_concepts.length === 0) {
    listEl.innerHTML = `<div style="color:var(--text-muted); text-align:center; padding:1rem;">No positive concept contributors found.</div>`;
  } else {
    listEl.innerHTML = pred.top_concepts
      .map((c) => {
        const dupBadge = c.is_duplicate
          ? `<span class="similar-wording-tag" title="Similar wording (word overlap ≥ 0.5 with '${c.duplicate_of}'). Measured redundancy is on the Compare page.">duplicate</span>`
          : "";

        return `
          <div class="concept-item-row">
            <div class="concept-name-wrap" title="${c.concept}">
              <span class="concept-name">${c.concept}</span>
              ${dupBadge}
            </div>
            <div class="concept-bar-track">
              <div class="concept-bar-fill ${barColorClass}" style="width: ${c.bar_pct.toFixed(1)}%;"></div>
            </div>
            <div class="concept-raw-val">${c.contribution.toFixed(2)}</div>
          </div>
        `;
      })
      .join("");
  }

  // Stat Tiles
  document.getElementById(`${prefix}-stat-concepts`).textContent = pred.stats.n_concepts ?? "—";
  document.getElementById(`${prefix}-stat-nec`).textContent = pred.stats.concepts_per_class !== null ? pred.stats.concepts_per_class.toFixed(1) : "—";
  document.getElementById(`${prefix}-stat-redundancy`).textContent = pred.stats.redundancy !== null ? pred.stats.redundancy.toFixed(2) : "—";
}

/* ==========================================================================
   Concepts View Logic
   ========================================================================== */

async function fetchClasses(dataset) {
  if (AppState.classesCache[dataset]) {
    return AppState.classesCache[dataset];
  }
  try {
    const res = await fetch(`/api/classes?dataset=${dataset}`);
    if (!res.ok) return [];
    const data = await res.json();
    AppState.classesCache[dataset] = data.classes || [];
    return AppState.classesCache[dataset];
  } catch {
    return [];
  }
}

async function populateClassDropdown() {
  const clsSelect = document.getElementById("concepts-class-select");
  if (!clsSelect) return;
  const classes = await fetchClasses(AppState.selectedDataset);
  clsSelect.innerHTML = classes
    .map((c, idx) => `<option value="${idx}">${idx + 1}. ${c}</option>`)
    .join("");
}

async function loadConceptsView() {
  const clsSelect = document.getElementById("concepts-class-select");
  if (!clsSelect || clsSelect.children.length === 0) {
    await populateClassDropdown();
  }

  const classIdx = parseInt(clsSelect.value, 10) || 0;
  const colOrig = document.getElementById("concepts-col-orig");
  const colFilt = document.getElementById("concepts-col-filt");

  try {
    const url = `/api/class_concepts?dataset=${AppState.selectedDataset}&class_idx=${classIdx}&budget=${AppState.selectedBudget}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error("Failed to load class concepts");
    const data = await res.json();

    function renderConceptRows(items, colorClass) {
      if (!items || items.length === 0) {
        return `<div class="not-run-text" style="padding:1rem;">No concept weights found.</div>`;
      }
      const maxW = Math.max(...items.map((it) => Math.abs(it.weight)), 0.1);
      return items
        .map((it) => {
          const barW = (Math.abs(it.weight) / maxW) * 100;
          return `
            <div class="concept-weight-item">
              <div style="flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; padding-right:0.5rem;" title="${it.concept}">
                <strong>${it.concept}</strong>
              </div>
              <div style="width:100px; height:6px; background:#f1f5f9; border-radius:3px; overflow:hidden; margin:0 0.5rem;">
                <div class="concept-bar-fill ${colorClass}" style="width:${barW}%;"></div>
              </div>
              <div style="width:40px; text-align:right; font-weight:600; color:var(--text-secondary);">
                ${it.weight.toFixed(2)}
              </div>
            </div>
          `;
        })
        .join("");
    }

    colOrig.innerHTML = renderConceptRows(data.original_concepts, "purple");
    colFilt.innerHTML = renderConceptRows(data.filtered_concepts, "teal");
  } catch (err) {
    colOrig.innerHTML = `<div class="not-run-text">${err.message}</div>`;
    colFilt.innerHTML = `<div class="not-run-text">${err.message}</div>`;
  }
}

/* ==========================================================================
   About Page Logic
   ========================================================================== */

async function loadAboutData() {
  const tbody = document.querySelector("#table-about-models tbody");
  if (!tbody) return;

  try {
    const res = await fetch("/api/about_info");
    if (!res.ok) throw new Error("Failed to load about info");
    const data = await res.json();

    tbody.innerHTML = (data.models || [])
      .map((m) => `
        <tr>
          <td><strong>${m.dataset}</strong></td>
          <td><code>${m.folder_name}</code></td>
          <td>${m.method}</td>
          <td>${m.budget ? m.budget + "%" : "—"}</td>
          <td>${m.seed !== null ? m.seed : "—"}</td>
          <td><span class="header-badge grey-badge">${m.label}</span></td>
        </tr>
      `)
      .join("");
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="6" class="not-run-text">${err.message}</td></tr>`;
  }
}
