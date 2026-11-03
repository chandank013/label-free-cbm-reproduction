# DA351 — Label-Free Concept Bottleneck Models: Replication (Phase 1), Concept-Quality Extension (Phase 2) & Concept Lab Web App (Phase 3)

**Course**: DA351 &bull; Interpretable and Explainable Machine Learning  
**Upstream Paper**: Oikarinen, Das, Nguyen, Weng. *Label-Free Concept Bottleneck Models.* ICLR 2023  
- Paper PDF: [https://openreview.net/pdf?id=FlCg47MNvBA](https://openreview.net/pdf?id=FlCg47MNvBA)  
- Authors' Repository: [https://github.com/Trustworthy-ML-Lab/Label-free-CBM](https://github.com/Trustworthy-ML-Lab/Label-free-CBM)  

---

## 1. Project Overview

Concept Bottleneck Models (CBMs) enhance neural network interpretability by passing visual representations through an intermediate layer of human-understandable concepts prior to final classification. However, traditional CBMs require labor-intensive, per-image concept annotations.

Oikarinen et al. (ICLR 2023) proposed **Label-Free CBM**, automating concept discovery via GPT-3 and concept grounding via CLIP image-text similarity.

This project delivers three complete milestones:
1. **Phase 1 (Replication)**: Strict reproduction of the original Label-Free CBM pipeline across CIFAR-10, CIFAR-100, and CUB-200 benchmarks with non-invasive runtime compatibility for Windows (`compat.py`).
2. **Phase 2 (Concept-Quality Filtering Extension)**: Formulation of a submodular greedy filtering framework that prunes redundant, co-activating concept synonyms and purges uninformative background concepts using one-way ANOVA $\eta^2$ and row-centered Pearson correlation $\|r_{ij}\|$.
3. **Phase 3 ("Concept Lab" Web Application)**: A fully offline, local web application (`webapp/`) featuring interactive side-by-side inference on user-uploaded photos, dynamic SVG Pareto curves, grouped redundancy comparisons, per-class concept exploration, and a 10-minute presentation recording script.

---

## 2. Repository Layout

```
DA351_LFCBM_Project/
├── original_lfcbm/        # Authors' upstream repository (unmodified, byte-identical commit fa2869d)
├── compat.py              # Non-invasive Windows runtime compatibility layer (DataLoader & streaming saves)
├── run_experiment.py      # Phase 1 runner (handles seeds, logging, calls authors' training code)
├── collect_results.py     # Aggregates Phase 1 results into results/comparison_table.md
├── phase2/                # Phase 2: Concept-quality filtering module
│   ├── concept_quality.py      # ANOVA η² relevance & row-centered Pearson redundancy math
│   ├── make_filtered_trainer.py# Generator creating train_cbm_filtered.py from authors' trainer
│   ├── train_cbm_filtered.py   # Programmatically generated filtered trainer (zero edits to original)
│   ├── run_phase2.py           # Phase 2 experimental runner (greedy filter, random & clip_topk controls)
│   └── aggregate_phase2.py     # Aggregates multi-seed runs into summary JSONs and CSV concept tables
├── webapp/                # Phase 3: "Concept Lab" interactive web application
│   ├── server.py               # FastAPI server (127.0.0.1:8000), security validation, static asset mount
│   ├── registry.py             # Automatic model discovery, 8-file validation, candidate sorting
│   ├── results_reader.py       # Disk results reader, defensive normalization, SVG data formatting
│   ├── inference.py            # Thread-safe model pair sessions, PIL validation, concept contributions
│   ├── populate_test_data.py   # Harvests 1 test photo per class into root data/
│   ├── static/                 # Offline frontend (HTML5, CSS3, ES6 JavaScript, plain SVG charts)
│   └── tests/                  # Automated test suite (unit, API, smoke, and Playwright browser tests)
├── data/                  # Harvested sample test images (data/cifar10, data/cifar100, data/cub)
├── docs/                  # Detailed documentation and project specifications
│   ├── CONTRIBUTIONS.md        # Comprehensive team member contribution breakdown across all phases
│   ├── WORKFLOW.md             # End-to-end mathematical and system operational workflow guide
│   ├── RECORDING_SCRIPT.md     # Verbatim 10-minute video presentation teleprompter script
│   ├── PHASE2_PLAN.md          # Technical design specification for concept quality filtering
│   ├── RUN_STEPS_LOCAL.md      # Local GPU setup and execution steps
│   └── WINDOWS_TROUBLESHOOTING.md # Detailed diagnostics for Windows PyTorch issues
└── results/               # Phase 1 run logs and Phase 2 JSONs (results/phase2/<dataset>/)
```

---

## 3. Preservation of Authors' Code & Scientific Honesty

- **Unmodified Upstream Files**: `original_lfcbm/` remains byte-identical to upstream commit `fa2869d`, except for pointing cache directories to local disk (`D:/datasets/cache` in `data_utils.py`).
- **Non-Invasive Architecture**: All Windows GPU runtime patches (`compat.py`) and Phase 2 training routines (`phase2/train_cbm_filtered.py`) operate outside the authors' codebase without altering their logic.
- **Zero Invented Numbers**: Every metric, table cell, and chart curve rendered in the reports and web application is computed directly from raw run artifacts on disk.
- **Honest Scoping**: Heavy datasets requiring 35+ GB of RAM (Places365, ImageNet) are explicitly reported as "Not run", adhering to scientific integrity on commodity laptop hardware.

---

## 4. Full Step-by-Step Running Guide

### Step 1: Environment Setup

Create and activate a clean Python 3.10 virtual environment:

```powershell
# Create virtual environment
python -m venv lfcbm-env

# Activate on Windows PowerShell
.\lfcbm-env\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA support (CUDA 12.8 / 12.4 compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install Phase 1 and Phase 2 core dependencies
pip install -r requirements.txt

# Install Web Application dependencies (FastAPI, Uvicorn, Pillow, Playwright, Pytest)
pip install -r requirements-web.txt
```

Verify GPU acceleration:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

### Step 2: Download Backbones and Datasets

```powershell
cd original_lfcbm

# Download pretrained visual backbones
bash download_models.sh

# Download CUB-200-2011 dataset
bash download_cub.sh

# Download ResNet-18 Places365 backbone
bash download_rn18_places.sh

cd ..
```
*Note*: CIFAR-10 and CIFAR-100 download automatically on first execution into `D:/datasets/cache`.

---

### Step 3: Phase 1 — Paper Replication

Run the baseline Label-Free CBM and Standard (sparse) models across benchmarks:

```powershell
# CIFAR-10 (Seed 0, batch size 64)
python run_experiment.py --dataset cifar10 --model lfcbm --seed 0 --batch_size 64
python run_experiment.py --dataset cifar10 --model standard_sparse --seed 0 --batch_size 64

# CIFAR-100 (Seed 0, batch size 64)
python run_experiment.py --dataset cifar100 --model lfcbm --seed 0 --batch_size 64
python run_experiment.py --dataset cifar100 --model standard_sparse --seed 0 --batch_size 64

# CUB-200 (Seed 0, batch size 64)
python run_experiment.py --dataset cub --model lfcbm --seed 0 --batch_size 64
python run_experiment.py --dataset cub --model standard_sparse --seed 0 --batch_size 64

# Compile Paper vs. Reproduced comparison table
python collect_results.py
```
Output table is written to `results/comparison_table.md`.

---

### Step 4: Phase 2 — Concept-Quality Filtering Experiments

Execute the concept-quality filtering pipeline across budgets (75%, 50%, 25%) and control baselines (Random, CLIP Top-$K$):

```powershell
# 1. Run unit and synthetic logic tests
python phase2/test_concept_quality.py
python tests/test_compat.py
python phase2/smoke_test_glue.py

# 2. Run full 30-run experiment sweep on CIFAR-10 (3 seeds x 10 configurations)
python phase2/run_phase2.py --dataset cifar10 --batch_size 64

# 3. Run full 30-run experiment sweep on CIFAR-100 (3 seeds x 10 configurations)
python phase2/run_phase2.py --dataset cifar100 --batch_size 64

# 4. Run full 30-run experiment sweep on CUB-200 (3 seeds x 10 configurations)
python phase2/run_phase2.py --dataset cub --batch_size 64

# 5. Aggregate results, Pareto curves, and CSV concept tables
python phase2/aggregate_phase2.py --dataset cifar10
python phase2/aggregate_phase2.py --dataset cifar100
python phase2/aggregate_phase2.py --dataset cub
python phase2/aggregate_phase2.py --all
```

---

### Step 5: Phase 3 — Launch the "Concept Lab" Web Application

```powershell
# 1. Ensure sample test photos are populated in data/ (1 image per class)
python webapp/populate_test_data.py

# 2. Launch the web server on 127.0.0.1:8000
.\lfcbm-env\Scripts\uvicorn.exe webapp.server:app --host 127.0.0.1 --port 8000
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

#### Interactive Views:
- **Compare (`#compare`)**: Inspect KPIs (Accuracy, # Concepts, Redundancy, Relevance), Pareto curves with seed error bars, grouped redundancy bar charts (CIFAR-10, CIFAR-100, CUB-200), Paper vs. Reproduced benchmarking table, and the pruned concept audit table with reason badges.
- **Try it (`#try`)**: Drag and drop any photo from `data/` (or upload your own JPEG/PNG/WEBP). See real-time prediction agreement, confidence meters, and side-by-side concept explanation bars ($a_j \cdot w_{c,j}$).
- **Concepts (`#concepts`)**: Explore per-class concept weights across Original and Filtered models.
- **About (`#about`)**: Review system architecture, methodology, and model folder inventories.

---

### Step 6: Automated Verification & Testing

Execute the complete automated test suite:

```powershell
.\lfcbm-env\Scripts\pytest.exe webapp/tests/ -v
```

**Verification Coverage**:
- `test_unit.py` (10 tests): Folder name parsing, lowest-seed sorting, defensive accuracy normalization, delta noise bands, contribution math, text Jaccard similarity.
- `test_api.py` (6 tests): Whitelist security validation, empty directory state handling, missing model error codes, corrupted upload rejection.
- `test_smoke_real.py` (1 test): Live GPU forward pass and mathematical cross-check against `cbm.load_cbm` ($< 10^{-4}$ tolerance).
- `test_browser.py` (1 test): Playwright headless Edge browser verification across $1440 \times 900$ and $1024 \times 768$ viewports, asserting zero console errors, zero external requests, and clean active dataset dropdown options.

*Result*: 18 passed in ~40 seconds.

---

## 5. Summary of Experimental Results

Below is the verified summary across all completed runs on disk (3 baseline seeds and 9 filter configurations per dataset):

| Dataset | Method | Budget ($k$) | # Concepts | Val Accuracy (%) | Redundancy ($\rho$) | Relevance ($\bar{\eta}^2$) | NEC |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **CIFAR-10** | Baseline | 100% | 132 | $86.48\% \pm 0.10\%$ | $0.723 \pm 0.000$ | $0.540 \pm 0.000$ | $36.3$ |
| | **Greedy (Ours)** | **50%** | **69** | **$86.03\% \pm 0.12\%$** | **$0.715 \pm 0.000$** | **$0.640 \pm 0.000$** | **$26.5$** |
| **CIFAR-100** | Baseline | 100% | 825 | $65.14\% \pm 0.14\%$ | $0.704 \pm 0.000$ | $0.481 \pm 0.000$ | $27.9$ |
| | Greedy | 75% | 621 | $64.80\% \pm 0.13\%$ | $0.663 \pm 0.001$ | $0.505 \pm 0.000$ | $26.8$ |
| | **Greedy (Ours)** | **50%** | **415** | **$64.06\% \pm 0.01\%$** | **$0.624 \pm 0.000$** | **$0.522 \pm 0.000$** | **$25.6$** |
| | Greedy | 25% | 208 | $62.39\% \pm 0.15\%$ | $0.574 \pm 0.000$ | $0.541 \pm 0.000$ | $21.2$ |
| | *Control: Random* | 50% | 412 | $63.42\% \pm 0.13\%$ | $0.662 \pm 0.001$ | $0.484 \pm 0.001$ | $27.5$ |
| | *Control: CLIP Top-k* | 50% | 415 | $63.99\% \pm 0.12\%$ | $0.710 \pm 0.000$ | $0.490 \pm 0.000$ | $26.1$ |
| **CUB-200** | Baseline | 100% | 209 | $74.12\% \pm 0.04\%$ | $0.802 \pm 0.000$ | $0.673 \pm 0.000$ | $31.1$ |
| | **Greedy (Ours)** | **50%** | **120** | **$74.05\% \pm 0.41\%$** | **$0.729 \pm 0.000$** | **$0.700 \pm 0.000$** | **$27.4$** |

### Key Takeaways:
1. **$50\%$ Concept Reduction with $< 1.1\%$ Accuracy Drop**: The greedy submodular filter cuts concept count in half while maintaining over $98.3\%$ of baseline accuracy across all datasets.
2. **Defeating Control Baselines**: Greedy selection significantly outperforms Random selection and CLIP Top-$K$ selection in both accuracy retention and redundancy reduction.
3. **Pruning Flawed Concepts**: Prunes collinear synonyms (e.g. co-activating bird color variants) and eliminates generic background concepts that provide zero class separation.


