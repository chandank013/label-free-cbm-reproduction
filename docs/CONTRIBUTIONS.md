# Contribution Log — DA351 (Phase 1, Phase 2 & Web Application)

**Course**: DA351 &bull; Interpretable and Explainable Machine Learning  
**Project**: Replication and Concept-Quality Extension of Label-Free Concept Bottleneck Models (Oikarinen et al., ICLR 2023)  
**Team Size**: 5 members &bull; Equal distribution (20% each)  

---

## 1. Executive Contribution Summary

| # | Team Member | Primary Role | Phase 1: Replication | Phase 2: Concept-Quality Extension | Web App: Concept Lab | Contrib. |
|---|---|---|---|---|---|:---:|
| **1** | **Chandan Kumar** | Paper & Theoretical Lead | Paper analysis; formalization of 2-stage CBM pipeline; Phase 0 proposal; baseline comparison scripts | Theoretical justification of concept quality; formulation of ANOVA $\eta^2$ class-discriminability metric; literature ties (LaBo, mRMR, VLG-CBM) | Documentation lead; About page scientific disclosures; video presentation teleprompter script (`docs/RECORDING_SCRIPT.md`) | **20%** |
| **2** | **Aayush Jha** | Core Experiments Lead (CIFAR) | Environment setup; CIFAR-10 & CIFAR-100 baseline runs (seeds 0, 1, 2); validation logging via `run_experiment.py` | Executed multi-seed CIFAR-10 and full 30 CIFAR-100 runs (baseline, greedy $k=75,50,25$, random, clip_topk); verification of metric JSONs | Test image harvesting for CIFAR-10 (10 images) and CIFAR-100 (100 images) in `data/`; validation of prediction APIs | **20%** |
| **3** | **Hari Krishna Sharma** | System Architecture & CUB Lead | CUB-200-2011 download & extraction; fixed CUB train split issue; qualitative concept checks via `evaluate_cbm.ipynb` | Executed CUB-200 Phase 2 experiments (baseline, greedy, random, clip_topk over 3 seeds); analyzed concept semantic collapse in fine-grained birds | Designed and built full-stack web app (`webapp/`): FastAPI backend, thread-safe session manager (`inference.py`), plain SVG charts (`charts.js`), and Playwright browser tests | **20%** |
| **4** | **Prem Kishan** | Infrastructure & Platform Lead | Windows CUDA `pin_memory`/`num_workers` runtime monkey-patch (`compat.py`); memory-efficient activation saving; GPU memory profiling | Engineered row-centered Pearson correlation matrix computation; optimized submodular greedy selection loop to eliminate GPU OOM spikes | Web app server lifecycle management (`server.py`); Uvicorn daemon binding to `127.0.0.1:8000`; upload validation pipeline (EXIF, RGB conversion, PIL security) | **20%** |
| **5** | **Lavanya Dharmadhikari** | Evaluation & Reporting Lead | Official ImageNet extraction pipeline; Places365 integrity checker; `collect_results.py` comparison table; repo structure & git hygiene | Aggregation pipeline (`aggregate_phase2.py`); concept removal audit tables; redundant concept pair tagging ($r \ge 0.80$); NEC metric calculation | Frontend UI implementation (`index.html`, `app.css`, `app.js`) matching pixel mockups; Compare and Try-it tab layouts; dynamic filter chip badges | **20%** |

---

## 2. Detailed Task Breakdown by Phase

### Phase 1 — Paper Replication (ICLR 2023)
- **Paper & Architecture Understanding** (*Chandan*): Studied Oikarinen et al. (ICLR 2023), mapping out the three-stage process: offline GPT-3 concept generation, frozen visual backbone activation caching (CLIP RN50 / ResNet18), and sparse linear layer classification via GLM-SAGA.
- **Experimental Execution** (*Aayush, Hari*): Ran both Label-Free CBM and Standard (sparse) across CIFAR-10, CIFAR-100, and CUB-200. Verified accuracy against author-reported numbers.
- **Platform Engineering & Bug Fixes** (*Prem*): Diagnosed and resolved Windows-specific PyTorch failures without modifying upstream author files:
  - Resolved `DataLoader(pin_memory=True)` CUDA initialization failures by runtime monkey-patching in `compat.py`.
  - Fixed Windows paging file crashes during large activation saves by implementing memory-mapped streaming chunk saves.
  - Authored [`docs/WINDOWS_TROUBLESHOOTING.md`](file:///c:/XAI-Project/DA351_LFCBM_Project/docs/WINDOWS_TROUBLESHOOTING.md) and [`docs/LOCAL_GPU_SETUP.md`](file:///c:/XAI-Project/DA351_LFCBM_Project/docs/LOCAL_GPU_SETUP.md).
- **Data Infrastructure & Reporting** (*Lavanya*): Structured `original_lfcbm/data_utils.py` download paths, built ImageNet preparation scripts (`scripts/imagenet_official/`), and implemented `collect_results.py` to compile `results/comparison_table.md`.

---

### Phase 2 — Concept-Quality Filtering Extension
- **Research Gap Identification** (*Chandan, Hari*): Proved that the original authors' 3 filters (text length, CLIP score threshold, projection interpretability) failed to ensure class discriminability and allowed severe activation collinearity (e.g., redundant concepts like "a curved beak" vs "a hooked bill" co-activating on the same images).
- **Mathematical Formulation** (*Chandan, Prem*): Formulated concept quality through submodular optimization:
  $$\max_{S \subseteq V, |S| \le k} \sum_{j \in S} \eta_j^2 - \lambda \sum_{j \in S} \max_{m \in S, m \neq j} |r_{jm}|$$
  where $\eta_j^2$ is the one-way ANOVA effect size (class relevance) and $|r_{jm}|$ is the row-centered absolute Pearson correlation of activations on held-out validation data.
- **Non-Invasive Architecture** (*Prem, Hari*): Built `phase2/make_filtered_trainer.py` to programmatically inject concept quality filtering into `original_lfcbm/train_cbm.py` (generating `phase2/train_cbm_filtered.py`), ensuring 0 manual edits to author source files.
- **Full Empirical Campaign** (*Aayush, Hari, Lavanya*):
  - Executed all 30 planned runs on CIFAR-100 (3 baseline seeds, 9 greedy runs at 75%, 50%, 25% budgets, 9 random controls, and 9 CLIP Top-$K$ controls).
  - Executed full seed sweeps for CIFAR-10 and CUB-200.
  - Proved that greedy filtering reduces pairwise activation redundancy by up to $61.3\%$ (CUB) and $11.4\%$ (CIFAR-100) with negligible accuracy impact ($< 1.1\%$).
  - Showed that the CLIP Top-$K$ control actually *increases* activation redundancy, proving that high-scoring concepts cluster on duplicate features.
- **Reporting & Statistical Aggregation** (*Lavanya*): Built `phase2/aggregate_phase2.py` to compute seed-averaged metrics, standard deviations, and generate CSV concept tables mapping nearest kept neighbors and correlation scores.

---

### Phase 3 — "Concept Lab" Web Application
- **Frontend Architecture & Mockup Matching** (*Lavanya, Hari*):
  - Created offline, single-page application (`webapp/static/index.html`, `app.css`, `app.js`, `charts.js`).
  - Faithfully reproduced the two design mockups: dark navy header (`#0F1B3D`), teal accents, light blue-grey canvas (`#F3F6FB`), side-by-side comparison cards, and custom metric badges.
  - Implemented 4 full views: `#compare`, `#try`, `#concepts`, and `#about`.
  - Built zero-dependency plain SVG charting library for Pareto curves (with seed error bars) and grouped redundancy bar charts (active datasets: CIFAR-10, CIFAR-100, CUB-200).
- **Backend & Model Serving Engine** (*Hari, Prem*):
  - Built FastAPI server (`webapp/server.py`) binding exclusively to `127.0.0.1:8000` with strict whitelist parameter validation.
  - Built `webapp/inference.py`: Thread-safe, lazy-loading model pair session manager using official authors' loader `cbm.load_cbm` and visual backbone transforms (`clip_RN50`, `resnet18_cub`).
  - Implemented exact per-concept contribution decomposition ($a_j \cdot w_{c,j}$) for the top 6 contributing concepts and dynamic NEC computation.
  - Built text similarity duplicate detection via token Jaccard overlap ($\ge 0.50$).
  - Built secure upload validation: max 10MB, PIL image verification, EXIF rotation handling, RGB normalization, zero disk persistence.
- **Model Registry & Data Readers** (*Hari, Aayush*):
  - Built `webapp/registry.py`: Scans `original_lfcbm/saved_models/`, verifies all 8 required model files, stably sorts candidates (lowest seed, newest timestamp wins), and exposes active datasets.
  - Built `webapp/results_reader.py`: Reads `results/` and `results/phase2/` files directly from disk, defensively normalizes percentage vs fraction values, and generates audit chips for pruned concepts.
- **Test Image Data Harvesting & Quality Control** (*Aayush*):
  - Harvested test sample photos into root `data/` for every single class: 10 images for CIFAR-10, 100 images for CIFAR-100, and 200 images for CUB-200.
  - Verified inference accuracy across all test classes.
- **Verification & Testing Suite** (*Hari, Prem*):
  - 10 unit tests (`webapp/tests/test_unit.py`): folder parsing, sorting, normalization, noise bands, contribution math, Jaccard similarity.
  - 6 API tests (`webapp/tests/test_api.py`): whitelist rejections, empty folder handling, missing model rejections, upload error codes.
  - Real-model smoke test (`webapp/tests/test_smoke_real.py`): Live forward pass on GPU and cross-validation against raw `cbm.load_cbm` ($< 10^{-4}$ tolerance).
  - Headless browser verification (`webapp/tests/test_browser.py`): Playwright Edge testing across $1440 \times 900$ and $1024 \times 768$ viewports, asserting zero console errors, zero external requests, and clean active dataset dropdown options.
- **Course Presentation & Video Script** (*Chandan*):
  - Drafted comprehensive 10-minute teleprompter script ([`docs/RECORDING_SCRIPT.md`](file:///c:/XAI-Project/DA351_LFCBM_Project/docs/RECORDING_SCRIPT.md)) covering all 6 mandated sections with exact screen cues and time allocations.

---

## 3. Shared Governance & Team Agreements

1. **Strict Codebase Non-Invasion**: All members strictly adhered to preserving `original_lfcbm/` unaltered (with the sole exception of the dataset cache path edit). All fixes and extensions were introduced through `compat.py`, `phase2/`, and `webapp/`.
2. **Zero Scientific Invention**: Every number, table row, and chart point displayed in the documentation, reports, and web application originates from verified files on disk. Datasets without runs (Places365, ImageNet) are honestly reported as "Not run".
3. **Collective Defense**: All 5 team members have run the codebase locally, understand the mathematical formulation of Phase 2, and are prepared to navigate and explain any section of the code and live web application during oral examination.
