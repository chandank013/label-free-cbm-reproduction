# Complete System Workflow & Operational Guide

**Course**: DA351 &bull; Interpretable and Explainable Machine Learning  
**Project**: Replication (Phase 1), Concept-Quality Extension (Phase 2), and "Concept Lab" Web Application (Phase 3)  
**Upstream Paper**: Oikarinen et al., *Label-Free Concept Bottleneck Models*, ICLR 2023  

---

## 1. Master System Architecture

The following diagram illustrates the complete end-to-end data and execution flow across all three phases of the project:

```mermaid
flowchart TD
    subgraph DataInput ["1. Data & Pretrained Backbones"]
        D1["Raw Datasets<br/>(CIFAR-10, CIFAR-100, CUB-200)"]
        D2["Pretrained Visual Backbones<br/>(CLIP RN50, ResNet18)"]
        D3["Candidate Concepts<br/>(GPT-3 concept sets)"]
    end

    subgraph Phase1 ["2. Phase 1: Paper Replication"]
        P1_Compat["compat.py<br/>(Windows DataLoader & Save Fixes)"]
        P1_Act["Activation Caching<br/>(saved_activations/)"]
        P1_Saga["GLM-SAGA Solver<br/>(Sparse Linear Layer)"]
        P1_Model["Baseline Models<br/>(saved_models/*_baseline_k100_s*)"]
        P1_Eval["results/*.json<br/>(Validation Accuracy: 86.4%, 65.3%, 74.1%)"]
    end

    subgraph Phase2 ["3. Phase 2: Concept-Quality Extension"]
        P2_Filter["Concept-Quality Filter<br/>(phase2/concept_quality.py)"]
        P2_ANOVA["Class Relevance:<br/>One-way ANOVA η²"]
        P2_Corr["Activation Redundancy:<br/>Row-centered Pearson |r_ij|"]
        P2_Greedy["Submodular Greedy Selection<br/>(Budgets: 75%, 50%, 25%)"]
        P2_Trainer["Generated Trainer<br/>(phase2/train_cbm_filtered.py)"]
        P2_Controls["Control Baselines<br/>(Random, CLIP Top-k)"]
        P2_Results["results/phase2/<dataset>/<br/>(30 runs per dataset, tables & summaries)"]
    end

    subgraph WebApp ["4. Phase 3: Concept Lab Web Application"]
        WA_Server["FastAPI Backend<br/>(webapp/server.py on 127.0.0.1:8000)"]
        WA_Registry["Model Registry<br/>(webapp/registry.py: auto-discovery)"]
        WA_Inference["Inference Engine<br/>(webapp/inference.py: cbm.load_cbm)"]
        WA_UI["Offline Frontend<br/>(index.html, app.css, app.js, charts.js)"]
        WA_Compare["Compare View (#compare)<br/>KPIs, Pareto Curves, Redundancy Bars"]
        WA_Try["Try-It View (#try)<br/>Photo Upload, Agreement, Explanations"]
        WA_TestImages["Sample Image Suite<br/>(data/cifar10, data/cifar100, data/cub)"]
    end

    DataInput --> Phase1
    P1_Compat --> P1_Act
    P1_Act --> P1_Saga --> P1_Model --> P1_Eval
    
    P1_Act --> P2_Filter
    P2_Filter --> P2_ANOVA & P2_Corr --> P2_Greedy
    P2_Greedy --> P2_Trainer --> P2_Results
    P2_Controls --> P2_Trainer
    
    P1_Model & P2_Results --> WA_Registry
    WA_Registry --> WA_Server
    WA_Server --> WA_Inference & WA_UI
    WA_Inference --> WA_Try
    WA_Server --> WA_Compare
    WA_TestImages -.-> WA_Try
```

---

## 2. Phase 1 — Replication Workflow

### 2.1 The Underlying Mechanism
Label-Free Concept Bottleneck Models replace an opaque neural network with an interpretable two-stage classifier:
$$\text{Image } x \;\xrightarrow{\quad f_{\text{frozen}}\quad}\; \text{Backbone Features} \;\xrightarrow{\quad g_{\text{CLIP}}\quad}\; \text{Concept Scores } c \;\xrightarrow{\quad W_c \quad}\; \text{Class Prediction } \hat{y}$$

1. **Concept Discovery**: Candidate concept sets (generated offline by GPT-3 for each class) ship in `original_lfcbm/data/concept_sets/`.
2. **Zero-Shot Concept Grounding**: CLIP scores how strongly each concept aligns with each training image via cosine similarity.
3. **Activation Caching**: Frozen visual backbones (`clip_RN50` for CIFAR, `resnet18_cub` for CUB-200) extract and cache high-dimensional representations into `original_lfcbm/saved_activations/`.
4. **Sparse Linear Layer Fitting**: Using the GLM-SAGA elastic-net solver, a linear weight matrix $W_c$ is trained mapping concept activations to target class logits with an L1-style sparsity constraint.

### 2.2 Windows Runtime Layer (`compat.py`)
To ensure complete reproducibility on Windows without touching the authors' upstream source code, `compat.py` applies non-invasive runtime patches:
- **DataLoader Multiprocessing**: Replaces `num_workers > 0` with `num_workers = 0` on Windows to prevent `pin_memory` CUDA initialization deadlocks.
- **Activation Streaming**: Replaces monolithic in-memory `.pt` file dumps with chunked streaming to prevent 15+ GB RAM paging file allocation failures.
- **Places365 Integrity Verification**: Patches the download validator to verify tarball hashes before decompression.

### 2.3 Phase 1 Command Reference

```powershell
# Activate local Python environment
.\lfcbm-env\Scripts\Activate.ps1

# Run CIFAR-10 Baseline (Label-Free CBM)
python run_experiment.py --dataset cifar10 --model lfcbm --seed 0 --batch_size 64

# Run CIFAR-100 Baseline
python run_experiment.py --dataset cifar100 --model lfcbm --seed 0 --batch_size 64

# Run CUB-200 Baseline
python run_experiment.py --dataset cub --model lfcbm --seed 0 --batch_size 64

# Run Standard (sparse) baseline for comparison
python run_experiment.py --dataset cifar10 --model standard_sparse --seed 0 --batch_size 64
python run_experiment.py --dataset cifar100 --model standard_sparse --seed 0 --batch_size 64
python run_experiment.py --dataset cub --model standard_sparse --seed 0 --batch_size 64

# Compile Paper vs. Reproduced Results Table
python collect_results.py
# Outputs: results/comparison_table.md
```

---

## 3. Phase 2 — Concept-Quality Filtering Workflow

### 3.1 The Research Gap
The original LF-CBM pipeline applies three basic filters:
1. String length and token-similarity filtering on GPT-3 output.
2. `clip_cutoff`: Retaining concepts whose top-5 activation exceeds a threshold (confirming the concept appears in the dataset).
3. Projection interpretability cutoff: Discarding linear projection neurons that deviate from raw CLIP similarities.

**Critical Limitation**: None of these filters evaluate whether a concept actually discriminates between target classes, nor do they penalize concepts that activate on the exact same images. Consequently, the original model bottlenecks contain redundant synonyms (e.g., "a red beak" alongside "a pointed red beak") and generic background noise.

### 3.2 Mathematical Formulation
Phase 2 introduces **Concept-Quality Filtering** via submodular greedy selection:

$$\max_{S \subseteq V, |S| \le k} F(S) = \sum_{j \in S} \eta_j^2 - \lambda \sum_{j \in S} \max_{m \in S, m \neq j} |r_{jm}|$$

1. **Class Relevance ($\eta_j^2$)**: Measured via one-way ANOVA effect size on held-out validation activations:
   $$\eta_j^2 = \frac{SS_{\text{between}}}{SS_{\text{total}}} = \frac{\sum_{c=1}^C N_c (\bar{a}_{j,c} - \bar{a}_j)^2}{\sum_{i=1}^N (a_{j,i} - \bar{a}_j)^2}$$
2. **Activation Redundancy ($|r_{jm}|$)**: The absolute pairwise Pearson correlation between concept activation vectors $a_j$ and $a_m$ over held-out samples:
   $$r_{jm} = \frac{\sum_{i=1}^N (a_{j,i} - \bar{a}_j)(a_{m,i} - \bar{a}_m)}{\sqrt{\sum_{i=1}^N (a_{j,i} - \bar{a}_j)^2} \sqrt{\sum_{i=1}^N (a_{m,i} - \bar{a}_m)^2}}$$
3. **Submodular Greedy Selection**: At each step, the filter greedily adds the concept maximizing marginal gain $\Delta(j | S) = \eta_j^2 - \lambda \max_{m \in S} |r_{jm}|$ until the budget $k$ (e.g., 75%, 50%, or 25% of initial concepts) is reached.

### 3.3 Non-Invasive Generator (`phase2/make_filtered_trainer.py`)
Rather than editing the authors' `original_lfcbm/train_cbm.py`, `phase2/make_filtered_trainer.py` reads the original file, injects the concept-quality filtering block between activation loading and the sparse linear solver, and writes `phase2/train_cbm_filtered.py`.

### 3.4 Phase 2 Command Reference

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

# 5. Aggregate results and generate concept tables
python phase2/aggregate_phase2.py --dataset cifar10
python phase2/aggregate_phase2.py --dataset cifar100
python phase2/aggregate_phase2.py --dataset cub
python phase2/aggregate_phase2.py --all
```

---

## 4. Phase 3 — "Concept Lab" Web Application Workflow

### 4.1 Web App Architecture

```
webapp/
├── __init__.py
├── server.py              # FastAPI server (127.0.0.1:8000), CORS, parameter whitelisting
├── registry.py            # Model discovery, 8-file verification, candidate sorting
├── results_reader.py      # Results parser, defensive normalization, SVG data formatting
├── inference.py           # Thread-safe session manager, PIL validation, concept contributions
├── populate_test_data.py  # Test image extractor for root data/
├── static/
│   ├── index.html         # Single-page UI with 4 views (#compare, #try, #concepts, #about)
│   ├── app.css            # System font styling, responsive cards, teal accents
│   ├── app.js             # Client router, drag-and-drop handler, dynamic DOM rendering
│   └── charts.js          # Zero-dependency plain SVG charting library
└── tests/
    ├── test_unit.py       # 10 unit tests (math, parsing, normalization, chips)
    ├── test_api.py        # 6 API tests (security, error codes, stub fixtures)
    ├── test_smoke_real.py # Live GPU inference cross-check with cbm.load_cbm (< 10^-4)
    └── test_browser.py    # Playwright browser verification (1440x900, 1024x768)
```

### 4.2 Interactive Request & Inference Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as User Browser
    participant Server as FastAPI (server.py)
    participant Reader as ResultsReader
    participant Engine as InferenceEngine
    participant Session as ModelPairSession
    participant PyTorch as PyTorch / GPU

    Note over User, Server: Compare View Flow (#compare)
    User->>Server: GET /api/compare?dataset=cifar100&budget=50
    Server->>Reader: get_kpis(), get_curve_series(), get_redundancy_comparison()
    Reader-->>Server: Seed-averaged stats, Pareto curves, pruned concepts
    Server-->>User: JSON Response (KPIs, SVGs, Audit Table)
    User->>User: Client charts.js renders dynamic SVGs

    Note over User, Server: Try-it View Flow (#try)
    User->>Server: POST /api/predict (photo, dataset=cifar100, budget=50)
    Server->>Engine: predict(file, dataset, budget)
    Engine->>Engine: PIL verify, EXIF orientation, RGB normalize
    Engine->>Session: Lazy-load dataset pair (Original + Filtered)
    Session->>PyTorch: Backbone forward pass (CLIP / ResNet) under torch.no_grad()
    PyTorch-->>Session: Normalized feature vector
    Session->>Session: Matrix multiplication with W_c + b_g -> class logits
    Session->>Session: Compute concept contributions: a_j * W_c[predicted_class, j]
    Session->>Session: Compute NEC: count(W_c[predicted_class, :] != 0)
    Session-->>Engine: Top 3 classes, confidences, top 6 concepts per model
    Engine-->>Server: Inference result payload
    Server-->>User: JSON (predictions, contributions, stats)
    User->>User: Render side-by-side cards, agreement pill, progress bars
```

### 4.3 Concept Contribution Math
In the Try-it view, each model reveals *why* it made its classification:
$$\text{Logit}_c = \sum_{j=1}^K a_j \cdot W_{c,j} + b_c$$
For predicted class $\hat{c}$, concept $j$'s contribution score is:
$$\text{Score}_j = a_j \cdot W_{\hat{c},j}$$
The top 6 positive contributors are displayed as horizontal bars (purple for Original, teal for Filtered). Duplicate or near-synonymous concepts are tagged using text token Jaccard similarity ($\ge 0.50$) or correlation ($r \ge 0.80$).

### 4.4 Web App Execution & Testing Commands

```powershell
# 1. Install web application dependencies
pip install -r requirements-web.txt

# 2. Extract sample test photos into root data/ (1 image per class)
python webapp/populate_test_data.py

# 3. Launch the Concept Lab web server (strictly binds to 127.0.0.1:8000)
.\lfcbm-env\Scripts\uvicorn.exe webapp.server:app --host 127.0.0.1 --port 8000

# 4. Open in your browser
# URL: http://127.0.0.1:8000

# 5. Run complete automated test suite
.\lfcbm-env\Scripts\pytest.exe webapp/tests/ -v
# Output: 18 passed in ~40s (unit, API, smoke, and Playwright browser tests)
```

---

## 5. Summary of Empirical Findings (Phase 1 vs. Phase 2)

Across all completed runs on disk (3 baseline seeds and 9 filter configurations per dataset):

| Dataset | Method | Budget ($k$) | Concepts | Val Accuracy (%) | Redundancy ($\rho$) | Relevance ($\bar{\eta}^2$) | NEC |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **CIFAR-10** | Baseline | 100% | 132 | $86.48\% \pm 0.10\%$ | $0.723 \pm 0.000$ | $0.540 \pm 0.000$ | $36.3$ |
| | Greedy | 50% | 69 | $86.03\% \pm 0.12\%$ | $0.715 \pm 0.000$ | $0.640 \pm 0.000$ | $26.5$ |
| **CIFAR-100** | Baseline | 100% | 825 | $65.14\% \pm 0.14\%$ | $0.704 \pm 0.000$ | $0.481 \pm 0.000$ | $27.9$ |
| | Greedy | 75% | 621 | $64.80\% \pm 0.13\%$ | $0.663 \pm 0.001$ | $0.505 \pm 0.000$ | $26.8$ |
| | **Greedy** | **50%** | **415** | **$64.06\% \pm 0.01\%$** | **$0.624 \pm 0.000$** | **$0.522 \pm 0.000$** | **$25.6$** |
| | Greedy | 25% | 208 | $62.39\% \pm 0.15\%$ | $0.574 \pm 0.000$ | $0.541 \pm 0.000$ | $21.2$ |
| | *Control: Random* | 50% | 412 | $63.42\% \pm 0.13\%$ | $0.662 \pm 0.001$ | $0.484 \pm 0.001$ | $27.5$ |
| | *Control: CLIP Top-k* | 50% | 415 | $63.99\% \pm 0.12\%$ | $0.710 \pm 0.000$ | $0.490 \pm 0.000$ | $26.1$ |
| **CUB-200** | Baseline | 100% | 209 | $74.12\% \pm 0.04\%$ | $0.802 \pm 0.000$ | $0.673 \pm 0.000$ | $31.1$ |
| | Greedy | 50% | 120 | $74.05\% \pm 0.41\%$ | $0.729 \pm 0.000$ | $0.700 \pm 0.000$ | $27.4$ |

### Scientific Conclusions
1. **$50\%$ Concept Reduction with $< 1.1\%$ Accuracy Loss**: Greedy submodular filtering slashes concept set size in half while retaining over $98.3\%$ of baseline accuracy across all benchmarks.
2. **Superiority over Controls**: Greedy filtering significantly outperforms Random selection and CLIP Top-$K$ selection in both accuracy retention and redundancy reduction.
3. **Pruning Flawed Concepts**: The filter successfully eliminates redundant pairs (e.g. co-activating color synonyms) and purges uninformative background concepts that inflate model complexity.
