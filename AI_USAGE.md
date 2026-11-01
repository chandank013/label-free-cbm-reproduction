# AI Usage Log — DA351 Phase 1

**Tool used**: Claude (Anthropic), web/chat interface
**Project**: Replication of "Label-Free Concept Bottleneck Models" (Oikarinen et al., ICLR 2023)

---

## Session 1 — Paper selection & Phase 0 proposal

**Prompts used** (in order):
- "DA351 – Explainable AI (XAI) Course Project: Research Replication and Extension Objective..." (pasted the full course project brief)
- "Provide best paper of each domains also suggest the best one"
- "yes" (requesting the Phase 0 proposal slide)

**Output received**: A shortlist of three candidate papers, one per domain (Computer Vision: Label-Free CBM, ICLR 2023; NLP: Faithfulness Tests for Natural Language Explanations, ACL 2023; Tabular/Healthcare: a SHAP-based tumor classification paper), with a recommendation to select Label-Free CBM and a justification. A one-slide PPTX proposal (title, problem statement, dataset, XAI method, reason for selection) built from that recommendation.

**How output was modified**: *[Team to fill in — e.g. "Added team name/roll numbers to the slide", "Confirmed paper choice with faculty before finalizing", "Edited reason-for-selection wording"]*

---

## Session 2 — Replication scaffold setup (Phase 1)

**Prompts used**:
- "yes i want to reproduce on all dataset"
- "Free Google Colab (limited GPU hours)" (in response to a compute-availability question)

**Output received**: A GitHub-ready repo scaffold consisting of: `original_lfcbm/` (the authors' unmodified public code, cloned from `github.com/Trustworthy-ML-Lab/Label-free-CBM`, kept with full attribution), `run_experiment.py` (a wrapper adding fixed-seed reproducibility, per-dataset hyperparameter configs matching the authors' own `training_commands.txt`, and run logging), `collect_results.py` (auto-generates a paper-vs-reproduced comparison table), a `README.md` documenting per-dataset feasibility on limited compute, and templates for the AI usage log and 1-page replication summary.

**How output was modified**: *[Team to fill in — e.g. "Ran on our own Colab/local GPU, no code changes to hyperparameters", "Adjusted which datasets we prioritized given time constraints"]*

---

## Session 3 — Switching to local GPU, driver/CUDA setup

**Prompts used**:
- "can i run on my local laptop gpu 5050"
- "in which folder it will saved automatically saved"

**Output received**: Guidance on Blackwell-generation GPU (RTX 50-series) PyTorch/CUDA compatibility requirements, VRAM/disk budgeting for an 8GB-class laptop GPU, and confirmation (via inspecting `torchvision`'s source) of exactly where each dataset auto-downloads to (`~/.cache`) versus what needed manual setup (CUB, pretrained backbones).

**How output was modified**: No code changes — this was environment/setup guidance we followed directly (installing the correct PyTorch build for our GPU, confirming dataset cache locations).

---

## Session 4 — Debugging Windows-specific runtime errors

**Prompts used** (error tracebacks pasted verbatim from our terminal, across several rounds):
- A `torch.AcceleratorError: CUDA error: resource already mapped` traceback (pin-memory thread crash)
- A `torch.OutOfMemoryError: CUDA out of memory` traceback
- "why not to keep in project directory folder" / "i will add to .gitignore"

**Output received**: Diagnosis that `original_lfcbm/utils.py` hardcoded `pin_memory=True` with `num_workers=8`, which triggers a known PyTorch/Windows CUDA race condition; a one-line fix (`pin_memory=False`). Diagnosis that the default `batch_size=512` exceeds our GPU's available VRAM; added `--batch_size`/`--saga_batch_size` CLI flags to `run_experiment.py` to override per-run. Guidance on `.gitignore` contents for large datasets/venvs.

**How output was modified**: We applied the `pin_memory=False` patch to our local copy of `original_lfcbm/utils.py` directly (one-line find-and-replace across 4 occurrences). We used `--batch_size 64` (later 32 for ResNet50-based datasets) instead of the paper's default 512.

---

## Session 5 — CUB dataset download failure

**Prompts used**:
- A `FileNotFoundError: ... data/CUB/train` traceback from `run_experiment.py`

**Output received**: Diagnosis that `download_cub.sh` (which uses `wget`) silently failed on Windows since `wget` isn't available by default; a Windows-native replacement using `curl`/`tar` (both built into Windows 10/11) to manually download and extract the CUB-200-2011 dataset and run the authors' `split_cub_dataset.py`.

**How output was modified**: Ran the provided `curl`/`tar` commands as-is in place of the broken shell script; no code changes needed beyond that.

---

## Session 6 — Places365 download corruption and server reliability

**Prompts used** (across multiple rounds, with tracebacks/screenshots pasted):
- A `FileNotFoundError` deep inside a `data_256_standard` path
- "can i solve this problem [torchvision's download_images bug]"
- A second `RuntimeError: File not found or corrupted` for the devkit file
- "Why this error [is it] due to run the command in powershell if we run that in cmd would it be correct"
- "It means due to slow network"
- A screenshot confirming a successful MD5 checksum match after re-downloading

**Output received**: Diagnosis that `torchvision`'s `Places365.download_images()` only checks folder *existence*, not completeness, before skipping re-download — meaning an interrupted download leaves permanently-corrupted cached data. A runtime monkeypatch added to `run_experiment.py` (`patch_places365_integrity_check()`) that samples expected files and forces re-download if any are missing. Manual, resumable download instructions using `curl -C -` and later `aria2c` (multi-connection, auto-retry) to work around a known-unreliable source server (`data.csail.mit.edu`), plus exact MD5 checksums to verify downloads before trusting them. Clarification that PowerShell's built-in `curl` alias (`Invoke-WebRequest`) behaves differently from the real `curl.exe`, which can itself cause silent download corruption.

**How output was modified**: Applied the monkeypatch to our local `run_experiment.py`. Used `aria2c` instead of the paper's/torchvision's built-in downloader for the large Places365 archive. Verified every downloaded archive's MD5 before proceeding, per the provided checksums.

---

## Session 7 — Windows shared-memory and system RAM limits during training

**Prompts used**:
- A `RuntimeError: Couldn't open shared file mapping ... error code: <1455>` traceback
- "So training getting slow ?" (after applying a `num_workers` fix)
- A `RuntimeError: ... not enough memory: you tried to allocate 14773944320 bytes` traceback after a 6-hour run
- "Why this error [is it due to] slow network" / follow-up debugging

**Output received**: Diagnosis of a Windows-specific shared-memory/page-file limitation tied to `num_workers=8` in DataLoader calls (fix: reduce to `num_workers=0`), with an explanation of the resulting data-loading slowdown trade-off. Diagnosis that the authors' `save_target_activations`/`save_clip_image_features` functions accumulate all per-image activations in a Python list before a single `torch.cat()` call, requiring ~2x the final tensor size in RAM simultaneously — exceeding available system memory for Places365's ~1.8M images. A tested, idempotent patch script (`patch_memory_efficient_save.py`) that pre-allocates the output tensor and streams batches into it directly, roughly halving peak RAM usage.

**How output was modified**: Applied `num_workers=0` to `original_lfcbm/utils.py`. Ran the provided `patch_memory_efficient_save.py` script to patch the memory-inefficient save functions. Increased our Windows page file size and moved it to a drive with more free space, per the provided guidance.

---

## Session 8 — Full ImageNet dataset setup (official source)

**Prompts used**:
- "I want to use original dataset from here [image-net.org registration route, listing the three official archive files]"
- "after changing will [DATASET_ROOTS edit] step 9 run on full dataset"
- "I also do not want to take this --imagenet_subset in ImageNet also so how many file code will have to change"
- "Above places365 data have been kept in d drive download folder"

**Output received**: Two custom scripts — `extract_train_tars.py` (handles the official ImageNet train archive's nested per-class-tar structure, which differs from a typical flat archive) and `reorganize_val_devkit.py` (reorganizes the flat, unlabeled official validation set into ImageFolder format using the devkit's `meta.mat` + ground-truth text file, as an alternative to a Kaggle CSV-based approach). Confirmation that only one line in `original_lfcbm/data_utils.py` (the `DATASET_ROOTS` paths) needed changing to use the full official dataset instead of the val-only substitution.

**How output was modified**: *[Team to fill in — e.g. "Ran extract_train_tars.py and reorganize_val_devkit.py on our downloaded ImageNet archives", "Edited DATASET_ROOTS in data_utils.py to point to our local paths"]*

---

## Session 9 — Presentation preparation

**Prompts used**:
- "provide the content of RUN_STEPS_LOCAL.md" (consolidating the local setup process into one reference doc)
- The professor's evaluation format email, pasted verbatim, followed by: "Description for this presentation to the teacher"
- "in this is it training from scretch or fine tuning is happenng then on which model"
- "How many classes in each dataset"

**Output received**: A timed, segment-by-segment presentation script matching the professor's specified format (paper summary, live results/code walkthrough, summary), including guidance on framing the real debugging work (Sessions 4–8 above) as a demonstration of reproducibility skill rather than a shortfall. A technical clarification that all backbones (CLIP, ResNet18/50) are used entirely frozen with no fine-tuning — only a small sparse linear layer is trained — and the class counts for each of the five datasets.

**How output was modified**: *[Team to fill in — e.g. "Filled in actual reproduced-accuracy numbers before presenting", "Rehearsed the script and adjusted timing", "Chose our own Phase 2 research gap angle based on the class-count discussion"]*

---

## Declaration

We confirm that all code execution, dataset downloading, and result
generation was performed by our team on our own compute (local GPU / Google
Colab, as documented per session above). AI assistance was used for
scaffolding the replication code around the original authors' public
repository, diagnosing and fixing real runtime errors we encountered during
our own execution (with every fix verified by us by actually running it),
and documentation/presentation preparation, as logged above. No AI tool was
used to fabricate, estimate, or guess at any reported accuracy result — all
numbers in our replication summary come from our own completed training
runs.

**Team members and roles**: *[fill in — names, roll numbers, and which
dataset(s)/tasks each member owned]*

<!-- 
## Session 4 — Phase 2 (concept quality filtering)
- **Prompt(s) used**: [paste your actual prompt, e.g. "Lets start phase 2 ... Improve the Quality of Automatically Generated Concepts"]
- **Output received**: literature check (LaBo, VLG-CBM, mRMR), a gap analysis of what LF-CBM already filters,
  the `phase2/` code (filter, metrics, generated trainer, runner, aggregator, rating study), synthetic tests, `docs/PHASE2_PLAN.md`.
- **How output was modified**: [what your team changed, which thresholds/budgets you chose and why, any bugs found when running on real data]
- **Verification status at hand-over**: synthetic-data unit tests pass; glue code smoke-tested with a numpy stand-in only.
  All real-data results must come from your own runs.


## Session 5 — Restructure: authors' code restored, Phase 2 on all five datasets
- **Prompt(s) used**: [paste your actual prompt, e.g. "provide the zip ... removing all the changes ... D:/datasets/cache ... phase 2 on all 5 data"]
- **Output received**: `original_lfcbm/` restored to upstream except the 8 download-path edits; the Windows/RAM fixes moved
  into `compat.py` (runtime, switch-off with `--no_compat`); Phase 2 runner/aggregation extended to cifar10/cifar100/cub/places365/imagenet.
- **How output was modified**: [what you changed; which fixes you actually needed on your machine]
- **Verification status at hand-over**: logic tests pass with a stand-in for torch; nothing run against real torch or datasets by the assistant. -->
