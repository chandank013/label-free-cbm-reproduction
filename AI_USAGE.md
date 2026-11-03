# AI Usage Log — DA351 Explainable AI Course Project

Covers Phase 0 (paper selection), Phase 1 (replication), Phase 2 (extension) and the demo web app.
Course policy: AI tools are allowed, but every team must record the tool, the prompts and how the output was modified. This file does that.

## 1. Tools used

| Tool | What it was used for |
|---|---|
| **Claude (Anthropic)**, claude.ai chat, including its code-execution sandbox and web search | Paper shortlist, Phase 0 slide, Phase 1 repo scaffold and debugging help, Phase 2 design and code, literature check, documents, web-app planning and prompts |
| **ChatGPT (image generation)** | Rendering the two UI mockups (Try-it page, Compare page) from prompts written by Claude |
| **AI coding agent (Antigravity)** | Implementing the web app from the specification prompt written by Claude (prompt P63 below) |

**What the AI could and could not run.** Claude's sandbox has no access to the team's GPU, datasets or saved models. It ran only *synthetic-data* tests (planted irrelevant/duplicate concepts, logic tests with stand-ins for PyTorch) and generated files. **Every accuracy, concept count, redundancy and relevance number in this project was produced by the team running the code on its own machine.** No AI tool produced or altered experimental results.

## 2. How to read the prompt log

- Prompts are quoted **verbatim from the chat history** (spelling and typos kept), numbered P1–P72 in order.
- `[square brackets]` mean pasted material (terminal logs, screenshots, files, tables) that is summarised here instead of copied.
- Stages: **A** Phase 0 · **B** Phase 1 · **C** Phase 2 · **D** Web app · **E** Phase 2 deliverables and documentation.

---

## Stage A — Phase 0: paper selection and proposal slide

- **P1** `[Pasted the full course brief: "DA351 – Explainable AI (XAI) Course Project: Research Replication and Extension" — team rules, Phase 0/1/2 tasks and deadlines, AI-usage policy, plagiarism policy, reproducibility requirements]`
- **P2** `[Button answer to "What would you like help with first?"]` → "Finding/selecting a paper (Phase 0)"
- **P3** "Provide best paper ofeach domains also suggest the best one"
- **P4** "yes" *(asked Claude to draft the Phase 0 one-slide proposal)*

**AI output:** shortlist of one paper per domain (CV, NLP, tabular/healthcare) with a recommendation of *Label-Free Concept Bottleneck Models* (ICLR 2023); a one-slide PPTX proposal (`DA351_Phase0_Proposal.pptx`).
**How it was modified / checked:** the team chose the paper. Later checking found a typo in the OpenReview link on the slide (see Section 4).

---

## Stage B — Phase 1: replication of the paper

- **P5** "yes i want to reproduce on all dataset"
- **P6** `[Button answer to "What compute do you have access to for training?"]` → "Free Google Colab (limited GPU hours)"
- **P7** "can i run on my local laptop gpu 5050"
- **P8** "after running all .sh command still i have to download the data separately of directly run the command of training"
- **P9** "in which folder it will saved automatically saved"
- **P10** "Why this error" `[pasted traceback: torch.AcceleratorError "CUDA error: resource already mapped" in the pin-memory thread, CIFAR-10 run]`
- **P11** `[Pasted terminal output: "torch.OutOfMemoryError: CUDA out of memory", 6.00 GiB GPU, CIFAR-10 run — no typed text]`
- **P12** "tell me the training through next dataset i have to used same like this command python run_experiment.py --dataset cifar10 --model lfcbm --seed 0 --batch_size 64"
- **P13** "So provide the content of RUN_STEPS_LOCAL.md to update there provide in markdown i will paste there # Step-by-Step: Running the LF-CBM Replication Locally (RTX 5060)" `[followed by the team's pasted draft of the 12-step local guide]`
- **P14** "in phase 8 can we run loacally on full data just tell me"
- **P15** "also can we run step 9 locally"
- **P16** "i also do not want to  take this --imagenet_subset in ImageNet also so how many file code will have to change"
- **P17** "after changing will DATASET_ROOTS = { "imagenet_train": "D:/datasets/ImageNet/train/", "imagenet_val": "D:/datasets/ImageNet/val/", "cub_train":"data/CUB/train", "cub_val":"data/CUB/test" } step 9 run on full dataset"
- **P18** "I want to use original dataset from here **Option B — Official image-net.org** 1. Register at https://image-net.org/download.php — approval can take a few days for new accounts, so start this immediately if you haven't 2. Download `ILSVRC2012_img_train.tar` (~138GB) and `ILSVRC2012_img_val.tar` (~6.3GB), plus the devkit (`ILSVRC2012_devkit_t12.tar.gz`) for ground truth 3. This route needs `reorganize_val_devkit.py` (Step 3, devkit variant) instead of the Kaggle CSV variant, since it doesn't include `LOC_val_solution.csv`"
- **P19** "Why this error" `[pasted traceback: FileNotFoundError 'data/CUB/train' on the CUB run]`
- **P20** "thanks"
- **P21** `[Pasted terminal output: Places365 run, FileNotFoundError on ...\data_256_standard\a\airfield\00000001.jpg — no typed text]`
- **P22** "can i solve this problem def download_images(self) -> None: if path.exists(self.images_dir): return   # <-- just returns, no integrity check at all ..."
- **P23** "I have already download the 5th data alsoin main directory so where i have to place that data of 138 gb and training data and how to run'"
- **P24** "why not to keep in project directory folder"
- **P25** "i will add to .gitignore"
- **P26** `[Pasted terminal output: Places365 devkit download failed at ~24 kB/s, RuntimeError "File not found or corrupted", then categories_places365.txt missing — no typed text]`
- **P27** "Is the above error is due to run the command in powershell if we run that in cmd would it be correct"
- **P28** "I am taking about this error" `[pasted the same devkit error again]`
- **P29** "It means due to slow network"
- **P30** `[Screenshot: cmd session — curl download of filelist_places365-standard.tar and certutil MD5 check — no typed text]`
- **P31** `[Screenshot: Places365 run showing a 26.1 GB download at ~121 kB/s — no typed text]`
- **P32** "winget install aria2.aria2 aria2c -c -x 8 -s 8 --max-tries=0 --retry-wait=10 -d "$env:USERPROFILE\.cache" -o train_256_places365standard.tar http://data.csail.mit.edu/places/places365/train_256_places365standard.tar CMD command"
- **P33** `[Pasted terminal output: Places365 run, "RuntimeError: Couldn't open shared file mapping ... error code: <1455>" — no typed text]`
- **P34** "Have I to change the code of util"
- **P35** "Why training getting slow ?"
- **P36** "error hai" `[pasted output: Places365 finished 56,359 batches, then "not enough memory: you tried to allocate 14773944320 bytes", plus follow-up commands]`
- **P37** "Above placenet365 data have been kept in d drive download folder"
- **P38** "What i have to do now"
- **P39** "Dear All, The project evaluation on 29th Sept 2026 is attached here for your reference. Check the time for your team and attend without fail. If you are absent for the presentation, there will be no marks for Phase 1. Also, during the presentation, no need to prepare any slide. First 2 to 3 minutes - Open the research article and summarize it Next 10 minutes - Show and explain the results from the research paper and then show your code and compare the results with the research paper. Last 2 minutes - Summarize the results Description for this presentation to the teacher"
- **P40** "in this is it training from scretch or fine tuning is happening tell me if fine tuning is happenng then on which model"
- **P41** `[Pasted the paper's results table: Standard / Standard (sparse) / Label-free CBM across the five datasets]` "What is this standerd accuracy that is not in my result"
- **P42** "yes" *(asked Claude to document the "Standard row" explanation in the README and docs)*
- **P43** "How many file have you change and why"
- **P44** "Provide workflow  and working with explaination  in markdown format"
- **P45** "Provide contribution of five people"
- **P46** "Thanks"

**AI output:** `run_experiment.py` (seeded wrapper around the authors' code), `collect_results.py`, README and docs (dataset commands, local-GPU setup, Windows troubleshooting, workflow, contribution log, presentation script, replication-summary template), `scripts/imagenet_official/` (ImageNet extraction helpers), the original in-place patches to the authors' `utils.py` and `patch_memory_efficient_save.py`, and diagnoses of each error the team pasted.
**How it was modified / checked:** the team ran every command on its own laptop and reported each failure; fixes were applied only after those real errors (pin-memory crash, Windows paging-file error, missing CUB data because `wget` is not available on Windows, corrupted/interrupted Places365 downloads, a ~14.8 GB RAM spike). Downloads were redone with `curl.exe`/`aria2c` and MD5 checks. The in-place edits to the authors' files were later **removed** (P52) and replaced by a runtime layer, `compat.py`.

---

## Stage C — Phase 2: concept-quality filtering

- **P47** "Lets start phase 2 Phase 2 — Our Extension After reproducing the original method, we will investigate its limitations. One possible limitation is concept quality. 4The model may generate concepts that are: Irrelevant For example: • • • • Animal Object Beautiful Nature These may not be useful for identifying the class. Redundant For example: • • • • Long beak Large beak Pointed beak Sharp beak Several concepts may describe almost the same visual characteristic. Therefore, our proposed Phase 2 extension is: Improve the Quality of Automatically Generated Concepts We can add a Concept Quality Filtering module: Label-Free CBM → Generated Concepts → Concept Filter → Better Concepts → Prediction The filter can attempt to: • • • Remove irrelevant concepts Remove redundant concepts Keep useful and meaningful concepts Then we will compare: Original Label-Free CBM VS Improved Label-Free CBM using metrics such as: • • • Classification accuracy Number of concepts Concept redundancy 5• • Concept relevance Interpretability Phase 1 Paper → Label-Free CBM → CIFAR-10/CUB-200 → Reproduce Results → Evaluate ↓ Phase 2 Analyze Limitations → Improve Concept Quality → Filter Concepts → Improved LF-CBM → Compare Results Our final comparison will be: Original Label-Free CBM VS Our Improved Label-Free CBM"
- **P48** "Tell me how many dataset used for phase 2"
- **P49** "Then why did you write phase 1 in zip file name just tell me only"
- **P50** "How many changes have you make like this patch_memory_efficient_save to fix error in the author' repo code till now tell me only"
- **P51** "I want to ask why the the first data saved in the cashed folder of c drive can i saved that in another drive just tell me only"
- **P52** "Now provide the zip file of the code by removing all the changes that you make till now and change the path of the of data to download in "D:/datasets/cache" only, keep this pretrained model weights in C drive  and provide zip file with phase 2 and i want to implements phase 2 on all 5 data not only on 3 data. Now provide the zip"
- **P53** "Discuss with me how the pgase two  resul will different from the original tell me"

**AI output:** an analysis of what LF-CBM already filters (so the gap is stated accurately); a literature check by web search (LaBo, CVPR 2023; VLG-CBM, NeurIPS 2024; mRMR, TPAMI 2005); the Phase 2 method and code in `phase2/` (`concept_quality.py`, `make_filtered_trainer.py` → generated `train_cbm_filtered.py`, `run_phase2.py`, `aggregate_phase2.py`, `rating_study.py`, tests); `compat.py` (runtime Windows/RAM fixes, replacing edits to the authors' files); the restructured project in which `original_lfcbm/` equals upstream commit `fa2869d` except for the dataset download path; `docs/PHASE2_PLAN.md`; and a discussion of expected results backed by one synthetic simulation.
**How it was modified / checked:** the algorithm and code were written by Claude from the team's outline (P47). The team ran all experiments (3 seeds, 3 budgets, controls and ablations on CIFAR-10, CIFAR-100, CUB-200) and produced every result file. Claude's synthetic tests found a design flaw (an averaged redundancy penalty let duplicates through), which led to the maximum-correlation penalty now used (Section 4).

---

## Stage D — Demo web app

- **P54** "I have to make a web app that would be show the comprision of the both pahse one and two and also there we can test both phase with the real world image so discuss first how can we implement this and provide the prompt to generate the image with ChatGPT of the frontend"
- **P55** "discuss the teck steck used for web app"
- **P56** "First tell me which folder and file and  are required from directory that nee in backend and frontend"
- **P57** "Also tell me where the web app will located in the directory"
- **P58** "We have done the phase one and phae 2 on first three dataset only"
- **P59** "I have all the download models in my saved_models folder and so Tell me which file code i need to change now"
- **P60** "show me the full and final directory with the web app in the chat"
- **P61** "Thanks"
- **P62** `[Uploaded the two ChatGPT-generated mockup images: Try-it page and Compare page]` "Provide the zip file code of web app with this fronend design that i will add to the main directory"
- **P63** "Provide prompt to paste in the antygravity that will taste what ever required and add web app same as the attached frontend image in the main directory"
- **P64** "Phase two is running now so what the next and when i have to handover the project to anty gravity to make web app"

**AI output:** the web-app design (FastAPI backend + plain HTML/JS frontend, no database, local only), the folder plan, two prompts for ChatGPT image generation, and the specification prompt for the coding agent (P63) including data contracts, honesty rules (no invented numbers, "Not run" states) and a test plan. Claude could **not** produce the web-app zip requested in P62 (its sandbox was unavailable at that time), so it wrote the agent prompt instead.
**How it was modified / checked:** the team generated the mockup images with ChatGPT, implemented the web app (with the coding agent, from the prompt above) and added it to the project. The web app only reads the team's own result files and trained models.

---

## Stage E — Phase 2 deliverables and documentation

- **P65** `[Pasted the Phase 2 evaluation brief: tasks, 10-minute video structure, mandatory deliverables (video, original paper PDF, results-and-comparison PDF, code + README + GitHub link), evaluation considerations, submission checklist]`
- **P66** "yes" *(asked Claude to draft the one-page gap explanation and the video script)*
- **P67** "First tell have you added page number"
- **P68** "Do you want me to turn the one-pager into a Word file when my tools are working again, or adjust the wording first? Now  Yes"
- **P69** "Provide the final dosc file of above according to the guidelines"
- **P70** `[Uploaded two PDFs: (1) "Experimental_Results_Comparison_Report.pdf", a sample report from a different project (counterfactual simulatability of language-model explanations), used only as a format reference; (2) the team's paper PDF "3646_label_free_concept_bottleneck_research_paper1.pdf"]` "I want the docs file in this format not same but take some references from this and here is our real research paper what ever required to you paste in this docs tell me first with page number i will paste you and als if any things required from the saved jsona and csv result first tell i will give you then you will give me the final docs so tell me if any thing required you first"
- **P71** `[Pasted result files: results/phase2/cub_summary.md, results/phase2/ALL_datasets_summary.md, results/comparison_table.md, and the rating-study key and blank sheet as CSV]` "Run counts: 30 each github chandank013 rating output" … "also try to follow the submission guidelines"
- **P72** "provide the ai usage and the fina; read me of phase one and phase 2 and add every prompt that i gave in the history add in the ai usage i will not paste from my side so it will fine you add and i also imple ment the web page so add that in the readme.md also provide only ai usage and readme only"

**AI output:** a one-page research-gap explanation and a timed video script (drafts, with result placeholders); an extraction of paper page numbers and a list of corrections after reading the paper PDF; the Phase 2 *Experimental Results and Comparison* document, with **[PUBLISHED, p.N]** and **[OURS]** tags, built from the team's pasted result files; this file and the final README.
**How it was modified / checked:** page numbers were checked against the team's 28-page PDF. All numbers in the results document come from the team's pasted files. The sample report from another project was used only for structure ideas (published-vs-ours tagging, extract slots with page numbers, setup comparison, "what it does and does not show"); **none of its text, numbers or figures was copied.** Placeholders were left for items only the team can supply (team names, hardware, screenshots, figures).

---

## 3. Files in the repository written or substantially drafted with AI help

| File or folder | Written by | Team's role |
|---|---|---|
| `run_experiment.py`, `collect_results.py`, `compat.py` | Claude | Ran, debugged on real hardware |
| `phase2/` (filter, trainer generator, runner, aggregation, rating tool, tests) | Claude | Ran all experiments; chose to run 3 of 5 datasets |
| `scripts/imagenet_official/`, `scripts/move_cache_to_d.bat` | Claude | Used as needed |
| `docs/` (plan, workflow, troubleshooting, dataset commands, presentation script, contributions) | Claude, from the team's messages | Reviewed; filled in the real details |
| `README.md`, `AI_USAGE.md` | Claude, from the chat history | Reviewed |
| `webapp/` | Coding agent, from a prompt written by Claude | Integrated and ran it |
| UI mockup images | ChatGPT, from prompts written by Claude | Generated and supplied the images |
| `original_lfcbm/` | Authors (Oikarinen et al., ICLR 2023) | One edit: dataset download path in `data_utils.py` |

## 4. Errors in AI output that were found and corrected

Recorded so that nobody relies on an earlier statement that was wrong.

1. **Paper link typo.** The Phase 0 slide gave the OpenReview id `FlCq47MNvBA`; the correct id is `FlCg47MNvBA`.
2. **Wrong "published" numbers.** The AI took the Label-free CBM numbers from the authors' GitHub README (86.37 / 65.27 / 74.59 / 43.71 / 71.98). The paper's Table 2 (p.6) says 86.40 / 65.13 / 74.31 / 43.68 / 71.95 (mean over three runs). Reports use Table 2.
3. **What is trained.** The AI said the sparse final layer was "the only trained part". The paper trains a concept projection layer *and* a sparse final layer; only the backbone is frozen.
4. **Incomplete description of the authors' filters.** Phase 2 was first described with three filters. The paper (p.4–5) has five, including a text-space duplicate filter (> 0.9); "animal" is already removed on CIFAR-10 for being too close to "dog" (p.14).
5. **Design flaw in the filter.** An averaged redundancy penalty let duplicates through; found by a synthetic test and replaced by the maximum-correlation penalty.
6. **Memory bug.** The top-k baseline sorted a full copy of the concept-score matrix; rewritten to work in chunks.
7. **Missing fix in a delivered zip.** One zip lacked the `num_workers` fix that its documentation claimed; caught on re-check.
8. **Stale script.** `move_cache_to_d.bat` would have moved model weights off the C: drive; replaced.
9. **Concepts-per-class figures.** First computed from the shipped concept files; corrected with the paper's final counts (p.6).
10. **Wrong date assumption.** An early plan assumed a start date that did not match the real date.
11. **Open issue.** *Standard (sparse)* on CIFAR-10 gave 88.33% against 82.96% published. The suspected cause is a default λ (0.0007) in our wrapper, instead of the authors' script default. It has **not** been re-run; the result is reported as not reproduced.

## 5. Declaration

All code execution, dataset downloading and result generation were performed by the team on its own machine. AI assistance was used for scaffolding, code, debugging advice, documentation and drafting, as logged above. No AI tool was used to fabricate, estimate or alter experimental results. Third-party code (the authors' LF-CBM repository), datasets and papers are cited in `README.md`.
