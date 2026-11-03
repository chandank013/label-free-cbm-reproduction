# Contribution Log — DA351 Phase 1

**Paper**: Label-Free Concept Bottleneck Models (Oikarinen et al., ICLR 2023)
**Team size**: 5

> Replace `Member 1`–`Member 5` with actual names and roll numbers before
> submitting. Percentages are a starting point — adjust to reflect what
> actually happened on your team.

| # | Name / Roll No. | Role | Key Tasks | Contribution |
|---|---|---|---|---|
| 1 | Chandan Kumar | Paper Lead | Read and summarized the paper; wrote the Phase 0 one-slide proposal (problem statement, dataset, XAI method, justification); prepared the paper-summary segment of the presentation | 20% |
| 2 | Aayush Jha | Core Experiments (small datasets) | Set up the project environment; ran and validated CIFAR-10 and CIFAR-100 for both Label-free CBM and Standard (sparse); filled in `results/*.json` accuracy from `evaluate_cbm.ipynb` for these datasets | 20% |
| 3 | Hari Krishna Sharma | Core Experiments (CUB-200) | Downloaded and extracted CUB-200-2011; ran Label-free CBM and Standard (sparse) on CUB; investigated and fixed the missing `data/CUB/train` issue (incomplete `download_cub.sh` run); produced qualitative concept-explanation examples from `evaluate_cbm.ipynb` | 20% |
| 4 | Prem Kishan | Infrastructure & Debugging (Places365) | Diagnosed and fixed the Windows-specific `pin_memory`/`num_workers` CUDA errors; diagnosed the Places365 download corruption issue and implemented the integrity-check patch; diagnosed and patched the out-of-memory crash during large-scale activation saving; documented all fixes in `docs/WINDOWS_TROUBLESHOOTING.md` | 20% |
| 5 | Lavanya Dharmadhikari | ImageNet, Repo & Reporting | Set up the official ImageNet download/extraction pipeline (`scripts/imagenet_official/`); maintained the GitHub repo structure, `README.md`, and `.gitignore`; wrote `collect_results.py` output into the final comparison table; compiled `AI_USAGE.md` and the 1-page replication summary; prepared the live-demo script for the evaluation | 20% |

## Shared responsibilities (all members)

- Reviewed and approved the final `results/comparison_table.md` before submission
- Contributed to the "Discussion" section of the replication summary, particularly explaining divergences between paper and reproduced numbers
- Present for and able to answer questions on any part of the codebase during the Phase 1 evaluation, per the "no slides, live code" format

## Notes for Phase 2

Whoever leads the paper-gap discussion for Phase 2 should build on whatever
limitation the team actually observed hands-on during replication — the
Windows/infrastructure debugging (Member 4) and the CUB concept-quality
qualitative check (Member 3) are both good candidates for a genuine, evidence-backed
research gap rather than a generic one taken from related work alone.
