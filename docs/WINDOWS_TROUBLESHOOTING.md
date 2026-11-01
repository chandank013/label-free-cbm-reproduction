# Windows + laptop-GPU troubleshooting

Every problem below was hit for real while replicating on a Windows laptop.
Items 1–4 are handled **automatically** by `compat.py` (applied at runtime by
`run_experiment.py` and `phase2/run_phase2.py`); the authors' files in
`original_lfcbm/` are not edited (the only edit there is the download path in
`data_utils.py`). Pass `--no_compat` to run the authors' code exactly as published.

| # | Symptom | Cause | Handled by compat.py? |
|---|---|---|---|
| 1 | `CUDA error: resource already mapped` (pin-memory thread) | `utils.py` uses `pin_memory=True` with `num_workers=8`; on Windows + recent NVIDIA drivers the threads race to pin the same page | yes — Windows only: `pin_memory=False` |
| 2 | `Couldn't open shared file mapping ... error code 1455` | Windows backs worker shared memory with the page file; it was too small | yes — Windows only: `num_workers=0` |
| 3 | `FileNotFoundError` inside `...data_256_standard\a\airfield\00000001.jpg` | torchvision's `Places365.download_images()` only checks that the image **folder exists**, so an interrupted extraction is reused forever | yes — samples 50 expected files and re-downloads if any are missing |
| 4 | `RuntimeError: not enough memory: you tried to allocate 14773944320 bytes` at the very end of a long run | the authors build a list of per-batch tensors and `torch.cat` it: list **and** result in RAM (~2× final size) | yes — output tensor pre-allocated, chunks freed as copied (same saved files) |
| 5 | `FileNotFoundError: 'data/CUB/train'` | `download_cub.sh` needs `wget`, absent on Windows, so CUB never downloaded | no — manual, below |
| 6 | Places365 / devkit download crawls or fails (`File not found or corrupted`) | `data.csail.mit.edu` is slow/flaky; torchvision cannot resume | no — manual, below |
| 7 | `expandable_segments not supported on this platform` | CUDA allocator option is Linux-only | harmless; `compat.py` no longer sets it on Windows |

Trade-off of #2: data loading is single-process, so the GPU waits for JPEG decoding.
Alternative that keeps workers: enlarge the Windows page file (Advanced System Settings →
Performance → Advanced → Virtual Memory → custom size on a drive with space → reboot) and
run with `--no_compat` only if you also accept #1/#4 risks.

## Where data and weights go

| What | Where |
|---|---|
| CIFAR-10/100, Places365 (auto-download) | `D:/datasets/cache` (set in `original_lfcbm/data_utils.py`) |
| CUB-200 | `original_lfcbm/data/CUB/` |
| ImageNet | `D:/datasets/ImageNet/{train,val}/` (or `--imagenet_root`) |
| Cached activations, trained models | `original_lfcbm/saved_activations/`, `original_lfcbm/saved_models/` |
| Pretrained weights (CLIP, ResNet, pytorchcv) | default torch locations on **C:** (`~/.torch/models`, `~/.cache/torch`) |

Keep datasets out of the git-tracked folder (`.gitignore` covers `*.tar`, `datasets/`, `saved_*`).

## 5. CUB download without `wget`

```powershell
cd original_lfcbm
curl.exe -L -o CUB_200_2011.tgz https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar xf CUB_200_2011.tgz -C data
del CUB_200_2011.tgz
cd data
python split_cub_dataset.py
rmdir /s /q CUB_200_2011
del attributes.txt
cd ..
dir data\CUB\train          # ~200 class folders
```

## 6. Slow or corrupted Places365 downloads

**PowerShell trap:** typing `curl` in PowerShell runs `Invoke-WebRequest`, not curl — no `-L`/`-C -`/`-o`
support, and it can save a broken file silently. Use `curl.exe`, or run in `cmd`. `Get-Command curl` shows which you have.

Download with resume, then verify the MD5 *before* trusting the file (names/hashes from torchvision's source):

```powershell
winget install aria2.aria2          # multi-connection + endless retry, same in cmd and PowerShell
aria2c -c -x 8 -s 8 --max-tries=0 --retry-wait=10 -d "D:\datasets\cache" -o filelist_places365-standard.tar http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar
aria2c -c -x 8 -s 8 --max-tries=0 --retry-wait=10 -d "D:\datasets\cache" -o train_256_places365standard.tar http://data.csail.mit.edu/places/places365/train_256_places365standard.tar
certutil -hashfile "D:\datasets\cache\filelist_places365-standard.tar" MD5     # 35a0585fee1fa656440f3ab298f8479c
certutil -hashfile "D:\datasets\cache\train_256_places365standard.tar" MD5     # 53ca1c756c3d1e7809517cc47c5561c5
```
If a previous extraction was interrupted, delete `D:\datasets\cache\data_256_standard` (compat.py also detects it).
With the correct tars present, the next run skips the download and only extracts.

## RAM, honestly

`compat.py` removes the 2× spike in #4, not the underlying requirement. LF-CBM holds whole-train-set arrays at once; from
the code and dataset sizes, the baseline needs roughly 33 GB (Places365) / 35 GB (ImageNet) for `target_features` plus the CLIP
concept scores alone, before overhead. A 16 GB laptop needs a very large page file on a fast drive (slow) or cannot finish.

## Say it in the report

Under "deviations from the original setup": (a) download folder changed to `D:/datasets/cache`;
(b) runtime compatibility layer — Windows `pin_memory`/`num_workers` settings, memory-efficient tensor assembly
(identical saved files), Places365 integrity check — none of which changes the model, hyper-parameters or metrics.
