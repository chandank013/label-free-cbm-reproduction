"""
Builds phase2/train_cbm_filtered.py from the authors' UNMODIFIED
original_lfcbm/train_cbm.py by inserting four small blocks at exact anchor
strings. Each anchor must occur exactly once, otherwise this script stops --
so a changed upstream file can never be patched silently or wrongly.

Why generate instead of hand-copying: the authors' file stays the single
source of truth, the diff is tiny and auditable
(`diff original_lfcbm/train_cbm.py phase2/train_cbm_filtered.py`), and
filter_mode='all' reproduces the original pipeline exactly, so the baseline
and the improved model run through the same code.

    python phase2/make_filtered_trainer.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "original_lfcbm", "train_cbm.py")
DST = os.path.join(HERE, "train_cbm_filtered.py")

HEADER = '''# ======================================================================
# GENERATED FILE -- do not edit by hand. Re-create with:
#     python phase2/make_filtered_trainer.py
# Derived from original_lfcbm/train_cbm.py (Oikarinen et al., ICLR 2023,
# https://github.com/Trustworthy-ML-Lab/Label-free-CBM). Everything outside
# the blocks marked "PHASE 2" is the authors' code, unchanged.
# ======================================================================
'''

IMPORTS = '''import torch
import numpy as np
import csv
import concept_quality as cq
'''

# --- block A: after the authors' clip_cutoff filter, before the projection layer is learned
BLOCK_A = '''
    # ===================== PHASE 2: concept quality filter (inserted) =====================
    # Runs on the concepts that survived the authors' clip_cutoff, using TRAIN data only.
    orig_idx = np.where((highest > args.clip_cutoff).numpy())[0]            # position in the original concept file
    _keep, phase2_diag = cq.run_filter(
        getattr(args, "filter_mode", "all"),
        clip_features.numpy(), np.asarray(data_utils.get_targets_only(d_train)), concepts, len(classes),
        keep_frac=getattr(args, "keep_frac", 1.0), seed=getattr(args, "seed", 0),
        w_rel=getattr(args, "w_rel", 1.0), w_red=getattr(args, "w_red", 1.0),
        agg=getattr(args, "agg", "max"), center_rows=getattr(args, "center_rows", True))
    if not _keep.all():
        _keep_t = torch.from_numpy(_keep)
        concepts = [c for c, k_ in zip(concepts, _keep) if k_]
        clip_features = clip_features[:, _keep_t]
        val_clip_features = val_clip_features[:, _keep_t]
        orig_idx = orig_idx[_keep]
    print("[phase2] filter_mode={} kept {}/{} concepts".format(
        phase2_diag["filter_mode"], len(concepts), phase2_diag["n_before_filter"]))
    # ======================================================================================
'''

# --- block B: final concept set is known (after the interpretability cutoff); measure it
BLOCK_B = '''    # ===================== PHASE 2: measure the FINAL concept set (inserted) =====================
    with torch.no_grad():
        _interp = interpretable.cpu()
        phase2_Xv = val_clip_features[:, _interp].numpy()
        phase2_yv = np.asarray(data_utils.get_targets_only(d_val))
        phase2_text = None
        try:
            _t = torch.load(text_save_name, map_location="cpu").float()[torch.from_numpy(orig_idx[_interp.numpy()])]
            phase2_text = (_t / torch.norm(_t, dim=1, keepdim=True)).numpy()
        except Exception as e:                      # text-space metrics are optional
            print("[phase2] text-space metrics skipped:", e)
    # ==============================================================================================
    del clip_features, val_clip_features
'''

# --- block C: end of train_cbm_and_save, after the model + metrics are written
BLOCK_C = '''
    # ===================== PHASE 2: persist the report (inserted) =====================
    phase2_summary, phase2_rows = cq.final_report(
        phase2_Xv, phase2_yv, len(classes), concepts, W_g.numpy(), text_emb=phase2_text,
        center_rows=getattr(args, "center_rows", True))
    phase2_summary.update(
        acc_val=float(output_proj['path'][0]['metrics']['acc_val']),
        n_classes=len(classes), seed=getattr(args, "seed", 0), dataset=args.dataset,
        filter_mode=phase2_diag["filter_mode"], keep_frac=phase2_diag["keep_frac"],
        n_before_filter=phase2_diag["n_before_filter"], n_after_filter=phase2_diag["n_after_filter"],
        w_rel=getattr(args, "w_rel", 1.0), w_red=getattr(args, "w_red", 1.0), agg=getattr(args, "agg", "max"))
    with open(os.path.join(save_name, "phase2_report.json"), "w") as f:
        json.dump({"summary": phase2_summary, "removed_by_filter": phase2_diag["removed"]}, f, indent=2)
    with open(os.path.join(save_name, "concept_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(phase2_rows[0].keys()))
        w.writeheader()
        w.writerows(phase2_rows)
    print("[phase2] acc_val={:.4f}  concepts={}  NEC={:.2f}".format(
        phase2_summary["acc_val"], phase2_summary["n_concepts"], phase2_summary["NEC"]))
    # ==================================================================================

if __name__=='__main__':'''


def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        sys.exit(f"ANCHOR PROBLEM [{label}]: expected exactly 1 occurrence, found {n}. "
                 f"The upstream train_cbm.py changed; update the anchors before generating.")
    return text.replace(old, new)


def main():
    src = open(SRC).read()
    out = src
    out = replace_once(out, "import torch\nimport os\n", IMPORTS + "import os\n", "imports")
    out = replace_once(
        out, "    val_clip_features = val_clip_features[:, highest>args.clip_cutoff]\n",
        "    val_clip_features = val_clip_features[:, highest>args.clip_cutoff]\n" + BLOCK_A, "A: after clip_cutoff")
    out = replace_once(out, "    del clip_features, val_clip_features\n", BLOCK_B, "B: before del features")
    out = replace_once(
        out, 'datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))',
        'datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + getattr(args, "tag", ""))', "D: unique save dir")
    out = replace_once(out, "\nif __name__=='__main__':", BLOCK_C, "C: end of function")
    open(DST, "w").write(HEADER + out)
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
