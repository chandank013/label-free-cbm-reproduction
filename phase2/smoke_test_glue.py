"""
Smoke test for the code blocks that make_filtered_trainer.py inserts into the
authors' train_cbm_and_save(). It executes those exact blocks (A, B, C) on tiny
synthetic data, so NameErrors / wrong variable order / bad shapes show up in
seconds instead of after a long run.

    python phase2/smoke_test_glue.py

If real PyTorch is installed (it is on your laptop), the blocks run against it.
If not (e.g. a minimal CI box), a small numpy stand-in for the handful of tensor
operations used is substituted -- that still catches scoping/ordering bugs but
does NOT test real-torch behaviour, and the script says so loudly.
"""
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import concept_quality as cq                       # noqa: E402
import make_filtered_trainer as mft                # noqa: E402

try:
    import torch
    USING = "REAL torch"
except ImportError:                                  # numpy stand-in (see docstring)
    USING = "NUMPY STAND-IN (real-torch behaviour NOT tested)"

    class T:
        def __init__(self, a): self.a = np.asarray(a)
        def numpy(self): return self.a
        def cpu(self): return self
        def float(self): return T(self.a.astype(np.float32))
        def abs(self): return T(np.abs(self.a))
        def sum(self, *a, **k): return T(self.a.sum(*a, **k))
        def item(self): return self.a.item()
        def __gt__(self, o): return T(self.a > o)
        def __truediv__(self, o): return T(self.a / (o.a if isinstance(o, T) else o))
        def __getitem__(self, i):
            if isinstance(i, tuple):
                i = tuple(x.a if isinstance(x, T) else x for x in i)
            else:
                i = i.a if isinstance(i, T) else i
            return T(self.a[i])
        @property
        def shape(self): return self.a.shape

    class _NoGrad:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    torch = types.SimpleNamespace(
        from_numpy=lambda a: T(a), no_grad=lambda: _NoGrad(), tensor=lambda a: T(a),
        norm=lambda t, dim, keepdim: T(np.linalg.norm(t.a, axis=dim, keepdims=keepdim)),
        load=lambda path, map_location=None: T(np.load(path)),
    )
    # block code calls torch.from_numpy(bool_array) and indexes with it
    sys.modules["torch"] = torch


def wrap(a):
    return torch.from_numpy(a) if USING.startswith("NUMPY") else torch.from_numpy(a)


def run(mode, keep_frac):
    rs = np.random.RandomState(0)
    N, Nv, C, K0 = 600, 200, 5, 40
    y, yv = rs.randint(0, C, N), rs.randint(0, C, Nv)
    tmp = tempfile.mkdtemp()
    text_path = os.path.join(tmp, "text.npy")
    text_all = rs.normal(size=(K0, 8)); text_all /= np.linalg.norm(text_all, axis=1, keepdims=True)
    if USING.startswith("REAL"):
        torch.save(torch.from_numpy(text_all.astype(np.float32)), text_path)
    else:
        np.save(text_path, text_all)
        text_path += ".npy" if not text_path.endswith(".npy") else ""

    highest_np = np.linspace(0.20, 0.35, K0)                       # some concepts below the cutoff
    concepts = [f"concept {i}" for i in range(K0)]
    cutoff = 0.25
    survivors = [c for c, h in zip(concepts, highest_np) if h > cutoff]
    K1 = len(survivors)

    class A: pass
    args = A(); args.clip_cutoff = cutoff; args.dataset = "fake"; args.seed = 0
    args.filter_mode = mode; args.keep_frac = keep_frac

    class DU:
        @staticmethod
        def get_targets_only(name): return list(y if name == "train" else yv)

    ns = dict(
        torch=torch, np=np, os=os, json=__import__("json"), csv=__import__("csv"), cq=cq, args=args,
        highest=torch.from_numpy(highest_np.astype(np.float32)), data_utils=DU, d_train="train", d_val="val",
        concepts=survivors, classes=[f"c{i}" for i in range(C)], text_save_name=text_path,
        clip_features=torch.from_numpy(rs.normal(size=(N, K1)).astype(np.float32)),
        val_clip_features=torch.from_numpy(rs.normal(size=(Nv, K1)).astype(np.float32)),
        save_name=tmp,
    )
    # what the authors' code does between block A and block B: drop non-interpretable concepts
    middle = ("    _m = np.ones(len(concepts), dtype=bool); _m[::5] = False\n"
              "    interpretable = torch.from_numpy(_m)\n"
              "    concepts = [c for c, k_ in zip(concepts, _m) if k_]\n")
    # what the authors' code does before block C: fit the final layer
    after = ("    W_g = torch.from_numpy(np.random.RandomState(1).normal(size=(len(classes), len(concepts))).astype(np.float32) "
             "* (np.random.RandomState(2).rand(len(classes), len(concepts)) > 0.6))\n"
             "    output_proj = {'path': [{'metrics': {'acc_val': 0.75}}]}\n")
    block_c = mft.BLOCK_C.rsplit("if __name__", 1)[0]
    block_b = mft.BLOCK_B.replace("    del clip_features, val_clip_features\n", "")
    src = "def _run():\n" + mft.BLOCK_A + middle + block_b + after + block_c
    src += "    return phase2_summary, phase2_diag\n"
    src = "def _run():\n" + "\n".join(f"    {n} = _ns[{n!r}]" for n in ns) + "\n" + src.split("def _run():\n", 1)[1]
    glb = dict(_ns=ns)
    exec(src, glb)
    summary, diag = glb["_run"]()
    return summary, diag, K1


if __name__ == "__main__":
    print("backend:", USING, "\n")
    for mode, frac in [("all", 1.0), ("greedy", 0.5), ("clip_topk", 0.5), ("random", 0.5)]:
        s, d, K1 = run(mode, frac)
        assert d["n_before_filter"] == K1
        assert d["n_after_filter"] == (K1 if mode == "all" else max(1, round(frac * K1)))
        assert s["n_concepts"] <= d["n_after_filter"] and s["NEC"] >= 0 and 0 <= s["acc_val"] <= 1
        print(f"PASS {mode:10s} {K1} -> {d['n_after_filter']} after filter -> {s['n_concepts']} final | NEC={s['NEC']:.2f}")
    print("\nglue blocks execute end-to-end (scoping/order/shape OK for this backend)")
