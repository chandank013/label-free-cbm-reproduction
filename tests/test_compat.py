"""
Tests for compat.py using stand-ins for torch and the authors' utils module.
They check OUR logic (ordering, scoping of the eval'd hook, kwargs rewriting,
buffer consumption). They cannot check real-torch behaviour -- run the real
pipeline on CIFAR-10 for that (docs/RUN_STEPS.md).

    python tests/test_compat.py
"""
import os
import sys
import types

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# ---- minimal stand-ins -------------------------------------------------------
SAVED = {}
fake_torch = types.ModuleType("torch")
fake_torch.empty = lambda shape, dtype=None: np.empty(shape, dtype=dtype)
fake_torch.save = lambda obj, path: SAVED.__setitem__(path, np.array(obj))
fake_torch.no_grad = lambda: types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s, *a: False)
class _NG:                                                   # context manager
    def __enter__(self): return self
    def __exit__(self, *a): return False
fake_torch.no_grad = lambda: _NG()
fake_torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
sys.modules["torch"] = fake_torch
if "tqdm" not in sys.modules:
    try:
        import tqdm  # noqa: F401
    except ImportError:
        m = types.ModuleType("tqdm"); m.tqdm = lambda it, *a, **k: it; sys.modules["tqdm"] = m

import compat  # noqa: E402


class Arr(np.ndarray):                                       # ndarray with .to()/.cpu() like a tensor
    def to(self, device): return self
    def cpu(self): return self
    def detach(self): return self
def A(x): return np.asarray(x).view(Arr)


def test_cat_inplace_matches_concatenate_and_consumes_list():
    chunks = [A(np.random.RandomState(i).rand(n, 4).astype(np.float32)) for i, n in enumerate([3, 5, 1, 7])]
    expected = np.concatenate([np.asarray(c) for c in chunks])
    out = compat.cat_inplace(chunks)
    assert out.dtype == np.float32 and out.shape == (16, 4)
    assert np.array_equal(out, expected), "row order must be preserved"
    assert chunks == [], "chunks must be freed as they are consumed"


def test_safe_loader_only_rewrites_kwargs_on_windows():
    seen = {}
    real = lambda *a, **kw: seen.update(args=a, kw=kw)
    for win in (True, False):
        compat.IS_WINDOWS = win
        compat.make_safe_loader(real)("ds", 64, num_workers=8, pin_memory=True)
        want = (0, False) if win else (8, True)
        assert (seen["kw"]["num_workers"], seen["kw"]["pin_memory"]) == want
        assert seen["args"] == ("ds", 64)


def _fake_utils(tmpdir):
    u = types.SimpleNamespace()
    u._make_save_dir = lambda name: None
    u._all_saved = lambda names: any(os.path.exists(p) for p in names.values())
    def get_activation(outputs, pool_mode):
        def hook(model, inp, out): outputs.append(A(out.mean(axis=(2, 3))))
        return hook
    u.get_activation = get_activation
    class DL:
        def __init__(self, ds, bs, **kw): self.ds, self.bs, self.kw = ds, bs, kw
        def __iter__(self):
            for i in range(0, len(self.ds), self.bs): yield A(self.ds[i:i + self.bs]), None
    u.DataLoader = DL
    return u


class _Layer:
    def __init__(self): self.hooks = []
    def register_forward_hook(self, fn):
        self.hooks.append(fn); return types.SimpleNamespace(remove=lambda: self.hooks.remove(fn))
class _Model:                                               # model with a dotted layer path like 'features.final_pool'
    def __init__(self):
        self.layer4 = _Layer(); self.features = types.SimpleNamespace(final_pool=_Layer())
    def __call__(self, x):
        for L in (self.layer4, self.features.final_pool):
            for h in list(L.hooks): h(self, x, np.asarray(x))


def test_saved_target_activations_equal_authors_semantics_incl_dotted_layer():
    u = _fake_utils(None)
    save_target, save_clip = compat.make_memory_efficient_saves(u)
    data = np.random.RandomState(0).rand(23, 6, 3, 3).astype(np.float32)       # 23 images, batch 5 -> uneven last batch
    for layer in ("layer4", "features.final_pool"):
        SAVED.clear()
        save_target(_Model(), data, "mem/{}.pt", target_layers=[layer], batch_size=5, device="cpu")
        (path, arr), = SAVED.items()
        assert path == f"mem/{layer}.pt"
        assert np.allclose(arr, data.mean(axis=(2, 3)), atol=1e-6), f"{layer}: wrong activations"
    # hooks must be removed afterwards (authors' code removes them)
    m = _Model(); save_target(m, data, "x/{}.pt", target_layers=["layer4"], batch_size=5, device="cpu")
    assert m.layer4.hooks == []


def test_save_clip_image_features_roundtrip_and_skip_if_exists(tmp_path=None):
    import tempfile
    d = tempfile.mkdtemp()
    u = _fake_utils(d)
    _, save_clip = compat.make_memory_efficient_saves(u)
    class M:                                                  # encode_image like CLIP
        def encode_image(self, x): return A(np.asarray(x).reshape(len(x), -1)[:, :4])
    data = np.random.RandomState(1).rand(11, 2, 2, 2).astype(np.float32)
    path = os.path.join(d, "clip.pt").replace("\\", "/")
    SAVED.clear(); save_clip(M(), data, path, batch_size=4, device="cpu")
    assert np.allclose(SAVED[path], data.reshape(11, -1)[:, :4])
    open(path, "w").write("x")                                # file exists -> must return without recomputing
    SAVED.clear(); save_clip(M(), data, path, batch_size=4, device="cpu")
    assert SAVED == {}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print("PASS ", t.__name__)
    print(f"\n{len(tests)}/{len(tests)} passed (stand-in torch: logic only)")
