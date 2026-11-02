import io
from pathlib import Path
import pytest
import torch
from PIL import Image

from webapp.registry import ModelInfo, ModelRegistry, sort_model_candidates
from webapp.results_reader import normalize_acc, ResultsReader
from webapp.inference import InferenceEngine, word_jaccard


def test_folder_name_parsing():
    # Tagged Phase 2 model
    path1 = Path("original_lfcbm/saved_models/cifar10_cbm_2026_11_01_15_53_31_greedy_k50_s0")
    info1 = ModelRegistry.parse_folder_name(path1)
    assert info1 is not None
    assert info1.dataset == "cifar10"
    assert info1.timestamp == "2026_11_01_15_53_31"
    assert info1.method == "greedy"
    assert info1.budget == 50
    assert info1.seed == 0
    assert info1.is_tagged is True

    # Tagged baseline model
    path2 = Path("original_lfcbm/saved_models/cub_cbm_2026_11_01_16_32_26_baseline_k100_s1")
    info2 = ModelRegistry.parse_folder_name(path2)
    assert info2 is not None
    assert info2.dataset == "cub"
    assert info2.method == "baseline"
    assert info2.budget == 100
    assert info2.seed == 1

    # Untagged Phase 1 model
    path3 = Path("original_lfcbm/saved_models/cifar100_cbm_2026_10_24_18_38")
    info3 = ModelRegistry.parse_folder_name(path3)
    assert info3 is not None
    assert info3.dataset == "cifar100"
    assert info3.timestamp == "2026_10_24_18_38"
    assert info3.method == "untagged"
    assert info3.is_tagged is False
    assert info3.label == "Phase 1 (untagged)"

    # Untagged author model
    path4 = Path("original_lfcbm/saved_models/cub_lf_cbm")
    info4 = ModelRegistry.parse_folder_name(path4)
    assert info4 is not None
    assert info4.dataset == "cub"
    assert info4.is_tagged is False


def test_newest_wins_and_lowest_seed():
    # Candidates with different seeds and timestamps
    c1 = ModelInfo("cifar10", "c1", Path("/tmp/c1"), "2026_11_01_10_00_00", "greedy", 50, 1, True, "m1")
    c2 = ModelInfo("cifar10", "c2", Path("/tmp/c2"), "2026_11_01_12_00_00", "greedy", 50, 0, True, "m2")
    c3 = ModelInfo("cifar10", "c3", Path("/tmp/c3"), "2026_11_02_15_00_00", "greedy", 50, 0, True, "m3")

    sorted_res = sort_model_candidates([c1, c2, c3])
    # Lowest seed is 0; between c2 and c3 (both seed 0), c3 has newer timestamp (2026_11_02 > 2026_11_01)
    assert sorted_res[0].folder_name == "c3"
    assert sorted_res[1].folder_name == "c2"
    assert sorted_res[2].folder_name == "c1"


def test_skipping_incomplete_folders(tmp_path):
    complete_dir = tmp_path / "cub_cbm_2026_11_01_12_00_00_baseline_k100_s0"
    complete_dir.mkdir()
    required = ["args.txt", "metrics.txt", "concepts.txt", "W_c.pt", "W_g.pt", "b_g.pt", "proj_mean.pt", "proj_std.pt"]
    for req in required:
        (complete_dir / req).write_text("sample content")

    incomplete_dir = tmp_path / "cub_cbm_2026_11_01_12_00_00_greedy_k50_s0"
    incomplete_dir.mkdir()
    # Write only some files
    (incomplete_dir / "args.txt").write_text("sample")
    (incomplete_dir / "W_c.pt").write_text("sample")

    assert ModelRegistry.is_folder_complete(complete_dir) is True
    assert ModelRegistry.is_folder_complete(incomplete_dir) is False

    reg = ModelRegistry(tmp_path)
    models = reg.get_all_models()
    assert len(models) == 1
    assert models[0].folder_name == complete_dir.name


def test_percent_fraction_normalisation():
    # Value in [0, 1.5] should be treated as fraction
    assert normalize_acc(0.864) == pytest.approx(0.864)
    assert normalize_acc(0.0) == 0.0
    assert normalize_acc(1.0) == 1.0

    # Value > 1.5 should be treated as percentage and converted to fraction
    assert normalize_acc(86.4) == pytest.approx(0.864)
    assert normalize_acc(65.27) == pytest.approx(0.6527)
    assert normalize_acc(100.0) == pytest.approx(1.0)

    # Edge cases
    assert normalize_acc(None) is None
    assert normalize_acc(-1) is None


def test_not_run_handling(tmp_path):
    reader = ResultsReader(tmp_path)
    kpis = reader.get_kpis("places365", 50)
    assert kpis["accuracy"]["status"] == "not_run"
    assert kpis["accuracy"]["original"] is None
    assert kpis["accuracy"]["badge_color"] == "grey"
    assert kpis["accuracy"]["badge_text"] == "Not run"

    redundancy = reader.get_redundancy_comparison(50)
    places = next(r for r in redundancy if r["dataset"] == "places365")
    assert places["status"] == "not_run"
    assert places["original"] is None


def test_reason_chip_rule():
    # Test duplicate chip vs low relevance chip rule:
    # "duplicate of '<nearest_kept>' (r=<value>)" in an amber chip when corr_with_nearest_kept >= 0.80,
    # otherwise a purple "low relevance (eta2=<value>)" chip.
    item_dup = {"concept": "a birdfeeder", "relevance": 0.52, "nearest_kept": "a bird feeder", "corr_with_nearest_kept": 0.98}
    item_rel = {"concept": "a cage", "relevance": 0.13, "nearest_kept": "a port", "corr_with_nearest_kept": 0.29}

    corr_dup = item_dup["corr_with_nearest_kept"]
    is_dup = corr_dup is not None and corr_dup >= 0.80
    assert is_dup is True
    reason_chip_dup = f"duplicate of '{item_dup['nearest_kept']}' (r={corr_dup:.2f})"
    assert "duplicate of 'a bird feeder' (r=0.98)" == reason_chip_dup

    corr_rel = item_rel["corr_with_nearest_kept"]
    is_rel_dup = corr_rel is not None and corr_rel >= 0.80
    assert is_rel_dup is False
    reason_chip_rel = f"low relevance (η²={item_rel['relevance']:.2f})"
    assert "low relevance (η²=0.13)" == reason_chip_rel


def test_delta_colour_and_seed_noise_rule(tmp_path):
    # Rule:
    # green only if change is in good direction, red if worse,
    # grey if |delta| <= seed-to-seed std of the original (label "within seed noise").
    # With one seed show "1 seed - no error bars".
    ds_dir = tmp_path / "results" / "phase2" / "cifar10"
    ds_dir.mkdir(parents=True)

    # 3 seeds for baseline with std = 0.010
    # accuracies: 0.850, 0.860, 0.870 -> mean = 0.860, std = 0.010
    import json
    for s, acc in enumerate([0.850, 0.860, 0.870]):
        run = {
            "summary": {
                "dataset": "cifar10", "method": "baseline", "seed": s, "keep_frac": 1.0,
                "acc_val": acc, "n_concepts": 100, "redundancy_nn_abs_corr": 0.5, "relevance_mean": 0.5
            }
        }
        (ds_dir / f"cifar10_baseline_k100_s{s}.json").write_text(json.dumps(run))

    # Case A: Filtered has delta = +0.005 (which is <= std 0.010) -> "within seed noise", grey
    for s in range(3):
        run = {
            "summary": {
                "dataset": "cifar10", "method": "greedy", "seed": s, "keep_frac": 0.5,
                "acc_val": 0.865, "n_concepts": 50, "redundancy_nn_abs_corr": 0.4, "relevance_mean": 0.6
            }
        }
        (ds_dir / f"cifar10_greedy_k50_s{s}.json").write_text(json.dumps(run))

    reader = ResultsReader(tmp_path)
    kpis = reader.get_kpis("cifar10", 50)
    assert kpis["accuracy"]["badge_color"] == "grey"
    assert kpis["accuracy"]["badge_text"] == "within seed noise"

    # Case B: Filtered has delta = +0.020 (> std 0.010) -> improvement, green
    for s in range(3):
        run = {
            "summary": {
                "dataset": "cifar10", "method": "greedy", "seed": s, "keep_frac": 0.5,
                "acc_val": 0.880, "n_concepts": 50, "redundancy_nn_abs_corr": 0.4, "relevance_mean": 0.6
            }
        }
        (ds_dir / f"cifar10_greedy_k50_s{s}.json").write_text(json.dumps(run))

    kpis2 = reader.get_kpis("cifar10", 50)
    assert kpis2["accuracy"]["badge_color"] == "green"
    assert "0.020" in kpis2["accuracy"]["badge_text"]


def test_contribution_math_synthetic():
    # Explanation = per-concept contribution to the predicted class (concept activation * final-layer weight)
    # Return top 6 positive contributors
    proj_c = torch.tensor([[2.0, -1.0, 3.0, 1.5, 0.5, -2.0, 4.0]])
    # W_g for 2 classes, 7 concepts
    W_g = torch.tensor([
        [1.0, 2.0, 0.5, -1.0, 3.0, 1.0, 2.0],  # class 0
        [-1.0, 1.0, 2.0, 3.0, 1.0, -1.0, 0.0],  # class 1
    ])
    # For class 0:
    # contributions = proj_c * W_g[0]:
    # c0: 2.0 * 1.0 = 2.0
    # c1: -1.0 * 2.0 = -2.0
    # c2: 3.0 * 0.5 = 1.5
    # c3: 1.5 * -1.0 = -1.5
    # c4: 0.5 * 3.0 = 1.5
    # c5: -2.0 * 1.0 = -2.0
    # c6: 4.0 * 2.0 = 8.0
    # Positive contributors:
    # c6 (8.0), c0 (2.0), c2 (1.5), c4 (1.5)
    contrib = proj_c[0] * W_g[0]
    pos_idx = torch.nonzero(contrib > 0).squeeze(-1)
    sorted_idx = pos_idx[torch.argsort(contrib[pos_idx], descending=True)]
    top_values = [contrib[i].item() for i in sorted_idx]
    assert top_values == [8.0, 2.0, 1.5, 1.5]


def test_upload_validation():
    engine = InferenceEngine(ModelRegistry(Path("/nonexistent")))

    # 1. Non-image file (random bytes)
    with pytest.raises(ValueError, match="Corrupt or unreadable"):
        engine.process_image(b"not an image file")

    # 2. Oversize (> 10MB)
    huge_bytes = b"0" * (10 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="exceeds 10 MB limit"):
        engine.process_image(huge_bytes)

    # 3. Valid image
    img = Image.new("RGB", (64, 64), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    processed = engine.process_image(buf.getvalue())
    assert processed.size == (64, 64)
    assert processed.mode == "RGB"

    # 4. EXIF Orientation
    # Create image with EXIF orientation 6 (90 CW)
    img_rot = Image.new("RGB", (100, 50), color="red")
    exif = img_rot.getexif()
    exif[0x0112] = 6  # Orientation tag = 6 (transposed)
    buf_rot = io.BytesIO()
    img_rot.save(buf_rot, format="JPEG", exif=exif)
    processed_rot = engine.process_image(buf_rot.getvalue())
    # Should be rotated to 50x100
    assert processed_rot.size == (50, 100)


def test_word_jaccard_duplicate_wording():
    # Similar wording test: Jaccard >= 0.5 after stop word removal
    c1 = "a red crest"
    c2 = "the red crest"
    assert word_jaccard(c1, c2) == 1.0  # both reduce to {"red", "crest"}

    c3 = "a pointed beak"
    c4 = "a sharp pointed beak"
    # c3: {"pointed", "beak"}, c4: {"sharp", "pointed", "beak"}
    # inter = 2, union = 3 -> 2/3 = 0.667 >= 0.5
    assert word_jaccard(c3, c4) >= 0.5

    c5 = "a black mask"
    c6 = "a yellow wing"
    assert word_jaccard(c5, c6) == 0.0
