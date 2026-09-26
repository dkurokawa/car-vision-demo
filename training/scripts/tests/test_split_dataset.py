from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest
import split_dataset as sd
from classes import CLASS_NAMES


def _make_class_dir(root: Path, cls: str, filenames: list[str]) -> None:
    d = root / cls
    d.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        (d / name).write_bytes(b"fake")


def _make_all_four_classes(root: Path, counts: dict[str, int] | None = None) -> None:
    """全 4 クラス分のフォルダを用意する (デフォルトはクラスごと 3 枚)."""
    counts = counts or dict.fromkeys(CLASS_NAMES, 3)
    for cls in CLASS_NAMES:
        n = counts[cls]
        _make_class_dir(root, cls, [f"{cls}_{i}.jpg" for i in range(n)])


def _write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "class"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _seed_split_dir(split_dir: Path, fingerprint: str, contents: dict[str, list[str]]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for name in sd.SPLIT_NAMES:
        lines = contents.get(name, [])
        (split_dir / f"{name}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
    (split_dir / sd.FINGERPRINT_FILENAME).write_text(fingerprint + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# split_files
# ---------------------------------------------------------------------------


def test_split_is_deterministic_for_the_same_seed() -> None:
    files = [f"img_{i:03d}.jpg" for i in range(37)]
    first = sd.split_files(files, seed=42)
    second = sd.split_files(files, seed=42)
    assert first == second


def test_split_does_not_depend_on_input_order() -> None:
    files = [f"img_{i:03d}.jpg" for i in range(37)]
    reversed_files = list(reversed(files))
    assert sd.split_files(files, seed=42) == sd.split_files(reversed_files, seed=42)


def test_different_seeds_can_produce_different_splits() -> None:
    files = [f"img_{i:03d}.jpg" for i in range(50)]
    a = sd.split_files(files, seed=1)
    b = sd.split_files(files, seed=2)
    assert a != b


def test_split_has_no_duplicates_and_covers_every_file() -> None:
    files = [f"img_{i:03d}.jpg" for i in range(41)]  # not evenly divisible
    result = sd.split_files(files, seed=7)

    all_assigned = result["train"] + result["val"] + result["test"]
    assert sorted(all_assigned) == sorted(files)
    assert len(set(all_assigned)) == len(files)
    assert len(all_assigned) == len(files)


def test_split_ratios_are_approximately_respected() -> None:
    files = [f"img_{i:04d}.jpg" for i in range(1000)]
    result = sd.split_files(files, seed=42, ratios=(0.70, 0.15, 0.15))
    assert len(result["train"]) == 700
    assert len(result["val"]) == 150
    assert len(result["test"]) == 150


def test_split_files_rejects_ratios_not_summing_to_one() -> None:
    with pytest.raises(ValueError):
        sd.split_files(["a.jpg", "b.jpg", "c.jpg"], seed=1, ratios=(0.5, 0.3, 0.1))


def test_split_files_rejects_fewer_than_three_files() -> None:
    with pytest.raises(ValueError):
        sd.split_files(["a.jpg", "b.jpg"], seed=1)


def test_split_files_with_exactly_three_files_gives_one_each() -> None:
    result = sd.split_files(["a.jpg", "b.jpg", "c.jpg"], seed=1)
    assert len(result["train"]) == 1
    assert len(result["val"]) == 1
    assert len(result["test"]) == 1


@pytest.mark.parametrize("n", range(sd.MIN_IMAGES_PER_CLASS, 12))
def test_split_files_always_guarantees_val_and_test_floor(n: int) -> None:
    files = [f"img_{i:03d}.jpg" for i in range(n)]
    result = sd.split_files(files, seed=42)
    assert len(result["val"]) >= 1
    assert len(result["test"]) >= 1
    assert len(result["train"]) + len(result["val"]) + len(result["test"]) == n


# ---------------------------------------------------------------------------
# images_by_class (labels come from the current folder, not the manifest;
# tested in isolation here without the fixed-4-class gate, which lives in
# validate_class_directories() and is exercised separately below)
# ---------------------------------------------------------------------------


def test_images_by_class_groups_by_folder_name(tmp_path: Path) -> None:
    _make_class_dir(tmp_path, "left_front", ["a.jpg", "b.jpg"])
    _make_class_dir(tmp_path, "right_rear", ["c.jpg"])

    by_class = sd.images_by_class(tmp_path)

    assert by_class == {"left_front": ["a.jpg", "b.jpg"], "right_rear": ["c.jpg"]}


def test_images_by_class_ignores_underscore_prefixed_dirs(tmp_path: Path) -> None:
    _make_class_dir(tmp_path, "left_front", ["a.jpg"])
    _make_class_dir(tmp_path, "_unsorted/left_front", ["staged.jpg"])
    _make_class_dir(tmp_path, "_external/epfl", ["ext.jpg"])

    by_class = sd.images_by_class(tmp_path)

    assert by_class == {"left_front": ["a.jpg"]}


def test_images_by_class_ignores_non_image_files_and_empty_dirs(tmp_path: Path) -> None:
    (tmp_path / "left_front").mkdir()
    (tmp_path / "left_front" / "notes.txt").write_bytes(b"not an image")
    (tmp_path / "empty_class").mkdir()

    assert sd.images_by_class(tmp_path) == {}


def test_images_by_class_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    assert sd.images_by_class(tmp_path / "does-not-exist") == {}


def test_images_by_class_uses_current_folder_even_if_manifest_disagrees(tmp_path: Path) -> None:
    # Manifest says "x.jpg" belongs to left_front, but the file now actually
    # sits under right_front/ (manually re-sorted). The folder wins.
    _make_class_dir(tmp_path, "right_front", ["x.jpg"])
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [{"file": "x.jpg", "class": "left_front"}])

    by_class = sd.images_by_class(tmp_path)

    assert by_class == {"right_front": ["x.jpg"]}
    assert "left_front" not in by_class


# ---------------------------------------------------------------------------
# validate_class_directories (G1: fixed 4-class set)
# ---------------------------------------------------------------------------


def test_validate_class_directories_passes_when_all_four_present(tmp_path: Path) -> None:
    _make_all_four_classes(tmp_path)
    sd.validate_class_directories(tmp_path)  # must not raise


def test_validate_class_directories_ignores_underscore_prefixed_dirs(tmp_path: Path) -> None:
    _make_all_four_classes(tmp_path)
    _make_class_dir(tmp_path, "_unsorted/left_front", ["staged.jpg"])
    sd.validate_class_directories(tmp_path)  # must not raise


def test_validate_class_directories_raises_for_unknown_folder(tmp_path: Path) -> None:
    _make_all_four_classes(tmp_path)
    _make_class_dir(tmp_path, "top_down", ["a.jpg"])  # not one of the 4 fixed classes

    with pytest.raises(ValueError) as exc_info:
        sd.validate_class_directories(tmp_path)
    assert "top_down" in str(exc_info.value)


def test_validate_class_directories_raises_for_missing_class_folder(tmp_path: Path) -> None:
    counts = dict.fromkeys(CLASS_NAMES, 3)
    _make_all_four_classes(tmp_path, counts)
    shutil.rmtree(tmp_path / "right_rear")

    with pytest.raises(ValueError) as exc_info:
        sd.validate_class_directories(tmp_path)
    assert "right_rear" in str(exc_info.value)


def test_validate_class_directories_raises_for_empty_class_folder(tmp_path: Path) -> None:
    _make_all_four_classes(tmp_path)
    (tmp_path / "right_rear").mkdir(exist_ok=True)
    for p in (tmp_path / "right_rear").iterdir():
        p.unlink()

    with pytest.raises(ValueError) as exc_info:
        sd.validate_class_directories(tmp_path)
    assert "right_rear" in str(exc_info.value)


def test_validate_class_directories_raises_when_data_dir_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        sd.validate_class_directories(tmp_path / "does-not-exist")


# ---------------------------------------------------------------------------
# warn_missing_manifest_entries
# ---------------------------------------------------------------------------


def test_warn_missing_manifest_entries_reports_per_class_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    by_class = {"left_front": ["a.jpg", "b.jpg"], "right_rear": ["c.jpg"]}
    manifest_filenames = {"a.jpg"}  # b.jpg and c.jpg have no manifest row

    sd.warn_missing_manifest_entries(by_class, manifest_filenames)

    out = capsys.readouterr().out
    assert "left_front" in out and "1 枚" in out
    assert "right_rear" in out


def test_warn_missing_manifest_entries_silent_when_everything_recorded(
    capsys: pytest.CaptureFixture[str],
) -> None:
    by_class = {"left_front": ["a.jpg"]}
    sd.warn_missing_manifest_entries(by_class, {"a.jpg"})
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# validate_class_sizes
# ---------------------------------------------------------------------------


def test_validate_class_sizes_passes_when_all_classes_have_enough_images() -> None:
    sd.validate_class_sizes({"left_front": ["a", "b", "c"], "right_rear": ["d", "e", "f", "g"]})


def test_validate_class_sizes_raises_listing_all_undersized_classes() -> None:
    by_class = {
        "left_front": ["a", "b", "c"],
        "right_front": ["d", "e"],  # too small
        "right_rear": ["f"],  # too small
    }
    with pytest.raises(ValueError) as exc_info:
        sd.validate_class_sizes(by_class)
    message = str(exc_info.value)
    assert "right_front" in message
    assert "right_rear" in message
    assert "left_front" not in message


# ---------------------------------------------------------------------------
# compute_fingerprint (G2: seed and ratios are part of the fingerprint)
# ---------------------------------------------------------------------------


def test_compute_fingerprint_is_stable_regardless_of_dict_or_list_order() -> None:
    a = sd.compute_fingerprint({"left_front": ["b.jpg", "a.jpg"], "right_rear": ["c.jpg"]}, seed=1)
    b = sd.compute_fingerprint({"right_rear": ["c.jpg"], "left_front": ["a.jpg", "b.jpg"]}, seed=1)
    assert a == b


def test_compute_fingerprint_changes_when_contents_change() -> None:
    a = sd.compute_fingerprint({"left_front": ["a.jpg"]}, seed=1)
    b = sd.compute_fingerprint({"left_front": ["a.jpg", "b.jpg"]}, seed=1)
    assert a != b


def test_compute_fingerprint_changes_when_seed_changes() -> None:
    by_class = {"left_front": ["a.jpg"]}
    a = sd.compute_fingerprint(by_class, seed=1)
    b = sd.compute_fingerprint(by_class, seed=2)
    assert a != b


def test_compute_fingerprint_changes_when_ratios_change() -> None:
    by_class = {"left_front": ["a.jpg"]}
    a = sd.compute_fingerprint(by_class, seed=1, ratios=(0.70, 0.15, 0.15))
    b = sd.compute_fingerprint(by_class, seed=1, ratios=(0.60, 0.20, 0.20))
    assert a != b


# ---------------------------------------------------------------------------
# generate_or_load_splits (integration: fingerprint reuse/invalidation
# including seed/ratios, integrity check on reuse, folder-based relabeling,
# undersized-class errors, fixed 4-class enforcement)
# ---------------------------------------------------------------------------


def test_reuses_split_when_content_is_valid_and_fingerprint_matches(tmp_path: Path) -> None:
    data_dir = tmp_path / "exterior"
    _make_all_four_classes(
        data_dir, {"left_front": 5, "left_rear": 3, "right_front": 3, "right_rear": 3}
    )
    manifest_path = data_dir / "manifest.csv"
    split_dir = tmp_path / "splits" / "1"

    fresh = sd.generate_or_load_splits(data_dir, manifest_path, split_dir, seed=1)

    # Perturb a *valid* copy of the genuinely-computed result: swap which
    # specific left_front file is in train vs. val. This keeps full
    # coverage/no-dup/no-unknown and each class's val/test >= 1 all true (so
    # it passes the G3 integrity check), but is a different assignment than
    # what split_files() would recompute -- proving reuse, not recompute
    # (a recompute would deterministically give back `fresh`).
    perturbed = {name: list(paths) for name, paths in fresh.items()}
    train_lf = [p for p in perturbed["train"] if p.startswith("left_front/")]
    val_lf = [p for p in perturbed["val"] if p.startswith("left_front/")]
    assert train_lf and val_lf, "left_front should have entries in both train and val"
    swap_out, swap_in = train_lf[0], val_lf[0]
    perturbed["train"] = [swap_in if p == swap_out else p for p in perturbed["train"]]
    perturbed["val"] = [swap_out if p == swap_in else p for p in perturbed["val"]]

    by_class = sd.images_by_class(data_dir)
    fingerprint = sd.compute_fingerprint(by_class, seed=1)
    _seed_split_dir(split_dir, fingerprint, perturbed)

    result = sd.generate_or_load_splits(data_dir, manifest_path, split_dir, seed=1)

    assert result == perturbed
    assert result != fresh


def test_regenerates_when_loaded_split_is_incomplete_even_if_fingerprint_matches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # This replaces the old "reuses a 3-line stale split" case: once the
    # integrity check (G3) exists, content that covers only 3 of the 13
    # target images must be rejected and regenerated even though its
    # (forged) fingerprint matches.
    data_dir = tmp_path / "exterior"
    _make_all_four_classes(
        data_dir, {"left_front": 4, "left_rear": 3, "right_front": 3, "right_rear": 3}
    )
    manifest_path = data_dir / "manifest.csv"
    split_dir = tmp_path / "splits" / "1"

    by_class = sd.images_by_class(data_dir)
    fingerprint = sd.compute_fingerprint(by_class, seed=1)
    incomplete = {
        "train": ["left_front/left_front_0.jpg"],
        "val": ["left_front/left_front_1.jpg"],
        "test": ["left_front/left_front_2.jpg"],
    }
    _seed_split_dir(split_dir, fingerprint, incomplete)

    result = sd.generate_or_load_splits(data_dir, manifest_path, split_dir, seed=1)

    assert result != incomplete
    all_assigned = result["train"] + result["val"] + result["test"]
    assert len(all_assigned) == 13
    assert len(set(all_assigned)) == 13
    assert "整合していません" in capsys.readouterr().out


def test_regenerates_when_loaded_split_has_a_duplicate_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "exterior"
    _make_all_four_classes(data_dir)
    manifest_path = data_dir / "manifest.csv"
    split_dir = tmp_path / "splits" / "1"

    fresh = sd.generate_or_load_splits(data_dir, manifest_path, split_dir, seed=1)
    by_class = sd.images_by_class(data_dir)
    fingerprint = sd.compute_fingerprint(by_class, seed=1)

    # Duplicate one train path into val too (same fingerprint, corrupted content).
    duplicated = {name: list(paths) for name, paths in fresh.items()}
    duplicated["val"] = [*duplicated["val"], duplicated["train"][0]]
    _seed_split_dir(split_dir, fingerprint, duplicated)

    result = sd.generate_or_load_splits(data_dir, manifest_path, split_dir, seed=1)

    assert result == fresh  # regenerated back to the real deterministic split
    assert "整合していません" in capsys.readouterr().out


def test_regenerates_when_fingerprint_does_not_match(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "exterior"
    _make_all_four_classes(
        data_dir, {"left_front": 4, "left_rear": 3, "right_front": 3, "right_rear": 3}
    )
    manifest_path = data_dir / "manifest.csv"

    stale = {
        "train": ["left_front/left_front_0.jpg"],
        "val": ["left_front/left_front_1.jpg"],
        "test": ["left_front/left_front_2.jpg"],
    }
    split_dir = tmp_path / "splits" / "42"
    _seed_split_dir(split_dir, "stale-fingerprint-that-does-not-match", stale)

    result = sd.generate_or_load_splits(data_dir, manifest_path, split_dir, seed=42)

    assert result != stale
    all_assigned = result["train"] + result["val"] + result["test"]
    assert len(all_assigned) == 13
    assert "不一致" in capsys.readouterr().out

    by_class = sd.images_by_class(data_dir)
    assert (split_dir / sd.FINGERPRINT_FILENAME).read_text(encoding="utf-8").strip() == (
        sd.compute_fingerprint(by_class, seed=42)
    )


def test_regenerates_when_seed_changes_even_with_the_same_images(tmp_path: Path) -> None:
    data_dir = tmp_path / "exterior"
    _make_all_four_classes(data_dir, dict.fromkeys(CLASS_NAMES, 10))
    manifest_path = data_dir / "manifest.csv"
    split_dir = tmp_path / "splits" / "shared"

    result_seed1 = sd.generate_or_load_splits(data_dir, manifest_path, split_dir, seed=1)
    result_seed2 = sd.generate_or_load_splits(data_dir, manifest_path, split_dir, seed=2)

    # Without seed in the fingerprint, the second call would wrongly reuse
    # the seed=1 result. With it, the image set being identical must not be
    # enough to reuse across different seeds.
    assert result_seed1 != result_seed2


def test_regenerates_when_ratios_change_even_with_the_same_images_and_seed(tmp_path: Path) -> None:
    data_dir = tmp_path / "exterior"
    _make_all_four_classes(data_dir, dict.fromkeys(CLASS_NAMES, 20))
    manifest_path = data_dir / "manifest.csv"
    split_dir = tmp_path / "splits" / "shared"

    result_a = sd.generate_or_load_splits(
        data_dir, manifest_path, split_dir, seed=1, ratios=(0.70, 0.15, 0.15)
    )
    result_b = sd.generate_or_load_splits(
        data_dir, manifest_path, split_dir, seed=1, ratios=(0.50, 0.25, 0.25)
    )

    assert len(result_a["train"]) != len(result_b["train"])


def test_force_regenerates_even_when_fingerprint_matches(tmp_path: Path) -> None:
    data_dir = tmp_path / "exterior"
    _make_all_four_classes(
        data_dir, {"left_front": 4, "left_rear": 3, "right_front": 3, "right_rear": 3}
    )
    manifest_path = data_dir / "manifest.csv"
    split_dir = tmp_path / "splits" / "1"

    by_class = sd.images_by_class(data_dir)
    fingerprint = sd.compute_fingerprint(by_class, seed=1)  # matches current data...

    # ...but the seeded split assignment itself is not what split_files()
    # would actually produce, so --force must recompute it instead of
    # trusting the (fingerprint-valid but stale) file on disk.
    stale = {
        "train": ["left_front/left_front_0.jpg"],
        "val": ["left_front/left_front_1.jpg"],
        "test": ["left_front/left_front_2.jpg"],
    }
    _seed_split_dir(split_dir, fingerprint, stale)

    result = sd.generate_or_load_splits(data_dir, manifest_path, split_dir, seed=1, force=True)

    assert result != stale
    all_assigned = result["train"] + result["val"] + result["test"]
    assert len(all_assigned) == 13


def test_relabeled_image_ends_up_under_its_new_class(tmp_path: Path) -> None:
    data_dir = tmp_path / "exterior"
    _make_all_four_classes(data_dir)  # 3 each, satisfies MIN_IMAGES_PER_CLASS
    # x.jpg is manually moved into right_front/ on top of the 3 auto-made files.
    (data_dir / "right_front" / "x.jpg").write_bytes(b"fake")
    manifest_path = data_dir / "manifest.csv"
    # manifest still says x.jpg is left_front (stale provenance record from
    # when it was first collected/sorted); the file has since been moved.
    _write_manifest(manifest_path, [{"file": "x.jpg", "class": "left_front"}])

    result = sd.generate_or_load_splits(data_dir, manifest_path, tmp_path / "splits" / "1", seed=1)

    all_assigned = result["train"] + result["val"] + result["test"]
    assert "right_front/x.jpg" in all_assigned
    assert "left_front/x.jpg" not in all_assigned


def test_images_without_manifest_entries_are_still_included(tmp_path: Path) -> None:
    data_dir = tmp_path / "exterior"
    _make_all_four_classes(data_dir)
    (data_dir / "left_front" / "manual.jpg").write_bytes(b"fake")
    manifest_path = data_dir / "manifest.csv"
    # None of left_front's files were ever recorded in the manifest.
    _write_manifest(manifest_path, [])

    result = sd.generate_or_load_splits(data_dir, manifest_path, tmp_path / "splits" / "1", seed=1)

    all_assigned = result["train"] + result["val"] + result["test"]
    assert "left_front/manual.jpg" in all_assigned


def test_generate_or_load_splits_raises_for_undersized_class(tmp_path: Path) -> None:
    data_dir = tmp_path / "exterior"
    _make_all_four_classes(data_dir)
    # Shrink left_front down to 2 -- below MIN_IMAGES_PER_CLASS -- while the
    # other three classes stay valid, so this exercises validate_class_sizes()
    # rather than validate_class_directories() (which only cares about >=1).
    shutil.rmtree(data_dir / "left_front")
    _make_class_dir(data_dir, "left_front", ["a.jpg", "b.jpg"])
    manifest_path = data_dir / "manifest.csv"

    with pytest.raises(ValueError) as exc_info:
        sd.generate_or_load_splits(data_dir, manifest_path, tmp_path / "splits" / "1", seed=1)
    assert "left_front" in str(exc_info.value)


def test_generate_or_load_splits_raises_when_no_images_at_all(tmp_path: Path) -> None:
    data_dir = tmp_path / "exterior"
    data_dir.mkdir()
    manifest_path = data_dir / "manifest.csv"

    with pytest.raises(ValueError):
        sd.generate_or_load_splits(data_dir, manifest_path, tmp_path / "splits" / "1", seed=1)

    # No split files should have been written for an all-empty target set.
    assert not (tmp_path / "splits" / "1").exists()


def test_generate_or_load_splits_raises_for_unknown_class_folder(tmp_path: Path) -> None:
    data_dir = tmp_path / "exterior"
    _make_all_four_classes(data_dir)
    _make_class_dir(data_dir, "top_down", ["a.jpg", "b.jpg", "c.jpg"])
    manifest_path = data_dir / "manifest.csv"

    with pytest.raises(ValueError) as exc_info:
        sd.generate_or_load_splits(data_dir, manifest_path, tmp_path / "splits" / "1", seed=1)
    assert "top_down" in str(exc_info.value)


# ---------------------------------------------------------------------------
# read_manifest (unchanged: still just parses file/class columns)
# ---------------------------------------------------------------------------


def test_read_manifest_parses_file_and_class_columns(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "class",
                "source",
                "source_url",
                "author",
                "license",
                "downloaded_at",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file": "a.jpg",
                "class": "left_front",
                "source": "pexels",
                "source_url": "https://example.com/a",
                "author": "someone",
                "license": "Pexels License",
                "downloaded_at": "2026-01-01T00:00:00+00:00",
            }
        )

    entries = sd.read_manifest(manifest_path)

    assert entries == [sd.ManifestEntry(file="a.jpg", cls="left_front")]


def test_read_manifest_returns_empty_list_when_missing(tmp_path: Path) -> None:
    assert sd.read_manifest(tmp_path / "does-not-exist.csv") == []


# ---------------------------------------------------------------------------
# main() CLI surface
# ---------------------------------------------------------------------------


def test_main_exits_with_error_status_when_no_images(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "exterior"
    data_dir.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        sd.main(["--data-dir", str(data_dir), "--split-dir", str(tmp_path / "splits" / "1")])

    assert exc_info.value.code == 1
    assert "left_front" in capsys.readouterr().err
