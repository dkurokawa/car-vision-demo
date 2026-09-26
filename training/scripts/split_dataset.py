#!/usr/bin/env python3
"""車外観 4 角度クラス分類用データセットの train/val/test 分割.

## クラス集合

対象クラスは `classes.py` の `CLASS_NAMES` (left_front / left_rear /
right_front / right_rear) に固定する。`apps/inference-demo/src/classifier.ts`
の `CLASS_NAMES` と要素・順序が一致していなければならない (モデル出力の
インデックスに対応するため。一致は `tests/test_classes.py` が検証する)。

## ラベルの決め方 (クラス)

画像のクラスは manifest.csv の `class` 列ではなく、**画像がいま実際に置かれて
いる `<data-dir>/<class>/` というフォルダ名**で決める。手で振り分け直した
結果 (誤ラベルの修正・再分類) を正とするため。`_` で始まるディレクトリ
(`_unsorted/`, `_external/` など収集・展開の作業用) はクラスとして扱わない。

`_` で始まらないのに `CLASS_NAMES` に無いフォルダ (未知のクラス) がある場合、
または `CLASS_NAMES` のいずれかのフォルダが無い・空 (画像 0 枚) の場合は、
分割対象を黙って一部に絞らずエラー終了する
(validate_class_directories() 参照)。

`data/raw/exterior/manifest.csv`（`data/scripts/download_datasets.py` が書き
出す）は出典 (source/author/license/URL) の記録にのみ使う。manifest に行が
無い画像 (手で追加した画像・外部データセットなど) も分割対象に含めるが、
クラスごとに件数を警告として表示する。

## 再利用の条件 (fingerprint)

分割対象一覧 (`<class>/<file>` をソートしたもの) に加えて `--seed` と
`--ratios` も含めた sha256 を `training/splits/<seed>/fingerprint.txt` に
保存する (対象は変わらなくても seed/ratios を変えれば別の分割になるため)。
次回実行時は、現在の対象一覧・seed・ratios から計算した fingerprint が
保存済みのものと一致する場合だけ既存の分割を再利用し、一致しなければ
(画像が増減・再分類された、または seed/ratios を変えた場合) 警告を出して
作り直す。`--force` を付けると fingerprint に関わらず常に作り直す。
分割対象が 0 件の場合は空の分割ファイルを書き出さずにエラー終了する。

fingerprint が一致しても、読み込んだ 3 ファイルの中身自体が壊れていたり
手で書き換えられていたりする可能性はゼロではないため、再利用前に
「対象一覧の全パスがちょうど 1 回ずつ現れる (欠落・重複・未知のパスが無い)・
各クラスの val/test が最低 1 件ずつある」ことも検査する
(_loaded_split_is_consistent() 参照)。満たさなければ同様に警告して作り直す。

## 分割そのもの

分割はクラスごとに independent に行い、`--seed` を固定すれば同じ入力から常に
同じ分割になる（`random.Random(seed)` によるシャッフルで、入力の列挙順には
依存しない）。3 枚以上あるクラスには val / test に最低 1 枚ずつを割り振り、
残りを train にする (小さいクラスが val=0 や test=0 のまま学習に回らないよう
にするため)。3 枚未満のクラスがあれば、val/test の最低枚数を確保できない
ため分割全体をエラー終了する。結果は
`training/splits/<seed>/{train,val,test}.txt` に `<class>/<file>` 形式で
書き出す。

`training/scripts/prepare_dataset.sh` はこのスクリプトを呼び出し、書き出された
一覧をもとに `training/datasets/exterior/{train,val,test}/<class>/` へ実ファイル
をコピーするだけの薄いラッパ。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from classes import CLASS_NAMES

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "exterior"
DEFAULT_SPLIT_ROOT = REPO_ROOT / "training" / "splits"
DEFAULT_SEED = 42
SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")
DEFAULT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)
FINGERPRINT_FILENAME = "fingerprint.txt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
# val/test に最低 1 枚ずつ割り振るのに必要な最小枚数 (train が 0 枚でもよい).
MIN_IMAGES_PER_CLASS = 3


@dataclass(frozen=True)
class ManifestEntry:
    file: str
    cls: str


def read_manifest(manifest_path: Path) -> list[ManifestEntry]:
    """manifest.csv の file/class 列だけを読む (source/license 等はここでは不要).

    分割対象クラスの決定には使わない (images_by_class() 参照)。ここで読んだ
    `file` 列は、出典記録の有無を確認する目的でのみ使う
    (warn_missing_manifest_entries() 参照)。
    """
    if not manifest_path.exists():
        return []
    entries: list[ManifestEntry] = []
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            file_name = (row.get("file") or "").strip()
            cls = (row.get("class") or "").strip()
            if file_name and cls:
                entries.append(ManifestEntry(file=file_name, cls=cls))
    return entries


def validate_class_directories(data_dir: Path) -> None:
    """クラスは `classes.CLASS_NAMES` の固定 4 クラスに限る.

    - `data_dir` 自体が無ければエラー.
    - `_` で始まらないのに `CLASS_NAMES` に無いフォルダ (未知のクラス) が
      あればエラー (フォルダ名を列挙).
    - `CLASS_NAMES` のいずれかのフォルダが無い、または画像が 1 枚も無い
      (空) 場合もエラー (列挙する)。一部のクラスだけで黙って学習を始めて
      しまうのを防ぐ。
    """
    if not data_dir.exists():
        raise ValueError(f"{data_dir} が存在しません")

    known = set(CLASS_NAMES)
    unknown = sorted(
        entry.name
        for entry in data_dir.iterdir()
        if entry.is_dir() and not entry.name.startswith("_") and entry.name not in known
    )
    if unknown:
        raise ValueError(
            f"未知のクラスフォルダがあります (固定 {len(CLASS_NAMES)} クラス "
            f"{list(CLASS_NAMES)} 以外は対象外): {', '.join(unknown)}"
        )

    def _has_images(cls_dir: Path) -> bool:
        return cls_dir.is_dir() and any(
            p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in cls_dir.iterdir()
        )

    missing_or_empty = [cls for cls in CLASS_NAMES if not _has_images(data_dir / cls)]
    if missing_or_empty:
        raise ValueError(
            "次のクラスに画像がありません (フォルダが無い、または空です): "
            f"{', '.join(missing_or_empty)}"
        )


def images_by_class(data_dir: Path) -> dict[str, list[str]]:
    """`data_dir` 直下のディレクトリ名をクラスとして画像を集約する.

    manifest ではなく実際のフォルダ配置を正とする (手で振り分け直した画像は
    新しいフォルダのクラスとして扱われる)。`_` で始まるディレクトリ
    (`_unsorted/` 等) と画像 1 枚も無いディレクトリは対象外。

    クラス集合を `CLASS_NAMES` に限定する検証は行わない
    (validate_class_directories() を先に呼ぶこと)。
    """
    by_class: dict[str, list[str]] = {}
    if not data_dir.exists():
        return by_class
    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        files = sorted(
            p.name for p in entry.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if files:
            by_class[entry.name] = files
    return by_class


def warn_missing_manifest_entries(
    by_class: dict[str, list[str]], manifest_filenames: set[str]
) -> None:
    """manifest に出典が記録されていない画像の件数をクラスごとに警告表示する."""
    for cls in sorted(by_class):
        missing = [f for f in by_class[cls] if f not in manifest_filenames]
        if missing:
            print(
                f"⚠️  {cls}: manifest.csv に記録の無い画像が {len(missing)} 枚あります"
                " (手動追加または外部データセットの可能性)"
            )


def validate_class_sizes(by_class: dict[str, list[str]]) -> None:
    """val/test に最低 1 枚ずつ配れない (3 枚未満の) クラスがあればまとめてエラーにする."""
    too_small = {
        cls: len(files) for cls, files in by_class.items() if len(files) < MIN_IMAGES_PER_CLASS
    }
    if too_small:
        details = ", ".join(f"{cls}={n}枚" for cls, n in sorted(too_small.items()))
        raise ValueError(
            f"val/test に最低1枚ずつ割り振るには各クラス{MIN_IMAGES_PER_CLASS}枚以上必要です。"
            f"不足しているクラス: {details}"
        )


def compute_fingerprint(
    by_class: dict[str, list[str]],
    seed: int,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> str:
    """分割対象一覧 (`<class>/<file>` をソートしたもの) + seed + ratios の sha256 hex digest.

    seed/ratios を含めるのは、対象画像が同じでも別の分割になり得るため
    (対象一覧だけが一致していても、seed/ratios が違えば別物として扱う).
    """
    lines = sorted(f"{cls}/{file_name}" for cls, files in by_class.items() for file_name in files)
    payload_lines = [f"seed={seed}", f"ratios={ratios[0]},{ratios[1]},{ratios[2]}", *lines]
    payload = "\n".join(payload_lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_files(
    files: Sequence[str],
    seed: int,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> dict[str, list[str]]:
    """`files` を train/val/test に決定的に分割する.

    同じ (files の集合, seed, ratios) なら常に同じ結果になり、`files` の並び順
    には依存しない (先にソートしてからシャッフルする)。重複なし・3 分割の
    合計は入力件数と一致する。

    `len(files) >= MIN_IMAGES_PER_CLASS` (3) のときは val・test に必ず
    最低 1 枚ずつ割り振る (ratio の丸めで 0 枚になるのを防ぐ)。3 枚未満は
    その保証ができないため ValueError。
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    if any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be non-negative, got {ratios}")
    if len(files) < MIN_IMAGES_PER_CLASS:
        raise ValueError(
            f"need at least {MIN_IMAGES_PER_CLASS} files to guarantee 1 val + 1 test, got {len(files)}"
        )

    shuffled = sorted(files)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = min(round(n * ratios[0]), n)
    n_val = min(round(n * ratios[1]), n - n_train)
    n_test = n - n_train - n_val

    # ratio の丸めで val/test が 0 枚になった場合、train (次点で val) から
    # 1 枚借りて最低保証を満たす。train/val/test の合計は常に n のまま.
    if n_test == 0:
        n_test = 1
        if n_train > 0:
            n_train -= 1
        else:
            n_val -= 1
    if n_val == 0:
        n_val = 1
        if n_train > 0:
            n_train -= 1
        else:
            n_test -= 1

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _loaded_split_is_consistent(
    loaded: dict[str, list[str]], by_class: dict[str, list[str]]
) -> bool:
    """fingerprint が一致していても、読み込んだ分割の中身自体を検査する.

    - 対象一覧 (`by_class` から導ける全パス) がちょうど 1 回ずつ現れる
      (欠落・重複が無い)。
    - 対象一覧に無い未知のパスが混じっていない。
    - 各クラスの val・test に最低 1 件ずつある。

    どれか一つでも満たさなければ False (呼び出し側が作り直す)。
    """
    expected = {f"{cls}/{file_name}" for cls, files in by_class.items() for file_name in files}
    all_loaded = [p for name in SPLIT_NAMES for p in loaded.get(name, [])]

    if len(all_loaded) != len(set(all_loaded)):
        return False  # duplicate path across (or within) splits
    if set(all_loaded) != expected:
        return False  # missing and/or unknown paths

    for cls in by_class:
        val_count = sum(1 for p in loaded.get("val", []) if p.startswith(f"{cls}/"))
        test_count = sum(1 for p in loaded.get("test", []) if p.startswith(f"{cls}/"))
        if val_count < 1 or test_count < 1:
            return False

    return True


def load_existing_splits(
    split_dir: Path, expected_fingerprint: str, by_class: dict[str, list[str]]
) -> dict[str, list[str]] | None:
    """fingerprint が一致し、かつ中身が整合している既存の分割があれば読み込んで返す.

    fingerprint ファイルや分割ファイルが欠けている、fingerprint が一致しない
    (対象一覧・seed・ratios が変わった)、または読み込んだ中身が
    `by_class` と整合しない場合は None を返し、呼び出し側が作り直す。
    後者 2 つの場合は再利用しない旨を警告表示する。
    """
    fingerprint_path = split_dir / FINGERPRINT_FILENAME
    paths = {name: split_dir / f"{name}.txt" for name in SPLIT_NAMES}
    if not fingerprint_path.exists() or not all(p.exists() for p in paths.values()):
        return None

    stored_fingerprint = fingerprint_path.read_text(encoding="utf-8").strip()
    if stored_fingerprint != expected_fingerprint:
        print(
            f"⚠️  {split_dir} の分割対象一覧・seed・ratios が前回から変わっています"
            " (fingerprint 不一致)。既存の分割は再利用せず作り直します。"
        )
        return None

    loaded = {
        name: [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        for name, p in paths.items()
    }
    if not _loaded_split_is_consistent(loaded, by_class):
        print(
            f"⚠️  {split_dir} の分割ファイルの中身が対象一覧と整合していません"
            " (欠落・重複・未知のパス、または val/test が空のクラスがあります)。"
            "既存の分割は再利用せず作り直します。"
        )
        return None

    return loaded


def write_splits(split_dir: Path, combined: dict[str, list[str]], fingerprint: str) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        lines = combined.get(name, [])
        text = "\n".join(lines) + ("\n" if lines else "")
        (split_dir / f"{name}.txt").write_text(text, encoding="utf-8")
    (split_dir / FINGERPRINT_FILENAME).write_text(fingerprint + "\n", encoding="utf-8")


def generate_or_load_splits(
    data_dir: Path,
    manifest_path: Path,
    split_dir: Path,
    seed: int,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    force: bool = False,
) -> dict[str, list[str]]:
    validate_class_directories(data_dir)
    by_class = images_by_class(data_dir)
    validate_class_sizes(by_class)

    manifest_filenames = {entry.file for entry in read_manifest(manifest_path)}
    warn_missing_manifest_entries(by_class, manifest_filenames)

    fingerprint = compute_fingerprint(by_class, seed, ratios)

    if not force:
        existing = load_existing_splits(split_dir, fingerprint, by_class)
        if existing is not None:
            return existing

    combined: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for cls in sorted(by_class):
        splits = split_files(by_class[cls], seed, ratios)
        for name in SPLIT_NAMES:
            combined[name].extend(f"{cls}/{file_name}" for file_name in splits[name])

    write_splits(split_dir, combined, fingerprint)
    return combined


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="分割の乱数シード")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="クラス別画像ディレクトリ (default: data/raw/exterior)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest.csv のパス (default: <data-dir>/manifest.csv)",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help="分割結果の出力先 (default: training/splits/<seed>)",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=DEFAULT_RATIOS,
        metavar=("TRAIN", "VAL", "TEST"),
        help="train/val/test の比率 (合計 1.0, default: 0.70 0.15 0.15)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="fingerprint が一致していても既存の分割を使わず作り直す",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir: Path = args.data_dir
    manifest_path: Path = args.manifest or (data_dir / "manifest.csv")
    seed: int = args.seed
    split_dir: Path = args.split_dir or (DEFAULT_SPLIT_ROOT / str(seed))
    ratios: tuple[float, float, float] = tuple(args.ratios)

    try:
        splits = generate_or_load_splits(
            data_dir, manifest_path, split_dir, seed, ratios, args.force
        )
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    for name in SPLIT_NAMES:
        print(f"{name}: {len(splits[name])} 件")
    print(f"書き出し先: {split_dir}")


if __name__ == "__main__":
    main()
