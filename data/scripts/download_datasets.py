#!/usr/bin/env python3
"""車外観画像を公開ソースから収集するスクリプト.

対象クラス (車外観の4方向):
  - left_front  (左前)
  - left_rear   (左後)
  - right_front (右前)
  - right_rear  (右後)

収集ソース:
  1. Unsplash API  (UNSPLASH_ACCESS_KEY)
  2. Pexels  API   (PEXELS_API_KEY)
  3. Pixabay API   (PIXABAY_API_KEY)

API キーは car-vision/.env に置く (gitignore 済み):
  UNSPLASH_ACCESS_KEY=...
  PEXELS_API_KEY=...
  PIXABAY_API_KEY=...

注:
  - 収集した画像は data/raw/exterior/_unsorted/<class>/ に保存される
  - クラス振り分けは別途 (手動 or 学習済みモデル) で行う
  - 保存した画像ごとに data/raw/exterior/manifest.csv に
    file,class,source,source_url,author,license,downloaded_at を追記する。
    出典・ライセンス・元URLの復元に使う。API が返さない項目は空欄にする
  - --target はクラスあたりの「総」目標枚数。既存枚数は data/raw/exterior/<class>/
    (振り分け済み) と _unsorted/<class>/ (振り分け前) の実ファイル数の合計で数え
    (manifest.csv の件数では数えない。統一のため count_collected() に一本化)、
    不足分だけ新規に取得する
  - Unsplash は API ガイドライン上、実際に使う (保存する) 画像ごとに
    `links.download_location` への GET が必須。失敗したらその画像はスキップする
    (収集自体は継続する)
  - EPFL/Stanford/Kaggle データセットは利用規約上 API 自動 DL 不可のため
    手動取得 (案内のみ)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
RAW_DIR = BASE_DIR / "raw"
EXTERIOR_DIR = RAW_DIR / "exterior"
UNSORTED_DIR = EXTERIOR_DIR / "_unsorted"
MANIFEST_PATH = EXTERIOR_DIR / "manifest.csv"
MANIFEST_FIELDS = [
    "file",
    "class",
    "source",
    "source_url",
    "author",
    "license",
    "downloaded_at",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

CLASS_QUERIES: dict[str, list[str]] = {
    "left_front": [
        "car front left quarter view",
        "sedan left front angle",
        "suv front left three quarter",
        "car driver side front",
        "vehicle front left perspective",
        "car left headlight angle",
    ],
    "left_rear": [
        "car rear left quarter view",
        "sedan left rear angle",
        "suv rear left three quarter",
        "car driver side rear",
        "vehicle rear left perspective",
        "car left taillight angle",
    ],
    "right_front": [
        "car front right quarter view",
        "sedan right front angle",
        "suv front right three quarter",
        "car passenger side front",
        "vehicle front right perspective",
        "car right headlight angle",
        "automobile right front fender",
        "japanese car right front view",
        "car right hand drive front quarter",
    ],
    "right_rear": [
        "car rear right quarter view",
        "sedan right rear angle",
        "suv rear right three quarter",
        "car passenger side rear",
        "vehicle rear right perspective",
        "car right taillight angle",
        "automobile right rear bumper",
        "japanese car right rear view",
        "car right hand drive rear quarter",
    ],
}


@dataclass(frozen=True)
class ImageCandidate:
    """1 件の候補画像とその出典メタデータ."""

    source: str
    url: str
    source_url: str | None
    author: str | None
    license: str | None
    # Unsplash のみ: 実際に保存する前に GET する必要がある URL
    # (https://help.unsplash.com/en/articles/2511245). 他ソースでは None.
    download_location: str | None = None


def load_env_keys() -> dict[str, str | None]:
    """.env から API キーを読み込む (値はログに出さない)."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with env_path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

    return {
        "unsplash": os.environ.get("UNSPLASH_ACCESS_KEY"),
        "pexels": os.environ.get("PEXELS_API_KEY"),
        "pixabay": os.environ.get("PIXABAY_API_KEY"),
    }


def _hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _existing_hashes(root: Path) -> set[str]:
    hashes: set[str] = set()
    if not root.exists():
        return hashes
    for p in root.rglob("*"):
        if p.suffix.lower() in IMAGE_SUFFIXES:
            try:
                hashes.add(_hash_bytes(p.read_bytes()))
            except OSError:
                continue
    return hashes


def count_existing(class_dir: Path) -> int:
    """`class_dir` 直下にある画像枚数 (サブディレクトリは見ない)."""
    if not class_dir.exists():
        return 0
    return sum(1 for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def count_collected(class_name: str) -> int:
    """`--target` の既存枚数として数える対象.

    振り分け済み data/raw/exterior/<class>/ と、振り分け前
    _unsorted/<class>/ の両方を実ファイル数で数えて合計する (manifest.csv の
    行数では数えない — 二重に真実の源を持たないため、実ファイルの存在の
    方に統一する)。
    """
    return count_existing(EXTERIOR_DIR / class_name) + count_existing(UNSORTED_DIR / class_name)


def compute_shortfall(existing: int, target: int) -> int:
    """`target` (クラスあたりの総目標枚数) に対して、あと何枚必要か."""
    return max(0, target - existing)


def _download(url: str, dest: Path, timeout: int = 30) -> bool:
    try:
        r = requests.get(url, timeout=timeout, allow_redirects=True)
        if r.status_code != 200:
            return False
        if "image" not in r.headers.get("content-type", ""):
            return False
        dest.write_bytes(r.content)
        return True
    except requests.RequestException:
        return False


def fetch_unsplash(query: str, access_key: str, per_page: int = 30) -> list[ImageCandidate]:
    """Unsplash Search API で画像候補一覧を取得 (作者/掲載ページ付き)."""
    candidates: list[ImageCandidate] = []
    params: dict[str, str | int] = {
        "query": query,
        "per_page": per_page,
        "orientation": "landscape",
    }
    try:
        r = requests.get(
            "https://api.unsplash.com/search/photos",
            params=params,
            headers={"Authorization": f"Client-ID {access_key}"},
            timeout=30,
        )
        r.raise_for_status()
        for item in r.json().get("results", []):
            url = item.get("urls", {}).get("regular")
            if not url:
                continue
            user = item.get("user") or {}
            candidates.append(
                ImageCandidate(
                    source="unsplash",
                    url=url,
                    source_url=item.get("links", {}).get("html"),
                    author=user.get("name"),
                    license="Unsplash License",
                    download_location=item.get("links", {}).get("download_location"),
                )
            )
    except requests.RequestException as e:
        print(f"  ! Unsplash error for '{query}': {e}", file=sys.stderr)
    return candidates


def fetch_pexels(query: str, api_key: str, per_page: int = 30) -> list[ImageCandidate]:
    """Pexels Search API で画像候補一覧を取得 (作者/掲載ページ付き)."""
    candidates: list[ImageCandidate] = []
    params: dict[str, str | int] = {
        "query": query,
        "per_page": per_page,
        "orientation": "landscape",
    }
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            params=params,
            headers={"Authorization": api_key},
            timeout=30,
        )
        r.raise_for_status()
        for item in r.json().get("photos", []):
            url = item.get("src", {}).get("large")
            if not url:
                continue
            candidates.append(
                ImageCandidate(
                    source="pexels",
                    url=url,
                    source_url=item.get("url"),
                    author=item.get("photographer"),
                    license="Pexels License",
                )
            )
    except requests.RequestException as e:
        print(f"  ! Pexels error for '{query}': {e}", file=sys.stderr)
    return candidates


def fetch_pixabay(query: str, api_key: str, per_page: int = 30) -> list[ImageCandidate]:
    """Pixabay API で画像候補一覧を取得 (投稿者/掲載ページ付き)."""
    candidates: list[ImageCandidate] = []
    params: dict[str, str | int] = {
        "key": api_key,
        "q": query,
        "per_page": per_page,
        "image_type": "photo",
        "orientation": "horizontal",
    }
    try:
        r = requests.get(
            "https://pixabay.com/api/",
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        for item in r.json().get("hits", []):
            url = item.get("largeImageURL") or item.get("webformatURL")
            if not url:
                continue
            candidates.append(
                ImageCandidate(
                    source="pixabay",
                    url=url,
                    source_url=item.get("pageURL"),
                    author=item.get("user"),
                    license="Pixabay License",
                )
            )
    except requests.RequestException as e:
        print(f"  ! Pixabay error for '{query}': {e}", file=sys.stderr)
    return candidates


def confirm_unsplash_download(download_location: str, access_key: str, timeout: int = 30) -> bool:
    """Unsplash API ガイドライン準拠: 実際に使う画像ごとに、保存前に
    `links.download_location` へ GET する (Authorization 付き).
    https://help.unsplash.com/en/articles/2511245-guideline-triggering-a-download

    成功 (200) なら True. 失敗したら False を返すだけで、呼び出し側がその
    画像をスキップする (収集自体は続行する)."""
    try:
        r = requests.get(
            download_location,
            headers={"Authorization": f"Client-ID {access_key}"},
            timeout=timeout,
        )
        return r.status_code == 200
    except requests.RequestException:
        return False


def append_manifest_row(manifest_path: Path, row: dict[str, str]) -> None:
    """1 件ぶんを manifest.csv に追記する (ヘッダは無ければ作る)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def collect(
    class_name: str,
    queries: Iterable[str],
    target: int,
    keys: dict[str, str | None],
    seen_hashes: set[str],
) -> int:
    """1 クラスぶん収集する. `target` は今回の収集ぶんの上限枚数 (既存枚数を
    差し引いた不足分は呼び出し側で計算する). 戻り値は新規保存枚数."""
    dest_root = UNSORTED_DIR / class_name
    dest_root.mkdir(parents=True, exist_ok=True)

    candidates: list[ImageCandidate] = []
    for q in queries:
        if keys["unsplash"]:
            candidates.extend(fetch_unsplash(q, keys["unsplash"]))
        if keys["pexels"]:
            candidates.extend(fetch_pexels(q, keys["pexels"]))
        if keys["pixabay"]:
            candidates.extend(fetch_pixabay(q, keys["pixabay"]))

    if not candidates:
        print(f"  ! '{class_name}': API キー未設定または検索結果ゼロ")
        return 0

    saved = 0
    pbar = tqdm(candidates, desc=f"  {class_name}", unit="img")
    for candidate in pbar:
        if saved >= target:
            break
        if candidate.source == "unsplash":
            # Unsplash API guideline: confirm the download *before* actually
            # fetching/saving the image, not after. Skip this image (without
            # downloading its bytes) if the key is missing, the search result
            # didn't include a download_location, or the ping fails.
            access_key = keys["unsplash"]
            if not candidate.download_location or not access_key:
                continue
            if not confirm_unsplash_download(candidate.download_location, access_key):
                continue
        tmp = dest_root / f".tmp_{int(time.time() * 1000)}.bin"
        if not _download(candidate.url, tmp):
            continue
        try:
            data = tmp.read_bytes()
            h = _hash_bytes(data)
            if h in seen_hashes:
                tmp.unlink(missing_ok=True)
                continue
            seen_hashes.add(h)
            final_name = f"{candidate.source}_{h[:12]}.jpg"
            final = dest_root / final_name
            tmp.rename(final)
            append_manifest_row(
                MANIFEST_PATH,
                {
                    "file": final_name,
                    "class": class_name,
                    "source": candidate.source,
                    "source_url": candidate.source_url or "",
                    "author": candidate.author or "",
                    "license": candidate.license or "",
                    "downloaded_at": datetime.now(UTC).isoformat(),
                },
            )
            saved += 1
            pbar.set_postfix(saved=saved)
        finally:
            tmp.unlink(missing_ok=True)
        time.sleep(0.1)

    return saved


def print_manual_dataset_notice() -> None:
    print(
        """
公開研究用データセットは利用規約上 API 自動 DL 不可なので手動取得:

  - EPFL Multi-View Car Dataset
      https://www.epfl.ch/labs/cvlab/data/data-pose-index-php/
      -> data/raw/exterior/_external/epfl/ に展開

  - Stanford Cars Dataset
      https://ai.stanford.edu/~jkrause/cars/car_dataset.html
      -> data/raw/exterior/_external/stanford/ に展開

  - Kaggle (Front/Rear Images of Car など)
      kaggle datasets download -d kushkunal/front-and-rear-images-of-car
      -> data/raw/exterior/_external/kaggle/ に展開

取得後は手動 or 別ツールで 4 角度クラスに振り分け.
詳細・ライセンスは data/EXTERNAL_DATASETS.md を参照.
""".strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=int,
        default=200,
        help="クラスあたりの総目標枚数 (既存分を数え、不足分だけ取得する. default: 200)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=list(CLASS_QUERIES.keys()),
        default=list(CLASS_QUERIES.keys()),
        help="収集対象クラス (default: 全 4 クラス)",
    )
    args = parser.parse_args()

    keys = load_env_keys()
    available = [k for k, v in keys.items() if v]
    print(f"📦 利用可能な API: {', '.join(available) if available else '(なし)'}")
    if not available:
        print(
            "⚠️  .env に UNSPLASH_ACCESS_KEY / PEXELS_API_KEY / PIXABAY_API_KEY を設定してください"
        )
        print_manual_dataset_notice()
        sys.exit(1)

    UNSORTED_DIR.mkdir(parents=True, exist_ok=True)
    seen = _existing_hashes(EXTERIOR_DIR)
    print(f"📂 既存画像ハッシュ: {len(seen)} 件 (重複除外用)")

    totals: dict[str, int] = {}
    for cls in args.classes:
        existing = count_collected(cls)
        shortfall = compute_shortfall(existing, args.target)
        print(f"\n🔍 {cls}: 既存 {existing} 枚 / 目標 {args.target} 枚 (不足 {shortfall} 枚)")
        if shortfall == 0:
            print(f"  ✅ '{cls}': 既に目標枚数に達しています")
            totals[cls] = 0
            continue
        totals[cls] = collect(cls, CLASS_QUERIES[cls], shortfall, keys, seen)

    print("\n" + "=" * 60)
    print("収集結果")
    print("=" * 60)
    for cls, n in totals.items():
        print(f"  {cls:12s} +{n} 枚")
    print(f"\n保存先: {UNSORTED_DIR.relative_to(PROJECT_ROOT)}/<class>/")
    print(f"manifest: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print_manual_dataset_notice()


if __name__ == "__main__":
    main()
