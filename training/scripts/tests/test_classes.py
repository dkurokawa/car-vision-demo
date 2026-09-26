from __future__ import annotations

import re

from classes import CLASS_NAMES
from split_dataset import REPO_ROOT

CLASSIFIER_TS_PATH = REPO_ROOT / "apps" / "inference-demo" / "src" / "classifier.ts"


def test_class_names_is_the_fixed_four_classes_in_order() -> None:
    assert CLASS_NAMES == ("left_front", "left_rear", "right_front", "right_rear")


def _extract_classifier_ts_class_names(content: str) -> list[str]:
    match = re.search(r"CLASS_NAMES\s*=\s*\[([^\]]*)\]", content)
    assert match is not None, "CLASS_NAMES array not found in classifier.ts"
    return [name.strip().strip("'\"") for name in match.group(1).split(",") if name.strip()]


def test_class_names_matches_classifier_ts_element_for_element() -> None:
    """`classes.CLASS_NAMES` と classifier.ts の CLASS_NAMES は要素・順序が
    一致していなければならない (モデル出力インデックス i が CLASS_NAMES[i]
    に対応するため)。"""
    content = CLASSIFIER_TS_PATH.read_text(encoding="utf-8")
    ts_names = _extract_classifier_ts_class_names(content)

    assert ts_names == list(CLASS_NAMES)


def test_extract_classifier_ts_class_names_parses_a_typical_declaration() -> None:
    sample = "export const CLASS_NAMES = ['left_front', 'left_rear', 'right_front', 'right_rear'] as const"
    assert _extract_classifier_ts_class_names(sample) == [
        "left_front",
        "left_rear",
        "right_front",
        "right_rear",
    ]
