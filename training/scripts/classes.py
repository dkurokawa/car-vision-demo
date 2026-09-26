"""車外観 4 角度クラスの固定クラス集合 (SSOT).

`train.py` と `split_dataset.py` はここから import する (二重定義を避ける)。
順序を含めて `apps/inference-demo/src/classifier.ts` の `CLASS_NAMES` と
一致していなければならない — モデルの出力ロジットのインデックス i が
`CLASS_NAMES[i]` に対応するため、順序がずれるとブラウザ側の推論結果が
別のクラスとして表示される。一致は
`training/scripts/tests/test_classes.py` が classifier.ts を読んで検証する。
"""

from __future__ import annotations

CLASS_NAMES: tuple[str, str, str, str] = (
    "left_front",
    "left_rear",
    "right_front",
    "right_rear",
)
