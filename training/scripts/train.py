#!/usr/bin/env python3
"""車外観 4 角度クラス分類モデル (YOLOv8-cls) の学習スクリプト.

クラス:
  - left_front  (左前)
  - left_rear   (左後)
  - right_front (右前)
  - right_rear  (右後)

データセット構造 (YOLOv8 classification 形式. training/scripts/split_dataset.py
の train/val/test 分割に対応):
  /app/datasets/exterior/
    train/{left_front,left_rear,right_front,right_rear}/*.jpg
    val/{left_front,left_rear,right_front,right_rear}/*.jpg
    test/{left_front,left_rear,right_front,right_rear}/*.jpg

出力:
  - PyTorch: /app/runs/classify/exterior_angle/weights/best.pt
  - ONNX:    /app/models/exterior_angle.onnx
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from classes import CLASS_NAMES
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from ultralytics import YOLO

DATA_DIR = Path("/app/datasets/exterior")
MODEL_NAME = "yolov8s-cls.pt"
EPOCHS = 50
# Ultralytics classification training/inference (both `model.train()` below
# and the `model.predict()` calls in evaluate_model()) preprocess with its
# built-in classify_transforms: resize the short side to IMAGE_SIZE, then
# center-crop IMAGE_SIZE x IMAGE_SIZE. apps/inference-demo/src/preprocess.ts
# (centerCropBox / toTensorData) implements the same transform for the
# browser demo so predictions are comparable between the two. If IMAGE_SIZE
# changes here, it must change there too.
IMAGE_SIZE = 640
BATCH_SIZE = 16
PROJECT = "/app/runs/classify"
RUN_NAME = "exterior_angle"

# Disable horizontal flip aug: flipping left_front into right_front (and
# vice versa) is the failure mode we're explicitly trying to fix.
FLIPLR = 0.0


def print_header(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def report_dataset_counts() -> None:
    print("📂 データセット確認:")
    for split in ("train", "val", "test"):
        print(f"  {split}:")
        for cls in CLASS_NAMES:
            d = DATA_DIR / split / cls
            n = len(list(d.glob("*"))) if d.exists() else 0
            print(f"    - {cls:12s}: {n} 枚")


def train_model() -> YOLO:
    print_header("🚀 モデル学習開始")
    model = YOLO(MODEL_NAME)
    print("学習パラメータ:")
    print(f"  base model = {MODEL_NAME}")
    print(f"  epochs     = {EPOCHS}")
    print(f"  imgsz      = {IMAGE_SIZE}")
    print(f"  batch      = {BATCH_SIZE}")
    print(f"  data       = {DATA_DIR}\n")

    model.train(
        data=str(DATA_DIR),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
        verbose=True,
        fliplr=FLIPLR,
    )
    print_header("✅ 学習完了")
    return model


def evaluate_model(model: YOLO) -> dict | None:
    """最終評価は test split で行う (val は学習中のモデル選択用)."""
    print_header("📊 モデル評価 (test split)")
    test_dir = DATA_DIR / "test"
    y_true: list[int] = []
    y_pred: list[int] = []

    for idx, cls in enumerate(CLASS_NAMES):
        d = test_dir / cls
        if not d.exists():
            print(f"⚠️  {d} がありません")
            continue
        images = [p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        for img in images:
            r = model.predict(str(img), verbose=False)
            y_true.append(idx)
            y_pred.append(int(r[0].probs.top1))

    if not y_true:
        print("❌ 評価データが見つかりません")
        return None

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("評価指標 (macro 平均):")
    print(f"  accuracy  = {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  precision = {prec:.4f}")
    print(f"  recall    = {rec:.4f}")
    print(f"  f1        = {f1:.4f}\n")

    print("分類レポート:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    out = Path(PROJECT) / RUN_NAME / "confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"💾 混同行列: {out}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}


def export_model() -> None:
    print_header("📦 ONNX エクスポート")
    pt_path = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
    if not pt_path.exists():
        print(f"❌ best.pt が見つかりません: {pt_path}")
        return
    model = YOLO(str(pt_path))
    model.export(format="onnx", imgsz=IMAGE_SIZE)

    onnx_src = pt_path.parent / "best.onnx"
    onnx_dest = Path("/app/models/exterior_angle.onnx")
    onnx_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_src, onnx_dest)
    print(f"✅ {onnx_dest}")


def main() -> None:
    print_header("🎓 車外観 4 角度クラス分類モデル学習")
    report_dataset_counts()
    model = train_model()
    metrics = evaluate_model(model)
    export_model()
    print_header("🎉 完了")
    if metrics:
        print(f"  accuracy = {metrics['accuracy'] * 100:.2f}%")
        print(f"  f1       = {metrics['f1'] * 100:.2f}%")


if __name__ == "__main__":
    main()
