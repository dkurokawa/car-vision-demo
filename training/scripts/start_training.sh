#!/bin/bash
#
# コンテナ内で学習を起動する.

set -euo pipefail

echo "🎓 車外観 4 角度クラス分類モデル学習を開始します"
echo

python3 /app/scripts/train.py

echo
echo "✅ 学習完了"
echo
echo "📁 出力先:"
echo "  - モデル (PyTorch): /app/runs/classify/exterior_angle/weights/best.pt"
echo "  - モデル (ONNX):    /app/models/exterior_angle.onnx"
echo "  - 混同行列:         /app/runs/classify/exterior_angle/confusion_matrix.png"
