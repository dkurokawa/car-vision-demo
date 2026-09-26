#!/bin/bash
#
# data/raw/exterior/<class>/* (画像がいま置かれているフォルダ名がそのまま
# クラスになる) を YOLOv8 classification 形式 (train/val/test 分割) に整形する.
#
# 分割そのものは training/scripts/split_dataset.py が担う (seed 固定・対象一覧の
# fingerprint が前回と一致すれば再利用・3 枚未満のクラスがあればエラー終了)。
# このスクリプトはその分割結果 (training/splits/<seed>/{train,val,test}.txt)
# を読んで実ファイルを展開するだけの薄いラッパ.
#
# 出力: training/datasets/exterior/{train,val,test}/<class>/
# クラス名は分割結果に書かれたものをそのまま使う (固定の4クラスに限らない.
# 手で振り分け直したフォルダや新しいクラスが増えても追従する).
# 既存の出力ディレクトリは削除して作り直す.
#
# 環境変数:
#   SEED  分割の乱数シード (default: 42). split_dataset.py 側の default と
#         一致させること.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$TRAINING_DIR/.." && pwd)"

SRC_DIR="$REPO_ROOT/data/raw/exterior"
DEST_DIR="$TRAINING_DIR/datasets/exterior"
SEED="${SEED:-42}"
SPLIT_DIR="$TRAINING_DIR/splits/$SEED"

SPLITS=(train val test)

echo "📦 データセット準備 (seed=$SEED)"
echo "  src       = $SRC_DIR"
echo "  dest      = $DEST_DIR"
echo "  split_dir = $SPLIT_DIR"
echo

python3 "$SCRIPT_DIR/split_dataset.py" --seed "$SEED" --data-dir "$SRC_DIR" --split-dir "$SPLIT_DIR"
echo

for split in "${SPLITS[@]}"; do
  if [[ ! -f "$SPLIT_DIR/$split.txt" ]]; then
    echo "❌ $SPLIT_DIR/$split.txt がありません (split_dataset.py が失敗した可能性があります)"
    exit 1
  fi
done

if [[ -d "$DEST_DIR" ]]; then
  rm -r "$DEST_DIR"
fi

# クラス名を固定リストで決め打ちしない (split_dataset.py 側がフォルダ名を
# そのままクラスとして扱うため、分割結果に出てきた分だけ都度ディレクトリを作る).
for split in "${SPLITS[@]}"; do
  count=0
  while IFS= read -r rel_path; do
    [[ -z "$rel_path" ]] && continue
    src_file="$SRC_DIR/$rel_path"
    if [[ ! -f "$src_file" ]]; then
      echo "  ! 見つかりません: $src_file (manifest.csv と実ファイルがずれている可能性があります)"
      continue
    fi
    dest_file="$DEST_DIR/$split/$rel_path"
    mkdir -p "$(dirname "$dest_file")"
    cp "$src_file" "$dest_file"
    count=$((count + 1))
  done <"$SPLIT_DIR/$split.txt"
  echo "  ✅ $split: $count 件"
done

echo
echo "📊 出力サマリ"
for split in "${SPLITS[@]}"; do
  [[ -d "$DEST_DIR/$split" ]] || continue
  for cls_dir in "$DEST_DIR/$split"/*/; do
    [[ -d "$cls_dir" ]] || continue
    cls="$(basename "$cls_dir")"
    n=$(find "$cls_dir" -type f | wc -l | tr -d ' ')
    echo "  $split/$cls: $n"
  done
done
