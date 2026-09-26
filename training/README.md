# Training

YOLOv8 classification で **車外観 4 角度クラス** (left_front / left_rear / right_front / right_rear) を学習する.

GPU + Docker + NVIDIA Container Toolkit を前提とする (RTX5080 で確認).

## 手順

### 1. データセット準備 (ホスト側)

`data/raw/exterior/<class>/` に画像を振り分けた状態で (`<class>` は実際に
置いたフォルダ名。手で振り分け直した結果がそのままクラスになる):

```bash
bash training/scripts/prepare_dataset.sh
```

分割は `training/scripts/split_dataset.py` が担う:

- **クラス集合**は `training/scripts/classes.py` の `CLASS_NAMES`
  (left_front / left_rear / right_front / right_rear) に固定する。
  `apps/inference-demo/src/classifier.ts` の `CLASS_NAMES` と要素・順序が
  一致していなければならず (モデル出力のインデックスに対応するため)、
  一致は `tests/test_classes.py` が classifier.ts を読んで検証する。
  `_` で始まらないのに 4 クラスに無いフォルダ (未知のクラス) があれば
  フォルダ名を挙げてエラー終了し、4 クラスのどれかが欠けている
  (フォルダが無い・空) 場合もエラー終了する。
- **ラベル**は `data/raw/exterior/manifest.csv`
  (`data/scripts/download_datasets.py` が書き出す) の `class` 列ではなく、
  画像が**いま実際に置かれているフォルダ名**で決まる。manifest は出典
  (source/author/license/URL) の記録にのみ使い、manifest に行が無い画像
  (手で置いた画像・外部データセットなど) も分割対象に含める。ただし
  記録の無い画像はクラスごとの枚数を警告表示する。
- seed 固定・クラスごとに train/val/test = 70/15/15 へ決定的に分割する
  (`SEED` 環境変数で変更可能, default `42`)。3 枚以上あるクラスには
  val・test に最低 1 枚ずつを保証し (小さいクラスが val=0/test=0 のまま
  学習に回らないようにするため)、3 枚未満のクラスがあればどのクラスが
  何枚不足しているかを表示してエラー終了する。
- 分割結果は `training/splits/<seed>/{train,val,test}.txt` と、対象一覧
  (`<class>/<file>` の集合) + seed + ratios の sha256 を書いた
  `fingerprint.txt` に書き出す。次回実行時は、今の対象一覧・seed・ratios
  から計算した fingerprint が保存済みのものと一致する場合だけ再利用し、
  一致しなければ (画像が増減・再分類された、または seed/ratios を変えた)
  警告を出して作り直す。fingerprint が一致していても、読み込んだ 3
  ファイルの中身自体 (欠落・重複・未知のパスが無いか、各クラスの
  val/test が 1 件以上あるか) を検査し、壊れていれば同様に作り直す。
  `python3 training/scripts/split_dataset.py --force` で fingerprint に
  関わらず常に作り直せる。分割対象が 0 件ならエラー終了する (空の分割は
  書き出さない)。

`prepare_dataset.sh` は上の分割結果を読んで実ファイルを
`training/datasets/exterior/{train,val,test}/<class>/` へコピーするだけの
薄いラッパ。

### 2. 学習 (コンテナ)

```bash
cd training
docker compose up --build
```

`val` は学習中のモデル選択に、`test` は学習後の最終評価
(`train.py` の `evaluate_model()`) に使う。

完了後の出力 (ホスト側パス):

| 内容 | パス |
|------|------|
| PyTorch weights | `training/runs/classify/exterior_angle/weights/best.pt` |
| ONNX export | `training/models/exterior_angle.onnx` |
| 混同行列 (test split) | `training/runs/classify/exterior_angle/confusion_matrix.png` |

### 3. ONNX をフロントへ配置

```bash
cp training/models/exterior_angle.onnx apps/inference-demo/public/models/
```

## ハイパーパラメータ

`training/scripts/train.py` 冒頭の定数を直接編集:

- `MODEL_NAME` (default `yolov8s-cls.pt`)
- `EPOCHS` (default `50`)
- `IMAGE_SIZE` (default `640`)
- `BATCH_SIZE` (default `16`)
