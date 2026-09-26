# 外部公開データセットの手動取得

API で取れる Unsplash/Pexels/Pixabay と異なり、研究用データセットは
ライセンス上ブラウザ承諾やアカウント登録を挟むため、**手動取得**.

取得後は `data/raw/exterior/_external/<source>/` に展開して、別途
角度クラス (`left_front` / `left_rear` / `right_front` / `right_rear`)
に振り分ける.

---

## EPFL Multi-View Car Dataset

20 視点 x 多数車種.  角度情報がメタデータに含まれているため
**ラベル付け済みデータとして最も価値が高い**.

- URL: <https://www.epfl.ch/labs/cvlab/data/data-pose-index-php/>
- ファイル: `car_ims.tgz` (約 1.5 GB)
- ライセンス: 研究用途限定。画像は本リポに含めない

```bash
mkdir -p data/raw/exterior/_external/epfl
cd data/raw/exterior/_external/epfl
# ブラウザでダウンロードリンクに同意して取得後:
tar xzf car_ims.tgz
```

角度はファイル名 `<carID>_<angle>.jpg` の `<angle>` から 0-19 度数で取れる.
4 クラス相当への mapping は例えば:

| angle (度数) | クラス |
|-------------|--------|
| 0-89        | right_front |
| 90-179      | right_rear |
| 180-269     | left_rear |
| 270-359     | left_front |

実際の角度割当は研究室の README に従う.

---

## Stanford Cars Dataset

196 車種 x 16,185 枚.  角度ラベルは無いが車そのものが鮮明.

- URL: <http://ai.stanford.edu/~jkrause/cars/car_dataset.html>
- ファイル: `cars_train.tgz` / `cars_test.tgz` (各 ~2 GB)
- ライセンス: 研究用途限定。画像は本リポに含めない

```bash
mkdir -p data/raw/exterior/_external/stanford
cd data/raw/exterior/_external/stanford
# ブラウザで取得後:
tar xzf cars_train.tgz
tar xzf cars_test.tgz
```

注意: 角度ラベルが無いので **使う場合は手動 or 既存モデル経由でクラス振り分けが必須**.

---

## Kaggle データセット

Front/Rear の 2 クラスに限定されるが手早く増やせる.

- 例: `kushkunal/front-and-rear-images-of-car`
  <https://www.kaggle.com/datasets/kushkunal/front-and-rear-images-of-car>
- ライセンス: データセットごとに異なる。利用前に個別に確認する。画像は本リポに含めない

```bash
mkdir -p data/raw/exterior/_external/kaggle
cd data/raw/exterior/_external/kaggle
# Kaggle CLI が必要:  pip install kaggle, ~/.kaggle/kaggle.json に API token 配置
kaggle datasets download -d kushkunal/front-and-rear-images-of-car
unzip front-and-rear-images-of-car.zip
```

---

## まとめ

| ソース | 角度ラベル | サイズ目安 | 自動取得 |
|--------|----------|----------|--------|
| Unsplash / Pexels / Pixabay | クエリ依存 (ノイジー) | 数百枚/サービス | ✅ `download_datasets.py` |
| EPFL | ◎ あり | 1.5 GB | ❌ ブラウザ承諾必須 |
| Stanford Cars | ✗ 無し | 4 GB | ❌ ブラウザ承諾必須 |
| Kaggle (Front/Rear) | △ 2 クラスのみ | ~300 MB | ⚠️ Kaggle CLI + token 必要 |
