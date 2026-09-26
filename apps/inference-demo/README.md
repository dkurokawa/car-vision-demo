# inference-demo

車外観 4 角度 (left_front / left_rear / right_front / right_rear) を ONNX Runtime Web でブラウザ推論するデモ.

Vite + React + onnxruntime-web.

## セットアップ

学習済みモデルはリポジトリに含めない. `../../training/` で学習して得た ONNX を
`public/models/exterior_angle.onnx` に置いてから起動する:

```bash
npm install
cp ../../training/models/exterior_angle.onnx public/models/
npm run dev
```

`*.onnx` は `.gitignore` 済み. モデルが無くても `npm run build` は通る (実行時に
`/models/exterior_angle.onnx` の取得に失敗した場合はエラー表示になる).

## 品質チェック

```bash
npm run lint   # eslint (typescript-eslint strictTypeChecked)
npm test       # vitest (src/**/*.test.ts)
```

## ビルド

```bash
npm run build      # tsc -b && vite build → dist/
npm run preview    # dist/ を確認
```
