import { useRef, useState } from 'react'
import { CLASS_LABELS_JA, CLASS_NAMES, classify, type Prediction } from './classifier'

type Status =
  | { kind: 'idle' }
  | { kind: 'running' }
  | { kind: 'done'; result: Prediction }
  | { kind: 'error'; message: string }

export function App() {
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [status, setStatus] = useState<Status>({ kind: 'idle' })
  const imgRef = useRef<HTMLImageElement | null>(null)
  // Bumped on every new file / run, so an in-flight classification whose
  // image has since changed can recognize itself as stale and drop its
  // result instead of overwriting a newer one.
  const requestIdRef = useRef(0)

  const onFile = (file: File) => {
    requestIdRef.current += 1
    if (imageUrl) URL.revokeObjectURL(imageUrl)
    setImageUrl(URL.createObjectURL(file))
    setStatus({ kind: 'idle' })
  }

  const run = async () => {
    if (!imgRef.current || !imgRef.current.complete) return
    const requestId = ++requestIdRef.current
    setStatus({ kind: 'running' })
    try {
      const result = await classify(imgRef.current)
      if (requestIdRef.current !== requestId) return // superseded by a newer image/run
      setStatus({ kind: 'done', result })
    } catch (e) {
      if (requestIdRef.current !== requestId) return
      const message = e instanceof Error ? e.message : String(e)
      setStatus({ kind: 'error', message })
    }
  }

  return (
    <main
      style={{
        maxWidth: 760,
        margin: '0 auto',
        padding: '32px 24px',
        display: 'flex',
        flexDirection: 'column',
        gap: 24,
      }}
    >
      <header>
        <h1 style={{ margin: 0, fontSize: 28 }}>car-vision</h1>
        <p style={{ color: '#475569', marginTop: 4 }}>
          車外観 4 角度 (左前 / 左後 / 右前 / 右後) を ONNX Runtime Web で分類するデモ
        </p>
      </header>

      <section
        style={{
          border: '1px dashed #cbd5e1',
          borderRadius: 8,
          padding: 24,
          background: '#fff',
        }}
      >
        <label
          style={{
            display: 'inline-block',
            padding: '8px 14px',
            border: '1px solid #1f2937',
            borderRadius: 6,
            background: '#1f2937',
            color: '#fff',
          }}
        >
          画像を選択
          <input
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) onFile(file)
            }}
          />
        </label>

        {imageUrl && (
          <div style={{ marginTop: 16 }}>
            <img
              ref={imgRef}
              src={imageUrl}
              alt=""
              style={{ maxWidth: '100%', borderRadius: 6, display: 'block' }}
              onLoad={() => {
                setStatus({ kind: 'idle' })
              }}
            />
            <button
              onClick={() => {
                void run()
              }}
              disabled={status.kind === 'running'}
              style={{
                marginTop: 12,
                padding: '8px 16px',
                background: '#0ea5e9',
                color: '#fff',
                border: 'none',
                borderRadius: 6,
              }}
            >
              {status.kind === 'running' ? '推論中...' : '推論する'}
            </button>
          </div>
        )}
      </section>

      {status.kind === 'error' && (
        <section style={{ background: '#fee2e2', padding: 16, borderRadius: 6, color: '#991b1b' }}>
          <strong>エラー:</strong> {status.message}
          <p style={{ marginTop: 8, marginBottom: 0, fontSize: 13 }}>
            <code>apps/inference-demo/public/models/exterior_angle.onnx</code> が配置されているか確認してください.
          </p>
        </section>
      )}

      {status.kind === 'done' && (
        <section style={{ background: '#fff', padding: 16, borderRadius: 6, border: '1px solid #e2e8f0' }}>
          <h2 style={{ marginTop: 0 }}>
            予測: {status.result.labelJa}{' '}
            <span style={{ color: '#64748b', fontSize: 14 }}>
              ({(status.result.confidence * 100).toFixed(1)}%)
            </span>
          </h2>
          <ul style={{ paddingLeft: 16 }}>
            {CLASS_NAMES.map((c) => (
              <li key={c}>
                {CLASS_LABELS_JA[c]} ({c}): {(status.result.probabilities[c] * 100).toFixed(1)}%
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  )
}
