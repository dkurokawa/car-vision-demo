import * as ort from 'onnxruntime-web'
import { centerCropBox, toTensorData } from './preprocess'

export const CLASS_NAMES = ['left_front', 'left_rear', 'right_front', 'right_rear'] as const
export type ClassName = (typeof CLASS_NAMES)[number]

export const CLASS_LABELS_JA: Record<ClassName, string> = {
  left_front: '左前',
  left_rear: '左後',
  right_front: '右前',
  right_rear: '右後',
}

const MODEL_URL = '/models/exterior_angle.onnx'
const IMAGE_SIZE = 640
const EXPECTED_INPUT_SHAPE = [1, 3, IMAGE_SIZE, IMAGE_SIZE]
const EXPECTED_OUTPUT_SHAPE = [1, CLASS_NAMES.length]
// Ultralytics' `model.export(format="onnx")` names the single input/output
// this way by default (see training/scripts/train.py's export_model()). If
// training ever renames them, this check should fail loudly rather than
// silently feeding the wrong tensor.
const EXPECTED_INPUT_NAME = 'images'
const EXPECTED_OUTPUT_NAME = 'output0'

let sessionPromise: Promise<ort.InferenceSession> | null = null

type ValueMetadata = ort.InferenceSession.ValueMetadata

/** Subset of `ort.InferenceSession` needed for shape validation; kept small so tests can pass in plain objects. */
export interface SessionShape {
  inputNames: readonly string[]
  outputNames: readonly string[]
  // Only available on newer onnxruntime-web builds; when absent we fall
  // back to validating the output tensor's actual dtype/shape at runtime
  // (see extractLogits()).
  inputMetadata?: readonly ValueMetadata[]
  outputMetadata?: readonly ValueMetadata[]
}

function formatShape(shape: readonly (number | string)[]): string {
  return `[${shape.join(',')}]`
}

/** A symbolic (string) dimension is a declared wildcard, so it always matches. */
function shapeMatches(actual: readonly (number | string)[], expected: readonly number[]): boolean {
  return actual.length === expected.length && actual.every((d, i) => typeof d === 'string' || d === expected[i])
}

function assertTensorMetadata(
  meta: ValueMetadata | undefined,
  expectedShape: readonly number[],
  label: string,
): void {
  if (!meta) return // not available on this onnxruntime-web build; checked at runtime instead
  if (!meta.isTensor) {
    throw new Error(`${label} が Tensor ではありません (期待: float32 Tensor / 実際: 非 Tensor の値)`)
  }
  if (meta.type !== 'float32') {
    throw new Error(`${label} dtype が一致しません (期待: float32 / 実際: ${meta.type})`)
  }
  if (!shapeMatches(meta.shape, expectedShape)) {
    throw new Error(
      `${label} shape が一致しません (期待: ${formatShape(expectedShape)} / 実際: ${formatShape(meta.shape)})`,
    )
  }
}

/**
 * Throws a descriptive error unless the session exposes exactly one input
 * and one output, named the way Ultralytics' ONNX export names them
 * ("images" / "output0"). If the runtime also exposes tensor metadata,
 * also validates dtype/shape against what the classifier expects.
 */
export function assertSessionShape(session: SessionShape): void {
  if (session.inputNames.length !== 1) {
    throw new Error(
      `想定外の入力数です (期待: 1 / 実際: ${String(session.inputNames.length)}: [${session.inputNames.join(', ')}])`,
    )
  }
  if (session.outputNames.length !== 1) {
    throw new Error(
      `想定外の出力数です (期待: 1 / 実際: ${String(session.outputNames.length)}: [${session.outputNames.join(', ')}])`,
    )
  }
  if (session.inputNames[0] !== EXPECTED_INPUT_NAME) {
    throw new Error(`入力名が一致しません (期待: "${EXPECTED_INPUT_NAME}" / 実際: "${session.inputNames[0]}")`)
  }
  if (session.outputNames[0] !== EXPECTED_OUTPUT_NAME) {
    throw new Error(`出力名が一致しません (期待: "${EXPECTED_OUTPUT_NAME}" / 実際: "${session.outputNames[0]}")`)
  }

  assertTensorMetadata(session.inputMetadata?.[0], EXPECTED_INPUT_SHAPE, '入力')
  assertTensorMetadata(session.outputMetadata?.[0], EXPECTED_OUTPUT_SHAPE, '出力')
}

function dimsMatch(dims: readonly number[], expected: readonly number[]): boolean {
  return dims.length === expected.length && dims.every((d, i) => d === expected[i])
}

/**
 * Validates that an ONNX input tensor matches the shape/dtype the model
 * expects, throwing a descriptive error otherwise. We build this tensor
 * ourselves, so this is a defensive self-check against preprocessing bugs
 * rather than a check on untrusted data.
 */
export function assertInputTensor(tensor: ort.Tensor): void {
  if (tensor.type !== 'float32') {
    throw new Error(`入力 dtype が一致しません (期待: float32 / 実際: ${tensor.type})`)
  }
  if (!dimsMatch(tensor.dims, EXPECTED_INPUT_SHAPE)) {
    throw new Error(
      `入力 shape が一致しません (期待: ${formatShape(EXPECTED_INPUT_SHAPE)} / 実際: ${formatShape(tensor.dims)})`,
    )
  }
}

/**
 * Validates the model's output tensor and returns its data as a
 * `Float32Array`. Replaces a blind `as Float32Array` cast with a runtime
 * check: only after confirming `type === 'float32'` do we treat `.data` as
 * such. This is also the fallback shape/dtype check for onnxruntime-web
 * builds that don't expose `outputMetadata` (see assertSessionShape()).
 */
export function extractLogits(tensor: ort.Tensor): Float32Array {
  if (tensor.type !== 'float32') {
    throw new Error(`出力 dtype が一致しません (期待: float32 / 実際: ${tensor.type})`)
  }
  if (!dimsMatch(tensor.dims, EXPECTED_OUTPUT_SHAPE)) {
    throw new Error(
      `出力 shape が一致しません (期待: ${formatShape(EXPECTED_OUTPUT_SHAPE)} / 実際: ${formatShape(tensor.dims)})`,
    )
  }
  // Safe: verified above that this tensor's dtype is float32.
  return tensor.data as Float32Array
}

async function loadSession(): Promise<ort.InferenceSession> {
  const session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })
  assertSessionShape(session)
  return session
}

/**
 * Returns the cached session, creating it on first use. If creation fails
 * (network error, shape mismatch, ...), the failed promise is discarded so
 * the next call retries instead of replaying the same rejection forever.
 */
export async function getSession(): Promise<ort.InferenceSession> {
  sessionPromise ??= loadSession().catch((error: unknown) => {
    sessionPromise = null
    throw error
  })
  return sessionPromise
}

export interface Prediction {
  className: ClassName
  labelJa: string
  confidence: number
  probabilities: Record<ClassName, number>
}

function softmax(logits: Float32Array): Float32Array {
  const max = Math.max(...logits)
  const exp = logits.map((v) => Math.exp(v - max))
  const sum = exp.reduce((a, b) => a + b, 0)
  return Float32Array.from(exp.map((v) => v / sum))
}

function buildInputTensor(image: HTMLImageElement): ort.Tensor {
  const { naturalWidth: width, naturalHeight: height } = image
  const { sx, sy, sSize } = centerCropBox(width, height, IMAGE_SIZE)

  const canvas = document.createElement('canvas')
  canvas.width = IMAGE_SIZE
  canvas.height = IMAGE_SIZE
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    throw new Error('failed to acquire a 2d canvas context')
  }
  ctx.drawImage(image, sx, sy, sSize, sSize, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
  const { data } = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE)
  const tensorData = toTensorData(data, IMAGE_SIZE, IMAGE_SIZE)

  const tensor = new ort.Tensor('float32', tensorData, EXPECTED_INPUT_SHAPE)
  assertInputTensor(tensor)
  return tensor
}

export async function classify(image: HTMLImageElement): Promise<Prediction> {
  const session = await getSession()
  const input = buildInputTensor(image)
  const inputName = session.inputNames[0]
  const outputName = session.outputNames[0]
  const output = await session.run({ [inputName]: input })
  const logits = extractLogits(output[outputName])

  const probs = softmax(logits)
  let bestIdx = 0
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[bestIdx]) bestIdx = i
  }
  const probabilities = Object.fromEntries(
    CLASS_NAMES.map((c, i) => [c, probs[i]]),
  ) as Record<ClassName, number>
  const className = CLASS_NAMES[bestIdx]
  return {
    className,
    labelJa: CLASS_LABELS_JA[className],
    confidence: probs[bestIdx],
    probabilities,
  }
}
