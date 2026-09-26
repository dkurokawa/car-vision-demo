// vi.fn()/vi.mocked() mock objects trip @typescript-eslint/unbound-method
// (it can't tell the extracted "method" is a plain mock function, not one
// that reads `this`), so it's disabled for this whole mocking-heavy file.
/* eslint-disable @typescript-eslint/unbound-method */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { InferenceSession, Tensor } from 'onnxruntime-web'

vi.mock('onnxruntime-web', () => ({
  InferenceSession: { create: vi.fn() },
  Tensor: vi.fn(),
}))

import * as ort from 'onnxruntime-web'
import {
  CLASS_NAMES,
  assertInputTensor,
  assertSessionShape,
  extractLogits,
  getSession,
} from './classifier'

const create = vi.mocked(ort.InferenceSession.create)

beforeEach(() => {
  create.mockReset()
})

describe('CLASS_NAMES', () => {
  it('matches the training label order', () => {
    // Must stay in lockstep with training/scripts/train.py's CLASS_NAMES —
    // the model's output index i corresponds to CLASS_NAMES[i].
    expect(CLASS_NAMES).toEqual(['left_front', 'left_rear', 'right_front', 'right_rear'])
  })
})

const VALID_INPUT_METADATA = {
  name: 'images',
  isTensor: true,
  type: 'float32',
  shape: [1, 3, 640, 640],
} as const
const VALID_OUTPUT_METADATA = {
  name: 'output0',
  isTensor: true,
  type: 'float32',
  shape: [1, 4],
} as const

describe('assertSessionShape', () => {
  it('accepts a session with exactly 1 "images" input and 1 "output0" output', () => {
    expect(() => {
      assertSessionShape({ inputNames: ['images'], outputNames: ['output0'] })
    }).not.toThrow()
  })

  it('accepts matching input/output metadata when the runtime exposes it', () => {
    expect(() => {
      assertSessionShape({
        inputNames: ['images'],
        outputNames: ['output0'],
        inputMetadata: [VALID_INPUT_METADATA],
        outputMetadata: [VALID_OUTPUT_METADATA],
      })
    }).not.toThrow()
  })

  it('rejects a session with zero or multiple inputs', () => {
    expect(() => {
      assertSessionShape({ inputNames: [], outputNames: ['output0'] })
    }).toThrow(/入力数/)
    expect(() => {
      assertSessionShape({ inputNames: ['a', 'b'], outputNames: ['output0'] })
    }).toThrow(/入力数/)
  })

  it('rejects a session with zero or multiple outputs', () => {
    expect(() => {
      assertSessionShape({ inputNames: ['images'], outputNames: [] })
    }).toThrow(/出力数/)
  })

  it('rejects an input name other than the Ultralytics default "images"', () => {
    expect(() => {
      assertSessionShape({ inputNames: ['input'], outputNames: ['output0'] })
    }).toThrow(/入力名.*期待: "images".*実際: "input"/)
  })

  it('rejects an output name other than the Ultralytics default "output0"', () => {
    expect(() => {
      assertSessionShape({ inputNames: ['images'], outputNames: ['dense_1'] })
    }).toThrow(/出力名.*期待: "output0".*実際: "dense_1"/)
  })

  it('rejects input metadata that describes a non-tensor value', () => {
    expect(() => {
      assertSessionShape({
        inputNames: ['images'],
        outputNames: ['output0'],
        inputMetadata: [{ name: 'images', isTensor: false }],
      })
    }).toThrow(/入力.*Tensor ではありません/)
  })

  it('rejects output metadata that describes a non-tensor value', () => {
    expect(() => {
      assertSessionShape({
        inputNames: ['images'],
        outputNames: ['output0'],
        outputMetadata: [{ name: 'output0', isTensor: false }],
      })
    }).toThrow(/出力.*Tensor ではありません/)
  })

  it('rejects input metadata with the wrong dtype', () => {
    expect(() => {
      assertSessionShape({
        inputNames: ['images'],
        outputNames: ['output0'],
        inputMetadata: [{ ...VALID_INPUT_METADATA, type: 'uint8' }],
      })
    }).toThrow(/入力.*dtype.*期待: float32.*実際: uint8/)
  })

  it('rejects input metadata with a mismatched shape', () => {
    expect(() => {
      assertSessionShape({
        inputNames: ['images'],
        outputNames: ['output0'],
        inputMetadata: [{ ...VALID_INPUT_METADATA, shape: [1, 3, 224, 224] }],
      })
    }).toThrow(/入力.*shape.*期待: \[1,3,640,640\].*実際: \[1,3,224,224\]/)
  })

  it('rejects output metadata with the wrong dtype', () => {
    expect(() => {
      assertSessionShape({
        inputNames: ['images'],
        outputNames: ['output0'],
        outputMetadata: [{ ...VALID_OUTPUT_METADATA, type: 'int64' }],
      })
    }).toThrow(/出力.*dtype.*期待: float32.*実際: int64/)
  })

  it('rejects output metadata with a mismatched shape (e.g. a 1000-class head)', () => {
    expect(() => {
      assertSessionShape({
        inputNames: ['images'],
        outputNames: ['output0'],
        outputMetadata: [{ ...VALID_OUTPUT_METADATA, shape: [1, 1000] }],
      })
    }).toThrow(/出力.*shape.*期待: \[1,4\].*実際: \[1,1000\]/)
  })

  it('tolerates a symbolic (string) dimension, such as a dynamic batch size', () => {
    expect(() => {
      assertSessionShape({
        inputNames: ['images'],
        outputNames: ['output0'],
        inputMetadata: [{ ...VALID_INPUT_METADATA, shape: ['batch', 3, 640, 640] }],
      })
    }).not.toThrow()
  })

  it('does not require metadata to be present (falls back to runtime output validation)', () => {
    expect(() => {
      assertSessionShape({ inputNames: ['images'], outputNames: ['output0'] })
    }).not.toThrow()
  })
})

describe('assertInputTensor', () => {
  it('accepts the expected [1,3,640,640] float32 tensor', () => {
    const tensor = { type: 'float32', dims: [1, 3, 640, 640] } as unknown as Tensor
    expect(() => {
      assertInputTensor(tensor)
    }).not.toThrow()
  })

  it('rejects a non-float32 dtype', () => {
    const tensor = { type: 'int32', dims: [1, 3, 640, 640] } as unknown as Tensor
    expect(() => {
      assertInputTensor(tensor)
    }).toThrow(/float32/)
  })

  it('rejects a mismatched shape', () => {
    const tensor = { type: 'float32', dims: [1, 3, 224, 224] } as unknown as Tensor
    expect(() => {
      assertInputTensor(tensor)
    }).toThrow(/shape/)
  })
})

describe('extractLogits', () => {
  it('returns the underlying data for a valid [1,4] float32 output', () => {
    const data = new Float32Array([0.1, 0.2, 0.3, 0.4])
    const tensor = { type: 'float32', dims: [1, 4], data } as unknown as Tensor
    expect(extractLogits(tensor)).toBe(data)
  })

  it('rejects a non-float32 output dtype', () => {
    const tensor = {
      type: 'int64',
      dims: [1, 4],
      data: new BigInt64Array(4),
    } as unknown as Tensor
    expect(() => extractLogits(tensor)).toThrow(/float32/)
  })

  it('rejects an output shape that does not match the 4-class head', () => {
    const tensor = {
      type: 'float32',
      dims: [1, 1000],
      data: new Float32Array(1000),
    } as unknown as Tensor
    expect(() => extractLogits(tensor)).toThrow(/shape/)
  })
})

describe('getSession retry after failure', () => {
  it('discards a failed session promise so the next call retries model loading', async () => {
    const fakeSession = {
      inputNames: ['images'],
      outputNames: ['output0'],
    } as unknown as InferenceSession
    create.mockRejectedValueOnce(new Error('network error')).mockResolvedValueOnce(fakeSession)

    await expect(getSession()).rejects.toThrow('network error')
    expect(create).toHaveBeenCalledTimes(1)

    await expect(getSession()).resolves.toBe(fakeSession)
    expect(create).toHaveBeenCalledTimes(2)
  })
})
