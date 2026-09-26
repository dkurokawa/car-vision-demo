import { describe, expect, it } from 'vitest'
import { centerCropBox, toTensorData } from './preprocess'

describe('centerCropBox', () => {
  it('returns the full image for a square input', () => {
    expect(centerCropBox(100, 100, 640)).toEqual({ sx: 0, sy: 0, sSize: 100 })
  })

  it('crops symmetric side margins for a landscape image', () => {
    // 200x100: short side is height (100), so the crop is a centered
    // 100x100 square with 50px trimmed off each side horizontally.
    expect(centerCropBox(200, 100, 640)).toEqual({ sx: 50, sy: 0, sSize: 100 })
  })

  it('crops symmetric top/bottom margins for a portrait image', () => {
    // 100x200: short side is width (100), so the crop trims 50px off the
    // top and bottom.
    expect(centerCropBox(100, 200, 640)).toEqual({ sx: 0, sy: 50, sSize: 100 })
  })

  it('does not depend on the target size (only on the aspect ratio)', () => {
    expect(centerCropBox(200, 100, 640)).toEqual(centerCropBox(200, 100, 224))
  })

  it('handles odd-sized dimensions without throwing', () => {
    const box = centerCropBox(201, 100, 640)
    expect(box.sSize).toBe(100)
    expect(box.sx).toBeCloseTo(50.5)
    expect(box.sy).toBe(0)
  })

  it.each([
    [0, 100, 640],
    [100, 0, 640],
    [100, 100, 0],
    [-1, 100, 640],
    [100, -1, 640],
    [100, 100, -1],
    [Number.NaN, 100, 640],
  ])('rejects invalid dimensions (%p, %p, %p)', (width, height, size) => {
    expect(() => centerCropBox(width, height, size)).toThrow()
  })
})

describe('toTensorData', () => {
  it('normalizes to 0-1 and lays out as planar R,G,B (NCHW without batch dim)', () => {
    // 2x1 image: pixel 0 = pure red, pixel 1 = pure green (alpha ignored).
    const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 128])
    const tensor = toTensorData(rgba, 2, 1)

    // R plane: [1, 0]
    expect(tensor[0]).toBeCloseTo(1)
    expect(tensor[1]).toBeCloseTo(0)
    // G plane: [0, 1]
    expect(tensor[2]).toBeCloseTo(0)
    expect(tensor[3]).toBeCloseTo(1)
    // B plane: [0, 0]
    expect(tensor[4]).toBeCloseTo(0)
    expect(tensor[5]).toBeCloseTo(0)
    expect(tensor.length).toBe(2 * 1 * 3)
  })

  it('scales mid-range values correctly', () => {
    const rgba = new Uint8ClampedArray([51, 102, 153, 255])
    const tensor = toTensorData(rgba, 1, 1)
    expect(tensor[0]).toBeCloseTo(51 / 255)
    expect(tensor[1]).toBeCloseTo(102 / 255)
    expect(tensor[2]).toBeCloseTo(153 / 255)
  })

  it('throws when the buffer length does not match width x height', () => {
    const rgba = new Uint8ClampedArray(4 * 3) // claims 3 pixels
    expect(() => toTensorData(rgba, 2, 1)).toThrow() // but declared as 2x1 = 2 pixels
  })
})
