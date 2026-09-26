/**
 * Pure preprocessing helpers, kept free of any canvas/DOM dependency so they
 * can be unit tested directly.
 *
 * The pipeline mirrors Ultralytics' classification inference preprocessing
 * (as used by `training/scripts/train.py`'s YOLO evaluation and export):
 * resize the short side to `size` while keeping the aspect ratio, then take
 * a centered `size x size` crop, normalize to 0-1, and lay out as NCHW.
 *
 * Resizing the short side to `size` and then center-cropping `size x size`
 * is mathematically equivalent to first taking a centered square crop of
 * side `min(width, height)` from the original image and then scaling that
 * square to `size x size`. `centerCropBox` returns that square directly (in
 * source-image coordinates) so the caller can do the crop-and-scale in a
 * single `drawImage` call.
 */

export interface CropBox {
  /** Left edge of the square crop, in source-image pixel coordinates. */
  sx: number
  /** Top edge of the square crop, in source-image pixel coordinates. */
  sy: number
  /** Side length of the square crop, in source-image pixel coordinates. */
  sSize: number
}

/**
 * Computes the centered square region of `width x height` that corresponds
 * to "resize short side to `size`, then center-crop `size x size`".
 *
 * `size` only needs to be a positive number for this calculation (the crop
 * box itself does not depend on it, since the equivalence above holds for
 * any target size); it is still required so callers can't accidentally
 * request a crop for an invalid target size.
 */
export function centerCropBox(width: number, height: number, size: number): CropBox {
  if (!Number.isFinite(width) || width <= 0) {
    throw new Error(`invalid width: ${String(width)}`)
  }
  if (!Number.isFinite(height) || height <= 0) {
    throw new Error(`invalid height: ${String(height)}`)
  }
  if (!Number.isFinite(size) || size <= 0) {
    throw new Error(`invalid size: ${String(size)}`)
  }
  const sSize = Math.min(width, height)
  const sx = (width - sSize) / 2
  const sy = (height - sSize) / 2
  return { sx, sy, sSize }
}

/**
 * Converts an RGBA pixel buffer (as returned by `CanvasRenderingContext2D
 * .getImageData`) already cropped/resized to `width x height` into a
 * normalized NCHW `Float32Array` of shape `[3, height, width]` (no batch
 * dimension; callers add that when building the ONNX tensor).
 *
 * Values are scaled to 0-1 and channel order is R, G, B — matching the
 * Ultralytics classification model's expected input.
 */
export function toTensorData(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
): Float32Array {
  const plane = width * height
  if (rgba.length !== plane * 4) {
    throw new Error(
      `rgba length ${String(rgba.length)} does not match ${String(width)}x${String(height)} (expected ${String(plane * 4)})`,
    )
  }
  const out = new Float32Array(plane * 3)
  for (let i = 0; i < plane; i++) {
    const offset = i * 4
    out[i] = (rgba[offset] ?? 0) / 255
    out[i + plane] = (rgba[offset + 1] ?? 0) / 255
    out[i + 2 * plane] = (rgba[offset + 2] ?? 0) / 255
  }
  return out
}
