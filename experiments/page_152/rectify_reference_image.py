#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rectify a scanned page image by detecting its outer border and applying
quadrilateral-to-rectangle warping.

This script is OpenCV-free and uses Pillow + NumPy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _fit_line_y_from_x(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    # Fit y = m*x + b
    m, b = np.polyfit(xs, ys, 1)
    return float(m), float(b)


def _fit_line_x_from_y(ys: np.ndarray, xs: np.ndarray) -> tuple[float, float]:
    # Fit x = m*y + b
    m, b = np.polyfit(ys, xs, 1)
    return float(m), float(b)


def _intersect_yx_lines(
    m_yx: float, b_yx: float, m_xy: float, b_xy: float
) -> tuple[float, float]:
    # y = m_yx*x + b_yx
    # x = m_xy*y + b_xy
    # => x = m_xy*(m_yx*x + b_yx) + b_xy
    denom = 1.0 - m_xy * m_yx
    if abs(denom) < 1e-9:
        raise ValueError("Degenerate line intersection")
    x = (m_xy * b_yx + b_xy) / denom
    y = m_yx * x + b_yx
    return float(x), float(y)


def _trim_black_borders(img: Image.Image, threshold: int = 8, pad: int = 2) -> Image.Image:
    arr = np.asarray(img)
    if arr.ndim == 3:
        mask = np.any(arr > threshold, axis=2)
    else:
        mask = arr > threshold
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return img

    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, arr.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, arr.shape[1])
    return img.crop((x0, y0, x1, y1))


def _trace_border_points(
    gray: np.ndarray, dark_thresh: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = gray.shape

    x_start, x_end = int(w * 0.03), int(w * 0.97)
    y_start, y_end = int(h * 0.03), int(h * 0.97)
    top_limit = int(h * 0.25)
    bot_start = int(h * 0.75)
    left_limit = int(w * 0.25)
    right_start = int(w * 0.75)

    top_xs, top_ys = [], []
    bot_xs, bot_ys = [], []
    left_ys, left_xs = [], []
    right_ys, right_xs = [], []

    # For each x, trace first dark pixel from top and last dark from bottom.
    for x in range(x_start, x_end, 2):
        col_top = gray[:top_limit, x]
        idx_top = np.where(col_top < dark_thresh)[0]
        if idx_top.size:
            top_xs.append(x)
            top_ys.append(int(idx_top[0]))

        col_bot = gray[bot_start:, x]
        idx_bot = np.where(col_bot < dark_thresh)[0]
        if idx_bot.size:
            bot_xs.append(x)
            bot_ys.append(bot_start + int(idx_bot[-1]))

    # For each y, trace first dark from left and last dark from right.
    for y in range(y_start, y_end, 2):
        row_left = gray[y, :left_limit]
        idx_left = np.where(row_left < dark_thresh)[0]
        if idx_left.size:
            left_ys.append(y)
            left_xs.append(int(idx_left[0]))

        row_right = gray[y, right_start:]
        idx_right = np.where(row_right < dark_thresh)[0]
        if idx_right.size:
            right_ys.append(y)
            right_xs.append(right_start + int(idx_right[-1]))

    return (
        np.asarray(top_xs),
        np.asarray(top_ys),
        np.asarray(bot_xs),
        np.asarray(bot_ys),
        np.asarray(left_ys),
        np.asarray(left_xs),
        np.asarray(right_ys),
        np.asarray(right_xs),
    )


def _robust_fit_y_from_x(xs: np.ndarray, ys: np.ndarray, n_iter: int = 5) -> tuple[float, float]:
    if len(xs) < 20:
        raise RuntimeError("Too few points for robust y=f(x) fit")
    mask = np.ones(len(xs), dtype=bool)
    m, b = 0.0, 0.0
    for _ in range(n_iter):
        m, b = _fit_line_y_from_x(xs[mask], ys[mask])
        resid = ys - (m * xs + b)
        scale = 1.4826 * np.median(np.abs(resid[mask])) + 1e-6
        new_mask = np.abs(resid) < 2.5 * scale
        if np.array_equal(new_mask, mask) or new_mask.sum() < max(20, int(0.25 * len(xs))):
            break
        mask = new_mask
    return m, b


def _robust_fit_x_from_y(ys: np.ndarray, xs: np.ndarray, n_iter: int = 5) -> tuple[float, float]:
    if len(ys) < 20:
        raise RuntimeError("Too few points for robust x=f(y) fit")
    mask = np.ones(len(ys), dtype=bool)
    m, b = 0.0, 0.0
    for _ in range(n_iter):
        m, b = _fit_line_x_from_y(ys[mask], xs[mask])
        resid = xs - (m * ys + b)
        scale = 1.4826 * np.median(np.abs(resid[mask])) + 1e-6
        new_mask = np.abs(resid) < 2.5 * scale
        if np.array_equal(new_mask, mask) or new_mask.sum() < max(20, int(0.25 * len(ys))):
            break
        mask = new_mask
    return m, b


def rectify_image(input_path: Path, output_path: Path, dark_quantile: float = 0.12) -> None:
    img = Image.open(input_path).convert("L")
    gray = np.asarray(img)
    h, w = gray.shape

    dark_thresh = int(np.quantile(gray, dark_quantile))
    (
        top_xs,
        top_ys,
        bot_xs,
        bot_ys,
        left_ys,
        left_xs,
        right_ys,
        right_xs,
    ) = _trace_border_points(gray, dark_thresh)

    # Keep side-specific quantiles so interior strokes do not dominate.
    top_keep = top_ys <= np.quantile(top_ys, 0.45)
    bot_keep = bot_ys >= np.quantile(bot_ys, 0.55)
    left_keep = left_xs <= np.quantile(left_xs, 0.45)
    right_keep = right_xs >= np.quantile(right_xs, 0.55)

    top_xs, top_ys = top_xs[top_keep], top_ys[top_keep]
    bot_xs, bot_ys = bot_xs[bot_keep], bot_ys[bot_keep]
    left_ys, left_xs = left_ys[left_keep], left_xs[left_keep]
    right_ys, right_xs = right_ys[right_keep], right_xs[right_keep]

    if min(len(top_xs), len(bot_xs), len(left_ys), len(right_ys)) < 50:
        raise RuntimeError(
            f"Not enough border points detected after filtering: "
            f"top={len(top_xs)}, bottom={len(bot_xs)}, left={len(left_ys)}, right={len(right_ys)}"
        )

    mt, bt = _robust_fit_y_from_x(top_xs, top_ys)
    mb, bb = _robust_fit_y_from_x(bot_xs, bot_ys)
    ml, bl = _robust_fit_x_from_y(left_ys, left_xs)
    mr, br = _robust_fit_x_from_y(right_ys, right_xs)

    tl = _intersect_yx_lines(mt, bt, ml, bl)
    tr = _intersect_yx_lines(mt, bt, mr, br)
    brc = _intersect_yx_lines(mb, bb, mr, br)
    blc = _intersect_yx_lines(mb, bb, ml, bl)

    # Clamp corners into image bounds for stability.
    def _clamp(pt: tuple[float, float]) -> tuple[float, float]:
        x = min(max(pt[0], 0.0), w - 1.0)
        y = min(max(pt[1], 0.0), h - 1.0)
        return x, y

    tl = _clamp(tl)
    tr = _clamp(tr)
    brc = _clamp(brc)
    blc = _clamp(blc)

    # Ensure corner ordering is consistent if fits flipped.
    if tl[0] > tr[0]:
        tl, tr = tr, tl
        blc, brc = brc, blc
    if tl[1] > blc[1]:
        tl, blc = blc, tl
        tr, brc = brc, tr

    # Estimate output size from opposite side lengths.
    width_top = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    width_bot = np.hypot(brc[0] - blc[0], brc[1] - blc[1])
    height_left = np.hypot(blc[0] - tl[0], blc[1] - tl[1])
    height_right = np.hypot(brc[0] - tr[0], brc[1] - tr[1])
    out_w = int(round(min(width_top, width_bot)))
    out_h = int(round(min(height_left, height_right)))

    # Keep output dimensions bounded to avoid accidental blow-ups.
    out_w = max(200, min(out_w, int(w * 1.2)))
    out_h = max(200, min(out_h, int(h * 1.2)))

    src_quad = [tl[0], tl[1], blc[0], blc[1], brc[0], brc[1], tr[0], tr[1]]
    rectified = Image.open(input_path).convert("RGB").transform(
        (out_w, out_h),
        Image.Transform.QUAD,
        data=src_quad,
        resample=Image.Resampling.BICUBIC,
    )
    rectified = _trim_black_borders(rectified, threshold=8, pad=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rectified.save(output_path)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Detected corners (TL, TR, BR, BL):")
    print(f"  TL: ({tl[0]:.1f}, {tl[1]:.1f})")
    print(f"  TR: ({tr[0]:.1f}, {tr[1]:.1f})")
    print(f"  BR: ({brc[0]:.1f}, {brc[1]:.1f})")
    print(f"  BL: ({blc[0]:.1f}, {blc[1]:.1f})")
    print(f"Output size: {out_w} x {out_h}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rectify scanned page border.")
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument("output", type=Path, help="Output image path")
    parser.add_argument(
        "--dark-quantile",
        type=float,
        default=0.12,
        help="Quantile used to choose dark-pixel threshold (default: 0.12)",
    )
    args = parser.parse_args()

    rectify_image(args.input, args.output, dark_quantile=args.dark_quantile)


if __name__ == "__main__":
    main()
