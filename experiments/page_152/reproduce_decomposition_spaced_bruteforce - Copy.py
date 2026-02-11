# -*- coding: utf-8 -*-
"""
Page 152: Brute-force spaced decomposition matching (dot-only outputs).

Searches per-filter parameters to better match the scanned reference plot:
  references/page_152/filter_decomposition.png

Constraints used for this brute-force pass:
  - Real-valued Ormsby only (analytic=False)
  - Sparse outputs only (spacing + offset decimation)
  - Dot plotting only (no interpolation lines)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.image as mpimg

from src.filters import ormsby_filter, apply_ormsby_filter
from src.filters.decimation import decimate_signal


# ============================================================================
# CONFIG
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "../../data/raw/^dji_w.csv")
REFERENCE_IMAGE = os.path.join(SCRIPT_DIR, "../../references/page_152/filter_decomposition.png")

DATE_START = "1921-04-29"
DATE_END = "1965-05-21"
DISPLAY_START = "1935-01-01"
DISPLAY_END = "1954-02-01"

FS = 52
TWOPI = 2 * np.pi
NYQ_RAD = np.pi * FS

# Approximate reference-plot geometry from the scanned figure.
# x-range: inner plotting frame
REF_X0 = 33
REF_X1 = 748
# y row boundaries of decomposition panels (1..6)
REF_ROW_BOUNDS = [256, 472, 564, 615, 645, 688, 706]

OUT_PLOT = "page152_bruteforce_dots_bestfit.png"
OUT_CSV = "page152_bruteforce_best_params.csv"


FILTER_SPECS = [
    {
        "type": "lp",
        "f_pass": 0.85,
        "f_stop": 1.25,
        "f_center": (0.85 + 1.25) / 2,
        "bandwidth": 1.25 - 0.85,
        "nw": 1393,
        "index": 0,
        "label": "LP-1: Trend (>5 yr)",
    },
    {
        "type": "bp",
        "f1": 0.85,
        "f2": 1.25,
        "f3": 2.05,
        "f4": 2.45,
        "f_center": (1.25 + 2.05) / 2,
        "bandwidth": 2.05 - 1.25,
        "nw": 1393,
        "index": 1,
        "label": "BP-2: ~3.8 yr",
    },
    {
        "type": "bp",
        "f1": 3.20,
        "f2": 3.55,
        "f3": 6.35,
        "f4": 6.70,
        "f_center": (3.55 + 6.35) / 2,
        "bandwidth": 6.35 - 3.55,
        "nw": 1245,
        "index": 2,
        "label": "BP-3: ~1.3 yr",
    },
    {
        "type": "bp",
        "f1": 7.25,
        "f2": 7.55,
        "f3": 9.55,
        "f4": 9.85,
        "f_center": (7.55 + 9.55) / 2,
        "bandwidth": 9.55 - 7.55,
        "nw": 1745,
        "index": 3,
        "label": "BP-4: ~0.7 yr",
    },
    {
        "type": "bp",
        "f1": 13.65,
        "f2": 13.95,
        "f3": 19.35,
        "f4": 19.65,
        "f_center": (13.95 + 19.35) / 2,
        "bandwidth": 19.35 - 13.95,
        "nw": 1299,
        "index": 4,
        "label": "BP-5: ~0.4 yr",
    },
    {
        "type": "bp",
        "f1": 28.45,
        "f2": 28.75,
        "f3": 35.95,
        "f4": 36.25,
        "f_center": (28.75 + 35.95) / 2,
        "bandwidth": 35.95 - 28.75,
        "nw": 1299,
        "index": 5,
        "label": "BP-6: ~0.2 yr",
    },
]


def _make_odd(n):
    n_int = max(3, int(round(n)))
    return n_int if (n_int % 2 == 1) else (n_int + 1)


def _moving_average(x, n=7):
    if n <= 1:
        return x
    w = np.ones(n, dtype=float) / n
    return np.convolve(x, w, mode="same")


def load_reference_targets():
    """Extract one normalized trace per decomposition row from reference image."""
    img = mpimg.imread(REFERENCE_IMAGE)
    if img.ndim == 3:
        gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    else:
        gray = img.astype(float)
    if gray.max() > 1.5:
        gray = gray / 255.0

    x0 = max(0, REF_X0)
    x1 = min(gray.shape[1] - 1, REF_X1)
    x_slice = slice(x0, x1 + 1)

    targets = []
    for i in range(6):
        y0 = REF_ROW_BOUNDS[i]
        y1 = REF_ROW_BOUNDS[i + 1]
        y0 = max(0, y0)
        y1 = min(gray.shape[0] - 1, y1)
        if y1 <= y0 + 4:
            raise ValueError(f"Invalid row bounds for row {i + 1}: {y0}, {y1}")

        region = 1.0 - gray[y0:y1, x_slice]
        # Remove persistent horizontal lines by subtracting row median darkness.
        region_hp = region - np.median(region, axis=1, keepdims=True)

        y_rel = np.argmax(region_hp, axis=0)
        conf = region_hp[y_rel, np.arange(region_hp.shape[1])]
        y_abs = y_rel.astype(float) + y0

        # Drop low-confidence picks, then fill.
        cutoff = np.percentile(conf, 45.0)
        y_abs[conf < cutoff] = np.nan

        x = np.arange(y_abs.size)
        good = np.isfinite(y_abs)
        if good.sum() < 2:
            y_fill = np.full_like(y_abs, (y0 + y1) / 2.0)
        else:
            y_fill = np.interp(x, x[good], y_abs[good])

        y_fill = _moving_average(y_fill, n=9)

        mid = 0.5 * (y0 + y1)
        half_h = max(1e-6, 0.5 * (y1 - y0))
        amp = (mid - y_fill) / half_h
        amp = amp - np.mean(amp)
        targets.append(amp)

    return targets


def load_display_data():
    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df_hurst = df[df.Date.between(DATE_START, DATE_END)].copy()

    dates = pd.to_datetime(df_hurst["Date"].values)
    close = df_hurst["Close"].values.astype(float)

    mask = (dates >= pd.to_datetime(DISPLAY_START)) & (dates <= pd.to_datetime(DISPLAY_END))
    idx = np.where(mask)[0]
    s_idx, e_idx = idx[0], idx[-1] + 1

    return dates[s_idx:e_idx], close[s_idx:e_idx]


def build_candidate_edges(spec, center_shift, bw_scale, skirt_scale):
    """Build LP/BP edges in rad/year from simple shift/scale controls."""
    if spec["type"] == "lp":
        bw0 = spec["f_stop"] - spec["f_pass"]
        c0 = 0.5 * (spec["f_pass"] + spec["f_stop"])
        c = c0 + center_shift
        bw = max(0.08, bw0 * bw_scale)
        f_pass = max(0.01, c - 0.5 * bw)
        f_stop = min(NYQ_RAD - 0.01, c + 0.5 * bw)
        if not (0 < f_pass < f_stop):
            return None
        return {"type": "lp", "f_pass": f_pass, "f_stop": f_stop}

    bw0 = spec["f3"] - spec["f2"]
    sk_l0 = spec["f2"] - spec["f1"]
    sk_r0 = spec["f4"] - spec["f3"]
    c0 = 0.5 * (spec["f2"] + spec["f3"])

    c = c0 + center_shift
    bw = max(0.08, bw0 * bw_scale)
    sk_l = max(0.04, sk_l0 * skirt_scale)
    sk_r = max(0.04, sk_r0 * skirt_scale)

    f2 = c - 0.5 * bw
    f3 = c + 0.5 * bw
    f1 = f2 - sk_l
    f4 = f3 + sk_r

    if not (0.01 < f1 < f2 <= f3 < (NYQ_RAD - 0.01) and f4 < (NYQ_RAD - 0.01)):
        return None
    return {"type": "bp", "f1": f1, "f2": f2, "f3": f3, "f4": f4}


def score_candidate(y_sparse, idx_sparse, target_trace, x_map):
    """Scale-fit candidate to target and return normalized RMSE score."""
    xpix = x_map[idx_sparse]
    t = target_trace[xpix]
    y = np.asarray(y_sparse, dtype=float)

    if len(y) < 8 or np.std(y) < 1e-9 or np.std(t) < 1e-9:
        return np.inf, 0.0, 0.0

    X = np.column_stack([y, np.ones_like(y)])
    a, b = np.linalg.lstsq(X, t, rcond=None)[0]
    pred = a * y + b
    rmse = np.sqrt(np.mean((pred - t) ** 2))
    nrmse = rmse / (np.std(t) + 1e-9)

    corr = np.corrcoef(y, t)[0, 1]
    if not np.isfinite(corr):
        corr = 0.0

    # Slightly reward correlation magnitude.
    score = nrmse - 0.05 * abs(corr)
    return score, a, b


def filter_output_for_candidate(signal_disp, spec, edge_spec, spacing, offset):
    """Run one candidate: decimate, filter real-valued, return sparse output."""
    signal_dec, idx_sparse = decimate_signal(signal_disp, spacing=spacing, offset=offset)
    fs_dec = FS / spacing
    nyq_dec = np.pi * fs_dec

    if edge_spec["type"] == "lp":
        if edge_spec["f_stop"] >= nyq_dec:
            return None
    else:
        if edge_spec["f4"] >= nyq_dec:
            return None

    nw_dec = _make_odd(spec["nw"] / spacing)

    if edge_spec["type"] == "lp":
        f_edges_cyc = np.array([edge_spec["f_pass"], edge_spec["f_stop"]]) / TWOPI
        h = ormsby_filter(
            nw=nw_dec,
            f_edges=f_edges_cyc,
            fs=fs_dec,
            filter_type="lp",
            analytic=False,
        )
    else:
        f_edges_cyc = np.array([edge_spec["f1"], edge_spec["f2"], edge_spec["f3"], edge_spec["f4"]]) / TWOPI
        h = ormsby_filter(
            nw=nw_dec,
            f_edges=f_edges_cyc,
            fs=fs_dec,
            filter_type="bp",
            method="modulate",
            analytic=False,
        )

    out = apply_ormsby_filter(signal_dec, h, mode="reflect", fs=fs_dec)
    y = out["signal"]
    if np.iscomplexobj(y):
        y = y.real
    return y, idx_sparse


def search_best_for_filter(signal_disp, spec, target_trace, x_map):
    """Two-stage search for one filter."""
    label = spec["label"]
    print(f"Searching {label} ...")

    # Coarse ranges.
    if label == "BP-2: ~3.8 yr":
        spacings = [5, 6, 7, 8]
    elif spec["type"] == "lp":
        spacings = [4, 5, 6, 7]
    else:
        spacings = [2, 3, 4, 5]

    center_shifts = [-0.40, -0.20, 0.00, 0.20, 0.40]
    bw_scales = [0.85, 1.00, 1.15]
    skirt_scales = [0.85, 1.00, 1.15]

    best = {
        "score": np.inf,
        "spacing": None,
        "offset": None,
        "edge_spec": None,
        "center_shift": None,
        "bw_scale": None,
        "skirt_scale": None,
        "fit_a": None,
        "fit_b": None,
        "signal_sparse": None,
        "indices_sparse": None,
    }

    eval_count = 0
    for spacing in spacings:
        for offset in range(1, spacing + 1):
            for center_shift in center_shifts:
                for bw_scale in bw_scales:
                    for skirt_scale in skirt_scales:
                        if spec["type"] == "lp":
                            skirt_scale_use = 1.0
                        else:
                            skirt_scale_use = skirt_scale

                        edge_spec = build_candidate_edges(
                            spec=spec,
                            center_shift=center_shift,
                            bw_scale=bw_scale,
                            skirt_scale=skirt_scale_use,
                        )
                        if edge_spec is None:
                            continue

                        result = filter_output_for_candidate(
                            signal_disp=signal_disp,
                            spec=spec,
                            edge_spec=edge_spec,
                            spacing=spacing,
                            offset=offset,
                        )
                        eval_count += 1
                        if result is None:
                            continue
                        y_sparse, idx_sparse = result
                        score, a, b = score_candidate(y_sparse, idx_sparse, target_trace, x_map)
                        if score < best["score"]:
                            best.update(
                                {
                                    "score": float(score),
                                    "spacing": int(spacing),
                                    "offset": int(offset),
                                    "edge_spec": edge_spec,
                                    "center_shift": float(center_shift),
                                    "bw_scale": float(bw_scale),
                                    "skirt_scale": float(skirt_scale_use),
                                    "fit_a": float(a),
                                    "fit_b": float(b),
                                    "signal_sparse": y_sparse.copy(),
                                    "indices_sparse": idx_sparse.copy(),
                                }
                            )

    # Local refinement around coarse best.
    if best["spacing"] is not None:
        spacing0 = best["spacing"]
        spacing_ref = sorted(set([max(2, spacing0 - 1), spacing0, spacing0 + 1]))
        shift0 = best["center_shift"]
        bw0 = best["bw_scale"]
        sk0 = best["skirt_scale"]

        center_ref = sorted(set(np.round(np.arange(shift0 - 0.20, shift0 + 0.201, 0.10), 3)))
        bw_ref = sorted(set(np.round(np.arange(bw0 - 0.10, bw0 + 0.101, 0.05), 3)))
        sk_ref = sorted(set(np.round(np.arange(sk0 - 0.10, sk0 + 0.101, 0.05), 3)))

        for spacing in spacing_ref:
            for offset in range(1, spacing + 1):
                for center_shift in center_ref:
                    for bw_scale in bw_ref:
                        for skirt_scale in sk_ref:
                            if spec["type"] == "lp":
                                skirt_scale_use = 1.0
                            else:
                                skirt_scale_use = skirt_scale

                            edge_spec = build_candidate_edges(
                                spec=spec,
                                center_shift=center_shift,
                                bw_scale=bw_scale,
                                skirt_scale=skirt_scale_use,
                            )
                            if edge_spec is None:
                                continue

                            result = filter_output_for_candidate(
                                signal_disp=signal_disp,
                                spec=spec,
                                edge_spec=edge_spec,
                                spacing=spacing,
                                offset=offset,
                            )
                            eval_count += 1
                            if result is None:
                                continue

                            y_sparse, idx_sparse = result
                            score, a, b = score_candidate(y_sparse, idx_sparse, target_trace, x_map)
                            if score < best["score"]:
                                best.update(
                                    {
                                        "score": float(score),
                                        "spacing": int(spacing),
                                        "offset": int(offset),
                                        "edge_spec": edge_spec,
                                        "center_shift": float(center_shift),
                                        "bw_scale": float(bw_scale),
                                        "skirt_scale": float(skirt_scale_use),
                                        "fit_a": float(a),
                                        "fit_b": float(b),
                                        "signal_sparse": y_sparse.copy(),
                                        "indices_sparse": idx_sparse.copy(),
                                    }
                                )

    print(
        f"  done ({eval_count} evals): score={best['score']:.4f}, "
        f"spacing={best['spacing']}, offset={best['offset']}"
    )
    return best


def save_best_params_csv(best_results):
    rows = []
    for spec, best in zip(FILTER_SPECS, best_results):
        row = {
            "label": spec["label"],
            "type": spec["type"],
            "score": best["score"],
            "spacing": best["spacing"],
            "offset": best["offset"],
            "center_shift": best["center_shift"],
            "bw_scale": best["bw_scale"],
            "skirt_scale": best["skirt_scale"],
        }
        es = best["edge_spec"]
        if es["type"] == "lp":
            row["f_pass"] = es["f_pass"]
            row["f_stop"] = es["f_stop"]
            row["f1"] = np.nan
            row["f2"] = np.nan
            row["f3"] = np.nan
            row["f4"] = np.nan
        else:
            row["f_pass"] = np.nan
            row["f_stop"] = np.nan
            row["f1"] = es["f1"]
            row["f2"] = es["f2"]
            row["f3"] = es["f3"]
            row["f4"] = es["f4"]
        rows.append(row)

    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(SCRIPT_DIR, OUT_CSV)
    df_out.to_csv(out_csv, index=False)
    print(f"Saved best params: {out_csv}")


def plot_best_dot_outputs(dates_disp, best_results):
    fig, axes = plt.subplots(6, 1, figsize=(16, 13), sharex=True)

    for ax, spec, best in zip(axes, FILTER_SPECS, best_results):
        idx = best["indices_sparse"]
        y = best["signal_sparse"]
        ax.plot(dates_disp[idx], y, linestyle="None", marker=".", color="k", markersize=2.2)
        ax.axhline(0.0, color="gray", linewidth=0.4)
        ax.grid(True, alpha=0.2)
        ax.set_ylabel(spec["label"], fontsize=8, rotation=0, labelpad=75, ha="left")
        ax.tick_params(axis="y", labelsize=7)

        es = best["edge_spec"]
        if es["type"] == "lp":
            txt = (
                f"s={best['spacing']} o={best['offset']}  "
                f"[{es['f_pass']:.2f}, {es['f_stop']:.2f}]"
            )
        else:
            txt = (
                f"s={best['spacing']} o={best['offset']}  "
                f"[{es['f1']:.2f}, {es['f2']:.2f}, {es['f3']:.2f}, {es['f4']:.2f}]"
            )
        ax.text(
            0.99,
            0.90,
            txt,
            transform=ax.transAxes,
            fontsize=7,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Date")
    plt.xticks(rotation=45)

    fig.suptitle(
        "Page 152 Brute-Force Match (Dot-Only, Real Ormsby, No Interpolation)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    out_plot = os.path.join(SCRIPT_DIR, OUT_PLOT)
    fig.savefig(out_plot, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_plot}")
    return fig


def main():
    print("=" * 72)
    print("Page 152 Brute-Force Decomposition Match (Dot-Only)")
    print("=" * 72)
    print(f"Reference image: {REFERENCE_IMAGE}")

    targets = load_reference_targets()
    dates_disp, signal_disp = load_display_data()

    w_target = len(targets[0])
    x_map = np.round(np.linspace(0, w_target - 1, len(signal_disp))).astype(int)

    best_results = []
    for spec, target in zip(FILTER_SPECS, targets):
        best = search_best_for_filter(
            signal_disp=signal_disp, spec=spec, target_trace=target, x_map=x_map
        )
        best_results.append(best)

    print()
    print("Best-fit summary:")
    for spec, best in zip(FILTER_SPECS, best_results):
        es = best["edge_spec"]
        if es["type"] == "lp":
            edge_txt = f"[{es['f_pass']:.2f}, {es['f_stop']:.2f}]"
        else:
            edge_txt = f"[{es['f1']:.2f}, {es['f2']:.2f}, {es['f3']:.2f}, {es['f4']:.2f}]"
        print(
            f"  {spec['label']:<20s}  score={best['score']:.4f}  "
            f"s={best['spacing']}  o={best['offset']}  edges={edge_txt}"
        )

    save_best_params_csv(best_results)
    plot_best_dot_outputs(dates_disp, best_results)
    plt.show()


if __name__ == "__main__":
    main()
