#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive Page 152 filter tuner.

Desktop UI (Tkinter + Matplotlib) for tuning filter specs live:
  - Select filter 1..6 from a dropdown
  - Tune center, bandwidth, and skirt width for bandpass filters
  - Tune pass/stop edges for LP filter 1
  - Apply one-shot global adjustments to all filters together
  - Redraw unified decomposition layout after each change
"""

import copy
import os
import sys
import tkinter as tk
from tkinter import ttk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.dates as mdates
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from src.filters import ormsby_filter, apply_ormsby_filter
from src.filters.decimation import decimate_signal


FS = 52
TWOPI = 2 * np.pi

DATE_START = "1921-04-29"
DATE_END = "1965-05-21"
DATE_START = "1900-04-29"
DATE_END = "2020-05-21"

DISPLAY_START = "1934-12-25"
DISPLAY_END = "1952-01-28"

PLOT_TOP = 600
PLOT_BOTTOM = 0
PRICE_CENTER = 369
PRICE_HEIGHT = 259
BP_CENTERS = {
    "bp2": 234,
    "bp3": 167,
    "bp4": 128,
    "bp5": 105,
    "bp6": 88,
}
BP_NAMES = ["bp2", "bp3", "bp4", "bp5", "bp6"]
BP_LABELS = ["2", "3", "4", "5", "6"]
BP_SCALE = 1.55
X_PAD_LEFT_WEEKS = 94
X_PAD_RIGHT_WEEKS = 55

FILTER_SPECS = [
    {
        "type": "lp",
        "f_pass": 0.85,
        "f_stop": 1.25,
        "nw": 1393,
        "spacing": 1,
        "startidx": 0,
        "index": 0,
        "label": "1",
    },
    {
        "type": "bp",
        "f1": 0.85,
        "f2": 1.25,
        "f3": 2.05,
        "f4": 2.45,
        "nw": 1393,
        "spacing": 1,
        "startidx": 0,
        "index": 1,
        "label": "2",
    },
    {
        "type": "bp",
        "f1": 3.20,
        "f2": 3.55,
        "f3": 6.35,
        "f4": 6.70,
        "nw": 1245,
        "spacing": 1,
        "startidx": 0,
        "index": 2,
        "label": "3",
    },
    {
        "type": "bp",
        "f1": 7.25,
        "f2": 7.55,
        "f3": 9.55,
        "f4": 9.85,
        "nw": 1745,
        "spacing": 1,
        "startidx": 0,
        "index": 3,
        "label": "4",
    },
    {
        "type": "bp",
        "f1": 13.65,
        "f2": 13.95,
        "f3": 19.35,
        "f4": 19.65,
        "nw": 1299,
        "spacing": 1,
        "startidx": 0,
        "index": 4,
        "label": "5",
    },
    {
        "type": "bp",
        "f1": 28.45,
        "f2": 28.75,
        "f3": 35.95,
        "f4": 36.25,
        "nw": 1299,
        "spacing": 1,
        "startidx": 0,
        "index": 5,
        "label": "6",
    },
]


def _find_reference_image(script_dir: str) -> str | None:
    candidates = [
        os.path.join(script_dir, "../../references/page_152/filter_decomposition_v4_greyscale.png"),
        os.path.join(script_dir, "../../references/page_152/filter_decomposition_v2 copy_rectified.png"),
        os.path.join(script_dir, "../../references/page_152/filter_decomposition_v2 copy.png"),
        os.path.join(script_dir, "../../references/page_152/filter_decomposition_v2.png"),
    ]
    for path in candidates:
        full = os.path.abspath(path)
        if os.path.exists(full):
            return full
    return None


class Page152TunerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Page 152 Interactive Filter Tuner")
        self.root.geometry("1680x980")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.abspath(os.path.join(script_dir, "../../data/raw/^dji_w.csv"))
        self.ref_image_path = _find_reference_image(script_dir)
        self.ref_img = mpimg.imread(self.ref_image_path) if self.ref_image_path else None

        self.filter_specs = copy.deepcopy(FILTER_SPECS)
        self.outputs = [None] * len(self.filter_specs)
        self._pending_after_id = None
        self._updating_controls = False

        self._load_data()
        self._compute_all_outputs()
        self.bp_max_amp = self._compute_bp_max_amp()

        self._build_ui()
        self._load_selected_filter_into_controls()
        self._draw_plot()

    def _load_data(self) -> None:
        df = pd.read_csv(self.csv_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[df.Date.between(DATE_START, DATE_END)].copy()

        self.close_prices = df.Close.values.astype(float)
        self.dates = pd.to_datetime(df.Date.values)
        self.n_points = len(self.close_prices)

        mask = (self.dates >= pd.to_datetime(DISPLAY_START)) & (self.dates <= pd.to_datetime(DISPLAY_END))
        disp_idx = np.where(mask)[0]
        self.s_idx, self.e_idx = disp_idx[0], disp_idx[-1] + 1
        self.disp_dates = self.dates[self.s_idx : self.e_idx]
        self.disp_prices = self.close_prices[self.s_idx : self.e_idx]

        self.xlim_left = self.disp_dates[0] - pd.Timedelta(weeks=X_PAD_LEFT_WEEKS)
        self.xlim_right = self.disp_dates[-1] + pd.Timedelta(weeks=X_PAD_RIGHT_WEEKS)

    def _build_ui(self) -> None:
        controls = ttk.Frame(self.root, padding=8)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        plot_area = ttk.Frame(self.root, padding=8)
        plot_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(controls, text="Filter").pack(anchor="w")
        self.filter_var = tk.StringVar(value="1")
        self.filter_combo = ttk.Combobox(
            controls,
            textvariable=self.filter_var,
            values=[str(i) for i in range(1, 7)],
            width=10,
            state="readonly",
        )
        self.filter_combo.pack(anchor="w", pady=(0, 8))
        self.filter_combo.bind("<<ComboboxSelected>>", self._on_filter_changed)

        self.status_var = tk.StringVar(value="")
        self.selected_spec_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self.selected_spec_var, justify="left").pack(anchor="w", pady=(0, 10))

        self.param1_label_var = tk.StringVar(value="Center (rad/yr)")
        self.param2_label_var = tk.StringVar(value="Bandwidth (rad/yr)")
        self.param3_label_var = tk.StringVar(value="Skirt Width (rad/yr)")

        ttk.Label(controls, textvariable=self.param1_label_var).pack(anchor="w")
        self.param1_var = tk.DoubleVar(value=1.0)
        self.param1_scale = tk.Scale(
            controls,
            from_=0.05,
            to=60.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=300,
            variable=self.param1_var,
            command=lambda _v: self._on_param_changed(),
        )
        self.param1_scale.pack(anchor="w")

        ttk.Label(controls, textvariable=self.param2_label_var).pack(anchor="w")
        self.param2_var = tk.DoubleVar(value=1.0)
        self.param2_scale = tk.Scale(
            controls,
            from_=0.05,
            to=60.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=300,
            variable=self.param2_var,
            command=lambda _v: self._on_param_changed(),
        )
        self.param2_scale.pack(anchor="w")

        ttk.Label(controls, textvariable=self.param3_label_var).pack(anchor="w")
        self.param3_var = tk.DoubleVar(value=1.0)
        self.param3_scale = tk.Scale(
            controls,
            from_=0.01,
            to=20.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=300,
            variable=self.param3_var,
            command=lambda _v: self._on_param_changed(),
        )
        self.param3_scale.pack(anchor="w")

        all_frame = ttk.LabelFrame(controls, text="Adjust All Filters (One Shot)", padding=6)
        all_frame.pack(anchor="w", fill=tk.X, pady=(8, 8))

        ttk.Label(all_frame, text="BP center shift (rad/yr)").pack(anchor="w")
        self.all_bp_center_shift_var = tk.DoubleVar(value=0.0)
        tk.Scale(
            all_frame,
            from_=-8.0,
            to=8.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=300,
            variable=self.all_bp_center_shift_var,
        ).pack(anchor="w")

        ttk.Label(all_frame, text="BP bandwidth scale").pack(anchor="w")
        self.all_bp_bandwidth_scale_var = tk.DoubleVar(value=1.0)
        tk.Scale(
            all_frame,
            from_=0.50,
            to=2.00,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=300,
            variable=self.all_bp_bandwidth_scale_var,
        ).pack(anchor="w")

        ttk.Label(all_frame, text="BP skirt scale").pack(anchor="w")
        self.all_bp_skirt_scale_var = tk.DoubleVar(value=1.0)
        tk.Scale(
            all_frame,
            from_=0.50,
            to=2.50,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=300,
            variable=self.all_bp_skirt_scale_var,
        ).pack(anchor="w")

        ttk.Label(all_frame, text="LP pass-edge shift (rad/yr)").pack(anchor="w")
        self.all_lp_pass_shift_var = tk.DoubleVar(value=0.0)
        tk.Scale(
            all_frame,
            from_=-2.0,
            to=2.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=300,
            variable=self.all_lp_pass_shift_var,
        ).pack(anchor="w")

        ttk.Label(all_frame, text="LP stop-edge shift (rad/yr)").pack(anchor="w")
        self.all_lp_stop_shift_var = tk.DoubleVar(value=0.0)
        tk.Scale(
            all_frame,
            from_=-2.0,
            to=2.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=300,
            variable=self.all_lp_stop_shift_var,
        ).pack(anchor="w")

        ttk.Button(all_frame, text="Apply to all filters", command=self._apply_all_filter_controls).pack(
            anchor="w", pady=(6, 2)
        )
        ttk.Button(all_frame, text="Reset all-adjust sliders", command=self._reset_all_adjustment_controls).pack(
            anchor="w", pady=(0, 2)
        )

        self.overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls,
            text="Show reference overlay",
            variable=self.overlay_var,
            command=self._draw_plot,
        ).pack(anchor="w", pady=(8, 4))

        self.autoscale_bp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls,
            text="Auto recompute BP scale range",
            variable=self.autoscale_bp_var,
            command=self._draw_plot,
        ).pack(anchor="w", pady=(0, 8))

        ttk.Button(controls, text="Print specs to console", command=self._print_specs).pack(anchor="w", pady=(4, 4))
        ttk.Button(controls, text="Reset selected filter", command=self._reset_selected_filter).pack(anchor="w", pady=(0, 8))
        ttk.Button(controls, text="Reset all filters", command=self._reset_all_filters).pack(anchor="w", pady=(0, 8))
        ttk.Label(controls, textvariable=self.status_var, justify="left", foreground="#9a0000").pack(anchor="w")

        self.figure = Figure(figsize=(13, 8), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_area)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_area)
        toolbar.update()

    def _selected_idx(self) -> int:
        return int(self.filter_var.get()) - 1

    @staticmethod
    def _derive_bp_edges(
        center: float, bandwidth: float, skirt: float, nyq_rad: float
    ) -> tuple[float, float, float, float] | None:
        center = max(0.02, float(center))
        bandwidth = max(0.02, float(bandwidth))
        skirt = max(0.01, float(skirt))

        f2 = max(0.01, center - bandwidth / 2.0)
        f3 = f2 + bandwidth
        f1 = max(0.001, f2 - skirt)
        f4 = f3 + skirt

        if f4 > nyq_rad:
            shift = f4 - nyq_rad
            f1 = max(0.001, f1 - shift)
            f2 = max(f1 + 0.01, f2 - shift)
            f3 = max(f2 + 0.01, f3 - shift)
            f4 = nyq_rad

        if not (f1 < f2 <= f3 < f4):
            return None
        return f1, f2, f3, f4

    def _compute_filter_output(self, spec: dict) -> np.ndarray:
        spacing = int(spec.get("spacing", 1))
        startidx = int(spec.get("startidx", 0))
        fs_eff = FS / spacing if spacing > 1 else FS

        nw = int(spec["nw"])
        if spacing > 1:
            nw = int(nw / spacing)
            if nw % 2 == 0:
                nw += 1

        if spec["type"] == "lp":
            f_edges = np.array([spec["f_pass"], spec["f_stop"]], dtype=float) / TWOPI
            h = ormsby_filter(nw=nw, f_edges=f_edges, fs=fs_eff, filter_type="lp", analytic=False)
        else:
            f_edges = np.array([spec["f1"], spec["f2"], spec["f3"], spec["f4"]], dtype=float) / TWOPI
            h = ormsby_filter(
                nw=nw,
                f_edges=f_edges,
                fs=fs_eff,
                filter_type="bp",
                method="modulate",
                analytic=False,
            )

        if spacing > 1:
            decimated, indices = decimate_signal(self.close_prices, spacing, startidx + 1)
            result = apply_ormsby_filter(decimated, h, mode="reflect", fs=fs_eff)
            full_output = np.full(self.n_points, np.nan, dtype=float)
            signal_out = result["signal"]
            for out_idx, orig_idx in enumerate(indices):
                if out_idx < len(signal_out):
                    full_output[orig_idx] = signal_out[out_idx]
            return full_output

        result = apply_ormsby_filter(self.close_prices, h, mode="reflect", fs=FS)
        return result["signal"].astype(float)

    def _compute_all_outputs(self) -> None:
        for i, spec in enumerate(self.filter_specs):
            self.outputs[i] = self._compute_filter_output(spec)

    def _compute_bp_max_amp(self) -> float:
        bp_max_amp = 0.0
        for i in range(1, len(self.outputs)):
            sig = self.outputs[i][self.s_idx : self.e_idx]
            valid = sig[~np.isnan(sig)] if np.any(np.isnan(sig)) else sig
            if len(valid):
                bp_max_amp = max(bp_max_amp, float(np.max(np.abs(valid))))
        return bp_max_amp

    def _on_filter_changed(self, _event=None) -> None:
        self._load_selected_filter_into_controls()

    def _load_selected_filter_into_controls(self) -> None:
        idx = self._selected_idx()
        spec = self.filter_specs[idx]
        self._updating_controls = True
        try:
            if spec["type"] == "bp":
                center = (spec["f2"] + spec["f3"]) / 2.0
                bandwidth = spec["f3"] - spec["f2"]
                skirt = ((spec["f2"] - spec["f1"]) + (spec["f4"] - spec["f3"])) / 2.0

                self.param1_label_var.set("Center (rad/yr)")
                self.param2_label_var.set("Bandwidth (rad/yr)")
                self.param3_label_var.set("Skirt Width (rad/yr)")
                self.param3_scale.configure(state=tk.NORMAL)

                self.param1_scale.configure(from_=0.10, to=60.0)
                self.param2_scale.configure(from_=0.02, to=20.0)
                self.param3_scale.configure(from_=0.01, to=12.0)

                self.param1_var.set(center)
                self.param2_var.set(max(0.02, bandwidth))
                self.param3_var.set(max(0.01, skirt))
            else:
                self.param1_label_var.set("Pass Edge f_pass (rad/yr)")
                self.param2_label_var.set("Stop Edge f_stop (rad/yr)")
                self.param3_label_var.set("Skirt Width (N/A for LP)")
                self.param3_scale.configure(state=tk.DISABLED)

                self.param1_scale.configure(from_=0.02, to=12.0)
                self.param2_scale.configure(from_=0.03, to=15.0)

                self.param1_var.set(spec["f_pass"])
                self.param2_var.set(spec["f_stop"])
                self.param3_var.set(0.5)
        finally:
            self._updating_controls = False

        self._refresh_selected_spec_text()
        self._draw_plot()

    def _refresh_selected_spec_text(self) -> None:
        spec = self.filter_specs[self._selected_idx()]
        if spec["type"] == "bp":
            txt = (
                f"Filter {spec['label']} (BP)\n"
                f"f1={spec['f1']:.3f}  f2={spec['f2']:.3f}\n"
                f"f3={spec['f3']:.3f}  f4={spec['f4']:.3f}\n"
                f"nw={spec['nw']} spacing={spec['spacing']} startidx={spec['startidx']}"
            )
        else:
            txt = (
                f"Filter {spec['label']} (LP)\n"
                f"f_pass={spec['f_pass']:.3f}  f_stop={spec['f_stop']:.3f}\n"
                f"nw={spec['nw']} spacing={spec['spacing']} startidx={spec['startidx']}"
            )
        self.selected_spec_var.set(txt)

    def _on_param_changed(self) -> None:
        if self._updating_controls:
            return
        if self._pending_after_id is not None:
            self.root.after_cancel(self._pending_after_id)
        self._pending_after_id = self.root.after(140, self._apply_current_controls)

    def _apply_current_controls(self) -> None:
        self._pending_after_id = None
        idx = self._selected_idx()
        spec = self.filter_specs[idx]
        fs_eff = FS / spec.get("spacing", 1)
        nyq_rad = np.pi * fs_eff * 0.999

        try:
            if spec["type"] == "bp":
                center = max(0.02, float(self.param1_var.get()))
                bandwidth = max(0.02, float(self.param2_var.get()))
                skirt = max(0.01, float(self.param3_var.get()))
                bp_edges = self._derive_bp_edges(center=center, bandwidth=bandwidth, skirt=skirt, nyq_rad=nyq_rad)
                if bp_edges is None:
                    self.status_var.set("Invalid BP edge ordering. Adjust sliders.")
                    return
                spec["f1"], spec["f2"], spec["f3"], spec["f4"] = bp_edges
            else:
                f_pass = max(0.02, float(self.param1_var.get()))
                f_stop = max(f_pass + 0.02, float(self.param2_var.get()))
                f_stop = min(f_stop, nyq_rad)
                f_pass = min(f_pass, f_stop - 0.02)

                spec["f_pass"] = f_pass
                spec["f_stop"] = f_stop

            self.outputs[idx] = self._compute_filter_output(spec)
            if self.autoscale_bp_var.get():
                self.bp_max_amp = self._compute_bp_max_amp()

            self.status_var.set("")
            self._refresh_selected_spec_text()
            self._draw_plot()
        except Exception as exc:
            self.status_var.set(f"Filter compute failed: {exc}")

    def _apply_all_filter_controls(self) -> None:
        bp_center_shift = float(self.all_bp_center_shift_var.get())
        bp_bandwidth_scale = max(0.50, float(self.all_bp_bandwidth_scale_var.get()))
        bp_skirt_scale = max(0.50, float(self.all_bp_skirt_scale_var.get()))
        lp_pass_shift = float(self.all_lp_pass_shift_var.get())
        lp_stop_shift = float(self.all_lp_stop_shift_var.get())

        try:
            updated_specs = copy.deepcopy(self.filter_specs)

            for spec in updated_specs:
                fs_eff = FS / spec.get("spacing", 1)
                nyq_rad = np.pi * fs_eff * 0.999

                if spec["type"] == "bp":
                    center = (spec["f2"] + spec["f3"]) / 2.0 + bp_center_shift
                    bandwidth = (spec["f3"] - spec["f2"]) * bp_bandwidth_scale
                    skirt = ((spec["f2"] - spec["f1"]) + (spec["f4"] - spec["f3"])) / 2.0 * bp_skirt_scale
                    bp_edges = self._derive_bp_edges(center=center, bandwidth=bandwidth, skirt=skirt, nyq_rad=nyq_rad)
                    if bp_edges is None:
                        self.status_var.set(f"Invalid all-filter BP update at filter {spec['label']}.")
                        return
                    spec["f1"], spec["f2"], spec["f3"], spec["f4"] = bp_edges
                else:
                    f_pass = max(0.02, spec["f_pass"] + lp_pass_shift)
                    f_stop = max(f_pass + 0.02, spec["f_stop"] + lp_stop_shift)
                    f_stop = min(f_stop, nyq_rad)
                    f_pass = min(f_pass, f_stop - 0.02)
                    spec["f_pass"] = f_pass
                    spec["f_stop"] = f_stop

            self.filter_specs = updated_specs
            self._compute_all_outputs()
            if self.autoscale_bp_var.get():
                self.bp_max_amp = self._compute_bp_max_amp()
            self._reset_all_adjustment_controls(set_status=False)
            self.status_var.set("Applied one-shot adjustment to all filters.")
            self._load_selected_filter_into_controls()
        except Exception as exc:
            self.status_var.set(f"All-filter adjustment failed: {exc}")

    def _reset_all_adjustment_controls(self, set_status: bool = True) -> None:
        self.all_bp_center_shift_var.set(0.0)
        self.all_bp_bandwidth_scale_var.set(1.0)
        self.all_bp_skirt_scale_var.set(1.0)
        self.all_lp_pass_shift_var.set(0.0)
        self.all_lp_stop_shift_var.set(0.0)
        if set_status:
            self.status_var.set("All-filter adjustment sliders reset.")

    def _reset_selected_filter(self) -> None:
        idx = self._selected_idx()
        self.filter_specs[idx] = copy.deepcopy(FILTER_SPECS[idx])
        self.outputs[idx] = self._compute_filter_output(self.filter_specs[idx])
        if self.autoscale_bp_var.get():
            self.bp_max_amp = self._compute_bp_max_amp()
        self._load_selected_filter_into_controls()

    def _reset_all_filters(self) -> None:
        self.filter_specs = copy.deepcopy(FILTER_SPECS)
        self._compute_all_outputs()
        if self.autoscale_bp_var.get():
            self.bp_max_amp = self._compute_bp_max_amp()
        self._reset_all_adjustment_controls(set_status=False)
        self.status_var.set("All filters reset to defaults.")
        self._load_selected_filter_into_controls()

    def _draw_plot(self) -> None:
        self.ax.clear()

        if self.overlay_var.get() and self.ref_img is not None:
            x_left = mdates.date2num(self.xlim_left)
            x_right = mdates.date2num(self.xlim_right)
            # Keep the overlay axis-aligned only (no rotation/skew transform).
            self.ax.imshow(
                self.ref_img,
                aspect="auto",
                alpha=0.6,
                extent=[x_left, x_right, PLOT_BOTTOM, PLOT_TOP],
                origin="upper",
                interpolation="nearest",
                zorder=0,
            )

        p = self.disp_prices
        p_min, p_max = np.min(p), np.max(p)
        p_bottom = PRICE_CENTER - PRICE_HEIGHT / 2
        p_scaled = (p - p_min) / (p_max - p_min) * PRICE_HEIGHT + p_bottom
        self.ax.plot(self.disp_dates, p_scaled, "k-", linewidth=1.0, zorder=3)

        lp_sig = self.outputs[0][self.s_idx : self.e_idx]
        lp_scaled = (lp_sig - p_min) / (p_max - p_min) * PRICE_HEIGHT + p_bottom
        self.ax.plot(self.disp_dates, lp_scaled, "b--", linewidth=1.3, alpha=0.8, zorder=2)
        self.ax.text(
            0.01,
            (PRICE_CENTER - PLOT_BOTTOM) / (PLOT_TOP - PLOT_BOTTOM),
            "1",
            transform=self.ax.transAxes,
            fontsize=10,
            ha="center",
            va="center",
            fontweight="bold",
            zorder=5,
        )

        for out_idx, bp_name, bp_label in zip(range(1, 6), BP_NAMES, BP_LABELS):
            center_y = BP_CENTERS[bp_name]
            sig = self.outputs[out_idx][self.s_idx : self.e_idx]
            sig_scaled = sig * BP_SCALE + center_y
            if np.any(np.isnan(sig)):
                valid = ~np.isnan(sig_scaled)
                self.ax.plot(self.disp_dates[valid], sig_scaled[valid], "b.", markersize=2, zorder=2)
            else:
                self.ax.plot(self.disp_dates, sig_scaled, "b-", linewidth=0.3, zorder=2)
            self.ax.axhline(center_y, color="gray", linewidth=0.3, linestyle="-", alpha=0.3)
            y_frac = (center_y - PLOT_BOTTOM) / (PLOT_TOP - PLOT_BOTTOM)
            self.ax.text(
                0.01,
                y_frac,
                bp_label,
                transform=self.ax.transAxes,
                fontsize=10,
                ha="center",
                va="center",
                fontweight="bold",
                zorder=5,
            )

        self.ax.set_xlabel("Date", fontsize=11, fontweight="bold")
        self.ax.set_title(
            "Page 152 Interactive Tuner: Six-Filter Structural Decomposition",
            fontsize=13,
            fontweight="bold",
            pad=14,
        )
        self.ax.set_xlim(self.xlim_left, self.xlim_right)
        self.ax.set_ylim(PLOT_BOTTOM, PLOT_TOP)
        self.ax.set_yticks([])
        self.ax.xaxis.set_major_locator(mdates.YearLocator(1))
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for lbl in self.ax.get_xticklabels():
            lbl.set_rotation(45)
            lbl.set_ha("right")
            lbl.set_fontsize(9)
        self.ax.grid(True, alpha=0.9, linestyle="-", linewidth=0.4, axis="x")
        self.ax.set_axisbelow(True)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _print_specs(self) -> None:
        print("FILTER_SPECS = [")
        for spec in self.filter_specs:
            if spec["type"] == "lp":
                line = (
                    f"    {{'type': 'lp', 'f_pass': {spec['f_pass']:.4f}, "
                    f"'f_stop': {spec['f_stop']:.4f}, 'nw': {spec['nw']}, "
                    f"'spacing': {spec['spacing']}, 'startidx': {spec['startidx']}, "
                    f"'index': {spec['index']}, 'label': '{spec['label']}'}}"
                )
            else:
                line = (
                    f"    {{'type': 'bp', 'f1': {spec['f1']:.4f}, 'f2': {spec['f2']:.4f}, "
                    f"'f3': {spec['f3']:.4f}, 'f4': {spec['f4']:.4f}, 'nw': {spec['nw']}, "
                    f"'spacing': {spec['spacing']}, 'startidx': {spec['startidx']}, "
                    f"'index': {spec['index']}, 'label': '{spec['label']}'}}"
                )
            print(line + ",")
        print("]")
        self.status_var.set("Current specs printed to terminal.")


def main() -> None:
    root = tk.Tk()
    Page152TunerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
