# -*- coding: utf-8 -*-
"""
Export Filter Specs in Machine-Readable Format (JSON/CSV)

Exports:
  1. 6-filter Ormsby specs (from pipeline derivation)
  2. Nominal model — all confirmed lines from narrowband CMW
  3. Narrowband CMW parameters

All outputs go to data/processed/

Reference: prd/nominal_model_pipeline.md, Remaining Work item 4
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import numpy as np
import pandas as pd

from src.pipeline.derive_nominal_model import derive_nominal_model, load_data
from src.pipeline.comb_bank import (
    design_narrowband_cmw_bank, run_cmw_comb_bank, extract_lines_from_narrowband
)
from src.pipeline.filter_design import design_analysis_filters

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')


def main():
    print("=" * 70)
    print("EXPORT FILTER SPECS — Machine-Readable Formats")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Step 1: Run pipeline ---
    print("\nStep 1: Running weekly pipeline...")
    result = derive_nominal_model(
        symbol='djia', freq='weekly',
        start='1921-04-29', end='1965-05-21',
        verbose=False
    )
    w0 = result.w0
    print(f"  w0 = {w0:.4f} rad/yr")

    # --- Step 2: Design 6-filter specs ---
    print("\nStep 2: Designing 6-filter bank...")
    filters = design_analysis_filters(
        group_boundaries=result.group_boundaries,
        w0=result.w0, fs=result.fs
    )

    # Export Ormsby specs
    ormsby_export = []
    for spec in filters['ormsby_specs']:
        entry = {k: v for k, v in spec.items() if not callable(v)}
        ormsby_export.append(entry)

    path = os.path.join(OUT_DIR, 'filter_specs_ormsby.json')
    with open(path, 'w') as f:
        json.dump(ormsby_export, f, indent=2)
    print(f"  Saved: {path}")

    # Also as CSV
    df_ormsby = pd.DataFrame(ormsby_export)
    path = os.path.join(OUT_DIR, 'filter_specs_ormsby.csv')
    df_ormsby.to_csv(path, index=False, float_format='%.4f')
    print(f"  Saved: {path}")

    # --- Step 3: Weekly narrowband CMW lines ---
    print("\nStep 3: Weekly narrowband CMW...")
    weekly_data = load_data('djia', 'weekly', '1921-04-29', '1965-05-21')
    nb_weekly_params = design_narrowband_cmw_bank(
        w0=w0, max_N=34, fs=weekly_data['fs'],
        fwhm_factor=0.5, omega_min=0.5
    )
    log_weekly = np.log(weekly_data['close'])
    nb_weekly_result = run_cmw_comb_bank(log_weekly, weekly_data['fs'], nb_weekly_params, analytic=True)
    weekly_lines = extract_lines_from_narrowband(nb_weekly_result, w0)
    print(f"  Weekly confirmed: {len(weekly_lines)}")

    # --- Step 4: Daily narrowband CMW lines ---
    print("\nStep 4: Daily narrowband CMW (N=2..80)...")
    daily_data = load_data('djia', 'daily', '1921-04-29', '1965-05-21')
    nb_daily_params = design_narrowband_cmw_bank(
        w0=w0, max_N=80, fs=daily_data['fs'],
        fwhm_factor=0.5, omega_min=0.5
    )
    log_daily = np.log(daily_data['close'])
    nb_daily_result = run_cmw_comb_bank(log_daily, daily_data['fs'], nb_daily_params, analytic=True)
    daily_lines = extract_lines_from_narrowband(nb_daily_result, w0)
    print(f"  Daily confirmed: {len(daily_lines)}")

    # Export confirmed lines
    for label, lines in [('weekly_33lines', weekly_lines), ('daily_79lines', daily_lines)]:
        # JSON
        path = os.path.join(OUT_DIR, f'nominal_model_{label}.json')
        with open(path, 'w') as f:
            json.dump(lines, f, indent=2)
        print(f"  Saved: {path}")

        # CSV
        if lines:
            df = pd.DataFrame(lines)
            path = os.path.join(OUT_DIR, f'nominal_model_{label}.csv')
            df.to_csv(path, index=False, float_format='%.6f')
            print(f"  Saved: {path}")

    # --- Step 5: CMW parameters ---
    print("\nStep 5: Exporting CMW parameters...")

    # Clean numpy types for JSON serialization
    def clean_for_json(params):
        cleaned = []
        for p in params:
            entry = {}
            for k, v in p.items():
                if isinstance(v, (np.integer, np.floating)):
                    entry[k] = float(v)
                else:
                    entry[k] = v
            cleaned.append(entry)
        return cleaned

    path = os.path.join(OUT_DIR, 'narrowband_cmw_params.json')
    with open(path, 'w') as f:
        json.dump({
            'w0': float(w0),
            'fwhm_factor': 0.5,
            'weekly_params': clean_for_json(nb_weekly_params),
            'daily_params': clean_for_json(nb_daily_params),
        }, f, indent=2)
    print(f"  Saved: {path}")

    # --- Step 6: Combined summary ---
    summary = {
        'w0_rad_yr': float(w0),
        'w0_period_yr': float(2 * np.pi / w0),
        'source': 'DJIA 1921-1965 (Hurst baseline)',
        'ormsby_6filters': ormsby_export,
        'weekly_confirmed_lines': len(weekly_lines),
        'daily_confirmed_lines': len(daily_lines),
        'boundaries_used': [float(b) for b in filters['boundaries_used']],
    }
    path = os.path.join(OUT_DIR, 'pipeline_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {path}")

    # List all exports
    print("\n" + "=" * 70)
    print("ALL EXPORTS:")
    print("=" * 70)
    for fname in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"  {fname:<45} {size:>8} bytes")

    print("\nDone.")


if __name__ == '__main__':
    main()
