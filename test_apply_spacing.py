#!/usr/bin/env python3
"""Test filter application with spacing."""

import sys
import os
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from src.filters import ormsby_filter, apply_ormsby_filter
from src.filters.decimation import decimate_signal

# Load small test data
df = pd.read_csv('data/raw/^dji_w.csv')
df['Date'] = pd.to_datetime(df['Date'])
df_hurst = df[df.Date.between('1921-04-29', '1965-05-21')].copy()
close_prices = df_hurst.Close.values[:500]  # Use first 500 samples

print(f"Test data length: {len(close_prices)}")

# Create filter
nw = 199
fs = 52/7
f_edges = np.array([0.85, 1.25, 2.05, 2.45]) / (2*np.pi)

print(f"Creating filter (nw={nw}, fs={fs})...")
h = ormsby_filter(nw=nw, f_edges=f_edges, fs=fs,
                  filter_type='bp', method='modulate', analytic=False)
print(f"Filter created: {len(h)} taps")

# Decimate input
spacing = 7
startidx = 2
print(f"Decimating with spacing={spacing}, startidx={startidx}...")
decimated, indices = decimate_signal(close_prices, spacing, startidx + 1)
print(f"Decimated: {len(decimated)} samples, {len(indices)} indices")

# Apply filter
print(f"Applying filter to decimated signal...")
try:
    result = apply_ormsby_filter(decimated, h, mode='reflect', fs=fs)
    print(f"Filter applied successfully")
    print(f"Result keys: {result.keys()}")
    print(f"Signal length: {len(result['signal'])}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTest complete!")
