#!/usr/bin/env python3
"""Quick test of spacing-related filter creation."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from src.filters import ormsby_filter

print("Testing filter creation with spacing parameters...")

# Original (no spacing)
nw_orig = 1393
fs_orig = 52

print(f"1. Original filter (nw={nw_orig}, fs={fs_orig})...")
f_edges = np.array([0.85, 1.25, 2.05, 2.45]) / (2*np.pi)
try:
    h1 = ormsby_filter(nw=nw_orig, f_edges=f_edges, fs=fs_orig,
                       filter_type='bp', method='modulate', analytic=False)
    print(f"   Success! Length: {len(h1)}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

# With spacing=7
spacing = 7
nw_spaced = int(nw_orig / spacing)
if nw_spaced % 2 == 0:
    nw_spaced += 1
fs_spaced = fs_orig / spacing

print(f"2. Spaced filter (nw={nw_spaced}, fs={fs_spaced})...")
try:
    h2 = ormsby_filter(nw=nw_spaced, f_edges=f_edges, fs=fs_spaced,
                       filter_type='bp', method='modulate', analytic=False)
    print(f"   Success! Length: {len(h2)}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTest complete!")
