import sys
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np

from src.spectral.lanczos import lanczos_spectrum
from src.spectral.peak_detection import find_spectral_peaks, find_spectral_troughs
from src.spectral.envelopes import fit_upper_envelope, fit_lower_envelope, envelope_model

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../../data/raw/^dji_w.csv')
df = pd.read_csv(csv_path)
datapointsperyr = 52

dataspacing = 1
df2 = df[df.Date.between('1921-04-29','1965-05-21')] # discord dates  29-04-1921 to 14-05-1965
#df2numpy = df['Close'].to_numpy()

[w,wRad,cosprt,sinprt,amp,phRad,phGrad] = lanczos_spectrum(df2.Close.values, dataspacing, datapointsperyr)

# Convert to radians/year
omega_yr = w * 52  # w is rad/week → *52 = rad/year

# Detect peaks and troughs in the spectrum
peak_idx, peak_freq, peak_amp = find_spectral_peaks(amp, omega_yr, min_distance=3)
trough_idx, trough_freq, trough_amp = find_spectral_troughs(amp, omega_yr, min_distance=3)

print(f"Detected {len(peak_idx)} peaks and {len(trough_idx)} troughs")

# Fit envelopes to peaks and troughs
upper_fit = fit_upper_envelope(peak_freq, peak_amp)
lower_fit = fit_lower_envelope(trough_freq, trough_amp)

print(f"\nUpper envelope: a(ω) = {upper_fit['k']:.4f} / ω")
print(f"  R² = {upper_fit['r_squared']:.4f}")
print(f"  RMSE = {upper_fit['rmse']:.4f}")

print(f"\nLower envelope: a(ω) = {lower_fit['k']:.4f} / ω")
print(f"  R² = {lower_fit['r_squared']:.4f}")
print(f"  RMSE = {lower_fit['rmse']:.4f}")

# Hardcoded values from original script (for comparison)
# ktop = 0.1875  # original hardcoded upper slope
# kbot = 0.0575  # original hardcoded lower slope

# Generate envelope lines using fitted parameters
upper_peak_amp_line = envelope_model(wRad[:], upper_fit['k'])
lower_peak_amp_line = envelope_model(wRad[:], lower_fit['k'])


# Plot amplitude vs. omega (rad/yr)
plt.figure(figsize=(10, 6))
plt.plot(omega_yr[:], amp[:], 'b.-', markersize=4, label='Spectrum')
plt.plot(omega_yr[:], upper_peak_amp_line[:], 'r-', linewidth=2,
         label=f'Upper envelope: k={upper_fit["k"]:.4f}')
plt.plot(omega_yr[:], lower_peak_amp_line[:], 'g-', linewidth=2,
         label=f'Lower envelope: k={lower_fit["k"]:.4f}')
plt.plot(peak_freq, peak_amp, 'ro', markersize=6, alpha=0.6, label='Detected peaks')
plt.plot(trough_freq, trough_amp, 'go', markersize=6, alpha=0.6, label='Detected troughs')

plt.xscale('linear')
plt.yscale('log')

plt.xlim(-0.1, 22)  # Hurst shows up to ~10 rad/yr
plt.ylim(0.45, 90)  # Hurst shows up from 0 to 80 amplitude

ax = plt.gca()
ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())

# Set major tick interval to 1
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))

# Set minor tick interval to 0.2
ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.2))
ax.minorticks_on()
ax.grid(which='major', color='#666666', linestyle='-', alpha=0.6)
ax.grid(which='minor', color='#999999', linestyle=':', alpha=0.4)

plt.xticks(minor=True)

plt.xlabel("Angular Frequency ω (radians per year)")
plt.ylabel("Amplitude (log scaled price)")
plt.title("Lanczos Spectrum – DJIA 1921–1965 (Hurst Window)\nwith Fitted Envelopes")
plt.legend(loc='upper right', fontsize=9)
plt.grid(True)


# Same plot with frequency as period (high to low frequency = low to high period)
plt.figure(figsize=(10, 6))
plt.plot(wRad[:],amp[:], 'b.-', markersize=4, label='Spectrum')
plt.plot(wRad[:], upper_peak_amp_line[:], 'r-', linewidth=2,
         label=f'Upper envelope')
plt.plot(wRad[:], lower_peak_amp_line[:], 'g-', linewidth=2,
         label=f'Lower envelope')
plt.grid(True)
plt.legend()

ax = plt.gca()
ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
plt.xlabel('Period (years)')
plt.ylabel('Amplitude')
plt.title('Fourier-Lanczos Spectrum (Period Domain)\nDJIA 1921-1965')
plt.show()

