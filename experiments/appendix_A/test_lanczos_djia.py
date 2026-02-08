import sys
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np

from src.spectral.lanczos import lanczos_spectrum

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../../data/raw/^dji_w.csv')
df = pd.read_csv(csv_path)
datapointsperyr = 52

dataspacing = 1
df2 = df[df.Date.between('1921-04-29','1965-05-21')] # discord dates  29-04-1921 to 14-05-1965
#df2numpy = df['Close'].to_numpy()

[w,wRad,cosprt,sinprt,amp,phRad,phGrad] = lanczos_spectrum(df2.Close.values, dataspacing, datapointsperyr)

ktop = 0.1875 # slope of upper line
kbot = 0.0575 # slope of lower line

# Convert to radians/year
omega_yr = w * 52  # w is rad/week → *52 = rad/year
upper_peak_amp_line = wRad[:]*ktop
lower_peak_amp_line = wRad[:]*kbot


# Plot amplitude vs. omega (rad/yr)
plt.figure(figsize=(10, 6))
plt.plot(omega_yr[:], amp[:], 'b.-', markersize=4)  # 
plt.plot(omega_yr[:], upper_peak_amp_line[:], '-', markersize=4)  # 
plt.plot(omega_yr[:], lower_peak_amp_line[:], '-', markersize=4)  # 
plt.xscale('linear')
plt.yscale('log')

#plt.ticklabel_format(style='plain', axis='y', useOffset=False)
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
plt.title("Lanczos Spectrum – DJIA 1921–1965 (Hurst Window)")
plt.grid(True)


# same plot frequencys high to low in points not radians
plt.figure()
plt.plot(wRad[:],amp[:])
plt.plot(wRad[:],wRad[:]*ktop,linestyle='--')
plt.plot(wRad[:],wRad[:]*kbot,linestyle='--')
#plt.plot((2*np.pi) / wRad[1:-1],amp[1:-1]);
plt.grid(True)

# plt.ticklabel_format(style='plain', axis='y', useOffset=False)
ax = plt.gca()
ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
plt.title('Fourier DJIA plotted as points and amplitude high frequency to low frequency')
plt.show()

