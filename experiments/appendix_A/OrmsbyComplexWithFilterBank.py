# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 00:09:41 2026

@author: Dime
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import scipy

from funcDesignFilterBank import design_ormsby_filter_bank, print_filter_specs, create_filter_kernels, plot_filter_bank_response, apply_filter_bank, create_time_frequency_heatmap
from funcOrmsby import ormsby_filter, funcOrmsby3, apply_ormsby_filter


def load_data2(csv_file):
    """Load DJIA data from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        #data.dates = df['Date'].values
        #data.close_prices = df['Close'].values
        #print(f"Loaded {len(self.dates)} data points")
        #print(f"Date range: {self.dates[0]} to {self.dates[-1]}")
    
        return df
    
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        print("Creating synthetic data as fallback...")
        dates = pd.date_range('1920-01-01', '2024-01-01', freq='W')
        dates = dates.values
        close_prices = 100 + 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 52) + \
                           np.cumsum(np.random.randn(len(dates)) * 0.5)

def find_date_index2( date_str,dates ):
    """Find index closest to given date string"""
    target_date = pd.to_datetime(date_str)
    dates_pd = pd.to_datetime(dates)
    idx = np.argmin(np.abs(dates_pd - target_date))
    return idx


plt.close('all')
t=52
fs=52
dt=1/t
oprice = load_data2('^DJI_w.csv')
signal = oprice['Close']
dates  = oprice['Date']

display_start_idx = 1
display_end_idx = 6000

display_start_idx = find_date_index2('1935-01-01',dates)
display_end_idx = find_date_index2('1951-12-31',dates)    

#results = self.apply_filter_bank(signal, filters)
start_idx = display_start_idx
end_idx = display_end_idx

dates_display = pd.to_datetime(dates[start_idx:end_idx])
signal_display = signal[start_idx:end_idx]

spacing = 1
nw = 199*7
w1 = 0.85; w2 = 1.25; w3 = 2.05; w4 = 2.45
datapointsperyr = 52
f2 = funcOrmsby3(199*7,w1,w2,w3,w4,datapointsperyr,1,"bp")

#f3 = funcOrmsby2.funcOrmsby2(249*5,3.2,3.55,6.35,6.7,datapointsperyr,1,"bp")
#f4 = funcOrmsby2.funcOrmsby2(349*5,7.25,7.55,9.55,9.85,datapointsperyr,1,"bp")
#f5 = funcOrmsby2.funcOrmsby2(649*2,13.65,13.95,19.35,19.65,datapointsperyr,1,"bp")
#f6 = funcOrmsby2.funcOrmsby2(1299*1,28.45,28.75,35.95,36.25,datapointsperyr,1,"bp")

dt = 1/52
twopi = 2*np.pi 
w_edgeslp = np.array([w1,w2])
w_edgesbp = np.array([w1,w2,w3,w4])
f_edgeslp = w_edgeslp / twopi
f_edgesbp = w_edgesbp / twopi


ff2s = ormsby_filter(nw=nw,f_edges=f_edgesbp,fs=fs,filter_type="bp",method='subtract',analytic=False)     # Optional: enforce H(0)=1 for LP
ff2mc = ormsby_filter(nw=nw,f_edges=f_edgesbp,fs=fs,filter_type="bp",method='modulate',analytic=True)     # Optional: enforce H(0)=1 for LP
#method 'subtract' or 'modulate' (for bp only)


# Low PASS Analysis
if True:
    # compare classic Ormsby Low pass to new ormby low pass
    hlp1 = funcOrmsby3(199*7,w1,w2,w1,w2,datapointsperyr,1,"lp")
    hlp2 = ormsby_filter(nw=nw,f_edges=f_edgeslp,fs=fs,filter_type="lp",method='',analytic=False)     # Optional: enforce H(0)=1 for LP
    hlp3 = ormsby_filter(nw=nw,f_edges=f_edgeslp,fs=fs,filter_type="lp",method='',analytic=False)     # Optional: enforce H(0)=1 for LP
    hlp4 = ormsby_filter(nw=nw,f_edges=f_edgeslp,fs=fs,filter_type="lp",method='',analytic=True)     # Optional: enforce H(0)=1 for LP

    lp1 = apply_ormsby_filter(signal=signal,h=hlp1,mode='reflect',fs=fs)
    lp2 = apply_ormsby_filter(signal=signal,h=hlp2,mode='reflect',fs=fs)
    lp3 = apply_ormsby_filter(signal=signal,h=hlp3,mode='reflect',fs=fs)
    lp4 = apply_ormsby_filter(signal=signal,h=hlp4,mode='reflect',fs=fs)
        
    fig, (ax1,ax2) = plt.subplots(2, 1)    
    
    # compare kernals
    ax1.plot(hlp1, label="lp orig fun")
    ax1.plot(hlp2, '--', label="lp2 nodccon")  #qwen says dont use this one half amplutide expected is no dc 
    ax1.plot(hlp3, label="lp new constraint DC") 
    ax1.plot(np.real(hlp4), linestyle='-.',label="lp real complex cDC") 
    #ax1.plot(np.imag(hlp4), linestyle='-.',label="lp imag complex cDC") # plots but high valued
    ax1.legend(loc="upper right")
    
    # lp1 classic fun and lp3 match lp2 bad half amp with dc setting
    # lp 3 and lp4 (real) also match
    #ax2.plot(oprice['Date'],oprice['Close'], label='price')
    #ax2.plot(oprice['Date'],lp1['signal'], label='bp1')
    #ax2.plot(oprice['Date'],lp2['signal'], linestyle='--',label='lp2 no dc half')
    ax2.plot(oprice['Date'],lp3['signal'], linestyle='--', label='lp3')
    ax2.plot(oprice['Date'],lp4['signal'], linestyle='-.', label='lp4 complex')
    ax2.plot(oprice['Date'],lp4['envelope'], linestyle='solid', label='lp4 complex env')
    ax2.legend(loc="upper right")
    #fig.legend()
    fig.show
    
# BandPass Analysis
if True:
    # compare classic Ormsby Bandpass pass to new ormby band pass created via subtract and rotate and complex
    hbp1 = funcOrmsby3(199*7,w1,w2,w3,w4,datapointsperyr,1,"bp")
    hbp2 = ormsby_filter(nw=nw,f_edges=f_edgesbp,fs=fs,filter_type="bp",method='subtract',analytic=False)     # Optional: enforce H(0)=1 for LP
    hbp3 = ormsby_filter(nw=nw,f_edges=f_edgesbp,fs=fs,filter_type="bp",method='modulate',analytic=False)     # Optional: enforce H(0)=1 for LP
    hbp4 = ormsby_filter(nw=nw,f_edges=f_edgesbp,fs=fs,filter_type="bp",method='modulate',analytic=True)     # Optional: enforce H(0)=1 for LP
    hbp5 = ormsby_filter(nw=nw,f_edges=f_edgesbp,fs=fs,filter_type="bp",method='subtract',analytic=True)     # Optional: enforce H(0)=1 for LP
    #hbp5 = ormsby_bp_subtract_complex(f_edgesbp[0], f_edgesbp[1], f_edgesbp[2], f_edgesbp[3], nw, fs)
    #hder = ormsby_filter(nw=nw,f_edges=f_edgeslp,fs=fs,filter_type="deriv",method='',analytic=True)
    
    bp1 = apply_ormsby_filter(signal=signal,h=hbp1,mode='reflect',fs=fs)
    bp2 = apply_ormsby_filter(signal=signal,h=hbp2,mode='reflect',fs=fs)
    bp3 = apply_ormsby_filter(signal=signal,h=hbp3,mode='reflect',fs=fs)
    bp4 = apply_ormsby_filter(signal=signal,h=hbp4,mode='reflect',fs=fs)
    bp5 = apply_ormsby_filter(signal=signal,h=hbp5,mode='reflect',fs=fs)
    
    fig, (ax1,ax2) = plt.subplots(2, 1)    
    
    # compare kernals
    #ax1.plot(hbp1, label="Bp orig fun")
    #ax1.plot(hbp2, '-.', label="Bp2 subtract")  #qwen says dont use this one half amplutide expected is no dc 
    #ax1.plot(hbp3, '--',label="bp2 modulate") 
    ax1.plot(np.real(hbp4), linestyle='-.',label="bp real mod complex cDC") 
    ax1.plot(np.imag(hbp4), linestyle='-.',label="bp imag mod complex cDC") # plots but high valued
    ax1.plot(np.real(hbp5), linestyle='solid',label="bp real sub complex cDC") 
    ax1.plot(np.imag(hbp5), linestyle='solid',label="bp imag sub complex cDC") # plots but high valued

    ax1.legend(loc="upper right")
    
    # lp1 classic fun and lp3 match lp2 bad half amp with dc setting
    # lp 3 and lp4 (real) also match
    ax2.plot(oprice['Date'],oprice['Close'], label='price')
    ax2.plot(oprice['Date'],bp1['signal'], label='bp1')
    ax2.plot(oprice['Date'],bp2['signal'], linestyle='--', label='bp2 subtract')
    ax2.plot(oprice['Date'],bp3['signal'], linestyle='--', label='bp3 modulated lp')
    ax2.plot(oprice['Date'],bp4['signal'], linestyle='-.', label='bp4 complex')
    ax2.plot(oprice['Date'],bp4['envelope'], linestyle='solid', label='bp4 complex env')
    ax2.plot(oprice['Date'],bp5['signal'], linestyle='-.', label='bp4 sub complex')
    ax2.plot(oprice['Date'],bp5['envelope'], linestyle='solid', label='bp4 sub complex env')
    
    ax2.legend(loc="upper right")
    #fig.legend()
    fig.show


# Page 152 Analysis
if False:
    # compare classic Ormsby Bandpass pass to new ormby band pass created via subtract and rotate and complex
    hbp1 = funcOrmsby3(199*7,w1,w2,w3,w4,datapointsperyr,1,"bp")
    hbp2 = ormsby_filter(nw=nw,f_edges=f_edgesbp,fs=fs,filter_type="bp",method='subtract',analytic=False)     # Optional: enforce H(0)=1 for LP
    hbp3 = ormsby_filter(nw=nw,f_edges=f_edgesbp,fs=fs,filter_type="bp",method='modulate',analytic=False)     # Optional: enforce H(0)=1 for LP
    hbp4 = ormsby_filter(nw=nw,f_edges=f_edgesbp,fs=fs,filter_type="bp",method='modulate',analytic=True)     # Optional: enforce H(0)=1 for LP
    hbp5 = ormsby_filter(nw=nw,f_edges=f_edgesbp,fs=fs,filter_type="bp",method='subtract',analytic=True)     # Optional: enforce H(0)=1 for LP
    #hbp5 = ormsby_bp_subtract_complex(f_edgesbp[0], f_edgesbp[1], f_edgesbp[2], f_edgesbp[3], nw, fs)
    
    bp1 = apply_ormsby_filter(signal=signal,h=hbp1,mode='reflect',fs=fs)
    bp2 = apply_ormsby_filter(signal=signal,h=hbp2,mode='reflect',fs=fs)
    bp3 = apply_ormsby_filter(signal=signal,h=hbp3,mode='reflect',fs=fs)
    bp4 = apply_ormsby_filter(signal=signal,h=hbp4,mode='reflect',fs=fs)
    bp5 = apply_ormsby_filter(signal=signal,h=hbp5,mode='reflect',fs=fs)
    
    fig, (ax1,ax2) = plt.subplots(2, 1)    
    
    # compare kernals
    #ax1.plot(hbp1, label="Bp orig fun")
    #ax1.plot(hbp2, '-.', label="Bp2 subtract")  #qwen says dont use this one half amplutide expected is no dc 
    #ax1.plot(hbp3, '--',label="bp2 modulate") 
    ax1.plot(np.real(hbp4), linestyle='-.',label="bp real mod complex cDC") 
    ax1.plot(np.imag(hbp4), linestyle='-.',label="bp imag mod complex cDC") # plots but high valued
    ax1.plot(np.real(hbp5), linestyle='solid',label="bp real sub complex cDC") 
    ax1.plot(np.imag(hbp5), linestyle='solid',label="bp imag sub complex cDC") # plots but high valued

    ax1.legend(loc="upper right")
    
    # lp1 classic fun and lp3 match lp2 bad half amp with dc setting
    # lp 3 and lp4 (real) also match
    ax2.plot(oprice['Date'],oprice['Close'], label='price')
    ax2.plot(oprice['Date'],bp1['signal'], label='bp1')
    ax2.plot(oprice['Date'],bp2['signal'], linestyle='--', label='bp2 subtract')
    ax2.plot(oprice['Date'],bp3['signal'], linestyle='--', label='bp3 modulated lp')
    ax2.plot(oprice['Date'],bp4['signal'], linestyle='-.', label='bp4 complex')
    ax2.plot(oprice['Date'],bp4['envelope'], linestyle='solid', label='bp4 complex env')
    ax2.plot(oprice['Date'],bp4['frequency'], linestyle='solid', label='bp4 complex freq')
    ax2.plot(oprice['Date'],bp4['phasew'], linestyle='solid', label='bp4 complex freq')
    
    ax2.plot(oprice['Date'],bp5['signal'], linestyle='-.', label='bp4 sub complex')
    ax2.plot(oprice['Date'],bp5['envelope'], linestyle='solid', label='bp4 sub complex env')
    
    ax2.legend(loc="upper right")
    #fig.legend()
    fig.show

        
# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    import sys
    sys.path.append('.')  # Ensure imports work
    
    # Design filter bank
    print("Designing filter bank...")
    specs = design_ormsby_filter_bank(
        fs=52,
        nw_base=199*7,
        f_lp_pass=0.85,
        f_lp_stop=1.25,
        Q_factor=4.0,
        spacing_mode='balanced',  # Try: 'constant_q', 'balanced', 'overlap_3db'
        max_freq=None
    )
    
    print("\nTry different modes:")
    print("  'balanced'    : Adjusts bandwidth for continuous coverage → best unity gain")
    print("  'overlap_3db' : Overlaps at -3dB points → good for analysis")
    print("  'constant_q'  : Strict constant-Q → best frequency resolution")
    
    # Print specifications
    print_filter_specs(specs)
    
    # Create filter kernels
    print("Creating filter kernels...")
    filters = create_filter_kernels(
        filter_specs=specs,
        fs=52,
        filter_type='modulate',
        analytic=True
    )
    print(f"Created {len(filters)} filter kernels\n")
    
    # Plot frequency responses
    print("Plotting frequency responses...")
    fig = plot_filter_bank_response(filters, fs=52)
    plt.show()
    
    # Example: Apply to synthetic signal
    print("\nExample with synthetic signal:")
    t = np.arange(0, 1000) / 52  # 1000 weeks
    signal = (np.sin(2 * np.pi * 1.5 * t) +      # 1.5 rad/yr component
              0.5 * np.sin(2 * np.pi * 5.0 * t) + # 5.0 rad/yr component
              0.3 * np.random.randn(len(t)))      # noise
    
    # Apply filter bank
    results = apply_filter_bank(signal, filters, fs=52, mode='reflect')
    
    # Create heatmap
    fig, tf_matrix = create_time_frequency_heatmap(results)
    plt.show()
    
    print("\n" + "=" * 80)
    print("Filter bank demonstration complete!")
    print("=" * 80)