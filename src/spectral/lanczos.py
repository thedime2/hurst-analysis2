# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 00:05:45 2023
@author: localadmin
"""

import numpy as np  


"""
Replication Note: Lanczos Spectrum Data Length and Frequency Resolution

This implementation reproduces the Fourier–Lanczos spectral analysis described
in Appendix A of J.M. Hurst, *The Profit Magic of Stock Transaction Timing*.

Hurst states that the Appendix A spectrum used 2229 data points, providing a
frequency resolution of 0.568 radians per year. When applied to weekly DJIA
data spanning approximately 29 April 1921 through mid-1965 (~44 years, ~2297–
2299 samples), the theoretical Fourier/Lanczos frequency resolution is:

    Δω ≈ 2π / T ≈ 0.14 radians per year

A resolution of 0.568 radians per year would correspond to an effective record
length of approximately 11 years, which is incompatible with:
- The low-frequency content shown in Figure AI-1
- The identification of fine frequency spacing at ~0.3676 radians per year
- The subsequent derivation of the Nominal Model

Empirical replication confirms that the Appendix A Lanczos spectrum must have
been computed over the full ~44-year DJIA record. The stated resolution value
is therefore interpreted as an editorial or explanatory error, not a
methodological one.

Fine spectral structure is inferred through envelope curvature, trough
regularity, and later validation using overlapping comb filter banks, rather
than from discrete Fourier bin spacing alone. Accordingly, this implementation
treats the Lanczos spectrum as a full-record analysis.
"""
    
def lanczos_spectrum(data, dataspacing, datapointsperyr):
    
    # w = np.array()
    # wRad = np.array()
    # cosprt=np.array()
    # sinprt=np.array()    
    # amp=np.array()    
    # phRad=np.array()
    # phGrad=np.array()
    
    if dataspacing == 0:
        print("data spacing 0 setting to 1...")
        dataspacing = 1 # % avoid div 0 err

    if data.size % 2 == 0: #% even length passed
        print("even number of data points - removing last one...")
        #data = data(1:end-1)  #% resize the data -1 to get odd
        data = data[0:-1]
    
    #numpy size is num elements so 2x2 = 4 where as len(x) will be length of 1st elem eg 2
       
    m         = len(data)
    midpoint  = int(((m+1)/2)-1) # add -1 for 0 based indx (python make int for use in idx)
    adata     = data[midpoint::-1] # data midpoint back to start 
    bdata     = data[midpoint::] # data after midpoint to end
    
    seq1      = adata + bdata #% new series is midpoint  (mid) + (an+bn  ... a1+b1) the a+b series
    seq1[0]   = seq1[0]/2.0   #% first value of 1st series midpoint (or as we calced it twice itself / 2.0 
    seq1[-1] = seq1[-1]/2.0 #% last value of 1st series is itself / 2.0 
    
    seq2      = adata - bdata #% new series (a-b) 
    seq2[2]   = 0 # first replace 0
    seq2[-1]  = 0 # last value of 2nd series is 0;
    
    qty       = int((m-1)/2) # % page 173
    Z         = np.pi / qty
    
    ## allow for faster vectorised code to fill the w array
    #i         = 0:1:qty # create i as vector  from 0 to qty stepping 1 eg ( 0 1 2 3 4 5)
    i         = np.arange(0,qty+1,1,dtype=int) # create i as vector  from 0 to qty stepping 1 eg ( 0 1 2 3 4 5)
    w         = i*(Z/dataspacing) #create w in vectorised code w(x) = i(x)*Z/dataspacing 
    #wT        = w.reshape(1,len(w))    #python convert from col to row
    
    #print(w)
    #wRad[1:len(w)] = 2*pi./w[1:-1] #vectorized w into rad (skip element 1 so no div/0 err [0 10 5 3.333 2.5 2]
    
    wRad = np.zeros(len(w)) # init array
   # wRad = wRad.reshape(len(wRad),1) # covert col to row
    
    wRad[1:] = 2*np.pi / w[1:] #vectorized w into rad (skip element 1 so no div/0 err [0 10 5 3.333 2.5 2]
    wRad[0] = np.inf # first omega is inf
    #wRadT = wRad.reshape(1,len(wRad)) # first omega is inf
        
    #%amplitude of seq 1 which is cos
    #%n=0:1:qty  %helper array 0,1,2,3,4,5 ... len (seq1)-1  or qty(5)
    
    ## create matrix in 2 steps k x kk then cos and sin of it remove 2 nested for loops by pre filling for vectorised code
    kvector = np.arange(0,qty+1,1,dtype=int) #0:1:qty 
    kkvector = np.arange(0,qty+1,1,dtype=int).reshape(int(qty+1),1)  #0:1:qty  #make is a col vector
    k_by_kk = kvector * kkvector #  % multiple loop counters
    k_by_kk_x_z = k_by_kk * Z # % could be done above but broken out for readability
    cos_mtx = np.cos(k_by_kk_x_z) # % create a matrix of cos and sines * loop * loop * z
    sin_mtx = np.sin(k_by_kk_x_z) # % create a matrix of cos and sines * loop * loop * z
    
    ##disp("Constructing SEQ 1 and 2 * cos and sin k by kk *Z vectors")
    aa = seq1 * cos_mtx 
    aa = aa.transpose() #  rotate matrix so appears same as in matlab
    
    bb = seq2 * sin_mtx #  % sequence 1 * cos
    bb = bb.transpose() #  rotate matrix so appears same as in matlab
    
    a = sum(aa)
    b = sum(bb)
    
    #print("computing sin and cos prts")
    cosprt = a / qty # % vector 1+k (+1 as not zero based array)
    sinprt = b / qty  
    
    ## then divide first cos value and last cos values by 2
    
    cosprt[0]  =  cosprt[0] / 2
    cosprt[-1] =  cosprt[-1] /2 
    
    sinprt[0]  = 0  #sin of 1st is 0    
    
    ##  --- compute amplitudes ---
    amp = np.sqrt(cosprt ** 2 + sinprt ** 2) # % .^ dot power for each elementwise in cosprt and sinprt vectors
     
    #print("computing Phase Vector...")
    #turn phase array initial computation into vectorised code 
    
    phaseArray = np.arctan(sinprt / cosprt)  # %./ vector division
    #print("computing Phases..."); 
    #%  --  could be vectorized but the for the cos phase 180 degrees adjustment check
    
    phRad = np.zeros(len(w)) # init array
    phGrad = np.zeros(len(w)) # init array
    
    for k in range(qty+1):   #= 0:(qty) pyhton default 0 to n but n not looped 
        #print(k)     
        phase2 = 0.0
        phase2 = phaseArray[k]
        
        if cosprt[k] <= 0: 
            phase2 = phase2 - np.pi

        phRad[k] = phase2
        phGrad[k] = phase2 * 180 / np.pi
     #end  
        
    return w,wRad,cosprt,sinprt,amp,phRad,phGrad

def nextpow2b(N):
    """ Function for finding the next power of 2 """
    n = 1
    while n < N: n *= 2
    return n

def nextpow2(x):
    """returns the smallest power of two that is greater than or equal to the
    absolute value of x.

    This function is useful for optimizing FFT operations, which are
    most efficient when sequence length is an exact power of two.

    :Example:

    .. doctest::

        >>> from spectrum import nextpow2
        >>> x = [255, 256, 257]
        >>> nextpow2(x)
        array([8, 8, 9])

    """
    res = np.ceil(np.log2(x))
    return res.astype('int')  #we want integer values only but ceil gives float

