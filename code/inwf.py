from astropy.io import fits
import numpy as np
import pandas as pd
import pylab as pl
from scipy.integrate import simps
import read_filter
#######################################################################################################
# select put in wave and flux
def input_wave_flux(wave,flux):
    
    ########################################################################################### 
    
    wave=wave
    flux=flux
    
    narray=int((wave[-1]-wave[0])/1.755555) 
    wavearr=wave[0]+1.755555*pl.frange(narray-1)
        
    galflux=np.interp(wavearr,wave,flux)      #erg/s/A/cm2
    
    
###############################################################################    
    return wavearr, galflux*10**12

################################################################################