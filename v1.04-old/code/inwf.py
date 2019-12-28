from astropy.io import fits
import numpy as np
import pandas as pd
from scipy.integrate import simps
import read_filter
#######################################################################################################
# select put in wave and flux
def input_wave_flux(wave,flux):
    
    ########################################################################################### 
    
    wave=wave
    flux=flux
    delta_lambda=1.755555
    narray=int((wave[-1]-wave[0])/delta_lambda) 
    wavearr=wave[0]+delta_lambda*np.float64(np.arange(narray))
        
    galflux=np.interp(wavearr,wave,flux)      #erg/s/A/cm2
    
    
###############################################################################    
    return wavearr, galflux*10**12

################################################################################
