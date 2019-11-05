from astropy.io import fits
import numpy as np
import pandas as pd
import pylab as pl
from scipy.integrate import simps
import read_filter
#######################################################################################################
# select put in wave and flux
def input_wave_flux(wave,flux,redshift,filtera):
    
    resulte=read_filter.filteraa(filtera)
    wavefilter=resulte[0]
    fluxfilter=resulte[1]
    vmin=resulte[2]
    vmax=resulte[3]
    filtereff=resulte[4]
    ###########################################################################################
    z=redshift
    print('redshift:',z)    
    
    wavearr=wave/(1+z) #A
    galflux=flux       #erg/s/A/cm2
    
    # select out the array of r band filter
    ii=np.logical_and(wavearr >= vmin, wavearr <= vmax)
    wavetmp2=wavearr[ii]
    x=np.interp(wavetmp2,wavefilter,fluxfilter)
    integratef4=x*wavetmp2    
    integconst=simps(integratef4,wavetmp2) # int(lambda*Rlambda*dlambda)
    
    delta_l=wavearr[1:]-wavearr[:-1]
    nndel=list(delta_l)
    nnd=nndel[-1]
    nndel.append(nnd)
    delta_lambda=np.array(nndel)
    
###############################################################################    
    return wavearr, galflux*10**12, integconst,delta_lambda,filtereff

################################################################################