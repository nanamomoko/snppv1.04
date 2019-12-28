from astropy.io import fits
import numpy as np
import pandas as pd
from scipy.integrate import simps
import read_filter
#########################################################################################
#select input parameters
def input_mag_model(targetmag,galtpl,filtera):
    
    resulte=read_filter.filteraa(filtera)
    wavefilter=resulte[0]
    fluxfilter=resulte[1]
    vmin=resulte[2]
    vmax=resulte[3]
    filtereff=resulte[4]
    
  ####################################################################################
    # define wavelength array,
    #cover the range of 350nm to 1050nm, depend on the spectral resolution wanted. 

    delta_lambda=1.755555 # has to be in unit of A
    print('delta_lambda:', delta_lambda)

    narray=int((10000.0-3500.0)/delta_lambda) 
    wavearr=3500.0+delta_lambda*np.float64(np.arange(narray))
    # select out the array of V band filter
    ii=np.logical_and(wavearr >= vmin, wavearr <= vmax)
    wavetmp2=wavearr[ii]
    x=np.interp(wavetmp2,wavefilter,fluxfilter)
    integratef4=x*wavetmp2
    integconst=simps(integratef4,wavetmp2) # int(lambda*Rlambda*dlambda)
    
    lambdav=filtereff #A
    ###############################################################
    # define basic target brightness, parameters constantly change
    itarget=targetmag    # in Johnson V mag/arcsec^2 unit
    print('itarget:',itarget)
    
    itarget_jy=3631.0*10**(-itarget/2.5+3.0)  # target flux in V in mJy/arcsec^2 unit
    itarget_nm=itarget_jy*3.0/(lambdav/100.0)**2 #target flux in V in 10^(-12)erg/s/cm^2/A (/arcsec^2 ?)
    
    galtpl=galtpl
    tplfile=galtpl
    print('tplfile:',tplfile)   

    sfgal=fits.open(tplfile)
    wavegal=sfgal[1].data['wave'] # A
    galflux2=sfgal[1].data['flux']
    galflux1=np.interp(wavearr,wavegal,galflux2)

    #;normalize the galaxy spectrum to the V band magnitude specified.
    ii=np.logical_and(wavegal >= vmin, wavegal <= vmax)
    wavetmp=wavegal[ii]
    fluxtmp=galflux2[ii]
    x=np.interp(wavetmp,wavefilter,fluxfilter)
    vfluxtmp=x*wavetmp*fluxtmp #bandpass*lambda*F_gal_lambda
    galintegrate=simps(vfluxtmp,wavetmp)
    galnorm=itarget_nm*integconst/galintegrate
    galflux=galnorm*galflux1   # the unit should now be in 10^(-12)erg/s/A/cm^2 (/arcsec^2 ?)

##################################################################
    return wavearr, galflux

##############################################################
