import numpy as np
import pandas as pd
################################################################################
def filteraa(filtera):
    #load the filters          
    filtersel=filtera #'../sdss_g0.par'            
    filterpath='./'
    filterfile=filterpath+filtersel   # ;fluxfilter: max=1, min=0, no particular unit
    print(filterfile)   

    ia=0
    with open(filterfile,'r') as fh:
        for line in fh:
            if line.startswith('#'):
                ia=ia+1
                continue

    band=pd.read_csv(filterfile,sep='\s+',header=None,skiprows=ia)
    wavefilter=np.array(band[0])
    fluxfilter=np.array(band[1])
    wavefilter=wavefilter  # A
    vmin=wavefilter[0]
    vmax=wavefilter[-1]

    # find the central wavelength, effective wavelength, and FWHM of the given filter
    filtermid=(vmax-vmin)*0.5  #A, central wavelength
    dwave=wavefilter[1:]-wavefilter[:-1]
    filtereff=np.nansum(dwave*wavefilter[1:]*fluxfilter[1:])/np.nansum(dwave*fluxfilter[1:]) #A, effective wavelength
    rmax=np.max(fluxfilter)
    nnn=np.where(fluxfilter > 0.5*rmax)[0]
    FWHMmin=wavefilter[nnn[0]]
    FWHMmax=wavefilter[nnn[-1]]
    filterwid=FWHMmax-FWHMmin  #A, FWHM
    
    return wavefilter,fluxfilter,vmin,vmax,filtereff,FWHMmin,FWHMmax