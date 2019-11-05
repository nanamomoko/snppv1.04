#
# Name:
#	snpp
# PURPOSE:
#	calculate the S/N per pixel for CSST and simulate a noisy spectrum for any given template.
#IUTPUTS:
#   targetmag      the surface brightness of the target you want to calculate the S/N (defult: 17 mag/arcsec^2)
#   deltal         the delta lambda per pixel, in unit of nm (defult: 1.755555 A)
#   galtpl         the filename of star-forming galaxy template you want to use.
#   filtera         the filter you chosed to estimate the S/N (defult: sdss_g)
#   specsample     pixels per spectral resolution element (defult: 2)
#   readnoise      read noise, in e/pix. (defult: 5.0)
#   fovp           diameter of fiber (or spaxel) in arcsec (defult: 0.2 arcsec)
#   npixel_width   the width of the spectrum on the CCD (defult: 2.0)
#   obstime        in seconds, single integration time (defult: 300s)
#   repeatnum      repeat number (defult: 20.0)
#   skyv           V band sky brightness in Johnson V mag/arcsec^2 unit (defult: 22.5 mag/arcsec^2)
#   qinput         the throughput correct factor (defult: 1.0)
#   slitwidth      suit to the slit case. the length assumed to be 0.15 arcsec
#   skyperpixel    a second way of estimating the Sky, if know the sky photon number per pixel
#
#IDL
# v5: 15 August 2018      writen by Lei Hao, rivised by Jun Yin
# v7: 10 Sep 2019  by Jun Yin
#
#python
# V1.04beta by Mengting Ju
##################################################################
##################################################################

from __future__ import print_function
from astropy.io import fits
import numpy as np
from snpp import snpp
import inwf
import inmg

#################################################################
##################################################################

def snpp_example(): 
    
    select=1 # 1 or 2
    resu=snpp_model(select)
    wavearr,galflux,integconst,delta_lambda,filtera,sampling=\
    resu[0],resu[1],resu[2],resu[3],resu[4],resu[5]
    
    filename='../results/mg_weak_30020_25.fits'
    ss=snpp(wavearr=wavearr,galflux=galflux,integconst=integconst,
            deltal=delta_lambda,filename=filename,
            specsample=sampling,filtera=filtera,
            readnoise=5.0,fovp=0.2,npixel_width=2.0,
            obstime=300,repeatnum=20,skyv=22.5,qinput=1.0, 
            slitwidth=None,skyperpixel=True)
    
################################################################

def snpp_model(s):
    
    if s==1:
    
        #select model and magnitude
        targetmag=17.
        galtpl='../obs/SFgal_tpl/SFgal_texp_FeH0_tau5_Ew10.fits'
        deltal=1.755555
        filtera='../obs/filters/sdss_g0.par'
        sampling=2.0

        result=inmg.input_mag_model(targetmag,galtpl,deltal, filtera,sampling)
        wavearr=result[0]   #A
        galflux=result[1]   #10^-12 erg/s/A/cm2
        integconst=result[2]
        delta_lambda=np.array([deltal]*len(wavearr))    

    elif s==2:
        
        #select put in wave and flux    
        filee=fits.open('MockGal-M21Z0.01-FOV6R0.1-W350n1000n.fits')
        fluxx=np.sum(np.sum(filee[1].data[:20,:20],1),0)  #erg/s/A/cm2
        wavee=filee[2].data    #A
        redshift=0.01
        filtera='sdss_r0.par'
        sampling=2.0

        result=inwf.input_wave_flux(wavee,fluxx,redshift,filtera)
        wavearr=result[0]  #A
        galflux=result[1]  #10^-12 erg/s/A/cm2
        integconst=result[2]
        delta_lambda=result[3]
    
    return wavearr,galflux,integconst,delta_lambda,filtera,sampling


#########################################################

if __name__=='__main__':
    snpp_example()
    
