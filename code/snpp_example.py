#
# Name:
#	snpp
# PURPOSE:
#	calculate the noise and generate a simulated spectrum with noise for any given template or spectrum.
#IUTPUTS:
#   targetmag      the surface brightness of the target you want to calculate the S/N (defult: 17 mag/arcsec^2)
#   galtpl         the filename of star-forming galaxy template you want to use.
#   filtera         the filter you chosed to estimate the S/N (defult: sdss_g)
#   readnoise      read noise, in e/pix. (defult: 5.0)
#   fovp           diameter of fiber (or spaxel) in arcsec (defult: 0.2 arcsec)
#   npixel_width   the width of the spectrum on the CCD (defult: 2.0)
#   obstime        in seconds, single integration time (defult: 300s)
#   repeatnum      repeat number (defult: 20.0)
#   skyr           r band sky brightness in Johnson r mag/arcsec^2 unit (defult: 22.5 mag/arcsec^2)
#   qinput         the throughput correct factor (defult: 1.0)
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
from snpp import *

#################################################################
##################################################################

def snpp_example(): 
    
    select=1 # 1 or 2
    resu=snpp_model(select)
    wavearr,galflux=resu[0],resu[1]
    
    filename='../results/mg_weak_30020_5.5.fits'
    
    ss=snpp(wavearr=wavearr,galflux=galflux,
            filename=filename,
            readnoise=5.5,fovp=0.2,npixel_width=2.0,
            obstime=300,repeatnum=20,skyr=22.5,qinput=1.0, 
            skyperpixel=True)
    
################################################################

def snpp_model(s):
    
    if s==1:
    
        #select model and magnitude
        targetmag=17.
        galtpl='../obs/SFgal_tpl/SFgal_texp_FeH0_tau5_Ew5.fits'
        filtera='../obs/filters/sdss_g0.par'

        result=input_mag_model(targetmag,galtpl,filtera)
        wavearr=result[0]   #A
        galflux=result[1]   #10^-12 erg/s/A/cm2   

    elif s==2:
        
        #select put in wave and flux    
        filee=fits.open('MockGal-M21Z0.01-W350n1000n.fits')
        fluxx=filee[1].data  #erg/s/A/cm2
        wavee=filee[2].data    #A

        result=input_wave_flux(wavee,fluxx)
        wavearr=result[0]  #A
        galflux=result[1]  #10^-12 erg/s/A/cm2
    
    return wavearr,galflux


#########################################################

if __name__=='__main__':
    snpp_example()
    
