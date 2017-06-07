"""
Code created by Christian Flores
"""

#CAMBIAR PATHS, PA and Disk inclination
import numpy as np
import astropy.io.fits as pf
import math
import sys
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#
#Global Variables
#
#-----------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------
#
#constants in mks system
#
#-----------------------------------------------------------------------------
#incli = 152.0*math.pi/180.0					#Disk inclination
incli = 0.


PA = 24.00*math.pi/180.0

# constants in SI units
h=6.62606957*10**(-34)          # [Jxs] Planck's constant
c=2.99792458*10**8                 # [m/s] light speed
k=1.3806488*10**(-23)               # [J/K] Boltzmann's constant
rest_freq_C18O= 219.56036e9					#C18O rest frequency
rest_freq_12CO= 230.53800e9					#12CO rest frequency
rest_freq_13CO= 220.39868e9					#13CO rest frequency

A_C18=6.011e-07						#Einstein Coefficients in s^-1
A_C13=6.038e-07
A_C12=6.910e-07


###### Transition 2-1
## A_C18=6.266e-08						#Einstein Coefficients in s^-1
## A_C13=6.294e-08
## A_C12=7-203e-08
####

#-----------------------------------------------------------------------------
#
#data manipulation constants
#
#-----------------------------------------------------------------------------

ruido=0.00467203						# image noise

#-----------------------------------------------------------------------------
#
#.fits information
#
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#12CO
path_12CO='/home/felipe/Escritorio/ID/HD142527/HD142527_12CO21.clean.fits'
hdulist12=pf.open(path_12CO)
prihdr12=hdulist12[0].header					#header information
scidata12=hdulist12[0].data					#data matrix
cdelt3_12=prihdr12['CDELT3']					#frequency interval
cval3_12=prihdr12['CRVAL3']					#initial frequency
bmajaxis_12=prihdr12['BMAJ']					#beam mayor and minor axis
bminaxis_12=prihdr12['BMIN']
chan_12=prihdr12['NAXIS3']						#number of channels
#-----------------------------------------------------------------------------
#13CO
path_13CO='/home/felipe/Escritorio/ID/HD142527/HD142527_13CO_contrest.fits'
hdulist13=pf.open(path_13CO)
prihdr13=hdulist13[0].header
scidata13=hdulist13[0].data
cdelt3_13=prihdr13['CDELT3']
cval3_13=prihdr13['CRVAL3']
bmajaxis_13=prihdr13['BMAJ']
bminaxis_13=prihdr13['BMIN']
chan_13=prihdr13['NAXIS3']						#number of channels

#-----------------------------------------------------------------------------
#C18O
path_C18O='/home/felipe/Escritorio/ID/HD142527/HD142527_C18O21.clean.fits'
hdulist18=pf.open(path_C18O)
prihdr18=hdulist18[0].header
scidata18=hdulist18[0].data
cdelt3_18=prihdr18['CDELT3']
cval3_18=prihdr18['CRVAL3']
bmajaxis_18=prihdr18['BMAJ']
bminaxis_18=prihdr18['BMIN']
chan_18=prihdr18['NAXIS3']						#number of channels
