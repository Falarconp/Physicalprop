"""
Code created by Christian Flores
"""
import collision_cross_section as ccs
from constants import *
import numpy as np
import astropy.io.fits as pf
import matplotlib.pyplot as plt
import scipy.ndimage.measurements as meas
import glob
import os
import math
from iminuit import Minuit
import sys
import scipy.signal as senal
#import tidy_fitting

#-----------------------------------------------------------------------------
#
#getting initial values
#
#-----------------------------------------------------------------------------


frac=ccs.molecular_fraction(isotopologue)			#getting molecular fraction
m=ccs.molecular_mass(isotopologue) 				#getting molecular mass in Kg
cross_sect=ccs.cross_section(isotopologue)			#getting cross section
sigma=cross_sect/(m*10**(9))   					# efective area in cm^2/micrograms
kdivm=k/m     							# boltzman constant divided by mass



#-----------------------------------------------------------------------------
#
#scaling factors from mks unit to Jy/beam
#
#-----------------------------------------------------------------------------

scaling12=(1e26*bmajaxis_12*bminaxis_12*(3600**2)*math.pi/(4*math.log(2)) )/(4.25e10)

scaling13=(1e26*bmajaxis_13*bminaxis_13*(3600**2)*math.pi/(4*math.log(2)) )/(4.25e10)

scaling18=(1e26*bmajaxis_18*bminaxis_18*(3600**2)*math.pi/(4*math.log(2)) )/(4.25e10)


#------------------------------------------------------------------------------------------------
#
#planck function in frequency, W/(m^2*Hz*sr) (mks system)
#
#------------------------------------------------------------------------------------------------
def bbody(t,nu):
    return (1.4745e-50)*nu**3.0/(math.exp(4.79922e-11*nu/t)-1.0)

#------------------------------------------------------------------------------------------------
#
#gaussian profile normalized
#
#------------------------------------------------------------------------------------------------

def phi(t,nu,nu0,vturb,angle):
    pa=(-20+90+angle)*math.pi/180.0  #-20 is the PA from east of north, and 90 to get the semi-minor axis
    Q=1
    shear=(Q*math.sin(incli)**2)*(math.tan(incli)**2)*math.sin(2*pa)**2
    deltav=(nu0/c)*math.sqrt(2.0*kdivm*t+2.0*vturb**2 )*(1+shear)**(0.5)
    #deltav= (nu0/c)*math.sqrt(2.0*kdivm*t+2.0*vturb**2 )
    phi0=deltav*math.sqrt(math.pi)
    gauss=math.exp(-((nu-nu0)**2.0)/(deltav**2.0))
    return gauss/phi0

#------------------------------------------------------------------------------------------------
#
#Voigt Profile normalized to the gaussian
#
#------------------------------------------------------------------------------------------------

def phi2(t,nu,nu0,vturb):
    if isotopologue=="12CO":
	frequency=np.linspace(cval3_12,cval3_12+cdelt3_12*(chan-1),num=chan)
    if isotopologue=="13CO":
	frequency=np.linspace(cval3_13,cval3_13+cdelt3_13*(chan-1),num=chan)
    if isotopologue=="C18O":
	frequency=np.linspace(cval3_18+cdelt3_18*(chan-1),cval3_18,num=chan)

    deltav=(nu0/c)*math.sqrt(2.0*kdivm*t+2.0*vturb**2)
    phi0=deltav*math.sqrt(math.pi)
    gauss=np.empty_like(frequency)
    for i in range(len(frequency)):
    	gauss[i]=math.exp(-((frequency[i]-nu0)**2.0)/(deltav**2.0))
    lorenciana=lorentzian_profile(t,isotopologue,nu0,frequency)
    #line_profile=np.convolve(gauss,lorenciana)
    line_profile=senal.fftconvolve(gauss,lorenciana)
    max_posi_voigt=meas.maximum_position(line_profile)[0]
    max_posi_gauss=meas.maximum_position(gauss)[0]
    max_gauss=gauss[max_posi_gauss]#meas.maximum(gauss)
    max_voigt=line_profile[max_posi_voigt]#meas.maximum(line_profile)
    scal=max_gauss/max_voigt
    pos_ini=max_posi_voigt-max_posi_gauss
    profile=line_profile[pos_ini:pos_ini+100]

    contador=int((nu-frequency[0])/(frequency[1]-frequency[0]))
    #return profile*scal/phi0[contador],gauss/phi0[contador],lorenciana/phi0[contador]
    return profile[contador]*scal/phi0

#------------------------------------------------------------------------------------------------
#
#optical depth
#
#nco is given in microgr/cm^2
#
#------------------------------------------------------------------------------------------------
def tau(t,nu,nu0,nco,vturb,angle):
    #return nco*frac*sigma*phi(t,nu,nu0,vturb)
    return nco*frac*sigma*phi(t,nu,nu0,vturb,angle)/math.cos(incli)

def tau2(t,nu,nu0,nco,vturb):
    return nco*frac*sigma*phi2(t,nu,nu0,vturb)    ### optical depth using the voigt profile line shape

#------------------------------------------------------------------------------------------------
#
# radiative transfer equation in Jy/beams
#
#------------------------------------------------------------------------------------------------

def intensity(t,nu,nu0,alpha,i0,nco,vturb,angle):
	if isotopologue=="12CO":
		scaling=scaling12
		nulevel=rest_freq_12CO

	if isotopologue=="13CO":
		scaling=scaling13
		nulevel=rest_freq_13CO

	if isotopologue=="C18O":
		scaling=scaling18
		nulevel=rest_freq_C18O
	return bbody(t,nu)*(1.0-math.exp(-tau(t,nu,nu0,nco,vturb,angle)))*scaling + i0*math.exp(-tau(t,nu,nu0,nco,vturb,angle))*(nu/nulevel)**alpha

#------------------------ Intensity including the cool 12CO molecular cloud-----------------------

def intensity_12CO(t,nu,nu0,alpha,i0,nco,vturb,t2,nu02,nco2,vturb2,angle):
	if isotopologue=="12CO":
		scaling=scaling12
		nulevel=rest_freq_12CO
	return ( bbody(t,nu)*(1.0-math.exp(-tau(t,nu,nu0,nco,vturb,angle)))*scaling  + i0*math.exp(-tau(t,nu,nu0,nco,vturb,angle))*(nu/nulevel)**alpha )*math.exp(-tau(t2,nu,nu02,nco2,vturb2,angle))

#------------------------------------------------------------------------------------------------
#
# radiative transfer equation without the continuumin Jy/beams
#
#------------------------------------------------------------------------------------------------

def intensity_no_continuum(t,nu,nu0,nco,vturb,angle):
	if isotopologue=="12CO":
		scaling=scaling12
		nulevel=rest_freq_12CO

	if isotopologue=="13CO":
		scaling=scaling13
		nulevel=rest_freq_13CO

	if isotopologue=="C18O":
		scaling=scaling18
		nulevel=rest_freq_C18O
	return bbody(t,nu)*(1.0-math.exp(-tau(t,nu,nu0,nco,vturb,angle)))*scaling
#------------------------------------------------------------------------------------------------
#
# Continuum level
#
#------------------------------------------------------------------------------------------------

def intensity_continuum_level(t,nu,nu0,alpha,i0,nco,vturb,angle):
	if isotopologue=="12CO":
		scaling=scaling12
		nulevel=rest_freq_12CO

	if isotopologue=="13CO":
		scaling=scaling13
		nulevel=rest_freq_13CO

	if isotopologue=="C18O":
		scaling=scaling18
		nulevel=rest_freq_C18O
	return i0*math.exp(-tau(t,nu,nu0,nco,vturb,angle))*(nu/nulevel)**alpha


#------------------------------------------------------------------------------------------------
#
# reading temperature in Kelvins
#
#------------------------------------------------------------------------------------------------

def brigth_temp(Pv,nu,name):
	if name=="12CO":
		scaling=scaling12
	elif name=="13CO":
		scaling=scaling13
	elif name=="C18O":
		scaling=scaling18
	return 4.79922e-11*nu/(math.log((1.4745e-50*nu**3.0)/(Pv/scaling)+1))

#------------------------------------------------------------------------------------------------
#
# Lorentzian profile
#
#------------------------------------------------------------------------------------------------
def lorentzian_profile(t,name,nu0,frequency):
	freq=frequency
	profile=np.empty_like(frequency)
	density=1e11
	velocity=math.sqrt(2*k*t/ccs.molecular_mass(name))*100 #velocity in cm/s
	collision_cross=ccs.cross_section(name)
	collision_freq=density*velocity*collision_cross

	if name=="C18O":
		gamma=A_C18+2*collision_freq
	elif name=="12CO":
		gamma=A_C12+2*collision_freq
	elif name=="13CO":
		gamma=A_C13+2*collision_freq
	for i in range(len(freq)):
		profile[i]=(gamma/( (freq[i]-nu0)**2+(gamma/(4*math.pi))**2 ))/(4*math.pi**2)

	return profile
