"""
This code provides gives temperature in the disk using emission from three isotopologues
of CO in aa protoplanetary disk.

Code created by Felipe Alarcon
Minimize chi2
"""

#    alpha=2.3
#    Chequear como quitar el continuo de emisiones de CO y perfilar, se puede estar infravalorando la columna de CO


import collision_cross_section	as ccs				# auxiliar quantities
import help_functions as hf					# finding maximum
from constants import *

import numpy as np
import scipy as sp
from scipy.integrate import quad
import astropy.io.fits as pf
import matplotlib.pyplot as plt
import scipy.ndimage.measurements as meas
import glob
import os
import math
from iminuit import Minuit
import sys
from multiprocessing import Pool

isotopologue = raw_input("¿isotopologue?")

frac = ccs.molecular_fraction(isotopologue)			#getting molecular fraction
m = ccs.molecular_mass(isotopologue) 				#getting molecular mass in Kg
cross_sect = ccs.cross_section(isotopologue)			#getting cross section
sigma = cross_sect/(m*10**(9))   					# efective area in cm^2/micrograms
kdivm = k/m     							# boltzman constant divided by mass

nu12i = 2.30532001862e11
nu12f = 2.30538615193e11
nu13i = 2.20393827872e11
nu13f = 2.20397944829e11
nu18i = 2.19555967397e11
nu18f = 2.19559335690e11


def bbody(t,nu):
    """
    Blackbody flux for a given temperature and frequency W/(m^2*Hz*sr) (mks system)
    """
    return (1.4745e-50)*nu**3.0/(sp.exp(4.79922e-11*nu/t)-1.0)


def phi(t,nu,nu0,vturb,angle, m):
    pa = (-20+90+angle)*math.pi/180.0  #-20 is the PA from east of north, and 90 to get the semi-minor axis
    Q = 1
    shear = (Q*math.sin(incli)**2)*(math.tan(incli)**2)*math.sin(2*pa)**2
    deltav = (nu0/c)*math.sqrt(2.0*k*t/m + 2.0*vturb**2 )*(1+shear)**(0.5)
    #deltav= (nu0/c)*math.sqrt(2.0*kdivm*t+2.0*vturb**2 )
    phi0 = deltav*math.sqrt(math.pi)
    gauss=sp.exp(-((nu-nu0)**2.0)/(2*(deltav**2.0)))
    return gauss/phi0


def tau(t,nu,nu0,nco,vturb,angle,iso=12):
    """
    Optical depth with turbulent velocity and thermal velocity
    """
    #return nco*frac*sigma*phi(t,nu,nu0,vturb)
    if iso==12:
        sigma = ccs.cross_section("12CO")
        m = ccs.molecular_mass("12CO")
    elif iso==13:
        sigma = ccs.cross_section("13CO")
        m = ccs.molecular_mass("13CO")
    elif iso==18:
        sigma = ccs.cross_section("C18O")
        m = ccs.molecular_mass("C18O")
    sigma = sigma/(m*10**9)
    return nco*sigma*phi(t,nu,nu0,vturb,angle, m)/math.cos(incli)


def intensity(nu, T, nu0, alpha, nco, vturb, angle, i0, hdr, iso=12):
    """
    Solution to radiative transfer equation
    """
    bminaxis =  hdr['BMIN']
    bmajaxis = hdr['BMAJ']
    scaling = (1e26*bmajaxis*bminaxis*(3600**2)*math.pi/(4*math.log(2)))/(4.25e10)
    # if iso==12:
    #     scaling=scaling12
    # elif iso==13:
    #     scaling=scaling13
    # elif iso==18:
    #     scaling=scaling18
    blackbody = bbody(T,nu)*(1.0-sp.exp(-tau(T,nu,nu0,nco,vturb,angle, iso)))*scaling
    thermal = i0*sp.exp(-tau(T,nu,nu0,nco,vturb,angle, iso))*(nu/nu0)**alpha
    return  blackbody + thermal


#def tau2(t,nu,nu0,nco,vturb):
#    return nco*frac*sigma*phi2(t,nu,nu0,vturb)    ### optical depth using the voigt profile line shape


def flux_int(nui, nuf, T, alpha, i0, nco, vturb, angle, datos, hdr, iso=12):
    """
    Chi^2 obtained integrating function from radiative transfer
    in the spectral window of each isotopologue divided by molecular fractions.
    """
    nu013 = (nu13i+nu13f)/2.
    nu012 = (nu12i+nu12f)/2.
    nu018 = (nu18i+nu18f)/2.
    if iso==12:
        integral = quad(intensity, nu12i, nu12f, args=(T, nu012 ,alpha, nco, vturb, angle, i0, hdr, 12))
    elif iso==13:
        integral = quad(intensity, nu13i, nu13f, args=(T, nu013 ,alpha, nco/70., vturb, angle, i0, hdr, 13))
    elif iso==18:
        integral = quad(intensity, nu18i, nu18f, args=(T, nu018 ,alpha, nco/500., vturb, angle, i0, hdr, 18))
    chi2 = sp.sum((datos-integral[0])**2)
    return chi2


def datafits(namefile):
    """
    Open a FITS image and return datacube and header, namefile without '.fits'
    """
    datacube = pf.open(namefile + '.fits')[0].data
    hdr = pf.open(namefile + '.fits')[0].header
    return datacube, hdr



def minprofilesfitmoment(image13CO, imageC18O, r=80):
    """
    Fits a temperature profile, turbulent velocity and column density
    using three CO isotopologues lines with iminuit package.
    """
    print('Opening FITS images and fitting functions')

#  Isotopologue images
#    cubo12, head12 = datafits(image12CO)
    cubo13, head13 = datafits(image13CO)
    cubo18, head18 = datafits(imageC18O)

#    std12, hstd12 = datafits(err12CO)
    std13, hstd13 = datafits(err13CO)
    std18, hstd18 = datafits(errC18O)

    TC18O = []
    T13CO = []
#    T12CO = []
    errpars = []
    errmodel = []


    for i in range(len(cubo)):
        t18 = []
#        t12 = []
        t13 = []
        e = []

        # p = Pool(3)
        # result = p.map(paralellprocess, range(len(cubo[0])))
        # p.close()

        for j in range(len(cubo[0])):
            if (i-256)**2 + (j-256)**2 > r**2:
                e.append([0,0,0])
                t18.append(0)
#                t12.append(0)
                t13.append(0)
                continue

#            err12 = std12[i][j]
            err13 = std13[i][j]
            err18 = std18[i][j]

#            c12 = cubo12[i][j]/1000.
            c13 = cubo13[i][j]/1000.
            c18 = cubo18[i][j]/1000.

            alpha = 2.3
            angle = 28*sp.pi/180.

# Escribir valores para vturb, NcolCO Array??
# iminuit least squares fit
            # f12 = lambda T12: flux_int(nu12i, nu12f, T12, alpha,
            #                            i0, NcolCO, vturb, angle, head12, 12)
            # m12 = Minuit(f12, T12=20, errordef=err12,
            #             error_T12=.5, limit_T12=(0,500))
            # m12.migrad(1e6)

            f13 = lambda T13, alpha, i0, NcolCO, vturb,: flux_int(nu13i, nu13f,
                                                                  T13*100, alpha,
                                                                  i0*1e-2, NcolCO*1e12,
                                                                  vturb*600, angle,
                                                                  head13, 13)
            m13 = Minuit(f13, T13=0.2, alpha=1.7, i0=0.1, vturb=0.4, NcolCO=
                         errordef=err13, error_T13=0.02, error_alpha=0.1,
                         error_io= ,error_vturb=0.03, error_NcolCO=0.01,
                         limit_T13=(0.01,1), limit_alpha=(0.1,2.5), limit_io=(0.001,1),
                         limit_NcolCO=(0,1), limit_vturb=(0.01,1))
            m13.migrad(1e6)

            f18 = lambda T18: flux_int(nu18i, nu18f, T18*100, alpha,
                                       i0, NcolCO, vturb, angle, head18, 18)
            m18 = Minuit(f18, T18=0.2, errordef=err18,
                        error_T18=.05, limit_T18=(0,1))
            m18.migrad(1e6)

#            error = [m13.errors['T13'], m18.errors['T18'], m12.errors['T12']];
            error = [m13.errors['T13'], m18.errors['T18']]


            e.append(error)
            t18.append(m18.values['T18']*100)
            t13.append(m13.values['T13']*100)
#            t12.append(m12.values['T12'])

        TC18O.append(t18)
        T13CO.append(t13)
#        T12CO.append(t12)
        errpars.append(e)

    TC18O = sp.array(TC18O);
    T13CO = sp.array(T13CO);
#    T12CO = sp.array(T12CO);
    errpars = sp.array(errpars)
    print('Calculated Moments')
    errpars = sp.swapaxes(errpars,0,1);
    errpars = sp.swapaxes(errpars,0,2);
    err_TC18O = errpars[1, :, :]
#    err_T12CO = errpars[1, :, :]
    err_T13CO = errpars[0, :, :]

    r2 = pf.PrimaryHDU(TC18O)
#    r3 = pf.PrimaryHDU(T12CO)
    r4 = pf.PrimaryHDU(T13CO)
    r6 = pf.PrimaryHDU(err_TC18O)
    r7 = pf.PrimaryHDU(err_T13CO)
#    r8 = pf.PrimaryHDU(err_T12CO)
    head1 = head13
    r2.header = head1;
    head1['BUNIT'] = 'K'
    head1['BTYPE'] = 'Temperature'
#    r3.header = head1
    r4.header = head1
    r6.header = head1
    r7.header = head1
#    r8.header = head1
    outputimage ='Temp_profile/HD142527'
    out2 = outputimage + '_TC18O.fits'
#    out3 = inputimage + '_T12CO.fits'
    out4 = outputimage + '_T13CO.fits'
    out6 = outputimage + '_errpars_TC18O.fits'
    out7 = outputimage + '_errpars_T13CO.fits'
#    out8 = inputimage + '_errpars_T12CO.fits'
    print('Writing images')
    r2.writeto(out2, clobber=True)
#    r3.writeto(out3, clobber=True)
    r4.writeto(out4, clobber=True)
    r6.writeto(out6, clobber=True)
    r7.writeto(out7, clobber=True)
#    r8.writeto(out8, clobber=True)
    pf.writeto(out2, TC18O, head1, clobber=True)
    pf.writeto(out3, T12CO, head2, clobber=True)
    pf.writeto(out4, T13CO, head3, clobber=True)
    pf.writeto(out6, err_TC18O, head1, clobber=True)
    pf.writeto(out8, err_T12CO, head3, clobber=True)
    pf.writeto(out7, err_T13CO, head2, clobber=True)
    return


input13CO = '/home/felipe/Escritorio/ID/HD142527/HD142527_13CO_contrest'
input12CO = '/home/felipe/Escritorio/ID/HD142527/HD142527_12CO.clean'
inputC180 = '/home/felipe/Escritorio/ID/HD142527/HD142527_C18O.clean'
minprofilesfitmoment(input13CO, inputC180)
