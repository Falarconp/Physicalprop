"""
This code provides the functions that are necessary in order to get the
gas density structure, turbulent velocity and temperature profiles
of a protoplanetary disk.

Code created by Felipe Alarcon
Minimize chi2
"""

#    alpha=1

import collision_cross_section	as ccs				# auxiliar quantities
import help_functions as hf					# finding maximum
from constants import *

import numpy as np
import scipy as sp
from scipy.integrate import quad
import pyfits
import matplotlib.pyplot as plt
import scipy.ndimage.measurements as meas
import glob
import os
import math
from iminuit import Minuit
import sys
from multiprocessing import Pool

# isotopologue = raw_input("isotopologo")

# frac = ccs.molecular_fraction(isotopologue)			#getting molecular fraction
# m = ccs.molecular_mass(isotopologue) 				#getting molecular mass in Kg
# cross_sect = ccs.cross_section(isotopologue)			#getting cross section
# sigma = cross_sect/(m*10**(9))   					# efective area in cm2/micrograms
# kdivm = k/m     							# boltzman constant divided by mass

d12 = 156400.1727294922
nu12i = 345796068100.0849 - d12*29
nu12f = 345796068100.0849 + d12*30

d13 = 1.495217280883789e5
nu13i = 3.305880399826639e11 - d13*29
nu13f = 3.305880399826639e11 + d13*30

d18 = 148953.013671875
nu18i = 329330626976.5068 - d18*29
nu18f = 329330626976.5068 + d18*30


def datafits(namefile):
    """
    Open a FITS image and return datacube and header, namefile without '.fits'
    """
    datacube = pf.open(namefile + '.fits')[0].data
    hdr = pf.open(namefile + '.fits')[0].header
    return datacube, hdr


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
    bminaxis =  0.1
    bmajaxis = 0.1
    scaling = (1e26*bmajaxis*bminaxis*math.pi/(4*math.log(2)))/(4.25e10)
    # if iso==12:
    #     scaling=scaling12
    # elif iso==13:
    #     scaling=scaling13
    # elif iso==18:
    #     scaling=scaling18
    #def tau2(t,nu,nu0,nco,vturb):
    blackbody = bbody(T,nu)*(1.0-sp.exp(-tau(T,nu,nu0,nco,vturb,angle, iso)))*scaling
    thermal = i0*sp.exp(-tau(T,nu,nu0,nco,vturb,angle, iso))*(nu/nu0)**alpha
    return  blackbody + thermal


def flux_int(T, alpha, i0, nco, vturb, angle, datos, hdr12, hdr13, hdr18):
    """
    Chi^2 obtained integrating function from radiative transfer
    in the spectral window of each isotopologue divided by molecular fractions.
    """
    nu013 = 3.305880399826639e11
    nu012 = 345796068100.0849
    nu018 = 329330626976.5068
    integral12 = quad(intensity, nu12i, nu12f, args=(T, nu012 ,alpha, nco, vturb, angle, i0, hdr12, 12))
    integral13 = quad(intensity, nu13i, nu13f, args=(T, nu013 ,alpha, nco/70., vturb, angle, i0, hdr13, 13))
    integral18 = quad(intensity, nu18i, nu18f, args=(T, nu018 ,alpha, nco/500., vturb, angle, i0, hdr18, 18))
    chi2 = sp.sum((datos[0]-integral12[0]*nu012/c)**2) + sp.sum((datos[1]-integral13[0]*nu013/c)**2) + ((datos[2]-integral18[0]*nu018/c)**2)
    return chi2


def minprofilesfitmoment(image12CO, image13CO, imageC18O, r=80):
    """
    Fits a temperature profile, turbulent velocity and column density
    using three CO isotopologues lines with iminuit package.
    """
    print('Opening FITS images and fitting functions')

#  Isotopologue images
    cubo12, head12 = datafits(image12CO)
    cubo13, head13 = datafits(image13CO)
    cubo18, head18 = datafits(imageC18O)

    std12 = 0.0055880332*sp.sqrt(43.)
    std13 = 0.0059544998*sp.sqrt(28.)
    std18 = 0.0041698562*sp.sqrt(23.)
    stderr = max(std12, std13, std18)

    alpha = 2.3
    angle = 28*sp.pi/180
    i0=0.
    Temperature = sp.zeros((256,256))
    Denscol = sp.zeros((256,256))
    Turbvel = sp.zeros((256,256))
    errpars = sp.zeros((256,256,3))
    errmodel = sp.zeros((256,256))

    def iterate(num):
        i = num/256
        j = num%256
# km/s to mks
        c12 = cubo12[i][j]*1000.
        c13 = cubo13[i][j]*1000.
        c18 = cubo18[i][j]*1000.

        data = [c12, c13, c18]
        if (i-128)**2 + (j-128)**2 > r**2 or data==[0,0,0] :
            return

# iminuit least squares fit
            # f = lambda Temp,vturb,NcolCO, i0: flux_int(Temp, alpha, i0, NcolCO,
            #                                            vturb, angle, data, head12,
            #                                            head13, head18)
            # m = Minuit(f, Temp=20, vturb=300, NcolCO=1e19, i0=.1, errordef=stderr,
            #             error_Temp=.5, error_NcolCO=1e12, error_vturb=5, error_i0=0.001,
            #             limit_Temp=(10,500), limit_vturb=(0, 10000),
            #             limit_NcolCO=(0, 1e20))

        f = lambda Temp,vturb,NcolCO: flux_int(Temp*100., alpha, i0, 10**19*NcolCO,
                                                       600.*vturb, angle, data, head12,
                                                       head13, head18)
        m = Minuit(f, Temp=0.2, vturb=0.1, NcolCO=0.02, errordef=stderr,
                    error_Temp=.02, error_NcolCO=0.005, error_vturb=0.005,
                    limit_Temp=(0.01,1), limit_vturb=(0.001, 1), limit_NcolCO=(0.001, 1))

        m.migrad(1e9)
        fit = [m.values['Temp']*100., m.values['vturb']*600., 10**24*m.values['NcolCO']];
        error = [m.errors['Temp']*100., m.errors['vturb']*600., 10**24*m.errors['NcolCO']];
        errmod = f(fit[0], fit[1], fit[2])
#            errmod = f(m.values['Temp'], m.values['vturb'], m.values['NcolCO'])

        Temperature[i,j] = fit[0]
        Denscol[i,j] = fit[2]
        Turbvel[i,j] = fit[1]
        errpars[i,j] = error
        errmodel[i,j] = errmod
        return

    pool = Pool(3)
    pool.map(iterate, range(65536))

# Parameter error (confidence intervals)
    Temperature = sp.array(Temperature);
    Turbvel = sp.array(Turbvel);
    Denscol = sp.array(Denscol);
    errpars = sp.array(errpars)
    errmodel= sp.array(errmodel)
    print('Calculated Fits')
    err_temp = errpars[:, :, 0]
    err_v_turb = errpars[:, :, 1]
    err_NCO = errpars[:, :, 2]

    r2 = pf.PrimaryHDU(Temperature)
    r3 = pf.PrimaryHDU(Turbvel)
    r4 = pf.PrimaryHDU(Denscol)
    r5 = pf.PrimaryHDU(errmodel)
    r6 = pf.PrimaryHDU(err_temp)
    r7 = pf.PrimaryHDU(err_v_turb)
    r8 = pf.PrimaryHDU(err_NCO)
    head1 = head12
    head2 = head13
    head3 = head13
    r2.header = head1;
    head1['BUNIT'] = 'K'
    head2['BUNIT'] = 'cms/s'
    head2['BTYPE'] = 'Velocity'
    head3['BUNIT'] = 'm-2'
    head3['BTYPE'] = 'Column Density'
    r3.header = head2
    r4.header = head3
    r6.header = head1
    r7.header = head2
    r8.header = head3
    inputimage = 'Sims'
    out2 = inputimage + '_Temp.fits'
    out3 = inputimage + '_v_turb.fits'
    out4 = inputimage + '_NCO.fits'
    out5 = inputimage + '_errpars_temp.fits'
    out6 = inputimage + '_errfit.fits'
    out7 = inputimage + '_errpars_vturb.fits'
    out8 = inputimage + '_errpars_NCO.fits'
    print('Writing images')
    r2.writeto(out2, clobber=True)
    r3.writeto(out3, clobber=True)
    r4.writeto(out4, clobber=True)
    r5.writeto(out5, clobber=True)
    r6.writeto(out6, clobber=True)
    r7.writeto(out7, clobber=True)
    r8.writeto(out8, clobber=True)
    pf.writeto(out2, Temperature, head1, clobber=True)
    pf.writeto(out3, Turbvel, head2, clobber=True)
    pf.writeto(out4, Denscol, head3, clobber=True)
    pf.writeto(out5, err_temp, head1, clobber=True)
    pf.writeto(out6, errmodel, head1, clobber=True)
    pf.writeto(out7, err_v_turb, head2, clobber=True)
    pf.writeto(out8, err_NCO, head3, clobber=True)
    return


image13CO = '/home/felipe/Escritorio/ID/HD142527/Simulations_CO3-2/Vortex_image_13co32_o500_i159.00_phi90.00_PA-24.00.fits_MOM0'
image12CO = '/home/felipe/Escritorio/ID/HD142527/Simulations_CO3-2/Vortex_image_12co32_o500_i159.00_phi90.00_PA-24.00.fits_MOM0'
imageC18O = '/home/felipe/Escritorio/ID/HD142527/Simulations_CO3-2/Vortex_image_c18o32_o500_i159.00_phi90.00_PA-24.00.fits_MOM0'


minprofilesfitmoment(image12CO, image13CO, imageC180, r=10)
