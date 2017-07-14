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

# constants in SI units
h=6.62606957*10**(-34)          # [Jxs] Planck's constant
c=2.99792458*10**8                 # [m/s] light speed
k=1.3806488*10**(-23)               # [J/K] Boltzmann's constant

nu12i = 2.305273879105E+11+ 27.*1.537983987427E+05
nu12f = 2.305273879105E+11 + 75.*1.537983987427E+05
nu13i = 2.204030910252E+11 - 34.*1.470341725464E+05
nu13f = 2.204030910252E+11 - 62.*1.470341725464E+05
nu18i = 2.195647542472E+11 - 37.*1.464749057007E+05
nu18f = 2.195647542472E+11 - 60.*1.464749057007E+05
std = 0.
frac = 1.


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


def phi2(t,nu,nu0,vturb,m):
    deltav = (nu0/c)*math.sqrt(2.0*k*t/m + 2.0*vturb**2 )**(0.5)
    phi0 = deltav*math.sqrt(math.pi)
    gauss=sp.exp(-((nu-nu0)**2.0)/(2*(deltav**2.0)))
    return gauss/phi0


def tau(t,nu,nu0,nco,vturb,angle,iso=12):
    """
    Optical depth with turbulent velocity and thermal velocity
    sigma is cross area cm^2
    """
    #return nco*frac*sigma*phi(t,nu,nu0,vturb)
    sigma = ccs.cross_section("13CO")
    m = ccs.cross_section("13CO")
    if iso==12:
        sigma = ccs.cross_section("12CO")
        m = ccs.molecular_mass("12CO")
    elif iso==13:
        sigma = ccs.cross_section("13CO")
        m = ccs.molecular_mass("13CO")
    elif iso==18:
        sigma = ccs.cross_section("C18O")
        m = ccs.molecular_mass("C18O")
#    sigma = sigma/(m*10**9)
    return nco*sigma*phi(t,nu,nu0,vturb,angle, m)/math.cos(math.pi-incli)


def tau2(t,nu,nu0,nco,vturb, iso=12):
    sigma = ccs.cross_section("13CO")
    m = ccs.cross_section("13CO")
    if iso==12:
        frac=1.
        sigma = ccs.cross_section("12CO")
        m = ccs.molecular_mass("12CO")
    elif iso==13:
        frac = 1/80.
        sigma = ccs.cross_section("13CO")
        m = ccs.molecular_mass("13CO")
    elif iso==18:
        frac = 1/500.
        sigma = ccs.cross_section("C18O")
        m = ccs.molecular_mass("C18O")
    return nco*frac*sigma*phi2(t,nu,nu0,vturb,m)


def intensity(nu, T, nu0, alpha, nco, vturb, angle, i0, hdr, iso=12):
    """
    Solution to radiative transfer equation
    """
    bminaxis =  hdr['BMIN']
    bmajaxis = hdr['BMAJ']
    scaling = (1e26*bmajaxis*bminaxis*(3600**2)*math.pi/(4*math.log(2)))/(4.25e10)
    blackbody = bbody(T,nu)*(1.0-sp.exp(-tau(T,nu,nu0,nco,vturb,angle, iso)))*scaling
    cont = i0*sp.exp(-tau(T,nu,nu0,nco,vturb,angle, iso))*(nu/nu0)**alpha
    return  blackbody + cont


def intensity_err(nu, nu0, T, alpha, i0, nco, vturb, angle, datos, hdr, iso=12):
    """
    Chi squared of data with fit of spectral line, both normalized.
    """
    if iso==13:
        nco/=80.
    elif iso==18:
        nco/=560.
    cloud_abs = sp.exp(-tau2(T_cloud, nu, nu_cloud, nco_cloud, v_cloud, iso))
    weights = sp.ones(len(nu))
    weights[datos<=2*std] = 0.5
    weights[datos<=1*std] = 0.2
    model = intensity(nu, T, nu0, alpha, nco, vturb, angle, i0, hdr, iso)*cloud_abs
    model /= model.sum()
    aux = (datos-model)**2
    chi = aux.sum()
    return chi


def flux_int(T, alpha, i0, nco, vturb, angle, datos, hdr12, hdr13, hdr18):
    """
    Chi^2 obtained integrating function from radiative transfer
    in the spectral window of each isotopologue divided by molecular fractions.
    """
    nu013 = (nu13i+nu13f)/2.
    nu012 = (nu12i+nu12f)/2.
    nu018 = (nu18i+nu18f)/2.
    integral12 = quad(intensity, nu12i, nu12f, args=(T, nu012 ,alpha, nco, vturb, angle, i0, hdr12, 12))
    integral13 = quad(intensity, nu13i, nu13f, args=(T, nu013 ,alpha, nco/70., vturb, angle, i0, hdr13, 13))
    integral18 = quad(intensity, nu18i, nu18f, args=(T, nu018 ,alpha, nco/500., vturb, angle, i0, hdr18, 18))
    chi2 = sp.sum((datos[0]-integral12[0])**2) + sp.sum((datos[1]-integral13[0])**2) + ((datos[2]-integral18[0])**2)
    return chi2



def minlinesfitmoment(image, iso=12, r=80):
    """
    Fits a temperature profile, turbulent velocity and column density
    using three CO isotopologues lines with iminuit package.
    """
    print('Opening FITS images and fitting functions')

#  Isotopologue image and centroid map
    cubo, head = datafits(image)
    if iso!=13:
        cubo=cubo[0]

    if iso==12:
        cubo = cubo[28:77,:,:]
        nu0 = (nu12i+nu12f)/2.
        nu = sp.linspace(nu12i, nu12f, 49)
        std = 0.0055880332
        model = sp.zeros((512,512,49))
        f=1.
        dnu = abs(nu[-1] - nu[-2])
        ncoscale = 5e14
    elif iso==13:
        cubo = cubo[35:64,:,:]
        nu0 = (nu13i+nu13f)/2.
        nu= sp.linspace(nu13i, nu13f, 29)
        std = 0.0059544998
        model = sp.zeros((512,512,29))
        dnu = abs(nu[-1] - nu[-2])
        f=1e-2
        ncoscale = 5e14
    else:
        cubo = cubo[38:62,:,:]
        nu0 = (nu18i+nu18f)/2.
        nu = sp.linspace(nu18i, nu18f, 24)
        std = 0.0041698562
        dnu = abs(nu[-1] - nu[-2])
        ncoscale = 5e14
        model = sp.zeros((512,512,24))
        f=0.0017

    contador = 0
    for i in range(256):
        for j in range(256):
            datos = cubo[:,i+128,j+128]
            contador += nu[datos==datos.min()]
    nu_cloud = contador/(256.*256.)

#Change to mks and m1 from velocity to frequency
    cubomax = cubo.max()

    alpha = 2.3
    angle = 28.*sp.pi/180.
    i0=0.
    Temperature = sp.zeros((512,512))
    Denscol = sp.zeros((512,512))
    Turbvel = sp.zeros((512,512))
    errpars = sp.zeros((512,512,3))
    errmodel = sp.zeros((512,512))
    arange = sp.arange(len(nu))


    for i in range(512):
        for j in range(512):
# km/s to mks
            data = cubo[:,i,j]

            datamax = data.max()
            stdc = std/datamax
            index = arange[data==datamax]
            indexi = max(0, index-4)
            centroid = nu[index]
            indexf = min(len(nu), index+4)
            indexes = sp.arange(indexi,indexf)
            aux = data[indexi:indexf]
            indexes = indexes[aux>=0]
            data = data[indexes]
            nus = nu[indexes]
            if (i-256)**2 + (j-256)**2 > r**2 :
                continue

#             if iso!=18 :
# #leer Tdust, nco from dust, vturb and nu0
#                 T_cont = T18[i,j]
#                 vturb_cont = vturb18[i,j]
#                 nco_cont = nco18[i,j]
#                 data-= intensity(nus, T_cont, fit[3], alpha, vturb_cont, nco_cont, angle, i0, head, iso)

            s = data.sum()
            data/= data.sum()


            r_ij = sp.sqrt((i-256)**2 +(j-256)**2)

            if r_ij>20 and r_ij<50:
                tscale = 120.
#      ncoscale *= (r_ij*280/128.)**(-1/1.5)*0.1
                t0 = 304*(r_ij*150/80.)**(-1/1.2)/120.
                tlim =(0.1,1.)
            elif r_ij>=50.:
                tscale = 80.
#            ncoscale *= (r_ij*280/128.)**(-1/1.5)
                t0 = 400*(r_ij*150./80.)**(-1/1.2)/100.
                tlim =(0.05,1.)
            else:
                tscale=250.
                t0 = 0.8
                tlim = (0.02,1.)

# iminuit least squares fit
            f = lambda Temp,vturb,NcolCO, nu0: intensity_err(nus, nu0, Temp*tscale, alpha, i0, ncoscale*NcolCO,
            600.*vturb, angle, data, head, iso)
            m = Minuit(f, Temp=t0, vturb=0.03, NcolCO=0.000001, nu0=centroid, errordef=stdc,
                        error_Temp=.001, error_NcolCO=0.0000005, error_vturb=0.005,
                        error_nu0=dnu/10., limit_Temp=tlim, limit_vturb=(0.002, 1),
                        limit_NcolCO=(0.0000003, 1), limit_nu0=(centroid-2*dnu, centroid+2*dnu))

            m.migrad(1e9)
            errmod = f(m.values['Temp'], m.values['vturb'], m.values['NcolCO'], m.values['nu0'])
            fit = [m.values['Temp']*tscale, m.values['vturb']*600., ncoscale*m.values['NcolCO'], m.values['nu0']];
            error = [m.errors['Temp']*tscale, m.errors['vturb']*600., ncoscale*m.errors['NcolCO']];

            Temperature[i,j] = fit[0]
            Denscol[i,j] = fit[2]
            Turbvel[i,j] = fit[1]
            errpars[i,j] = error
            errmodel[i,j] = errmod
            model[i,j] = intensity(nu, fit[0], fit[3], alpha, fit[2], fit[1], angle, i0, head, iso)
            model[i,j] = model[i,j]/(model[i,j].sum())*2.*s

# Parameter error (confidence intervals)
    Temperature = sp.nan_to_num(Temperature)
    Turbvel = sp.nan_to_num(Turbvel)
    Denscol = sp.nan_to_num(Denscol)
    errpars = sp.nan_to_num(errpars)
    errmod = sp.nan_to_num(errmod)
    errmodel= sp.array(errmodel)
    model = sp.array(model)
    model = sp.swapaxes(model, 0,1)
    model = sp.swapaxes(model,0,2)
    print('Calculated Fits')
    err_temp = errpars[:, :, 0]
    err_v_turb = errpars[:, :, 1]
    err_NCO = errpars[:, :, 2]

    r1 = pf.PrimaryHDU(model)
    r2 = pf.PrimaryHDU(Temperature)
    r3 = pf.PrimaryHDU(Turbvel)
    r4 = pf.PrimaryHDU(Denscol)
    r5 = pf.PrimaryHDU(errmodel)
    r6 = pf.PrimaryHDU(err_temp)
    r7 = pf.PrimaryHDU(err_v_turb)
    r8 = pf.PrimaryHDU(err_NCO)
    head1 = head
    head2 = head
    head3 = head
    r1.header = head
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
    inputimage = '/home/felipe/Escritorio/ID/HD142527/Physical_condition_32/lines_' + str(iso)
    out1 = inputimage + '_model.fits'
    out2 = inputimage + '_Temp.fits'
    out3 = inputimage + '_v_turb.fits'
    out4 = inputimage + '_NCO.fits'
    out5 = inputimage + '_errpars_temp.fits'
    out6 = inputimage + '_errfit.fits'
    out7 = inputimage + '_errpars_vturb.fits'
    out8 = inputimage + '_errpars_NCO.fits'
    print('Writing images')
    r1.writeto(out1, clobber=True)
    r2.writeto(out2, clobber=True)
    r3.writeto(out3, clobber=True)
    r4.writeto(out4, clobber=True)
    r5.writeto(out5, clobber=True)
    r6.writeto(out6, clobber=True)
    r7.writeto(out7, clobber=True)
    r8.writeto(out8, clobber=True)
    pf.writeto(out1, model, head, clobber=True)
    pf.writeto(out2, Temperature, head1, clobber=True)
    pf.writeto(out3, Turbvel, head2, clobber=True)
    pf.writeto(out4, Denscol, head3, clobber=True)
    pf.writeto(out5, err_temp, head1, clobber=True)
    pf.writeto(out6, errmodel, head1, clobber=True)
    pf.writeto(out7, err_v_turb, head2, clobber=True)
    pf.writeto(out8, err_NCO, head3, clobber=True)
    return


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

    Temperature = []
    Denscol = []
    Turbvel = []
    errpars = []
    errmodel = []


    for i in range(len(cubo12)):
        Temp = []
        Density = []
        vturb = []
        e = []
        model = []
        for j in range(len(cubo12[0])):
            if (i-256)**2 + (j-256)**2 > r**2:
                e.append([0,0,0])
                Temp.append(0)
                Density.append(0)
                vturb.append(0)
                model.append(0)
                continue

# km/s to mks
            c12 = cubo12[i][j]/1000.
            c13 = cubo13[i][j]/1000.
            c18 = cubo18[i][j]/1000.

            data = [c12, c13, c18]
            alpha = 2.3
            angle = 28*sp.pi/180.
            i0 = 0

# iminuit least squares fit
            # f = lambda Temp,vturb,NcolCO, i0: flux_int(Temp, alpha, i0, NcolCO,
            #                                            vturb, angle, data, head12,
            #                                            head13, head18)
            # m = Minuit(f, Temp=20, vturb=300, NcolCO=1e19, i0=.1, errordef=stderr,
            #             error_Temp=.5, error_NcolCO=1e12, error_vturb=5, error_i0=0.001,
            #             limit_Temp=(10,500), limit_vturb=(0, 10000),
            #             limit_NcolCO=(0, 1e20))

            f = lambda Temp,vturb,NcolCO: flux_int(Temp*80., alpha, i0, 10**(NcolCO*20.),
                                                       700.*vturb, angle, data, head12,
                                                       head13, head18)
            m = Minuit(f, Temp=0.2, vturb=0.1, NcolCO=0.95, errordef=stderr,
                        error_Temp=0.01, error_NcolCO=0.05, error_vturb=0.05,
                        limit_Temp=(0.01,1), limit_vturb=(0, 1),
                        limit_NcolCO=(0, 1))

            m.migrad(1e6)
            fit = [m.values['Temp']*80., m.values['vturb']*700., 10**(m.values['NcolCO']*20.)];
            error = [m.errors['Temp']*80., m.errors['vturb']*700., m.errors['NcolCO']];
#            errmod = f(fit[0], fit[1], fit[2], m.values['i0'])
            errmod = f(m.values['Temp'], m.values['vturb'], m.values['NcolCO'])

# Parameter error (confidence intervals)

            e.append(error)
            Temp.append(fit[0])
            vturb.append(fit[1])
            Density.append(fit[2])
            model.append(errmod)

        Temperature.append(Temp)
        Denscol.append(Density)
        Turbvel.append(vturb)
        errpars.append(e)
        errmodel.append(model)

    Temperature = sp.array(Temperature);
    Turbvel = sp.array(Turbvel);
    Denscol = sp.array(Denscol);
    errpars = sp.array(errpars)
    errmodel= sp.array(errmodel)
    model = sp.nan_to_num(model)
    Denscol = sp.nan_to_num(Denscol)
    Turbvel = sp.nan_to_num(Turbvel)
    Temperature = sp.nan_to_num(Temperature)
    print('Calculated Moments')
    errpars = sp.swapaxes(errpars,0,1);
    errpars = sp.swapaxes(errpars,0,2);
    err_temp = errpars[0, :, :]
    err_v_turb = errpars[1, :, :]
    err_NCO = errpars[2, :, :]

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
    inputimage = 'HD142527'
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


image12CO = '/home/felipe/Escritorio/ID/HD142527/HD142527_12CO21_cont.clean'
image13CO = '/home/felipe/Escritorio/ID/HD142527/HD142527_13CO_contrest'
imageC18O = '/home/felipe/Escritorio/ID/HD142527/HD142527_C18O21.clean'

mom1_C18O = '/home/felipe/Escritorio/ID/HD142527/HD142527_C18O21.clean_MOM1'
mom1_13CO = '/home/felipe/Escritorio/ID/HD142527/HD142527_13CO_contrest_MOM1'
mom1_12CO = '/home/felipe/Escritorio/ID/HD142527/HD142527_12CO21_cont.clean_MOM1'

minlinesfitmoment(imageC18O, r=120, iso=18)
#minlinesfitmoment(image13CO, r=80, iso=13)
#minlinesfitmoment(image12CO, mom1_12CO, r=80, iso=12)
