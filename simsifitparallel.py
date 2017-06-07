"""
This code provides the functions that are necessary in order to get the
gas density structure, turbulent velocity and temperature profiles
of a protoplanetary disk.

Code adapted by Felipe Alarcon
Minimize chi2
"""

####################       CAMBIAR SHAPE NU



import collision_cross_section	as ccs				# auxiliar quantities
import help_functions as hf					# finding maximum
from constants import *

import numpy as np
import scipy as sp
from scipy.integrate import quad
import pyfits as pf
import matplotlib.pyplot as plt
import scipy.ndimage.measurements as meas
import glob
import os
import math
from iminuit import Minuit
import sys
from astropy.convolution import Gaussian2DKernel, convolve_fft
from multiprocessing import Pool


# constants in SI units
h=6.62606957*10**(-34)          # [Jxs] Planck's constant
c=2.99792458*10**8                 # [m/s] light speed
k=1.3806488*10**(-23)               # [J/K] Boltzmann's constant
# isotopologue = raw_input("isotopologo")

#2-1
# d12 = -38772.7057800293
# nu12i = 230538019386.3495 - d12*59
# nu12f = 230538019386.3495 + d12*60
#
# d13 = -37067.44424438477
# nu13i = 220398702661.8233 - d13*59
# nu13f = 220398702661.8233 + d13*60
#
# d18 = -36926.43966674805
# nu18i = 219560372563.2249 - d18*59.
# nu18f = 219560372563.2249 + d18*60.

#3-2
d12 = -58157.20745849609
nu12i = 345796018978.6035 - d12*59
nu12f = 345796018978.6035 + d12*60

d13 = -55599.46649169922
nu13i = 330587993021.5317 - d13*59
nu13f = 330587993021.5317 + d13*60

d18 = -55387.990234375
nu18i = 329330580193.9937 - d18*59.
nu18f = 329330580193.9937 + d18*60.


def datafits(namefile):
    """
    Open a FITS image and return datacube and header, namefile without '.fits'
    """
    datacube = pf.open(namefile + '.fits')[0].data
    hdr = pf.open(namefile + '.fits')[0].header
    return datacube, hdr


def bbody(t,nu):
    """
    Blackbody flux for a given temperature and frequency W/(m^2*Hz*sr) (SI system)
    """
    return (1.4745e-50)*nu**3.0/(sp.exp(4.79922e-11*nu/t)-1.0)


def phi(t,nu,nu0,vturb,angle, m):
    """
    nco in microgram*cm^-2, vturb m*s^-1
    """
    pa = (-20+90+angle)*math.pi/180.0  #-20 is the PA from east of north, and 90 to get the semi-minor axis
    Q = 1
    shear = (Q*math.sin(incli)**2)*(math.tan(incli)**2)*math.sin(2*pa)**2
    deltav = (nu0/c)*math.sqrt(k*t/m + vturb**2 )*(1+shear)**(0.5)
    phi0 = deltav*math.sqrt(2*math.pi)
    gauss=sp.exp(-((nu-nu0)**2.0)/(2*(deltav**2.0)))
    norm = sp.sum(gauss)
    return gauss/norm


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
        nco /= 80.
    elif iso==18:
        sigma = ccs.cross_section("C18O")
        m = ccs.molecular_mass("C18O")
        nco /= 560.
#    sigma = sigma/(m*10**9)
    return -nco*sigma*phi(t,nu,nu0,vturb,angle, m)/math.cos(math.pi-incli)


def intensity(nu, T, nu0, nco, vturb, angle, hdr, iso=12):
    """
    Solution to radiative transfer equation
    """
    if iso==13:
        nco/= 80.
    if iso==18:
        nco/=500.
    pix_deg = abs(hdr['CDELT1'])
    pix_rad = pix_deg*sp.pi/180.
    bminaxis =  pix_deg  #arcsec
    bmajaxis = pix_deg  #arcsec,
    scaling = 1e26*pix_rad**2    ##SI to Jy/pix^2
#    scaling = (1e26*bmajaxis*bminaxis*math.pi/(4*math.log(2)))/(4.25e10)  #scaling to beam Si to Jy and arcsec^2 to strrad
    blackbody = bbody(T,nu)*(1.0-sp.exp(-tau(T,nu,nu0,nco,vturb,angle, iso)))*scaling
    return  blackbody


def intensity_continuum(nu, T, nu0, alpha, nco, vturb, angle, i0, hdr, v18, T18, nco18,iso=12):
    """
    Solution to radiative transfer equation qith continuum
    """
    pix_deg = abs(hdr['CDELT1'])
    pix_rad = pix_deg*sp.pi/180.
    bminaxis =  pix_deg  #arcsec
    bmajaxis = pix_deg  #arcsec,
    scaling = 1e26*pix_rad**2    ##SI to Jy/pix^2
    if iso==13:
        nco/= 80.
    cont = i0*sp.exp(tau(T18,nu,nu0,nco18,v18,angle, 18))*(nu/nu0)**alpha
    if iso==18:
        nco/=500.
        cont = i0*sp.exp(-tau(T,nu,nu0,nco,vturb,angle, iso))*(nu/nu0)**alpha
    blackbody = bbody(T,nu)*(1.0-sp.exp(-tau(T,nu,nu0,nco,vturb,angle, iso)))*scaling
    return  blackbody + cont


def intensity_dust_err(nu, nu0, T, alpha, i0, nco, vturb, angle, datos, hdr, v18, T18, nco18,iso=12):
    """
    Chi squared of data with fit of spectral line, considering continuum absorption from midplane.
    """
    model = intensity_continuum(nu, T, nu0, alpha, nco, vturb, angle, i0, hdr, v18, T18, nco18, iso)
    aux = (datos-model)**2
    chi = sp.sum(aux)
    return chi


def intensity_err(nu, nu0, T, nco, vturb, angle, datos, hdr, iso=12):
    """
    Chi squared of data with fit of spectral line, both normalized.
    """
    model = intensity(nu, T, nu0, nco, vturb, angle, hdr, iso)
    aux = (datos-model)**2
    chi = sp.sum(aux)
    return chi


def flux_int(T, alpha, i0, nco, vturb, angle, datos, hdr12, hdr13, hdr18):
    """
    Chi^2 obtained integrating function from radiative transfer
    in the spectral window of each isotopologue divided by molecular fractions.
    Frequencies in Hz
    """
    # nu013 = 220398702661.8233
    # nu012 = 230538019386.3495
    # nu018 = 219560372563.2249
    nu018 = 329330580193.9937
    nu013 = 330587993021.5317
    nu012 = 345796018978.6035
    integral12 = quad(intensity, nu12i, nu12f, args=(T, nu012 ,alpha, nco, vturb, angle, i0, hdr12, 12))
    integral13 = quad(intensity, nu13i, nu13f, args=(T, nu013 ,alpha, nco/70., vturb, angle, i0, hdr13, 13))
    integral18 = quad(intensity, nu18i, nu18f, args=(T, nu018 ,alpha, nco/500., vturb, angle, i0, hdr18, 18))
    chi2 = sp.sum((datos[0]-integral12[0]*nu012/c)**2) + sp.sum((datos[1]-integral13[0]*nu013/c)**2) + ((datos[2]-integral18[0]*nu018/c)**2)
    return chi2


def Convolv(cubo, head, Beam=0.25):
    resol = abs(head['CDELT1'])*3600
    stdev = Beam / (2 * sp.sqrt (2 * sp.log(2)))
    stdev /= resol
    x_size = int(8*stdev + 1.)

    print 'convolution with gaussian'
    print '\tbeam '+str(Beam)+' arcsec'
    print '\tbeam '+str(stdev)+' pixels'

    # circular Gaussian
    beam = Gaussian2DKernel (stddev = stdev, x_size = x_size, y_size = x_size,
                             model ='integrate')
    smooth =  np.zeros((256, 256))
    smooth += convolve_fft(cubo[:,:], beam)
    print '\tsmoothed'
    return smooth

def minlinesfitmoment(image, cont=False, iso=12, r=80, Convolve=False, Beam=0.25):
    """
    Fits a temperature profile, turbulent velocity and column density
    using three CO isotopologues lines with iminuit package.
    """
    print('Opening FITS images and fitting functions')

#  Isotopologue image and centroid map
    cubo, head = datafits(image)

#Gaussian Convolution
    if False:
        resol = abs(head['CDELT1'])*3600
        stdev = Beam / (2 * sp.sqrt (2 * sp.log(2)))
        stdev /= resol
        x_size = int(8*stdev + 1.)

        print 'convolution with gaussian'
        print '\tbeam '+str(Beam)+' arcsec'
        print '\tbeam '+str(stdev)+' pixels'

        # circular Gaussian
        beam = Gaussian2DKernel (stddev = stdev, x_size = x_size, y_size = x_size,
                                 model ='integrate')
        smooth =  np.zeros((120, 256, 256))
        for k in range(120):
            smooth[k, :,:] += convolve_fft(cubo[k,:,:], beam)
        print '\tsmoothed'
        cubo = smooth

    inc18o = '/home/felipe/Escritorio/nuevos_cubos/Faceon/Simslines_18'

    if iso==12:
        nu0 = 345796018978.6035
        nu = sp.linspace(nu12i, nu12f, 120)
        f=1.
        ncoscale = 1e18
        if cont:
            nco_c18o = pf.open(inc18o + '_NCO.fits')[0].data
            vturb_c18o = pf.open(inc18o + '_v_turb.fits')[0].data
            Temp_c18o = pf.open(inc18o + '_Temp.fits')[0].data
        dnu = nu[1]-nu[0]
    elif iso==13:
        ncoscale = 1e18
        nu0 = 330587993021.5317
        nu= sp.linspace(nu13i, nu13f, 120)
        f=1e-2
        dnu = nu[1]-nu[0]
        if cont:
            nco_c18o = pf.open(inc18o + '_NCO.fits')[0].data
            vturb_c18o = pf.open(inc18o + '_v_turb.fits')[0].data
            Temp_c18o = pf.open(inc18o + '_Temp.fits')[0].data
    else:
        ncoscale = 1e18
        nu0 = 329330580193.9937
        nu = sp.linspace(nu18i, nu18f, 120)
        f=0.0017
        dnu = nu[1]-nu[0]


    cubomax = cubo.max()

    stderr = sp.std(cubo)
    alpha = 2.3
    i0 = 0.0
    angle = 28.*sp.pi/180.
    Temperature = sp.zeros((256,256))
    Denscol = sp.zeros((256,256))
    Turbvel = sp.zeros((256,256))
    errpars = sp.zeros((256,256,3))
    errmodel = sp.zeros((256,256))
    model = sp.zeros((256,256,120))
    mom2 = sp.zeros((256,256))
    velocities = sp.linspace(-3.,3.,120)

    for i in range(256):
        for j in range(256):

            if (i-128)**2 + (j-128)**2 > r**2 :
                continue

            data = cubo[:,i,j]

            m0 = sp.integrate.simps(data, nu)
            m1 = sp.integrate.simps(velocities*data, nu)/m0
            mom2[i,j] = sp.integrate.simps(data*(velocities-m1)**2, nu)*1000/m0

            datamax = data.max()
            datamin = data.min()
            data1 = data -data.min()
            centroid = (nu*data1).sum()/data1.sum()

            data = data[40:80]
            i0 = datamin
            if i0==0:
                i0=1e-10
            i0_lim=(0.6*i0,1.2*i0)

            r_ij = sp.sqrt((i-128)**2 +(j-128)**2)

            if r_ij>20 and r_ij<50:
                tscale = 120.
                t0 = 350*(r_ij*280/128.)**(-1/1.5)/120.
                tlim =(0.05,1.)
            elif r_ij>=50.:
                tscale = 80.
                t0 = 350*(r_ij*280./128.)**(-1/1.5)/80.
                tlim =(0.05,1.)
            else:
                tscale=200.
                t0 = 0.4
                tlim = (0.05,1.)

            if cont:

                if iso==18:
                    f = lambda Temp,vturb,NcolCO, i0, nu0: intensity_dust_err(nu[40:80], nu0, Temp*tscale, alpha, i0, ncoscale*NcolCO,
                                                                600.*vturb, angle, data, head, 600.*vturb, tscale*Temp,
                                                                ncoscale*NcolCO, iso)
                else:
                    nco18 = nco_c18o[i,j]
                    temp18 = Temp_c18o[i,j]
                    vturb18 = vturb_c18o[i,j]
                    f = lambda Temp,vturb,NcolCO, i0, nu0: intensity_dust_err(nu[40:80], nu0, Temp*tscale, alpha, i0, ncoscale*NcolCO,
                                                                600.*vturb, angle, data, head, vturb18, temp18,
                                                                nco18, iso)

                m = Minuit(f, Temp=t0, vturb=0.04, NcolCO=0.00001, errordef=stderr/datamax, i0=i0, nu0=centroid,
                            error_Temp=.005, error_NcolCO=0.0000005, error_vturb=0.005, error_i0=i0/100.,
                            error_nu0=abs(dnu)/5., limit_Temp=tlim, limit_vturb=(0.002, 1),
                            limit_NcolCO=(0.0000001, 1), limit_i0=i0_lim, limit_nu0=(centroid-abs(dnu), centroid+abs(dnu)))
                m.migrad(1e9)

                errmod = f(m.values['Temp'], m.values['vturb'], m.values['NcolCO'], m.values['i0'], m.values['nu0'])
                fit = [m.values['Temp']*tscale, m.values['vturb']*600., ncoscale*m.values['NcolCO'],
                       m.values['i0'], m.values['nu0']];
                i0model = fit[3]

                if iso!=18:
                    model[i,j] = intensity_continuum(nu, fit[0], fit[4], alpha, fit[2], fit[1],
                                                    angle, fit[3], head, vturb18, temp18, nco18, iso)
                else:
                    model[i,j] = intensity_continuum(nu, fit[0], fit[4], alpha, fit[2], fit[1],
                                                    angle, fit[3], head, fit[1], fit[0], fit[2], iso)

            else:
                data = data1[40:80]
#                data1/=data1.sum()
                f = lambda Temp,vturb,NcolCO, nu0: intensity_err(nu[40:80], nu0, Temp*tscale, ncoscale*NcolCO,
                                                            600.*vturb, angle, data, head, iso)
                m = Minuit(f, Temp=t0, vturb=0.04, NcolCO=0.00001, errordef=stderr/datamax, nu0=centroid,
                            error_Temp=.005, error_NcolCO=0.0000005, error_vturb=0.005, error_nu0=abs(dnu)/5.,
                            limit_Temp=tlim, limit_vturb=(0.002, 1), limit_NcolCO=(0.0000001, 1),
                            limit_nu0=(centroid-1*abs(dnu), nu0+1*abs(dnu)))
                m.migrad(1e9)
                errmod = f(m.values['Temp'], m.values['vturb'], m.values['NcolCO'], m-values['nu0'])
                fit = [m.values['Temp']*tscale, m.values['vturb']*600., ncoscale*m.values['NcolCO'], m.values['nu0']];
                model[i,j] = intensity(nu, fit[0], fit[3], fit[2], fit[1], angle, head, iso)

            error = [m.errors['Temp']*tscale, m.errors['vturb']*600., ncoscale*m.errors['NcolCO']];

            Temperature[i,j] = fit[0]
            Denscol[i,j] = fit[2]
            Turbvel[i,j] = fit[1]
            errpars[i,j] = error
            errmodel[i,j] = errmod



# Parameter error (confidence intervals)
    errmodel= sp.array(errmodel)
    model = sp.array(model)
    model = sp.swapaxes(model, 0,1)
    model = sp.swapaxes(model,0,2)
    print('Calculated Fits')
    err_temp = errpars[:, :, 0]
    err_v_turb = errpars[:, :, 1]
    err_NCO = errpars[:, :, 2]
    model = sp.nan_to_num(model)
    Denscol = sp.nan_to_num(Denscol)
    Turbvel = sp.nan_to_num(Turbvel)
    Temperature = sp.nan_to_num(Temperature)

    if Convolve:
        Denscol = Convolv(Denscol, head)
        Temperature = Convolv(Temperature, head)
        Turbvel = Convolv(Turbvel, head)
        mom2 = Convolv(mom2, head)

    r1 = pf.PrimaryHDU(model)
    r2 = pf.PrimaryHDU(Temperature)
    r3 = pf.PrimaryHDU(Turbvel)
    r4 = pf.PrimaryHDU(Denscol)
    r5 = pf.PrimaryHDU(errmodel)
    r6 = pf.PrimaryHDU(err_temp)
    r7 = pf.PrimaryHDU(err_v_turb)
    r8 = pf.PrimaryHDU(err_NCO)
    r9 = pf.PrimaryHDU(mom2)
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
    r9.header = head2
    inputimage = '/home/felipe/Escritorio/nuevos_cubos/Faceon/Simslines_' + str(iso)
#    inputimage = '/home/felipe/Escritorio/ID/vortex/planetvortex/template/products/Simslines/' +str(iso)
#    inputimage = '/home/felipe/Escritorio/ID/HD142527/Simulations_CO3-2/smooth_Simslines_' + str(iso)
    out1 = inputimage + '_model.fits'
    out2 = inputimage + '_Temp.fits'
    out3 = inputimage + '_v_turb.fits'
    out4 = inputimage + '_NCO.fits'
    out5 = inputimage + '_errpars_temp.fits'
    out6 = inputimage + '_errfit.fits'
    out7 = inputimage + '_errpars_vturb.fits'
    out8 = inputimage + '_errpars_NCO.fits'
    out9 = inputimage + '_mom2.fits'
    print('Writing images')
    r1.writeto(out1, clobber=True)
    r2.writeto(out2, clobber=True)
    r3.writeto(out3, clobber=True)
    r4.writeto(out4, clobber=True)
    r5.writeto(out5, clobber=True)
    r6.writeto(out6, clobber=True)
    r7.writeto(out7, clobber=True)
    r8.writeto(out8, clobber=True)
    r9.writeto(out9, clobber=True)
    pf.writeto(out1, model, head, clobber=True)
    pf.writeto(out2, Temperature, head1, clobber=True)
    pf.writeto(out3, Turbvel, head2, clobber=True)
    pf.writeto(out4, Denscol, head3, clobber=True)
    pf.writeto(out5, err_temp, head1, clobber=True)
    pf.writeto(out6, errmodel, head1, clobber=True)
    pf.writeto(out7, err_v_turb, head2, clobber=True)
    pf.writeto(out8, err_NCO, head3, clobber=True)
    pf.writeto(out8, mom2, head2, clobber=True)
    return

image12CO = '/home/felipe/Escritorio/nuevos_cubos/Faceon/image_12co32_dust_o500_i180.00_phi90.00_PA-24.00'
image13CO = '/home/felipe/Escritorio/nuevos_cubos/Faceon/image_13co32_dust_o500_i180.00_phi90.00_PA-24.00'
imageC18O = '/home/felipe/Escritorio/nuevos_cubos/Faceon/image_c18o32_dust_o500_i180.00_phi90.00_PA-24.00'

#image12CO = '/home/felipe/Escritorio/ID/vortex/planetvortex/template/products/image_12co32_o0_i180.00_phi90.00_PA-24.00'
#image13CO = '/home/felipe/Escritorio/ID/vortex/planetvortex/template/products/image_13co32_o0_i180.00_phi90.00_PA-24.00'
#imageC18O = '/home/felipe/Escritorio/ID/vortex/planetvortex/template/products/image_c18o32_o0_i180.00_phi90.00_PA-24.00'

minlinesfitmoment(imageC18O, r=80, iso=18, Convolve=False, cont=True)
minlinesfitmoment(image13CO, r=80, iso=13, Convolve=False, cont=True)
minlinesfitmoment(image12CO, r=80, iso=12, Convolve=False, cont=True)
