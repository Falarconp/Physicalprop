import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from astropy.io import fits as pf
import math
import seaborn as sns
import collision_cross_section	as ccs
import astropy.constants as const
from scipy.interpolate import griddata
from scipy.integrate import simps
import astropy.units as u
#from Hydro3D import *

######### Guardarr z de cada punto, dz con dcos, interpolar grilla polar en grilla plano, itegrar,
## grid already in cm

h = const.h.cgs.value
c = const.c.cgs.value
k = const.k_B.cgs.value

nu12 = 345796018978.6035
nu13 = 330587993021.5317
nu18 = 329330580193.9937

#indir = '/home/felipe/Escritorio/ID/planetvortex_p01/'
#orbitnumber = '500'

#rho = Field(field='gasdens'+orbitnumber+'.dat', directory=indir)
#ene = Field(field='gasenergy'+orbitnumber+'.dat', directory=indir)
#vx  = Field('gasvy'+orbitnumber+'.dat', staggered='y', directory=indir)
#vy  = Field('gasvz'+orbitnumber+'.dat', staggered='z', directory=indir)
#vz  = Field('gasvx'+orbitnumber+'.dat', staggered='x', directory=indir)


def tau(t,nu,nu0,nco,vturb,angle,iso=12):
    """
    Optical depth with turbulent velocity and thermal velocity
    sigma is cross area cm^2
    """
    sigma = ccs.cross_section("13CO")
    if iso==12:
        m = ccs.cross_section("13CO")
        sigma = ccs.cross_section("12CO")
        m = ccs.molecular_mass("12CO")
    elif iso==13:
        sigma = ccs.cross_section("13CO")
        m = ccs.molecular_mass("13CO")
    elif iso==18:
        sigma = ccs.cross_section("C18O")
        m = ccs.molecular_mass("C18O")
    return nco*sigma*phi(t,nu,nu0,vturb,angle, m)/math.cos(incli)

def input_pars(iso='12', incli=0):

    incli = incli*sp.pi/180.

#---Input Files
    gridfile='/home/felipe/Escritorio/ID/vortex/planetvortex/template/amr_grid.inp'
    velfile = '/home/felipe/Escritorio/ID/vortex/planetvortex/template/gas_velocity.inp'
    densfile = '/home/felipe/Escritorio/ID/vortex/planetvortex/template/numberdens_'
    kappafile = '/home/felipe/Escritorio/ID/vortex/planetvortex/template/dustkappa_astrosilicate_ext0.inp'

    scaling = (u.au).to(u.cm)

#----Isotopologue Data
    if iso=='12':
        densfile = densfile +'12c16o.inp'
        nu0 = nu12
        m = ccs.molecular_mass("12CO")
        sigma = ccs.cross_section("12CO")
    elif iso=='13':
        densfile = densfile +'13c16o.inp'
        nu0 = nu13
        m = ccs.molecular_mass("13CO")
        sigma = ccs.cross_section("13CO")
    else:
        densfile = densfile +'12c18o.inp'
        nu0 = nu18
        m = ccs.molecular_mass("C18O")
        sigma = ccs.cross_section("C18O")

    gasdens = sp.loadtxt(densfile,skiprows=2)

#------------ 0:radial, 1:colatitude, 2:azimuth
    gasvel = sp.loadtxt(velfile,skiprows=2)

#------------READING GRID
    arch=open(gridfile,'r')
    arch.readline(5)
    arch.readline()
    arch.readline()
    arch.readline()
    arch.readline()
    line=arch.readline()
    dat=line.split()

    Nr=int(dat[0])
    Nth=int(dat[1])
    Nphi=int(dat[2])


    Redge=np.fromstring(arch.readline(),dtype=float, sep=' ')/1.496e13
    Thedge=np.fromstring(arch.readline(),dtype=float, sep=' ')
    Phiedge=np.fromstring(arch.readline(),dtype=float, sep=' ')

    dR=Redge[1:]-Redge[:-1]
    R=Redge[:-1]+dR

#    R *= scaling

    dTh=Thedge[1:]-Thedge[:-1]
    Th=Thedge[:-1]+dTh

    dPhi=Phiedge[1:]-Phiedge[:-1]
    Phi=Phiedge[:-1]+dPhi

    arch.close()

    res = 256

    zmax = R[-1]*sp.cos(Th[0])

    Temperature = sp.zeros((res,res))
    velocity = sp.zeros((res,res))
    density = sp.zeros((res,res))
    dsurf = sp.zeros((res,res))
    taus = sp.zeros((res,res,2))

#----------------------SETING GAS TEMPERATURE
    print "Calculating input Parameters..."

    rhos = sp.linspace(-R[-1]*sp.cos(incli), R[-1]*sp.cos(incli), res)
    drho = rhos[1] - rhos[0]

    depth = sp.cos(Th[0]+incli)-sp.cos(Th+incli)
    depth_2 = sp.cos(Th[0]-incli) - sp.cos(Th-incli)

    x = sp.linspace(-R[-1], R[-1], res)
    y = sp.linspace(-R[-1], R[-1], res)
    X0, Y0 = sp.meshgrid(x,y)

    densities = sp.zeros((res,res,Nth))
    vturb = sp.zeros((res,res,Nth))
    Temp = sp.zeros((res,res,Nth))
    v_th = sp.zeros((res,res,Nth))

    for k in xrange(Nth):
        Theta = Th[k]
        coor = sp.zeros((Nr*Nphi,2))
        griddens = sp.zeros(Nr*Nphi)
        gridvturb = sp.zeros(Nr*Nphi)
        for j in xrange(Nphi):
            phi=Phi[j]
            for i in xrange(Nr):
                if phi>sp.pi:
                    Theta += incli
                elif phi< sp.pi:
                    Theta -= incli
                Rho = R[i]*sp.sin(Theta)
                index = i + Nr*(Nth*j + k)
                griddens[Nr*j + i] = gasdens[index]
                gridvturb[Nr*j + i] = gasvel[index,0]*sp.cos(Theta) - gasvel[index,1]*sp.sin(Theta) - gasvel[index,2]*sp.sin(phi)*sp.sin(incli)
                coor[Nr*j+ i] = sp.array([Rho*sp.cos(phi), Rho*sp.sin(phi)])
        densities[:,:,k] = griddata(coor, griddens, (X0,Y0), method='cubic', fill_value=0)
        vturb[:,:,k] = griddata(coor, gridvturb, (X0,Y0), method='cubic', fill_value=0)

    print "Calculating optical depth and physical parameters..."

    for j in range(res):
        for i in range(res):
            dist = sp.sqrt(X0[i,j]**2 + Y0[i,j]**2)
            if Y0[i,j]>0:
                z = dist*depth*scaling
            else:
                z = dist*depth_2*scaling
            for k in range(1,len(z)):
                if dist <30:
                    Temp[i,j,k] = 70*sp.cos(incli)
                    v_th[i,j,k] = sp.sqrt(k*Temp[i,j,k]/m)
                else:
                    Temp[i,j,k] = 70 * (dist/30)**(-0.5)
                    v_th[i,j,k] = sp.sqrt(k*Temp[i,j,k]/m)
                ncol = densities[i,j,k-1]*0.5 + densities[i,j,k]*0.5
                v_turb = vturb[i,j,k-1]*0.5 + vturb[i,j,k]*0.5
                deltav = (nu0/c)*math.sqrt(v_turb**2 )
                phi0 = deltav*math.sqrt(2*math.pi)
                if taus[i,j,0] <1:
                    taus[i,j,1] = z[k]
                    taus[i,j,0]  += ncol*sigma*(z[k]-z[k-1])/phi0
                    density[i,j] = simps(densities[i,j,:k],z[:k])
                    Temperature[i,j] = simps(Temp[i,j,:k]*densities[i,j,:k],z[:k])/density[i,j]
                    centroid =  simps(vturb[i,j,:k]*densities[i,j,:k],z[:k])/density[i,j]
                    v_t = sp.sqrt(k*(Temp[i,j,k-1]*0.5 + Temp[i,j,k]*0.5)/m)
                    velocity[i,j] = simps(((vturb[i,j,:k]-centroid)**2 - v_th[i,j,:k]**2)*densities[i,j,:k],z[:k])/density[i,j]
                    velocity[i,j] = sp.sqrt(velocity[i,j])
            dsurf[i,j] = simps(densities[i,j,:], z)
            dsurf *= m/const.N_A.value

    Temperature = sp.nan_to_num(Temperature)
    velocity = sp.nan_to_num(velocity)
    taus = sp.nan_to_num(taus)

    print('Saving FITS images...')

    r1 = pf.PrimaryHDU(velocity)
    r2 = pf.PrimaryHDU(Temperature)
    r3 = pf.PrimaryHDU(density)
    r4 = pf.PrimaryHDU(taus[:,:,1])

    r1.header['BITPIX'] = -32

    r1.header['NAXIS'] = 2
    r1.header['NAXIS1'] = 256
    r1.header['NAXIS2'] = 256
    r1.header['EPOCH']  = 2000.0
    r1.header['EQUINOX'] = 2000.0
    r1.header['LONPOLE'] = 180.0
    r1.header['CTYPE1'] = 'RA---TAN'
    r1.header['CTYPE2'] = 'DEC--TAN'
    r1.header['CRVAL1'] = float(0.0)
    r1.header['CRVAL2'] = float(0.0)
    r1.header['CDELT1'] = 4.77430501022309E-06
    r1.header['CDELT2'] = 4.77430501022309E-06
    r1.header['CUNIT1'] = 'DEG     '
    r1.header['CUNIT2'] = 'DEG     '
    r1.header['CRPIX1'] = float((res-1)/2)
    r1.header['CRPIX2'] = float((res-1)/2)
    r1.header['BUNIT'] = 'cm/s'
    r1.header['BTYPE'] = 'Velocity'
    r1.header['BSCALE'] = 1
    r1.header['BZERO'] = 0
    head1 = r1.header
    head1['BUNIT'] = 'Kelvin'
    head1['BTYPE'] = 'Temperature'
    head2 = r1.header
    head3 = r1.header
    head2['BUNIT'] = 'cm^-2'
    head2['BTYPE'] = 'Column Density'
    head3['BUNIT'] = 'cm'
    head3['BTYPE'] = 'Optical Depth'
    r2.header = head1
    r3.header = head2
    r4.header = head3
    inputimage = '/home/felipe/Escritorio/nuevos_cubos/Input/Inp_pars_incli_' + str(incli) +'_' + str(iso)
    out2 = inputimage + '_Temp.fits'
    out1 = inputimage + '_v_turb.fits'
    out3 = inputimage + '_NCO.fits'
    out4 = inputimage + '_opt_depth.fits'
    print('Writing images')
    r1.writeto(out1, clobber=True)
    r2.writeto(out2, clobber=True)
    r3.writeto(out3, clobber=True)
    r4.writeto(out4, clobber=True)
    pf.writeto(out1, velocity, r1.header, clobber=True)
    pf.writeto(out2, Temperature, head1, clobber=True)
    pf.writeto(out3, density, head2, clobber=True)
    pf.writeto(out4, taus[:,:,1], head3, clobber=True)
    return

input_pars(iso='12', incli=0.)
input_pars(iso='13', incli=0.)
input_pars(iso='18', incli=0.)
