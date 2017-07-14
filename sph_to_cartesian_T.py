import pyfits
from astropy.io import fits
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pylab
import math
import copy
import sys


#python view.py dust_temperature.bdat
infile = 'dust_temperature.bdat'

#f = open(infile,'r')
nr, ntheta, nphi = (128, 128, 128)

#im_nx, im_ny = tuple(np.array(f.readline().split(),dtype=int))
#nlam = int(f.readline())
#pixsize_x, pixsize_y = tuple(np.array(f.readline().split(),dtype=float))


#images = np.loadtxt(infile, skiprows=3)

#head=np.fromfile(infile,dtype=int,count=10)

dt = np.dtype('float32')
allbdata=np.fromfile(infile,dtype=dt,count=-1)

#print allbdata.shape ," vs ", 2*128*128*128,"\n"

images=allbdata[8:]

print images.shape ,"\n"

cube = images.reshape(2,nphi,ntheta,nr)

#print cube.shape

nx=2*nr
ny=2*nr
nz=2*nr

cubecart = np.zeros((nx,ny,nz))

#hdu = fits.PrimaryHDU(cube)
#hdu.writeto('Tcube.fits',clobber=True)

#datadir="./RT_experiments/cavityouterdisk185/"
datadir="."
import sys
sys.path.append(datadir)
from ParametricDiskr import *
from set_M import M
write_AMRgrid(M, Plot=False, PunchGrid=False)

RRs = M.xm
Thetas =  M.ym
Phis = M.zm

RRs = RRs[0:-1]
Thetas =  Thetas[0:-1]
Phis = Phis[0:-1]

Phis = Phis - np.pi

#print "min Thetas ", np.min(Thetas),"\n"
#print "max Thetas ", np.max(Thetas),"\n"
#print "min Phis ", np.min(Phis),"\n"
#print "max Phis ", np.max(Phis),"\n"
#
rmax = max(RRs)
pixsize= (rmax/nr)
xxs = pixsize*(np.arange(nx)-nx/2)
yys = pixsize*(np.arange(ny)-ny/2)
zzs = pixsize*(np.arange(nz)-nz/2)

for k in range(nz):
    print k," < ",nz
    #for k in ([nz/2.]):
    #for k in ([0]):
    for j in range(ny):
        for i in range(nx):
#        for i in ([nx/2.]):
#        for i in ([nx-1]):
            x=xxs[i]
            y=yys[j]
            z=zzs[k]
            r=np.sqrt(x**2+y**2+z**2)
#            print "r ",r,"x ",x,"y ",y ,",z",z
            Theta=np.arccos(z/r)
            Phi=np.arctan2(y,x)
#            print "Theta ",Theta,"Phi",Phi

            iPhi=np.argmin(np.fabs(Phis-Phi))
            iTheta=np.argmin(np.fabs(Thetas-Theta))
            iR=np.argmin(np.fabs(RRs-r))
#
#            print 'ir ',iR,' iPhi',iPhi,' iTheta', iTheta
            T=cube[1,iPhi,iTheta,iR]
#            print "T= ",T," \n"
            cubecart[k,j,i] = T



hdu = fits.PrimaryHDU(cubecart)

h = hdu.header
h['CRPIX1']=nx/2+1
h['CRPIX2']=ny/2+1
h['CRPIX3']=nz/2+1
h['CRVAL1']=0.
h['CRVAL2']=0.
h['CRVAL3']=0.
h['CDELT1']="%e" % (pixsize)
h['CDELT2']="%e" % (pixsize)
h['CDELT3']="%e" % (pixsize)

hdu.writeto('Tcart.fits',clobber=True)


# im = np.log10(im)
#plt.imshow(im, cmap = 'plasma', origin='lower',aspect='auto', vmax = im.max()/1e5)
#plt.axis('equal')
#plt.show()
