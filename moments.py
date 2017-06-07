import scipy as sp
from astropy.io import fits as pf
import matplotlib.pyplot as plt
import seaborn as sns

velocity = sp.linspace(-3,3.120)


def datafits(namefile):
    """
    Open a FITS image and return datacube and header, namefile without '.fits'
    """
    datacube = pf.open(namefile + '.fits')[0].data
    hdr = pf.open(namefile + '.fits')[0].header
    return datacube, hdr

def nu_sample(datacube, hdr):
    length = len(datacube)
    nus = sp.zeros(length)
    index = hdr['CRPIX3']
    dnu = hdr['CDELT3']
    nu_0 = hdr['CRVAL3'] - index*dnu
    nu_f = hdr['CRVAL3'] + (length-index-1)*dnu
    nus = sp.arange(nu_0, nu_f, dnu)
    return nus


def m0(datacube, hdr):
    mom_map = sp.zeros([len(datacube[0]), len(datacube[0][0])])
    nus = nu_sample(datacube, hdr)
    for i in range(datacube[0]):
        for j in range(datacube[0][0]):
            mom_map[i][j] = sp.integrate.simps(nus, datacube)
    return mom_map


def m1(datacube, hdr):


def m2(datacube, hdr):


def m3(datacube, hdr):


def m7(datacube, hdr):


def m8(datacube, hdr):
    mom_map = sp.zeros([len(datacube[0]), len(datacube[0][0])])
    nus = nu_sample(datacube, hdr)
    index = sp.arange(120)
    for i in range(datacube[0]):
        for j in range(datacube[0][0]):
            datos = datacube[:,i,j]
            indice = index[datos==datos.max()]
            mom_map[i][j] = velocity[indice]
    return mom_map


def moment_map(imagefile, mom=0):
    name = imagefile+'fits'
    datacube, hdr = datafits(name)
    print('\tCalculating moment map')
    out = name + '_mom' + str(mom) + '.fits'
    print('\tWriting image')
    if m==0:
        mom_map, head = m0(datacube, hdr)
    elif m==1:
        mom_map, head = m1(datacube, hdr)
    elif m==2:
        mom_map, head = m2(datacube, hdr)
    elif m==7:
        mom_map, head = m7(datacube, hdr)
    elif m==8:
        mom_map, head = m8(datacube, hdr)
    r = pf.PrimaryHDU(mom_map)
    r.header = head
    r.writeto(out, clobber=True)
    pf.writeto(out, mom_map, head, clobber=True)
    return
