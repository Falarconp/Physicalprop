"""
Code created by Christian Flores
"""

#------------------------------------------------------------------------------------------------
#module imported as ccs
#------------------------------------------------------------------------------------------------
from constants import rest_freq_C18O
from constants import rest_freq_12CO
from constants import rest_freq_13CO

from constants import A_C18
from constants import A_C12
from constants import A_C13

import numpy as np
import astropy.io.fits as pf
import matplotlib.pyplot as plt
#--------------------------------------------------------------------------------
import scipy.ndimage.measurements as meas
import glob
import os
import math

#constants in cgs units
#--------------------------------------------------------------------------------
#h=6.6260755e-27
c1=2.9979245e10
#m_e=9.1093897e-28
#q_e=4.80320425e-10

g1=5							#level 2 rotational degenerancy
g2=7							#level 3 rotational

## g1=3
## g2=5

mmass_12CO = 27.99491461956 				#molecular mass of 12CO in g/mol
mmass_13CO = 28.9982694574				#molecular mass of 13CO in g/mol
mmass_C18O = 29.99916100 					#molecular mass of C18O in g/mol

avog=6.0221413e+23					#avogadro number mol^-1

#-----------------------------------------------------------------------------
#
#line absorption cross section derived from Einstein Coefficients in [cm^2/s]
#
#-----------------------------------------------------------------------------
def cross_section(name):

	if name=="C18O":
		nu=rest_freq_C18O				#C18O rest frequency
		A21=A_C18
	elif name=="12CO":
		nu=rest_freq_12CO				#12CO rest frequency
		A21=A_C12
	elif name=="13CO":
		nu=rest_freq_13CO				#13CO rest frequency
		A21=A_C13
	# B21=(c**2/(2*h*nu**3)) * A21
	# B12=(g2/g1)*B21
	# f12=B12*h*nu*m_e*c/(4*math.pi**2 *q_e**2)
	# sigma=math.pi*q_e**2*f12/(m_e*c)
	sigma = (g2*c1**2*A21)/(g1*8*math.pi*nu**2)
	return sigma

#-----------------------------------------------------------------------------
#
#molecular mass of molecules in Kg
#
#-----------------------------------------------------------------------------
def molecular_mass(name):
	if name=="C18O":
		m = mmass_C18O
	elif name=="12CO":
		m = mmass_12CO
	elif name=="13CO":
		m = mmass_13CO
	else:
		m = mmass_12CO
	return m*1e-3/avog

#-----------------------------------------------------------------------------
#
#molecular fraction of the isotopologues
#
#-----------------------------------------------------------------------------
def molecular_fraction(name):
	if name=="C18O":
		g=(1/500.0)#*1.26
	elif name=="12CO":
		g=1.0
	elif name=="13CO":
		g=1/70.0
	else:
		g=1.
	return g
