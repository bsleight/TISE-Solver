import numpy as np
from scipy.stats import norm
###A file containing different types of 3D potentials
###hbar = m = 1 for all potentials

def coulomb(x,y,z):
	###Hydrogen Atom potential in natual units
	return -1/np.sqrt((x**2 + y**2 + z**2))

def gaussian(x, y, z, mu=0, sigma=1, A=1):
	###Gaussian potential
	###A: amplitude, sigma: standard deviation, mu: center
	return A*(norm.pdf(x, loc=mu, scale=sigma) * norm.pdf(y, loc=mu, scale=sigma) * norm.pdf(z, loc=mu, scale=sigma))

def harmonic_osc(x, y, z, w=1):
	###Harmonic Osscillator potential
	###w:natural frequency
	return 0.5*(w**2)*(x**2 + y**2 + z**2)
