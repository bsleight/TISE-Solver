#Byron Sleight 11/7/22
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from scipy import sparse
import potentials
import argparse

def main(ptype, N, L, l, w, mu, sigma, A, n):
	###main function for plotting eigenvalues and eigenvectors for 3D potential
	###set grid spacing
	h = L/N

	###generate position mesh grid
	x, y, z = np.meshgrid(np.linspace(-L/2,L/2,N), np.linspace(-L/2,L/2,N), np.linspace(-L/2,L/2,N))

	###calculate kinetic energy and angular momentum operators
	T = get_kinetic(N,h)
	Lz = get_Lz(N, x, y, h)

	###conditionals to run correct potential type
	if ptype == 'QHO':
		###Harmonic Oscillator Potential
		print("Calculation for Harmonic Oscillator Potential")
		V = potentials.harmonic_osc(x, y, z, w)

	elif ptype == 'gaussian':
		###Gaussian Potential
		print("Calculation for Gaussian Potential")
		V = potentials.gaussian(x, y, z, mu=0, sigma=sigma, A=A)

	elif ptype == 'well':
		###Infinite 3D well
		print("Calculation for Infinite Well Potential")
		V = potentials.harmonic_osc(x, y, z, w =0)

	elif ptype == 'coulomb':
		###Coulumb potential
		print("Calculation for Coulumb Potential")
		V = potentials.coulomb(x,y,z)

	else:
		###the potential type entered is invalid
		print('Invalid Potential type')

	###generate Hamiltonian as a sum of kinetic and potential matrices
	H = T + sparse.diags(V.reshape(N**3),(0))
	###perturbed Hamiltonian
	H_perturb = H + l*Lz

	###calculate perturbed and unperturbed eigenvectors and eigenvalues
	E, eigenvecs = eigsh(H, k=n, which='SM')
	E_perturb, eigenvecs_perturb = eigsh(H_perturb, k=n, which='SM')

	###reshape eigenvectors to make them plottable
	psi = [eigenvecs.T[i].reshape((N,N,N)) for i in range(n)]
	psi_perturb = [eigenvecs_perturb.T[i].reshape((N,N,N)) for i in range(n)]

	###plot eigenvalues and eigenvectors
	plot_eigenvalues(E, E_perturb)
	plot_prob_dist(x,y,z,psi, psi_perturb)
	plt.show()


def get_kinetic(N,h):
	###kinetic energy matrix
	###generating 1D laplacian matrix via finite diff. approx.
	ones = np.ones([N])
	diags = np.array([ones, -2*ones, ones])
	D = sparse.spdiags(diags, np.array([-1,0,1]), N, N)/h**2

	###use kronecker products to represent each component of the 3D laplacian
	I = np.identity(N)
	d2dx2 = sparse.kron(sparse.kron(D,I),I)  #d^2/dx^2
	d2dy2 =  sparse.kron(sparse.kron(I,D),I) #d^2/dy^2
	d2dz2 = sparse.kron(sparse.kron(I,I),D) #d^2/dz^2
	laplace_3d = d2dx2 + d2dy2 + d2dz2 #full 3D laplacian

	###return kinetic energy operator (hbar = m =1)
	T = -(1/2) * laplace_3d
	return T


def get_Lz(N, x, y, h):
	###Lz component of angular momentum operator
	###1D finite diff. approx. for first derivative
	ones = np.ones([N])
	diags = np.array([ -ones, ones])
	d = sparse.spdiags(diags, np.array([ 0,1]),N, N)/h

	###use kronecker products to represent the 1D derivative in 3D
	I = np.identity(N)
	ddx = sparse.kron(sparse.kron(d,I),I) #d/dx
	ddy =  sparse.kron(sparse.kron(I,d),I) #d/dy

	###momentum operators (hbar = 1)
	px = -1j*ddx
	py = -1j*ddy

	###position operators (diagonal in position space)
	pos_diags = np.linspace(-L/2, L/2, N, dtype = float)
	points = sparse.spdiags(pos_diags, np.array([0]),N, N)
	xhat = sparse.kron(sparse.kron(points,I),I) #x operator
	yhat =  sparse.kron(sparse.kron(I,points),I) #y operator

	###return Lz
	Lz = xhat*py - yhat*px
	return Lz


def plot_prob_dist(X, Y, Z, psi, psi_perturb):
	###helper function for plotting prod. dens.
	###plot for unperturbed ground state
	fig = plt.figure(3)
	ax1 = fig.add_subplot(221,projection="3d")
	ax1.set_xlabel('x')
	ax1.set_ylabel('y')
	ax1.set_zlabel('z')
	ax1.set_title('Ground State Probability Distribution (Unperturbed)')
	dens = np.conjugate(psi[0])*psi[0] #|psi|^2

	max_dens = np.max(dens) #maximum value of |psi|^2
	frac = 0.05 #denotes the fraction of max_dens a point needs to be greater than to be included in plot

	###plotting prob. dens. with a mask so any points <= frac*max_dens don't crowd the density plot
	plot1 = ax1.scatter(X[dens >= frac*max_dens],Y[dens >= frac*max_dens],Z[dens >= frac*max_dens],
	c=dens[dens >= frac*max_dens],s=0.001,cmap = cm.seismic, alpha=1,antialiased=True )

	fig.colorbar(plot1, ax=ax1, location = 'left', label = 'Prob. Dens.', format='%.0e') #set colorbar

	###plot for unperturbed excited state
	ax2 = fig.add_subplot(222,projection="3d")
	ax2.set_xlabel('x')
	ax2.set_ylabel('y')
	ax2.set_zlabel('z')
	ax2.set_title('First Excited State Probability Distribution (Unperturbed)')
	dens = np.conjugate(psi[1])*psi[1] #|psi|^2

	###plotting prob. dens. with a mask so any points <= frac*max_dens don't crowd the density plot
	plot2 = ax2.scatter(X[dens >= frac*max_dens],Y[dens >= frac*max_dens],Z[dens >= frac*max_dens],
	c=dens[dens >= frac*max_dens],s=0.001,cmap = cm.seismic,alpha=1,antialiased=True )

	fig.colorbar(plot2, ax=ax2,location = 'left', label = 'Prob. Dens.', format='%.0e') #set colorbar

	###plot for perturbed ground state
	ax3 = fig.add_subplot(223,projection="3d")
	ax3.set_xlabel('x')
	ax3.set_ylabel('y')
	ax3.set_zlabel('z')
	ax3.set_title('Ground State Probability Distribution (Perturbed)')
	dens = np.conjugate(psi_perturb[0])*psi_perturb[0] #|psi|^2

	###plotting prob. dens. with a mask so any points <= frac*max_dens don't crowd the density plot
	plot3 = ax3.scatter(X[dens >= frac*max_dens],Y[dens >= frac*max_dens],Z[dens >= frac*max_dens],
	c=dens[dens >= frac*max_dens],s=0.001,cmap = cm.seismic,alpha=1,antialiased=True )

	fig.colorbar(plot3, ax=ax3, location = 'left', label = 'Prob. Dens.', format='%.0e') #set colorbar

	###plot for perturbed excited state
	ax4 = fig.add_subplot(224,projection="3d")
	ax4.set_xlabel('x')
	ax4.set_ylabel('y')
	ax4.set_zlabel('z')
	ax4.set_title('First Excited State Probability Distribution (Perturbed)')
	dens = np.conjugate(psi_perturb[1])*psi_perturb[1] #|psi|^2

	###plotting prob. dens. with a mask so any points <= frac*max_dens don't crowd the density plot
	plot4 = ax4.scatter(X[dens >= frac*max_dens],Y[dens >= frac*max_dens],Z[dens >= frac*max_dens],
	c=dens[dens >= frac*max_dens],s=0.001,cmap = cm.seismic,alpha=1,antialiased=True )

	fig.colorbar(plot4, ax=ax4, location = 'left', label = 'Prob. Dens.', format='%.0e') #set colorbar


def plot_eigenvalues(eigenvalues, eigenvalues_perturb):
	###helper function for plotting eigenvalues
	###sort eigenvalue arrays for plotting
	eigenvalues.sort()
	eigenvalues_perturb.sort()

	###n denotes the index of each energy in the eigenvalue array
	n = np.arange(0, len(eigenvalues), 1)

	###plot perturbed and unperturbed energies
	plt.figure(1)
	plt.subplot(211)
	plt.grid()
	plt.scatter(n, eigenvalues, marker='_')
	plt.title("Energy Spectrum Unperturbed")
	plt.xlabel('Eigenvalue Index')
	plt.ylabel(r'$E_n$ ')

	plt.subplot(212)
	plt.grid()
	plt.scatter(n, eigenvalues_perturb, marker='_')
	plt.title("Energy Spectrum Perturbed")
	plt.xlabel('Eigenvalue Index')
	plt.ylabel(r'$E_n $')

	plt.figure(2)
	plt.subplot(111)
	plt.grid()
	plt.scatter(n, eigenvalues, marker='_', label = 'unperturbed')
	plt.scatter(n, eigenvalues_perturb, marker='_', label = 'perturbed')
	plt.legend()
	plt.title("Energy Spectrums Compared")
	plt.xlabel('Eigenvalue Index')
	plt.ylabel(r'$E_n $')

if __name__ == '__main__':
    ###argparser for taking in parameters from command line
    parser = argparse.ArgumentParser()

	###get all the args
    parser.add_argument('-ptype', help='Type of Potential to plot [gaussian, coulomb, QHO]', type=str, default = 'QHO')
    parser.add_argument('-w', help='Harmonic Oscillator Natural Frequency', type=float, default = 1)
    parser.add_argument('-A', help='Gaussian Potential Amplitude', type=float, default = 1)
    parser.add_argument('-mu', help='Gaussian Potential Center', type=float, default = 1)
    parser.add_argument('-sigma', help='Gaussian Potential Standard Deviation', type=float, default = 1)
    parser.add_argument('-N', help='Number of Points in 1D (total grid points = N^3)', type=int, default = 50)
    parser.add_argument('-L', help='Length of Box', type=float, default = 7)
    parser.add_argument('-l', help = "Perturbation Strength", type = float, default = 0.1)
    parser.add_argument('-n', help='Number of Eigenvalues calculated', type=int, default = 4)

    args = parser.parse_args()._get_kwargs()

	###assigning args to variables
    N = [arg[1] for arg in args if arg[0] == 'N'][0]
    L = [arg[1] for arg in args if arg[0] == 'L'][0]
    A = [arg[1] for arg in args if arg[0] == 'A'][0]
    mu = [arg[1] for arg in args if arg[0] == 'mu'][0]
    sigma = [arg[1] for arg in args if arg[0] == 'sigma'][0]
    w = [arg[1] for arg in args if arg[0] == 'w'][0]
    ptype = [arg[1] for arg in args if arg[0] == 'ptype'][0]
    n = [arg[1] for arg in args if arg[0] == 'n'][0]
    l = [arg[1] for arg in args if arg[0] == 'l'][0]

    main(ptype, N, L, l, w, mu, sigma, A, n)
