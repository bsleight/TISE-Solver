# TISE-Solver
Solves the TISE for a specified potential type and optional perturbation and plots the resulting energy spectrum and probability distributions. 

The main file can be run from the command line and takes the following parameters: 

-ptype       Type of Potential to plot ('gaussian', 'coulomb', 'well', or 'QHO') (default 'QHO', type: str) 
-w           Harmonic Oscillator Natural Frequency (default 1, type: float)
-A           Gaussian Potential Amplitude (default 1, type: float)
-mu          Gaussian Potential Center (default 1, type: float)
-sigma       Gaussian Potential Standard Deviation (default 1, type: float)
-N           Number of Points in 1D (total grid points = N^3) (default 50, type: int)
-L           Length of Box (default 7, type: float)
-l           Perturbation Strength (default 0.1, type: float)
-n           Number of Eigenvalues calculated (default 4, type: int)

The script will calculate the perturbed and unperturbed energy eigenvalues and probability distributions for the given potential type. 
The perturbation term is the L_z operator added to the Hamiltonian with a strength specified by the l parameter. 
