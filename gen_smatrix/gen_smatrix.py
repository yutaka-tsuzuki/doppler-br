import numpy as np
from scipy import constants

# constants
mc2 = constants.m_e * constants.c**2

# gamma matrices
G = np.array([	\
	[	\
		[0, 0, 1, 0],	\
		[0, 0, 0, 1],	\
		[1, 0, 0, 0],	\
		[0, 1, 0, 0]	\
	], [	\
		[0, 0, 0, 1],	\
		[0, 0, 1, 0],	\
		[0,	-1, 0, 0],	\
		[-1, 0, 0, 0]	\
	], [	\
		[0, 0, 0, -1j],	\
		[0, 0, 1j, 0],	\
		[0,	1j, 0, 0],	\
		[-1j, 0, 0, 0]	\
	], [	\
		[0, 0, 1, 0],	\
		[0, 0, 0, -1],	\
		[-1, 0, 0, 0],	\
		[0, 1, 0, 0]	\
	]	\
], dtype = "complex128")

# spinors of a free electron
# according to Relativistic Quantum Mechanics by W. Greiner
def u(p, s) :
	# energy
	E = np.sqrt(mc2**2 + np.dot(p, p) * c**2)
	# normalization factor
	factor = np.sqrt(0.5 * (E + mc2) / mc2)
	# spinor
	a_u = np.zeros((4, ), dtype = "complex128")
	# spin + or -
	if s > 0.0 :	# spin +
		a_u[0] = 1
		a_u[2] = p[2] * constants.c / (E + mc2)
		a_u[3] = (p[0] + 1j * p[1]) * constants.c / (E + mc2)
	else :	# spin -
		a_u[1] = 1
		a_u[2] = (p[0] - 1j * p[1]) * constants.c / (E + mc2)
		a_u[3] = - p[2] * constants.c / (E + mc2)
	
	return a_u

def ubar(p, s) :
	# energy
	E = np.sqrt(mc2**2 + np.dot(p, p) * c**2)
	# normalization factor
	factor = np.sqrt(0.5 * (E + mc2) / mc2)
	# spinor
	a_u = np.zeros((4, ), dtype = "complex128")
	# spin + or -
	if s > 0.0 :	# spin +
		a_u[0] = 1
		a_u[2] = p[2] * constants.c / (E + mc2)
		a_u[3] = (p[0] - 1j * p[1]) * constants.c / (E + mc2)
	else :	# spin -
		a_u[1] = 1
		a_u[2] = (p[0] + 1j * p[1]) * constants.c / (E + mc2)
		a_u[3] = - p[2] * constants.c / (E + mc2)
	
	return a_u

# contraction
def contr(v) :
	mat = np.zeros((4, 4))
	for i in range(4) :
		mat += G[i] * v[i]
	return mat

# fermion propagator
def propag(p, k) :
	# denominator
	denom = 2.0 * np.dot(p, k)
	# matrix
	mat = conrt(p) + contr(k) + constants.m_e * constants.c * np.eye(4)
