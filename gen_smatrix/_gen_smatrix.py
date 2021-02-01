import numpy as np
from scipy import constants
import matplotlib.pyplot as plt


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
	E = np.sqrt(1.0 + np.dot(p, p))
	# normalization factor
	factor = np.sqrt(0.5 * (E + 1.0))
	# spinor
	a_u = np.zeros((4, ), dtype = "complex128")
	# spin + or -
	if s > 0.0 :	# spin +
		a_u[0] = 1
		a_u[2] = p[2] / (E + 1.0)
		a_u[3] = (p[0] + 1j * p[1]) / (E + 1.0)
	else :	# spin -
		a_u[1] = 1
		a_u[2] = (p[0] - 1j * p[1]) / (E + 1.0)
		a_u[3] = - p[2] / (E + 1.0)
	
	return a_u

def ubar(p, s) :
	# energy
	E = np.sqrt(1.0 + np.dot(p, p))
	# normalization factor
	factor = np.sqrt(0.5 * (E + 1.0))
	# spinor
	a_u = np.zeros((4, ), dtype = "complex128")
	# spin + or -
	if s > 0.0 :	# spin +
		a_u[0] = 1
		a_u[2] = p[2] / (E + 1.0)
		a_u[3] = (p[0] - 1j * p[1]) / (E + 1.0)
	else :	# spin -
		a_u[1] = 1
		a_u[2] = (p[0] + 1j * p[1]) / (E + 1.0)
		a_u[3] = - p[2] / (E + 1.0)
	
	return a_u

# contraction
def contr4(v) :
	mat = np.zeros((4, 4), dtype = "complex128")
	for i in range(4) :
		mat += G[i] * v[i]
	return mat

def contr3(s, v) :
	mat = np.zeros((4, 4), dtype = "complex128")
	mat += G[0] * s
	for i in range(1, 4) :
		mat += G[i] * v[i-1]
		#mat -= G[i] * v[i-1]
	return mat

def contr3alt(s, v) :
	mat = np.zeros((4, 4), dtype = "complex128")
	mat += G[0] * s
	for i in range(1, 4) :
		#mat += G[i] * v[i-1]
		mat -= G[i] * v[i-1]
	return mat

# fermion propagator
def propag(p, k, sgn = 1.0) :
	# energy
	ep = np.sqrt(np.dot(p, p) + 1.0)
	ek = np.linalg.norm(k)
	# inner product
	pk = ep * ek - np.dot(p, k)
	# matrix
	mat = np.zeros((4, 4), dtype = "complex128")
	if sgn > 0.0 :
		mat = 0.5j * (contr3(ep, p) + contr3(ek, k) + np.eye(4, dtype = "complex128"))
		mat /= pk
	else :
		mat = 0.5j * (contr3(ep, p) - contr3(ek, k) + np.eye(4, dtype = "complex128"))
		mat /= -pk
	return mat

# second-order Compton scattering amplitude
def amp_compton(p_in, s_in, k_in, e_in, p_out, s_out, k_out, e_out) :
	# factor, fine-structure constant
	factor = -1 * constants.alpha
	# u
	term_u_out = ubar(p_out, s_out)
	term_u_in = u(p_in, s_in)
	# e contraction
	term_e_out = contr3alt(0.0, e_out)
	term_e_in = contr3alt(0.0, e_in)
	# propagator
	pr1 = propag(p_in, k_in)
	pr2 = propag(p_in, k_out, sgn = -1.0)
	# calculation
	v = np.matmul( G[0], np.matmul( term_e_out, np.matmul( pr1, np.matmul(term_e_in, term_u_in.T) ) ) ).T
	v += np.matmul( G[0], np.matmul( term_e_in, np.matmul( pr2, np.matmul(term_e_out, term_u_in.T) ) ) ).T
	amp = factor * np.dot(term_u_out, v)

	return amp

# second-order Compton scattering S-matrix, NOT normalized properly
def smatrix_compton(p_in, s_in, k_in, e_in, p_out, s_out, k_out, e_out) :
	# amplitude
	amp = amp_compton(p_in, s_in, k_in, e_in, p_out, s_out, k_out, e_out)
	# factor
	ene_in = np.sqrt(np.dot(p_in, p_in) + 1.0)
	ene_out = np.sqrt(np.dot(p_out, p_out) + 1.0)
	k0_in = np.linalg.norm(k_in)
	k0_out = np.linalg.norm(k_out)
	factor = np.sqrt(ene_in * ene_out * k0_in * k0_out)

	return amp / factor

# cross section, NOT normalized properly
def cs_compton(p_in, s_in, k_in, e_in, p_out, s_out, k_out, e_out) :
	# smatrix
	smatrix = smatrix_compton(p_in, s_in, k_in, e_in, p_out, s_out, k_out, e_out)
	# factor
	#kappa = k_in / np.linalg.norm(k_in)
	#v_rel = np.linalg.norm(k_in - p_in) * constants.c

	#return (smatrix * smatrix.conjugate()).real / v_rel
	return (smatrix * smatrix.conjugate()).real

# test
def main() :
	# norm
	def find_norm(p_in, k_in, theta, phi) :
		kappa = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
		norm_k_in = np.linalg.norm(k_in)
		ene_in = np.sqrt(np.dot(p_in, p_in) + 1.0)
		numer = ene_in * norm_k_in - np.dot(p_in, k_in)
		denom = ene_in + norm_k_in - np.dot(p_in + k_in, kappa)
		return numer / denom
	
	def get_ortho3(k) :
		v = np.array([1.0, 0.0, 0.0])
		v2 = np.array([0.0, 1.0, 0.0])
		if np.dot(v, k) > np.dot(v2, k) :
			v = v2
		u1 = np.cross(k, v)
		u1 /= np.linalg.norm(u1)
		u2 = np.cross(k, u1)
		u2 /= np.linalg.norm(u2)
		return u1, u2
	
	ene_in = 1.0e6 * constants.e	# J
	# k_in, e_in
	k_in = np.array([0.0, 0.0, ene_in / mc2])
	e_in = np.array([1.0, 0.0, 0.0])
	p_in = np.zeros((3, ))
	# storage
	a_theta = np.linspace(0.0, constants.pi, 100)
	a_val = np.array([])
	for theta in a_theta :
		phi = 0.5 * constants.pi
		k0_out = find_norm(p_in, k_in, theta, phi)
		k_out = np.array([k0_out * np.sin(theta) * np.cos(phi), k0_out * np.sin(theta) * np.sin(phi), k0_out * np.cos(theta)])
		k0_out_orth = find_norm(p_in, k_in, theta, 0.0)
		k_out_orth = np.array([k0_out_orth * np.sin(theta) * np.cos(0.0), k0_out_orth * np.sin(theta) * np.sin(0.0), k0_out_orth * np.cos(theta)])
		p_out = k_in + p_in - k_out
		p_out_orth = k_in + p_in - k_out_orth
		e_out_1, e_out_2 = get_ortho3(k_out)
		e_out_1_orth, e_out_2_orth = get_ortho3(k_out_orth)
		# amplitude
		#smatrix = 0.0
		#smatrix_orth = 0.0
		den = 0.0
		for s_in in [-0.5, 0.5] :
			for s_out in [-0.5, 0.5] :
				cs_1 = cs_compton(p_in, s_in, k_in, e_in, p_out, s_out, k_out, e_out_1)
				cs_2 = cs_compton(p_in, s_in, k_in, e_in, p_out, s_out, k_out, e_out_2)
				cs_orth_1 = cs_compton(p_in, s_in, k_in, e_in, p_out_orth, s_out, k_out_orth, e_out_1_orth)
				cs_orth_2 = cs_compton(p_in, s_in, k_in, e_in, p_out_orth, s_out, k_out_orth, e_out_2_orth)
				#den = cs_1 + cs_2 + cs_orth_1 + cs_orth_2
				#den = cs_2
				den = cs_orth_1 + cs_orth_2
		a_val = np.append(a_val, den * np.sin(theta))
		#a_val = np.append(a_val, k0_out)
	# plot
	plt.plot(a_theta, a_val)
	plt.show()


if __name__ == "__main__" :
	main()
