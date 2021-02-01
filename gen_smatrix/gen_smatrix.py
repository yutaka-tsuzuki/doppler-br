import numpy as np
import math
from scipy import constants
import matplotlib.pyplot as plt


# constants
mc2 = constants.m_e * constants.c**2

# gamma matrices
G = np.array([	\
	[	\
		[1, 0, 0, 0],	\
		[0, 1, 0, 0],	\
		[0, 0, -1, 0],	\
		[0, 0, 0, -1]	\
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

# 4-momentum class
class Momentum4 :

	def __init__(self, p3 = np.array([0.0, 0.0, 0.0])) :
		self.p4 = np.zeros((4, ))
		for i in range(3) :
			self.p4[i + 1] = p3[i]

	#def set_p3(self, p3) :
	#	for i in range(3) :
	#		self.p4[i + 1] = p3[i]
		
	def contr(self) :
		mat = np.zeros((4, 4), dtype = "complex128")
		mat += G[0] * self.p4[0]
		for i in range(3) :
			mat += G[i + 1] * self.p4[i + 1]
			#mat -= G[i + 1] * self.p4[i + 1]
		return mat
	
	def __add__(self, k) :
		q = Momentum4()
		for i in range(4) :
			q.p4[i] = self.p4[i] + k.p4[i]
		return q
	
	def __sub__(self, k) :
		q = Momentum4()
		for i in range(4) :
			q.p4[i] = self.p4[i] - k.p4[i]
		return q

	def __mul__(self, k) :
		pr = self.p4[0] * k.p4[0]
		for i in range(3) :
			pr -= self.p4[i + 1] * k.p4[i + 1]
		return pr

class Polarization4(Momentum4) :
	
	def contrv(self) :
		mat = np.zeros((4, 4), dtype = "complex128")
		mat += G[0] * self.p4[0]
		for i in range(3) :
			mat -= G[i + 1] * self.p4[i + 1]
			#mat += G[i + 1] * self.p4[i + 1]
		return mat

class Momentum4Photon(Momentum4) :
	
	def __init__(self, p3 = np.array([0.0, 0.0, 0.0])) :
		super().__init__(p3)
		self.p4[0] = np.linalg.norm(p3)

class Momentum4Electron(Momentum4) :
	
	def __init__(self, p3 = np.array([0.0, 0.0, 0.0])) :
		super().__init__(p3)
		self.p4[0] = np.sqrt(1.0 + np.dot(p3, p3))


# spinors of a free electron
# according to Relativistic Quantum Mechanics by W. Greiner
def u(p, s) :
	# energy
	E = p.p4[0]
	# normalization factor
	factor = np.sqrt(0.5 * (E + 1.0))
	# spinor
	a_u = np.zeros((4, ), dtype = "complex128")
	# spin + or -
	if s > 0.0 :	# spin +
		a_u[0] = 1.0
		a_u[1] = 0.0
		a_u[2] = p.p4[3] / (E + 1.0)
		a_u[3] = (p.p4[1] + 1j * p.p4[2]) / (E + 1.0)
	else :	# spin -
		a_u[0] = 0.0
		a_u[1] = 1.0
		a_u[2] = (p.p4[1] - 1j * p.p4[2]) / (E + 1.0)
		a_u[3] = - p.p4[3] / (E + 1.0)
	
	return a_u

def ubar(p, s) :
	# energy
	E = p.p4[0]
	# normalization factor
	factor = np.sqrt(0.5 * (E + 1.0))
	# spinor
	a_u = np.zeros((4, ), dtype = "complex128")
	# spin + or -
	if s > 0.0 :	# spin +
		a_u[0] = 1.0
		a_u[1] = 0.0
		a_u[2] = p.p4[3] / (E + 1.0)
		a_u[3] = (p.p4[1] - 1j * p.p4[2]) / (E + 1.0)
	else :	# spin -
		a_u[0] = 0.0
		a_u[1] = 1.0
		a_u[2] = (p.p4[1] + 1j * p.p4[2]) / (E + 1.0)
		a_u[3] = - p.p4[3] / (E + 1.0)
	
	return np.matmul(a_u, G[0])

# fermion propagator
def propag(p, k, sgn = 1.0) :
	# matrix
	mat = np.zeros((4, 4), dtype = "complex128")
	if sgn > 0.0 :
		pk = (p + k) * (p + k) - 1.0
		mat = (0.5j / pk) * (p.contr() + k.contr() + np.eye(4, dtype = "complex128"))
	else :
		pk = (p - k) * (p - k) - 1.0
		mat = (0.5j / pk) * (p.contr() - k.contr() + np.eye(4, dtype = "complex128"))
	return mat

# second-order Compton scattering amplitude
def amp_compton(p_in, s_in, k_in, e_in, p_out, s_out, k_out, e_out) :
	# factor, fine-structure constant
	factor = -1 * constants.alpha
	# u
	term_u_out = ubar(p_out, s_out)
	term_u_in = u(p_in, s_in)
	# e contraction
	term_e_out = e_out.contrv()
	term_e_in = e_in.contrv()
	# propagator
	pr1 = propag(p_in, k_in)
	pr2 = propag(p_in, k_out, sgn = -1.0)
	# calculation
	v1 = np.matmul( term_e_out, np.matmul( pr1, np.matmul(term_e_in, term_u_in.T) ) ).T
	v2 = np.matmul( term_e_in, np.matmul( pr2, np.matmul(term_e_out, term_u_in.T) ) ).T
	amp1 = factor * np.dot(term_u_out, v1)
	amp2 = factor * np.dot(term_u_out, v2)

	return amp1 + amp2

# second-order Compton scattering S-matrix, NOT normalized properly
def smatrix_compton(p_in, s_in, k_in, e_in, p_out, s_out, k_out, e_out) :
	# amplitude
	amp = amp_compton(p_in, s_in, k_in, e_in, p_out, s_out, k_out, e_out)
	# factor
	ene_in = p_in.p4[0]
	ene_out = p_out.p4[0]
	k0_in = k_in.p4[0]
	k0_out = k_out.p4[0]
	factor = np.sqrt(ene_in * ene_out * k0_in * k0_out)
	#factor = ene_out * k0_out

	#return amp / factor
	#return amp * factor
	return amp

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
def energy_ratio(e_init, theta) : 
	return 1.0 / (1.0 + (e_init / 511.0) * (1 - math.cos(theta)))

# r0 [m]
def KN_cross_diff(e_init, theta, r0 = 2.818e-15) :
	hoge = energy_ratio(e_init, theta)
	s = math.sin(theta)
	return 0.5 * r0 * r0 * hoge * hoge * (hoge + 1.0 / hoge - s * s)

def KN_cross_theta(e_init, theta) : 
	return 2.0 * math.pi * KN_cross_diff(e_init, theta) * math.sin(theta)

# test
def main() :
	# norm
	def find_norm(p_in, k_in, theta, phi) :
		kappa = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
		norm_k_in = k_in.p4[0]
		ene_in = p_in.p4[0]
		numer = p_in * k_in
		denom = ene_in + norm_k_in - np.dot((p_in + k_in).p4[1:4], kappa)
		return numer / denom
	
	def get_ortho3(k) :
		k3 = k.p4[1:4]
		v = np.array([1.0, 0.0, 0.0])
		v2 = np.array([0.0, 1.0, 0.0])
		if np.dot(v, k3) > np.dot(v2, k3) :
			v = v2
		u1 = np.cross(k3, v)
		u1 /= np.linalg.norm(u1)
		u2 = np.cross(k3, u1)
		u2 /= np.linalg.norm(u2)
		return Polarization4(u1), Polarization4(u2)
	
	ene_in = 1000.0e3 * constants.e	# J
	# k_in, e_in
	k_in = Momentum4Photon([0.0, 0.0, ene_in / mc2])
	e_in = Polarization4([1.0, 0.0, 0.0])
	p_in = Momentum4Electron()
	# storage
	a_theta = np.linspace(0.0, constants.pi, 181)
	a_val = np.array([])
	for theta in a_theta :
		phi = 0.25 * constants.pi
		k0_out = find_norm(p_in, k_in, theta, phi)
		k_out = Momentum4Photon([k0_out * np.sin(theta) * np.cos(phi), k0_out * np.sin(theta) * np.sin(phi), k0_out * np.cos(theta)])
		#p_out = (k_in + p_in) - k_out
		p_out = Momentum4Electron((k_in + p_in - k_out).p4[1:4])
		e_out_1, e_out_2 = get_ortho3(k_out)
		# S-matrix
		den = 0.0
		#for s_in, s_out in [[-0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5]] :
		for s_in, s_out in [[0.5, 0.5], [-0.5, -0.5]] :
			smatrix_1 = smatrix_compton(p_in, s_in, k_in, e_in, p_out, s_out, k_out, e_out_1)
			smatrix_2 = smatrix_compton(p_in, s_in, k_in, e_in, p_out, s_out, k_out, e_out_2)
			den += (smatrix_1 * smatrix_1.conjugate()).real + (smatrix_2 * smatrix_2.conjugate()).real
		a_val = np.append(a_val, den * np.sin(theta))
	# plot
	plt.plot(a_theta, a_val / np.sum(a_val))
	y_kn = np.array([KN_cross_theta(ene_in / 1.0e3 / constants.e, theta) for theta in a_theta])
	plt.plot(a_theta, y_kn / np.sum(y_kn))
	plt.show()


if __name__ == "__main__" :
	main()
