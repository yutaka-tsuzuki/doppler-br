import numpy as np
from scipy import constants
import matplotlib.pyplot as plt

from gen_dist import gen_dist_p
from gen_smatrix import gen_smatrix

# constants
N = 3
L = 0
M = 0
Z = 16
E_in = 75.0e3 * constants.e	# J
P_UNIT = Z * constants.hbar / gen_dist_p.R_BOHR


n_entry = 10000

# sampling amplitudes
class SampleAmp :
	def __init__(self, n, l, m, z, eps = 1.0) :
		#p0 = np.abs(np.random.normal(0.0, eps))
		#theta_init, phi_init = gen_dist_p.random_thetaphi()
		#self.p = np.array([p0 * np.sin(theta_init) * np.cos(phi_init), p0 * np.sin(theta_init) * np.sin(phi_init), p0 * np.cos(theta_init)])
		self.s = np.random.choice([-0.5, 0.5])
		self.n = n
		self.l = l
		self.m = m
		self.z = z
		self.eps = eps
		self.sampler = gen_dist_p.SampleMC(n, l, m, z, eps)
		self.p_in = self.sampler.sample() * P_UNIT / (constants.m_e * constants.c)
		self.s_in = np.random.choice([-0.5, 0.5])
		#theta_k_init, phi_k_init = gen_dist_p.random_thetaphi()
		#k0 = self.find_norm(self.p_in, self.k_in, theta_k, phi_k)
		#self.k = np.array([k0 * np.sin(theta_k_init) * np.cos(phi_k_init), k0 * np.sin(theta_k_init) * np.sin(phi_k_init), k0 * np.cos(theta_k_init)])

	def find_norm(self, p_in, k_in, theta, phi) :
		kappa = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
		norm_k_in = np.linalg.norm(k_in)
		ene_in = np.sqrt(np.dot(p_in, p_in) + 1.0)
		numer = ene_in * norm_k_in - np.dot(p_in, k_in)
		denom = ene_in + norm_k_in - np.dot(p_in + k_in, kappa)
		return numer / denom
	
	def set_in(self, k_in, e_in) :
		self.k_in = k_in
		self.e_in = e_in
		self.p_in = self.sampler.sample() * P_UNIT / (constants.m_e * constants.c)
		self.s_in = np.random.choice([-0.5, 0.5])
		ene_in = np.sqrt(np.dot(self.p_in, self.p_in) + 1.0) + np.linalg.norm(self.k_in)
		theta_k, phi_k = gen_dist_p.random_thetaphi()
		k0 = self.find_norm(self.p_in, self.k_in, theta_k, phi_k)
		self.k = np.array([k0 * np.sin(theta_k) * np.cos(phi_k), k0 * np.sin(theta_k) * np.sin(phi_k), k0 * np.cos(theta_k)])

	def get_ortho3(self, k) :
		v = np.array([1.0, 0.0, 0.0])
		v2 = np.array([0.0, 1.0, 0.0])
		if np.dot(v, k) < np.dot(v2, k) :
			v = v2
		u1 = np.cross(k, v)
		u1 /= np.linalg.norm(u1)
		u2 = np.cross(k, u1)
		u2 /= np.linalg.norm(u2)
		return u1, u2
	
	def sample(self) :
		# test
		self.p_in = np.zeros((3, ))
		ene_in = np.sqrt(np.dot(self.p_in, self.p_in) + 1.0) + np.linalg.norm(self.k_in)
		# sample w_new
		s_new = np.random.choice([-0.5, 0.5])
		theta_k_new, phi_k_new = gen_dist_p.random_thetaphi()
		k0_new = self.find_norm(self.p_in, self.k_in, theta_k_new, phi_k_new)
		k_new = np.array([k0_new * np.sin(theta_k_new) * np.cos(phi_k_new), k0_new * np.sin(theta_k_new) * np.sin(phi_k_new), k0_new * np.cos(theta_k_new)])
		# calculate k_out from conservation law
		p_out = self.p_in + self.k_in - self.k
		p_out_new = self.p_in + self.k_in - k_new
		e_out_1, e_out_2 = self.get_ortho3(self.k)
		e_out_new_1, e_out_new_2 = self.get_ortho3(k_new)
		# amplitude
		amp_1 = gen_smatrix.amp_compton(self.p_in, self.s_in, self.k_in, self.e_in, p_out, self.s, self.k, e_out_1)
		amp_2 = gen_smatrix.amp_compton(self.p_in, self.s_in, self.k_in, self.e_in, p_out, self.s, self.k, e_out_2)
		den = ((amp_1 + amp_2) * (amp_1 + amp_2).conjugate()).real
		amp_new_1 = gen_smatrix.amp_compton(self.p_in, self.s_in, self.k_in, self.e_in, p_out_new, s_new, k_new, e_out_new_1)
		amp_new_2 = gen_smatrix.amp_compton(self.p_in, self.s_in, self.k_in, self.e_in, p_out_new, s_new, k_new, e_out_new_2)
		den_new = ((amp_new_1 + amp_new_2) * (amp_new_1 + amp_new_2).conjugate()).real
		# probability
		prob = den_new / den
		if prob > 1.0 :
			prob = 1.0
		rr = np.random.uniform(0.0, 1.0)
		if rr < prob :
			self.k = k_new
			self.s = s_new
			p_out = p_out_new

		return p_out, self.s, self.k


def main() :
	# sampler
	sampler = SampleAmp(N, L, M, Z, 1.0)
	# k_in, e_in
	#k_in = np.array([0.0, 0.0, E_in / (constants.m_e * constants.c**2)])
	k_in = np.array([0.0, 0.0, 0.001])
	e_in = np.array([1.0, 0.0, 0.0])
	sampler.set_in(k_in, e_in)
	# storage
	a_theta = np.zeros((n_entry, ))
	for i_entry in range(n_entry) :
		p_out, s_out, k_out = sampler.sample()
		theta = np.arccos(np.dot(k_in, k_out) / (np.linalg.norm(k_in) * np.linalg.norm(k_out)))
		a_theta[i_entry] = theta
		if (i_entry + 1) % 1000 == 0 :
			print("%d / %d" % (i_entry + 1, n_entry))
	plt.hist(a_theta, bins = 40, range = (0.0, constants.pi), density = True)
	plt.xlim(0.0, constants.pi)
	plt.show()


if __name__ == "__main__" :
	main()
