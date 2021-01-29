import numpy as np
from scipy import constants

from gen_dist import gen_dist_p
from gen_smatrix import gen_smatrix

# sampling amplitudes
class SampleAmp :
	def __init__(self, n, l, m, z, eps = 1.0e-3) :
		p0 = np.abs(np.random.normal(0.0, eps))
		theta_init, phi_init = random_thetaphi()
		self.p = np.array([p0 * np.sin(theta_init) * np.cos(phi_init), p0 * np.sin(theta_init) * np.sin(phi_init), p0 * np.cos(theta_init)])
		self.s = np.random.choice([-0.5, 0.5])
		self.n = n
		self.l = l
		self.m = m
		self.z = z
		self.eps = eps
		self.sampler = SampleMC(n, l, m, z, eps)
		self.p_in = self.sampler.sample()
		self.s_in = np.random.choice([-0.5, 0.5])
	
	def set_in(self, k_in, e_in) :
		self.k_in = k_in
		self.e_in = e_in
		self.p_in = self.sampler.sample()
		self.s_in = np.random.choice([-0.5, 0.5])

	def get_ortho3(k) :
		v = np.array([1.0, 0.0, 0.0])
		if np.dot(v, k) / np.linalg.norm(k) < 1.0e-3 :
			v = np.array([0.0, 1.0, 0.0])
		u1 = np.cross(k, v)
		u1 /= np.linalg.norm(u1)
		u2 = np.cross(k, u1)
		u2 /= np.linalg.norm(u2)
		return u1, u2
		
	def sample(self) :
		# sample w_new
		p_new = np.random.normal(self.p, self.eps, (3, ))
		s_new = np.random.choice([-0.5, 0.5])
		# calculate k_out from conservation law
		k_out = self.p_in + self.k_in - self.p
		k_out_new = self.p_in + self.k_in - p_new
		e_out_1, e_out_2 = self.get_ortho3(k_out)
		e_out_new_1, e_out_new_2 = self.get_ortho3(k_out_new)
		# amplitude
		amp_1 = gen_smatrix.amp_compton(self.p_in, self.s_in, self.k_in, self.e_in, self.p, self.s, k_out, e_out_1)
		amp_2 = gen_smatrix.amp_compton(self.p_in, self.s_in, self.k_in, self.e_in, self.p, self.s, k_out, e_out_2)
		amp = amp_1 + amp_2
		amp_new_1 = gen_smatrix.amp_compton(self.p_in, self.s_in, self.k_in, self.e_in, p_new, s_new, k_out_new, e_out_new_1)
		amp_new_2 = gen_smatrix.amp_compton(self.p_in, self.s_in, self.k_in, self.e_in, p_new, s_new, k_out_new, e_out_new_2)
		amp_new = amp_new_1 + amp_new_2
		# probability
		prob = amp_new / amp
		if prob > 1.0 :
			prob = 1.0
		rr = np.random.uniform(0.0, 1.0)
		if rr < prob :
			self.p = p_new
			self.s = s_new

		return self.p, self.s

