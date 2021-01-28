import numpy as np
from scipy import special, constants
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


n_entry = 100000

# constants
R_BOHR = 4.0 * constants.pi * constants.epsilon_0 * constants.hbar**2 / (constants.m_e * constants.e**2)


# methods
def Y(l, m, theta, phi) :
	# because of absurdity of scipy we should permute the ordering of variables
	return special.sph_harm(m, l, phi, theta)

def R(n, l, r) :
	# normalized radius
	#rho = 2.0 * r / (n * R_BOHR)
	rho = 2.0 * r / n
	# first term, a normalization constant with respect to n and l
	term_1 = np.sqrt( (8.0 * special.factorial(n - l - 1)) / (n**3 * 2.0 * n * special.factorial(n + l)) )
	# second term, exp(-0.5 * rho) * rho^l
	term_2 = np.exp(-0.5 * rho) * np.power(rho, l)
	# third term, a generalized Laguerre polynomial
	term_3 = special.eval_genlaguerre(n - l - 1, 2 * l + 1, rho)

	return term_1 * term_2 * term_3

def RY(n, l, m, x) :
	return R(n, l, x[0]) * Y(l, m, x[1], x[2])

def RY2(n, l, m, x) :
	ry = RY(n, l, m, x)
	return (ry * ry.conjugate()).real

def random_thetaphi() :
	# random variables
	phi = np.random.uniform(0.0, 2.0 * constants.pi)
	cos_theta = np.random.uniform(-1.0, 1.0)
	theta = np.arccos(cos_theta)
	
	return theta, phi

def gradRY2(n, l, m, x, eps = 1.0e-9) :
	ry2_0 = RY2(n, l, m, x)
	df_dr = (RY2(n, l, m, x + np.array([eps, 0, 0])) - ry2_0) / eps
	df_dtheta = (RY2(n, l, m, x + np.array([0, eps, 0])) - ry2_0) / eps
	df_dphi = (RY2(n, l, m, x + np.array([0, 0, eps])) - ry2_0) / eps
	return np.array([df_dr, df_dtheta, df_dphi])

# Monte Carlo sempling
class SampleMC :
	def __init__(self, n, l, m, r0 = 1.0 * R_BOHR, eps = 1.0e-3) :
		theta_init, phi_init = random_thetaphi()
		self.w = np.array([r0 * np.sin(theta_init) * np.cos(phi_init), r0 * np.sin(theta_init) * np.sin(phi_init), r0 * np.cos(theta_init)])
		self.n = n
		self.l = l
		self.m = m
		self.r0 = r0
		self.eps = eps
		self.var = np.sqrt(2.0 * self.eps)

	# Langevin sampling
	def sample(self) :
		# sample g
		g = np.random.normal(np.array([0.0, 0.0, 0.0]), self.var, (3, ))
		# convert old vector to spherical coordinates
		r = np.linalg.norm(self.w)
		theta = np.arctan2(np.sqrt(self.w[0]**2 + self.w[1]**2), self.w[2])
		phi = np.arctan2(self.w[1], self.w[0])
		if phi < 0.0 :
			phi += 2.0 * constants.pi
		# gradient
		ry2 = RY2(self.n, self.l, self.m, np.array([r, theta, phi]))
		grad_s = gradRY2(self.n, self.l, self.m, np.array([r, theta, phi]))
		# new vector
		r_grad = grad_s[0]
		theta_grad = grad_s[1]
		phi_grad = grad_s[2]
		x_grad = np.sin(theta) * np.cos(phi) * r_grad + np.cos(theta) * np.cos(phi) * theta_grad / r - np.sin(phi) * phi_grad / (r * np.sin(phi))
		y_grad = np.sin(theta) * np.sin(phi) * r_grad + np.cos(theta) * np.sin(phi) * theta_grad / r - np.cos(phi) * phi_grad / (r * np.sin(phi))
		z_grad = np.cos(theta) * r_grad - np.sin(theta) * theta_grad / r
		w_new = self.w + self.eps * np.array([x_grad, y_grad, z_grad]) / ry2 + g
		self.w = w_new

		return self.w
	
	# calculation
	def get_integrand_p(self, p) :
		# sample new vector
		self.sample()
		# convert old vector to spherical coordinates
		r = np.linalg.norm(self.w)
		theta = np.arctan2(np.sqrt(self.w[0]**2 + self.w[1]**2), self.w[2])
		phi = np.arctan2(self.w[1], self.w[0])
		# probability
		ry2 = RY2(self.n, self.l, self.m, np.array([r, theta, phi]))
		# integrand
		ry = RY(self.n, self.l, self.m, np.array([r, theta, phi]))
		ker = np.exp(complex(- 1.0j * np.dot(p, self.w)))

		return ker / ry, ry2

# main
def main() :
	# sampler
	sampler = SampleMC(3, 2, 2, 1.0, 1.0e-2)
	# momentum
	p = np.random.normal(0.0, 1.0, (3, ))	# * constants.hbar / R_BOHR
	# storage
	a_integ = np.zeros((n_entry, ), dtype = "complex128")
	a_prob = np.zeros((n_entry, ))
	a_sum_integ = np.zeros((n_entry, ), dtype = "complex128")
	a_sum_prob = np.zeros((n_entry, ))
	# sample
	for i_entry in range(n_entry) :
		integ, prob = sampler.get_integrand_p(p)
		a_integ[i_entry] = integ
		a_prob[i_entry] = prob
		a_sum_integ[i_entry] = np.sum(a_integ) / (i_entry + 1)
		a_sum_prob[i_entry] = np.sum(a_prob) * np.sum(a_prob)
		if (i_entry + 1) % 1000 == 0 :
			print("%d / %d" % (i_entry + 1, n_entry))
	# plot
	plt.plot(range(1, n_entry + 1), (a_sum_integ * a_sum_integ.conjugate()).real)
	plt.yscale("log")
	#plt.ylim(1.0e-3, 1.0e3)
	plt.grid()
	plt.show()


# magic code
if __name__ == "__main__" :
	main()
