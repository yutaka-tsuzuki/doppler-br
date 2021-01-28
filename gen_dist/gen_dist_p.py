import numpy as np
from scipy import special, constants
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


n_entry = 10000

# constants
R_BOHR = 4.0 * constants.pi * constants.epsilon_0 * constants.hbar**2 / (constants.m_e * constants.e**2)


# methods
# spherical harmonics
def Y(l, m, theta, phi) :
	# because of absurdity of scipy we should permute the ordering of variables
	return special.sph_harm(m, l, phi, theta)

# radial function in coordinate space
# r should be provided by the unit of Bohr radius
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

#def gradRY2(n, l, m, x, eps = 1.0e-9) :
#	ry2_0 = RY2(n, l, m, x)
#	df_dr = (RY2(n, l, m, x + np.array([eps, 0, 0])) - ry2_0) / eps
#	df_dtheta = (RY2(n, l, m, x + np.array([0, eps, 0])) - ry2_0) / eps
#	df_dphi = (RY2(n, l, m, x + np.array([0, 0, eps])) - ry2_0) / eps
#	return np.array([df_dr, df_dtheta, df_dphi])

# radial funciton in momentum space
# see Hey. APJ 61, 28 (1993) for detail
# p should be provided by the atomic unit, p_unit = constants.hbar * Z / R_BOHR
def F(n, l, p) :
	# first term
	term_1 = - (-1.0j)**l * np.sqrt( (2.0 * special.factorial(n - l - 1)) / (constants.pi * special.factorial(n + l)) )
	# second term
	term_2 = n**2 * 2**(2 * (l + 1)) * special.factorial(l)
	# third term
	term_3 = n**l * p**l / (n**2 * p**2 + 1)**(l + 2)
	# fourth term, Gegenbauer polynomial
	term_4 = special.eval_gegenbauer(n - l - 1, l + 1, [(n**2 * p**2 - 1) / (n**2 * p**2 + 1)])[0]

	return term_1 * term_2 * term_3 * term_4

def FY(n, l, m, k) :
	return F(n, l, k[0]) * Y(l, m, k[1], k[2])

def FY2(n, l, m, k) :
	fy = FY(n, l, m, k)
	return (fy * fy.conjugate()).real

#def gradFY2(n, l, m, k, eps = 1.0e-9) :
#	fy2_0 = FY2(n, l, m, k)
#	df_dp = (FY2(n, l, m, k + np.array([eps, 0, 0])) - fy2_0) / eps
#	df_dtheta = (FY2(n, l, m, k + np.array([0, eps, 0])) - fy2_0) / eps
#	df_dphi = (FY2(n, l, m, k + np.array([0, 0, eps])) - fy2_0) / eps
#	return np.array([df_dp, df_dtheta, df_dphi])

#def FY2d3(n, l, m, k) :
#	fy2 = FY2(n, l, m, k)
#	return fy2 / ((k[0])**2 * np.sin(k[1]))

#def gradFY2d3(n, l, m, k, eps = 1.0e-9) :
#	fy2d3_0 = FY2d3(n, l, m, k)
#	df_dp = (FY2d3(n, l, m, k + np.array([eps, 0, 0])) - fy2d3_0) / eps
#	df_dtheta = (FY2d3(n, l, m, k + np.array([0, eps, 0])) - fy2d3_0) / eps
#	df_dphi = (FY2d3(n, l, m, k + np.array([0, 0, eps])) - fy2d3_0) / eps
#	return np.array([df_dp, df_dtheta, df_dphi])

def random_thetaphi() :
	# random variables
	phi = np.random.uniform(0.0, 2.0 * constants.pi)
	cos_theta = np.random.uniform(-1.0, 1.0)
	theta = np.arccos(cos_theta)
	
	return theta, phi

# Monte Carlo sempling
class SampleMC :
	def __init__(self, n, l, m, z, eps = 1.0e-3) :
		p0 = np.abs(np.random.normal(0.0, eps))
		theta_init, phi_init = random_thetaphi()
		self.w = np.array([p0 * np.sin(theta_init) * np.cos(phi_init), p0 * np.sin(theta_init) * np.sin(phi_init), p0 * np.cos(theta_init)])
		self.n = n
		self.l = l
		self.m = m
		self.z = z
		self.p0 = p0
		self.eps = eps
		self.var = np.sqrt(2.0 * self.eps)

	# Metropolis sampling
	def sample(self) :
		# sample w_new
		w_new = np.random.normal(self.w, self.eps, (3, ))
		# convert vectors to spherical coordinates
		p = np.linalg.norm(self.w)
		theta = np.arctan2(np.sqrt(self.w[0]**2 + self.w[1]**2), self.w[2])
		phi = np.arctan2(self.w[1], self.w[0])
		if phi < 0.0 :
			phi += 2.0 * constants.pi
		p_new = np.linalg.norm(w_new)
		theta_new = np.arctan2(np.sqrt(w_new[0]**2 + w_new[1]**2), w_new[2])
		phi_new = np.arctan2(w_new[1], w_new[0])
		if phi_new < 0.0 :
			phi_new += 2.0 * constants.pi
		# probability
		fy2 = FY2(self.n, self.l, self.m, np.array([p, theta, phi]))
		fy2_new = FY2(self.n, self.l, self.m, np.array([p_new, theta_new, phi_new]))
		prob = fy2_new / fy2
		if prob > 1.0 :
			prob = 1.0
		# condition
		rr = np.random.uniform(0.0, 1.0)
		if rr < prob :
			self.w = w_new

		return self.w

	# Metropolis sampling
	def sample_p(self) :
		# sample p_new
		p_new = np.random.normal(self.p0, self.eps)
		p = self.p0
		# probability
		f = F(self.n, self.l, p)
		f_new = F(self.n, self.l, p_new)
		f2p2 = (f * f.conjugate()).real * p**2
		f2p2_new = (f_new * f_new.conjugate()).real * p_new**2
		prob = f2p2_new / f2p2
		if prob > 1.0 :
			prob = 1.0
		rr = np.random.uniform(0.0, 1.0)
		if rr < prob :
			self.p0 = p_new

		return self.p0


# test
def plot_F() :
	a_p = np.linspace(0.0, 2.0, 100)
	a_f = np.zeros((100, ))
	for i in range(100) :
		f = F(4, 0, a_p[i])
		f2 = (f * f.conjugate()).real
		f2p2 = f2 * a_p[i]**2
		a_f[i] = f2p2
	plt.plot(a_p, a_f)
	plt.xlim(0.0, 2.0)
	plt.show()

# main
def main() :
	# sampler
	sampler = SampleMC(3, 2, -1, 1, 1.0e-1)
	# momentum
	#p = np.random.normal(0.0, 1.0, (3, ))	# * z * constants.hbar / R_BOHR
	# storage
	a_p_sample = np.zeros((n_entry, ))
	a_p_weight = np.zeros((n_entry, ))
	a_p_x = np.zeros((n_entry, ))
	a_p_y = np.zeros((n_entry, ))
	a_p_z = np.zeros((n_entry, ))
	# sample
	for i_entry in range(n_entry) :
		p_sample = sampler.sample()
		a_p_sample[i_entry] = np.linalg.norm(p_sample)
		a_p_weight[i_entry] = 1.0
		a_p_x[i_entry] = p_sample[0]
		a_p_y[i_entry] = p_sample[1]
		a_p_z[i_entry] = p_sample[2]
		if (i_entry + 1) % 1000 == 0 :
			print("%d / %d" % (i_entry + 1, n_entry))
	# plot
	plt.hist(a_p_sample, weights = a_p_weight, bins = 100, range = (0.0, 2.0), density = True)
	plt.xlim(0.0, 2.0)
	plt.grid()
	plt.show()
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = "3d")
	ax.scatter(a_p_x, a_p_y, zs = a_p_z, s = 1)
	plt.show()


# magic code
if __name__ == "__main__" :
	main()
