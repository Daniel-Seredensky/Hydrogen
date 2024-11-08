import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Hydrogen:
    # Physical constants
    def __init__(self,n,l,m,numSamples = 10000):
        self.a0 = 1.0  # Bohr radius in atomic units
        self.n=n
        self.l=l
        self.m=m
        self.numSamples = numSamples

    @staticmethod
    def getStates():
        return [[1,0,0],[2,0,0],[2,1,-1],[2,1,0],[2,1,1]]
    
    def radial_wave_function(self,r):
        rho = 2 * r / (self.n * self.a0)
        prefactor = np.sqrt((2 / (self.n * self.a0)) ** 3 * np.math.factorial(self.n - self.l - 1) / (2 * self.n * np.math.factorial(self.n + self.l)))
        laguerre_poly = sp.assoc_laguerre(rho, self.n - self.l - 1, 2 * self.l + 1)
        R_nl = prefactor * rho ** self.l * np.exp(-rho / 2) * laguerre_poly
        return R_nl

    def angular_wave_function(self,theta, phi):
        Y_lm = sp.sph_harm(self.m, self.l, phi, theta)
        return Y_lm

    def sample_radial(self, num_samples):
        r_max = self.n * self.a0 * 10
        r = np.linspace(0, r_max, 1000)
        R_nl = self.radial_wave_function(r)
        P_radial = r**2 * np.abs(R_nl)**2
        P_radial /= np.max(P_radial)  # Normalize for rejection sampling

        samples = []
        while len(samples) < num_samples:
            r_proposal = np.random.uniform(0, r_max)
            R_nl_proposal = self.radial_wave_function(r_proposal)
            P_proposal = r_proposal**2 * np.abs(R_nl_proposal)**2
            P_proposal /= np.max(P_radial)
            u = np.random.uniform(0, 1)
            if u < P_proposal:
                samples.append(r_proposal)
        return np.array(samples)

    def sample_angular(self,num_samples):
        theta = np.arccos(1 - 2 * np.random.uniform(0, 1, num_samples))
        phi = np.random.uniform(0, 2 * np.pi, num_samples)
        Y_lm = self.angular_wave_function(theta, phi)
        P_angular = np.abs(Y_lm)**2
        P_angular /= np.max(P_angular)  # Normalize for rejection sampling

        samples_theta = []
        samples_phi = []
        for i in range(num_samples):
            u = np.random.uniform(0, 1)
            if u < P_angular[i]:
                samples_theta.append(theta[i])
                samples_phi.append(phi[i])

        if len(samples_theta) < num_samples:
            additional_theta, additional_phi = self.sample_angular(num_samples - len(samples_theta))
            samples_theta.extend(additional_theta)
            samples_phi.extend(additional_phi)

        return np.array(samples_theta), np.array(samples_phi)

    def generate_samples(self,num_samples):
        r_samples = self.sample_radial(num_samples)
        theta_samples, phi_samples = self.sample_angular(num_samples)
        x = r_samples * np.sin(theta_samples) * np.cos(phi_samples)
        y = r_samples * np.sin(theta_samples) * np.sin(phi_samples)
        z = r_samples * np.cos(theta_samples)
        return x, y, z

    def visualize_electron_cloud(self):
        x,y,z = self.generate_samples(self.numSamples)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x,y,z, s=1, alpha=0.5)
        ax.set_title(f'Electron Cloud for n={self.n}, l={self.l}, m={self.m}')
        ax.set_xlabel('x (a.u.)')
        ax.set_ylabel('y (a.u.)')
        ax.set_zlabel('z (a.u.)')
        plt.show()
