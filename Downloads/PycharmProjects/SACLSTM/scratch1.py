from KineticsSandbox.system import system
import matplotlib.pyplot as plt
from KineticsSandbox.potential import D1
from overdamped import reweight
import numpy as np
from scipy import constants
import scipy

kb = constants.R * 0.001
m = 1
x = 1.5
v = 0
T = 300
xi = 50
dt = 0.01
h = 0.01

system = system.D1(m,x,v,T,xi,dt,h)

doublewell = D1.DoubleWell([1,0,1])
triplewell = D1.TripleWell([4])
bolhuis = D1.Bolhuis([0, 0, 1, 1, 1, 0])
reweighed_bolhuis  = D1.Bolhuis([0, 0, 1, 1, 1, 10])

y = np.linspace(-2, 2, 200)
potential = bolhuis.potential(y)
potential_reweighted = reweighed_bolhuis.potential(y)
plt.plot(y, potential, label = 'alpha = 0 ')
plt.plot(y, potential_reweighted, label = 'alpha = 4')
#plt.legend()

instance = reweight(system,100,200, doublewell, triplewell)

X, eta, delta_eta = instance.generate(int(1e7))

plt.hist(X, bins =1000, density=True)

transition_matrix = instance.MSM(X, 200)
transition_matrix = np.nan_to_num(transition_matrix, nan=0)
eq = instance.equilibrium_dist(transition_matrix)
plt.plot(eq)
plt.xlabel('coordinate along x')
plt.ylabel('frequency density')
plt.title('probability distribution (BAOAB)')

plt.hist(v, bins =1000, density=True)
plt.xlabel('coordinate along x ')
plt.ylabel('frequency density of velocities')
plt.title('maxwell boltzmann distribution (BAOAB)')

M = instance.reweighting_factor(X, eta, delta_eta,200)
reweighted_matrix = instance.reweighted_MSM(X, M, 200)
reweighted_matrix = np.nan_to_num(reweighted_matrix,nan=0)
eq = instance.equilibrium_dist(reweighted_matrix)
plt.plot(eq)
plt.title('reweighted distribution (BAOAB)')


instance1 = reweight(system, 100, 1000, triplewell, doublewell)
X1, eta1, delta_eta1 = instance1.generate(int(1e6))

instance2 = reweight(system, 100, 1000, doublewell, triplewell)
X2, eta2, delta_eta2 = instance2.generate(int(1e6))

eigen1 = np.zeros(10)
eigen2 = np.zeros(10)
eigen3 = np.zeros(10)
eig = np.zeros(100)
tau = np.linspace(10,900,10)


for i in range(10):

    T_ = instance2.MSM(X2, int(tau[i]))
    T_ = np.nan_to_num(T_, nan=0)
    eig = np.linalg.eigvals(T_)
    eig = np.sort(eig)
    print(eig[-2])
    eigen1[i] = -tau[i] / np.log(eig[-2])

    M = instance2.reweighting_factor(X2, eta2, delta_eta2, lagtime=int(tau[i]))
    Tr = instance2.reweighted_MSM(X2, M, int(tau[i]))
    Tr = np.nan_to_num(Tr, nan=0)
    eig = np.linalg.eigvals(Tr)
    eig = np.sort(eig)
    print(eig[-2])
    eigen2[i] = -tau[i] / np.log(eig[-2])

    T_ = instance2.MSM(X1, int(tau[i]))
    T_ = np.nan_to_num(T_, nan=0)
    eig = np.linalg.eigvals(T_)
    eig = np.sort(eig)
    print(eig[-2])
    eigen3[i] = -tau[i] / np.log(eig[-2])

plt.plot(tau, eigen1, label='bolhuis')
plt.plot(tau, eigen2, label ='reweighted eigenvector')
plt.plot(tau ,eigen3, label= 'simulated bolhuis')
plt.xlabel('lagtimes (n steps)')
plt.ylabel('implied timescales')
plt.title('implied timescales v lagtimes')
plt.legend()


x = np.linspace(min(X), max(X),100)

boltzmann = np.exp(-triplewell.potential(x)/(kb*T))

x_boltzmann = np.linspace(min(X), max(X), len(boltzmann))
x_pi = np.linspace(min(X), max(X), len(eq))

#plt.plot(x_pi,pi1/np.sum(pi1), label='simulated eigen vector')
plt.plot(x_boltzmann, boltzmann/np.sum(boltzmann), label = 'boltzmann distribution')
plt.plot(x_pi, eq/np.sum(eq), label = 'reweighted eigen vector')
plt.title('Overdamped EM scheme')
plt.xlabel('coordinate along x')
plt.ylabel('probability density')
plt.legend()

eigenvalues, eigenvectors = np.linalg.eig(reweighted_matrix.T)

# Sort eigenvalues and corresponding eigenvectors in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]


# Extract the second and third eigenvectors
second_eigenvector = eigenvectors[:, 2]
plt.plot(second_eigenvector)



bins = np.linspace(min(X)-1e-6, max(X) + 1e-6, 100 + 1)
bins
discretized_path = np.digitize(X, bins, right=False) - 1
max(discretized_path)

plt.hist(discretized_path,bins=100)

