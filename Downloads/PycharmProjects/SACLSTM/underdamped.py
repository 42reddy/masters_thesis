from KineticsSandbox.system import system
from KineticsSandbox.integrator import D1_stochastic
import numpy as np
from scipy import constants

kb = constants.R * 0.001


class underdamped():

    def __init__(self, system, n_states, tau, V_simulation, V_target):

        self.system = system
        self.n_states = n_states
        self.tau = tau
        self.V_simulation = V_simulation
        self.V_target = V_target

    def gradient(self, x):
        """

        :param x: position
        :return: the gradient of bias potential
        """
        return np.array([self.V_simulation.force_ana(x)[0] - self.V_target.force_ana(x)[0],
                         self.V_target.potential(x) - self.V_simulation.potential(x)],
                        dtype=object)

    def BAOAB(self, potential, eta_k=None):
        """
        Perform a full Langevin integration step for the BAOAB algorithm

        Parameters:
        - system (object): An object representing the physical system undergoing Langevin integration.
                          It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                          (friction coefficient), 'T' (temperature), and 'dt' (time step).
        - potential (object): An object representing the potential energy landscape of the system.
                             It should have a 'force' method that calculates the force at a given position.
        - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                            in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

        Returns:
        None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
        """

        D1_stochastic.B_step(self.system, potential, half_step=True)
        delta_eta = (np.exp(-self.system.xi * self.system.dt) * (self.system.dt / 2) * (self.gradient(self.system.x)[0])
                     / (np.sqrt(kb * self.system.T * self.system.m * (1 - np.exp(-2 * self.system.xi * self.system.dt)))))
        D1_stochastic.A_step(self.system, half_step=True)
        D1_stochastic.O_step(self.system, eta_k=eta_k[0])
        D1_stochastic.A_step(self.system, half_step=True)
        D1_stochastic.B_step(self.system, potential, half_step=True)

        return delta_eta

    def BOAOB(self, potential, eta_k=None):
        """
        Perform a full Langevin integration step for the BOAOB algorithm

        Parameters:
        - system (object): An object representing the physical system undergoing Langevin integration.
                          It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                          (friction coefficient), 'T' (temperature), and 'dt' (time step).
        - potential (object): An object representing the potential energy landscape of the system.
                             It should have a 'force' method that calculates the force at a given position.
        - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                            in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

        Returns:
        None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
        """

        D1_stochastic.B_step(self.system, potential, half_step=True)
        D1_stochastic.O_step(self.system, half_step=True, eta_k=eta_k[0])
        delta_eta = np.exp(-self.system.xi * self.system.dt / 2) * (self.system.dt / 2) * (self.gradient(self.system.x))[
            0] / (np.sqrt(kb * self.system.T * self.system.m * (1 - np.exp(-self.system.xi * self.system.dt))))
        D1_stochastic.A_step(self.system)
        D1_stochastic.O_step(self.system, half_step=True, eta_k=eta_k[1])
        delta_eta1 = (self.system.dt / 2) * (self.gradient(self.system.x))[
            0] / (np.sqrt(kb * self.system.T * self.system.m * (1 - np.exp(-self.system.xi * self.system.dt))))
        D1_stochastic.B_step(self.system, potential, half_step=True)

        return delta_eta, delta_eta1

    def ABOBA(self, potential, eta_k=None):
        """
        Perform a full Langevin integration step for the OABAO algorithm

        Parameters:
        - system (object): An object representing the physical system undergoing Langevin integration.
                          It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                          (friction coefficient), 'T' (temperature), and 'dt' (time step).
        - potential (object): An object representing the potential energy landscape of the system.
                             It should have a 'force' method that calculates the force at a given position.
        - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                            in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

        Returns:
        None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
        """

        D1_stochastic.A_step(self.system, half_step=True)
        D1_stochastic.B_step(self.system, potential, half_step=True)
        D1_stochastic.O_step(self.system, eta_k=eta_k[0])
        delta_eta = (np.exp(- self.system.xi * self.system.dt) + 1) * (self.gradient(self.system.x)[0] * self.system.dt / 2) / (
            np.sqrt(kb * self.system.T * self.system.m * (1 - np.exp(-2 * self.system.xi * self.system.dt))))
        D1_stochastic.B_step(self.system, potential, half_step=True)
        D1_stochastic.A_step(self.system, half_step=True)

        return delta_eta

    def generate(self, N):

        """
        simulates the system at double well potential and Euler Maruyama scheme
        """
        delta_eta = np.zeros(N)
        delta_eta1 = np.zeros(N)
        eta1 = np.zeros(N)
        eta2 = np.zeros(N)
        X = np.zeros(N)
        v = np.zeros(N)
        X[0] = self.system.x
        for i in range(N - 1):
            eta1[i] = np.random.normal(0, 1)  # random number corresponding to random force
            eta2[i] = np.random.normal(0, 1)
            eta = [eta1[i], eta2[i]]
            delta_eta[i]= self.ABOBA(self.V_simulation, eta_k=eta)
            # update the system position
            X[i + 1] = self.system.x
            v[i + 1] = self.system.v

        return X, v, eta1, delta_eta

    def MSM(self, X, lagtime):

        """
        :param X: simulation path
        :param n_states: number of markov states
        :return: Transition matrix

        """

        bins = np.linspace(min(X), max(X), self.n_states)
        discretized_path = np.digitize(X, bins, right=False) - 1

        """
        calculate the count matrix traversing through the discretized simulation path
        """

        count_matrix = np.zeros((self.n_states, self.n_states))

        for i in range(len(discretized_path) - lagtime):
            count_matrix[discretized_path[i], discretized_path[i + lagtime]] += 1

        count_matrix = 0.5 * (count_matrix + np.transpose(count_matrix))

        transition_matrix = count_matrix / np.sum(count_matrix, axis= 1, keepdims=True)

        return transition_matrix

    def equilibrium_dist(self, transition_matrix):
        A = np.transpose(transition_matrix) - np.eye(self.n_states)

        A = np.vstack((A, np.ones(self.n_states)))
        b = np.zeros(self.n_states)
        b = np.append(b, 1)
        pi = np.linalg.lstsq(A, b, rcond=None)[0]

        return pi

    def reweighting_factor(self, X, eta, delta_eta, lagtime, eta1=None, delta_eta1=None):

        """
        :param delta_eta1: second random number at target potential
        :param eta1: second random number at simulation
        :param eta: random number at simulation potential
        :param delta_eta: random number at target potential
        :param X: simulation path
        :return: reweighting factors for each path
        """

        len_paths = int((len(X) - lagtime))  # length of the paths generated from sliding window method

        """ calculate the reweighting factor for each observed path """
        M = np.zeros(len_paths)
        for i in range(len_paths):
            """calculate eta and delta_eta for each path"""
            eta_ = eta[i : i + lagtime ]
            delta_eta_ = delta_eta[i : i + lagtime]

            """calculate the reweighting factor"""

            if delta_eta1 is not None:    # switch between schemes with single and two random numbers
                eta_1 = eta1[i: i + lagtime]
                delta_eta_1 = delta_eta1[i: i + lagtime]

                M[i] = np.exp((self.V_simulation.potential(X[i]) - self.V_target.potential(X[i])) / (kb * self.system.T)) * (
                           np.exp(-np.sum(eta_ * delta_eta_)) * np.exp(-0.5 * np.sum(delta_eta_ ** 2))) * (
                           np.exp(-np.sum(eta_1 * delta_eta_1)) * np.exp(-0.5 * np.sum(delta_eta_1 ** 2)))
            else:
                M[i] = np.exp((self.V_simulation.potential(X[i]) - self.V_target.potential(
                    X[i])) / (kb * self.system.T)) * (np.exp(-np.sum(eta_ * delta_eta_)) * np.exp(-0.5 * np.sum(delta_eta_ ** 2)))

        return M

    def reweighted_MSM(self, X, M, lagtime):

        """
        :param X: simulated path
        :param paths: generated paths from X
        :param M: rewighting factors
        :param lag_time: lag time for the markov model
        :return: reweighted transition matrix
        """

        # X = X[(X <= 1.6) & (X >= -1.7)]
        bins = np.linspace(min(X), max(X), self.n_states + 1)
        discretized_path = np.zeros(len(X))

        for i in range(self.n_states):
            if i == 0:
                discretized_path[(X >= bins[0]) & (X < bins[1])] = 0
            elif i == self.n_states - 1:
                discretized_path[(X >= bins[self.n_states - 1]) & (X <= bins[self.n_states])] = self.n_states - 1
            else:
                discretized_path[(X >= bins[i]) & (X < bins[i+1])] = i
        discretized_path = discretized_path.astype(int)



        count_matrix = np.zeros((self.n_states, self.n_states))
        #discretized_path[discretized_path == self.n_states+1] = self.n_states
        #discretized_path[discretized_path == 0] = 1
        #discretized_path -=

        len_paths = int(len(X) - lagtime)
        for i in range(len_paths):
            path = discretized_path[i: i + lagtime]

            '''transitions = path[:self.tau - lag_time], path[lag_time:]  # Two slices for transitions

            # Count the transitions
            for start, end in zip(*transitions):
                count_matrix[start, end] += M[i]'''

            count_matrix[path[0], path[lagtime - 1]] += M[i]

        count_matrix = 0.5 * (count_matrix + np.transpose(count_matrix))

        transition_matrix = count_matrix / np.sum(count_matrix, axis=1, keepdims=True)

        return transition_matrix

    def implied_timescales(self, X, X2, eta, delta_eta, lagtimes):

        lagtimes = lagtimes.astype(int)

        eigen1 = np.zeros(len(lagtimes))
        eigen2 = np.zeros(len(lagtimes))
        eigen3 = np.zeros(len(lagtimes))

        for i in range(len(lagtimes)):
            print(i)
            T_direct = self.MSM(X, lagtimes[i])
            T_direct = np.nan_to_num(T_direct, nan=0)
            eig = np.linalg.eigvals(T_direct)
            eig = np.sort(eig)
            print(eig[-2])
            eigen1[i] = -lagtimes[i] / np.log(eig[-2])

            T_target = self.MSM(X2, lagtimes[i])
            T_target = np.nan_to_num(T_target, nan=0)
            eig = np.linalg.eigvals(T_target)
            eig = np.sort(eig)
            print(eig[-2])
            eigen2[i] = -lagtimes[i] / np.log(eig[-2])

            M = self.reweighting_factor(X, eta, delta_eta, lagtimes[i])
            T_reweighted = self.reweighted_MSM(X, M, lagtimes[i])
            T_reweighted = np.nan_to_num(T_reweighted, nan=0)
            eig = np.linalg.eigvals(T_reweighted)
            eig = np.sort(eig)
            print(eig[-2])
            eigen3[i] = -lagtimes[i] / np.log(eig[-2])

        return eigen1, eigen2, eigen3






