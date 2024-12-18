from scipy import constants
from KineticsSandbox.integrator import D1_stochastic
import numpy as np

kb = 0.001 * constants.R

class reweight():

    def __init__(self, system, n_states, tau, V_simulation, V_target):

        self.system = system
        self.n_states = n_states
        self.tau = tau
        self.V_simulation = V_simulation
        self.V_target = V_target

    def generate(self, N):

        """
        simulates the system at double well potential and Euler Maruyama scheme
        """
        eta = np.zeros(N)
        delta_eta = np.zeros(N)
        X = np.zeros(N)
        X[0] = self.system.x
        for i in range(N - 1):
            eta[i] = np.random.normal(0, 1)  # random number corresponding to random force
            delta_eta[i] = (np.sqrt(self.system.dt / (2 * kb * self.system.T * self.system.xi * self.system.m)) *
                            self.gradient(self.system.x)[0])
            D1_stochastic.EM(self.system, self.V_simulation, eta_k=eta[i])
            # update the system position
            X[i + 1] = self.system.x

        return X, eta, delta_eta

    def MSM(self, X, lagtime):

        """
        :param X: simulation path
        :param n_states: number of markov states
        :return: Transition matrix

        """

        bins = np.linspace(min(X) - 1e-6, max(X) + 1e-6, self.n_states + 1)
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

    def gradient(self, x):
        """

        :param x: position
        :param V_sim:  simulation potential
        :param V_target:  target potential
        :return: the gradient of bias potential
        """
        return np.array([self.V_simulation.force_ana(x)[0] - self.V_target.force_ana(x)[0],
                         self.V_target.potential(x) - self.V_simulation.potential(x)],
                        dtype=object)

    def reweighting_factor(self, X, eta, delta_eta, lagtime):

        """
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
            eta_ = eta[i : i + lagtime]
            delta_eta_ = delta_eta[i : i + lagtime]

            """calculate the reweighting factor"""

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


