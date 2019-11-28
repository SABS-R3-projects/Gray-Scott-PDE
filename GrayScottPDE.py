import pints
import numpy as np
from scipy.sparse import spdiags
from math import floor

class GrayScott(pints.ForwardModel):
    """
    The Gray Scott model inheriting the pints.ForwardModel class
    """
    def __init__(self, N, u0=None, v0=None):
        super().__init__()
        self.N = N
        N, N2, r = self.N, np.int(self.N / 2), 16

        # Check initial values
        if u0 is None:
            self.u0 = np.zeros((N, N))
            self.u0[N2 - r:N2 + r, N2 - r:N2 + r] = 0.50
        else:
            self.u0 = u0
            if len(self.u0) != N:
                raise ValueError('Initial value must have the same size as N')
            if np.any(self.u0 < 0):
                raise ValueError('Initial states can not be negative.')

        if v0 is None:
            self.v0 =np.zeros((N, N))
            self.v0[N2 - r:N2 + r, N2 - r:N2 + r] = 0.25
        else:
            self.v0 = v0
            if len(self.v0) != N:
                raise ValueError('Initial value must have the same size as N')
            if np.any(self.v0 < 0):
                raise ValueError('Initial states can not be negative.')

    def n_outputs(self):
        """
        Return the number of outputs
        :return:
        """
        return 2*self.N*self.N

    def n_parameters(self):
        """
        Return the dimension of the parameter vector. The Gray-Scott model has two parameters F and k
        """
        return 2

    def simulate(self, parameters, times):
        """
        Simulate the Gray-Scott PDE system with the finite difference method

        :param parameters: a two-dimensional parameter vector [F, k]
        :param times: time-points where the model is simulated, t0 = 0
        :return: simulated values of the PDE system [u, v]
        """

        def laplacian(N):
            """
            Calculate the Laplacian matrix using the using the five-point stencil finite difference method

            :param N: dimension of the Laplacian matrix to be calculated
            :return: the Laplacian matrix
            """
            e = np.ones(N ** 2)
            e2 = ([1] * (N - 1) + [0]) * N
            e3 = ([0] + [1] * (N - 1)) * N
            L = spdiags([-4 * e, e2, e3, e, e], [0, -1, 1, -N, N], N ** 2, N ** 2)
            return L

        def integrate(Nt, Du, Dv, F, k, L):
            """
            Solves the PDE

            :param Nt:
            :param Du:
            :param Dv:
            :param F:
            :param k:
            :param L:
            :return:
            """
            u_i = self.u0.reshape((self.N * self.N))
            v_i = self.v0.reshape((self.N * self.N))
            self.output = np.zeros(((floor(Nt/100)+1), 2*self.N*self.N))

            for i in range(Nt):
                uvv = u_i * v_i * v_i
                u_i = u_i + (Du * L.dot(u_i) - uvv + F * (1 - u_i))
                v_i = v_i + (Dv * L.dot(v_i) + uvv - (F + k) * v_i)
                if (i % 100 == 0):
                    self.output[floor((i+1)/100)] = np.hstack((u_i, v_i))

        L = laplacian(self.N)
        Du = parameters[0]
        Dv = parameters[1]
        F = parameters[2]
        k = parameters[3]
        Nt = len(times)

        integrate(Nt, Du, Dv, F, k, L)
        return self.output



