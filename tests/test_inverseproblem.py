import unittest
import numpy as np
import numpy as np
from scipy.integrate import odeint
from pints import ForwardModel, SumOfSquaresError, OptimisationController, XNES

import sys, os
cwd = os.getcwd()
repo = os.path.dirname(cwd)
sys.path.append(repo + '/python_files')
from Pde_solver import Solver
from Inference import Inference

def exponential_growth(x0, Lambda, times):
    """Analytic solution of exponential growth ODE.

    Arguments:
        x0 {float} -- Initial state, assuming t0=0.
        Lambda {float} -- Growth factor.
        times {np.ndarray} -- Times at which function will be evaluated.

    Returns:
        np.ndarray -- State values at time points in time.
    """
    return [x0[0] * np.exp(Lambda * times), x0[1] * np.exp(Lambda * times)]

class TestModel(ForwardModel):
    """Exponential model for testing.
    """

    def exponential_growth_ODE(self, x, t, Lambda):
        """Right hand side of exponential growth ODE.

        Arguments:
            x {float} -- Current state.
            t {float} -- Current time.
            Lambda {float} -- Growth factor.

        Returns:
            float -- dx / dt
        """
        return x * Lambda

    def n_parameters(self) -> int:
        """Returns number of parameters

        Returns:
            [int] -- Exponential growth model has two parameters
        """
        return 2

    def n_outputs(self):
        """Returns number of outputs."""
        return 2

    def simulate(self, parameters, times):
        x0 = [parameters[0], parameters[0]]
        Lambda = parameters[1]
        return odeint(self.exponential_growth_ODE, x0, times, args=(Lambda,))


def test_find_parameter():
    """Example based testing of self.find_parameter().
    """
    x0 = [0.5, 0.5]
    Lambda = 0.1
    parameters = np.array([0.5, Lambda])
    times = np.arange(0, 10, 0.1)

    analytical_solution = np.transpose(exponential_growth(x0, Lambda, times))

    model = TestModel()
    numerical_solution = model.simulate(parameters, times)

    # testing the scipys odeint works as expected.
    assert np.allclose(a=numerical_solution, b=analytical_solution, rtol=1.0e-7)

    # generate data
    noise_std = 0.1

    data_times = times
    data_ys = analytical_solution + noise_std * np.random.normal(size=analytical_solution.shape)

    inference = Inference(model, data_times, data_ys)

    estimated_parameters = inference.optimise()

    assert np.allclose(a=estimated_parameters, b=parameters, rtol=5.0e-02)
    return True





