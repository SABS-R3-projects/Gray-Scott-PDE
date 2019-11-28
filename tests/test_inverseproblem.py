import unittest

import numpy as np
from scipy.integrate import odeint
from pints import ForwardModel, SumOfSquaresError, OptimisationController, XNES

from grayscott.inverseproblem import InverseProblem

def exponential_growth(x0: float, Lambda: float, times: np.ndarray) -> np.ndarray:
    """Analytic solution of exponential growth ODE.

    Arguments:
        x0 {float} -- Initial state, assuming t0=0.
        Lambda {float} -- Growth factor.
        times {np.ndarray} -- Times at which function will be evaluated.

    Returns:
        np.ndarray -- State values at time points in time.
    """
    return x0 * np.exp(Lambda * times)


class TestModel(ForwardModel):
    """Exponential model for testing.
    """

    def exponential_growth_ODE(self, x: float, t: float, Lambda: float) -> float:
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

    def simulate(self, parameters: np.ndarray, times: np.ndarray) -> np.ndarray:
        x0, Lambda = parameters
        return odeint(self.exponential_growth_ODE, x0, times, args=(Lambda,))

def test_find_parameter():
    """Example based testing of self.find_parameter().
    """
    x0 = 1.0
    Lambda = 0.1
    parameters = np.array([x0, Lambda])
    times = np.arange(0, 10, 0.1)

    analytical_solution = exponential_growth(x0, Lambda, times)

    model = TestModel()
    numerical_solution = model.simulate(parameters, times)[:, 0]

    # testing the scipys odeint works as expected.
    assert np.allclose(a=numerical_solution, b=analytical_solution, rtol=1.0e-7)

    # generate data
    noise_std = 0.1

    number_data_points = len(times)
    data_times = times
    data_ys = analytical_solution + noise_std * np.random.randn(number_data_points)

    problem = InverseProblem(model, data_times, data_ys)
    initial_parameters = [0.9, 0.15] # x0, Lambda

    estimated_parameters, _ = problem.find_parameter(initial_parameters)

    assert np.allclose(a=estimated_parameters, b=parameters, rtol=5.0e-02)


def test_sample_posterior():
    """Example based testing of self.sample_posterior().
    """
    x0 = 1.0
    Lambda = 0.1
    parameters = np.array([x0, Lambda])
    times = np.arange(0, 10, 0.1)
    analytical_solution = exponential_growth(x0, Lambda, times)

    # generate data
    noise_std = 0.2
    number_data_points = len(times)
    data_times = times
    data_ys = analytical_solution + noise_std * np.random.randn(number_data_points)

    # find intitial parameters for sampling
    model = TestModel()
    problem = InverseProblem(model, data_times, data_ys)
    initial_parameters = [0.9, 0.15] # x0, Lambda
    estimated_parameters, _ = problem.find_parameter(initial_parameters)

    # prior bounds for sampling
    estimated_parameters = [estimated_parameters[0], estimated_parameters[1], noise_std]
    lower_bound = [0, 0, 0]
    upper_bound = [5, 10, 5]
    print(problem.sample_posterior(initial_parameter=estimated_parameters,
                                   lower_bound=lower_bound,
                                   upper_bound=upper_bound))

    assert True






