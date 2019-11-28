import numpy as np
from typing import List
from pints import SingleOutputProblem, SumOfSquaresError, OptimisationController, XNES
from pints import SingleChainMCMC, MCMCController, LogPosterior, GaussianLogLikelihood, UniformLogPrior

class InverseProblem(SingleOutputProblem):
    def __init__(self, model, times, values):
        super(InverseProblem, self).__init__(model, times, values)

    def find_parameter(self, initial_parameters: np.ndarray) -> List:
        """Minimises Least Squares Error to find optimal model parameters.

        Arguments:
            initial_parameters {np.ndarray} -- Initial point in parameter space for optimistation.
        
        Returns:
            List -- Parameter estimates, Least Squared Error
        """
        error_measure = SumOfSquaresError(self)
        optimisation = OptimisationController(error_measure, initial_parameters, method=XNES)

        estimated_parameters, score = optimisation.run()

        return [estimated_parameters, score]

    def sample_posterior(self, initial_parameter, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self._set_logposterior()

        mcmc = MCMCController(log_pdf=self.log_posterior,
                              chains=1,
                              x0=initial_parameter,
                              method=SingleChainMCMC)

        mcmc.set_max_iterations(1000)

        return mcmc.run()

    def _set_logposterior(self):
        """Sets log posterior.
        """
        self._set_log_likelihood()
        self._set_log_prior()

        self.log_posterior = LogPosterior(self.log_likelihood, self.log_prior)

    def _set_log_likelihood(self):
        """Sets log likelihood assuming Gaussian noise with unknown noise.
        """
        self.log_likelihood = GaussianLogLikelihood(self)

    def _set_log_prior(self):
        """Sets log prior to be uniform for all parameters within the given boundaries.
        """
        self.log_prior = UniformLogPrior(self.lower_bound, self.upper_bound)




