import numpy as np
from typing import List
from pints import SingleOutputProblem, SumOfSquaresError, OptimisationController, XNES

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