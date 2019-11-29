import pints

class Inference():
    """
    Parameter inference of Gray-Scott data

    Parameters:
    ---------------
        model:
            - the model implemented as a Pde_solver class
        times:
            - the time-points of the obtained values
        values:
            - the obtained values of the Gray-Scott data
    """
    def __init__(self, model, times, values):
        self.problem = pints.MultiOutputProblem(model, times, values)

    def optimise(self):
        """
        Parameter inference using SNES (Seperable Natural Evolution Strategy).

        Returns:
        ---------------
            found_parameters:
                - found optimal parameters
        """
        #Define a score function, i.e the sum of squares error
        score = pints.SumOfSquaresError(self.problem)

        #Define the boundaries for F and k according to literature
        boundaries = pints.RectangularBoundaries([0.01, 0.01], [1.0, 1.0])

        #Starting point within the boundaries
        x0 = [0.05, 0.05]

        #Run SNES
        found_parameters, found_value = pints.optimise(score, x0, boundaries=boundaries, method=pints.SNES)
        return found_parameters