import pints

class Inference():
    def __init__(self, model, times, values):
        self.problem = pints.MultiOutputProblem(model, times, values)

    def optimise(self):
        score = pints.SumOfSquaresError(self.problem)
        boundaries = pints.RectangularBoundaries([0.01, 0.01, 0.01, 0.01], [1.0, 1.0, 0.1, 0.1])
        x0 = [0.05, 0.05, 0.05, 0.05]
        found_parameters, found_value = pints.optimise(score, x0, boundaries=boundaries, method=pints.SNES)
        print(found_parameters)
