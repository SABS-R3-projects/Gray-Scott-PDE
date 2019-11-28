import numpy as np
import pints

from GrayScottPDE import GrayScott
from Plots import Plot

N = 200
model = GrayScott(N)
parameters = [0.14, 0.06, 0.060, 0.062]
times = np.arange(0,10000)
values = model.simulate(parameters, times)
plot = Plot(N)
plot.animation(values)

# Add some noise
#values += np.random.normal(0, 0.02, values.shape)

# Create an object with links to the model and time series
#problem = pints.MultiOutputProblem(model, times, values)

# Select a score function
#score = pints.SumOfSquaresError(problem)

# Select some boundaries
#boundaries = pints.RectangularBoundaries([0.3, 0.3], [0.7, 0.7])

# Select a starting point
#x0 = [0.5, 0.5]

# Perform an optimization using SNES (see docs linked above).
#found_parameters, found_value = pints.optimise(score, x0, boundaries=boundaries, method=pints.SNES)

#print(found_parameters)