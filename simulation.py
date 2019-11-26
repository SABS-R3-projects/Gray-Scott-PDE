from GrayScottPDE import GrayScott
import numpy as np
import pints

N = 100
model = GrayScott(N)
parameters = [0.6, 0.65]
times = np.arange(0,10)
values = model.simulate(parameters, times)
print(values.shape)
# Add some noise
values += np.random.normal(0, 0.02, values.shape)

# Create an object with links to the model and time series
problem = pints.MultiOutputProblem(model, times, values)

# Select a score function
score = pints.SumOfSquaresError(problem)

# Select some boundaries
boundaries = pints.RectangularBoundaries([0.3, 0.3], [0.7, 0.7])

# Select a starting point
x0 = [0.5, 0.5]

# Perform an optimization using SNES (see docs linked above).
found_parameters, found_value = pints.optimise(score, x0, boundaries=boundaries, method=pints.SNES)

print(found_parameters)