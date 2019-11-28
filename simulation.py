from GrayScottPDE import GrayScott
import numpy as np
import pints
import matplotlib.pyplot as plt

N = 200
model = GrayScott(N)
parameters = [0.060, 0.062]
times = np.arange(0,30000)
values = model.simulate(parameters, times)
print(values.shape)
values1 = values[0:N*N]
plt.pcolor(values1.reshape((N, N)), cmap=plt.cm.RdBu)
plt.savefig("GSPDE.png")
values2 = values[N*N:2*N*N]
plt.pcolor(values2.reshape((N, N)), cmap=plt.cm.RdBu)
plt.savefig("GSPDE2.png")
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