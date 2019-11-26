from GrayScottPDE import GrayScott
import numpy as np

N = 20
rdSolver = GrayScott(N)
parameters = [0.6, 0.65]
times = np.arange(0,5)
values = rdSolver.simulate(parameters, times)
print(values)