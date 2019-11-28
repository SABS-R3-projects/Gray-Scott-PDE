import numpy as np
import pints

from GrayScottPDE import GrayScott
from Plots import Plot
from Inference import Inference

N = 200
model = GrayScott(N)
parameters = [0.14, 0.06, 0.060, 0.062]
times = np.arange(0,10*10+10, 10)
values = model.simulate(parameters, times)
plot = Plot(N)
plot.animation(values, "animations2.gif")

#inference = Inference(model, times, values)
#inference.optimise()