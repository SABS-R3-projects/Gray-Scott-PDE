import matplotlib.pyplot as plt
from Pde_solver import Solver
from Inference import Inference

## Initiate PDE solver class 
solv = Solver(n_save_frames=20, n_time_points=4000, model='gray-scott')

## Set F & k parameters and solve
## spots: F=0.035, k=0.065
## maze-like: F=0.035, k=0.06

tmp = solv.solve(parameters=[0.035, 0.060])  # parameters = [F, k]

inference = Inference(solv, solv.save_times, solv.save_u_mat.reshape(solv.n_save_frames, solv.n_x * solv.n_y))
inference.optimise()
