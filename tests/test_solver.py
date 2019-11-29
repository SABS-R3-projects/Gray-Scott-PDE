## Convergence test
import sys, os
cwd = os.getcwd()
repo = os.path.dirname(cwd)
sys.path.append(repo + '/python_files')
from Pde_solver import Solver
import numpy as np

def convergence_test():
    def_n_times = 16000
    solv = Solver(n_save_frames=4000, n_time_points=def_n_times, model='gray-scott', n_grid=32, fix_seed=True)
    F = 0.035
    k = 0.06
    ## This particular model is known to converge
    tmp = solv.solve(parameters=[F, k], verbose=False, til_convergence=True)  # parameters = [F, k]
    assert solv.n_times < def_n_times  # soft test
    # assert solv.n_times == 4044  # hard test
    return True   # return success

def diffusion_test():
    """Test for correct diffusion and periodic boundary conditions:"""
    solv = Solver(n_save_frames=4000, n_time_points=300, model='heat', n_grid=32, fix_seed=True)

    assert np.isclose(solv.init_u_mat.sum(), solv.u_mat.sum())  # assert convergence of energy (i.e diffusion and boundary)
    return True
