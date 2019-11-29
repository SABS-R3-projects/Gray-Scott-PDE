## PDE solver
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib.pylab as plab
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import pints

class Solver(pints.ForwardModel):
    """The PDE solver for the Gray-Scott equation and heat equation.

    Parameters:
    ---------------
        n_grid: int
            - Size of x and y dimensions
        n_time_points: int, default=8000
            - Number of time points
        model: str, 'gray-scott' or 'heat', Default='gray-scott'
            - defining model to solve
        n_save_frames: int, Default=100
            - How many frames (sampled evenly between 0 and n_time_points) to save
            for e.g. plotting purposes
        fix_seed: bool, Default=False
            - If true, the numpy random seed is fixed. This allows users to recreate
            exactly the same results each time (because initial conditions contain
            some random noise).
    """
    def __init__(self, n_grid=256, n_time_points=8000, model='gray-scott',
                n_save_frames=100, fix_seed=False):

        self.solve_eq = model
        self.eps_1 = 0.14  # hard-coded. These values were found to work well with dx=dy=dt=1
        self.eps_2 = 0.06

        ## Create x and y grids
        self.n_x = n_grid
        self.x_start = 0
        self.x_end = n_grid - 1
        self.x_arr = np.linspace(self.x_start, self.x_end, self.n_x)
        self.dx = np.diff(self.x_arr)[0]

        self.n_y= n_grid
        self.y_start = 0
        self.y_end = n_grid - 1
        self.y_arr = np.linspace(self.y_start, self.y_end, self.n_y)
        self.dy = np.diff(self.y_arr)[0]

        ## Create time array for solving (t_arr) and for saving/plotting (save_frames)
        self.n_times = n_time_points
        self.t_start = 0
        self.t_end = n_time_points - 1
        self.t_arr = np.linspace(self.t_start, self.t_end, self.n_times)
        self.dt = np.diff(self.t_arr)[0]
        self.n_save_frames = n_save_frames
        self.save_frames = np.linspace(self.t_start, self.t_end, self.n_save_frames)
        self.save_frames = np.round(self.save_frames)

        if fix_seed:  # fix random seed if required
            np.random.seed(0)

        ## Set initial conditions:
        if model == 'gray-scott':
            self.u_mat = np.ones((self.n_x, self.n_y))
        elif model == 'heat':
            self.u_mat = np.zeros((self.n_x, self.n_y))
        self.v_mat = np.zeros((self.n_x, self.n_y))

        spat_slice = slice(int(7/16 * self.n_x), int(9/16 * self.n_x))   # create square with initial offset
        self.u_mat[spat_slice, spat_slice] = 0.5  # standard init values for gray scott model
        self.v_mat[spat_slice, spat_slice] = 0.25
        self.u_mat[:, :] += ((np.random.rand(self.n_x, self.n_y) - 0.5) * 0.02)  # add uniform noise
        self.v_mat[:, :] += ((np.random.rand(self.n_x, self.n_y) - 0.5) * 0.02)
        self.init_u_mat = self.u_mat.copy()  # save initial values
        self.init_v_mat = self.v_mat.copy()

        if model == 'gray-scott':
            self.interaction = True
            self.decay = True
        elif model == 'heat':  # if heat equation: disable other terms but diffusion
            self.interaction = False
            self.decay = False

    def diffusion_update(self, old_u, diff_coef=1):
        """Perform numerical 2D laplacian update (via central finite difference)
        with periodic boundary conditions.

        Parameters:
        --------------------
            old_u: 2D numpy array
                - 2D array of old values, of which the laplacian is to be computed.
            diff_coef: float/int. Default = 1
                - diffusion coefficient.

        Returns:
        --------------------
            diff: 2D numpy array with size old_u.shape
                - diffusion term of old_u
        """

        diff = np.zeros_like(old_u)
        ## Compute central differnece in 2 dimensions:
        diff = diff_coef * (((np.roll(old_u, shift=1, axis=0)
                            + np.roll(old_u, shift=-1, axis=0)
                            - 2 * old_u) / (self.dx ** 2))
                            + ((np.roll(old_u, shift=1, axis=1)
                            + np.roll(old_u, shift=-1, axis=1)
                            - 2 * old_u) / (self.dy ** 2)))
        return diff

    def update_uv(self, old_u_mat, old_v_mat):
        """Update step of gray-scott or heat equation.

        Parameters:
        ---------------
            old_u_mat: 2D numpy array
                - u matrix
            old_v_mat: 2D numpy arrayL
                - v matrix

        Returns:
        ---------------
            new_u_mat: 2D numpy array with size old_u_mat.shape
                - updated u matrix
            new_v_mat: 2D numpy array with size old_v_mat.shape
                - updated v matrix
        """

        new_u_mat = old_u_mat.copy()
        new_v_mat = old_v_mat.copy()

        ## Perform diffusion step:
        new_u_mat += self.dt * self.diffusion_update(old_u=old_u_mat, diff_coef=self.eps_1)
        new_v_mat += self.dt * self.diffusion_update(old_u=old_v_mat, diff_coef=self.eps_2)

        if self.solve_eq == 'gray-scott':  ## IF gray=-scott model, perform other two actions:
            if self.interaction:
                uvv = old_u_mat * np.power(old_v_mat, 2)
                new_u_mat -= self.dt * uvv
                new_v_mat += self.dt * uvv

            if self.decay:
                new_u_mat += self.dt * self.F * (1 - old_u_mat)
                new_v_mat -= self.dt * (self.k + self.F) * old_v_mat

        return new_u_mat, new_v_mat

    def solve(self, parameters):
        """Solving function for PDE.

        Arguments:
        ----------------
            Parameters: list or np array of size 2
                1st entry:
                    F: float
                        - parameter of gray-scott model
                2nd entry:
                    k: float
                        - parameter of gray-scott model
        Returns:
        ----------------
            save_u_mat: float of len (n_save_frames * n_x * n_y)
                - Returns all save u matrices, collapsed to 1 dimension
        """

        assert len(parameters) == 2
        self.F = parameters[0]  # find F and k from input
        self.k = parameters[1]

        print(f'Solving {self.solve_eq} model in {self.n_times} time steps.\n\n')

        ## Create matrices to save frames during solving at regular intervals
        self.save_u_mat = np.zeros((self.n_save_frames, self.n_x, self.n_y))
        self.save_v_mat = np.zeros((self.n_save_frames, self.n_x, self.n_y))
        self.save_times = np.zeros(self.n_save_frames)
        i_save = 0

        ## Forward difference time solving loop:
        for i_tau in tqdm(range(self.n_times)):
            if i_tau in self.save_frames:  # if at the save interval, save matrices
                self.save_u_mat[i_save, :, :] = self.u_mat.copy()
                self.save_v_mat[i_save, :, :] = self.v_mat.copy()
                self.save_times[i_save] = i_tau
                i_save += 1
            old_u = self.u_mat.copy()
            old_v = self.v_mat.copy()
            self.u_mat, self.v_mat = self.update_uv(old_u_mat=old_u, old_v_mat=old_v)  # do update
        self.save_u_mat[-1, :, :] = self.u_mat.copy()
        self.save_v_mat[-1, :, :] = self.v_mat.copy()

        return self.save_u_mat.reshape(-1)  # only return u for parameter inference

    def plot2d(self, save_figures=False):
        """Function to plot u and v matrix at their current state.

        Arguments:
        --------------
            save_figures: bool, default = False
                - If true, save figure as png file
        """
        filename_uv = f'u_matrix_F={self.F}_k={self.k}.png'  # define file name to save to
        plt.rcParams['figure.figsize'] = (12, 5)
        plt.subplot(121)  # plot u matrix
        plt.pcolor(self.u_mat, cmap=plt.cm.RdBu)
        plt.xlabel('u_1'); plt.ylabel('u_2'); plt.title(f'u matrix, F={self.F}, k={self.k} after {self.n_times} iterations')
        plt.colorbar()
        plt.subplot(122) # plot v matrix
        plt.pcolor(self.v_mat, cmap=plt.cm.RdBu)
        plt.xlabel('v_1'); plt.ylabel('v_2'); plt.title(f'v matrix, F={self.F}, k={self.k} after {self.n_times} iterations')
        plt.colorbar()
        if save_figures:
            plt.savefig(filename_uv, dpi=200)

    def animation(self, save_animation=True):
        """Function to create animation of evolution u matrix using the saved frames.

        Argumetns:
        -----------------
            save_animation: bool
                - If true, save animation to gif file
        """
        fig = plab.figure()
        plt.rcParams['figure.figsize'] = (10, 10)
        plab.pcolormesh(self.u_mat, cmap=plab.cm.RdBu)

        def animate(i):
            """Plotting function for animation"""
            if i < self.save_u_mat.shape[0]:
                plab.pcolormesh(np.squeeze(self.save_u_mat[i, :, :]), cmap=plab.cm.RdBu)

        anim = animation.FuncAnimation(fig, animate, frames=range(self.save_u_mat.shape[0]), blit=False)
        writer = PillowWriter(fps=20)
        filename_an = f'u_matrix_F={self.F}_k={self.k}.gif'
        plab.xlabel('u_1')
        plab.ylabel('u_2')
        plab.title(f'u matrix, F={self.F}, k={self.k} after {self.n_times} iterations.')
        if save_animation:
            anim.save(filename_an, writer=writer)

    def n_outputs(self):
        """Returns number of outputs."""
        return (self.n_x * self.n_y)

    def n_parameters(self):
        """Returns number of parameters for inference (F and K)"""
        return 2
    
    def simulate(self, parameters, times):
        value = self.solve(parameters)
        return value
