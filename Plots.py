import matplotlib.pyplot as plt
import matplotlib.pylab as plab
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

class Plot():
    """
    Defines an interface for plotting the Gray-Scott PDE system

    :param N: The dimension of the quadratic spatial matrix (NxN), i.e. the length of discrete spatial points along the one axis.
    """
    def __init__(self, N):
        self.N = N

    def plot2d(self, values_t, filename_u, filename_v):
        """
        Plots a 2d plot of the Gray-Scott PDE system given the values for u and v at a given time-point.

        :param values_t: a 1D array containing the values for u and v at a given time-point. If the quadratic spatial matrix is at the
        given time-point (NxN), then the length of the values_t should be 2*N*N, where the first N*N values correspond to u and the next
        N*N values correspond to v. In other words, values_t is a reshaped 1D array that contains all the values of two spatial matrices
        giving the concentration of u and v at a given time-point.
        :param filename_u: filename for outputting u
        :param filename_v: filename for outputting v
        :return: filename_u.png and filename_v.png
        """
        #The values corresponding to u are selected from the values_t
        values_t_u = values_t[0: self.N * self.N]

        #The values corresoinding to v are selected from the values_t
        values_t_v = values_t[self.N * self.N : len(values_t)]

        #Plotting u. Note that the 1D array (length N*N) is reshaped to a 2D array (NxN) for plotting
        plt.pcolor(values_t_u.reshape((self.N, self.N)), cmap=plt.cm.RdBu)

        #Saving the u plot
        plt.savefig(filename_u)

        # Plotting v. Note that the 1D array (length N*N) is reshaped to a 2D array (NxN) for plotting
        plt.pcolor(values_t_v.reshape((self.N, self.N)), cmap=plt.cm.RdBu)

        # Saving the v plot
        plt.savefig(filename_v)

    def animation(self, values, filename_u, filename_v):
        """
        Plots a 2d animation (gif) of the evolution of the PDE system

        :param values: a (t x n_t) 2D array where the values of the system n_t are given at the timepoints t. If the quadratic spatial matrix is at the
        given time-point (NxN), then the length of n_t should be 2*N*N, where the first N*N values correspond to u and the next
        N*N values correspond to v. In other words, values_t is a reshaped 1D array that contains all the values of two spatial matrices
        giving the concentration of u and v at a given time-point.
        :param filename_u: filename for outputting u
        :param filename_v: filename for outputting v
        :return: filename_u.gif and filename_v.gif
        """

        #Plotting u

        #First the values
        values_u = values[0][0: self.N * self.N]
        fig = plab.figure()
        plab.pcolormesh(values_u.reshape((self.N, self.N)), cmap=plab.cm.RdBu)

        def animate_u(i):
            values_u = values[i][0: self.N * self.N]
            plab.pcolormesh(values_u.reshape((self.N, self.N)), cmap=plab.cm.RdBu)

        anim_u = animation.FuncAnimation(fig, animate_u, frames=range(1, values.shape[0], 10), blit=False)
        writer = PillowWriter(fps=100)
        anim_u.save(filename_u, writer=writer)

        values_v = values[0][0: self.N * self.N]
        fig = plab.figure()
        plab.pcolormesh(values_v.reshape((self.N, self.N)), cmap=plab.cm.RdBu)

        def animate_v(i):
            values_v = values[i][0: self.N * self.N]
            plab.pcolormesh(values_v.reshape((self.N, self.N)), cmap=plab.cm.RdBu)

        anim_v = animation.FuncAnimation(fig, animate_v, frames=range(1, values.shape[0], 10), blit=False)
        writer = PillowWriter(fps=100)
        anim_v.save(filename_v, writer=writer)


