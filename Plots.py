import matplotlib.pyplot as plt
import matplotlib.pylab as plab
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

class Plot():
    def __init__(self, N):
        self.N = N

    def plot2d(self, values_t, filename_u, filename_v):
        values_t_u = values_t[0: self.N * self.N]
        values_t_v = values_t[self.N * self.N : len(values_t)]
        plt.pcolor(values_t_u.reshape((self.N, self.N)), cmap=plt.cm.RdBu)
        plt.savefig(filename_u)
        plt.pcolor(values_t_v.reshape((self.N, self.N)), cmap=plt.cm.RdBu)
        plt.savefig(filename_v)

    def animation(self, values, filename_u):
        values_u = values[0][0: self.N * self.N]

        fig = plab.figure()
        plab.pcolormesh(values_u.reshape((self.N, self.N)), cmap=plab.cm.RdBu)

        def animate(i):
            values_u = values[i][0: self.N * self.N]
            plab.pcolormesh(values_u.reshape((self.N, self.N)), cmap=plab.cm.RdBu)

        anim = animation.FuncAnimation(fig, animate, frames=range(1, values.shape[0]), blit=False)
        writer = PillowWriter(fps=100)
        anim.save(filename_u, writer=writer)
