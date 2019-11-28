import matplotlib.pyplot as plt

class Plot():
    def __init__(self, N):
        self.N = N

    def plot2d(self, values_t):
        values_t_u = values_t[0: self.N * self.N]
        values_t_v = values_t[self.N * self.N : len(values)]
        plt.pcolor(values_t_u.reshape((self.N, self.N)), cmap=plt.cm.RdBu)
        plt.savefig("GSPDE_u.png")
        plt.pcolor(values_t_v.reshape((self.N, self.N)), cmap=plt.cm.RdBu)
        plt.savefig("GSPDE_v.png")