from GSPDE import GrayScott

N = 256
rdSolver = GrayScott(N)
L = rdSolver.laplacian()
Du, Dv, F, K = 0.14, 0.06, 0.035, 0.065
Nt = 32000

rdSolver.initialise()
rdSolver.integrate(Nt, Du, Dv, F, K, L)
rdSolver.configPlot()