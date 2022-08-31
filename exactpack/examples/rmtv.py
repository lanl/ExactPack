import numpy as np
import matplotlib.pylab as plt

# import ExactPack solver and analysis tools
from exactpack.solvers.rmtv import Rmtv

#####################################################################
rmax = 1
r = np.linspace(0.001, rmax, 1000)

solver = Rmtv()

#####################################################################
# Note that currently the value of t is not used in the Rmtv solver. 
# The heat front position is set by solver.rf
t = 1.e-9
soln1 = solver(r, t)

soln1.plot('density',label='Rho')
soln1.plot('temperature',label='T')
plt.xlim(0.0,rmax)
plt.title('ExactPack solver class Rmtv')
# plt.ylim(0,1000)
plt.legend(loc=0)
plt.grid(True)
plt.show()
