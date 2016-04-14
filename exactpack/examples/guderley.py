import numpy as np
import matplotlib.pylab as plt
from matplotlib import rc, rcParams
#rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})

# import ExactPack solver and analysis tools
from exactpack.solvers.guderley import Guderley

#####################################################################
rmax = 3.0
r = np.linspace(0.0, rmax, 1000)

solver = Guderley(gamma=3.0)

#####################################################################
t = -1.
soln = solver(r,t)

soln.plot('density',label='density')
soln.plot('pressure')
soln.plot('velocity')
soln.plot('sie')
plt.xlim(0.0,rmax)
plt.title(r'ExactPack solver class Guderley: $t=-1\,{\rm s}$')
plt.ylim(-0.5,3.0)
plt.legend(loc=0)
plt.grid(True)
plt.show()
