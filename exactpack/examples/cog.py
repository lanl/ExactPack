import numpy as np
import matplotlib.pylab as plt
from matplotlib import rc, rcParams
#rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})

# import ExactPack solver and analysis tools
from exactpack.solvers.cog import Cog8
from exactpack.solvers.cog import Cog3
from exactpack.solvers.cog.cog8_timmes import Cog8 as Cog8Timmes

# construct sptial grid and choose time
rmax = 2.
r = np.linspace(0.0, rmax, 1000)
t = 1.0

#####################################################################
# solver object
solver = Cog8()
soln = solver(r,t)

# plot exact solution
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity')
soln.plot('temperature')
plt.xlim(0.0,rmax)
plt.ylim(0.0,5.0)
plt.title(r'ExactPack solver class Cog8')
plt.legend(loc=0)
plt.grid(True)
plt.show()

#####################################################################
# solver object
solver = Cog8Timmes()
soln = solver(r,t)

# plot exact solution
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity')
soln.plot('temperature')
plt.xlim(0.0,rmax)
plt.ylim(0.0,5.0)
plt.title(r'ExactPack solver class Cog8Timmes')
plt.legend(loc=0)
plt.grid(True)
plt.show()


#####################################################################
# solver object
solver = Cog3(geometry=3, rho0=1.8, b=1.2, v=0.5, Gamma=40.)
soln = solver(r,t)

# plot exact solution
soln.plot('density')
soln.plot('pressure', scale=0.01)
soln.plot('velocity')
soln.plot('temperature')
plt.xlim(0.0,rmax)
plt.ylim(-5.0,10.0)
plt.title(r'ExactPack solver class Cog3')
plt.legend(loc=0)
plt.grid(True)
plt.show()

