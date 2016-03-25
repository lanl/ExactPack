import numpy as np
import matplotlib.pylab as plt
from matplotlib import rc, rcParams
#rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})

# import ExactPack solver and analysis tools
from exactpack.solvers.noh.noh1 import Noh
from exactpack.solvers.noh.timmes import Noh as NohTimmes
from exactpack.solvers.noh import SphericalNoh, CylindricalNoh, PlanarNoh
from exactpack.analysis import CodeVerificationStudy
from exactpack.analysis.readers.rage import RageDump, rage_dump1D, rage_dump2D, sqrt

# construct spatial grid and choose time
# rmax = 1.2 # computational domain
rmax = 0.5   # analysis domain
r = np.linspace(0.0, rmax, 1000)
t = 0.6

#####################################################################
# solver object
solver = Noh()
soln = solver(r,t)
# plot exact solution
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity',scale=10)
soln.plot('sie',scale=10)
plt.xlim(0.0,rmax)
plt.title('ExactPack solver class Noh')
plt.ylim(-15,70)
plt.legend(loc=1)
plt.grid(True)
plt.show()

#####################################################################
# solver object
solver = NohTimmes(geometry=3, gamma=5.0/3.0)
soln = solver(r,t)
# plot exact solution
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity',scale=10)
soln.plot('sie',scale=10)
plt.xlim(0.0,rmax)
plt.title('ExactPack solver class Noh for Timmes')
plt.ylim(-15,70)
plt.legend(loc=1)
plt.grid(True)
plt.show()

#####################################################################
# solver object
solver = SphericalNoh()
soln = solver(r,t)
# plot exact solution
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity',scale=10)
soln.plot('sie',scale=10)
plt.xlim(0.0,rmax)
plt.title('ExactPack solver class SphericalNoh')
plt.ylim(-15,70)
plt.legend(loc=1)
plt.grid(True)
plt.show()

#####################################################################
# solver object
solver = CylindricalNoh()
soln = solver(r,t)
# plot exact solution
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity')
soln.plot('sie')
plt.xlim(0.0,rmax)
plt.title('ExactPack solver class CylindricalNoh')
plt.ylim(-2,18)
plt.legend(loc=1)
plt.grid(True)
plt.show()


#####################################################################
# solver object
solver = PlanarNoh()
soln = solver(r,t)
# plot exact solution
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity')
soln.plot('sie')
plt.xlim(0.0,rmax)
plt.title('ExactPack solver class PlanarNoh')
plt.ylim(-1.4,5)
plt.legend(loc=1)
plt.grid(True)
plt.show()




