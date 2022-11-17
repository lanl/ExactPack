import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', size=14)

# import ExactPack solver and analysis tools
from exactpack.solvers.noh.noh1 import Noh
from exactpack.solvers.noh import SphericalNoh, CylindricalNoh, PlanarNoh

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
fig = plt.figure(figsize=(10, 7))
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity',scale=10)
soln.plot('sie',scale=10)
plt.xlim(0.0,rmax)
plt.title('ExactPack solver class Noh')
plt.ylim(-15,70)
plt.legend(loc=1)
plt.grid(True)
plt.tight_layout()
plt.show()

#####################################################################
# solver object
solver = SphericalNoh()
soln = solver(r,t)
# plot exact solution
fig = plt.figure(figsize=(10, 7))
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity',scale=10)
soln.plot('sie',scale=10)
plt.xlim(0.0,rmax)
plt.title('ExactPack solver class SphericalNoh')
plt.ylim(-15,70)
plt.legend(loc=1)
plt.grid(True)
plt.tight_layout()
plt.show()

#####################################################################
# solver object
solver = CylindricalNoh()
soln = solver(r,t)
# plot exact solution
fig = plt.figure(figsize=(10, 7))
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity')
soln.plot('sie')
plt.xlim(0.0,rmax)
plt.title('ExactPack solver class CylindricalNoh')
plt.ylim(-2,18)
plt.legend(loc=1)
plt.grid(True)
plt.tight_layout()
plt.show()


#####################################################################
# solver object
solver = PlanarNoh()
soln = solver(r,t)
# plot exact solution
fig = plt.figure(figsize=(10, 7))
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity')
soln.plot('sie')
plt.xlim(0.0,rmax)
plt.title('ExactPack solver class PlanarNoh')
plt.ylim(-1.4,5)
plt.legend(loc=1)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.close()
