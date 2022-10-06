import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', size=14)

from exactpack.solvers.noh2 import Noh2, PlanarNoh2, CylindricalNoh2, SphericalNoh2
from exactpack.solvers.noh2.noh2_cog import Noh2Cog

#####################################################################
rmax = 1.2  # analysis domain
r = np.linspace(0.0, rmax, 10)
t1 = 0.6

#####################################################################
fig = plt.figure(figsize=(10, 7))
plt.subplot(211)
solver = Noh2()
soln1 = solver(r,t1)

plt.title('ExactPack solver class Noh2: direct and cog1')
soln1.plot('density')
soln1.plot('pressure')
soln1.plot('velocity',scale=10)
soln1.plot('specific_internal_energy')
plt.ylim(-50, 100)
plt.xlim(0, rmax)
plt.grid(True)
plt.legend(loc=0)

#####################################################################
plt.subplot(212)
solver = Noh2Cog()
soln2 = solver(r,t1)

soln2.plot('density')
soln2.plot('pressure')
soln2.plot('velocity',scale=10)
soln2.plot('specific_internal_energy')
plt.ylim(-50, 100)
plt.xlim(0, rmax)
plt.grid(True)
plt.legend(loc=0)
plt.tight_layout()
plt.show()

#####################################################################
solver = SphericalNoh2()
soln = solver(r,t1)

fig = plt.figure(figsize=(10, 7))
plt.title('Spherical Noh2')
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity',scale=10)
soln.plot('specific_internal_energy')
plt.ylim(-50, 80)
plt.xlim(0, rmax)
plt.grid(True)
plt.legend(loc=0)
plt.tight_layout()
plt.show()


#####################################################################
solver = CylindricalNoh2()
soln = solver(r,t1)

fig = plt.figure(figsize=(10, 7))
plt.title('Cylindrical Noh2')
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity',scale=10)
soln.plot('specific_internal_energy')
plt.ylim(-50, 20)
plt.xlim(0, rmax)
plt.grid(True)
plt.legend(loc=0)
plt.tight_layout()
plt.show()

#####################################################################
solver = PlanarNoh2()
soln = solver(r,t1)

fig = plt.figure(figsize=(10, 7))
plt.title('Planar Noh2')
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity',scale=10)
soln.plot('specific_internal_energy')
plt.ylim(-50, 10)
plt.xlim(0, rmax)
plt.grid(True)
plt.legend(loc=0)
plt.tight_layout()
plt.show()

plt.close()
