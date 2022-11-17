import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', size=14)


# import ExactPack solvers
from exactpack.solvers.cog import Cog8
from exactpack.solvers.cog import Cog3

# construct sptial grid and choose time
rmax = 2.
r = np.linspace(0.0, rmax, 1000)
t = 1.0

#####################################################################
# solver object
solver = Cog8()
soln = solver(r,t)

# plot exact solution
fig = plt.figure(figsize=(10, 7))
soln.plot('density')
soln.plot('velocity')
soln.plot('temperature')
plt.xlim(0.0,rmax)
plt.ylim(0.0,5.0)
plt.title(r'ExactPack solver class Cog8')
plt.legend(loc=0)
plt.grid(True)
plt.tight_layout()
plt.show()


#####################################################################
# solver object
solver = Cog3(geometry=3, rho0=1.8, b=1.2, v=0.5, Gamma=40.)
soln = solver(r,t)

# plot exact solution
fig = plt.figure(figsize=(10, 7))
soln.plot('density')
soln.plot('pressure', scale=0.01)
soln.plot('velocity')
soln.plot('temperature')
plt.xlim(0.0,rmax)
plt.ylim(-5.0,10.0)
plt.title(r'ExactPack solver class Cog3')
plt.legend(loc=0)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.close()