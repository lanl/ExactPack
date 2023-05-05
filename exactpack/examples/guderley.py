import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', size=14)

# import ExactPack solver and analysis tools
from exactpack.solvers.guderley import Guderley

#####################################################################
rmax = 3.0
r = np.linspace(0.0, rmax, 1000)

solver = Guderley(gamma=3.0)

#####################################################################
t = -1.
soln = solver(r,t)

fig = plt.figure(figsize=(10, 7))
soln.plot('density',label='density')
soln.plot('pressure')
soln.plot('velocity')
soln.plot('specific_internal_energy')
plt.xlim(0.0,rmax)
plt.title(r'ExactPack solver class Guderley: $t=-1\,{\rm s}$')
plt.ylim(-0.5,3.0)
plt.legend(loc=0)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.close()
