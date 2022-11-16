import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', size=14)

# import ExactPack solver and analysis tools
from exactpack.solvers.mader.timmes import Mader

#####################################################################
rmax = 5.0
r = np.linspace(0.0, rmax, 1000)
t = 5.e-6

#####################################################################
solver = Mader()
soln = solver(r,t)

fig = plt.figure(figsize=(10, 7))
soln.plot('density')
soln.plot('pressure', scale=1.e-11)
soln.plot('velocity',scale=1.e-5)  # looks wrong
plt.xlim(0.0, rmax)
plt.title('ExactPack solver class Mader')
plt.ylim(0, 3)
plt.legend(loc=1)
plt.grid(True)
plt.show()
plt.tight_layout()
plt.close()