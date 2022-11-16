import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', size=14)

from exactpack.solvers.heat import Hutchens1
from exactpack.solvers.heat import Hutchens2


# Hutchens
##########################################################
b = 1.0
r = np.linspace(0.0, b, 1000)
t = 0.1

solver = Hutchens1()
soln = solver(r, t)
fig = plt.figure(figsize=(10, 7))
soln.plot('temperature')
plt.xlim(0.0, b)
plt.title('Hutchens1: Heat Conduction in a Sphere')
plt.ylim(0, 5)
plt.legend(loc=0)
plt.grid(True)
plt.tight_layout()
plt.show()

##########################################################
b = 1.0
L = 2.0
rmax = 10
zmax = 10
r0 = np.linspace(0.0, b, rmax)
z0 = np.linspace(0.0, L, zmax)
r, z = np.meshgrid(r0, z0)
rzlist = (r, z)
t = 0  # dummy argument

solver = Hutchens2()
soln = solver(rzlist, t)
fig = plt.figure(figsize=(10, 7))
p1 = plt.contour(soln.position_r,
                 soln.position_z,
                 soln.temperature, 8)
plt.xlabel('r')
plt.ylabel('z')
plt.title('Hutchens2: Steady State Heat Conduction in a Cylinder')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.close()
