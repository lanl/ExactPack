#
#  Creates plots for the DSD solvers in ExactPack
#

import numpy as np
import matplotlib.pylab as plt

# import ExactPack solver and analysis tools
# from exactpack.solvers.dsd.ratestick import RateStick
from exactpack.solvers.dsd.cylexpansion import CylindricalExpansion
# from exactpack.solvers.dsd.explosivearc import ExplosiveArc

plt.rc('font', family='serif', size=12)

#####################################################################
# RateStick example
# Construct spatial grid and choose time

# solver object

# any other calculations for analysis

# plot exact solutions

#####################################################################
# CylindricalExpansion example
# Construct spatial grid and choose time
r = np.linspace(1, 4, 61)
theta = np.linspace(0.0, np.pi / 2.0, 181)

r2g, th2g = np.meshgrid(r, theta)
x2 = r2g * np.cos(th2g)
y2 = r2g * np.sin(th2g)
xy = np.vstack((x2.flatten(), y2.flatten())).T  # 2D grid

time = 0.0  # time does not matter, but needs to be set for the machinery

# solver object
soln = CylindricalExpansion()(xy, time)  # 2D using default parameters

# outer boundary of HE
obdyz = np.linspace(0.0, 4.0, 241)
obdyx = np.sqrt(16.0 - obdyz ** 2)

# inner boundary of HE
ibdyz = np.linspace(0.0, 1.0, 61)
ibdyx = np.sqrt(1.0 - ibdyz ** 2)

# boundary between HE regions
mbdyz = np.linspace(0.0, 2.0, 121)
mbdyx = np.sqrt(2.0 - mbdyz ** 2)

# calculate distance from det
r2 = np.sqrt(soln.position_x ** 2 + soln.position_y ** 2)

# plot exact solutions
fig = plt.figure()
fig.add_subplot(1, 2, 1, aspect=1.)
p1 = plt.plot(r2, soln.burntime, 'b.')
plt.xlabel('distance')
plt.ylabel('burn time')
plt.title('Burntime vs R')
plt.grid()

fig.add_subplot(1, 2, 2, aspect=1.)
p2 = plt.contour(soln.position_x.reshape(len(theta), len(r)),
                 soln.position_y.reshape(len(theta), len(r)),
                 soln.burntime.reshape(len(theta), len(r)))
plt.xlabel('x')
plt.ylabel('y')
plt.clabel(p2, inline=1, fmt='%1.1f')
plt.title('Burntime')
plt.plot(obdyx, obdyz, 'k-', lw=2)
plt.plot(ibdyx, ibdyz, 'k-', lw=2)
plt.plot(mbdyx, mbdyz, 'k--')

plt.tight_layout()
plt.show()

#####################################################################
# ExplosiveArc example
# Construct spatial grid and choose time

# solver object

# any other calculations for analysis

# plot exact solutions
