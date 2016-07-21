#
#  Creates plots for the DSD solvers in ExactPack
#

import numpy as np
import matplotlib.pylab as plt

# import ExactPack solver and analysis tools
from exactpack.solvers.dsd.ratestick import RateStick
from exactpack.solvers.dsd.cylexpansion import CylindricalExpansion
# from exactpack.solvers.dsd.explosivearc import ExplosiveArc

plt.rc('font', family='serif', size=12)

#####################################################################
# RateStick examples
# Case 1
# Initial condition 1, default Bdzil planar case (with slight change)
# Construct spatial grid and choose time
nodesx = 11                          # number of nodes in x-direction
nodesy = 51                          # number of nodes in y-direction
x1 = np.linspace(0.0, 1.0, nodesx)   # spacing must be greater than solver dx
y1 = np.linspace(0.0, 5.0, nodesy)

x12, y12 = np.meshgrid(x1, y1)
xy1 = np.vstack((x12.flatten(), y12.flatten())).T  # 2D grid points

time = 0.0  # time does not matter, but needs to be set for the machinery

# solver object
soln1 = RateStick(xnodes=nodesx, ynodes=nodesy)(xy1, time)

# Case 2
# Initial condition 1, Bdzil cylindrical case (with slight change)
# Construct spatial grid and choose time
nodesx = 11                          # number of nodes in x-direction
nodesy = 51                          # number of nodes in y-direction
x2 = np.linspace(0.0, 1.0, nodesx)   # spacing must be greater than solver dx
y2 = np.linspace(0.0, 5.0, nodesy)

x22, y22 = np.meshgrid(x2, y2)
xy2 = np.vstack((x22.flatten(), y22.flatten())).T  # 2D grid points

time = 0.0  # time does not matter, but needs to be set for the machinery

# solver object
soln2 = RateStick(geometry=2, xnodes=nodesx, ynodes=nodesy)(xy2, time)

# plot exact solutions on requested grid
tcontours = [0.25, 0.5, 1.0, 2.0, 5.0]
fig = plt.figure()
fig.add_subplot(1, 2, 1)
p1 = plt.contour(soln1.position_x.reshape(len(y1), len(x1)),
                 soln1.position_y.reshape(len(y1), len(x1)),
                 soln1.burntime.reshape(len(y1), len(x1)),
                 tcontours)
plt.xlabel('x')
plt.ylabel('y')
plt.clabel(p1, inline=1, fmt='%1.1f')
plt.title('IC=1, Planar Burntime')
plt.grid()

fig.add_subplot(1, 2, 2)
p2 = plt.contour(soln2.position_x.reshape(len(y2), len(x2)),
                 soln2.position_y.reshape(len(y2), len(x2)),
                 soln2.burntime.reshape(len(y2), len(x2)),
                 tcontours)
plt.xlabel('x')
plt.ylabel('y')
plt.clabel(p2, inline=1, fmt='%1.1f')
plt.title('IC=1, Cylindrical Burntime')
plt.grid()

plt.tight_layout()
plt.show()

# Case 3
# Initial condition 2, planar case
# Construct spatial grid and choose time
nodesx = 10                          # number of nodes in x-direction
nodesy = 81                          # number of nodes in y-direction
x3 = np.linspace(0.0, 0.9, nodesx)   # spacing must be greater than solver dx
y3 = np.linspace(0.0, 8.0, nodesy)

x32, y32 = np.meshgrid(x3, y3)
xy3 = np.vstack((x32.flatten(), y32.flatten())).T  # 2D grid points

time = 0.0  # time does not matter, but needs to be set for the machinery

# solver object
soln3 = RateStick(IC=2, xnodes=nodesx, ynodes=nodesy, R=0.9, omega_c=0.9,
                  D_CJ=0.8, alpha=0.05, t_f=11.0)(xy3, time)

# Case 4
# Initial condition 2, cylindrical case
# Construct spatial grid and choose time
nodesx = 10                          # number of nodes in x-direction
nodesy = 81                          # number of nodes in y-direction
x4 = np.linspace(0.0, 0.9, nodesx)   # spacing must be greater than solver dx
y4 = np.linspace(0.0, 8.0, nodesy)

x42, y42 = np.meshgrid(x4, y4)
xy4 = np.vstack((x42.flatten(), y42.flatten())).T  # 2D grid points

time = 0.0  # time does not matter, but needs to be set for the machinery

# solver object
soln4 = RateStick(geometry=2, IC=2, xnodes=nodesx, ynodes=nodesy, R=0.9,
                  omega_c=0.9, D_CJ=0.8, alpha=0.05, t_f=11.0)(xy4, time)

# plot exact solutions on requested grid
tcontours = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
fig = plt.figure()
fig.add_subplot(1, 2, 1)
p3 = plt.contour(soln3.position_x.reshape(len(y3), len(x3)),
                 soln3.position_y.reshape(len(y3), len(x3)),
                 soln3.burntime.reshape(len(y3), len(x3)),
                 tcontours)
plt.xlabel('x')
plt.ylabel('y')
plt.clabel(p3, inline=1, fmt='%1.1f')
plt.title('IC=2, Planar Burntime')
plt.grid()

fig.add_subplot(1, 2, 2)
p4 = plt.contour(soln4.position_x.reshape(len(y4), len(x4)),
                 soln4.position_y.reshape(len(y4), len(x4)),
                 soln4.burntime.reshape(len(y4), len(x4)),
                 tcontours)
plt.xlabel('x')
plt.ylabel('y')
plt.clabel(p4, inline=1, fmt='%1.1f')
plt.title('IC=2, Cylindrical Burntime')
plt.grid()

plt.tight_layout()
plt.show()

# Case 5
# Initial condition 3, planar case
# Construct spatial grid and choose time
nodesx = 11                         # number of nodes in x-direction
nodesy = 61                         # number of nodes in y-direction
x5 = np.linspace(0.0, 1.0, nodesx)  # spacing must be greater than solver dx
y5 = np.linspace(0.0, 6.0, nodesy)

x52, y52 = np.meshgrid(x5, y5)
xy5 = np.vstack((x52.flatten(), y52.flatten())).T  # 2D grid points

time = 0.0  # time does not matter, but needs to be set for the machinery

# solver object
soln5 = RateStick(IC=3, xnodes=nodesx, ynodes=nodesy, t_f=7.0)(xy5, time)

# Case 6
# Initial condition 3, cylindrical case
# Construct spatial grid and choose time
nodesx = 11                         # number of nodes in x-direction
nodesy = 61                         # number of nodes in y-direction
x6 = np.linspace(0.0, 1.0, nodesx)  # spacing must be greater than solver dx
y6 = np.linspace(0.0, 6.0, nodesy)

x62, y62 = np.meshgrid(x6, y6)
xy6 = np.vstack((x62.flatten(), y62.flatten())).T  # 2D grid points

time = 0.0  # time does not matter, but needs to be set for the machinery

# solver object
soln6 = RateStick(geometry=2, IC=3, xnodes=nodesx, ynodes=nodesy,
                  t_f=7.0)(xy6, time)

# plot exact solutions on requested grid
tcontours = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
fig = plt.figure()
fig.add_subplot(1, 2, 1)
p5 = plt.contour(soln5.position_x.reshape(len(y5), len(x5)),
                 soln5.position_y.reshape(len(y5), len(x5)),
                 soln5.burntime.reshape(len(y5), len(x5)),
                 tcontours)
plt.xlabel('x')
plt.ylabel('y')
plt.clabel(p5, inline=1, fmt='%1.1f')
plt.title('IC=3, Planar Burntime')
plt.grid()

fig.add_subplot(1, 2, 2)
p6 = plt.contour(soln6.position_x.reshape(len(y6), len(x6)),
                 soln6.position_y.reshape(len(y6), len(x6)),
                 soln6.burntime.reshape(len(y6), len(x6)),
                 tcontours)
plt.xlabel('x')
plt.ylabel('y')
plt.clabel(p6, inline=1, fmt='%1.1f')
plt.title('IC=3, Cylindrical Burntime')
plt.grid()

plt.tight_layout()
plt.show()

#####################################################################
# CylindricalExpansion example
# Construct spatial grid and choose time
r = np.linspace(1.0, 4.0, 61)
theta = np.linspace(0.0, np.pi / 2.0, 181)

r2g, th2g = np.meshgrid(r, theta)
x2 = r2g * np.cos(th2g)
y2 = r2g * np.sin(th2g)
xy = np.vstack((x2.flatten(), y2.flatten())).T  # 2D grid points

time = 0.0  # time does not matter, but needs to be set for the machinery

# solver object
soln = CylindricalExpansion()(xy, time)  # 2D using default parameters

# outer boundary of HE
obdyz = np.linspace(0.0, 4.0, 241)
obdyx = np.sqrt(16.0 - obdyz ** 2.0)

# inner boundary of HE
ibdyz = np.linspace(0.0, 1.0, 61)
ibdyx = np.sqrt(1.0 - ibdyz ** 2.0)

# boundary between HE regions
mbdyz = np.linspace(0.0, 2.0, 121)
mbdyx = np.sqrt(4.0 - mbdyz ** 2.0)

# calculate distance from det
r2 = np.sqrt(soln.position_x ** 2.0 + soln.position_y ** 2.0)

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
