#
#  Creates plots for the Kenamond solvers in ExactPack
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', size=14)

# import ExactPack solver and analysis tools
from exactpack.solvers.kenamond.kenamond1 import Kenamond1
from exactpack.solvers.kenamond.kenamond2 import Kenamond2
from exactpack.solvers.kenamond.kenamond3 import Kenamond3

#####################################################################
# Kenamond1 example
# construct spatial grid and choose time
x = np.linspace(0, 10, 41)
y = np.linspace(0, 10, 81)
z = np.linspace(0, 10, 41)
xy = np.array([[x0, y0] for x0 in x for y0 in y])  # 2D grid
xyz = np.array([[x0, y0, z0] for x0 in x for y0 in y for z0 in z])  # 3D grid

time = 0.0  # time does not matter, but needs to be set for the machinery

# solver object
soln2 = Kenamond1()(xy, time)  # 2D using default parameters
soln3 = Kenamond1(geometry=3, x_d=(0.0, 0.0, 0.0))(xyz, time)  # 3D prob

# calculate distance from det
r2 = np.sqrt(soln2.position_x ** 2 + soln2.position_y ** 2)
r3 = np.sqrt(soln3.position_x ** 2 + soln3.position_y ** 2 +
             soln3.position_z ** 2)

# plot exact solutions
fig = plt.figure(figsize=(10, 7))
plt.subplot(2, 2, 1)
plt.plot(r2, soln2.burntime, 'b.')
plt.xlabel('distance')
plt.ylabel('burn time')
plt.title('2D Burntime')
plt.axis('scaled')

plt.subplot(2, 2, 2)
plt.plot(r3, soln3.burntime, 'b.')
plt.xlabel('distance')
plt.ylabel('burn time')
plt.title('3D Burntime')
plt.axis('scaled')

plt.subplot(2, 2, 3)
p3 = plt.contour(soln2.position_x.reshape(len(x), len(y)),
                 soln2.position_y.reshape(len(x), len(y)),
                 soln2.burntime.reshape(len(x), len(y)))
plt.xlabel('x')
plt.ylabel('y')
plt.clabel(p3, inline=1, fmt='%1.1f')
plt.title('2D Burntime')
plt.axis('scaled')

plt.subplot(2, 2, 4)
p4 = plt.contour(soln3.position_x[::len(z)].reshape(len(x), len(y)),
                 soln3.position_y[::len(z)].reshape(len(x), len(y)),
                 soln3.burntime[::len(z)].reshape(len(x), len(y)))
plt.xlabel('x')
plt.ylabel('y')
plt.clabel(p4, inline=1, fmt='%1.1f')
plt.title('3D Burntime, z=0 plane')
plt.axis('scaled')

plt.tight_layout()
plt.show()

#####################################################################
# Kenamond2 example
# construct spatial grid and choose time
r = np.linspace(0, 10, 41)
theta2 = np.linspace(-np.pi / 2.0, np.pi / 2.0, 65)
theta3 = np.linspace(0.0, np.pi / 2.0, 33)
phi = np.linspace(0, np.pi, 65)

r2g, th2g = np.meshgrid(r, theta2)
x2 = r2g * np.cos(th2g)
y2 = r2g * np.sin(th2g)
xy = np.vstack((x2.flatten(), y2.flatten())).T  # 2D grid

rtp = np.array([[r0, t0, p0] for r0 in r for t0 in theta3 for p0 in phi])
x3 = rtp[:, 0] * np.cos(rtp[:, 1]) * np.sin(rtp[:, 2])
y3 = rtp[:, 0] * np.sin(rtp[:, 1]) * np.sin(rtp[:, 2])
z3 = rtp[:, 0] * np.cos(rtp[:, 2])
xyz = np.vstack((x3, y3, z3)).T  # 3D grid

time = 0.0  # time does not matter, but needs to be set for the machinery

# solver object
soln2 = Kenamond2()(xy, time)  # 2D using default parameters
soln3 = Kenamond2(geometry=3)(xyz, time)  # 3D problem

# locate 3D points on theta=0 (y=0) and theta=pi/2 (x=0) planes
xt0 = []
zt0 = []
bt0 = []
ytp = []
ztp = []
btp = []
for i in range(len(r)):
    t0start = i * len(theta3) * len(phi)
    grab = soln3.position_x[t0start:t0start + len(phi)]
    xt0 = np.append(xt0, grab)
    grab = soln3.position_z[t0start:t0start + len(phi)]
    zt0 = np.append(zt0, grab)
    grab = soln3.burntime[t0start:t0start + len(phi)]
    bt0 = np.append(bt0, grab)

    tpstart = len(phi) * ((i + 1) * len(theta3) - 1)
    grab = soln3.position_y[tpstart:tpstart + len(phi)]
    ytp = np.append(ytp, grab)
    grab = soln3.position_z[tpstart:tpstart + len(phi)]
    ztp = np.append(ztp, grab)
    grab = soln3.burntime[tpstart:tpstart + len(phi)]
    btp = np.append(btp, grab)

# outer boundary of HE
obdyz = np.linspace(-10.0, 10.0, 401)
obdyx = np.sqrt(100.0 - obdyz ** 2)

# boundary between HE regions
ibdyz = np.linspace(-3.0, 3.0, 121)
ibdyx = np.sqrt(9.0 - ibdyz ** 2)

# plot exact solutions
fig = plt.figure(figsize=(10, 7))
fig.add_subplot(2, 2, 1, aspect=1.)
p1 = plt.contour(soln2.position_x.reshape(len(theta2), len(r)),
                 soln2.position_y.reshape(len(theta2), len(r)),
                 soln2.burntime.reshape(len(theta2), len(r)), 8)
plt.xlabel('x')
plt.ylabel('y')
plt.clabel(p1, inline=1, fmt='%1.1f')
plt.title('2D Burntime')
plt.plot(obdyx, obdyz, 'k-', lw=2)
plt.plot(ibdyx, ibdyz, 'k--')

fig.add_subplot(2, 2, 2, aspect=1.)
p2 = plt.contour(xt0.reshape(len(r), len(phi)),
                 zt0.reshape(len(r), len(phi)),
                 bt0.reshape(len(r), len(phi)), 8)
plt.xlabel('x')
plt.ylabel('z')
plt.clabel(p2, inline=1, fmt='%1.1f')
plt.title('3D Burntime, theta=0 plane')
plt.plot(obdyx, obdyz, 'k-', lw=2)
plt.plot(ibdyx, ibdyz, 'k--')

fig.add_subplot(2, 2, 3, aspect=1.)
p3 = plt.contour(ytp.reshape(len(r), len(phi)),
                 ztp.reshape(len(r), len(phi)),
                 btp.reshape(len(r), len(phi)), 8)
plt.xlabel('y')
plt.ylabel('z')
plt.clabel(p3, inline=1, fmt='%1.1f')
plt.title('3D Burntime, theta=pi/2 plane')
plt.plot(obdyx, obdyz, 'k-', lw=2)
plt.plot(ibdyx, ibdyz, 'k--')

fig.add_subplot(2, 2, 4, aspect=1.)
start = int((len(phi) + 1) / 2 - 1)

p4 = plt.contour(soln3.position_x[start::len(phi)].reshape(len(r),
                                                           len(theta3)),
                 soln3.position_y[start::len(phi)].reshape(len(r),
                                                           len(theta3)),
                 soln3.burntime[start::len(phi)].reshape(len(r),
                                                         len(theta3)),
                 8)
plt.xlabel('x')
plt.ylabel('y')
plt.clabel(p4, inline=1, fmt='%1.1f')
plt.title('3D Burntime, z=0 plane')
plt.plot(obdyx[200:], obdyz[200:], 'k-', lw=2)
plt.plot(ibdyx[60:], ibdyz[60:], 'k--')
plt.grid(True)
plt.tight_layout()
plt.show()

#####################################################################
# Kenamond3 example
# construct spatial grid and choose time
r = np.linspace(3.0001, 10, 29)
theta2 = np.linspace(-np.pi / 2.0, np.pi / 2.0, 65)
theta3 = np.linspace(0.0, np.pi / 2.0, 33)
phi = np.linspace(0, np.pi, 65)

r2g, th2g = np.meshgrid(r, theta2)
x2 = r2g * np.cos(th2g)
y2 = r2g * np.sin(th2g)
xy = np.vstack((x2.flatten(), y2.flatten())).T  # 2D grid

rtp = np.array([[r0, t0, p0] for r0 in r for t0 in theta3 for p0 in phi])
x3 = rtp[:, 0] * np.cos(rtp[:, 1]) * np.sin(rtp[:, 2])
y3 = rtp[:, 0] * np.sin(rtp[:, 1]) * np.sin(rtp[:, 2])
z3 = rtp[:, 0] * np.cos(rtp[:, 2])
xyz = np.vstack((x3, y3, z3)).T  # 3D grid

time = 0.0  # time does not matter, but needs to be set for the machinery

# solver object
soln2 = Kenamond3()(xy, time)  # 2D using default parameters
soln3 = Kenamond3(geometry=3, x_d=[0.0, 0.0, 5.0])(xyz, time)  # 3D prob

# locate 3D points on theta=0 (y=0) and theta=pi/2 (x=0) planes
xt0 = []
zt0 = []
bt0 = []
ytp = []
ztp = []
btp = []
for i in range(len(r)):
    t0start = i * len(theta3) * len(phi)
    grab = soln3.position_x[t0start:t0start + len(phi)]
    xt0 = np.append(xt0, grab)
    grab = soln3.position_z[t0start:t0start + len(phi)]
    zt0 = np.append(zt0, grab)
    grab = soln3.burntime[t0start:t0start + len(phi)]
    bt0 = np.append(bt0, grab)

    tpstart = len(phi) * ((i + 1) * len(theta3) - 1)
    grab = soln3.position_y[tpstart:tpstart + len(phi)]
    ytp = np.append(ytp, grab)
    grab = soln3.position_z[tpstart:tpstart + len(phi)]
    ztp = np.append(ztp, grab)
    grab = soln3.burntime[tpstart:tpstart + len(phi)]
    btp = np.append(btp, grab)

# outer boundary of HE
obdyz = np.linspace(-10.0, 10.0, 401)
obdyx = np.sqrt(100.0 - obdyz ** 2)

# inner boundary of HE
ibdyz = np.linspace(-3.0, 3.0, 121)
ibdyx = np.sqrt(9.0 - ibdyz ** 2)

# plot exact solutions
fig = plt.figure(figsize=(10, 7))
fig.add_subplot(2, 2, 1, aspect=1.)
p1 = plt.contour(soln2.position_x.reshape(len(theta2), len(r)),
                 soln2.position_y.reshape(len(theta2), len(r)),
                 soln2.burntime.reshape(len(theta2), len(r)), 8)
plt.xlabel('x')
plt.ylabel('y')
plt.clabel(p1, inline=1, fmt='%1.1f')
plt.title('2D Burntime')
plt.plot(obdyx, obdyz, 'k-', lw=2)
plt.plot(ibdyx, ibdyz, 'k-', lw=2)

fig.add_subplot(2, 2, 2, aspect=1.)
p2 = plt.contour(xt0.reshape(len(r), len(phi)),
                 zt0.reshape(len(r), len(phi)),
                 bt0.reshape(len(r), len(phi)), 8)
plt.xlabel('x')
plt.ylabel('z')
plt.clabel(p2, inline=1, fmt='%1.1f')
plt.title('3D Burntime, theta=0 plane')
plt.plot(obdyx, obdyz, 'k-', lw=2)
plt.plot(ibdyx, ibdyz, 'k-', lw=2)

fig.add_subplot(2, 2, 3, aspect=1.)
p3 = plt.contour(ytp.reshape(len(r), len(phi)),
                 ztp.reshape(len(r), len(phi)),
                 btp.reshape(len(r), len(phi)), 8)
plt.xlabel('y')
plt.ylabel('z')
plt.clabel(p3, inline=1, fmt='%1.1f')
plt.title('3D Burntime, theta=pi/2 plane')
plt.plot(obdyx, obdyz, 'k-', lw=2)
plt.plot(ibdyx, ibdyz, 'k-', lw=2)

plt.subplot(2, 2, 4)
start = int((len(phi) + 1) / 2 - 1)
p4 = plt.contour(soln3.position_x[start::len(phi)].reshape(len(r),
                                                           len(theta3)),
                 soln3.position_y[start::len(phi)].reshape(len(r),
                                                           len(theta3)),
                 soln3.burntime[start::len(phi)].reshape(len(r),
                                                         len(theta3)),
                 8)
plt.xlabel('x')
plt.ylabel('y')
plt.clabel(p4, inline=1, fmt='%1.1f')
plt.title('3D Burntime, z=0 plane')
plt.plot(obdyx[200:], obdyz[200:], 'k-', lw=2)
plt.plot(ibdyx[60:], ibdyz[60:], 'k-', lw=2)
plt.axis('scaled')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.close()
