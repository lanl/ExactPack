import numpy as np
import matplotlib.pylab as plt

from exactpack.solvers.heat.cylindrical_sandwich import CylindricalSandwich

# CylindricalSandwich
##########################################################
a = 0.25
b = 0.85
t = 0.01
rn = 50
thetan = 50
r0 = np.linspace(a, b, rn)
theta0 = np.linspace(0, np.pi/2, thetan)
r, theta = np.meshgrid(r0, theta0)
rtheta_list = (r, theta)

solver = CylindricalSandwich()  # NonHomogeneousOnly = True
soln = solver(rtheta_list, t)

# contour plots
contourV = np.linspace(0.001, 1.5, 200)
pt1 = plt.contour(soln.angle_theta, soln.position_r, soln.temperature, contourV)
manual_locations = [(0.25, 0.55), (0.5, 0.55), (1.0, 0.55), (1.4, 0.55)]
plt.clabel(pt1, inline=1, fontsize=10, manual=manual_locations)

plt.title(r'Nonhomogenous solution $T(x,y,t)$ at $t=0.01$')
plt.ylim(a, b)
plt.xlim(0, np.pi/2)
plt.ylabel(r'$T$', fontsize=20)
plt.xlabel(r'$\theta$', fontsize=20)
plt.grid(True)
plt.savefig('cylindrical_sandwich.pdf')
plt.show()

# line outs
pt1 = plt.plot(soln.angle_theta[:, 2], soln.temperature[:, 2])
pt2 = plt.plot(soln.angle_theta[:, 10], soln.temperature[:, 10])
#
plt.title(r'Nonhomogenous solution $T(x,y,t)$ at $t=0.01$')
plt.ylim(a, b)
plt.xlim(0, np.pi/2)
plt.ylabel(r'$T$', fontsize=20)
plt.xlabel(r'$\theta$', fontsize=20)
plt.grid(True)
plt.savefig('cylindrical_sandwich.pdf')
plt.show()

# CylindricalSandwich static nonhomogeneous
##########################################################
t = 0.01
a = 0.25
b = 0.85
rn = 50
thetan = 50
r0 = np.linspace(a, b, rn)
theta0 = np.linspace(0, np.pi/2, thetan)
r, theta = np.meshgrid(r0, theta0)
rtheta_list = (r, theta)  # my way
solver = CylindricalSandwich(NonHomogeneousOnly=True)  # NonHomogeneousOnly = True
soln = solver(rtheta_list, t)
contourV = np.linspace(0.001, 1.5, 200)
pt1 = plt.contour(soln.angle_theta, soln.position_r, soln.temperature, contourV)
manual_locations = [(0.25, 0.55), (0.5, 0.55), (1.0, 0.55), (1.4, 0.55)]
plt.clabel(pt1, inline=1, fontsize=10, manual=manual_locations)
#
plt.title(r'Nonhomogenous solution $ \bar T(x,y)$')
plt.ylim(a, b)
plt.xlim(0, np.pi/2)
plt.ylabel(r'$r$', fontsize=20)
plt.xlabel(r'$\theta$', fontsize=20)
plt.grid(True)
plt.savefig('cylindrical_sandwich_static.pdf')
plt.show()
