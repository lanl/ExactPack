import numpy as np
import matplotlib.pylab as plt

from exactpack.solvers.heat import Rod1D
from exactpack.solvers.heat import PlanarSandwichDawes
from exactpack.solvers.heat import PlanarSandwich
from exactpack.solvers.heat import Rectangle


# rod1D
##########################################################
L = 2.0
x = np.linspace(0.0, L, 1000)
t0 = 1.0
t1 = 0.2
t2 = 0.1
t3 = 0.01
t4 = 0.001

# BC1 homogeneous
solver = Rod1D(alpha1=1, beta1=0, alpha2=1, beta2=0, TL=3, TR=4, Nsum=300)
soln0 = solver(x, t0)
soln1 = solver(x, t1)
soln2 = solver(x, t2)
soln3 = solver(x, t3)
soln4 = solver(x, t4)
soln0.plot('temperature', label=r'$t=1.000$')
soln1.plot('temperature', label=r'$t=0.200$')
soln2.plot('temperature', label=r'$t=0.100$')
soln3.plot('temperature', label=r'$t=0.010$')
soln4.plot('temperature', label=r'$t=0.001$')
plt.title('Rod1D: BC1 homogeneous')
plt.ylim(0, 5.0)
plt.xlim(0, L)
plt.legend(loc=2)
plt.grid(True)
plt.savefig('rod1D_BC1.pdf')
plt.show()

# BC1 nonhomogeneous
solver = Rod1D(alpha1=1, beta1=0, alpha2=1, gamma1=1, beta2=0, gamma2=0, TL=0, TR=0, Nsum=300)
soln0 = solver(x, t0)
soln1 = solver(x, t1)
soln2 = solver(x, t2)
soln3 = solver(x, t3)
soln4 = solver(x, t4)
soln0.plot('temperature', label=r'$t=1.000$')
soln1.plot('temperature', label=r'$t=0.200$')
soln2.plot('temperature', label=r'$t=0.100$')
soln3.plot('temperature', label=r'$t=0.010$')
soln4.plot('temperature', label=r'$t=0.001$')
plt.title('Rod1D: BC1 nonhomogeneous')
plt.ylim(0, 1.1)
plt.xlim(0, L)
plt.legend(loc=0)
plt.grid(True)
plt.savefig('rod1D_BC1_nonhomo.pdf')
plt.show()

# BC2 homogeneous
solver = Rod1D(alpha1=0, beta1=1, alpha2=0, beta2=1, TL=3, TR=4, Nsum=300)
t0 = 5.0
t1 = 0.4
soln0 = solver(x, t0)
soln1 = solver(x, t1)
soln2 = solver(x, t2)
soln3 = solver(x, t3)
soln4 = solver(x, t4)
soln0.plot('temperature', label=r'$t=5.000$')
soln1.plot('temperature', label=r'$t=0.500$')
soln2.plot('temperature', label=r'$t=0.100$')
soln3.plot('temperature', label=r'$t=0.010$')
soln4.plot('temperature', label=r'$t=0.001$')
plt.title('Rod1D: BC2 homogeneous')
plt.ylim(2.9, 4.1)
plt.xlim(0, L)
plt.legend(loc=2)
plt.grid(True)
plt.savefig('rod1D_BC2.pdf')
plt.show()

# BC2 nonhomogeneous
F = 1
solver = Rod1D(alpha1=0, beta1=1, gamma1=F, alpha2=0, beta2=1, gamma2=F, TL=0, TR=0, Nsum=300)
t0 = 5.0
t1 = 0.4
soln0 = solver(x, t0)
soln1 = solver(x, t1)
soln2 = solver(x, t2)
soln3 = solver(x, t3)
soln4 = solver(x, t4)
soln0.plot('temperature', label=r'$t=5.000$')
soln1.plot('temperature', label=r'$t=0.500$')
soln2.plot('temperature', label=r'$t=0.100$')
soln3.plot('temperature', label=r'$t=0.010$')
soln4.plot('temperature', label=r'$t=0.001$')
plt.title('Rod1D: BC2 nonhomogeneous')
plt.ylim(-1.5, 1.5)
plt.xlim(0, L)
plt.legend(loc=2)
plt.grid(True)
plt.savefig('rod1D_BC2_nonhomo.pdf')
plt.show()

# BC3 and BC4 homogeneous
solver3 = Rod1D(alpha1=1, beta1=0, alpha2=0, beta2=1, Nsum=300, TL=3, TR=4)
solver4 = Rod1D(alpha1=0, beta1=1, alpha2=1, beta2=0, Nsum=300, TL=4, TR=3)
#
plt.subplot(2, 1, 1)
plt.title('Rod1D: BC3 and BC4 homogeneous')
soln0 = solver3(x, t0)
soln1 = solver3(x, t1)
soln2 = solver3(x, t2)
soln3 = solver3(x, t3)
soln0.plot('temperature', label=r'$t=1.0$')
soln1.plot('temperature', label=r'$t=0.1$')
soln2.plot('temperature', label=r'$t=0.01$')
soln3.plot('temperature', label=r'$t=0.001$')
plt.ylim(-0.2, 4)
plt.xlim(0, L)
plt.legend(loc=0)
plt.grid(True)
#
plt.subplot(2, 1, 2)
soln0 = solver4(x, t0)
soln1 = solver4(x, t1)
soln2 = solver4(x, t2)
soln3 = solver4(x, t3)
soln0.plot('temperature', label=r'$t=1.0$')
soln1.plot('temperature', label=r'$t=0.1$')
soln2.plot('temperature', label=r'$t=0.01$')
soln3.plot('temperature', label=r'$t=0.001$')
plt.ylim(-0.2, 4)
plt.xlim(0, L)
plt.legend(loc=0)
plt.grid(True)
plt.savefig('rod1D_BC3_BC4.pdf')
plt.show()

# BC3 and BC4 nonhomogeneous
solver3 = Rod1D(alpha1=1, beta1=0, gamma1=1, alpha2=0, beta2=1, gamma2=0, Nsum=300, TL=0, TR=0)
solver4 = Rod1D(alpha1=0, beta1=1, gamma1=0, alpha2=1, beta2=0, gamma2=1, Nsum=300, TL=0, TR=0)
#
plt.subplot(2, 1, 1)
plt.title('Rod1D: BC3 and BC4 nonhomogeneous')
soln0 = solver3(x, t0)
soln1 = solver3(x, t1)
soln2 = solver3(x, t2)
soln3 = solver3(x, t3)
soln0.plot('temperature', label=r'$t=1.0$')
soln1.plot('temperature', label=r'$t=0.1$')
soln2.plot('temperature', label=r'$t=0.01$')
soln3.plot('temperature', label=r'$t=0.001$')
plt.ylim(-0.2, 1.2)
plt.xlim(0, L)
plt.legend(loc=0)
plt.grid(True)
#
plt.subplot(2, 1, 2)
soln0 = solver4(x, t0)
soln1 = solver4(x, t1)
soln2 = solver4(x, t2)
soln3 = solver4(x, t3)
soln0.plot('temperature', label=r'$t=1.0$')
soln1.plot('temperature', label=r'$t=0.1$')
soln2.plot('temperature', label=r'$t=0.01$')
soln3.plot('temperature', label=r'$t=0.001$')
plt.ylim(-0.2, 1.2)
plt.xlim(0, L)
plt.legend(loc=0)
plt.grid(True)
plt.savefig('rod1D_BC3_BC4_nonhomo.pdf')
plt.show()

# general BC's
solver1 = Rod1D(alpha1=0, beta1=-1, alpha2=1, beta2=2, L=2, TL=3, TR=4)
soln0 = solver1(x, t0)
soln1 = solver1(x, t1)
soln2 = solver1(x, t2)
soln3 = solver1(x, t3)
soln0.plot('temperature', label=r'$t=1.0$')
soln1.plot('temperature', label=r'$t=0.1$')
soln2.plot('temperature', label=r'$t=0.01$')
soln3.plot('temperature', label=r'$t=0.001$')
plt.title('Rod1D: general BC homogeneous')
plt.ylim(0, 4.5)
plt.xlim(0, L)
plt.legend(loc=0)
plt.grid(True)
plt.savefig('rod1D_BC1_general.pdf')
plt.show()

# Planar Sandwich
##########################################################
L = 2.0
ymax = 100
y = np.linspace(0.0, L, ymax)
t0 = 1.0
t4 = 0.600
t1 = 0.1
t2 = 0.01
t3 = 0.001

# Dawes' fortran implementation
solver = PlanarSandwichDawes(Nsum=20000, TB=1, TT=0)
soln0 = solver(y, t0)
soln4 = solver(y, t4)
soln1 = solver(y, t1)
soln2 = solver(y, t2)
soln3 = solver(y, t3)
soln0.plot('temperature', label=r'$t=1.0$')
soln4.plot('temperature', label=r'$t=0.600$')
soln1.plot('temperature', label=r'$t=0.1$')
soln2.plot('temperature', label=r'$t=0.01$')
soln3.plot('temperature', label=r'$t=0.001$')
plt.title('PlanarSandwichDawes')
plt.ylim(0, 1.1)
plt.xlim(0, L)
plt.legend(loc=0)
plt.grid(True)
plt.savefig('plandar_sandwich_dawes.pdf')
plt.show()

# Python implementation in 1D
solver = PlanarSandwich(Nsum=20000, TB=1, TT=0)
soln0 = solver(y, t0)
soln4 = solver(y, t4)
soln1 = solver(y, t1)
soln2 = solver(y, t2)
soln3 = solver(y, t3)
soln0.plot('temperature', label=r'$t=1.0$')
soln4.plot('temperature', label=r'$t=0.600$')
soln1.plot('temperature', label=r'$t=0.1$')
soln2.plot('temperature', label=r'$t=0.01$')
soln3.plot('temperature', label=r'$t=0.001$')
plt.title('PlanarSandwich')
plt.ylim(0, 1.1)
plt.xlim(0, L)
plt.legend(loc=0)
plt.grid(True)
plt.savefig('plandar_sandwich_1d.pdf')
plt.show()


# Rectangle
##########################################################
a = 2.0  # length along x-direction
b = 2.0  # lenght along y-direction
#
# static nonhomogeneous solution
#
xn = 50
yn = 50
x0 = np.linspace(0.0, a, xn)
y0 = np.linspace(0.0, b, yn)
x, y = np.meshgrid(x0, y0)
xylist = (x, y)
#
t = 0  # dummy variable
plt.title(r'Nonhomogenous solution $\bar T(x,y)$')
solver = Rectangle(NonHomogeneousOnly=True, Nsum=100)
soln = solver(xylist, t)
contourV = np.linspace(0.05, 1.0, 10)
pt1 = plt.contour(soln.position_x, soln.position_y, soln.temperature, contourV)
manual_locations = [(1, 0.5), (1, 1.0), (1, 1.5), (1, 2.0)]
plt.clabel(pt1, inline=1, fontsize=10, manual=manual_locations)
#
plt.ylim(0, a)
plt.xlim(0, b)
plt.grid(True)
plt.savefig('rectangle_static.pdf')
plt.show()

# plot the evolution in 2x2 plots at t1,t2,t3,t4
#
xn = 30
yn = 30
plt.subplot(2, 2, 1)
x0 = np.linspace(0.0, a, xn)
y0 = np.linspace(0.0, b, yn)
x, y = np.meshgrid(x0, y0)
xylist = (x, y)
#
t = 0.01
plt.title(r'$t=0.01$')
solver = Rectangle(Nsum=100)
soln = solver(xylist, t)
contourV = np.linspace(0.25, 1.0, 8)
pt1 = plt.contour(soln.position_x, soln.position_y, soln.temperature, contourV)
manual_locations = [(1, 1.0), (1, 2.0)]
plt.clabel(pt1, inline=1, fontsize=10, manual=manual_locations)
#
plt.subplot(2, 2, 2)
t = 0.1
plt.title(r'$t=0.1$')
solver = Rectangle(Nsum=100)
soln = solver(xylist, t)
contourV = np.linspace(0.2, 1.0, 10)
pt1 = plt.contour(soln.position_x, soln.position_y, soln.temperature, contourV)
manual_locations = [(1, 0.5), (1, 1.0), (1, 1.25), (1, 1.5), (1, 1.75), (1, 2.0)]
plt.clabel(pt1, inline=1, fontsize=10, manual=manual_locations)
#
plt.subplot(2, 2, 3)
t = 0.2
plt.title(r'$t=0.2$')
solver = Rectangle(Nsum=100)
soln = solver(xylist, t)
contourV = np.linspace(0.05, 1.0, 20)
pt1 = plt.contour(soln.position_x, soln.position_y, soln.temperature, contourV)
manual_locations = [(1, 0.5), (1, 1.0), (1, 1.25), (1, 1.5), (1, 1.75), (1, 2.0)]
plt.clabel(pt1, inline=1, fontsize=10, manual=manual_locations)
#
plt.subplot(2, 2, 4)
t = 0.5
plt.title(r'$t=0.5$')
solver = Rectangle(Nsum=100)
soln = solver(xylist, t)
contourV = np.linspace(0.05, 1.0, 20)
pt1 = plt.contour(soln.position_x, soln.position_y, soln.temperature, contourV)
manual_locations = [(1, 0.25), (1, 0.5), (1, 1.0), (1, 1.25), (1, 1.5), (1, 1.75), (1, 2.0)]
plt.clabel(pt1, inline=1, fontsize=10, manual=manual_locations)
#
plt.ylim(0, a)
plt.xlim(0, b)
plt.grid(True)
plt.savefig('rectangle.pdf')
plt.show()
