#
#  Creates plots for the Escape of HE Products (EHEP) problem
#  in ExactPack
#

from exactpack.solvers.ehep import EscapeOfHEProducts
import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='serif', size=12)

#  Initialize the solution object

solution = EscapeOfHEProducts(D=0.85, rho_0=1.6)

#  Set vector of times to evaluate the solution

xmax = 5.0
t = 5.0
NP = 201

xvec = np.linspace(0, xmax, NP)

#  Evaluate the solution

result = solution._run(xvec, t)

#  Plot the solution vs time

plt.subplot(2, 2, 1)
plt.plot(xvec, result['density'], 'k')
plt.ylabel(r'density [$\rm{g/cm^3}$]')

plt.subplot(2, 2, 2)
plt.plot(xvec, result['pressure'], 'k')
plt.ylabel(r'pressure [$\rm{Mbar}$]')

plt.subplot(2, 2, 3)
plt.plot(xvec, result['specific_internal_energy'], 'k')
plt.ylabel(r'specific internal energy [$\rm{Mbar-cm^3/g}$]')
plt.xlabel(r'position [$\rm{cm}$]', labelpad=20)

plt.subplot(2, 2, 4)
plt.plot(xvec, result['velocity'], 'k')
plt.ylabel(r'particle velocity [$\rm{cm/\mu s}$]')
plt.xlabel(r'position [$\rm{cm}$]', labelpad=20)

plt.suptitle('EHEP results at t={}$\mu\sec$'.format(t))

plt.tight_layout()

plt.show()
