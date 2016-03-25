import numpy as np
import matplotlib.pylab as plt
from matplotlib import rc
#rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

# import ExactPack solvers
from exactpack.solvers.sedov import SphericalSedov as SphericalSedovTimmes
from exactpack.solvers.sedov.kamm import Sedov as SedovKamm

#####################################################################
rmax = 1.2
r = np.linspace(0.0, rmax, 1000)
t = 1.

#####################################################################
solverKamm = SedovKamm(geometry=3, eblast=0.851072, gamma=1.4)
solnKamm = solverKamm(r, t)

plt.figure()
solnKamm.plot('density')
solnKamm.plot('pressure', scale=10)
solnKamm.plot('velocity', scale=10)
solnKamm.plot('sie')
plt.xlim(0.0, rmax)
plt.ylim(0.0, 6.5)
plt.title(r'ExactPack solver Kamm for Spherical at $t_{\rm max}=1\,{\rm s}$')
plt.legend(loc=2)
plt.grid(True)

#####################################################################
# construct spherical spatial grid and choose time
solver = SphericalSedovTimmes()
soln = solver(r, t)

plt.figure()
soln.plot('density')
soln.plot('pressure', scale=10)
soln.plot('velocity', scale=10)
soln.plot('sie')
plt.xlim(0.0, rmax)
plt.ylim(0.0, 6.5)
plt.title(r'ExactPack solvers Timmes at $t_{\rm max}=1\,{\rm s}$')
plt.legend(loc=0)
plt.grid(True)

####################################################
# Reproduce Table 3 of LA-UR-00-6053 p 19 [1]
# Comparison with Sedov's book, Table 3, LA-UR-00-6055, p 19
# f -> velocity -> sedov[1]
# g -> density  -> sedov[2]
# h -> pressure -> sedov[3]
# V -> sie [by convention]
#                  lambda  Sedov-f Sedov-g sedov-h
sedov = np.array([0.9913, 0.9814, 0.8379, 0.9109,
                  0.9773, 0.9529, 0.6457, 0.7993,
                  0.9622, 0.9237, 0.4978, 0.7078,
                  0.9342, 0.8744, 0.3241, 0.5923,
                  0.9080, 0.8335, 0.2279, 0.5241,
                  0.8747, 0.7872, 0.1509, 0.4674,
                  0.8359, 0.7397, 0.0967, 0.4272,
                  0.7950, 0.6952, 0.0621, 0.4021,
                  0.7493, 0.6496, 0.0379, 0.3856,
                  0.6788, 0.5844, 0.0174, 0.3732,
                  0.5794, 0.4971, 0.0052, 0.3672,
                  0.4560, 0.3909, 0.0009, 0.3656,
                  0.3600, 0.3086, 0.0002, 0.3655,
                  0.2960, 0.2538, 0.0000, 0.3655,
                  0.2000, 0.1714, 0.0000, 0.3655,
                  0.1040, 0.0892, 0.0000, 0.3655]).reshape(16, 4)

t = 1.0
solver = SedovKamm(geometry=3, eblast=0.851072, gamma=1.4)
lam = sedov[:, 0]
soln = solver(lam, t)  # these values reproduce those of Ref [1]

print 'Reproduction of Table 3 in LA-UR-00-6055 on p 19.'
print '{0}  {1:11} {2:7} {3} {4} {5} {6} {7}'.format('lambda', 'V', \
'Sedov-f', 'Exact-f', 'Sedov-g', 'Exact-g', 'Sedov-h', 'Exact-h')
print '==================================================================='
for i in range(len(lam)):
    print '{0:.4f}  {1:.8f}  {2:.4f}  {3:.4f}  {4:.4f}  {5:.4f}  {6:.4f}  \
    {7:.4f}'.format(soln[i][0], soln[i][2], sedov[i][1], soln[i][4], \
                    sedov[i][2], soln[i][1], sedov[i][3], soln[i][3])


####################################################
plt.show()

