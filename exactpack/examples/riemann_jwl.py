import numpy as np

import matplotlib.pylab as plt
from matplotlib import rc, rcParams
#rc('text', usetex=True)
rc('font', **{'family':'serif','serif':['Computer Modern']})

# import ExactPack solver and analysis tools
from exactpack.contrib.riemann_jwl.kamm import RiemannJWL, RiemannJWLLee, RiemannJWLShyue

################################################################
rmax = 100.
r = np.linspace(0.0, rmax, 100)


################################################################
t = 20.0
solver = RiemannJWLLee()
soln = solver(r,t)

# plot solution
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity')
plt.xlim(0.0,rmax)
plt.ylim(-0.5,5.0)
plt.legend(loc=2)
plt.title('RiemannJWLLee: at $t=20$s')
plt.grid(True)
plt.show()


################################################################
t = 12.0
solver = RiemannJWLShyue()
soln = solver(r,t)

soln.plot('density')
soln.plot('pressure')
soln.plot('velocity')
plt.xlim(0.0,rmax)
plt.ylim(-0.5,8.0)
plt.legend(loc=1)
plt.title('RiemannJWLSyue: at $t=12$s')
plt.grid(True)
plt.show()

################################################################
t = 20.0
solver = RiemannJWL(interface_loc=50., 
                    rhol=0.9525, pl=1., ul=0., 
                    rhor=3.810, pr=2., ur=0.,
                    rho0l=1.905, sie0l=0., gammal=0.8938, bigal=6.321e2, bigbl=-4.472e-2, r1l=1.13e1, r2l=1.13, 
                    rho0r=1.905, sie0r=0., gammar=0.8938, bigar=6.321e2, bigbr=-4.472e-2, r1r=1.13e1, r2r=1.13)
soln = solver(r,t)

soln.plot('density')
soln.plot('pressure')
soln.plot('velocity')
plt.xlim(0.0,rmax)
plt.ylim(-0.5,5.0)
plt.legend(loc=2)
plt.title('Riemann JWL: Lee at $t=20$s')
plt.grid(True)
plt.show()

# Defauls to Lee
################################################################
t = 20.0
solver = RiemannJWL()
soln = solver(r,t)

soln.plot('density')
soln.plot('pressure')
soln.plot('velocity')
plt.xlim(0.0,rmax)
plt.ylim(-0.5,5.0)
plt.legend(loc=2)
plt.title('Riemann JWL Default: Lee at $t=20$s')
plt.grid(True)
plt.show()
