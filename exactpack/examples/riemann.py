import numpy as np

import matplotlib.pylab as plt

# import ExactPack solver and analysis tools
from exactpack.solvers.riemann.riemann import Riemann
from exactpack.solvers.riemann import *

# create solution on r-grid
r = np.linspace(0.0, 1.0, 200)

#####################################################################
# Riemann
t = 0.25
solver = Riemann()
soln = solver(r,t)
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity')
# plot exact solution
plt.xlim(0.0,1.0)
plt.ylim(0.0,1.5)
plt.title('Riemann Kamm')
plt.legend(loc=0)
plt.grid(True)
plt.show()


#####################################################################
# test1: Sod
t = 0.25
solver = Sod()
soln = solver(r,t)
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity')
# plot exact solution
plt.xlim(0.0,1.0)
plt.ylim(0.0,1.5)
plt.title('Riemann test1: Sod at $t=0.25$s')
plt.legend(loc=0)
plt.grid(True)
plt.show()

#####################################################################
# test2: Einfeldt
t = 0.15
solver =  Einfeldt()
soln = solver(r,t)
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity')
# plot exact solution
plt.xlim(0.0,1.0)
plt.ylim(-2.1,2.1)
plt.title('Riemann test2: Einfeldt at $t=0.15$s')
plt.legend(loc=0)
plt.grid(True)
plt.show()

#####################################################################
# test3: StationaryContact
t = 0.012
solver = StationaryContact()
soln = solver(r,t)
soln.plot('density',scale=100)
soln.plot('pressure')
soln.plot('velocity')
# plot exact solution
plt.xlim(0.0,1.0)
plt.ylim(-500., 3000.)
plt.title('Riemann test3: Stationary Contact at $t=0.012$s')
plt.legend(loc=0)
plt.grid(True)
plt.show()

#####################################################################
# test4: SlowShock
t = 1.0
solver = SlowShock()
soln = solver(r,t)
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity')
# plot exact solution
plt.xlim(0.0,1.0)
plt.ylim(-10.0,25.5)
plt.title('Riemann test4: Slow Shock at $t=1$s')
plt.legend(loc=0)
plt.grid(True)
plt.show()

#####################################################################
# test5: ShockContactShock
t = 0.3
solver = ShockContactShock()
soln = solver(r,t)
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity')
# plot exact solution
plt.xlim(0.0,1.0)
plt.ylim(-1.2,2.5)
plt.title('Riemann test5: Shock-Contact-Shock at $t=0.3$s')
plt.legend(loc=0)
plt.grid(True)
plt.show()

#####################################################################
# test6: LeBlanc
t = 0.5
solver =  LeBlanc()
soln = solver(r,t)
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity')
# plot exact solution
plt.xlim(0.0,1.0)
plt.ylim(0.0,1.2)
plt.title('Riemann test6: LeBlanc at $t=0.5$s')
plt.legend(loc=0)
plt.grid(True)
plt.show()
