import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', size=14)

from exactpack.solvers.noh.noh1 import Noh
from exactpack.solvers.nohblackboxeos import PlanarNohBlackBox, CylindricalNohBlackBox, SphericalNohBlackBox

rmax = 1   # analysis domain
r = np.linspace(0.0, rmax, 1000)
t = 0.3

class ideal_gas_eos: 
    def __init__(self, gamma = 5/3): 
        self.gamma = gamma
        assert(gamma != 1), "Gamma cannot be equal to 1"

    def e(self, rho, P): 
        if(rho == 0.0):
            raise ValueError("Error: rho = 0. Please do not break math and divide by zero.")
        return P/(rho*(self.gamma -1.))     
    
    def de_dP(self, rho, P): 
        if(rho == 0.0):
            raise ValueError("Error: rho = 0. Please do not break math and divide by zero.")
        return 1./(rho*(self.gamma-1))
    
    def de_drho(self, rho,P): 
        if(rho == 0.0):
            raise ValueError("Error: rho = 0. Please do not break math and divide by zero.")
        return -P/(rho**2*(self.gamma-1))

ideal_gas = ideal_gas_eos()

#####################################################################
# solver object
solver = PlanarNohBlackBox(ideal_gas)
soln = solver(r,t)
# plot exact solution
fig = plt.figure(figsize=(10,7))
soln.plot('density')
soln.plot('pressure')
soln.plot('velocity',scale=10)
soln.plot('specific_internal_energy',scale=10)
plt.xlim(0.0,rmax)
plt.title('ExactPack solver class PlanarNohBlackBox')
plt.ylim(-15,70)
plt.legend(loc=1)
plt.grid(True)
plt.tight_layout()
plt.show()


#####################################################################
# solver object
cyl_solver = CylindricalNohBlackBox(ideal_gas)
cyl_soln = cyl_solver(r,t)
# plot exact solution
cyl_fig = plt.figure(figsize=(10,7))
cyl_soln.plot('density')
cyl_soln.plot('pressure')
cyl_soln.plot('velocity',scale=10)
cyl_soln.plot('specific_internal_energy',scale=10)
plt.xlim(0.0,rmax)
plt.title('ExactPack solver class PlanarNohBlackBox')
plt.ylim(-15,70)
plt.legend(loc=1)
plt.grid(True)
plt.tight_layout()
plt.show()


#####################################################################
# solver object
sph_solver = SphericalNohBlackBox(ideal_gas)
sph_soln = sph_solver(r,t)
# plot exact solution
sph_fig = plt.figure(figsize=(10,7))
sph_soln.plot('density')
sph_soln.plot('pressure')
sph_soln.plot('velocity',scale=10)
sph_soln.plot('specific_internal_energy',scale=10)
plt.xlim(0.0,rmax)
plt.title('ExactPack solver class PlanarNohBlackBox')
plt.ylim(-15,70)
plt.legend(loc=1)
plt.grid(True)
plt.tight_layout()
plt.show()