import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from exactpack.solvers.nohblackboxeos.equations_of_state import (ideal_gas_eos, noble_abel_eos, stiffened_gas_eos, carnahan_starling_eos, aluminum_eos)
from exactpack.solvers.nohblackboxeos import (
        NohBlackBoxEos,
    PlanarNohBlackBox,
    CylindricalNohBlackBox,
    SphericalNohBlackBox,
)

rc("font", size=14)


# from exactpack.solvers.noh.noh1 import Noh

rmax = 1  # analysis domain
r = np.linspace(0.0, rmax, 1000)
t = 0.3



####################################################################
################# EXAMPLES USING IDEAL GAS #########################
####################################################################
# Equation of State Object
ideal_gas = ideal_gas_eos()

#####################################################################
# solver object
solver = PlanarNohBlackBox(ideal_gas)
solver.set_new_solver_initial_guess([5, 1, 1])
soln = solver(r, t)
# plot exact solution
fig = plt.figure(figsize=(10, 7))
soln.plot("density")
soln.plot("pressure")
soln.plot("velocity", scale=10)
soln.plot("specific_internal_energy", scale=10)
plt.xlim(0.0, rmax)
plt.title("ExactPack solver class PlanarNohBlackBox using Ideal Gas EoS")
plt.ylim(-15, 70)
plt.legend(loc=1)
plt.grid(True)
plt.tight_layout()
plt.show()


######################################################################
## solver object
cyl_solver = CylindricalNohBlackBox(ideal_gas)
cyl_solver.set_new_solver_initial_guess([5, 1, 1])
cyl_soln = cyl_solver(r, t)
# plot exact solution
cyl_fig = plt.figure(figsize=(10, 7))
cyl_soln.plot("density")
cyl_soln.plot("pressure")
cyl_soln.plot("velocity", scale=10)
cyl_soln.plot("specific_internal_energy", scale=10)
plt.xlim(0.0, rmax)
plt.title("ExactPack solver class CylindricalNohBlackBox using Ideal Gas EoS")
plt.ylim(-15, 70)
plt.legend(loc=1)
plt.grid(True)
plt.tight_layout()
plt.show()


#####################################################################
# solver object
sph_solver = SphericalNohBlackBox(ideal_gas)
sph_solver.set_new_solver_initial_guess([50,1,0.5])
sph_soln = sph_solver(r, t)
# plot exact solution
sph_fig = plt.figure(figsize=(10, 7))
sph_soln.plot("density")
sph_soln.plot("pressure")
sph_soln.plot("velocity", scale=10)
sph_soln.plot("specific_internal_energy", scale=10)
plt.xlim(0.0, rmax)
plt.title("ExactPack solver class SphericalNohBlackBox using Ideal Gas EoS")
plt.ylim(-15, 70)
plt.legend(loc=1)
plt.grid(True)
plt.tight_layout()
plt.show()

####################################################################
################# EXAMPLES USING NOBLE ABLE ########################
####################################################################
noble_able = noble_abel_eos() # Defaul: b = 0.01, gamma = 5/3
######################################################################
# solver object

solver = PlanarNohBlackBox(noble_able)
solver.set_new_solver_initial_guess([5, 1, 1])
solver.solve_jump_conditions()
soln = solver(r, t)
# plot exact solution
fig = plt.figure(figsize=(10, 7))
soln.plot("density")
soln.plot("pressure")
soln.plot("velocity", scale=10)
soln.plot("specific_internal_energy", scale=10)
plt.xlim(0.0, rmax)
plt.title("ExactPack solver class PlanarNohBlackBox using Noble Able EoS")
plt.ylim(-15, 70)
plt.legend(loc=1)
plt.grid(True)
plt.tight_layout()
plt.show()



##################################################################################################
################## EXAMPLES USING STIFF GAS AND UNIQUE INITIAL CONDITIONS ########################
##################################################################################################
## Equation of State Object
stiff_gas = stiffened_gas_eos() # Default: p_inf = 1, C_s = sqrt(5/3) = sqrt(gamma), gamma = 5/3

#################################################################################################
## initial conditions; be aware the Noh Problem with stiff requires symmetry = 0
initial_conditions = {'velocity': -2, 'density': 3, 'pressure': 1}

#################################################################################################
## solver object

solver = PlanarNohBlackBox(stiff_gas, initial_conditions)
solver.set_new_solver_initial_guess([5,1,1])
solver.solve_jump_conditions()
soln = solver(r, t)
# plot exact solution
fig = plt.figure(figsize=(10, 7))
soln.plot("density")
soln.plot("pressure")
soln.plot("velocity", scale=10)
soln.plot("specific_internal_energy", scale=10)
plt.xlim(0.0, rmax)
plt.title("ExactPack solver class PlanarNohBlackBox using Stiff Gas EoS")
plt.ylim(-15, 70)
plt.legend(loc=1)
plt.grid(True)
plt.tight_layout()
plt.show()

#################################################################################################
################# EXAMPLES USING CARNAHAN-STARLING IN HIGHER GOEMETRY ###########################
#################################################################################################
# Equation of State Object
carnahan_starling = carnahan_starling_eos(b=0.01) #Default: b = 1

################################################################################################
# initial conditions; be aware the Noh Problem with stiff requires symmetry = 0
initial_conditions = {'velocity': -2, 'density': 3, 'pressure': 0}

################################################################################################
# solver object

cs_solver = SphericalNohBlackBox(carnahan_starling, initial_conditions)
cs_solver.set_new_solver_initial_guess([15,1,1])
cs_solver.solve_jump_conditions()
soln = cs_solver(r,t)
## plot exact solution
fig = plt.figure(figsize=(10, 7))
soln.plot("density")
soln.plot("pressure")
soln.plot("velocity", scale=10)
soln.plot("specific_internal_energy", scale=10)
plt.xlim(0.0, rmax)
plt.title("ExactPack solver class PlanarNohBlackBox using Carnahn-Starling EoS")
plt.ylim(-15, 100)
plt.legend(loc=1)
plt.grid(True)
plt.tight_layout()
plt.show()

###########################################################
################# EXAMPLES USING ALUMINUM #################
###########################################################
# Aluminum EOS

###########################################################
# solver object

al_eos = aluminum_eos()
al_initial_conditions = {'velocity': -1.5*0.524 * 1e3, 'density': 2.7, 'pressure': 20, 'symmetry': 0}
rho_e_D_guess = [2.7, 1.55174098e+08, 524230]
al_solver = PlanarNohBlackBox(al_eos, al_initial_conditions)
al_solver.set_new_solver_initial_guess(rho_e_D_guess)
al_solver.solve_jump_conditions()
al_r = np.linspace(0.0, 500000, 100000)
soln =  al_solver(al_r,t)
## plot exact solution
fig = plt.figure(figsize=(10, 7))
soln.plot("density")
soln.plot("pressure", scale = 1.0e-8)
soln.plot("velocity")
soln.plot("specific_internal_energy")
plt.xlim(0.0, 500000)
plt.title("ExactPack solver class PlanarNohBlackBox using Aluminum Steinberg EoS")
plt.ylim(-15,100)
plt.legend(loc=1)
plt.grid(True)
plt.tight_layout()
plt.show()

