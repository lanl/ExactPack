#
#  Creates plots for the radiative-shock solutions in ExactPack.
#

from exactpack.solvers.radshocks.nED_radshocks import ED_Solver
from exactpack.solvers.radshocks.nED_radshocks import nED_Solver
from exactpack.solvers.radshocks.nED_radshocks import Sn_Solver
from exactpack.solvers.radshocks.nED_radshocks import ie_Solver
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', size=14)

import numpy as np

#  Initialize the solution objects.
#  Four solvers are shown below: ED_Solver, nED_Solver, Sn_Solver, ie_Solver.
#  Six solution objects are initialized in order to show that the nED_Solver
#  produces ED_Solver solutions when \epsion is small, and how to produce a
#  FLD solutions (the Wilson sum-limiter in this case).

ED_soln  = ED_Solver(M0 = 1.2)
nED_soln = nED_Solver(M0 = 1.2)
FLD1_soln = nED_Solver(M0 = 1.2, problem = 'FLD_1')
AP_soln  = nED_Solver(M0 = 1.2, epsilon = 0.1)
Sn_soln  = Sn_Solver(M0 = 1.2, f_tol = 1.e-4)
ie_soln = ie_Solver(M0 = 1.4)

#  Set vector of times to evaluate the solution.

xmin = -10.0
xmax = 10.0
t = 0.0
NP = 1e5

xvec = np.linspace(xmin, xmax, int(NP))

#  Evaluate the solutions.

ED_result = ED_soln(xvec, t)
nED_result = nED_soln(xvec, t)
FLD1_result = FLD1_soln(xvec, t)
AP_result = AP_soln(xvec, t)
Sn_result = Sn_soln(xvec, t)
ie_result = ie_soln(xvec, t)

#  Call out certain variables that are plotted below.

x_ED = ED_result['position']
x_nED = nED_result['position']
x_FLD1 = FLD1_result['position']
x_AP = AP_result['position']
x_Sn = Sn_result['position']
x_ie = ie_result['position']
Tm_ED = ED_result['temperature']
Tm_nED = nED_result['temperature_mat']
Tr_nED = nED_result['temperature_rad']
Tm_FLD1 = FLD1_result['temperature_mat']
Tr_FLD1 = FLD1_result['temperature_rad']
Tm_AP = AP_result['temperature_mat']
Tr_AP = AP_result['temperature_rad']
Tm_Sn = Sn_result['temperature_mat']
Tr_Sn = Sn_result['temperature_rad']
VEF_Sn = Sn_result['VEF']
Ti_ie = ie_result['temperature_ion']
Tm_ie = ie_result['temperature_mat']
Te_ie = ie_result['temperature_elec']

# Define simple parameters to make the radiative-shock plots look nicer.
Tm_max = np.max(Tm_nED)
Tm_min = np.min(Tm_nED)
delta_T = (Tm_max - Tm_min) / 100.
arg_x0 = np.argmin(np.abs(Tm_nED - Tm_max + delta_T))
arg_x1 = np.argmin(np.abs(Tm_nED - Tm_min - delta_T))
x0 = 1.1 * x_nED[arg_x0]
x1 = x_nED[arg_x1]
x1 = max(-x0 / 10., x1)
y0 = Tm_min - delta_T
y1 = Tm_max + delta_T

print('x0, x1, y0, y1:', x0, x1, y0, y1)

# Plot the four radiative-shock solutions.

fig = plt.figure(figsize=(10, 7))
ax1 = fig.add_subplot(231)
ax1.plot(x_ED, Tm_ED)
plt.xticks([-0.002, 0., 0.004], ['-0.002', '0.', '0.004'])
plt.yticks([100, 119], ['100', '119'])
plt.xlim((x0, x1))
plt.ylim((y0, y1))
plt.title('ED_Solver')
plt.grid(True)
ax2 = fig.add_subplot(232)
ax2.plot(x_nED, Tm_nED)
ax2.plot(x_nED, Tr_nED)
plt.xticks([-0.002, 0., 0.004], ['-0.002', '0.', '0.004'])
plt.yticks([100, 119], ['100', '119'])
plt.xlim((x0, x1))
plt.ylim((y0, y1))
plt.title('nED_Solver')
plt.grid(True)
ax3 = fig.add_subplot(233)
ax3.plot(x_AP, Tm_AP)
ax3.plot(x_AP, Tr_AP, '--')
plt.xticks([-0.002, 0., 0.004], ['-0.002', '0.', '0.004'])
plt.yticks([100, 119], ['100', '119'])
plt.xlim((x0, x1))
plt.ylim((y0, y1))
plt.title('nED_Solver ($\epsilon = 0.1$)')
plt.grid(True)
ax4 = fig.add_subplot(234)
ax4.plot(x_Sn, Tm_Sn)
ax4.plot(x_Sn, Tr_Sn)
plt.xticks([-0.002, 0., 0.004], ['-0.002', '0.', '0.004'])
plt.yticks([100, 119], ['100', '119'])
plt.xlim((x0, x1))
plt.ylim((y0, y1))
ax5 = ax4.twinx()
ax5.plot(x_Sn, VEF_Sn)
plt.title('Sn_Solver')
plt.grid(True)
ax6 = fig.add_subplot(235)
ax6.plot(x_FLD1, Tm_FLD1)
ax6.plot(x_FLD1, Tr_FLD1)
plt.xticks([-0.002, 0., 0.004], ['-0.002', '0.', '0.004'])
plt.yticks([100, 119], ['100', '119'])
plt.xlim((x0, x1))
plt.ylim((y0, y1))
plt.title('FLD1_Solver')
plt.grid(True)

# Define simple parameters to make the ion-electron plots look nicer.
Ti_max = np.max(Ti_ie)
Ti_min = np.min(Ti_ie)
delta_T = (Ti_max - Ti_min) / 100.
arg_x0 = np.argmin(np.abs(Ti_ie - Ti_max + delta_T))
arg_x1 = np.argmin(np.abs(Ti_ie - Ti_min - delta_T))
x0 = 1.1 * x_ie[arg_x0]
x1 = x_ie[arg_x1]
x1 = max(-x0 / 10., x1)
y0 = Ti_min - delta_T
y1 = Ti_max + delta_T

# Plot the ion-electron shock solution.

ax7 = fig.add_subplot(236)
ax7.plot(x_ie, Ti_ie)
ax7.plot(x_ie, Tm_ie)
ax7.plot(x_ie, Te_ie)
# plt.xticks([-9, 0., 5], ['0.', '1.', '3.'])
# plt.yticks([100, 146], ['100', '140'])
# plt.xlim((x0, x1))
# plt.ylim((y0, y1))
plt.xlim((x0-0.001, x1+0.001))
plt.ylim((y0, y1))
plt.title('ie_Solver')
plt.grid(True)
plt.tight_layout()

print('x0, x1, y0, y1:', x0, x1, y0, y1)
plt.show()

plt.close()
