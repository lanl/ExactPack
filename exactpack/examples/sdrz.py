#  
#  Creates plots for the steady detonation reaction zone (SDRZ) problem
#  in ExactPack
#

from exactpack.solvers.sdrz import SteadyDetonationReactionZone
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', size=14)

import numpy as np

#  Initialize the solution object

solution = SteadyDetonationReactionZone(D=0.85, rho_0=1.6, gamma=3.0)

#  Set vector of times to evaluate the solution

tfinal = 2.0
NPt = 201   
xmax = 3.0
NPx = 201

tvec = np.linspace(0, tfinal, NPt)

#  Evaluate the solution

result = solution.run_tvec(tvec)

#  Plot the solution vs time
plt.figure('Time history of tracer particle that starts in the shock',
           figsize=(10,7))

plt.subplot(2,2,1)
plt.plot(tvec, result['density'],'k')
plt.ylabel(r'density [$\rm{g/cm^3}$]')
plt.grid(True)

plt.subplot(2,2,2)
plt.plot(tvec, result['pressure'],'k')
plt.ylabel(r'pressure [$\rm{Mbar}$]')
plt.grid(True)

plt.subplot(2,2,3)
plt.plot(tvec, result['velocity'],'k')
plt.ylabel(r'particle velocity [$\rm{cm/\mu s}$]')
plt.xlabel(r'time [$\rm{\mu s}$]', labelpad=20)
plt.grid(True)

plt.subplot(2,2,4)
plt.plot(tvec, result['reaction_progress'],'k')
plt.ylim([0,1.05])
plt.ylabel(r'reaction progress, $\rm \lambda$', labelpad=5)
plt.xlabel(r'time [$\rm{\mu s}$]', labelpad=20)
plt.grid(True)

plt.tight_layout()
plt.show()


#  Plot the solution snapshot at final time at vector xvec


xvec = np.linspace(0., xmax,  NPx)

result = solution._run(xvec, tfinal)

plt.figure('Solution snapshot at t={0} us'.format(tfinal), figsize=(10,7))

plt.subplot(2,2,1)
plt.plot(result['position'], result['density'],'k')
plt.ylabel(r'density [$\rm{g/cm^3}$]')
plt.grid(True)

plt.subplot(2,2,2)
plt.plot(result['position'], result['pressure'],'k')
plt.ylabel(r'pressure [$\rm{Mbar}$]')
plt.grid(True)

plt.subplot(2,2,3)
plt.plot(result['position'], result['velocity'],'k')
plt.ylabel(r'particle velocity [$\rm{cm/\mu s}$]')
plt.xlabel(r'absolute position [$\rm{cm}$]', labelpad=20)
plt.grid(True)

plt.subplot(2,2,4)
plt.plot(result['position'], result['reaction_progress'],'k')
plt.ylim([0,1.05])
plt.ylabel(r'reaction progress, $\rm \lambda$', labelpad=5)
plt.xlabel(r'absolute position [$\rm{cm}$]', labelpad=20)
plt.grid(True)

plt.tight_layout()
plt.show()

plt.close()
