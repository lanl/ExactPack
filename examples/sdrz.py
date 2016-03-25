#  
#  Creates plots for the steady detonation reaction zone (SDRZ) problem
#  in ExactPack
#

from exactpack.solvers.sdrz import SteadyDetonationReactionZone
import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='serif', size=12)

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

plt.figure('Time history of tracer particle that starts in the shock', figsize=(12,8))

plt.subplot(2,2,1)
plt.plot(tvec, result['density'],'k')
plt.ylabel(r'density [$\rm{g/cm^3}$]')

plt.subplot(2,2,2)
plt.plot(tvec, result['pressure'],'k')
plt.ylabel(r'pressure [$\rm{Mbar}$]')

plt.subplot(2,2,3)
plt.plot(tvec, result['velocity'],'k')
plt.ylabel(r'particle velocity [$\rm{cm/\mu s}$]')
plt.xlabel(r'time [$\rm{\mu s}$]', labelpad=20)

plt.subplot(2,2,4)
plt.plot(tvec, result['reaction_progress'],'k')
plt.ylim([0,1.05])
plt.ylabel(r'reaction progress, $\rm \lambda$', labelpad=30)
plt.xlabel(r'time [$\rm{\mu s}$]', labelpad=20)

plt.tight_layout()

#  Plot the solution snapshot at final time at vector xvec


xvec = np.linspace(0., xmax,  NPx)

result = solution._run(xvec, tfinal)

plt.figure('Solution snapshot at t={0} us'.format(tfinal), figsize=(12,8))

plt.subplot(2,2,1)
plt.plot(result['position'], result['density'],'k')
plt.ylabel(r'density [$\rm{g/cm^3}$]')

plt.subplot(2,2,2)
plt.plot(result['position'], result['pressure'],'k')
plt.ylabel(r'pressure [$\rm{Mbar}$]')

plt.subplot(2,2,3)
plt.plot(result['position'], result['velocity'],'k')
plt.ylabel(r'particle velocity [$\rm{cm/\mu s}$]')
plt.xlabel(r'absolute position [$\rm{cm}$]', labelpad=20)

plt.subplot(2,2,4)
plt.plot(result['position'], result['reaction_progress'],'k')
plt.ylim([0,1.05])
plt.ylabel(r'reaction progress, $\rm \lambda$', labelpad=30)
plt.xlabel(r'absolute position [$\rm{cm}$]', labelpad=20)

plt.tight_layout()

plt.show()
