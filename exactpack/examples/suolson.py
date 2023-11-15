import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', size=14)

# import ExactPack solver and analysis tools
from exactpack.solvers.suolson import SuOlson as mySuOlson

#####################################################################
rmax = 20.0
r = np.linspace(0.0, rmax, 1000)

solver = mySuOlson()

#####################################################################
t = 1.e-9
soln1 = solver(r,t)

t = 0.1e-9
soln2 = solver(r,t)

t = 0.01e-9
soln3 = solver(r,t)

fig = plt.figure(figsize=(10, 7))
soln1.plot('temperature_rad',label=r'$T_{\rm rad} ~ t=1\,{\rm ns}$')
soln1.plot('temperature_mat',label=r'$T_{\rm mat} ~ t=1\,{\rm ns}$')
soln2.plot('temperature_rad',label=r'$T_{\rm rad} ~ t=0.1\,{\rm ns}$')
soln2.plot('temperature_mat',label=r'$T_{\rm mat} ~ t=0.1\,{\rm ns}$')
soln3.plot('temperature_rad',label=r'$T_{\rm rad} ~ t=0.01\,{\rm ns}$')
soln3.plot('temperature_mat',label=r'$T_{\rm mat} ~ t=0.01\,{\rm ns}$')
plt.xlim(0.0,rmax)
plt.title('ExactPack solver class SuOlson')
plt.ylim(0,1000)
plt.legend(loc=0)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.close()
