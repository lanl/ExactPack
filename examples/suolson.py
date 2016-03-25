import numpy as np
import matplotlib.pylab as plt
from matplotlib import rc, rcParams
#rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})

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


soln1.plot('Tradiation',label=r'$T_{\rm rad} ~ t=1\,{\rm ns}$')
soln1.plot('Tmaterial',label=r'$T_{\rm mat} ~ t=1\,{\rm ns}$')
soln2.plot('Tradiation',label=r'$T_{\rm rad} ~ t=0.1\,{\rm ns}$')
soln2.plot('Tmaterial',label=r'$T_{\rm mat} ~ t=0.1\,{\rm ns}$')
soln3.plot('Tradiation',label=r'$T_{\rm rad} ~ t=0.01\,{\rm ns}$')
soln3.plot('Tmaterial',label=r'$T_{\rm mat} ~ t=0.01\,{\rm ns}$')
plt.xlim(0.0,rmax)
plt.title('ExactPack solver class SuOlson')
plt.ylim(0,1000)
plt.legend(loc=0)
plt.grid(True)
plt.show()
