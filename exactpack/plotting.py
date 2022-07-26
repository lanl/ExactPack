from math import ceil, log10

import matplotlib
import matplotlib.pyplot as plt
from numpy import abs, amax

def plot(soln, name, label=None, scale=None, **kwargs):
    """Plot a solution.

    Used internally by :meth:`exactpack.base.ExactSolution.plot` to
    do plotting, see documentation there for syntax.
    """

    if scale=='auto':
        scale = 10**-ceil(log10(amax(abs(soln[name]))))
    
    if label==None:
        if scale==None:
            label = name
        else:
            label = r"{} ($\times {}$)".format(name, scale)

    distance = soln.dtype.names[0]

    if scale==None:
        plt.plot(soln[distance], soln[name], label=label, **kwargs)
    else:
        plt.plot(soln[distance], scale*soln[name], label=label, **kwargs)
        
    lsize =18
    plt.xlabel(distance,fontsize=lsize)
    plt.tick_params(axis='both',labelsize=lsize)
    plt.legend(loc='best')
