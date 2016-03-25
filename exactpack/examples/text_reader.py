import os.path

from exactpack.analysis import CodeVerificationStudy, PointNorm
from exactpack.analysis.readers import TextReader
from exactpack.solvers.riemann import Sod
import matplotlib.pylab as plt


# The data used for this example was obtained using OpenFOAM.
# OpenFOAM (www.openfoam.org) is a freely available CFD suite,
# produced by the OpenFOAM Foundation (which is not affiliated with
# Los Alamos National Laboratory).
#
# This particular example is the case:
#   tutorials/compressible/sonicFoam/laminar/shockTube
# run at the default resolution, half the resolution, and double the
# resolution.  The data was then manually converted into a text file.

base = os.path.dirname(__file__)
study = CodeVerificationStudy([os.path.join(base, "data/coarse.dat"),
                               os.path.join(base, "data/medium.dat"),
                               os.path.join(base, "data/fine.dat"),],
                              Sod(pl=1e5, pr=1e4, interface_loc=0.0),
                              dxs=[0.02, 0.01, 0.005],
                              time=0.007,
                              reader=TextReader(names=['x_position',
                                                       'velocity_x',
                                                       'pressure',
                                                       'temperature']))

study.plot('velocity_x')
plt.show()

study.convergence('velocity_x', norm=PointNorm()).plot(fiducial=2.0/3.0)
plt.show()
