import os.path

from exactpack.analysis import *
from exactpack.analysis.readers import VTKReader, generic_VTK_name_mapping
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
# resolution.  

base = os.path.dirname(__file__)

study = Study([os.path.join(base, "data/shockTube_COARSE.vtk"),
               os.path.join(base, "data/shockTube_MED.vtk"),
               os.path.join(base, "data/shockTube_FINE.vtk"),],
              reference=Sod(pl=1e5, pr=1e4, interface_loc=0.0),
              study_parameters=[0.02, 0.01, 0.005],
              time=0.007,
              reader=VTKReader(name_mapping=generic_VTK_name_mapping)
              )

# Plot the pressure profiles
study.plot('pressure')
plt.title("Sod Shock Tube Profiles")
plt.show()

fiducials = { 'pressure': 1 }

fitstudy = FitConvergenceRate(study, fiducials=fiducials)
print(fitstudy)
fitstudy.plot('pressure')
plt.title("Sod Shock Tube Convergence: Best fit to exponential")
plt.xlabel("Zone size")
plt.show()

roachestudy = RoacheConvergenceRate(study, fiducials=fiducials)
print(roachestudy)
roachestudy.plot('pressure')
plt.title("Sod Shock Tube Convergence: Roache's Formula")
plt.xlabel("Zone size")
plt.show()

linregstudy = RegressionConvergenceRate(study, fiducials=fiducials)
print(linregstudy)
linregstudy.plot('pressure')
plt.title("Sod Shock Tube Convergence: Linear regression")
plt.xlabel("Zone size")
plt.show()
