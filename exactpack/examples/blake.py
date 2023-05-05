#
# Creates a plot for the default (LANL-standard)
# Spherical Blake problem in ExactPack.
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', size=14)

from exactpack.solvers.blake import Blake

# !!!!!!!!!!!!!!!!!!!! SI units ###ONLY### !!!!!!!!!!!!!!!!!!!!

# Set the grid and snapshot time
grid_dr = 0.002
dom_max = 1.0
npts = int(round(dom_max / grid_dr)) + 1
rmin = 0.1
rmax = dom_max                  # mesh limits: diff than domain.
radii = np.linspace(rmin, rmax, npts)
tsnap = 1.6e-4

# All elastic params defaulted, two other params set to default values.
blkslvr = Blake(cavity_radius=rmin, pressure_scale=1.0e6)  # solver
blksoln = blkslvr(radii, tsnap)                            # solution

# With dflt pressure_scale, these scale=scl and plt.ylim() values
# provide just enough space.
fig = plt.figure(figsize=(10, 7))
scl = 1.0e-6
blksoln.plot('pressure', scale=scl)
blksoln.plot('stress_dev_rr', scale=scl)
blksoln.plot('stress_dev_qq', scale=scl)
blksoln.plot('stress_diff', scale=scl)

plt.xlim(0.0, 1.0)
plt.ylim(-1.1, 1.6)
plt.title('Stresses')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 7))
scl = 1.0e5
blksoln.plot('strain_vol', scale=scl)
blksoln.plot('strain_rr', scale=scl)
blksoln.plot('strain_qq', scale=scl)
blksoln.plot('displacement', scale=scl)

plt.xlim(0.0, 1.0)
plt.ylim(-2.1, 1.1)
plt.title('Displacement and Strains')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.close()
