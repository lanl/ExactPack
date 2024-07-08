'''Example demonstrating Sedov solvers. Reproduces plots from Kamm & Timmes,
"On Efficient Generation of Numerically Robust Sedov Solutions," LA-UR-07-2849
'''


# import standard Python packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', size=14)

# import ExactPack solvers
from exactpack.solvers.sedov.sedov import Sedov

# pyplot default settings
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 16})
# rc('grid', c='0.5', ls='-', lw=0.5)

# set domain variables for plots
npts = 2001
rvec = np.linspace(0.0, 1.2, npts)
t = 1.0

#
# Figure 8: Standard test cases
#

solver_pla = Sedov(geometry=1, eblast=0.0673185,
                                    gamma=1.4, omega=0.)
solution_pla = solver_pla(r=rvec, t=t)

solver_cyl = Sedov(geometry=2, eblast=0.311357,
                                    gamma=1.4, omega=0.)
solution_cyl = solver_cyl(r=rvec, t=t)

solver_sph = Sedov(geometry=3, eblast=0.851072,
                                    gamma=1.4, omega=0.)
solution_sph = solver_sph(r=rvec, t=t)

fig = plt.figure(figsize=(10, 7))
plt.suptitle('''Sedov solutions for $\gamma=1.4$, standard cases.
    Compare to Fig. 8 from Kamm & Timmes 2007''')

plt.subplot(221)
solution_pla.plot('density')
solution_cyl.plot('density')
solution_sph.plot('density')
plt.xlim(0.0, 1.2)
plt.ylim(0.0, 6.5)
plt.xlabel('Position (cm)')
plt.ylabel('Density (g/cc)')
plt.grid(True)
L = plt.legend(loc='upper left', bbox_to_anchor=(0.25, 1.4), ncol=3,
               fancybox=True, shadow=True)
L.get_texts()[0].set_text('planar')
L.get_texts()[1].set_text('cylindrical')
L.get_texts()[2].set_text('spherical')

plt.subplot(222)
solution_pla.plot('velocity')
solution_cyl.plot('velocity')
solution_sph.plot('velocity')
plt.xlim(0.0, 1.2)
plt.ylim(0.0, 0.4)
plt.xlabel('Position (cm)')
plt.ylabel('Particle velocity (cm/s)')
plt.grid(True)
plt.gca().legend().set_visible(False)

plt.subplot(223)
solution_pla.plot('specific_internal_energy')
solution_cyl.plot('specific_internal_energy')
solution_sph.plot('specific_internal_energy')
plt.xlim(0.0, 1.2)
plt.ylim(1.e-2, 1.e5)
plt.xlabel('Position (cm)')
plt.ylabel('Specific internal energy (erg/g)')
plt.grid(True)
plt.gca().set_yscale('log', nonpositive='clip')
plt.gca().legend().set_visible(False)

plt.subplot(224)
solution_pla.plot('pressure')
solution_cyl.plot('pressure')
solution_sph.plot('pressure')
plt.xlim(0.0, 1.2)
plt.ylim(0.0, 0.15)
plt.xlabel('Position (cm)')
plt.ylabel('Pressure (erg/cc)')
plt.grid(True)
plt.gca().legend().set_visible(False)

plt.tight_layout()
fig.subplots_adjust(top=0.8)  # Makes room for suptitle
#plt.savefig('fig08.pdf')
plt.show()

#
# Figure 9: Singular test cases
#

solver_cyl = Sedov(geometry=2, eblast=2.45749,
                                    gamma=1.4, omega=1.66667)
solution_cyl = solver_cyl(r=rvec, t=t)

solver_sph = Sedov(geometry=3, eblast=4.90875,
                                    gamma=1.4, omega=2.33333)
solution_sph = solver_sph(r=rvec, t=t)

fig = plt.figure(figsize=(10, 7))
plt.suptitle('''Sedov solutions for $\gamma=1.4$, singular cases.
    Compare to Fig. 9 from Kamm & Timmes 2007''')

plt.subplot(221)
solution_cyl.plot('density')
solution_sph.plot('density')
plt.xlim(0.0, 1.2)
plt.ylim(0.0, 12.0)
plt.xlabel('Position (cm)')
plt.ylabel('Density (g/cc)')
plt.grid(True)
L = plt.legend(loc='upper left', bbox_to_anchor=(0.25, 1.4), ncol=2,
               fancybox=True, shadow=True)
L.get_texts()[0].set_text('cylindrical')
L.get_texts()[1].set_text('spherical')

plt.subplot(222)
solution_cyl.plot('velocity')
solution_sph.plot('velocity')
plt.xlim(0.0, 1.2)
plt.ylim(0.0, 0.8)
plt.xlabel('Position (cm)')
plt.ylabel('Particle velocity (cm/s)')
plt.grid(True)
plt.gca().legend().set_visible(False)

plt.subplot(223)
solution_cyl.plot('specific_internal_energy')
solution_sph.plot('specific_internal_energy')
plt.xlim(0.0, 1.2)
plt.ylim(1.e-5, 1.e0)
plt.xlabel('Position (cm)')
plt.ylabel('Specific internal energy (erg/g)')
plt.grid(True)
plt.gca().set_yscale('log', nonpositive='clip')
plt.gca().legend().set_visible(False)

plt.subplot(224)
solution_cyl.plot('pressure')
solution_sph.plot('pressure')
plt.xlim(0.0, 1.2)
plt.ylim(0.0, 0.7)
plt.xlabel('Position (cm)')
plt.ylabel('Pressure (erg/cc)')
plt.grid(True)
plt.gca().legend().set_visible(False)

plt.tight_layout()
fig.subplots_adjust(top=0.8)  # Makes room for suptitle
#plt.savefig('fig09.pdf')
plt.show()

#
# Figure 10: Vacuum test cases
#

solver_cyl = Sedov(geometry=2, eblast=2.67315,
                                    gamma=1.4, omega=1.7)
solution_cyl = solver_cyl(r=rvec, t=t)

solver_sph = Sedov(geometry=3, eblast=5.45670,
                                    gamma=1.4, omega=2.4)
solution_sph = solver_sph(r=rvec, t=t)

fig = plt.figure(figsize=(10, 7))
plt.suptitle('''Sedov solutions for $\gamma=1.4$, vacuum cases.
    Compare to Fig. 10 from Kamm & Timmes 2007''')

plt.subplot(221)
solution_cyl.plot('density')
solution_sph.plot('density')
plt.xlim(0.0, 1.2)
plt.ylim(0.0, 20.0)
plt.xlabel('Position (cm)')
plt.ylabel('Density (g/cc)')
plt.grid(True)
L = plt.legend(loc='upper left', bbox_to_anchor=(0.25, 1.4), ncol=2,
               fancybox=True, shadow=True)
L.get_texts()[0].set_text('cylindrical')
L.get_texts()[1].set_text('spherical')

plt.subplot(222)
solution_cyl.plot('velocity')
solution_sph.plot('velocity')
plt.xlim(0.0, 1.2)
plt.ylim(0.0, 0.8)
plt.xlabel('Position (cm)')
plt.ylabel('Particle velocity (cm/s)')
plt.grid(True)
plt.gca().legend().set_visible(False)

plt.subplot(223)
solution_cyl.plot('specific_internal_energy')
solution_sph.plot('specific_internal_energy')
plt.xlim(0.0, 1.2)
plt.ylim(1.e-5, 1.e0)
plt.xlabel('Position (cm)')
plt.ylabel('Specific internal energy (erg/g)')
plt.grid(True)
plt.gca().set_yscale('log', nonpositive='clip')
plt.gca().legend().set_visible(False)

plt.subplot(224)
solution_cyl.plot('pressure')
solution_sph.plot('pressure')
plt.xlim(0.0, 1.2)
plt.ylim(0.0, 0.7)
plt.xlabel('Position (cm)')
plt.ylabel('Pressure (erg/cc)')
plt.grid(True)
plt.gca().legend().set_visible(False)

plt.tight_layout()
fig.subplots_adjust(top=0.8)  # Makes room for suptitle
#plt.savefig('fig10.pdf')

plt.show()

plt.close()
