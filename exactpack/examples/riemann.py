# Example demonstrating Riemann solvers for both ideal-gas EOS and generalized EOS in the form of JWL.
# For the ideal-gas EOS, reproduces plots from Kamm, et. als., 2014 LA-UR-14-20418 "Standardized Definitions for Code Verification Test Problems".
# There are a standard set of six Riemann problems for an ideal-gas EOS.
# These are defined this 2014 report.
# For the generalized EOS (JWL in this case), reproduces plots from Kamm's 2015 LA-UR-15-21616 "Exact Compressible One Dimensional Riemann Solver for General Convex Equations of State".
# This 2015 report defines two additional ideal-gas EOS Riemann problems, and two generalized EOS Riemann problems for a JWL EOS.
# One of the ideal-gas EOS problems is for left & right states with different values of the adiabatic index, :math:`\gamma`.
# This makes for a total of 10 Riemann problems defined in these two documents.
# These same 10 are also defined below and run as part of this example script.

# import standard python packages
from matplotlib import pyplot as plt
from matplotlib import rc

from numpy import linspace

# import ExactPack solvers
from exactpack.solvers.riemann.ep_riemann import IGEOS_Solver, GenEOS_Solver

xvec = linspace(0., 1., int(1e5))
t_final = 0.25

# Define the 10 Riemann problems 
# The Sod shocktube problem solved using the ideal gas EOS (IGEOS) solver.
riem1_ig_soln = IGEOS_Solver(rl=1.0,   ul=0.,   pl=1.0,  gl=1.4,
                             rr=0.125, ur=0.,   pr=0.1,  gr=1.4,
                             xmin=0.,  xd0=0.5, xmax=1., t=0.25)

riem1_ig_result = riem1_ig_soln._run(xvec, t_final)

# The Sod shocktube problem solved using the generalized EOS (GenEOS) solver.
riem1_gen_soln = GenEOS_Solver(rl=1.0,   ul=0.,   pl=1.0,  gl=1.4,
                               rr=0.125, ur=0.,   pr=0.1,  gr=1.4,
                               xmin=0.,  xd0=0.5, xmax=1., t=0.25)

riem1_gen_result = riem1_gen_soln(xvec, t_final)


# The Sod shocktube flipping the left & right states, using the IGEOS solver.
# This problem is probably more relevant for checking that the analytic solution
# works correctly.
riem1Rev_ig_soln = IGEOS_Solver(rr=1.0,   ur=0.,   pr=1.0,  gr=1.4,
                                rl=0.125, ul=0.,   pl=0.1,  gl=1.4,
                                xmin=0.,  xd0=0.5, xmax=1., t=0.25)

riem1Rev_ig_result = riem1Rev_ig_soln(xvec, t_final)


# The Sod shocktube flipping the left & right states, using the GenEOS solver.
# This problem is probably more relevant for checking that the semi-analytic
# integrator works correctly.
riem1Rev_gen_soln = GenEOS_Solver(rr=1.0,   ur=0.,   pr=1.0,  gr=1.4,
                                  rl=0.125, ul=0.,   pl=0.1,  gl=1.4,
                                  xmin=0.,  xd0=0.5, xmax=1., t=0.25)

riem1Rev_gen_result = riem1Rev_gen_soln(xvec, t_final)


t_final = 0.15
# The Einfeldt problem solved using the IGEOS solver.
riem2_ig_soln = IGEOS_Solver(rl=1.,   ul=-2.0, pl=0.4,  gl=1.4,
                             rr=1.,   ur= 2.0, pr=0.4,  gr=1.4,
                             xmin=0., xd0=0.5, xmax=1., t=0.15)

riem2_ig_result = riem2_ig_soln(xvec, t_final)


# The reverse Einfeldt problem solved using the IGEOS solver.
riem2Rev_ig_soln = IGEOS_Solver(rr=1.,   ur=2.0,  pr=0.4,  gr=1.4,
                                rl=1.,   ul=-2.0, pl=0.4,  gl=1.4,
                                xmin=0., xd0=0.5, xmax=1., t=0.15)

riem2Rev_ig_result = riem2Rev_ig_soln(xvec, t_final)


# The Einfeldt problem solved using the GenEOS solver.
riem2_gen_soln = GenEOS_Solver(rl=1.,   ul=-2.0, pl=0.4,  gl=1.4,
                               rr=1.,   ur= 2.0, pr=0.4,  gr=1.4,
                               xmin=0., xd0=0.5, xmax=1., t=0.15)

riem2_gen_result = riem2_gen_soln(xvec, t_final)


# The reverse Einfeldt problem solved using the GenEOS solver.
riem2Rev_gen_soln = GenEOS_Solver(rr=1.,   ur= 2.0, pr=0.4,  gr=1.4,
                                  rl=1.,   ul=-2.0, pl=0.4,  gl=1.4,
                                  xmin=0., xd0=0.5, xmax=1., t=0.15)

riem2Rev_gen_result = riem2Rev_gen_soln(xvec, t_final)


t_final = 0.012
# The Stationary Contact problem solved using the IGEOS solver.
riem3_ig_soln=IGEOS_Solver(rl=1.,   ul=-19.59745, pl=1000.0,  gl=1.4,
                           rr=1.,   ur=-19.59745, pr=   0.01, gr=1.4,
                           xmin=0., xd0=0.8,      xmax=1.,    t=0.012)

riem3_ig_result = riem3_ig_soln(xvec, t_final)


# The reverse Stationary Contact problem solved using the IGEOS solver.
riem3Rev_ig_soln=IGEOS_Solver(rr=1.,   ur=19.59745, pr=1000.0,  gr=1.4,
                              rl=1.,   ul=19.59745, pl=   0.01, gl=1.4,
                              xmin=0., xd0=0.2,     xmax=1.,    t=0.012)

riem3Rev_ig_result = riem3Rev_ig_soln(xvec, t_final)


# The Stationary Contact problem solved using the GenEOS solver.
riem3_gen_soln = GenEOS_Solver(rl=1.,   ul=-19.59745, pl=1000.0,  gl=1.4,
                               rr=1.,   ur=-19.59745, pr=   0.01, gr=1.4,
                               xmin=0., xd0=0.8,      xmax=1.,    t=0.012)

riem3_gen_result = riem3_gen_soln(xvec, t_final)


# The reverse Stationary Contact problem solved using the GenEOS solver.
riem3Rev_gen_soln = GenEOS_Solver(rr=1.,   ur=19.59745, pr=1000.0, gr=1.4,
                                  rl=1.,   ul=19.59745, pl=  0.01, gl=1.4,
                                  xmin=0., xd0=0.2,     xmax=1.,   t=0.012)

riem3Rev_gen_result = riem3Rev_gen_soln(xvec, t_final)


t_final = 1.
# The Slow Shock problem solved using the IGEOS solver.
riem4_ig_soln = IGEOS_Solver(rl=3.857143, ul=-0.810631, pl=31./3., gl=1.4,
                             rr=1.,       ur=-3.44,     pr=1.,     gr=1.4,
                             xmin=0.,     xd0=0.5,      xmax=1.,   t=1.)

riem4_ig_result = riem4_ig_soln(xvec, t_final)


# The reverse Slow Shock problem solved using the IGEOS solver.
riem4Rev_ig_soln = IGEOS_Solver(rr=3.857143, ur=0.810631, pr=31./3., gr=1.4,
                                rl=1.,       ul=3.44,     pl=1.,     gl=1.4,
                                xmin=0.,     xd0=0.5,     xmax=1.,   t=1.)

riem4Rev_ig_result = riem4Rev_ig_soln(xvec, t_final)


# The Slow Shock problem solved using the GenEOS solver.
riem4_gen_soln = GenEOS_Solver(rl=3.857143, ul=-0.810631, pl=31./3., gl=1.4,
                               rr=1.,       ur=-3.44,     pr=1.,     gr=1.4,
                               xmin=0.,     xd0=0.5,      xmax=1.,   t=1.)

riem4_gen_result = riem4_gen_soln(xvec, t_final)


# The reverse Slow Shock problem solved using the GenEOS solver.
riem4Rev_gen_soln = GenEOS_Solver(rr=3.857143, ur=0.810631, pr=31./3., gr=1.4,
                                  rl=1.,       ul=3.44,     pl=1.,     gl=1.4,
                                  xmin=0.,     xd0=0.5,     xmax=1.,   t=1.)

riem4Rev_gen_result = riem4Rev_gen_soln(xvec, t_final)


t_final = 0.3
# The Shock Contact Shock problem solved using the IGEOS solver.
riem5_ig_soln = IGEOS_Solver(rl=1.0,  ul= 0.5, pl=1.,   gl=1.4,
                             rr=1.25, ur=-0.5, pr=1.,   gr=1.4,
                             xmin=0., xd0=0.5, xmax=1., t=0.3)

riem5_ig_result = riem5_ig_soln(xvec, t_final)


# The reversed Shock Contact Shock problem solved using the IGEOS solver.
riem5Rev_ig_soln = IGEOS_Solver(rr=1.0,  ur=-0.5, pr=1.,   gr=1.4,
                                rl=1.25, ul= 0.5, pl=1.,   gl=1.4,
                                xmin=0., xd0=0.5, xmax=1., t=0.3)

riem5Rev_ig_result = riem5Rev_ig_soln(xvec, t_final)


# The Shock Contact Shock problem solved using the GenEOS solver.
riem5_gen_soln = GenEOS_Solver(rl=1.0,  ul= 0.5, pl=1.,   gl=1.4,
                               rr=1.25, ur=-0.5, pr=1.,   gr=1.4,
                               xmin=0., xd0=0.5, xmax=1., t=0.3)

riem5_gen_result = riem5_gen_soln(xvec, t_final)


# The reversed Shock Contact Shock problem solved using the GenEOS solver.
riem5Rev_gen_soln = GenEOS_Solver(rr=1.0,  ur=-0.5, pr=1.,   gr=1.4,
                                  rl=1.25, ul= 0.5, pl=1.,   gl=1.4,
                                  xmin=0., xd0=0.5, xmax=1., t=0.3)

riem5Rev_gen_result = riem5Rev_gen_soln(xvec, t_final)


t_final = 0.5
# The LeBlanc problem solved using the IGEOS solver.
riem6_ig_soln = IGEOS_Solver(rl=1.0,  ul=0.,   pl=1./15.,     gl=5./3.,
                             rr=0.01, ur=0.,   pr=2./(3.e10), gr=5./3.,
                             xmin=0., xd0=0.3, xmax=1.,       t=0.5)

riem6_ig_result = riem6_ig_soln(xvec, t_final)


# The reversed LeBlanc problem solved using the IGEOS solver.
riem6Rev_ig_soln = IGEOS_Solver(rr=1.0,  ur=0.,   pr=1./15.,     gr=5./3.,
                                rl=0.01, ul=0.,   pl=2./(3.e10), gl=5./3.,
                                xmin=0., xd0=0.7, xmax=1.,       t=0.5)

riem6Rev_ig_result = riem6Rev_ig_soln(xvec, t_final)


# The LeBlanc problem solved using the GenEOS solver.
riem6_gen_soln = GenEOS_Solver(rl=1.0,  ul=0.,   pl=1./15.,     gl=5./3.,
                               rr=0.01, ur=0.,   pr=2./(3.e10), gr=5./3.,
                               xmin=0., xd0=0.3, xmax=1.,       t=0.5)
 
riem6_gen_result = riem6_gen_soln(xvec, t_final)


# The reversed LeBlanc problem solved using the GenEOS solver.
riem6Rev_gen_soln = GenEOS_Solver(rr=1.0,  ur=0.,   pr=1./15.,      gr=5./3.,
                                  rl=0.01, ul=0.,   pl=2./(3.e10) , gl=5./3.,
                                  xmin=0., xd0=0.7, xmax=1.,        t=0.5)

riem6Rev_gen_result = riem6Rev_gen_soln(xvec, t_final)


t_final = 0.2
# The Sod shocktube with different gammas for the left & right states, using the
# IGEOS solver.
riem7_ig_soln = IGEOS_Solver(rl=1.0,   ul=0.,   pl=2.0,  gl=2.0,
                                rr=0.125, ur=0.,   pr=0.1,  gr=1.4,
                                xmin=0.,  xd0=0.5, xmax=1., t=0.2)

riem7_ig_result = riem7_ig_soln(xvec, t_final)


# The Sod shocktube with different gammas for the left & right states, using the
# GenEOS solver.
riem7_gen_soln = GenEOS_Solver(rl=1.0,    ul=0.,  pl=2.0,  gl=2.0,
                                  rr=0.125,  ur=0.,  pr=0.1,  gr=1.4,
                                  xmin=0.,  xd0=0.5, xmax=1., t=0.2)

riem7_gen_result = riem7_gen_soln(xvec, t_final)


# The reversed Sod shocktube with different gammas for the left & right states,
# using the IGEOS solver.
riem7Rev_ig_soln = IGEOS_Solver(rr=1.0,   ur=0.,   pr=2.0,  gr=2.0,
                                   rl=0.125, ul=0.,   pl=0.1,  gl=1.4,
                                   xmin=0.,  xd0=0.5, xmax=1., t=0.2)

riem7Rev_ig_result = riem7Rev_ig_soln(xvec, t_final)


# The reversed Sod shocktube with different gammas for the left & right states,
# using the GenEOS solver.
riem7Rev_gen_soln=GenEOS_Solver(rr=1.0,   ur=0.,   pr=2.0,  gr=2.0,
                                   rl=0.125, ul=0.,   pl=0.1,  gl=1.4,
                                   xmin=0.,  xd0=0.5, xmax=1., t=0.2)

riem7Rev_gen_result = riem7Rev_gen_soln(xvec, t_final)


t_final = 20.
xvec = linspace(0., 100., int(1e5))
# The Lee JWL shocktube problem solved using the GenEOS solver.
riem_Lee_soln = GenEOS_Solver(rl=0.9525, ul=0.,   pl=1.,     gl=1.8938,
                              rr=3.81,   ur=0.,   pr=2.,     gr=1.8938,
                              xmin=0.,   xd0=50., xmax=100., t=20.,
                              A=632.1,   B=-0.04472,
                              R1=11.3,   R2=1.13,
                              r0=1.905,  e0=0.,
                              problem='JWL')

riem_Lee_result = riem_Lee_soln(xvec, t_final)


# The Lee JWL shocktube problem reversed solved using the GenEOS solver.
riem_LeeRev_soln = GenEOS_Solver(rr=0.9525, ur=0.,   pr=1.,     gr=1.8938,
                                 rl=3.81,   ul=0.,   pl=2.,     gl=1.8938,
                                 xmin=0.,   xd0=50., xmax=100., t=20.,
                                 A=632.1,   B=-0.04472,
                                 R1=11.3,   R2=1.13,
                                 r0=1.905,  e0=0.,
                                 problem='JWL')

riem_LeeRev_result = riem_LeeRev_soln(xvec, t_final)


t_final = 12.
# The Shyue JWL shocktube problem solved using the GenEOS solver.
riem_Shyue_soln = GenEOS_Solver(rl=1.7,  ul=0.,   pl=10.0,   gl=1.25,
                                rr=1.0,  ur=0.,   pr= 0.5,   gr=1.25,
                                xmin=0., xd0=50., xmax=100., t=12.,
                                A=8.545, B=0.205,
                                R1=4.6,  R2=1.35,
                                r0=1.84, e0=0.,
                                problem='JWL')

riem_Shyue_result = riem_Shyue_soln(xvec, t_final)


# The Shyue JWL shocktube problem reversed solved using the GenEOS solver.
riem_ShyueRev_soln = GenEOS_Solver(rr=1.7,  ur=0.,   pr=10.0,   gr=1.25,
                                   rl=1.0,  ul=0.,   pl= 0.5,   gl=1.25,
                                   xmin=0., xd0=50., xmax=100., t=12.,
                                   A=8.545, B=0.205,
                                   R1=4.6,  R2=1.35,
                                   r0=1.84, e0=0.,
                                   problem='JWL')

riem_ShyueRev_result = riem_ShyueRev_soln(xvec, t_final)


fig = plt.figure(figsize=(10,7))
x1_ig = riem1_ig_result['position']
u1_ig = riem1_ig_result['velocity']
p1_ig = riem1_ig_result['pressure']
r1_ig = riem1_ig_result['density']
e1_ig = riem1_ig_result['sie']
plt.plot(x1_ig, u1_ig, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x1_ig, p1_ig, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x1_ig, r1_ig, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x1_ig, e1_ig, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-0.5, 3.))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 1: Sod using the IGEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x1_gen = riem1_gen_result['position']
u1_gen = riem1_gen_result['velocity']
p1_gen = riem1_gen_result['pressure']
r1_gen = riem1_gen_result['density']
e1_gen = riem1_gen_result['sie']
plt.plot(x1_gen, u1_gen, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x1_gen, p1_gen, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x1_gen, r1_gen, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x1_gen, e1_gen, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-0.5, 3.))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 1: Sod using the GenEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x1Rev_ig = riem1Rev_ig_result['position']
u1Rev_ig = riem1Rev_ig_result['velocity']
p1Rev_ig = riem1Rev_ig_result['pressure']
r1Rev_ig = riem1Rev_ig_result['density']
e1Rev_ig = riem1Rev_ig_result['sie']
plt.plot(x1Rev_ig, u1Rev_ig, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x1Rev_ig, p1Rev_ig, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x1Rev_ig, r1Rev_ig, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x1Rev_ig, e1Rev_ig, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-1.1, 3.))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 1 reversed: Sod reversed using the IGEOS solver'
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x1Rev_gen = riem1Rev_gen_result['position']
u1Rev_gen = riem1Rev_gen_result['velocity']
p1Rev_gen = riem1Rev_gen_result['pressure']
r1Rev_gen = riem1Rev_gen_result['density']
e1Rev_gen = riem1Rev_gen_result['sie']
plt.plot(x1Rev_gen, u1Rev_gen, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x1Rev_gen, p1Rev_gen, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x1Rev_gen, r1Rev_gen, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x1Rev_gen, e1Rev_gen, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-1.1, 3.))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 1 reversed: Sod reversed using the GenEOS solver'
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x2_ig = riem2_ig_result['position']
u2_ig = riem2_ig_result['velocity']
p2_ig = riem2_ig_result['pressure']
r2_ig = riem2_ig_result['density']
e2_ig = riem2_ig_result['sie']
plt.plot(x2_ig, u2_ig, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x2_ig, p2_ig, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x2_ig, r2_ig, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x2_ig, e2_ig, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-3., 3.))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 2: Einfeldt - using the IGEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x2_gen = riem2_gen_result['position']
u2_gen = riem2_gen_result['velocity']
p2_gen = riem2_gen_result['pressure']
r2_gen = riem2_gen_result['density']
e2_gen = riem2_gen_result['sie']
plt.plot(x2_gen, u2_gen, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x2_gen, p2_gen, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x2_gen, r2_gen, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x2_gen, e2_gen, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-3., 3.))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 2: Einfeldt - using the GenEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x2Rev_ig = riem2Rev_ig_result['position']
u2Rev_ig = riem2Rev_ig_result['velocity']
p2Rev_ig = riem2Rev_ig_result['pressure']
r2Rev_ig = riem2Rev_ig_result['density']
e2Rev_ig = riem2Rev_ig_result['sie']
plt.plot(x2Rev_ig, u2Rev_ig, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x2Rev_ig, p2Rev_ig, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x2Rev_ig, r2Rev_ig, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x2Rev_ig, e2Rev_ig, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-3., 3.))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 2: reversed Einfeldt - using the IGEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x2Rev_gen = riem2Rev_gen_result['position']
u2Rev_gen = riem2Rev_gen_result['velocity']
p2Rev_gen = riem2Rev_gen_result['pressure']
r2Rev_gen = riem2Rev_gen_result['density']
e2Rev_gen = riem2Rev_gen_result['sie']
plt.plot(x2Rev_gen, u2Rev_gen, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x2Rev_gen, p2Rev_gen, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x2Rev_gen, r2Rev_gen, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x2Rev_gen, e2Rev_gen, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-3., 3.))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 2: reversed Einfeldt - using the GenEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x3_ig = riem3_ig_result['position']
u3_ig = riem3_ig_result['velocity']
p3_ig = riem3_ig_result['pressure']
r3_ig = riem3_ig_result['density']
e3_ig = riem3_ig_result['sie']
plt.plot(x3_ig, u3_ig, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x3_ig, p3_ig, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x3_ig, r3_ig, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x3_ig, e3_ig, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-500., 3000.))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 3: Stationary Contact - using the IGEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x3_gen = riem3_gen_result['position']
u3_gen = riem3_gen_result['velocity']
p3_gen = riem3_gen_result['pressure']
r3_gen = riem3_gen_result['density']
e3_gen = riem3_gen_result['sie']
plt.plot(x3_gen, u3_gen, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x3_gen, p3_gen, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x3_gen, r3_gen, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x3_gen, e3_gen, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-500., 3000.))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 3: Stationary Contact - using the GenEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x3Rev_ig = riem3Rev_ig_result['position']
u3Rev_ig = riem3Rev_ig_result['velocity']
p3Rev_ig = riem3Rev_ig_result['pressure']
r3Rev_ig = riem3Rev_ig_result['density']
e3Rev_ig = riem3Rev_ig_result['sie']
plt.plot(x3Rev_ig, u3Rev_ig, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x3Rev_ig, p3Rev_ig, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x3Rev_ig, r3Rev_ig, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x3Rev_ig, e3Rev_ig, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-500., 3000.))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 3: reversed Stationary Contact - using the IGEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x3Rev_gen = riem3Rev_gen_result['position']
u3Rev_gen = riem3Rev_gen_result['velocity']
p3Rev_gen = riem3Rev_gen_result['pressure']
r3Rev_gen = riem3Rev_gen_result['density']
e3Rev_gen = riem3Rev_gen_result['sie']
plt.plot(x3Rev_gen, u3Rev_gen, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x3Rev_gen, p3Rev_gen, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x3Rev_gen, r3Rev_gen, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x3Rev_gen, e3Rev_gen, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-500., 3000.))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 3: reversed Stationary Contact - using the GenEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x4_ig = riem4_ig_result['position']
u4_ig = riem4_ig_result['velocity']
p4_ig = riem4_ig_result['pressure']
r4_ig = riem4_ig_result['density']
e4_ig = riem4_ig_result['sie']
plt.plot(x4_ig, u4_ig, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x4_ig, p4_ig, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x4_ig, r4_ig, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x4_ig, e4_ig, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-4., 12.))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 4: Slow Shock - using the IGEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x4_gen = riem4_gen_result['position']
u4_gen = riem4_gen_result['velocity']
p4_gen = riem4_gen_result['pressure']
r4_gen = riem4_gen_result['density']
e4_gen = riem4_gen_result['sie']
plt.plot(x4_gen, u4_gen, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x4_gen, p4_gen, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x4_gen, r4_gen, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x4_gen, e4_gen, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-4., 12.))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 4: Slow Shock - using the GenEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x4Rev_ig = riem4Rev_ig_result['position']
u4Rev_ig = riem4Rev_ig_result['velocity']
p4Rev_ig = riem4Rev_ig_result['pressure']
r4Rev_ig = riem4Rev_ig_result['density']
e4Rev_ig = riem4Rev_ig_result['sie']
plt.plot(x4Rev_ig, u4Rev_ig, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x4Rev_ig, p4Rev_ig, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x4Rev_ig, r4Rev_ig, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x4Rev_ig, e4Rev_ig, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-4., 12.))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 4: reversed Slow Shock - using the IGEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x4Rev_gen = riem4Rev_gen_result['position']
u4Rev_gen = riem4Rev_gen_result['velocity']
p4Rev_gen = riem4Rev_gen_result['pressure']
r4Rev_gen = riem4Rev_gen_result['density']
e4Rev_gen = riem4Rev_gen_result['sie']
plt.plot(x4Rev_gen, u4Rev_gen, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x4Rev_gen, p4Rev_gen, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x4Rev_gen, r4Rev_gen, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x4Rev_gen, e4Rev_gen, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-4., 12.))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 4: reversed Slow Shock - using the GenEOS solver'
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x5_ig = riem5_ig_result['position']
u5_ig = riem5_ig_result['velocity']
p5_ig = riem5_ig_result['pressure']
r5_ig = riem5_ig_result['density']
e5_ig = riem5_ig_result['sie']
plt.plot(x5_ig, u5_ig, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x5_ig, p5_ig, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x5_ig, r5_ig, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x5_ig, e5_ig, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-1., 3.5))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 5: Shock Contact Shock - using the IGEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x5_gen = riem5_gen_result['position']
u5_gen = riem5_gen_result['velocity']
p5_gen = riem5_gen_result['pressure']
r5_gen = riem5_gen_result['density']
e5_gen = riem5_gen_result['sie']
plt.plot(x5_gen, u5_gen, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x5_gen, p5_gen, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x5_gen, r5_gen, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x5_gen, e5_gen, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-1., 3.5))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 5: Shock Contact Shock - using the GenEOS solver'
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x5Rev_ig = riem5Rev_ig_result['position']
u5Rev_ig = riem5Rev_ig_result['velocity']
p5Rev_ig = riem5Rev_ig_result['pressure']
r5Rev_ig = riem5Rev_ig_result['density']
e5Rev_ig = riem5Rev_ig_result['sie']
plt.plot(x5Rev_ig, u5Rev_ig, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x5Rev_ig, p5Rev_ig, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x5Rev_ig, r5Rev_ig, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x5Rev_ig, e5Rev_ig, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-1., 3.5))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 5: reversed Shock Contact Shock - using the IGEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x5Rev_gen = riem5Rev_gen_result['position']
u5Rev_gen = riem5Rev_gen_result['velocity']
p5Rev_gen = riem5Rev_gen_result['pressure']
r5Rev_gen = riem5Rev_gen_result['density']
e5Rev_gen = riem5Rev_gen_result['sie']
plt.plot(x5Rev_gen, u5Rev_gen, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x5Rev_gen, p5Rev_gen, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x5Rev_gen, r5Rev_gen, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x5Rev_gen, e5Rev_gen, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-1., 3.5))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 5: reversed Shock Contact Shock - using the IGEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x5Rev_gen = riem5Rev_gen_result['position']
u5Rev_gen = riem5Rev_gen_result['velocity']
p5Rev_gen = riem5Rev_gen_result['pressure']
r5Rev_gen = riem5Rev_gen_result['density']
e5Rev_gen = riem5Rev_gen_result['sie']
plt.plot(x5Rev_gen, u5Rev_gen, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x5Rev_gen, p5Rev_gen, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x5Rev_gen, r5Rev_gen, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x5Rev_gen, e5Rev_gen, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-1., 3.5))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 5: reversed Shock Contact Shock - using the IGEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x5Rev_gen = riem5Rev_gen_result['position']
u5Rev_gen = riem5Rev_gen_result['velocity']
p5Rev_gen = riem5Rev_gen_result['pressure']
r5Rev_gen = riem5Rev_gen_result['density']
e5Rev_gen = riem5Rev_gen_result['sie']
plt.plot(x5Rev_gen, u5Rev_gen, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x5Rev_gen, p5Rev_gen, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x5Rev_gen, r5Rev_gen, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x5Rev_gen, e5Rev_gen, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-1., 3.5))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 5: reversed Shock Contact Shock - using the GenEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x6_ig = riem6_ig_result['position']
u6_ig = riem6_ig_result['velocity']
p6_ig = riem6_ig_result['pressure']
r6_ig = riem6_ig_result['density']
e6_ig = riem6_ig_result['sie']
plt.plot(x6_ig, u6_ig, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x6_ig, p6_ig, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x6_ig, r6_ig, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x6_ig, e6_ig, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-0.2, 1.2))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 6: LeBlanc - using the IGEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x6_gen = riem6_gen_result['position']
u6_gen = riem6_gen_result['velocity']
p6_gen = riem6_gen_result['pressure']
r6_gen = riem6_gen_result['density']
e6_gen = riem6_gen_result['sie']
plt.plot(x6_gen, u6_gen, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x6_gen, p6_gen, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x6_gen, r6_gen, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x6_gen, e6_gen, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-0.2, 1.2))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 6: LeBlanc - using the GenEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x6Rev_ig = riem6Rev_ig_result['position']
u6Rev_ig = riem6Rev_ig_result['velocity']
p6Rev_ig = riem6Rev_ig_result['pressure']
r6Rev_ig = riem6Rev_ig_result['density']
e6Rev_ig = riem6Rev_ig_result['sie']
plt.plot(x6Rev_ig, u6Rev_ig, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x6Rev_ig, p6Rev_ig, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x6Rev_ig, r6Rev_ig, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x6Rev_ig, e6Rev_ig, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-0.2, 1.2))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 6: reversed LeBlanc - using the IGEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x6Rev_gen = riem6Rev_gen_result['position']
u6Rev_gen = riem6Rev_gen_result['velocity']
p6Rev_gen = riem6Rev_gen_result['pressure']
r6Rev_gen = riem6Rev_gen_result['density']
e6Rev_gen = riem6Rev_gen_result['sie']
plt.plot(x6Rev_gen, u6Rev_gen, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x6Rev_gen, p6Rev_gen, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x6Rev_gen, r6Rev_gen, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x6Rev_gen, e6Rev_gen, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-0.2, 1.2))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = 'Riemann Problem 6: reversed LeBlanc - using the GenEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x7_ig = riem7_ig_result['position']
u7_ig = riem7_ig_result['velocity']
p7_ig = riem7_ig_result['pressure']
r7_ig = riem7_ig_result['density']
e7_ig = riem7_ig_result['sie']
plt.plot(x7_ig, u7_ig, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x7_ig, p7_ig, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x7_ig, r7_ig, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x7_ig, e7_ig, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-0.2, 3.5))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = r'Riemann Problem 7: Sod w/ $\gamma_L \neq \gamma_R$ using the IGEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x7_gen = riem7_gen_result['position']
u7_gen = riem7_gen_result['velocity']
p7_gen = riem7_gen_result['pressure']
r7_gen = riem7_gen_result['density']
e7_gen = riem7_gen_result['sie']
plt.plot(x7_gen, u7_gen, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x7_gen, p7_gen, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x7_gen, r7_gen, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x7_gen, e7_gen, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-0.2, 3.5))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = r'Riemann Problem 7: Sod w/ $\gamma_L \neq \gamma_R$ using the GenEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x7Rev_ig = riem7Rev_ig_result['position']
u7Rev_ig = riem7Rev_ig_result['velocity']
p7Rev_ig = riem7Rev_ig_result['pressure']
r7Rev_ig = riem7Rev_ig_result['density']
e7Rev_ig = riem7Rev_ig_result['sie']
plt.plot(x7Rev_ig, u7Rev_ig, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x7Rev_ig, p7Rev_ig, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x7Rev_ig, r7Rev_ig, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x7Rev_ig, e7Rev_ig, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-0.2, 3.5))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = r'Riemann Problem 7 reversed: Sod w/ $\gamma_L \neq \gamma_R$ using the IGEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x7Rev_gen = riem7Rev_gen_result['position']
u7Rev_gen = riem7Rev_gen_result['velocity']
p7Rev_gen = riem7Rev_gen_result['pressure']
r7Rev_gen = riem7Rev_gen_result['density']
e7Rev_gen = riem7Rev_gen_result['sie']
plt.plot(x7Rev_gen, u7Rev_gen, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x7Rev_gen, p7Rev_gen, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x7Rev_gen, r7Rev_gen, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x7Rev_gen, e7Rev_gen, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 1.))
plt.ylim((-0.2, 3.5))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
title_str = r'Riemann Problem 7 reversed: Sod w/ $\gamma_L \neq \gamma_R$ using the GenEOS solver' 
plt.title(title_str, fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x_Shyue = riem_Shyue_result['position']
u_Shyue = riem_Shyue_result['velocity']
p_Shyue = riem_Shyue_result['pressure']
r_Shyue = riem_Shyue_result['density']
e_Shyue = riem_Shyue_result['sie']
plt.plot(x_Shyue, u_Shyue, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x_Shyue, p_Shyue, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x_Shyue, r_Shyue, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x_Shyue, e_Shyue, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 100.))
plt.ylim((-0.5, 25.))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
plt.title('Riemann Problem for Shyue JWL using the GenEOS solver', fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x_ShyueRev = riem_ShyueRev_result['position']
u_ShyueRev = riem_ShyueRev_result['velocity']
p_ShyueRev = riem_ShyueRev_result['pressure']
r_ShyueRev = riem_ShyueRev_result['density']
e_ShyueRev = riem_ShyueRev_result['sie']
plt.plot(x_ShyueRev, u_ShyueRev, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x_ShyueRev, p_ShyueRev, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x_ShyueRev, r_ShyueRev, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x_ShyueRev, e_ShyueRev, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 100.))
plt.ylim((-0.5, 25.))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
plt.title('Riemann Problem for Shyue JWL, reversed using the GenEOS solver', fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x_Lee = riem_Lee_result['position']
u_Lee = riem_Lee_result['velocity']
p_Lee = riem_Lee_result['pressure']
r_Lee = riem_Lee_result['density']
e_Lee = riem_Lee_result['sie']
plt.plot(x_Lee, u_Lee, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x_Lee, p_Lee, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x_Lee, r_Lee, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x_Lee, e_Lee, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 100.))
plt.ylim((-0.16, 4.))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
plt.title('Riemann Problem for Lee JWL using the GenEOS solver', fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


fig = plt.figure(figsize=(10,7))
x_LeeRev = riem_LeeRev_result['position']
u_LeeRev = riem_LeeRev_result['velocity']
p_LeeRev = riem_LeeRev_result['pressure']
r_LeeRev = riem_LeeRev_result['density']
e_LeeRev = riem_LeeRev_result['sie']
plt.plot(x_LeeRev, u_LeeRev, 'r', label=r'Velocity [$cm/s$]', linewidth=5.)
plt.plot(x_LeeRev, p_LeeRev, 'g', label=r'Pressure [$dyn/cm^2$]', linewidth=5.)
plt.plot(x_LeeRev, r_LeeRev, 'b', label=r'Density [$g/cm^3$]', linewidth=5.)
plt.plot(x_LeeRev, e_LeeRev, 'k', label=r'SIE [$erg$]', linewidth=5.)
plt.xlim((0., 100.))
plt.ylim((-0.16, 4.))
plt.xlabel(r'x [cm]', fontsize=20.)
plt.ylabel(r'Flow Property', fontsize=20.)
plt.xticks(fontsize=20.)
plt.yticks(fontsize=20.)
plt.grid(which='major', axis='both')
plt.title('Riemann Problem for Lee JWL, reversed using the GenEOS solver', fontsize=20.)
plt.legend(fontsize=20.)
plt.show()


# from exactpack.solvers.riemann.riemann import RiemannIGEOS
# riem1_ig_soln = RiemannIGEOS(rl=1.0,   ul=0.,   pl=1.0,  gl=1.4,
#                              rr=0.125, ur=0.,   pr=0.1,  gr=1.4,
#                              xmin=0.,  xd0=0.5, xmax=1., t=0.25)
# riem1_ig_soln.driver()
# 
# fig = plt.figure(figsize=(10,7))
# uNCSplow, uRCNplow = riem1_ig_soln.uNCSplow, riem1_ig_soln.uRCNplow
# uSCNphigh, uNCRphigh = riem1_ig_soln.uSCNphigh, riem1_ig_soln.uNCRphigh
# uaps, uRCVRps, ul_tilde = riem1_ig_soln.uaps, riem1_ig_soln.uRCVRps, riem1_ig_soln.ul_tilde
# plow, phigh, ps = riem1_ig_soln.plow, riem1_ig_soln.phigh, riem1_ig_soln.ps
# pstar, ustar = riem1_ig_soln.px, riem1_ig_soln.ux
# umin = min(min(uNCSplow), min(uRCNplow), min(uSCNphigh), min(uNCRphigh))
# label_str  = 'star state\n p = {:5g}\n u = {:5g}'.format(pstar, ustar)
# plt.plot(ustar, pstar, color='r', marker="*", markersize = 15, label=label_str)
# plt.plot(uNCSplow, plow, color = 'k')
# plt.plot(uRCNplow, plow, color = 'k')
# plt.plot(uSCNphigh, phigh, color = 'k')
# plt.plot(uNCRphigh, phigh, color = 'k')
# plt.plot(uaps, ps, '--k')
# plt.plot(uRCVRps, ps, 'r')
# plt.axvline(ul_tilde, color = 'k', linestyle = '--')
# plt.xlim((umin, 1.5 * ul_tilde))
# plt.ylim((0., phigh[-1]))
# plt.xlabel(r'u [$cm/s$]')
# plt.ylabel(r'p normalized to the left-state initial pressure, p$_l$')
# plt.legend(loc = 'upper right')
# title_str = 'P-U diagram relating the contact discontinuity star-state\n to the Riemann problem solution type (i.e., SCS, SCR, etc.).'
# plt.suptitle(title_str)
# plt.title(r'see Figure 3 in Gottlieb & Groths 1988 JCP')
# plt.annotate(text='SCS', xy=(-2, 1))
# plt.annotate(text='SCR', xy=(0, 1.75))
# plt.annotate(text='RCR', xy=(2, 1))
# plt.annotate(text='RCS', xy=(-0.5, 0.25))
# plt.annotate(text='RCVCR', xy=(1.2 * ul_tilde, 1.))
# plt.show()
# 
# 
# from exactpack.solvers.riemann.riemann import RiemannGenEOS
# riem_Lee = GenEOS_Solver(rl=0.9525, ul=0.,   pl=1.,     gl=1.8938,
#                          rr=3.81,   ur=0.,   pr=2.,     gr=1.8938,
#                          xmin=0.,   xd0=50., xmax=100., t=20.,
#                          A=632.1,   B=-0.04472,
#                          R1=11.3,   R2=1.13,
#                          r0=1.905,  e0=0.,
#                          problem='JWL')
# 
# fig = plt.figure(figsize=(10,7))
# gs = gridspec.GridSpec(1, 1, wspace = 0.0, top    = 0.5, left  = 0.1,
#                              hspace = 0.0, bottom = 0.0, right = 0.0)
# 
# integ_ps_left,integ_ps_right = riem_Lee.integ_ps_left, riem_Lee.integ_ps_right
# shock_ps_left,shock_ps_right = riem_Lee.shock_ps_left, riem_Lee.shock_ps_right
# uls, ulx, urs, urx = riem_Lee.uls, riem_Lee.ulx, riem_Lee.urs, riem_Lee.urx
# pl, ul, pr, ur = riem_Lee.pl, riem_Lee.ul, riem_Lee.pr, riem_Lee.ur
# px, ux1 = riem_Lee.px, riem_Lee.ux1
# ustar_str = 'ustar = {:3g}'.format(ux1)
# pstar_str = 'pstar = {:3g}'.format(px)
# plt.plot(ulx, shock_ps_left, 'b', label = 'left shock')
# plt.axvline(ul, color = 'b', linestyle = ':', label = 'ul, pl')
# plt.plot(uls, integ_ps_left, '--b', label = 'left rarefaction')
# plt.axhline(pl, color = 'b', linestyle = ':')
# plt.axvline(ux1, color = 'k', linestyle = '--', label = ustar_str)
# plt.plot(urx, shock_ps_right, 'r', label = 'right shock')
# plt.axvline(ur, color = 'r', linestyle = ':', label = 'ur, pr')
# plt.plot(urs, integ_ps_right, '--r', label = 'right rarefaction')
# plt.axhline(pr, color = 'r', linestyle = ':')
# plt.axhline(px, color = 'k', linestyle = '--', label = pstar_str)
# # plt.xlim((-0.8, 0.3))
# # plt.ylim((0, 2.75))
# plt.xlabel('u [$cm/s$]')
# plt.ylabel('p [$dyn/cm^2$]')
# plt.legend(ncol = 2)
# # title_str = "P-U diagram showing potential shock and rarefaction\n waves using $(p, u)_0$. The crossing curves give values for\n $p^{\star}$ & $u^{\star}$ and determine\n whether the wave is a shock or rarefaction."
# title_str = "P-U diagram for the Lee JWL problem."
# plt.title(title_str)
# plt.show()
