'''Example demonstrating Riemann solvers for both ideal-gas EOS and generalized EOS in the form of JWL.
For the ideal-gas EOS, reproduces plots from Kamm, et. als., 2014 LA-UR-14-20418 "Standardized Definitions for Code Verification Test Problems".
There are a standard set of six Riemann problems for an ideal-gas EOS.
These are defined this 2014 report.
For the generalized EOS (JWL in this case), reproduces plots from Kamm's 2015 LA-UR-15-21616 "Exact Compressible One Dimensional Riemann Solver for General Convex Equations of State".
This 2015 report defines two additional ideal-gas EOS Riemann problems, and two generalized EOS Riemann problems for a JWL EOS.
One of the ideal-gas EOS problems is for left & right states with different values of the adiabatic index, :math:`\gamma`.
This makes for a total of 10 Riemann problems defined in these two documents.
These same 10 are also defined below and run as part of this example script.
'''

# import standard python packages
from matplotlib import pyplot as plt
from matplotlib import rc, gridspec
from importlib import reload

# import ExactPack solvers
from exactpack.solvers.riemann.riemann import RiemannIGEOS, RiemannGenEOS

# # pyplot default settings
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 16})
# rc('grid', c='0.5', ls='-', lw=0.5)
# 
# # Define the 10 Riemann problems 
# # The Sod shocktube problem solved using the ideal gas EOS (IGEOS) solver.
# riem1_ig = RiemannIGEOS(rl=1.0,   ul=0., pl=1.0, gl=1.4, xmin=0., xd0=0.5,
#                         rr=0.125, ur=0., pr=0.1, gr=1.4, xmax=1., t=0.25)
# riem1_ig.driver()
# 
# # The Sod shocktube problem solved using the generalized EOS (GenEOS) solver.
# riem1_gen = RiemannGenEOS(rl=1.0,   ul=0., pl=1.0, gl=1.4, xmin=0., xd0=0.5,
#                           rr=0.125, ur=0., pr=0.1, gr=1.4, xmax=1., t=0.25)
# riem1_gen.driver()
# 
# # The Sod shocktube flipping the left & right states, using the IGEOS solver.
# # This problem is probably more relevant for checking that the analytic solution
# # and/or semi-analytic integrator works correctly.
# riem1Rev_ig = RiemannIGEOS(rr=1.0,   ur=0., pr=1.0, gr=1.4, xmin=0., xd0=0.5,
#                            rl=0.125, ul=0., pl=0.1, gl=1.4, xmax=1., t=0.25)
# riem1Rev_ig.driver()
# 
# # The Sod shocktube flipping the left & right states, using the GenEOS solver.
# # This problem is probably more relevant for checking that the analytic solution
# # and/or semi-analytic integrator works correctly.
# riem1Rev_gen = RiemannGenEOS(rr=1.0,  ur=0.,pr=1.0, gr=1.4, xmin=0., xd0=0.5,
#                              rl=0.125,ul=0.,pl=0.1, gl=1.4, xmax=1., t=0.25)
# riem1Rev_gen.driver()
# 
# # The Sod shocktube with different gammas for the left & right states, using the
# # IGEOS solver.
# riem1Mod_ig = RiemannIGEOS(rl=1.0,   ul=0., pl=2.0, gl=2.0, xmin=0., xd0=0.5,
#                            rr=0.125, ur=0., pr=0.1, gr=1.4, xmax=1., t=0.2)
# riem1Mod_ig.driver()
# 
# # The Sod shocktube with different gammas for the left & right states, using the
# # GenEOS solver.
# riem1Mod_gen = RiemannGenEOS(rl=1.0,   ul=0., pl=2.0, gl=2.0, xmin=0., xd0=0.5,
#                              rr=0.125, ur=0., pr=0.1, gr=1.4, xmax=1., t=0.2)
# riem1Mod_gen.driver()
# 
# # The reversed Sod shocktube with different gammas for the left & right states,
# # using the IGEOS solver.
# riem1ModRev_ig = RiemannIGEOS(rr=1.0,   ur=0., pr=2.0, gr=2.0, xmin=0., xd0=0.5,
#                               rl=0.125, ul=0., pl=0.1, gl=1.4, xmax=1., t=0.2)
# riem1ModRev_ig.driver()
# 
# # The reversed Sod shocktube with different gammas for the left & right states,
# # using the GenEOS solver.
# riem1ModRev_gen = RiemannGenEOS(rr=1.0,  ur=0.,pr=2.0, gr=2.0, xmin=0., xd0=0.5,
#                                 rl=0.125,ul=0.,pl=0.1, gl=1.4, xmax=1., t=0.2)
# riem1ModRev_gen.driver()
# 
# # The Einfeldt problem solved using the IGEOS solver.
# riem2_ig = RiemannIGEOS(rl=1., ul=-2.0, pl=0.4, gl=1.4, xmin=0., xd0=0.5,
#                         rr=1., ur= 2.0, pr=0.4, gr=1.4, xmax=1., t=0.15)
# riem2_ig.driver()
# 
# # The reverse Einfeldt problem solved using the IGEOS solver.
# riem2Rev_ig = RiemannIGEOS(rr=1., ur= 2.0, pr=0.4, gr=1.4, xmin=0., xd0=0.5,
#                            rl=1., ul=-2.0, pl=0.4, gl=1.4, xmax=1., t=0.15)
# riem2Rev_ig.driver()
# 
# # The Einfeldt problem solved using the GenEOS solver.
# riem2_gen = RiemannGenEOS(rl=1., ul=-2.0, pl=0.4, gl=1.4, xmin=0., xd0=0.5,
#                           rr=1., ur= 2.0, pr=0.4, gr=1.4, xmax=1., t=0.15)
# riem2_gen.driver()
# 
# # The reverse Einfeldt problem solved using the IGEOS solver.
# riem2Rev_gen = RiemannGenEOS(rr=1., ur= 2.0, pr=0.4, gr=1.4, xmin=0., xd0=0.5,
#                              rl=1., ul=-2.0, pl=0.4, gl=1.4, xmax=1., t=0.15)
# riem2Rev_gen.driver()
# 
# # The Stationary Contact problem solved using the IGEOS solver.
# riem3_ig = RiemannIGEOS(rl=1.,ul=-19.59745,pl=1000.0,  gl=1.4, xmin=0., xd0=0.8,
#                         rr=1.,ur=-19.59745,pr=   0.01, gr=1.4, xmax=1., t=0.012)
# riem3_ig.driver()
# 
# # The reverse Stationary Contact problem solved using the IGEOS solver.
# riem3Rev_ig = RiemannIGEOS(rr=1.,ur=19.59745,pr=1000.0,  gr=1.4, xmin=0., xd0=0.2,
#                            rl=1.,ul=19.59745,pl=   0.01, gl=1.4, xmax=1., t=0.012)
# riem3Rev_ig.driver()
# 
# # The Stationary Contact problem solved using the GenEOS solver.
# riem3_gen = RiemannGenEOS(rl=1.,ul=-19.59745,pl=1000.0, gl=1.4,xmin=0., xd0=0.8,
#                           rr=1.,ur=-19.59745,pr=   0.01,gr=1.4,xmax=1., t=0.012)
# riem3_gen.driver()
# 
# # The reverse Stationary Contact problem solved using the IGEOS solver.
# riem3Rev_gen = RiemannGenEOS(rr=1.,ur=19.59745,pr=1000.0,gr=1.4,xmin=0.,xd0=0.2,
#                              rl=1.,ul=19.59745,pl=  0.01,gl=1.4,xmax=1.,t=0.012)
# riem3Rev_gen.driver()
# 
# # The Slow Shock problem solved using the IGEOS solver.
# riem4_ig = RiemannIGEOS(rl=3.857143,ul=-0.810631,pl=31./3.,gl=1.4, xmin=0.,xd0=0.5,
#                         rr=1.,      ur=-3.44,    pr=1.,    gr=1.4, xmax=1., t=1.)
# riem4_ig.driver()
# 
# # The reverse Slow Shock problem solved using the IGEOS solver.
# riem4Rev_ig = RiemannIGEOS(rr=3.857143,ur=0.810631,pr=31./3.,gr=1.4, xmin=0.,xd0=0.5,
#                            rl=1.,      ul=3.44,    pl=1.,    gl=1.4, xmax=1., t=1.)
# riem4Rev_ig.driver()
# 
# # The Slow Shock problem solved using the GenEOS solver.
# riem4_gen = RiemannGenEOS(rl=3.857143,ul=-0.810631,pl=31./3.,gl=1.4,xmin=0.,xd0=0.5,
#                            rr=1.,      ur=-3.44,    pr=1.,    gr=1.4,xmax=1., t=1.)
# riem4_gen.driver()
# 
# # The reverse Slow Shock problem solved using the IGEOS solver.
# riem4Rev_gen = RiemannGenEOS(rr=3.857143,ur=0.810631,pr=31./3.,gr=1.4,xmin=0.,xd0=0.5,
#                              rl=1.,      ul=3.44,    pl=1.,    gl=1.4, xmax=1., t=1.)
# riem4Rev_gen.driver()
# 
# # The Shock Contact Shock problem solved using the IGEOS solver.
# riem5_ig = RiemannIGEOS(rl=1.0,  ul= 0.5, pl=1., gl=1.4, xmin=0., xd0=0.5,
#                         rr=1.25, ur=-0.5, pr=1., gr=1.4, xmax=1., t=0.3)
# riem5_ig.driver()
# 
# # The reversed Shock Contact Shock problem solved using the IGEOS solver.
# riem5Rev_ig = RiemannIGEOS(rr=1.0,  ur=-0.5, pr=1., gr=1.4, xmin=0., xd0=0.5,
#                            rl=1.25, ul= 0.5, pl=1., gl=1.4, xmax=1., t=0.3)
# riem5Rev_ig.driver()
# 
# # The Shock Contact Shock problem solved using the GenEOS solver.
# riem5_gen = RiemannGenEOS(rl=1.0,  ul= 0.5, pl=1., gl=1.4, xmin=0., xd0=0.5,
#                           rr=1.25, ur=-0.5, pr=1., gr=1.4, xmax=1., t=0.3)
# riem5_gen.driver()
# 
# # The reversed Shock Contact Shock problem solved using the IGEOS solver.
# riem5Rev_gen = RiemannGenEOS(rr=1.0, ur=-0.5, pr=1., gr=1.4, xmin=0., xd0=0.5,
#                              rl=1.25,ul= 0.5, pl=1., gl=1.4, xmax=1., t=0.3)
# riem5Rev_gen.driver()
# 
# # The LeBlanc problem solved using the IGEOS solver.
# riem6_ig = RiemannIGEOS(rl=1.0, ul=0.,pl=1./15.,    gl=5./3., xmin=0.,xd0=0.3,
#                         rr=0.01,ur=0.,pr=2./(3.e10),gr=5./3., xmax=1., t=0.5)
# riem6_ig.driver()
# 
# # The reversed LeBlanc problem solved using the IGEOS solver.
# riem6Rev_ig = RiemannIGEOS(rr=1.0, ur=0.,pr=1./15.,    gr=5./3., xmin=0.,xd0=0.7,
#                            rl=0.01,ul=0.,pl=2./(3.e10),gl=5./3., xmax=1., t=0.5)
# riem6Rev_ig.driver()
# 
# # The LeBlanc problem solved using the GenEOS solver.
# riem6_gen =RiemannGenEOS(rl=1.0, ul=0.,pl=2./3.*1.e-1, gl=5./3.,xmin=0.,xd0=0.3,
#                          rr=0.01,ur=0.,pr=2./3.*1.e-10,gr=5./3.,xmax=1., t=0.5)
# riem6_gen.driver()
# 
# # The reversed LeBlanc problem solved using the IGEOS solver.
# riem6Rev_gen = RiemannGenEOS(rr=1.0,ur=0.,pr=1./15.,   gr=5./3.,xmin=0.,xd0=0.7,
#                              rl=0.01,ul=0.,pl=2./(3.e10),gl=5./3.,xmax=1.,t=0.5)
# riem6Rev_gen.driver()
# 
# # The Lee JWL shocktube problem solved using the GenEOS solver.
# riem_Lee = RiemannGenEOS(rl=0.9525, ul=0., pl=1., gl=1.8938, xmin=0., xd0=50.,
#                          rr=3.81,   ur=0., pr=2., gr=1.8938, xmax=100., t=20.,
#                          A=632.1, B=-0.04472, R1=11.3, R2=1.13, r0=1.905, e0=0.,
#                          problem='JWL')
# riem_Lee.driver()
# 
# # The Lee JWL shocktube problem solved using the GenEOS solver.
# riem_LeeRev = RiemannGenEOS(rr=0.9525,ur=0., pr=1., gr=1.8938, xmin=0., xd0=50.,
#                             rl=3.81,  ul=0., pl=2., gl=1.8938, xmax=100., t=20.,
#                             A=632.1,B=-0.04472,R1=11.3,R2=1.13, r0=1.905, e0=0.,
#                             problem='JWL')
# riem_LeeRev.driver()
# 
# # The Shyue JWL shocktube problem solved using the GenEOS solver.
# riem_Shyue = RiemannGenEOS(rl=1.7, ul=0., pl=10.0, gl=1.25, xmin=0.,   xd0=50.,
#                            rr=1.0, ur=0., pr= 0.5, gr=1.25, xmax=100., t=12.,
#                            A=8.545, B=0.205, R1=4.6, R2=1.35, r0=1.84, e0=0.,
#                            problem='JWL')
# riem_Shyue.driver()
# 
# # The Shyue JWL shocktube problem solved using the GenEOS solver.
# riem_ShyueRev = RiemannGenEOS(rr=1.7,ur=0.,pr=10.0, gr=1.25, xmin=0.,   xd0=50.,
#                               rl=1.0,ul=0.,pl= 0.5, gl=1.25, xmax=100., t=12.,
#                               A=8.545, B=0.205, R1=4.6, R2=1.35, r0=1.84, e0=0.,
#                               problem='JWL')
# riem_ShyueRev.driver()
# 
# fig = plt.figure(figsize=(8,8))
# x1_ig, u1_ig, p1_ig, r1_ig, e1_ig = riem1_ig.x, riem1_ig.u, riem1_ig.p, riem1_ig.r, riem1_ig.e
# # plt.subplot(gs[0,0])
# plt.plot(x1_ig, u1_ig, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x1_ig, p1_ig, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x1_ig, r1_ig, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x1_ig, e1_ig, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-0.5, 3.))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 1: Sod\n using the IGEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x1_gen, u1_gen, p1_gen, r1_gen, e1_gen = riem1_gen.x, riem1_gen.u, riem1_gen.p, riem1_gen.r, riem1_gen.e
# # plt.subplot(gs[0,0])
# plt.plot(x1_gen, u1_gen, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x1_gen, p1_gen, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x1_gen, r1_gen, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x1_gen, e1_gen, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-0.5, 3.))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 1: Sod\n using the GenEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x1Rev_ig, u1Rev_ig, p1Rev_ig, r1Rev_ig, e1Rev_ig = riem1Rev_ig.x, riem1Rev_ig.u, riem1Rev_ig.p, riem1Rev_ig.r, riem1Rev_ig.e
# # plt.subplot(gs[0,1])
# plt.plot(x1Rev_ig, u1Rev_ig, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x1Rev_ig, p1Rev_ig, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x1Rev_ig, r1Rev_ig, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x1Rev_ig, e1Rev_ig, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-1.1, 3.))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 1 reversed: Sod reversed\n using the IGEOS solver'
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x1Rev_gen, u1Rev_gen, p1Rev_gen, r1Rev_gen, e1Rev_gen = riem1Rev_gen.x, riem1Rev_gen.u, riem1Rev_gen.p, riem1Rev_gen.r, riem1Rev_gen.e
# # plt.subplot(gs[0,1])
# plt.plot(x1Rev_gen, u1Rev_gen, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x1Rev_gen, p1Rev_gen, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x1Rev_gen, r1Rev_gen, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x1Rev_gen, e1Rev_gen, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-1.1, 3.))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 1 reversed: Sod reversed\n using the GenEOS solver'
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x1Mod_ig, u1Mod_ig, p1Mod_ig, r1Mod_ig, e1Mod_ig = riem1Mod_ig.x, riem1Mod_ig.u, riem1Mod_ig.p, riem1Mod_ig.r, riem1Mod_ig.e
# # plt.subplot(gs[1,0])
# plt.plot(x1Mod_ig, u1Mod_ig, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x1Mod_ig, p1Mod_ig, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x1Mod_ig, r1Mod_ig, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x1Mod_ig, e1Mod_ig, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-0.2, 3.5))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 1 modified: Sod w/ $\gamma_L \neq \gamma_R$\n using the IGEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x1Mod_gen, u1Mod_gen, p1Mod_gen, r1Mod_gen, e1Mod_gen = riem1Mod_gen.x, riem1Mod_gen.u, riem1Mod_gen.p, riem1Mod_gen.r, riem1Mod_gen.e
# # plt.subplot(gs[1,0])
# plt.plot(x1Mod_gen, u1Mod_gen, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x1Mod_gen, p1Mod_gen, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x1Mod_gen, r1Mod_gen, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x1Mod_gen, e1Mod_gen, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-0.2, 3.5))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 1 modified: Sod w/ $\gamma_L \neq \gamma_R$\n using the GenEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x1ModRev_ig, u1ModRev_ig, p1ModRev_ig, r1ModRev_ig, e1ModRev_ig = riem1ModRev_ig.x, riem1ModRev_ig.u, riem1ModRev_ig.p, riem1ModRev_ig.r, riem1ModRev_ig.e
# # plt.subplot(gs[1,0])
# plt.plot(x1ModRev_ig, u1ModRev_ig, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x1ModRev_ig, p1ModRev_ig, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x1ModRev_ig, r1ModRev_ig, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x1ModRev_ig, e1ModRev_ig, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-0.2, 3.5))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 1 modified reversed: Sod w/ $\gamma_L \neq \gamma_R$\n using the IGEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x1ModRev_gen, u1ModRev_gen, p1ModRev_gen, r1ModRev_gen, e1ModRev_gen = riem1ModRev_gen.x, riem1ModRev_gen.u, riem1ModRev_gen.p, riem1ModRev_gen.r, riem1ModRev_gen.e
# # plt.subplot(gs[1,0])
# plt.plot(x1ModRev_gen, u1ModRev_gen, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x1ModRev_gen, p1ModRev_gen, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x1ModRev_gen, r1ModRev_gen, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x1ModRev_gen, e1ModRev_gen, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-0.2, 3.5))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 1 modified reversed: Sod w/ $\gamma_L \neq \gamma_R$\n using the GenEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x2_ig, u2_ig, p2_ig, r2_ig, e2_ig = riem2_ig.x, riem2_ig.u, riem2_ig.p, riem2_ig.r, riem2_ig.e
# # plt.subplot(gs[1,1])
# plt.plot(x2_ig, u2_ig, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x2_ig, p2_ig, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x2_ig, r2_ig, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x2_ig, e2_ig, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-3., 3.))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 2: Einfeldt\n - using the IGEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x2_gen, u2_gen, p2_gen, r2_gen, e2_gen = riem2_gen.x, riem2_gen.u, riem2_gen.p, riem2_gen.r, riem2_gen.e
# # plt.subplot(gs[1,1])
# plt.plot(x2_gen, u2_gen, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x2_gen, p2_gen, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x2_gen, r2_gen, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x2_gen, e2_gen, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-3., 3.))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 2: Einfeldt\n - using the GenEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x2Rev_ig, u2Rev_ig, p2Rev_ig, r2Rev_ig, e2Rev_ig = riem2Rev_ig.x, riem2Rev_ig.u, riem2Rev_ig.p, riem2Rev_ig.r, riem2Rev_ig.e
# # plt.subplot(gs[1,1])
# plt.plot(x2Rev_ig, u2Rev_ig, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x2Rev_ig, p2Rev_ig, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x2Rev_ig, r2Rev_ig, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x2Rev_ig, e2Rev_ig, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-3., 3.))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 2: reversed Einfeldt\n - using the IGEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x2Rev_gen, u2Rev_gen, p2Rev_gen, r2Rev_gen, e2Rev_gen = riem2Rev_gen.x, riem2Rev_gen.u, riem2Rev_gen.p, riem2Rev_gen.r, riem2Rev_gen.e
# # plt.subplot(gs[1,1])
# plt.plot(x2Rev_gen, u2Rev_gen, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x2Rev_gen, p2Rev_gen, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x2Rev_gen, r2Rev_gen, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x2Rev_gen, e2Rev_gen, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-3., 3.))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 2: reversed Einfeldt\n - using the GenEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x3_ig, u3_ig, p3_ig, r3_ig, e3_ig = riem3_ig.x, riem3_ig.u, riem3_ig.p, riem3_ig.r, riem3_ig.e
# # plt.subplot(gs[2,0])
# plt.plot(x3_ig, u3_ig, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x3_ig, p3_ig, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x3_ig, r3_ig, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x3_ig, e3_ig, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-500., 3000.))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 3: Stationary Contact\n - using the IGEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x3_gen, u3_gen, p3_gen, r3_gen, e3_gen = riem3_gen.x, riem3_gen.u, riem3_gen.p, riem3_gen.r, riem3_gen.e
# # plt.subplot(gs[2,0])
# plt.plot(x3_gen, u3_gen, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x3_gen, p3_gen, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x3_gen, r3_gen, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x3_gen, e3_gen, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-500., 3000.))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 3: Stationary Contact\n - using the GenEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x3Rev_ig, u3Rev_ig, p3Rev_ig, r3Rev_ig, e3Rev_ig = riem3Rev_ig.x, riem3Rev_ig.u, riem3Rev_ig.p, riem3Rev_ig.r, riem3Rev_ig.e
# # plt.subplot(gs[2,0])
# plt.plot(x3Rev_ig, u3Rev_ig, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x3Rev_ig, p3Rev_ig, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x3Rev_ig, r3Rev_ig, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x3Rev_ig, e3Rev_ig, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-500., 3000.))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 3: reversed Stationary Contact\n - using the IGEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x3Rev_gen, u3Rev_gen, p3Rev_gen, r3Rev_gen, e3Rev_gen = riem3Rev_gen.x, riem3Rev_gen.u, riem3Rev_gen.p, riem3Rev_gen.r, riem3Rev_gen.e
# # plt.subplot(gs[2,0])
# plt.plot(x3Rev_gen, u3Rev_gen, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x3Rev_gen, p3Rev_gen, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x3Rev_gen, r3Rev_gen, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x3Rev_gen, e3Rev_gen, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-500., 3000.))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 3: reversed Stationary Contact\n - using the GenEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x4_ig, u4_ig, p4_ig, r4_ig, e4_ig = riem4_ig.x, riem4_ig.u, riem4_ig.p, riem4_ig.r, riem4_ig.e
# # plt.subplot(gs[2,1])
# plt.plot(x4_ig, u4_ig, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x4_ig, p4_ig, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x4_ig, r4_ig, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x4_ig, e4_ig, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-4., 12.))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 4: Slow Shock\n - using the IGEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x4_gen, u4_gen, p4_gen, r4_gen, e4_gen = riem4_gen.x, riem4_gen.u, riem4_gen.p, riem4_gen.r, riem4_gen.e
# # plt.subplot(gs[2,1])
# plt.plot(x4_gen, u4_gen, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x4_gen, p4_gen, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x4_gen, r4_gen, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x4_gen, e4_gen, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-4., 12.))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 4: Slow Shock\n - using the GenEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x4Rev_ig, u4Rev_ig, p4Rev_ig, r4Rev_ig, e4Rev_ig = riem4Rev_ig.x, riem4Rev_ig.u, riem4Rev_ig.p, riem4Rev_ig.r, riem4Rev_ig.e
# # plt.subplot(gs[2,1])
# plt.plot(x4Rev_ig, u4Rev_ig, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x4Rev_ig, p4Rev_ig, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x4Rev_ig, r4Rev_ig, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x4Rev_ig, e4Rev_ig, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-4., 12.))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 4: reversed Slow Shock\n - using the IGEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x4Rev_gen, u4Rev_gen, p4Rev_gen, r4Rev_gen, e4Rev_gen = riem4Rev_gen.x, riem4Rev_gen.u, riem4Rev_gen.p, riem4Rev_gen.r, riem4Rev_gen.e
# # plt.subplot(gs[2,1])
# plt.plot(x4Rev_gen, u4Rev_gen, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x4Rev_gen, p4Rev_gen, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x4Rev_gen, r4Rev_gen, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x4Rev_gen, e4Rev_gen, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-4., 12.))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 4: reversed Slow Shock\n - using the GenEOS solver'
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x5_ig, u5_ig, p5_ig, r5_ig, e5_ig = riem5_ig.x, riem5_ig.u, riem5_ig.p, riem5_ig.r, riem5_ig.e
# # plt.subplot(gs[3,0])
# plt.plot(x5_ig, u5_ig, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x5_ig, p5_ig, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x5_ig, r5_ig, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x5_ig, e5_ig, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-1., 3.5))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 5: Shock Contact Shock\n - using the IGEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x5_gen, u5_gen, p5_gen, r5_gen, e5_gen = riem5_gen.x, riem5_gen.u, riem5_gen.p, riem5_gen.r, riem5_gen.e
# # plt.subplot(gs[3,0])
# plt.plot(x5_gen, u5_gen, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x5_gen, p5_gen, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x5_gen, r5_gen, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x5_gen, e5_gen, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-1., 3.5))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 5: Shock Contact Shock\n - using the GenEOS solver'
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x5Rev_ig, u5Rev_ig, p5Rev_ig, r5Rev_ig, e5Rev_ig = riem5Rev_ig.x, riem5Rev_ig.u, riem5Rev_ig.p, riem5Rev_ig.r, riem5Rev_ig.e
# # plt.subplot(gs[3,0])
# plt.plot(x5Rev_ig, u5Rev_ig, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x5Rev_ig, p5Rev_ig, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x5Rev_ig, r5Rev_ig, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x5Rev_ig, e5Rev_ig, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-1., 3.5))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 5: reversed Shock Contact Shock\n - using the IGEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x5Rev_gen, u5Rev_gen, p5Rev_gen, r5Rev_gen, e5Rev_gen = riem5Rev_gen.x, riem5Rev_gen.u, riem5Rev_gen.p, riem5Rev_gen.r, riem5Rev_gen.e
# # plt.subplot(gs[3,0])
# plt.plot(x5Rev_gen, u5Rev_gen, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x5Rev_gen, p5Rev_gen, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x5Rev_gen, r5Rev_gen, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x5Rev_gen, e5Rev_gen, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-1., 3.5))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 5: reversed Shock Contact Shock\n - using the GenEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x6_ig, u6_ig, p6_ig, r6_ig, e6_ig = riem6_ig.x, riem6_ig.u, riem6_ig.p, riem6_ig.r, riem6_ig.e
# # plt.subplot(gs[3,1])
# plt.plot(x6_ig, u6_ig, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x6_ig, p6_ig, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x6_ig, r6_ig, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x6_ig, e6_ig, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-0.2, 1.2))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 6: LeBlanc\n - using the IGEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x6_gen, u6_gen, p6_gen, r6_gen, e6_gen = riem6_gen.x, riem6_gen.u, riem6_gen.p, riem6_gen.r, riem6_gen.e
# # plt.subplot(gs[3,1])
# plt.plot(x6_gen, u6_gen, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x6_gen, p6_gen, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x6_gen, r6_gen, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x6_gen, e6_gen, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-0.2, 1.2))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 6: LeBlanc\n - using the GenEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x6Rev_ig, u6Rev_ig, p6Rev_ig, r6Rev_ig, e6Rev_ig = riem6Rev_ig.x, riem6Rev_ig.u, riem6Rev_ig.p, riem6Rev_ig.r, riem6Rev_ig.e
# # plt.subplot(gs[3,1])
# plt.plot(x6Rev_ig, u6Rev_ig, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x6Rev_ig, p6Rev_ig, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x6Rev_ig, r6Rev_ig, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x6Rev_ig, e6Rev_ig, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-0.2, 1.2))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 6: reversed LeBlanc\n - using the IGEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x6Rev_gen, u6Rev_gen, p6Rev_gen, r6Rev_gen, e6Rev_gen = riem6Rev_gen.x, riem6Rev_gen.u, riem6Rev_gen.p, riem6Rev_gen.r, riem6Rev_gen.e
# # plt.subplot(gs[3,1])
# plt.plot(x6Rev_gen, u6Rev_gen, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x6Rev_gen, p6Rev_gen, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x6Rev_gen, r6Rev_gen, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x6Rev_gen, e6Rev_gen, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 1.))
# plt.ylim((-0.2, 1.2))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# title_str = 'Riemann Problem 6: reversed LeBlanc\n - using the GenEOS solver' 
# plt.title(title_str)
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x_Lee, u_Lee, p_Lee, r_Lee, e_Lee = riem_Lee.x, riem_Lee.u, riem_Lee.p, riem_Lee.r, riem_Lee.e
# # plt.subplot(gs[4,0])
# plt.plot(x_Lee, u_Lee, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x_Lee, p_Lee, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x_Lee, r_Lee, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x_Lee, e_Lee, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 100.))
# plt.ylim((-0.16, 4.))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# plt.title('Riemann Problem for Lee JWL\n using the GenEOS solver')
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x_LeeRev, u_LeeRev, p_LeeRev, r_LeeRev, e_LeeRev = riem_LeeRev.x, riem_LeeRev.u, riem_LeeRev.p, riem_LeeRev.r, riem_LeeRev.e
# # plt.subplot(gs[4,0])
# plt.plot(x_LeeRev, u_LeeRev, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x_LeeRev, p_LeeRev, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x_LeeRev, r_LeeRev, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x_LeeRev, e_LeeRev, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 100.))
# plt.ylim((-0.16, 4.))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# plt.title('Riemann Problem for Lee JWL, reversed\n using the GenEOS solver')
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x_Shyue, u_Shyue, p_Shyue, r_Shyue, e_Shyue = riem_Shyue.x, riem_Shyue.u, riem_Shyue.p, riem_Shyue.r, riem_Shyue.e
# # plt.subplot(gs[4,1])
# plt.plot(x_Shyue, u_Shyue, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x_Shyue, p_Shyue, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x_Shyue, r_Shyue, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x_Shyue, e_Shyue, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 100.))
# plt.ylim((-0.5, 25.))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# plt.title('Riemann Problem for Shyue JWL\n using the GenEOS solver')
# plt.legend()
# plt.show()
# 
# 
# fig = plt.figure(figsize=(8,8))
# x_ShyueRev, u_ShyueRev, p_ShyueRev, r_ShyueRev, e_ShyueRev = riem_ShyueRev.x, riem_ShyueRev.u, riem_ShyueRev.p, riem_ShyueRev.r, riem_ShyueRev.e
# # plt.subplot(gs[4,1])
# plt.plot(x_ShyueRev, u_ShyueRev, 'r', label=r'Velocity [$cm/s$]')
# plt.plot(x_ShyueRev, p_ShyueRev, 'g', label=r'Pressure [$dyn/cm^2$]')
# plt.plot(x_ShyueRev, r_ShyueRev, 'b', label=r'Density [$g/cm^3$]')
# plt.plot(x_ShyueRev, e_ShyueRev, 'k', label=r'SIE [$erg$]')
# plt.xlim((0., 100.))
# plt.ylim((-0.5, 25.))
# plt.xlabel(r'x [cm]')
# plt.ylabel(r'Flow Property')
# plt.grid(which='major', axis='both')
# plt.title('Riemann Problem for Shyue JWL, reversed\n using the GenEOS solver')
# plt.legend()
# plt.show()
# 
# 
# # fig = plt.figure(figsize=(8,8))
# # uNCSplow, uRCNplow = riem1_ig.uNCSplow, riem1_ig.uRCNplow
# # uSCNphigh, uNCRphigh = riem1_ig.uSCNphigh, riem1_ig.uNCRphigh
# # uaps, uRCVRps, ul_tilde = riem1_ig.uaps, riem1_ig.uRCVRps, riem1_ig.ul_tilde
# # plow, phigh, ps = riem1_ig.plow, riem1_ig.phigh, riem1_ig.ps
# # pstar, ustar = riem1_ig.px, riem1_ig.ux
# # umin = min(min(uNCSplow), min(uRCNplow), min(uSCNphigh), min(uNCRphigh))
# # label_str  = 'star state\n p = {:5g}\n u = {:5g}'.format(pstar, ustar)
# # plt.plot(ustar, pstar, color='r', marker="*", markersize = 15, label=label_str)
# # plt.plot(uNCSplow, plow, color = 'k')
# # plt.plot(uRCNplow, plow, color = 'k')
# # plt.plot(uSCNphigh, phigh, color = 'k')
# # plt.plot(uNCRphigh, phigh, color = 'k')
# # plt.plot(uaps, ps, '--k')
# # plt.plot(uRCVRps, ps, 'r')
# # plt.axvline(ul_tilde, color = 'k', linestyle = '--')
# # plt.xlim((umin, 1.5 * ul_tilde))
# # plt.ylim((0., phigh[-1]))
# # plt.xlabel(r'u [$cm/s$]')
# # plt.ylabel(r'p normalized to the left-state initial pressure, p$_l$')
# # plt.legend(loc = 'upper right')
# # title_str = 'P-U diagram relating the contact discontinuity star-state\n to the Riemann problem solution type (i.e., SCS, SCR, etc.).'
# # plt.suptitle(title_str)
# # plt.title(r'see Figure 3 in Gottlieb & Groths 1988 JCP')
# # plt.annotate(text='SCS', xy=(-2, 1))
# # plt.annotate(text='SCR', xy=(0, 1.75))
# # plt.annotate(text='RCR', xy=(2, 1))
# # plt.annotate(text='RCS', xy=(-0.5, 0.25))
# # plt.annotate(text='RCVCR', xy=(1.2 * ul_tilde, 1.))
# # plt.show()
# # 
# # 
# # fig = plt.figure(figsize=(8,8))
# # gs = gridspec.GridSpec(1, 1, wspace = 0.0, top    = 0.5, left  = 0.1,
# #                              hspace = 0.0, bottom = 0.0, right = 0.0)
# # 
# # integ_ps_left,integ_ps_right = riem_Lee.integ_ps_left, riem_Lee.integ_ps_right
# # shock_ps_left,shock_ps_right = riem_Lee.shock_ps_left, riem_Lee.shock_ps_right
# # uls, ulx, urs, urx = riem_Lee.uls, riem_Lee.ulx, riem_Lee.urs, riem_Lee.urx
# # pl, ul, pr, ur = riem_Lee.pl, riem_Lee.ul, riem_Lee.pr, riem_Lee.ur
# # px, ux1 = riem_Lee.px, riem_Lee.ux1
# # ustar_str = 'ustar = {:3g}'.format(ux1)
# # pstar_str = 'pstar = {:3g}'.format(px)
# # plt.plot(ulx, shock_ps_left, 'b', label = 'left shock')
# # plt.axvline(ul, color = 'b', linestyle = ':', label = 'ul, pl')
# # plt.plot(uls, integ_ps_left, '--b', label = 'left rarefaction')
# # plt.axhline(pl, color = 'b', linestyle = ':')
# # plt.axvline(ux1, color = 'k', linestyle = '--', label = ustar_str)
# # plt.plot(urx, shock_ps_right, 'r', label = 'right shock')
# # plt.axvline(ur, color = 'r', linestyle = ':', label = 'ur, pr')
# # plt.plot(urs, integ_ps_right, '--r', label = 'right rarefaction')
# # plt.axhline(pr, color = 'r', linestyle = ':')
# # plt.axhline(px, color = 'k', linestyle = '--', label = pstar_str)
# # # plt.xlim((-0.8, 0.3))
# # # plt.ylim((0, 2.75))
# # plt.xlabel('u [$cm/s$]')
# # plt.ylabel('p [$dyn/cm^2$]')
# # plt.legend(ncol = 2)
# # # title_str = "P-U diagram showing potential shock and rarefaction\n waves using $(p, u)_0$. The crossing curves give values for\n $p^{\star}$ & $u^{\star}$ and determine\n whether the wave is a shock or rarefaction."
# # title_str = "P-U diagram for the Lee JWL problem."
# # plt.title(title_str)
# # plt.show()
