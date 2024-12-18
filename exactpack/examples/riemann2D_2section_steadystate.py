# '''Example demonstrating 2-section steadystate 2D Riemann solvers an ideal-gas
# equation of state. Developing this to handle a generalized EOS should be
# straight-forward, the same as done for the 1D Riemann problem, and should be
# done soon.
# While currently being restricted to an ideal-gas EOS, the solver does allow
# for different values of the adiabatic index in the left and right states.
# This solver reproduces four plots from three papers as examples:
# [1] 1985, Glaz & Wardlaw, High order godunov scheme for steady supersonic gas dynamics; see Fig 15. This is also Fig 6 in [2], and the plot for Mach number is corrected/correctly given in [4] as Fig. 9.2.
# [2] 1990, Loh & Hui, New lagrangian method for steady supersonic flow computation part1 godunov scheme; see Figs. 6 & 7.
# [3] 1999, Hui, Li & Li, Unified coordinate system for solving the two dimensional euler equations; see Figs. 3 & 4.
# [4] 2012, Computational fluid dynamics based on the unified coordinates, Hui & Xu.
# A nice description of the solution method is given in [2] Appendix A, and
# also in
# [5] 1994, New lagrangian method for three dimensional steady supersonic flows, Loh & Liou.
# Of course, the solver is more general than these four problem setups, the comparisons simply serve as examples. Similarly, the reversed state configurations also serve as examples, thus providing 8 problem setups in the examples file.
# '''

# import standard python packages
from numpy import linspace, array, inf
import matplotlib.pyplot as plt

# import ExactPack solvers
from exactpack.solvers.riemann2D_2section_steadystate.ep_riemann2D_2section_steadystate import IGEOS_Solver

from exactpack.solvers.riemann2D_2section_steadystate.plot_utils import plot_data

# values taken from 1999 Hui, Li, Li paper
# A Unified Coordinate System for Solving the Two-Dimensional Euler Equations,
# Journal of Computational Physics 153, 596-637; see Fig. 3 and above text.
state_top = [0.25, 0.5, 7.0, 0., 1.4]
state_bottom = [1., 1., 2.4, 0., 1.4]
N=101
xlim=(0., 0.02, 1.02)
ylim=(0., 0.5, 0.8)
ys = linspace(ylim[0], ylim[2], int(1e4)) - ylim[1]
xs = 0. * ys + xlim[2] - 2. * xlim[1]
solver = IGEOS_Solver(bottom_state=state_bottom, top_state=state_top)
soln = solver._run(xs, ys)
plot_data(solver, soln, N=N, xlim=xlim, ylim=ylim, var_str=['pressure', 'density'])


# values taken from 1990 Loh & Hui paper, see Figs. 6
# New Lagrangian Method fo Steady Supersonic Flow Computation
state_top = [0.25, 0.5, 4.0, 0., 1.4]
state_bottom = [1., 1., 2.4, 0., 1.4]
N=101
xlim=(0., 0.02, 0.42)
ylim=(0., 0.22, 0.42)
ys = linspace(ylim[0], ylim[2], int(1e4)) - ylim[1]
xs = 0. * ys + xlim[2] - 2. * xlim[1]
solver = IGEOS_Solver(bottom_state=state_bottom, top_state=state_top)
soln = solver._run(xs, ys)
plot_data(solver, soln, N=N, xlim=xlim, ylim=ylim, var_str=['pressure', 'density'])


# values taken from 1990 Loh & Hui paper, see Figs. 7
# New Lagrangian Method fo Steady Supersonic Flow Computation
state_top = [0.1, 0.2, 4.0, 0., 1.4]
state_bottom = [1., 1., 6., 15., 1.4]
N=101
xlim=(0., 0.02, 0.32)
ylim=(0., 0.01, 0.21)
ys = linspace(ylim[0], ylim[2], int(1e4)) - ylim[1]
xs = 0. * ys + xlim[2] - 2. * xlim[1]
solver = IGEOS_Solver(bottom_state=state_bottom, top_state=state_top)
soln = solver._run(xs, ys)
plot_data(solver, soln, N=N, xlim=xlim, ylim=ylim, var_str=['pressure', 'density'])



# values taken from 1985 Glaz & Wardlaw paper, see Fig. 16
# A High-Order Godunov Scheme for Steady Supersonic Gas Dynamics
state_top = [0.01, 0.05, 3.5, 0., 1.4]
state_bottom = [1., 1., 10., 23., 1.4]
N=101
xlim=(0., 0.02, 0.17)
ylim=(0., 0.01, 0.21)
ys = linspace(ylim[0], ylim[2], int(1e4)) - ylim[1]
xs = 0. * ys + xlim[2] - 2. * xlim[1]
solver = IGEOS_Solver(bottom_state=state_bottom, top_state=state_top)
soln = solver._run(xs, ys)
plot_data(solver, soln, N=N, xlim=xlim, ylim=ylim, var_str=['pressure', 'density'])


# values taken from 1999 Hui, Li, Li paper
# A Unified Coordinate System for Solving the Two-Dimensional Euler Equations,
# Journal of Computational Physics 153, 596-637; see Fig. 3 and above text.
# REVERSED STATE - REVERSED STATE - REVERSED STATE
staterev_top = [1., 1., 2.4, 0., 1.4]
staterev_bottom = [0.25, 0.5, 7.0, 0., 1.4]
N=101
xlim=(0., 0.02, 1.02)
ylim=(0., 0.3, 0.8)
ys_rev = linspace(ylim[0], ylim[2], int(1e4)) - ylim[1]
xs_rev = 0. * ys_rev + xlim[2] - 2. * xlim[1]
solver_rev = IGEOS_Solver(bottom_state=staterev_bottom, top_state=staterev_top)
soln_rev = solver_rev._run(xs_rev, ys_rev)
plot_data(solver_rev, soln_rev, N=N, xlim=xlim, ylim=ylim, var_str=['pressure', 'density'])


# values taken from 1990 Loh & Hui paper, see Figs. 6
# New Lagrangian Method fo Steady Supersonic Flow Computation
# REVERSED STATE - REVERSED STATE - REVERSED STATE
staterev_top = [1., 1., 2.4, 0., 1.4]
staterev_bottom = [0.25, 0.5, 4.0, 0., 1.4]
N=101
xlim=(0., 0.02, 0.42)
ylim=(0., 0.2, 0.42)
ys_rev = linspace(ylim[0], ylim[2], int(1e4)) - ylim[1]
xs_rev = 0. * ys_rev + xlim[2] - 2. * xlim[1]
solver_rev = IGEOS_Solver(bottom_state=staterev_bottom, top_state=staterev_top)
soln_rev = solver_rev._run(xs_rev, ys_rev)
plot_data(solver_rev, soln_rev, N=N, xlim=xlim, ylim=ylim, var_str=['pressure', 'density'])


# values taken from 1990 Loh & Hui paper, see Figs. 7
# New Lagrangian Method fo Steady Supersonic Flow Computation
# REVERSED STATE - REVERSED STATE - REVERSED STATE
staterev_top = [1., 1., 6., -15., 1.4]
staterev_bottom = [0.1, 0.2, 4.0, 0., 1.4]
N=101
xlim=(0., 0.02, 0.32)
ylim=(0., 0.2, 0.21)
ys_rev = linspace(ylim[0], ylim[2], int(1e4)) - ylim[1]
xs_rev = 0. * ys_rev + xlim[2] - 2. * xlim[1]
solver_rev = IGEOS_Solver(bottom_state=staterev_bottom, top_state=staterev_top)
soln_rev = solver_rev._run(xs_rev, ys_rev)
plot_data(solver_rev, soln_rev, N=N, xlim=xlim, ylim=ylim, var_str=['pressure', 'density'])


# values taken from 1985 Glaz & Wardlaw paper, see Fig. 16
# A High-Order Godunov Scheme for Steady Supersonic Gas Dynamics
# REVERSED STATE - REVERSED STATE - REVERSED STATE
staterev_top = [1., 1., 10., -23., 1.4]
staterev_bottom = [0.01, 0.05, 3.5, 0., 1.4]
N=101
xlim=(0., 0.02, 0.17)
ylim=(0., 0.2, 0.21)
ys_rev = linspace(ylim[0], ylim[2], int(1e4)) - ylim[1]
xs_rev = 0. * ys_rev + xlim[2] - 2. * xlim[1]
solver_rev = IGEOS_Solver(bottom_state=staterev_bottom, top_state=staterev_top)
soln_rev = solver_rev._run(xs_rev, ys_rev)
plot_data(solver_rev, soln_rev, N=N, xlim=xlim, ylim=ylim, var_str=['pressure', 'density'])
