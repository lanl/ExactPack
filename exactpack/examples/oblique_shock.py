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
from exactpack.solvers.oblique_shocks.oblique_shocks import TwoShockTheory

# xvec = linspace(0., 1., int(1e5))
# t_final = 0.25

solver = TwoShockTheory()
solver._run()
solver.plot_pressure_deflection()
