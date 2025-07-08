r""" ExactPack wrapper for the steadystate 2section 2D Riemann solver.
"""

from exactpack.base import ExactSolver, ExactSolution, print_when_verbose

from exactpack.solvers.riemann2D_2section_steadystate import riemann2D_2section_steadystate
from numpy import interp, mgrid, array, tan, sqrt, inf, where, linspace
from scipy.optimize import bisect
from scipy.integrate import quad
import matplotlib.pyplot as plt

class IGEOS_Solver(ExactSolver):
    r"""Computes the analytic solution to the steadystate 2-section 2D Riemann
        problem for an ideal-gas EOS. See [Glaz1985]_ for the original
        description of the problem setup; see [Loh1990]_ App. A or [Loh1994]_
        Section 4 for descriptions of the solution procedure. The problem
        default values match Figure 3 in [Hui1999]_.

        Default values are :math:`bottom\_state=[1., 1., 2.4, 0., 1.4], top\_state=[0.25, 0.5, 7.0, 0., 1.4]`.
    """

    parameters = {
        'bottom_state': "List of values for pressure, density, Mach number, flow angle in degrees, and the adiabatic index, all for the bottom state.",
        'top_state': "List of values for pressure, density, Mach number, flow angle in degrees, and the adiabatic index, all for the top state.",
        't': "The end time."
    }

    bottom_state = [1., 1., 2.4, 0., 1.4]
    top_state = [0.25, 0.5, 7.0, 0., 1.4]
    t    = 0.25
    L = 1 # cm
    geometry = 1 # convergence study requires us to set geometry

    def __init__(self, **kwargs):
        """Set default values if necessary and check for valid inputs.
        """
        super(IGEOS_Solver, self).__init__(**kwargs)

    @print_when_verbose
    def _run(self, r, t):
        self.t = t
        xs, ys = zip(*r)
        xs = array(xs)
        ys = array(ys)
        prob = riemann2D_2section_steadystate.SetupRiemannProblem(
               bottom_state = self.bottom_state,
               top_state = self.top_state)
        prob.assign_lineout_vals(xs, ys)

        self.pB, self.thetaB_rad = prob.pB, prob.thetaB_rad
        self.pBc, self.dBc = prob.bottom_compression_arrays[:2]
        self.pBe, self.dBe = prob.bottom_expansion_arrays[:2]
        self.pT, self.thetaT_rad = prob.pT, prob.thetaT_rad
        self.pTc, self.dTc = prob.top_compression_arrays[:2]
        self.pTe, self.dTe = prob.top_expansion_arrays[:2]

        self.deflection_angle_solution = prob.deflection_angle_solution
        self.pressure_solution = prob.pressure_solution
        self.bottom_state = prob.bottom_state
        self.top_state = prob.top_state
        self.bottom_star_vals = prob.bottom_star_vals
        self.top_star_vals = prob.top_star_vals
        self.angles = prob.angles
        self.morphology = prob.morphology
        self.lineout_vals = prob.lineout_vals
        lineout_vals = self.lineout_vals

        x = lineout_vals[0]
        y = lineout_vals[1]
        pressure = lineout_vals[3]
        density = lineout_vals[4]
        specific_internal_energy = lineout_vals[5]
        Mach = lineout_vals[6]
        u = lineout_vals[7]
        v = lineout_vals[8]
        speed = lineout_vals[9]

        return ExactSolution([x, y, pressure, density, specific_internal_energy, Mach, u, v, speed],
                             names=['x_position',
                                    'y_position',
                                    'pressure',
                                    'density',
                                    'specific_internal_energy',
                                    'Mach',
                                    'x_velocity',
                                    'y_velocity',
                                    'speed'])
