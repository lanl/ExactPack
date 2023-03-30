r"""A python solver for the spherical, isotropic, linear elastic Blake problem.

This Blake version handles only spherical geometry with a constant (for
:math:`t > 0`) pressure history as in :ref:`formul-soln` above.
"""

from ...base import ExactSolver, ExactSolution, print_when_verbose
from . import set_check_elastic_params as elas_prms_mod

import warnings
import math as ma
import numpy as np


class Blake(ExactSolver):
    r"""Compute a solution \"snapshot\" for the spherical Blake problem.

    In the spherical Blake problem for an isotropic, linear-elastic material,
    solution discontinuities do not develop from smooth initial data because
    the governing equations and BCs are linear :eq:`rPhiEq`, :eq:`bdryCon`.
    This class calculates the solution (the fundamental field being the
    displacement) in terms of the radial coordinate in the :math:`t = 0`
    *reference configuration* of the elastic space.  In this domain, the
    solution is smooth because the pressure history is applied to the boundary
    (cavity surface).  To assist with hydrocode comparison, we calculate the
    \"curr_posn\" array as the sum of the (input) referential radial coordinate
    and the displacement at the snapshot time.

    **Elastic Parameters**

    There are six parameters which are commonly used to characterize an
    isotropic, linear-elastic solid.

    ==================================== ========
    Parameter Name                       Symbol
    ==================================== ========
    First Lame Modulus                   :math:`\quad \lambda`
    Shear Modulus (Second Lame Modulus)  :math:`\quad G`
    Young's Modulus                      :math:`\quad E`
    Poisson's Ratio                      :math:`\quad \nu`
    Bulk Modulus                         :math:`\quad K`
    Longitudinal Modulus                 :math:`\quad M`
    ==================================== ========

    The material is determined by specifying any **two** of these six; the
    other four are calculated internally.  Consequently, class Blake requires
    any two of these to create a solver instance.  For further detail, see the
    documentation for :func:`set_elastic_params` in module
    :mod:`set_check_elastic_params`.
    """

    # non-material problem defn params and descriptions
    problem_params_desc = {
        'geometry': "3 = spherical (dimensionless)",
        'ref_density': "initial (uniform) mass density (kg/m**3)",
        'cavity_radius':  """initial radius of cavity surface to which the
        pressure history is applied (m)""",
        'pressure_scale': """scale of pressure history imposed on cavity
        surface (Pa).  In this version of the solver the pressure is constant
        for :math:`t > 0`.""",
        'blake_debug': """True/False flag to turn on available debugging code.
        Useful mainly for a developer/maintainer.""",
    }

    # elastic material params
    elas_prm_names = ('lame_mod', 'shear_mod', 'youngs_mod',
                      'poisson_ratio', 'bulk_mod',
                      'long_mod')
    elas_prm_descriptions = (
        ("Lame modulus (Pa)", "Shear Modulus (Pa)", "Young's modulus (Pa)",
         "Poisson's Ratio (dimensionless)", "Bulk modulus (Pa)",
         "Longitudinal modulus (Pa)"))
    elas_params_desc = dict(zip(elas_prm_names, elas_prm_descriptions))

    # Ordinals of the elas param names.
    elas_prm_order = dict(zip(elas_prm_names, range(6)))

    parameters = {}             # all constructor params
    parameters.update(problem_params_desc)
    parameters.update(elas_params_desc)

    # Default Values

    # Non-material defaults dict
    prob_param_dflts = {
        'geometry': 3,
        'ref_density': 3000.0,
        'cavity_radius': 0.1,
        'pressure_scale': 1.0e6,
        'blake_debug': False,
    }
    # class ExactSolver requires these as scalars here.
    for nm in prob_param_dflts:
        val = prob_param_dflts[nm]
        stmt = nm + ' = ' + str(val)
        exec(stmt, None, None)

    elas_prm_dflt_vals = (25.0e9, 25.0e9, 62.5e9,
                          0.25, 41.66666666666667e9,
                          75.0e9)
    elas_prm_dflts = dict(zip(elas_prm_names, elas_prm_dflt_vals))

    # Separate copy accessed by unit test code.
    elas_param_values = {}

    # Default snapshot time.
    # tsnap is *not* a class Blake instance param, but is part of
    # the LANL default Blake solution, per ref. [#bro08].
    tsnap_default = 1.6e-04

    def __init__(self, **kwargs):
        r"""**Blake Class Snapshot Solver: Constructor**

        This function does the material and problem setup to create an instance
        of the spherical Blake class and returns the solver object.  The solver
        produces an :class:`ExactSolution` \"snapshot\" on the argument radial
        grid 'r' and at time 't', when invoked.  When the constructor is called
        with *no arguments*, the solver for the *default* Blake problem is
        instantiated.

        For the non-material problem parameters, only those for which a
        non-default value is desired need be specified.  If *no* elastic
        (material) parameters are specified, the default material obtains.
        No other defaulting of the elastic parameters is performed.

        **Default Problem Definition**

        As our default for spherical Blake, we use the Los Alamos standard
        setup as defined by Brock [#bro08]_.  This problem definition was also
        used by Kamm and Ankeny [#kam09]_ in a hydrocode evaluation using their
        own Blake solver.

        **Spherical Snapshot Solver Output Fields**

        ================= ========
        Field Name        Description (physical components not tensor cmpts)
        ================= ========
        position          reference radial coordinate
        curr_posn         current (snapshot time) radial coordinate
        displacement      radial component of displacement vector
        strain_rr         radial component of (infinitesimal) strain
        strain_qq         circumfrential component of strain
        strain_vol        volume strain = trace(strain tensor)
        density           current mass density
        stress_rr         radial component of stress
        stress_qq         circumfrential component of stress
        pressure          -(1/3) trace(stress tensor)
        stress_dev_rr     radial component of stress deviator
        stress_dev_qq     circumfrential component of stress deviator
        stress_diff       abs(stress_rr - stress_qq)
        ================= ========

        **Spherical Coordinates**

        We use a common physics convention.

        (r, theta, phi) = (radius, co-latitiude, longitude), pos. phi measured
        from +x-axis to +y-axis, so that the basis vectors in this order form a
        right-handed triad.  Indices 'q' and 'f' are used to indicate the theta
        and phi coorindates, respectively.  In the current problem, by
        symmetry, the phi and theta components are equal, so only theta
        components are output.  """

        if len(kwargs) == 0:
            # Full default: no given params.
            elas_dflt = True
            kwargs = Blake.prob_param_dflts.copy()
        else:
            # Some user-given params.
            # *Individual* non-elastic parameters may be defaulted.
            # Elastic params must be specified as a pair:
            # either defaulted or exactly two params given.
            if len(set(kwargs.keys()) & set(Blake.elas_prm_names)) > 0:
                elas_dflt = False
            else:
                elas_dflt = True

        # Handle blake_debug parameter.
        if kwargs.get('blake_debug') is None:
            dbg_arg = self.prob_param_dflts['blake_debug']
        else:
            dbg_arg = kwargs['blake_debug']

        # Calc. full elastic param set.
        elas_param_vals = (
            elas_prms_mod.set_elastic_params(
                Blake.elas_prm_names, Blake.elas_prm_dflt_vals,
                Blake.elas_prm_order, elas_dflt, dbg_arg,
                **kwargs))
        kwargs.update(elas_param_vals)
        self.elas_param_values.update(elas_param_vals)

        # Parent-class constructor traps any invalid param names.
        super(Blake, self).__init__(**kwargs)

        # Sanity checks.
        if not self.geometry == 3:
            raise ValueError(
                'Invalid value of geometry parameter: ' +
                str(self.geometry) + '.' +
                '\nOnly spherical geometry (= 3) implemented at this time.')

        if not self.ref_density > 0.:
            raise ValueError('ref_density parameter is non-positive.')

        if not self.cavity_radius > 0.:
            raise ValueError('cavity_radius parameter is non-positive.')

        if not self.pressure_scale > 0.:
            raise ValueError('pressure_scale parameter is non-positive.')

        # Over-large pressures invalidate small strain model assumption.
        pscale_bulkmod_warn_ratio = 0.1
        if not self.pressure_scale < pscale_bulkmod_warn_ratio * self.bulk_mod:
            msg = ('\npressure_scale parameter (= ' +
                   str(self.pressure_scale) +
                   ') should be < 10% of bulk_mod = ' + str(self.bulk_mod))
            warnings.warn(msg)

        if not isinstance(self.blake_debug, bool):
            raise ValueError(
                'blake_debug parameter is not boolean (True of False).')

        return None

    @print_when_verbose
    def _run(self, radii, tsnap):
        """Run the solver instance."""

        # Local vars
        third = 1.0 / 3.0
        zero_r = np.zeros(radii.size, dtype=np.float64)

        geom = self.geometry
        cavrad = self.cavity_radius
        ref_dens = self.ref_density
        pscl = self.pressure_scale
        plda = self.lame_mod
        pg = self.shear_mod
        pe = self.youngs_mod
        pnu = self.poisson_ratio
        pk = self.bulk_mod
        pm = self.long_mod

        grid_min = np.amin(radii)
        if grid_min < 0.0:
            msg = ('Minimum coordinate (= ' + str(grid_min) + ') of radial '
                   'grid is negative.' +
                   '\n\t    Please re-grid.')
            raise ValueError(msg)

        # Internal constants (indep. of (radius, time)).
        # (n, b) = Hutchins' (alpha, beta)
        cl = pow(pm / ref_dens, 0.5)       # long. sound speed

        n = ((1. - 2. * pnu) / (1. - pnu)) * (cl / cavrad)

        b = pow(((1. - 2. * pnu) / ((1. - pnu)**2.)) *
                ((cl / cavrad)**2.), 0.5)

        b2pn2 = b**2 + n**2

        k1 = cavrad * pscl / (ref_dens * b2pn2)

        # (radius and time)-dependent vars
        radii2 = np.square(radii)
        k1onr2_r = k1 / radii2
        kr_r = (b2pn2 / (n * cl)) * radii - 1.0
        # tp_r(tsnap) = reduced time: when tp < 0, soln is ident. zero.
        tp_r = tsnap - (radii - cavrad) / cl

        # Scalars for radial strain
        eacts = ma.exp(n * (tsnap + cavrad / cl))
        emacts = ma.exp(-n * (tsnap + cavrad / cl))
        # arrays
        enrc_r = np.exp((n / cl) * radii)
        cosb_r = np.cos(b * tp_r)
        sinb_r = np.sin(b * tp_r)
        emntp_r = np.exp(-n * tp_r)

        # Formulae for some fields calculated with Mathematica 10.x
        # Formula for displacement translated directly from (Hutchens 2005).
        displ = k1onr2_r * (1.0 - emntp_r * (cosb_r - (n / b) * kr_r * sinb_r))

        # Arrays needed for radial strain
        costerm_r = b * (-2.0 * cl**2 + b2pn2 * radii2) * cosb_r
        sinterm_r = (2.0 * n * cl**2 - 2.0 * cl * b2pn2 *
                     radii + n * b2pn2 * radii2) * sinb_r
        radstrn = (
            emacts * k1onr2_r *
            (-2.0 * eacts * b * cl**2 +
             enrc_r * (sinterm_r - costerm_r)) / (radii * b * cl**2))

        # Where cond is False, displacement = strain_rad = zero.
        cond = np.logical_and(np.greater_equal(radii, cavrad),
                              np.greater(tp_r, 0.0))
        displacement = np.where(cond, displ, zero_r)
        strain_rad = np.where(cond, radstrn, zero_r)

        # All other fields are fcns of displacement, strain_rad, params.
        curr_posn = radii + displacement
        strain_hoop = displacement / radii
        strain_vol = strain_rad + 2.0 * strain_hoop
        density = ref_dens / (1.0 + strain_vol)
        #
        stress_rad = (plda + 2.0 * pg) * strain_rad + 2.0 * plda * strain_hoop
        stress_hoop = plda * strain_rad + 2.0 * (plda + pg) * strain_hoop
        pressure = -third * (stress_rad + 2.0 * stress_hoop)
        stress_dev_rad = stress_rad + pressure
        stress_dev_hoop = stress_hoop + pressure
        # J. Kamm's defn of principle stress diff. NOTE: abs(.)
        stress_diff = abs(stress_rad - stress_hoop)

        return ExactSolution([radii,
                              curr_posn,
                              displacement,
                              strain_rad,
                              strain_hoop,
                              strain_vol,
                              density,
                              stress_rad,
                              stress_hoop,
                              pressure,
                              stress_dev_rad,
                              stress_dev_hoop,
                              stress_diff],
                             names=['position',
                                    'curr_posn',
                                    'displacement',
                                    'strain_rr',
                                    'strain_qq',
                                    'strain_vol',
                                    'density',
                                    'stress_rr',
                                    'stress_qq',
                                    'pressure',
                                    'stress_dev_rr',
                                    'stress_dev_qq',
                                    'stress_diff'],
                             jumps=None)
