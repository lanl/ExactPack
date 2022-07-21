r"""Set and check values of a complete set of isotropic, linear elastic
parameters.
"""

# (Sat Sep 24 2016) walter
#
# In this version, function set_elastic_params() uses compile() and a namespace
# dict 'ns' to store internal param variables introduced by use of exec.
# This is to avoid the version-dependent problems with the 'exec' statement
# in the previous coding.


import warnings
import numpy as np

# The next four fcns are the isotropic, linear elastic, Positive-Definite
# strain energy function tests named using the indices from pg. 85 of the
# Gurtin reference in the Blake solver (viz, ./__init__.py) docstring.
# These are called *only* by the main fcn herein: set_elastic_params().
# Each issues a ValueError() when the given params yield a non-PD energy.


def check_ii(prmcase, lbda_name, lbda, G_name, G, blk_dbg_prm):
    r"""Terminate if the strain energy function is **not** PD.

    Apply Gurtin's condition (ii) [#gur83]_ to the elasticity parmeters to
    determine if the isotropic, linear elastic strain energy is positive
    definite (PD).
    """
    if not (G > 0. and 2 * G + 3 * lbda > 0.):
        msg = ('Given or calculated values of ' + lbda_name + ' = ' +
               str(lbda) + ' and ' + G_name + ' = ' + str(G) +
               '\n\t    do not yield a ' +
               'positive-definite strain energy function.')
        if blk_dbg_prm:
            msg = msg + '\n\t    prmcase = ' + str(prmcase)

        raise ValueError(msg)


def check_iii(prmcase, G_name, G, K_name, K, blk_dbg_prm):
    r"""Terminate if the strain energy function is **not** PD.

    Apply Gurtin's condition (iii) [#gur83]_ to the elasticity parmeters to
    determine if the isotropic, linear elastic strain energy is positive
    definite (PD).
    """
    if not (G > 0. and K > 0.):
        msg = ('Given or calculated values of ' + G_name + ' = ' + str(G) +
               ' and ' + K_name + ' = ' + str(K) + '\n\t    do not ' +
               'yield a positive-definite strain energy function.')
        if blk_dbg_prm:
            msg = msg + '\n\t    prmcase = ' + str(prmcase)

        raise ValueError(msg)


def check_iv(prmcase, G_name, G, nu_name, nu, blk_dbg_prm):
    r"""Terminate if the strain energy function is **not** PD.

    Apply Gurtin's condition (iv) [#gur83]_ to the elasticity parmeters to
    determine if the isotropic, linear elastic strain energy is positive
    definite (PD).
    """
    if not (G > 0. and -1.0 < nu < 0.5):
        msg = ('Given or calculated values of ' + G_name + ' = ' + str(G) +
               ' and ' + nu_name + ' = ' + str(nu) + '\n\t    do not ' +
               'yield a positive-definite strain energy function.')
        if blk_dbg_prm:
            msg = msg + '\n\t    prmcase = ' + str(prmcase)

        raise ValueError(msg)


def check_v(prmcase, E_name, E, nu_name, nu, blk_dbg_prm):
    r"""Terminate if the strain energy function is **not** PD.

    Apply Gurtin's condition (v) [#gur83]_ to the elasticity parmeters to
    determine if the isotropic, linear elastic strain energy is positive
    definite (PD).
    """
    if not (E > 0. and -1.0 < nu < 0.5):
        msg = ('Given or calculated values of ' + E_name + ' = ' + str(E) +
               ' and ' + nu_name + ' = ' + str(nu) + '\n\t    do not ' +
               'yield a positive-definite strain energy function.')
        if blk_dbg_prm:
            msg = msg + '\n\t    prmcase = ' + str(prmcase)

        raise ValueError(msg)


# Additional elastic parameter checking fcns.

def warn_negative_poisson(prmcase, nu, pnm, pv, blk_dbg_prm):
    r"""Issue a warning if Poisson's Ratio is non-positive.

    This function should be invoked whenever the isotropic linear elastic
    material defined by two positive moduli *could* have a non-positive
    Poisson's Ratio.  If the value *is* non-positive, an informational warning
    is issued as such materials are uncommon and the negative sign might be due
    to an input error.  To reliably specify a material with negative Poisson's
    Ratio, the poisson_ratio parameters should be used explicitly.
    """

    if nu > 0.0:
        return None
    else:
        msg = ('\nNon-positive poisson_ratio = ' + str(nu) +
               ', has been calculated from \n    ' +
               pnm[0] + ' = ' + str(pv[0]) +
               ', and ' + pnm[1] + ' = ' + str(pv[1]) + '.')
        msg = msg + ('\n    If this is *not* expected, ' +
                     'please check your input.')
        msg = msg + ('\n    If a negative poisson_ratio *is* desired, ' +
                     'specify it explicitly.')
        if blk_dbg_prm:
            msg = msg + '\n    prmcase = ' + str(prmcase)  # dbg

        warnings.warn(msg)
    return None


def term_cmplx_poisson(prmcase, pnm, pv, blk_dbg_prm):
    r"""Raise a value error if Poisson's Ratio will be complex.

    Certain combinations of two positive moduli *can* produce a complex value
    for Poisson's Ratio.
    """

    msg = ("\nSpecified values of " + pnm[0] + " = " + str(pv[0]) +
           ", and " + pnm[1] + " = " + str(pv[1]) + ", " +
           "\n    will produce a complex value for Poisson's Ratio." +
           "\n    Please check your inputs and try again.")

    if blk_dbg_prm:
        msg = msg + '\n    prmcase = ' + str(prmcase)  # dbg

    raise ValueError(msg)


def term_nan_poisson(prmcase, pnm, pv, blk_dbg_prm):
    r"""Raise a value error if Poisson's Ratio is a NaN.

    Certain combinations of two positive moduli *can* produce a NaN for
    Poisson's Ratio, due to divide by zero.
    """

    msg = ("\nSpecified values of " + pnm[0] + " = " + str(pv[0]) +
           ", and " + pnm[1] + " = " + str(pv[1]) + ", " +
           "\n    will produce a NaN value for the Poisson's Ratio " +
           "(to within a small rel. tolerance) due to divide by zero."
           "\n    Please check your inputs and try again.")

    if blk_dbg_prm:
        msg = msg + '\n    prmcase = ' + str(prmcase)  # dbg

    raise ValueError(msg)


def term_nan_lame(prmcase, pnm, pv, blk_dbg_prm):
    r"""Raise a value error if the first Lame modulus is a Nan.

    Certain combinations of two positive moduli *can* produce a NaN for the
    Lame modulus, due to divide by zero.
    """

    msg = ("\nSpecified values of " + pnm[0] + " = " + str(pv[0]) +
           ", and " + pnm[1] + " = " + str(pv[1]) + ", " +
           "\n    will produce a NaN value for the Lame modulus " +
           "(to within a small rel. tolerance) due to divide by zero."
           "\n    Please check your inputs and try again.")

    if blk_dbg_prm:
        msg = msg + '\n    prmcase = ' + str(prmcase)  # dbg

    raise ValueError(msg)

# END additional elastic parameter checking fcns.


def set_elastic_params(elas_prm_names, elas_prm_dflt_vals,
                       elas_prm_order, defaulted, blk_dbg_prm, **kwargs):
    r"""Calculate and check a full set of isotropic, linear elastic parameters.

    An isotropic, linear elastic solid is defined by specifying, via the kwargs
    dict, any **two** of the **six** elastic parameters defined in class Blake.
    This function computes the remaining four elastic parameters and performs
    some sanity checking.  In particular we require that,

    1. Each user-specified modulus parameter is positive.
    2. Each pair of user-specified parameters define a material which has a
       positive-definite (PD) strain energy function.
    3. If Poisson's Ratio is negative when calculated from user-specified
       moduli, a non-fatal warning message to that effect is issued.  A
       negative value is uncommon for materials in their linear elastic range.
       Moreover, if the user *intends* that Poisson's Ratio be negative, then
       it should be passed to the Blake constructor *explicitly*.  When
       Poisson's Ratio is inferred from moduli values, a square root sign
       ambiguity may prevent the correct sign being set.

    If conditions 1 or 2 aren't satisfied, an error message is issued and
    *ExactPack* terminates.  Otherwise, the full parameter set is returned as a
    dictionary of the material parameter names and values.  When True is passed
    as the value of argument "defaulted", a dictionary of default values is
    constructed from the first two arguments and returned.  This is the *only*
    default case accepted by this function.

    **NOTE** : It is the caller's responsibility to ensure that the order of
    elements in elas_prm_names and elas_prm_dflt_vals agrees with each other
    and with the ordinality specified in elas_prm_order.  Even if defaulted is
    *False*, the elas_prm_names and elas_prm_order arguments must still agree.
    In particular, the elas_prm_names list *must* contain the same six strings
    as does the corresponding attribute of :class:`Blake`.

    The blk_dbg_prm argument is True/False and enables available debugging
    code throughout this module when True.
    """

    # Sanity check defaults, construct default dict & return if
    # defaulted == True.
    nelprm = 6          # number of elastic parameters
    if(len(elas_prm_names) == nelprm and
       len(elas_prm_dflt_vals) == nelprm and
       len(elas_prm_order) == nelprm):
        elas_prm_dflts = dict(zip(elas_prm_names, elas_prm_dflt_vals))
        if defaulted:
            return elas_prm_dflts
    else:
        raise ValueError('Length of default argument(s) is invalid!')

    #           Setup for generic case

    # Internal param var names (same order as elas_prm_names):
    int_var_names = ('plda', 'pg', 'pe', 'pnu', 'pk', 'pm')

    # make a reverse dict from internal var names to external param names.
    ivar_pnms = dict(zip(int_var_names, elas_prm_names))

    # Create local elastic param vars w/ COMPLEX init. value.
    # ipr, ips are intermediate variables (called R, S in [#gur83])
    # which are computed in some cases.
    # Complex value makes lack of overrite easy to detect at end.

    ns = {}             # namespace for local vars introduced by exec.
    vnmlst = ['ipr', 'ips']     # Include these params in loop.
    vnmlst.extend(int_var_names)
    for nm in vnmlst:
        stmt = nm + ' = ' + str(1j)
        # print '\nstmt str is: ', stmt                 #dbg
        cobj = compile(stmt, '<string>', 'exec')
        exec(cobj, ns)

    #
    #          Generic case
    #
    # Store the valid param names from kwargs.key().
    elas_prm_args = {}
    for ky in elas_prm_dflts:
        if ky in kwargs:
            elas_prm_args[ky] = float(kwargs[ky])

    if len(elas_prm_args) != 2:
        raise ValueError(
            """Invalid number of elastic parameters specified!
            EXACTLY *two* of the six possible elastic parameters must be
            specified to create a non-default Blake instance!
            Misspelled parameter name(s) can also raise this error.
            """)

    #   Parameter Restrictions
    # Here we check only that: any GIVEN moduli are positive and
    # a GIVEN poisson_ratio (pnu) lies in the PD strain energy range:
    # -1.0 < pnu < 0.5.
    for ky in elas_prm_args:
        if (ky == 'poisson_ratio'):
            if not (-1.0 < elas_prm_args[ky] < 0.5):
                err_str = f'Specified value of {ky} is not in the open '
                err_str += 'interval: (-1.0, 0.5).'
                raise ValueError(err_str)
        else:
            # all other elas params are moduli
            if elas_prm_args[ky] <= 0.0:
                err_str = f'Specified value of {ky} is non-positive.'
                raise  ValueError(err_str)

    # Ensure elas_prm_order[ekey0] < elas_prm_order[ekey1].
    # Coding below depends on this.
    eprmkys = list(elas_prm_args.keys())
    if elas_prm_order[eprmkys[0]] < elas_prm_order[eprmkys[1]]:
        [eky0, eky1] = eprmkys
    else:
        if blk_dbg_prm:
            print('\neprmkeys reversed per elas_prm_order!')
        eprmkys.reverse()
        [eky0, eky1] = eprmkys

    # Set prmcase and local param vars.
    if eky0 == 'lame_mod':
        ns['plda'] = elas_prm_args[eky0]
        if eky1 == 'shear_mod':
            prmcase = 0
            ns['pg'] = elas_prm_args[eky1]
        elif eky1 == 'youngs_mod':
            prmcase = 1
            ns['pe'] = elas_prm_args[eky1]
        elif eky1 == 'poisson_ratio':
            prmcase = 2
            ns['pnu'] = elas_prm_args[eky1]
        elif eky1 == 'bulk_mod':
            prmcase = 3
            ns['pk'] = elas_prm_args[eky1]
        elif eky1 == 'long_mod':
            prmcase = 4
            ns['pm'] = elas_prm_args[eky1]

    elif eky0 == 'shear_mod':
        ns['pg'] = elas_prm_args[eky0]
        if eky1 == 'youngs_mod':
            prmcase = 5
            ns['pe'] = elas_prm_args[eky1]
        elif eky1 == 'poisson_ratio':
            prmcase = 6
            ns['pnu'] = elas_prm_args[eky1]
        elif eky1 == 'bulk_mod':
            prmcase = 7
            ns['pk'] = elas_prm_args[eky1]
        elif eky1 == 'long_mod':
            prmcase = 8
            ns['pm'] = elas_prm_args[eky1]

    elif eky0 == 'youngs_mod':
        ns['pe'] = elas_prm_args[eky0]
        if eky1 == 'poisson_ratio':
            prmcase = 9
            ns['pnu'] = elas_prm_args[eky1]
        elif eky1 == 'bulk_mod':
            prmcase = 10
            ns['pk'] = elas_prm_args[eky1]
        elif eky1 == 'long_mod':
            prmcase = 11
            ns['pm'] = elas_prm_args[eky1]

    elif eky0 == 'poisson_ratio':
        ns['pnu'] = elas_prm_args[eky0]
        if eky1 == 'bulk_mod':
            prmcase = 12
            ns['pk'] = elas_prm_args[eky1]
        elif eky1 == 'long_mod':
            prmcase = 13
            ns['pm'] = elas_prm_args[eky1]

    elif eky0 == 'bulk_mod':
        ns['pk'] = elas_prm_args[eky0]
        if eky1 == 'long_mod':
            prmcase = 14
            ns['pm'] = elas_prm_args[eky1]

    if blk_dbg_prm:
        print('prmcase = ', prmcase)

    # POSITIVE-DEFINITENESS (PD) of STRAIN ENERGY FUNCTION
    # Calculation and checking of full elas param set.
    #
    # There are four PD strain energy test functions at top of file.
    # We apply one of these tests in each prmcase.
    # RECALL: we have already verified that each GIVEN *modulus* is pos.
    # and, if GIVEN, pnu (poisson's ratio) satisfies: -1.0 < pnu < 0.5.
    # We ASSUME THESE CONDITIONS here.
    #
    # When positive moduli *could* yield a neg. value for pnu (Poisson) we
    # invoke warn_negative_poisson() which issues a non-fatal warning if pnu
    # actually *is* neg.  See the docstring for detail.
    #
    # When the given moduli: 1) will yield a complex param value; 2) are
    # sufficiently close to yielding a NaN param value, we invoke one of the
    # term_*_*() functions which issues a ValueError().  We invoke these
    # checking functions as early as possible in each prmcase block to avoid
    # possible un-trapped floating point exceptions (FPEs).

    # Tolerances for numpy.isclose() to control termination using
    # term_nan_lame() or term_nan_poisson().
    abstol = 0.0
    reltol = 1.0e-13

    #
    # eky0 == 'lame_mod'
    if prmcase == 0:
        # given -- lame_mod, shear_mod
        # Neg. poisson and FPEs not possible.
        check_ii(prmcase, eky0, ns['plda'], eky1, ns['pg'], blk_dbg_prm)
        ns['pe'] = (ns['pg'] * (3 * ns['plda'] + 2 * ns['pg']) /
                    (ns['plda'] + ns['pg']))
        ns['pnu'] = ns['plda'] / (2 * (ns['plda'] + ns['pg']))
    elif prmcase == 1:
        # given -- lame_mod, youngs_mod
        # Neg. poisson and FPEs not possible.
        ns['ipr'] = (
            pow(ns['pe']**2 + 9*ns['plda']**2 + 2*ns['pe']*ns['plda'], 0.5))
        ns['pnu'] = 2 * ns['plda'] / (ns['pe'] + ns['plda'] + ns['ipr'])
        ns['pg'] = 0.25 * (ns['pe'] - 3 * ns['plda'] + ns['ipr'])
        check_ii(prmcase, eky0, ns['plda'], ivar_pnms['pg'], ns['pg'],
                 blk_dbg_prm)
    elif prmcase == 2:
        # given -- lame_mod, poisson_ratio
        # Neg. poisson and FPEs not possible.
        ns['pe'] = ns['plda']*(1 + ns['pnu'])*(1 - 2 * ns['pnu'])/ns['pnu']
        ns['pg'] = ns['plda']*(1 - 2 * ns['pnu']) / (2*ns['pnu'])
        check_ii(prmcase, eky0, ns['plda'], ivar_pnms['pg'], ns['pg'],
                 blk_dbg_prm)
    elif prmcase == 3:
        # given -- lame_mod, bulk_mod
        # check for poisson_mod = NaN
        pnames = (eky0, eky1)
        pvals = (ns['plda'], ns['pk'])
        icb = 3*ns['pk']
        if np.isclose(ns['plda'], icb, rtol=reltol, atol=abstol):
            term_nan_poisson(prmcase, pnames, pvals, blk_dbg_prm)

        ns['pnu'] = ns['plda'] / (3 * ns['pk'] - ns['plda'])
        warn_negative_poisson(prmcase, ns['pnu'], pnames, pvals, blk_dbg_prm)
        ns['pg'] = 3 * (ns['pk'] - ns['plda']) / 2
        check_ii(prmcase, eky0, ns['plda'], ivar_pnms['pg'], ns['pg'],
                 blk_dbg_prm)
        ns['pe'] = (9 * ns['pk'] * (ns['pk'] - ns['plda']) /
                    (3 * ns['pk'] - ns['plda']))
    elif prmcase == 4:
        # given -- lame_mod, long_mod
        # Neg. poisson and FPEs not possible.
        ns['pg'] = (ns['pm'] - ns['plda']) / 2
        check_ii(prmcase, eky0, ns['plda'], ivar_pnms['pg'], ns['pg'],
                 blk_dbg_prm)
        ns['pe'] = ((ns['pm'] - ns['plda']) * (ns['pm'] + 2 * ns['plda']) /
                    (ns['pm'] + ns['plda']))
        ns['pnu'] = ns['plda'] / (ns['pm'] + ns['plda'])
    #
    # eky0 == 'shear_mod'
    elif prmcase == 5:
        # given -- shear_mod, youngs_mod
        # check for lame_mod = NaN
        pnames = (eky0, eky1)
        pvals = (ns['pg'], ns['pe'])
        icb = 3 * ns['pg']            # (pe = icb) --> div-zero
        if np.isclose(ns['pe'], icb, rtol=reltol, atol=abstol):
            term_nan_lame(prmcase, pnames, pvals, blk_dbg_prm)

        ns['plda'] = (ns['pg'] * (ns['pe'] - 2 * ns['pg']) /
                      (3 * ns['pg'] - ns['pe']))
        ns['pnu'] = (ns['pe'] / (2 * ns['pg'])) - 1
        warn_negative_poisson(prmcase, ns['pnu'], pnames, pvals, blk_dbg_prm)
        check_iv(prmcase, eky0, ns['pg'], ivar_pnms['pnu'], ns['pnu'],
                 blk_dbg_prm)
    elif prmcase == 6:
        # given -- shear_mod, poisson_ratio
        # Neg. poisson and FPEs not possible.
        check_iv(prmcase, eky0, ns['pg'], eky1, ns['pnu'], blk_dbg_prm)
        ns['plda'] = 2 * ns['pg'] * ns['pnu'] / (1 - 2 * ns['pnu'])
        ns['pe'] = 2 * ns['pg'] * (1 + ns['pnu'])
    elif prmcase == 7:
        # given -- shear_mod, bulk_mod
        pnames = (eky0, eky1)
        pvals = (ns['pg'], ns['pm'])
        ns['pnu'] = (3 * ns['pk'] - 2 * ns['pg'])/(6 * ns['pk'] + 2 * ns['pg'])
        warn_negative_poisson(prmcase, ns['pnu'], pnames, pvals, blk_dbg_prm)
        check_iii(prmcase, eky0, ns['pg'], eky1, ns['pk'], blk_dbg_prm)
        ns['plda'] = ns['pk'] - 2 * ns['pg'] / 3
        ns['pe'] = 9 * ns['pk'] * ns['pg'] / (3 * ns['pk'] + ns['pg'])
    elif prmcase == 8:
        # given -- shear_mod, long_mod
        # check for poisson_mod = NaN
        pnames = (eky0, eky1)
        pvals = (ns['pg'], ns['pm'])
        #                       # (pm = pg) --> div-zero
        if np.isclose(ns['pm'], ns['pg'], rtol=reltol, atol=abstol):
            term_nan_poisson(prmcase, pnames, pvals, blk_dbg_prm)

        ns['pnu'] = (ns['pm'] - 2 * ns['pg']) / (2 * ns['pm'] - 2 * ns['pg'])
        warn_negative_poisson(prmcase, ns['pnu'], pnames, pvals, blk_dbg_prm)
        check_iv(prmcase, eky0, ns['pg'], ivar_pnms['pnu'], ns['pnu'],
                 blk_dbg_prm)
        ns['plda'] = ns['pm'] - 2 * ns['pg']
        ns['pe'] = (ns['pg'] * (3 * ns['pm'] - 4 * ns['pg']) /
                    (ns['pm'] - ns['pg']))
    #
    # eky0 == 'youngs_mod'
    elif prmcase == 9:
        # given -- youngs_mod, poisson_ratio
        # Neg. poisson and FPEs not possible.
        check_v(prmcase, eky0, ns['pe'], eky1, ns['pnu'], blk_dbg_prm)
        ns['plda'] = ns['pe']*ns['pnu']/((1 + ns['pnu'])*(1 - 2 * ns['pnu']))
        ns['pg'] = 0.5 * ns['pe'] / (1 + ns['pnu'])
    elif prmcase == 10:
        # given -- youngs_mod, bulk_mod
        # check for lame_mod = NaN
        pnames = (eky0, eky1)
        pvals = (ns['pe'], ns['pk'])
        icb = 9 * ns['pk']            # (pe = icb) --> div-zero
        if np.isclose(ns['pe'], icb, rtol=reltol, atol=abstol):
            term_nan_lame(prmcase, pnames, pvals, blk_dbg_prm)

        ns['plda'] = (3 * ns['pk'] * (3 * ns['pk'] - ns['pe']) /
                      (9 * ns['pk'] - ns['pe']))
        ns['pnu'] = (3 * ns['pk'] - ns['pe']) / (6 * ns['pk'])
        warn_negative_poisson(prmcase, ns['pnu'], pnames, pvals, blk_dbg_prm)
        check_v(prmcase, eky0, ns['pe'], ivar_pnms['pnu'], ns['pnu'],
                blk_dbg_prm)
        ns['pg'] = 3 * ns['pk'] * ns['pe'] / (9 * ns['pk'] - ns['pe'])
    elif prmcase == 11:
        # given -- youngs_mod, long_mod
        # check for poisson = Complex
        # NOTE: sign(pnu) == sign(ips)
        #
        # Because materials with (pnu < 0) are unusual, we use the
        # pos. root here.  To handle materials with an intended NEGATIVE
        # poisson_ratio, the user should explicitly pass poisson_ratio
        # to the constructor.
        #
        # SIGN of f(pe, pm) = pe**2 + 9*pm**2 - 10*pe*pm.
        # In terms of pm and pg, (pm - pe) = (pm - 2*pg)**2 / (pm - pg).
        # The numerator has min value 0 on line pm = 2*pg and denom. > 0,
        # so pm > pe for all pm > pg > 0, which is the case for real matls.
        # Positivity of f(pe, pm) follows directly by considering
        # f(pe, alpha*pe) with alpha > 1.
        # Thus ips is real provided only that pm > pg.
        pnames = (eky0, eky1)
        pvals = (ns['pe'], ns['pm'])
        ips2 = ns['pe']**2 + 9 * ns['pm']**2 - 10*ns['pe']*ns['pm']
        if ips2 < 0.0:
            term_cmplx_poisson(prmcase, pnames, pvals, blk_dbg_prm)

        ns['ips'] = pow(ips2, 0.5)
        ns['pnu'] = 0.25 * (ns['pe'] - ns['pm'] + ns['ips']) / ns['pm']
        warn_negative_poisson(prmcase, ns['pnu'], pnames, pvals, blk_dbg_prm)
        check_v(prmcase, eky0, ns['pe'], ivar_pnms['pnu'], ns['pnu'],
                blk_dbg_prm)
        ns['plda'] = 0.25 * (ns['pm'] - ns['pe'] + ns['ips'])
        ns['pg'] = 0.125 * (3 * ns['pm'] + ns['pe'] - ns['ips'])
    #
    # eky0 == 'poisson_ratio'
    elif prmcase == 12:
        # given -- poisson_ratio, bulk_mod
        # Neg. poisson and FPEs not possible.
        ns['pg'] = 3 * ns['pk'] * (1 - 2 * ns['pnu']) / (2 * (1 + ns['pnu']))
        check_iv(prmcase, ivar_pnms['pg'], ns['pg'], eky0, ns['pnu'],
                 blk_dbg_prm)
        ns['plda'] = 3 * ns['pk'] * ns['pnu'] / (1 + ns['pnu'])
        ns['pe'] = 3 * ns['pk'] * (1 - 2 * ns['pnu'])
    elif prmcase == 13:
        # given -- poisson_ratio, long_mod
        # Neg. poisson and FPEs not possible.
        ns['pg'] = 0.5 * ns['pm'] * (1 - 2 * ns['pnu']) / (1 - ns['pnu'])
        check_iv(prmcase, ivar_pnms['pg'], ns['pg'], eky0, ns['pnu'],
                 blk_dbg_prm)
        ns['plda'] = ns['pm'] * ns['pnu'] / (1 - ns['pnu'])
        ns['pe'] = 2 * (1 + ns['pnu']) * ns['pg']
    #
    # eky0 == 'bulk_mod'
    elif prmcase == 14:
        # given -- bulk_mod, long_mod
        pnames = (eky0, eky1)
        pvals = (ns['pk'], ns['pm'])
        ns['pg'] = 0.75 * (ns['pm'] - ns['pk'])
        check_iii(prmcase, ivar_pnms['pg'], ns['pg'], eky0, ns['pk'],
                  blk_dbg_prm)
        ns['pnu'] = (3 * ns['pk'] - ns['pm']) / (3 * ns['pk'] + ns['pm'])
        warn_negative_poisson(prmcase, ns['pnu'], pnames, pvals, blk_dbg_prm)
        ns['plda'] = 0.5 * (3 * ns['pk'] - ns['pm'])
        ns['pe'] = 9*ns['pk']*(ns['pm'] - ns['pk']) / (3*ns['pk'] + ns['pm'])

    # end if prmcase

    # Verify that (plda, pg, pe, pnu) have been set.
    allset = not any((v for v in (ns['plda'], ns['pg'], ns['pe'], ns['pnu'])
                      if isinstance(v, complex)))
    errmsg = (
        """Internal: one of the internal params has not yet been set!
        Contact ExactPack team!
        """)
    assert allset, errmsg

    # One or both of bulk_mod, long_mod may not yet have been set.
    # plda and pg have been set.
    if isinstance(ns['pk'], complex):
        ns['pk'] = ns['plda'] + 2 * ns['pg'] / 3
    if isinstance(ns['pm'], complex):
        ns['pm'] = ns['plda'] + 2 * ns['pg']

    # Provided inputs are as described in docstring, this dict contains the
    # correct name-value mapping.
    return dict(zip(elas_prm_names, [ns['plda'], ns['pg'], ns['pe'],
                    ns['pnu'], ns['pk'], ns['pm']]))
