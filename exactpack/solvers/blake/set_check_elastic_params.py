r"""Set and check values of a complete set of isotropic, linear elastic
parameters.
"""

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
    as the corresponding attribute of :class:`Blake`.
    
    The blk_dbg_prm argument is True/False and enables available debugging
    code throughout this module when True.
    """

    # Sanity check defaults, construct default dict & return if
    # defaulted == True.
    nelprm = 6          # number of elastic parameters
    if (len(elas_prm_names) == nelprm and
        len(elas_prm_dflt_vals) == nelprm and
        len(elas_prm_order) == nelprm):
        elas_prm_dflts = dict(zip(elas_prm_names, elas_prm_dflt_vals))
        if defaulted:
            return elas_prm_dflts
    else:
        raise ValueError('Length of default argument(s) is invalid!')

    #           Setup for generic case

    # Internal param vars (same order as elas_prm_names):
    int_var_names = ('plda', 'pg', 'pe', 'pnu', 'pk', 'pm')
    # make a reverse dict from internal var names to external param names.
    ivar_pnms = dict(zip(int_var_names, elas_prm_names))

    # Create local elas param vars w/ COMPLEX init. value.
    # ipr, ips are intermediate variables (sometimes called R, S)
    # which are computed in some cases.
    ipr = ips = 1j
    for nm in int_var_names:
        stmt = nm + ' = ' + str(1j)
        exec(stmt, None, None)

    #          Generic case

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
                raise(
                    ValueError(
                        'Specified value of ' + ky +
                        ' is not in the open interval: (-1.0, 0.5).')
                )
        else:
            # all other elas params are moduli
            if elas_prm_args[ky] <= 0.0:
                raise(
                    ValueError(
                        'Specified value of ' + ky +
                        ' is non-positive.')
                )

    # Ensure elas_prm_order[ekey0] < elas_prm_order[ekey1]: 
    # coding below depends on this.
    eprmkys = elas_prm_args.keys()
    if elas_prm_order[eprmkys[0]] < elas_prm_order[eprmkys[1]]:
        [eky0, eky1] = eprmkys
    else:
        if blk_dbg_prm:
            print '\neprmkeys reversed per elas_prm_order!'
        eprmkys.reverse()
        [eky0, eky1] = eprmkys

    # Set prmcase and local param vars.
    if eky0 == 'lame_mod':
        plda = elas_prm_args[eky0]
        if eky1 == 'shear_mod':
            prmcase = 0
            pg = elas_prm_args[eky1]
        elif eky1 == 'youngs_mod':
            prmcase = 1
            pe = elas_prm_args[eky1]
        elif eky1 == 'poisson_ratio':
            prmcase = 2
            pnu = elas_prm_args[eky1]
        elif eky1 == 'bulk_mod':
            prmcase = 3
            pk = elas_prm_args[eky1]
        elif eky1 == 'long_mod':
            prmcase = 4
            pm = elas_prm_args[eky1]

    elif eky0 == 'shear_mod':
        pg = elas_prm_args[eky0]
        if eky1 == 'youngs_mod':
            prmcase = 5
            pe = elas_prm_args[eky1]
        elif eky1 == 'poisson_ratio':
            prmcase = 6
            pnu = elas_prm_args[eky1]
        elif eky1 == 'bulk_mod':
            prmcase = 7
            pk = elas_prm_args[eky1]
        elif eky1 == 'long_mod':
            prmcase = 8
            pm = elas_prm_args[eky1]

    elif eky0 == 'youngs_mod':
        pe = elas_prm_args[eky0]
        if eky1 == 'poisson_ratio':
            prmcase = 9
            pnu = elas_prm_args[eky1]
        elif eky1 == 'bulk_mod':
            prmcase = 10
            pk = elas_prm_args[eky1]
        elif eky1 == 'long_mod':
            prmcase = 11
            pm = elas_prm_args[eky1]

    elif eky0 == 'poisson_ratio':
        pnu = elas_prm_args[eky0]
        if eky1 == 'bulk_mod':
            prmcase = 12
            pk = elas_prm_args[eky1]
        elif eky1 == 'long_mod':
            prmcase = 13
            pm = elas_prm_args[eky1]

    elif eky0 == 'bulk_mod':
        pk = elas_prm_args[eky0]
        if eky1 == 'long_mod':
            prmcase = 14
            pm = elas_prm_args[eky1]

    if blk_dbg_prm:
        print 'prmcase = ', prmcase

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
        check_ii(prmcase, eky0, plda, eky1, pg, blk_dbg_prm)
        pe = pg * (3 * plda + 2 * pg) / (plda + pg)
        pnu = plda / (2 * (plda + pg))
    elif prmcase == 1:
        # given -- lame_mod, youngs_mod
        # Neg. poisson and FPEs not possible.
        ipr = pow(pe**2 + 9*plda**2 + 2*pe*plda, 0.5)
        pnu = 2 * plda / (pe + plda + ipr)
        pg = 0.25 * (pe - 3 * plda + ipr)
        check_ii(prmcase, eky0, plda, ivar_pnms['pg'], pg, blk_dbg_prm)
    elif prmcase == 2:
        # given -- lame_mod, poisson_ratio
        # Neg. poisson and FPEs not possible.
        pe = plda * (1 + pnu) * (1 - 2 * pnu) / pnu
        pg = plda * (1 - 2 * pnu) / (2 * pnu)
        check_ii(prmcase, eky0, plda, ivar_pnms['pg'], pg, blk_dbg_prm)
    elif prmcase == 3:
        # given -- lame_mod, bulk_mod
        # check for poisson_mod = NaN
        pnames = (eky0, eky1)
        pvals = (plda, pk)
        icb = 3*pk
        if np.isclose(plda, icb, rtol=reltol, atol=abstol):
            term_nan_poisson(prmcase, pnames, pvals, blk_dbg_prm)
        pnu = plda / (3 * pk - plda)
        warn_negative_poisson(prmcase, pnu, pnames, pvals, blk_dbg_prm)
        pg = 3 * (pk - plda) / 2
        check_ii(prmcase, eky0, plda, ivar_pnms['pg'], pg, blk_dbg_prm)
        pe = 9 * pk * (pk - plda) / (3 * pk - plda)
    elif prmcase == 4:
        # given -- lame_mod, long_mod
        # Neg. poisson and FPEs not possible.
        pg = (pm - plda) / 2
        check_ii(prmcase, eky0, plda, ivar_pnms['pg'], pg, blk_dbg_prm)
        pe = (pm - plda) * (pm + 2 * plda) / (pm + plda)
        pnu = plda / (pm + plda)
    #
    # eky0 == 'shear_mod'
    elif prmcase == 5:
        # given -- shear_mod, youngs_mod
        # check for lame_mod = NaN
        pnames = (eky0, eky1)
        pvals = (pg, pe)
        icb = 3 * pg            # (pe = icb) --> div-zero
        if np.isclose(pe, icb, rtol=reltol, atol=abstol):
            term_nan_lame(prmcase, pnames, pvals, blk_dbg_prm)
        plda = pg * (pe - 2 * pg) / (3 * pg - pe)
        pnu = (pe / (2 * pg)) - 1
        warn_negative_poisson(prmcase, pnu, pnames, pvals, blk_dbg_prm)
        check_iv(prmcase, eky0, pg, ivar_pnms['pnu'], pnu, blk_dbg_prm)
    elif prmcase == 6:
        # given -- shear_mod, poisson_ratio
        # Neg. poisson and FPEs not possible.
        check_iv(prmcase, eky0, pg, eky1, pnu, blk_dbg_prm)
        plda = 2 * pg * pnu / (1 - 2 * pnu)
        pe = 2 * pg * (1 + pnu)
    elif prmcase == 7:
        # given -- shear_mod, bulk_mod
        pnames = (eky0, eky1)
        pvals = (pg, pm)
        pnu = (3 * pk - 2 * pg) / (6 * pk + 2 * pg)
        warn_negative_poisson(prmcase, pnu, pnames, pvals, blk_dbg_prm)
        check_iii(prmcase, eky0, pg, eky1, pk, blk_dbg_prm)
        plda = pk - 2 * pg / 3
        pe = 9 * pk * pg / (3 * pk + pg)
    elif prmcase == 8:
        # given -- shear_mod, long_mod
        # check for poisson_mod = NaN
        pnames = (eky0, eky1)
        pvals = (pg, pm)
        #                       # (pm = pg) --> div-zero
        if np.isclose(pm, pg, rtol=reltol, atol=abstol):
            term_nan_poisson(prmcase, pnames, pvals, blk_dbg_prm)
        pnu = (pm - 2 * pg) / (2 * pm - 2 * pg)
        warn_negative_poisson(prmcase, pnu, pnames, pvals, blk_dbg_prm)
        check_iv(prmcase, eky0, pg, ivar_pnms['pnu'], pnu, blk_dbg_prm)
        plda = pm - 2 * pg
        pe = pg * (3 * pm - 4 * pg) / (pm - pg)
    #
    # eky0 == 'youngs_mod'
    elif prmcase == 9:
        # given -- youngs_mod, poisson_ratio
        # Neg. poisson and FPEs not possible.
        check_v(prmcase, eky0, pe, eky1, pnu, blk_dbg_prm)
        plda = pe * pnu / ((1 + pnu) * (1 - 2 * pnu))
        pg = 0.5 * pe / (1 + pnu)
    elif prmcase == 10:
        # given -- youngs_mod, bulk_mod
        # check for lame_mod = NaN
        pnames = (eky0, eky1)
        pvals = (pe, pk)
        icb = 9 * pk            # (pe = icb) --> div-zero
        if np.isclose(pe, icb, rtol=reltol, atol=abstol):
            term_nan_lame(prmcase, pnames, pvals, blk_dbg_prm)
        plda = 3 * pk * (3 * pk - pe) / (9 * pk - pe)
        pnu = (3 * pk - pe) / (6 * pk)
        warn_negative_poisson(prmcase, pnu, pnames, pvals, blk_dbg_prm)
        check_v(prmcase, eky0, pe, ivar_pnms['pnu'], pnu, blk_dbg_prm)
        pg = 3 * pk * pe / (9 * pk - pe)
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
        pvals = (pe, pm)
        ips2 = pe**2 + 9 * pm**2 - 10*pe*pm
        if ips2 < 0.0:
            term_cmplx_poisson(prmcase, pnames, pvals, blk_dbg_prm)
        ips = pow(ips2, 0.5)
        pnu = 0.25 * (pe - pm + ips) / pm
        warn_negative_poisson(prmcase, pnu, pnames, pvals, blk_dbg_prm)
        check_v(prmcase, eky0, pe, ivar_pnms['pnu'], pnu, blk_dbg_prm)
        plda = 0.25 * (pm - pe + ips)
        pg = 0.125 * (3 * pm + pe - ips)
    #
    # eky0 == 'poisson_ratio'
    elif prmcase == 12:
        # given -- poisson_ratio, bulk_mod
        # Neg. poisson and FPEs not possible.
        pg = 3 * pk * (1 - 2 * pnu) / (2 * (1 + pnu))
        check_iv(prmcase, ivar_pnms['pg'], pg, eky0, pnu, blk_dbg_prm)
        plda = 3 * pk * pnu / (1 + pnu)
        pe = 3 * pk * (1 - 2 * pnu)
    elif prmcase == 13:
        # given -- poisson_ratio, long_mod
        # Neg. poisson and FPEs not possible.
        pg = 0.5 * pm * (1 - 2 * pnu) / (1 - pnu)
        check_iv(prmcase, ivar_pnms['pg'], pg, eky0, pnu, blk_dbg_prm)
        plda = pm * pnu / (1 - pnu)
        pe = 2 * (1 + pnu) * pg
    #
    # eky0 == 'bulk_mod'
    elif prmcase == 14:
        # given -- bulk_mod, long_mod
        pnames = (eky0, eky1)
        pvals = (pk, pm)
        pg = 0.75 * (pm - pk)
        check_iii(prmcase, ivar_pnms['pg'], pg, eky0, pk, blk_dbg_prm)
        pnu = (3 * pk - pm) / (3 * pk + pm)
        warn_negative_poisson(prmcase, pnu, pnames, pvals, blk_dbg_prm)
        plda = 0.5 * (3 * pk - pm)
        pe = 9 * pk * (pm - pk) / (3 * pk + pm)

    # end if prmcase

    # Verify that (plda, pg, pe, pnu) have been set.
    allset = not any((v for v in (plda, pg, pe, pnu) if isinstance(v,complex)))
    errmsg = (
        """Internal: one of the internal params has not yet been set!
        Contact ExactPack team!
        """ )
    assert allset, errmsg

    # One or both of bulk_modd, long_mod may not yet have been set.
    # plda and pg have been set.
    if isinstance(pk, complex):
        pk = plda + 2 * pg / 3
    if isinstance(pm, complex):
        pm = plda + 2 * pg

    # Provided inputs are as described in docstring, this dict contains the
    # correct name-value mapping.
    return dict(zip(elas_prm_names, [plda, pg, pe, pnu, pk, pm]))
