r"""Unittests for all code in :module:`blake.set_check_elastic_params`."""

import warnings
import unittest
import numpy as np

from exactpack.solvers.blake import Blake
from exactpack.solvers.blake import set_check_elastic_params as elas_prms_mod

class TestSetCheckElasErrsWarns(unittest.TestCase):
    """Exercise all exception-producing auxilliary funtions
    in :mod:`blake.set_check_elastic_params`.

    The debugging argument in each test in this class is set to the debug_prm
    class attribute.  This must have value True for full coverage.
    """

    # See comments in class TestSetElasPrms.
    dflt_keys = Blake.elas_prm_names                   # orig. tuple
    dflt_vals = Blake.elas_prm_dflt_vals               # orig. tuple
    dflt_order = Blake.elas_prm_order                  # orig. dict
    debug_prm = True

    def test_warn_neg_poisson(self):
        """Test set_check_elastic_params.warn_negative_poisson()."""

        dflt_vals = TestSetCheckElasErrsWarns.dflt_vals
        dflt_order = TestSetCheckElasErrsWarns.dflt_order
        prmcase = 3
        dbg_prm = TestSetCheckElasErrsWarns.debug_prm
        pnu = -0.1              # does NOT correspond to prmnames, prmvals
        prmnames = ['lame_mod', 'bulk_mod']
        prmvals = [dflt_vals[dflt_order['lame_mod']],
                   dflt_vals[dflt_order['bulk_mod']]]

        with warnings.catch_warnings(record=True) as wcm:
            # wcm: warning context manager; UserWarning triggered by default.
            # All asserts go *inside* scope of 'with ... as wcm:'
            elas_prms_mod.warn_negative_poisson(
                prmcase, pnu, prmnames, prmvals, dbg_prm)
            # Verify warning category and message.
            # *Last* element of wcm contains most recent warning.
            assert len(wcm) > 0
            assert issubclass(wcm[-1].category, UserWarning)
            assert 'Non-positive poisson_ratio' in str(wcm[-1].message)

    #
    # Test Pos. Defn. strain energy checks.
    #
    def test_check_ii(self):
        """Test set_check_elastic_params.check_ii()."""
        prmcase = 0
        dbg_prm = TestSetCheckElasErrsWarns.debug_prm
        args = [prmcase, 'lame_mod', 0.0, 'shear_mod', 0.0, dbg_prm]
        kwargs = {}
        repatrn = 'Given or calculated values of lame_mod.*and shear_mod.*'
        self.assertRaisesRegexp(ValueError, repatrn,
                                elas_prms_mod.check_ii, *args, **kwargs)

    def test_check_iii(self):
        """Test set_check_elastic_params.check_iii()."""
        prmcase = 7
        dbg_prm = TestSetCheckElasErrsWarns.debug_prm
        args = [prmcase, 'shear_mod', 0.0, 'bulk_mod', 0.0, dbg_prm]
        kwargs = {}
        repatrn = 'Given or calculated values of shear_mod.*and bulk_mod.*'
        self.assertRaisesRegexp(ValueError, repatrn,
                                elas_prms_mod.check_iii, *args, **kwargs)

    def test_check_iv(self):
        """Test set_check_elastic_params.check_iv()."""
        prmcase = 7
        dbg_prm = TestSetCheckElasErrsWarns.debug_prm
        args = [prmcase, 'shear_mod', 0.0, 'poisson_ratio', 1.0, dbg_prm]
        kwargs = {}
        repatrn = 'Given or calculated values of shear_mod.*and poisson.*'
        self.assertRaisesRegexp(ValueError, repatrn,
                                elas_prms_mod.check_iv, *args, **kwargs)

    def test_check_v(self):
        """Test set_check_elastic_params.check_v()."""
        prmcase = 9
        dbg_prm = TestSetCheckElasErrsWarns.debug_prm
        args = [prmcase, 'youngs_mod', 0.0, 'poisson_ratio', 1.0, dbg_prm]
        kwargs = {}
        repatrn = 'Given or calculated values of youngs_mod.*and poisson.*'
        self.assertRaisesRegexp(ValueError, repatrn,
                                elas_prms_mod.check_v, *args, **kwargs)

    #
    # Test other parameter value checks.
    #

    def test_term_cmplx_poisson(self):
        """Test set_check_elastic_params.term_cmplx_poisson()."""

        dflt_vals = TestSetCheckElasErrsWarns.dflt_vals
        dflt_order = TestSetCheckElasErrsWarns.dflt_order
        dbg_prm = TestSetCheckElasErrsWarns.debug_prm
        args = [TestSetCheckElasErrsWarns.dflt_keys, dflt_vals, dflt_order,
                False, dbg_prm]
        long_val = 35.0e9  # will produce cmplx poisson_mod w/ dflt youngs_mod
        kwargs = dict(zip(('long_mod', 'youngs_mod'),
                          (long_val,  dflt_vals[dflt_order['youngs_mod']])))
        self.assertRaisesRegexp(
            ValueError, '.*complex value for Poisson.*',
            elas_prms_mod.set_elastic_params, *args, **kwargs)

    # In these test_nan_*_pcN() fcns, N is the prmcase var in
    # set_check_elastic_params.set_elastic_params()

    def test_term_nan_lame_pc5(self):
        """Test term_nan_lame() given: shear_mod, youngs_mod."""

        dflt_vals = TestSetCheckElasErrsWarns.dflt_vals
        dflt_order = TestSetCheckElasErrsWarns.dflt_order
        dbg_prm = TestSetCheckElasErrsWarns.debug_prm
        args = [TestSetCheckElasErrsWarns.dflt_keys, dflt_vals, dflt_order,
                False, dbg_prm]
        youngs_val = 3 * dflt_vals[dflt_order['shear_mod']]
        kwargs = dict(zip(('youngs_mod', 'shear_mod'),
                          (youngs_val, dflt_vals[dflt_order['shear_mod']])))
        self.assertRaisesRegexp(
            ValueError, '.*NaN value for the Lame.*',
            elas_prms_mod.set_elastic_params, *args, **kwargs)

    def test_term_nan_lame_pc10(self):
        """Test term_nan_lame() given: youngs_mod, bulk_mod."""

        dflt_vals = TestSetCheckElasErrsWarns.dflt_vals
        dflt_order = TestSetCheckElasErrsWarns.dflt_order
        dbg_prm = TestSetCheckElasErrsWarns.debug_prm
        args = [TestSetCheckElasErrsWarns.dflt_keys, dflt_vals, dflt_order,
                False, dbg_prm]
        youngs_val = 9 * dflt_vals[dflt_order['bulk_mod']]
        kwargs = dict(zip(('youngs_mod', 'bulk_mod'),
                          (youngs_val, dflt_vals[dflt_order['bulk_mod']])))
        self.assertRaisesRegexp(
            ValueError, '.*NaN value for the Lame.*',
            elas_prms_mod.set_elastic_params, *args, **kwargs)

    def test_term_nan_poisson_pc3(self):
        """Test term_nan_poisson() given: lame_mod, bulk_mod."""

        dflt_vals = TestSetCheckElasErrsWarns.dflt_vals
        dflt_order = TestSetCheckElasErrsWarns.dflt_order
        dbg_prm = TestSetCheckElasErrsWarns.debug_prm
        args = [TestSetCheckElasErrsWarns.dflt_keys, dflt_vals, dflt_order,
                False, dbg_prm]
        lame_val = 3 * dflt_vals[dflt_order['bulk_mod']]
        kwargs = dict(zip(('lame_mod', 'bulk_mod'),
                          (lame_val,  dflt_vals[dflt_order['bulk_mod']])))
        self.assertRaisesRegexp(
            ValueError, '.*a NaN value for the Poisson.*',
            elas_prms_mod.set_elastic_params, *args, **kwargs)

    def test_term_nan_poisson_pc8(self):
        """Test term_nan_poisson() given: shear_mod, long_mod."""
        """Produce a call to term_nan_poisson, given shear_mod, long_mod."""

        dflt_vals = TestSetCheckElasErrsWarns.dflt_vals
        dflt_order = TestSetCheckElasErrsWarns.dflt_order
        dbg_prm = TestSetCheckElasErrsWarns.debug_prm
        args = [TestSetCheckElasErrsWarns.dflt_keys, dflt_vals, dflt_order,
                False, dbg_prm]
        long_val = dflt_vals[dflt_order['shear_mod']]
        kwargs = dict(zip(('long_mod', 'shear_mod'),
                          (long_val,  dflt_vals[dflt_order['shear_mod']])))
        self.assertRaisesRegexp(
            ValueError, '.*a NaN value for the Poisson.*',
            elas_prms_mod.set_elastic_params, *args, **kwargs)


class TestSetElasticParams(unittest.TestCase):
    """Exercise all normal execution paths through
    :func:`blake.set_check_elastic_params.set_elastic_params`.

    The primary test is :func:`test_set_elastic_params` which invokes
    set_elastic_params() with each unique pair of elastic parameters from the
    default parameter set as arguments.  If agreement is not obtained (to
    within a small tolerance) an exception is issued.  A few other test
    functions cover the sanity checks near the top of the tested function.
    """

    # Get default elas param names, values and param order directly from the
    # class Blake.  These are the tuples defined as class Blake attributes and
    # passed to set_elastic_params() from Blake.__init__.
    #
    # Use the elas_prm_order dict, which encodes the Blake attribute tuple
    # ordering explicitly, if sequential access to elas_prm_dflts is needed.

    # Make these Blake.* available to this class.
    elas_prm_dflts = Blake.elas_prm_dflts               # orig. dict
    dflt_order = Blake.elas_prm_order                   # orig. dict
    dflt_keys = Blake.elas_prm_names                   # orig. tuple
    dflt_vals = Blake.elas_prm_dflt_vals               # orig. tuple
    debug_prm = False

    def test_set_elastic_params(self):
        r"""Test normal function of set_elastic_params().

        The main loop invokes set_elastic_params() with each pair of parameters
        from the default parameter set and checks that the entire default set
        is reproduced to within a small tolerance.  A few other invocations
        cover the remaining normal use cases.
        """

        # puts blank line after test title.
        print ' '
        # Local refs to TestSetElasPrms.* objects
        dflt_keys = TestSetElasticParams.dflt_keys               # orig. tuple
        dflt_vals = TestSetElasticParams.dflt_vals               # orig. tuple
        dflt_order = TestSetElasticParams.dflt_order             # orig. dict

        elas_prm_dflts = TestSetElasticParams.elas_prm_dflts     # orig. dict

        # Setup for testing multiple cases.
        abstol = 0.0
        reltol = 1.0e-15

        # TODO(?): move msgs inside loop w/format to include prmcase.
        errmsg = ('Current default output and default standard output: ' +
                  'DO NOT agree for rel. tol. = ' + str(reltol))

        # Loop over 15 distinct two-elas param cases,
        # calling set_elastic_params() directly.
        # This loop covers most of the normal-function code in
        # set_elastic_params().
        elas_dflt = False
        dbg_prm = TestSetElasticParams.debug_prm
        blk_cnt = 0
        done_eky0 = []
        for eky0 in dflt_keys:
            blk_cnt += 1
            done_eky0.append(eky0)
            for eky1 in dflt_keys:
                if eky1 not in done_eky0:
                    # print '\nIn test_check_elas_params: blk_cnt, eky0, eky1'
                    # print str(blk_cnt) + ', ' + eky0 + ', ' + eky1      #dbg
                    # This use of elas_prm_dflts is OK as we access by key.
                    kwargs = {k: v for k, v in elas_prm_dflts.iteritems()
                              if k in (eky0, eky1)}
                    kwargs.update({'geometry': 3})
                    # print '\nIn test_check_elas_params: kwargs = '      #dbg
                    # print kwargs                                        #dbg
                    ret_params = elas_prms_mod.set_elastic_params(
                        dflt_keys, dflt_vals, dflt_order,
                        elas_dflt, dbg_prm, **kwargs)
                    # print '\nIn test_check_elas_params: ret_params = '  #dbg
                    # print ret_params                                    #dbg

                    # Put full set elas param values into main kwargs dict.
                    blk_vals = [ret_params[k] for k in dflt_keys]
                    np.testing.assert_allclose(blk_vals, dflt_vals,
                                               atol=abstol, rtol=reltol,
                                               verbose=True, err_msg=errmsg)
                    del [blk_vals, ret_params, kwargs]      # cleanup

        # Cover the defaulted True case
        elas_dflt = True
        kwargs = {}
        ret_params = elas_prms_mod.set_elastic_params(
            dflt_keys, dflt_vals, dflt_order,
            elas_dflt, dbg_prm, **kwargs)

    #
    # Handle the other special cases in set_elastic_params().
    #
    def test_check_len_elas_prm_dflts(self):
        """Test set_elastic_params default args length check."""

        # Local copy of elas_prms_mod.set_elastic_params() input prms.
        dflt_order = TestSetElasticParams.dflt_order         # orig. dict
        dflt_keys = TestSetElasticParams.dflt_keys          # orig. tuple
        dflt_vals = TestSetElasticParams.dflt_vals          # orig. tuple

        wrong = list(dflt_keys)
        wrong.append(dflt_keys[0])
        dbg_prm = TestSetElasticParams.debug_prm
        args = [wrong, dflt_vals, dflt_order, False, dbg_prm]
        kwargs = dict(zip(dflt_keys[0:2], dflt_vals[0:2]))

        # Appears that to have assertRaisesRegexp() use regex
        # with any RE flag value, it must be compiled.
        # Stay with exact matches for now.
        self.assertRaisesRegexp(ValueError, 'Length of default',
                                elas_prms_mod.set_elastic_params,
                                *args, **kwargs)

    def test_check_len_elas_prm_args(self):
        """Test set_elastic_params elas param length check."""

        dflt_order = TestSetElasticParams.dflt_order         # orig. dict
        dflt_keys = TestSetElasticParams.dflt_keys          # orig. tuple
        dflt_vals = TestSetElasticParams.dflt_vals          # orig. tuple

        dbg_prm = TestSetElasticParams.debug_prm
        args = [dflt_keys, dflt_vals, dflt_order, False, dbg_prm]
        kwargs = dict(zip(dflt_keys[0:1], dflt_vals[0:1]))      # len != 2
        self.assertRaisesRegexp(
            ValueError, 'Invalid number of elastic',
            elas_prms_mod.set_elastic_params, *args, **kwargs)

    def test_check_given_poisson_range(self):
        """Test given poisson_ratio range check."""

        dflt_vals = TestSetElasticParams.dflt_vals
        dbg_prm = TestSetElasticParams.debug_prm
        args = [TestSetElasticParams.dflt_keys, dflt_vals,
                TestSetElasticParams.dflt_order, False, dbg_prm]
        bad_poisson = 10.0
        kwargs = dict(zip(('lame_mod', 'poisson_ratio'),
                          (dflt_vals[0], bad_poisson)))
        self.assertRaisesRegexp(
            ValueError, 'Specified value of poisson_ratio',
            elas_prms_mod.set_elastic_params, *args, **kwargs)

    def test_check_given_moduli_range(self):
        """Test given moduli range check."""

        dflt_vals = TestSetElasticParams.dflt_vals
        dbg_prm = TestSetElasticParams.debug_prm
        args = [TestSetElasticParams.dflt_keys, dflt_vals,
                TestSetElasticParams.dflt_order, False, dbg_prm]
        bad_lame = -1.0
        kwargs = dict(zip(('lame_mod', 'shear_mod'),
                          (bad_lame, dflt_vals[1])))
        self.assertRaisesRegexp(
            ValueError, 'Specified value of lame_mod is non-positive',
            elas_prms_mod.set_elastic_params, *args, **kwargs)


if __name__ == '__main__':
    unittest.main()
