r"""Unittests for the spherical, isotropic, linear elastic Blake solver.
"""

from exactpack.solvers.blake import Blake

import warnings
import unittest
import numpy as np


class TestBlakeParamErrWarnChecks(unittest.TestCase):
    r"""Tests :class:`exactpack.solvers.blake.blake.Blake` instance parameter checks.
    """

    elas_dflt_prms = Blake.elas_prm_dflts

    def test_defaults(self):
        """Test that default param values are present in the default solver."""

        lame_mod = 25.0e9
        shear_mod = 25.0e9
        youngs_mod = 62.5e9
        poisson_ratio = 0.25
        bulk_mod = 41.66666666666667e9
        long_mod = 75.0e9
        # other dflts
        geometry = 3
        ref_density = 3000.0
        cavity_radius = 0.1
        pressure_scale = 1.0e6
        blake_debug = False

        slvr = Blake()
        self.assertEqual(slvr.lame_mod, lame_mod)
        self.assertEqual(slvr.shear_mod, shear_mod)
        self.assertEqual(slvr.youngs_mod, youngs_mod)
        self.assertEqual(slvr.poisson_ratio, poisson_ratio)
        self.assertEqual(slvr.bulk_mod, bulk_mod)
        self.assertEqual(slvr.long_mod, long_mod)
        self.assertEqual(slvr.geometry, geometry)
        self.assertEqual(slvr.ref_density, ref_density)
        self.assertEqual(slvr.cavity_radius, cavity_radius)
        self.assertEqual(slvr.pressure_scale, pressure_scale)
        self.assertEqual(slvr.blake_debug, blake_debug)

    def test_assign_non_elastic_params(self):
        """Test that the non-elastic parameters are passed to the instance."""

        geom = 3
        refd = 6000.0
        crad = 0.25
        pscl = 1.0e8
        dbgflag = False

        slvr = Blake(geometry=geom, ref_density=refd, cavity_radius=crad,
                     pressure_scale=pscl, blake_debug=dbgflag)
        self.assertEqual(slvr.geometry, geom)
        self.assertEqual(slvr.ref_density, refd)
        self.assertEqual(slvr.cavity_radius, crad)
        self.assertEqual(slvr.pressure_scale, pscl)
        self.assertEqual(slvr.blake_debug, dbgflag)

    def test_check_cavity_radius(self):
        """Test cavity_radius parameter positive check."""
        self.assertRaisesRegex(ValueError, "cavity_radius.*non-positive",
                               Blake, cavity_radius=-1.0)

    def test_check_geometry(self):
        """Test geometry parameter range check."""
        self.assertRaisesRegex(ValueError, "value of geometry",
                               Blake, geometry=4)

    def test_check_pressure_scale_pos(self):
        """Test pressure_scale parameter positive check."""
        self.assertRaisesRegex(ValueError, "pressure_scale.*non-positive",
                               Blake, pressure_scale=-1.0)

    def test_check_ref_density(self):
        """Test ref_density parameter positive check."""
        self.assertRaisesRegex(ValueError, "ref_density.*non-positive",
                               Blake, ref_density=-1.0)

    def test_pscale_warn_check(self):
        """Test the pressure_scale parameter range warning."""
        with warnings.catch_warnings(record=True) as wcm:
            # Default bulk_mod big enough to trigger this warning.
            big_pscale = self.elas_dflt_prms['bulk_mod']
            blk_slvr = Blake(pressure_scale=big_pscale)  # trigger
            # test
            assert len(wcm) > 0
            assert issubclass(wcm[-1].category, UserWarning)
            assert 'pressure_scale parameter' in str(wcm[-1].message)

    def test_check_blake_debug(self):
        """Test that blake_debug parameter is boolean."""
        self.assertRaisesRegex(ValueError, "blake_debug.*boolean.*",
                               Blake, blake_debug='string')


class TestBlakeSolution(unittest.TestCase):
    r"""Tests :class:`exactpack.solvers.blake.blake.Blake`
    to confirm proper solution values in some specific cases.
    """

    elas_dflt_keys = Blake.elas_prm_names
    elas_dflt_vals = Blake.elas_prm_dflt_vals
    elas_dflt_order = Blake.elas_prm_order
    elas_dflt_prms = Blake.elas_prm_dflts

    errmsg_pre = ('Current output and default standard output: ' +
                  'DO NOT AGREE for rel. tol. = ')

    # Blake default regression standard data.
    blk_std_dat0 = np.array([
        +1.000000000000000e-01, +1.000010017698277e-01, +1.001769827695929e-06,
        +7.000000000000000e-01, +7.000000690942526e-01, +6.909425264323670e-08,
        +8.900000000000000e-01, +8.900000140747887e-01, +1.407478865289037e-08,
    ]).reshape(3, 3)

    blk_std_dat1 = np.array([
        -2.001179885130619e-05, +1.001769827695929e-05, +2.359770261238576e-08,
        +2.956403204185389e-07, +9.870607520462385e-08, +4.930524708277866e-07,
        -1.333617468905843e-06, +1.581436927291053e-08, -1.301988730360022e-06,
    ]).reshape(3, 3)

    blk_std_dat2 = np.array([
        +2.999999929206894e+03, -9.999999999999998e+05, +5.014748564132740e+05,
        +2.999998520843317e+03, +2.710832779162161e+04, +1.726161553092586e+04,
        +3.000003905971277e+03, -9.923059170429272e+04, -3.175899979535503e+04,
    ]).reshape(3, 3)

    blk_std_dat3 = np.array([
        -9.832376088493813e+02, -1.000983237608849e+06, +5.004916188044246e+05,
        -2.054385295115777e+04, +6.564474840463838e+03, -3.282237420231915e+03,
        +5.424953043166759e+04, -4.498106127262513e+04, +2.249053063631256e+04,
    ]).reshape(3, 3)

    blk_std_dat4 = np.array([
        +1.501474856413274e+06,
        +9.846712260695753e+03,
        +6.747159190893768e+04,
    ]).reshape(3, 1)

    blk_dflt_std = np.hstack((blk_std_dat0, blk_std_dat1, blk_std_dat2,
                              blk_std_dat3, blk_std_dat4, ))
    blk_dflt_std_shp = blk_dflt_std.shape

    # cleanup
    for n in range(0, 5):
        stmt = 'del blk_std_dat' + str(n)
        exec(stmt, None, None)

    # blk_dflt_std is regression standard output for the LANL-default parameter
    # set, computed on intl_test_grid (3 points).  Several output fields have
    # local abs. maxima near these positions.
    intl_test_grid = np.array((0.1, 0.7, 0.89))

    def test_blake_dflt_regress(self):
        r"""Regression test of default solver instance.

        Compares current output on :attr:`TestBlakeSolution.intl_test_grid`
        (three points) using the default solver instance against regression
        standard output, :attr:`TestBlakeSolution.blk_dflt_std`.
        """

        print(' ')
        tsnap = Blake.tsnap_default
        grid = TestBlakeSolution.intl_test_grid
        blk_dflt_std = self.blk_dflt_std
        blk_dflt_std_shp = self.blk_dflt_std_shp
        curr_slvr = Blake()
        curr_soln = curr_slvr(grid, tsnap)

        # ###NOTE### rec_array.view() requires that both both dtype and type
        # be specified to get correct conversion, even if dtype is not changed.
        # recarray.view(..., type=np.ndarray) is a 1D array.
        curr_soln_2d = np.reshape(
            curr_soln.view(dtype=np.float64, type=np.ndarray),
            blk_dflt_std_shp)

        abstol = 0.0
        reltol = 1.0e-15
        errmsg = TestBlakeSolution.errmsg_pre + str(reltol)

        np.testing.assert_allclose(curr_soln_2d, blk_dflt_std,
                                   atol=abstol, rtol=reltol,
                                   verbose=True, err_msg=errmsg)

    def test_blake_non_dflt_regress(self):
        r"""Regression test of solver instance with param values given.

        Same as test_blake_dflt_regress() but a full set of default parameter
        values are supplied.
        """

        elas_dflt_keys = self.elas_dflt_keys
        elas_dflt_prms = self.elas_dflt_prms

        # Use dflt values of bulk_mod, long_mod as elas param values.
        kwargs = {k: elas_dflt_prms[k] for k in elas_dflt_keys[4:6]}
        # Specify all the non-elastic params (dflt values).
        kwargs.update({'geometry': 3, 'ref_density': 3000.0,
                       'cavity_radius': 0.1, 'pressure_scale': 1.0e6})

        tsnap = Blake.tsnap_default
        grid = TestBlakeSolution.intl_test_grid
        blk_dflt_std = self.blk_dflt_std
        blk_dflt_std_shp = self.blk_dflt_std_shp
        curr_slvr = Blake(**kwargs)
        curr_soln = curr_slvr(grid, tsnap)
        curr_soln_2d = np.reshape(
            curr_soln.view(dtype=np.float64, type=np.ndarray),
            blk_dflt_std_shp)

        abstol = 0.
        reltol = 1.0e-12        # NOT 1.0e-15 as in full-default case.
        errmsg = TestBlakeSolution.errmsg_pre + str(reltol)

        # There is some loss of accuracy in computing the solution
        # in this case.  Non-elastic params are specified exactly here,
        # so they aren't the cause.
        # However, the given elastic params here are NOT those used internally
        # for default soln calculation.  Calculation of the internal
        # elastic params causes this loss of accuracy.
        # This is dependent on which params are given here.
        np.testing.assert_allclose(curr_soln_2d, blk_dflt_std,
                                   atol=abstol, rtol=reltol,
                                   verbose=True, err_msg=errmsg)


class TestBlakeRunChecks(unittest.TestCase):
    r"""Tests :class:`exactpack.solvers.blake.blake.Blake` run-time checks."""

    def test_radii_positive_check(self):
        """Test execution radial coordinates positive check."""

        blk_slvr = Blake()
        radii = np.array([-1., 1.])
        tsnap = Blake.tsnap_default
        args = [radii, tsnap]
        kwargs = {}
        self.assertRaisesRegex(ValueError, "Minimum coordinate.*is negative",
                               blk_slvr, *args, **kwargs)


class TestBlakeVsKamm(unittest.TestCase):
    r"""
    .. |nbsp| unicode:: 0xA0
       :trim:

    Test the :class:`exactpack.solvers.blake.blake.Blake` solver against
    J. |nbsp| Kamm's F90 Blake solver output.

    J. Kamm's Fortran90 spherical Blake solver [#kam09]_ has been run using our
    default parameter set, Brock [#bro08]_ on a 16-point radial grid.  This
    test runs exactly the same problem on the same grid at 64-bit precision.
    Our grid differs from that of Brock in that (SI units):

    * Our domain extends to 1.2 rather than 1.0.
    * Our grid spacing is significantly larger: :math:`dr = 0.075`.

    The grid is "cell-center" in that the points are offset from domain the
    boundaries by :math:`dr/2 = 0.0375`.  A feature of the Brock setup is that
    the radius of the leading edge of the waveform is at exactly
    :math:`r = 0.9` for the snapshot at :math:`t = 0.00016`.

    The Kamm solver output fields include: position, displacement, stress_rad,
    stress_hoop, pressure, stress_diff = abs(stress_rad - stress_hoop); it does
    not separately calculate strains.  The Kamm fields are incorporated as
    source here and each field is also computed by the *ExactPack* Blake solver
    (EPB).  This test demonstrates that to within a small tolerance, the two
    solvers agree.  Kamm's solver was developed independently of EPB, in a
    different source language and using slightly different analytic forms for
    the stresses.

    .. [#bro08] Brock, Jerry S. *Blake Test Problem Parameters*. Report
       LA-UR-08-3005, Los Alamos National Laboratory (LANL), Los Alamos,
       NM 87545: LANL (2008).

    .. [#kam09] Kamm, James R., and Lee A. Ankeny. *Analysis of the Blake
       Problem with RAGE*. Report LA-UR-09-01255, Los Alamos National
       Laboratory (LANL), Los Alamos, NM 87545: LANL (2009).
    """

    def test_blake_vs_kamm(self):
        r"""Test spherical Blake fields and compare against Kamm output."""

        # kamm_sph_dat: Kamm F90 solver output data.
        kamm_sph_dat1 = np.array([
            #    0                      1                       2
            #    position               displacement            stress_rad
            3.75000000000000E-02,  0.00000000000000E+00,  0.00000000000000E+00,
            1.12500000000000E-01,  7.91729548518875E-07, -7.02702469218342E+05,
            1.87500000000000E-01,  2.84225705240826E-07, -1.54399885073962E+05,
            2.62500000000000E-01,  1.41455165224225E-07, -6.01559358293565E+04,
            3.37500000000000E-01,  7.96700674058918E-08, -3.15798436396484E+04,
            4.12500000000000E-01,  4.74304458774152E-08, -1.70103854094798E+04,
            4.87500000000000E-01,  3.27363284735378E-08, -3.40137748463715E+03,
            5.62500000000000E-01,  3.39585162727519E-08,  1.21503162121726E+04,
            6.37500000000000E-01,  4.99358394789389E-08,  2.55080127582440E+04,
            7.12500000000000E-01,  7.26569497152086E-08,  2.55556598291575E+04,
            7.87500000000000E-01,  8.17337333358558E-08, -2.65424005640929E+03,
            8.62500000000000E-01,  4.52710826644583E-08, -6.78530461070724E+04,
            9.37500000000000E-01,  0.00000000000000E+00,  0.00000000000000E+00,
            1.01250000000000E+00,  0.00000000000000E+00,  0.00000000000000E+00,
            1.08750000000000E+00,  0.00000000000000E+00,  0.00000000000000E+00,
            1.16250000000000E+00,  0.00000000000000E+00,  0.00000000000000E+00,
        ]).reshape(16, 3)

        kamm_sph_dat2 = np.array([
            #    3                      4                       5
            #    stress_hoop            pressure                stress_diff
            0.00000000000000E+00,  0.00000000000000E+00,  0.00000000000000E+00,
            3.52232175830090E+05, -5.87294147279051E+02,  1.05493464504843E+06,
            7.48559073046019E+04,  1.56269015491954E+03,  2.29255792378564E+05,
            2.48544230486987E+04,  3.48236324398638E+03,  8.50103588780552E+04,
            9.14500703515220E+03,  4.42994318978133E+03,  4.07248506748006E+04,
            3.91177978823708E+03,  3.06227527766854E+03,  2.09221651977169E+04,
            4.46216109034108E+03, -1.84098156534834E+03,  7.86353857497823E+03,
            9.08099670372448E+03, -1.01041032065405E+04,  3.06931950844813E+03,
            1.50302316356158E+04, -1.85228253431586E+04,  1.04777811226282E+04,
            1.70164421319658E+04, -1.98628480310297E+04,  8.53921769719173E+03,
            7.76432562520280E+03, -4.29147039799877E+03,  1.04185656816121E+04,
            -1.82436643869509E+04, 3.47801249603247E+04,  4.96093817201215E+04,
            0.00000000000000E+00,  0.00000000000000E+00,  0.00000000000000E+00,
            0.00000000000000E+00,  0.00000000000000E+00,  0.00000000000000E+00,
            0.00000000000000E+00,  0.00000000000000E+00,  0.00000000000000E+00,
            0.00000000000000E+00,  0.00000000000000E+00,  0.00000000000000E+00,
        ]).reshape(16, 3)

        # finalize kamm data: column index is leftmost
        kamm_sph_dat = np.hstack((kamm_sph_dat1, kamm_sph_dat2))
        del kamm_sph_dat1
        del kamm_sph_dat2

        # set the grid
        grid_dr = 0.075
        dom_max = 1.2
        npts = int(round(dom_max / grid_dr))
        rmin = grid_dr / 2.
        rmax = dom_max - rmin  # mesh limits: diff than domain.
        radii = np.linspace(rmin, rmax, npts)
        # snapshot time
        tsnap = Blake.tsnap_default

        # attr_to_kmprms
        # key   = solution (recarray) attribute name.
        # value = (ndx, rtol), ndx = kamm array col index,
        #                      rtol = tolerance per key.
        attribs = ('position', 'displacement', 'stress_rr', 'stress_qq',
                   'pressure', 'stress_diff')
        rtol_scale = 1.e-15
        rtols = [rtol_scale * m for m in (1, 10, 100, 100, 200, 100)]
        values = zip(range(6), rtols)
        attr_to_kmprms = dict(zip(attribs, values))

        # Solver and solution
        brock_blk_solver = Blake()
        soln = brock_blk_solver(radii, tsnap)

        abstol = 0.0
        print('\nAbs. tolerance = ', abstol)

        for ky in attr_to_kmprms:
            cmd = 'global pyth; pyth = soln.' + ky
            exec(cmd, None, None)
            kamm = kamm_sph_dat[:, attr_to_kmprms[ky][0]]
            errormsg = ('Blake solver and Kamm data for: ' + ky +
                        ', DO NOT agree!')
            np.testing.assert_allclose(
                pyth, kamm, rtol=attr_to_kmprms[ky][1], atol=abstol,
                verbose=True, err_msg=errormsg)
            # If here then OK
            okmsg = ('Blake solver and Kamm data agree for: ' + ky +
                     ', \twith rel. tol. = ' + str(attr_to_kmprms[ky][1]))
            print(okmsg)


if __name__ == '__main__':
    unittest.main()
