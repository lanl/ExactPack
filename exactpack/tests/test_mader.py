"""Unit tests for the Mader rarefaction solver.
"""

import pytest

import numpy as np

from exactpack.solvers.mader.timmes import Mader


class TestMaderTimmes():
    r"""Regression tests for :class:`exactpack.solvers.mader.timmes.Mader`

    The comparison is made at :math:`r=0.7` and :math:`t=6.25 \times 10^{-6}`.
    Note: Mader currently returns a NaN if the spatial interval is a single
    point; therefore, the following tests use :math:`r=[0.7,0.8]`, and the
    first spatial point is selected. This is because Timmes' solution code
    returns cell averaged values.
    """

    def test_velocity(self):
        """Regression test for velocity."""
        sol = Mader(p_cj=3.0e11, d_cj=8.0e5, gamma=3.0, u_piston=0.0)
        # r must contain 2 elements, otherwise the density and pressure are nan
        r = np.array([0.7, 0.8])
        t = 6.25e-6
        solrt = sol(r, t)
        np.testing.assert_allclose(solrt.velocity[0], 144000.0)

    def test_pressure(self):
        """Regression test for pressure."""
        sol = Mader(p_cj=3.0e11, d_cj=8.0e5, gamma=3.0, u_piston=0.0)
        r = np.array([0.7, 0.8])
        t = 6.25e-6
        solrt = sol(r, t)
        np.testing.assert_allclose(solrt.pressure[0], 223599111111.10834)

    def test_sound(self):
        """Regression test for sound speed."""
        sol = Mader(p_cj=3.0e11, d_cj=8.0e5, gamma=3.0, u_piston=0.0)
        r = np.array([0.7, 0.8])
        t = 6.25e-6
        solrt = sol(r, t)
        np.testing.assert_allclose(solrt.sound_speed[0], 544000.0)

    def test_density(self):
        """Regression test for density."""
        sol = Mader(p_cj=3.0e11, d_cj=8.0e5, gamma=3.0, u_piston=0.0)
        r = np.array([0.7, 0.8])
        t = 6.25e-6
        solrt = sol(r, t)
        np.testing.assert_allclose(solrt.density[0], 2.26666666666663)

    def test_xdet(self):
        """Regression test for xdet"""
        sol = Mader(p_cj=3.0e11, d_cj=8.0e5, gamma=3.0, u_piston=0.0)
        r = np.array([0.7, 0.8])
        t = 6.25e-6
        solrt = sol(r, t)
        np.testing.assert_allclose(solrt.xdet[0], 4.3)
