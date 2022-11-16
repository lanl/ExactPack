"""Unittests for the 1D Riemann solvers.
"""

import unittest
import pytest
import warnings
from pytest import approx
from numpy import array, interp, diff, sqrt, abs, argmin

import numpy.random

from exactpack.solvers.riemann.riemann import *

warnings.simplefilter('ignore', category=RuntimeWarning)

class TestRiemannSetup():
    """Tests problem setup of :class:`exactpack.solvers.riemann.riemann`.

       These tests confirm proper assignment of variables, including default
       values.
    """

    def test_defaults(self):
        """Test that default values are set accurately and problem is initialized correctly.
        """

        # here are the defaults
        xmin, xd0, xmax, t = 0.0, 0.5, 1.0, 0.25
        rl, ul, pl, gl = 1.0,   0.0, 1.0, 1.4
        rr, ur, pr, gr = 0.125, 0.0, 0.1, 1.4
        A, B, R1, R2, r0, e0  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        problem = 'igeos'
        num_x_pts, num_int_pts, int_tol = 10001, 10001, 1.e-12

        soln = SetupRiemannProblem(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                                   rl=rl, ul=ul, pl=pl, gl=gl,
                                   rr=rr, ur=ur, pr=pr, gr=gr)

        assert soln.xmin == xmin
        assert soln.xd0 == xd0
        assert soln.xmax == xmax
        assert soln.t == t
        assert soln.rl == rl
        assert soln.ul == ul
        assert soln.pl == pl
        assert soln.gl == gl
        assert soln.rr == rr
        assert soln.ur == ur
        assert soln.pr == pr
        assert soln.gr == gr
        assert soln.A ==  A
        assert soln.B ==  B
        assert soln.R1 == R1
        assert soln.R2 == R2
        assert soln.r0 == r0
        assert soln.e0 == e0
        assert soln.problem == problem
        assert soln.num_int_pts == num_int_pts
        assert soln.num_x_pts == num_x_pts
        assert soln.int_tol == int_tol


# Determines density, pressure, and velocity values in a rarefaction zone.
# Copied from exactpack.solvers.riemann.utils.
def rho_p_u_rarefaction(p, r, u, g, x, xd0, t, self):
  sgn = 1 if ((p == self.pl) and (r == self.rl)  and (u == self.ul)) else -1
  a = sqrt(g * p / r)
  y = 2. / (g + 1.) + sgn * (g - 1.) / a / (g + 1.) * (u - (x - xd0) / t)
  v = 2. * (sgn * a + (g - 1.) * u / 2. + (x - xd0) / t) / (g + 1.)
  return r * y**(2. / (g - 1.)), p * y**(2. * g / (g - 1.)), v


class Test_Riemann1():
    """Tests ideal-gas EOS (IGEOS) and general EOS (GenEOS) solutions produced by :class:`exactpack.solvers.riemann.riemann` on the first Riemann problem (Sod).
    """

    # Riemann Problem 1
    xmin, xd0, xmax, t = 0.0, 0.5, 1.0, 0.25
    rl, ul, pl, gl = 1.0,   0.0, 1.0, 1.4
    rr, ur, pr, gr = 0.125, 0.0, 0.1, 1.4
    A, B, R1, R2, r0, e0  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    problem = 'igeos'
    num_x_pts, num_int_pts, int_tol = 10001, 10001, 1.e-12

    soln_ig = RiemannIGEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                           rl=rl, ul=ul, pl=pl, gl=gl,
                           rr=rr, ur=ur, pr=pr, gr=gr)
    soln_ig.driver()

    soln_gen = RiemannGenEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                            rl=rl, ul=ul, pl=pl, gl=gl,
                            rr=rr, ur=ur, pr=pr, gr=gr)
    soln_gen.driver()

    # Test that star state values are computed correctly.
    pstar =  0.30313017805042364
    ustar =  0.9274526200494746
    rstar1 = 0.42631942817827095
    rstar2 = 0.26557371170518734
    estar1 = 1.7776000694229792
    estar2 = 2.853540887990146
    astar1 = 0.9977254326100283
    astar2 = 1.26411348275164

    # Test that spatial region boundaries are computed correctly.
    # Xregs = Vregs * t + xd0
    Xregs = array([0.2041960108450192, 0.4824317968598616,
                   0.7318631550123686, 0.9380389330073916])
    Vregs = array([-1.1832159566199232, -0.07027281256055373,
                    0.9274526200494746, 1.7521557320295664])

    def test_riem1ig_star_states(self):
        """Using the ideal-gas EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_ig.px,  abs=1.e-12)
        assert self.ustar  == approx(self.soln_ig.ux,  abs=1.e-12)
        assert self.rstar1 == approx(self.soln_ig.rx1, abs=1.e-12)
        assert self.rstar2 == approx(self.soln_ig.rx2, abs=1.e-12)
        assert self.estar1 == approx(self.soln_ig.ex1, abs=1.e-12)
        assert self.estar2 == approx(self.soln_ig.ex2, abs=1.e-12)
        assert self.astar1 == approx(self.soln_ig.ax1, abs=1.e-12)
        assert self.astar2 == approx(self.soln_ig.ax2, abs=1.e-12)

    def test_riem1gen_star_states(self):
        """Using the general EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar        == approx(self.soln_gen.px,  abs=1.e-6)
        assert self.soln_gen.ux1 == approx(self.soln_gen.ux2, abs=1.e-10)
        assert self.ustar        == approx(self.soln_gen.ux1, abs=1.e-6)
        assert self.ustar        == approx(self.soln_gen.ux2, abs=1.e-6)
        assert self.rstar1       == approx(self.soln_gen.rx1, abs=1.e-6)
        assert self.rstar2       == approx(self.soln_gen.rx2, abs=1.e-6)
        assert self.estar1       == approx(self.soln_gen.ex1, abs=1.e-6)
        assert self.estar2       == approx(self.soln_gen.ex2, abs=1.e-5)
        assert self.astar1       == approx(self.soln_gen.ax1, abs=1.e-6)
        assert self.astar2       == approx(self.soln_gen.ax2, abs=1.e-6)

    def test_riem1ig_region_boundaries(self):
        """Using the ideal-gas EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_ig.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_ig.Xregs[1], abs=1.e-12)
        assert self.Xregs[2] == approx(self.soln_ig.Xregs[2], abs=1.e-12)
        assert self.Xregs[3] == approx(self.soln_ig.Xregs[3], abs=1.e-12)
        assert self.Vregs[0] == approx(self.soln_ig.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_ig.Vregs[1], abs=1.e-12)
        assert self.Vregs[2] == approx(self.soln_ig.Vregs[2], abs=1.e-12)
        assert self.Vregs[3] == approx(self.soln_ig.Vregs[3], abs=1.e-12)

    def test_riem1gen_region_boundaries(self):
        """Using the general EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that velocity and spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_gen.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_gen.Xregs[1], abs=1.e-6)
        assert self.Xregs[2] == approx(self.soln_gen.Xregs[2], abs=1.e-6)
        assert self.Xregs[3] == approx(self.soln_gen.Xregs[3], abs=1.e-6)
        assert self.Vregs[0] == approx(self.soln_gen.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_gen.Vregs[1], abs=1.e-5)
        assert self.Vregs[2] == approx(self.soln_gen.Vregs[2], abs=1.e-6)
        assert self.Vregs[3] == approx(self.soln_gen.Vregs[3], abs=1.e-5)

    def test_riem1ig_state0(self):
        """Using the ideal-gas EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test that any point in (xmin,Xregs[0]) returns left state values.
        x = diff([self.xmin, self.Xregs[0]])[0] *numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem1gen_state0(self):
        """Using the general EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x = diff([self.xmin, self.Xregs[0]])[0] *numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem1ig_state1(self):
        """Using the ideal-gas EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pl, self.rl, self.ul, self.gl,
                                      x, self.xd0, self.t, self.soln_ig)

        r_soln = interp(x, self.soln_ig.x, self.soln_ig.r)
        u_soln = interp(x, self.soln_ig.x, self.soln_ig.u)
        p_soln = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(r_soln, abs=1.e-12)
        assert u == approx(u_soln, abs=1.e-12)
        assert p == approx(p_soln, abs=1.e-12)

    def test_riem1gen_state1(self):
        """Using the general EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pl, self.rl, self.ul, self.gl,
                                      x, self.xd0, self.t, self.soln_gen)

        r_soln = interp(x, self.soln_gen.x, self.soln_gen.r)
        u_soln = interp(x, self.soln_gen.x, self.soln_gen.u)
        p_soln = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(r_soln, abs=1.e-8)
        assert u == approx(u_soln, abs=1.e-8)
        assert p == approx(p_soln, abs=1.e-8)

    def test_riem1ig_state2(self):
        """Using the ideal-gas EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any po2nt in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        r_soln = interp(x, self.soln_ig.x, self.soln_ig.r)
        u_soln = interp(x, self.soln_ig.x, self.soln_ig.u)
        p_soln = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(r_soln, abs=1.e-12)
        assert u == approx(u_soln, abs=1.e-12)
        assert p == approx(p_soln, abs=1.e-12)

    def test_riem1gen_state2(self):
        """Using the general EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        r_soln = interp(x, self.soln_gen.x, self.soln_gen.r)
        u_soln = interp(x, self.soln_gen.x, self.soln_gen.u)
        p_soln = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(r_soln, abs=1.e-6)
        assert u == approx(u_soln, abs=1.e-6)
        assert p == approx(p_soln, abs=1.e-6)

    def test_riem1ig_state3(self):
        """Using the ideal-gas EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        r_soln = interp(x, self.soln_ig.x, self.soln_ig.r)
        u_soln = interp(x, self.soln_ig.x, self.soln_ig.u)
        p_soln = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(r_soln, abs=1.e-12)
        assert u == approx(u_soln, abs=1.e-12)
        assert p == approx(p_soln, abs=1.e-12)

    def test_riem1gen_state3(self):
        """Using the general EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        r_soln = interp(x, self.soln_gen.x, self.soln_gen.r)
        u_soln = interp(x, self.soln_gen.x, self.soln_gen.u)
        p_soln = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(r_soln, abs=1.e-6)
        assert u == approx(u_soln, abs=1.e-6)
        assert p == approx(p_soln, abs=1.e-6)

    def test_riem1ig_state4(self):
        """Using the ideal-gas EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],Xregs[4]) returns correct values.
        x= diff([self.Xregs[3], self.xmax])[0]*numpy.random.rand()+self.Xregs[3]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]

        r_soln = interp(x, self.soln_ig.x, self.soln_ig.r)
        u_soln = interp(x, self.soln_ig.x, self.soln_ig.u)
        p_soln = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rr == approx(r_soln, abs=1.e-12)
        assert self.ur == approx(u_soln, abs=1.e-12)
        assert self.pr == approx(p_soln, abs=1.e-12)

    def test_riem1gen_state4(self):
        """Using the general EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],Xregs[4]) returns correct values.
        x = diff([self.Xregs[3], self.xmax])[0] * numpy.random.rand() + self.Xregs[3]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]

        r_soln = interp(x, self.soln_gen.x, self.soln_gen.r)
        u_soln = interp(x, self.soln_gen.x, self.soln_gen.u)
        p_soln = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rr == approx(r_soln, abs=1.e-12)
        assert self.ur == approx(u_soln, abs=1.e-12)
        assert self.pr == approx(p_soln, abs=1.e-12)

class Test_Riemann1_reversed():
    """Tests ideal-gas EOS (IGEOS) and general EOS (GenEOS) solutions produced by :class:`exactpack.solvers.riemann.riemann` on the first Riemann problem (Sod), with left and right states reversed.
    """

    # Riemann Problem 1 reversed
    xmin, xd0, xmax, t = 0.0, 0.5, 1.0, 0.25
    rl, ul, pl, gl = 0.125, 0.0, 0.1, 1.4
    rr, ur, pr, gr = 1.0,   0.0, 1.0, 1.4
    A, B, R1, R2, r0, e0  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    problem = 'igeos'
    num_x_pts, num_int_pts, int_tol = 10001, 10001, 1.e-12

    soln_ig = RiemannIGEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                           rl=rl, ul=ul, pl=pl, gl=gl,
                           rr=rr, ur=ur, pr=pr, gr=gr)
    soln_ig.driver()

    soln_gen = RiemannGenEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                             rl=rl, ul=ul, pl=pl, gl=gl,
                             rr=rr, ur=ur, pr=pr, gr=gr)
    soln_gen.driver()

    # Test that star state values are computed correctly.
    pstar =  0.30313017805042364
    ustar =  -0.9274526200482547
    rstar1 = 0.26557371170518734
    rstar2 = 0.42631942817827095
    estar1 = 2.853540887990146
    estar2 = 1.7776000694229792
    astar1 = 1.26411348275164
    astar2 = 0.9977254326100283

    # Test that spatial region boundaries are computed correctly.
    # Xregs = Vregs * t + xd0
    Xregs = array([0.06196106699260839, 0.26813684498793633,
                   0.5175682031404434,  0.7958039891549809])
    Vregs = array([-1.7521557320295664, -0.9274526200482547,
                    0.07027281256177365, 1.1832159566199232])

    def test_riem1revig_star_states(self):
        """Using the ideal-gas EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_ig.px,  abs=1.e-12)
        assert self.ustar  == approx(self.soln_ig.ux,  abs=1.e-12)
        assert self.rstar1 == approx(self.soln_ig.rx1, abs=1.e-12)
        assert self.rstar2 == approx(self.soln_ig.rx2, abs=1.e-12)
        assert self.estar1 == approx(self.soln_ig.ex1, abs=1.e-12)
        assert self.estar2 == approx(self.soln_ig.ex2, abs=1.e-12)
        assert self.astar1 == approx(self.soln_ig.ax1, abs=1.e-12)
        assert self.astar2 == approx(self.soln_ig.ax2, abs=1.e-12)

    def test_riem1revgen_star_states(self):
        """Using the general EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar        == approx(self.soln_gen.px,  abs=1.e-6)
        assert self.soln_gen.ux1 == approx(self.soln_gen.ux2, abs=1.e-10)
        assert self.ustar        == approx(self.soln_gen.ux1, abs=1.e-6)
        assert self.ustar        == approx(self.soln_gen.ux2, abs=1.e-6)
        assert self.rstar1       == approx(self.soln_gen.rx1, abs=1.e-6)
        assert self.rstar2       == approx(self.soln_gen.rx2, abs=1.e-6)
        assert self.estar1       == approx(self.soln_gen.ex1, abs=1.e-5)
        assert self.estar2       == approx(self.soln_gen.ex2, abs=1.e-6)
        assert self.astar1       == approx(self.soln_gen.ax1, abs=1.e-6)
        assert self.astar2       == approx(self.soln_gen.ax2, abs=1.e-6)

    def test_riem1revig_region_boundaries(self):
        """Using the ideal-gas EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0

        assert self.Xregs[0] == approx(self.soln_ig.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_ig.Xregs[1], abs=1.e-12)
        assert self.Xregs[2] == approx(self.soln_ig.Xregs[2], abs=1.e-12)
        assert self.Xregs[3] == approx(self.soln_ig.Xregs[3], abs=1.e-12)
        assert self.Vregs[0] == approx(self.soln_ig.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_ig.Vregs[1], abs=1.e-12)
        assert self.Vregs[2] == approx(self.soln_ig.Vregs[2], abs=1.e-12)
        assert self.Vregs[3] == approx(self.soln_ig.Vregs[3], abs=1.e-12)

    def test_riem1revgen_region_boundaries(self):
        """Using the general EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0

        assert self.Xregs[0] == approx(self.soln_gen.Xregs[0], abs=1.e-5)
        assert self.Xregs[1] == approx(self.soln_gen.Xregs[1], abs=1.e-6)
        assert self.Xregs[2] == approx(self.soln_gen.Xregs[2], abs=1.e-6)
        assert self.Xregs[3] == approx(self.soln_gen.Xregs[3], abs=1.e-12)
        assert self.Vregs[0] == approx(self.soln_gen.Vregs[0], abs=1.e-5)
        assert self.Vregs[1] == approx(self.soln_gen.Vregs[1], abs=1.e-6)
        assert self.Vregs[2] == approx(self.soln_gen.Vregs[2], abs=1.e-5)
        assert self.Vregs[3] == approx(self.soln_gen.Vregs[3], abs=1.e-12)

    def test_riem1revig_state0(self):
        """Using the ideal-gas EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test that any point in (xmin,Xregs[0]) returns left state values.
        x=diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem1revgen_state0(self):
        """Using the general EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test that any point in (xmin,Xregs[0]) returns left state values.
        x=diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem1revig_state1(self):
        """Using the ideal-gas EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem1revgen_state1(self):
        """Using the general EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-6)
        assert u == approx(ul, abs=1.e-6)
        assert p == approx(pl, abs=1.e-6)

    def test_riem1revig_state2(self):
        """Using the ideal-gas EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem1revgen_state2(self):
        """Using the general EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-6)
        assert u == approx(ul, abs=1.e-6)
        assert p == approx(pl, abs=1.e-6)

    def test_riem1revig_state3(self):
        """Using the ideal-gas EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pr, self.rr, self.ur, self.gr, x,
                                      self.xd0, self.t, self.soln_ig)

        r_soln = interp(x, self.soln_ig.x, self.soln_ig.r)
        u_soln = interp(x, self.soln_ig.x, self.soln_ig.u)
        p_soln = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(r_soln, abs=1.e-12)
        assert u == approx(u_soln, abs=1.e-12)
        assert p == approx(p_soln, abs=1.e-12)

    def test_riem1revgen_state3(self):
        """Using the general EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pr, self.rr, self.ur, self.gr, x,
                                      self.xd0, self.t, self.soln_gen)

        r_soln = interp(x, self.soln_gen.x, self.soln_gen.r)
        u_soln = interp(x, self.soln_gen.x, self.soln_gen.u)
        p_soln = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(r_soln, abs=1.e-8)
        assert u == approx(u_soln, abs=1.e-8)
        assert p == approx(p_soln, abs=1.e-8)

    def test_riem1revig_state4(self):
        """Using the ideal-gas EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],Xregs[4]) returns correct values.
        x = diff([self.Xregs[3], self.xmax])[0] * numpy.random.rand() + self.Xregs[3]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]

        r_soln = interp(x, self.soln_ig.x, self.soln_ig.r)
        u_soln = interp(x, self.soln_ig.x, self.soln_ig.u)
        p_soln = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rr == approx(r_soln, abs=1.e-12)
        assert self.ur == approx(u_soln, abs=1.e-12)
        assert self.pr == approx(p_soln, abs=1.e-12)

    def test_riem1revgen_state4(self):
        """Using the general EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],Xregs[4]) returns correct values.
        x = diff([self.Xregs[3], self.xmax])[0] * numpy.random.rand() + self.Xregs[3]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]

        r_soln = interp(x, self.soln_gen.x, self.soln_gen.r)
        u_soln = interp(x, self.soln_gen.x, self.soln_gen.u)
        p_soln = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rr == approx(r_soln, abs=1.e-12)
        assert self.ur == approx(u_soln, abs=1.e-12)
        assert self.pr == approx(p_soln, abs=1.e-12)


class Test_Riemann2():
    """Tests ideal-gas EOS (IGEOS) and general EOS (GenEOS) solutions produced by :class:`exactpack.solvers.riemann.riemann` on the second Riemann problem (Einfeldt) are correct.
    """

    # Riemann Problem 2
    xmin, xd0, xmax, t = 0.0, 0.5, 1.0, 0.15
    rl, ul, pl, gl = 1., -2.0, 0.4, 1.4
    rr, ur, pr, gr = 1.,  2.0, 0.4, 1.4
    A, B, R1, R2, r0, e0  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    problem = 'igeos'
    num_x_pts, num_int_pts, int_tol = 10001, 10001, 1.e-12

    soln_ig = RiemannIGEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                           rl=rl, ul=ul, pl=pl, gl=gl,
                           rr=rr, ur=ur, pr=pr, gr=gr)
    soln_ig.driver()

    soln_gen = RiemannGenEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                            rl=rl, ul=ul, pl=pl, gl=gl,
                            rr=rr, ur=ur, pr=pr, gr=gr)
    soln_gen.driver()

    # Test that star state values are computed correctly.
    pstar  = 0.0018938734192488482
    ustar  = 1.0587708487719283e-10
    rstar1 = 0.021852118200170755
    rstar2 = 0.021852118200170755
    estar1 = 0.21666931803824513
    estar2 = 0.21666931803824513
    astar1 = 0.3483314773336128
    astar2 = 0.3483314773336128

    # Test that spatial region boundaries are computed correctly.
    # Xregs = Vregs * t + xd0
    Xregs = array([0.08775027839678179, 0.44775027841583964,
                   0.5000000000158815,  0.5522497216159235,
                   0.9122497216032182])
    Vregs = array([-2.748331477354788,     -0.3483314772277357,
                    1.0587708487719283e-10, 0.3483314774394899,
                    2.748331477354788])

    def test_riem2ig_star_states(self):
        """Using the ideal-gas EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_ig.px,  abs=1.e-12)
        assert self.ustar  == approx(self.soln_ig.ux,  abs=1.e-12)
        assert self.rstar1 == approx(self.soln_ig.rx1, abs=1.e-12)
        assert self.rstar2 == approx(self.soln_ig.rx2, abs=1.e-12)
        assert self.estar1 == approx(self.soln_ig.ex1, abs=1.e-12)
        assert self.estar2 == approx(self.soln_ig.ex2, abs=1.e-12)
        assert self.astar1 == approx(self.soln_ig.ax1, abs=1.e-12)
        assert self.astar2 == approx(self.soln_ig.ax2, abs=1.e-12)

    def test_riem2gen_star_states(self):
        """Using the general EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar        == approx(self.soln_gen.px,  abs=1.e-5)
        assert self.soln_gen.ux1 == approx(self.soln_gen.ux2, abs=1.e-5)
        assert self.ustar        == approx(self.soln_gen.ux1, abs=1.e-5)
        assert self.ustar        == approx(self.soln_gen.ux2, abs=1.e-5)
        assert self.rstar1       == approx(self.soln_gen.rx1, abs=1.e-6)
        assert self.rstar2       == approx(self.soln_gen.rx2, abs=1.e-6)
        assert self.estar1       == approx(self.soln_gen.ex1, abs=1.e-5)
        assert self.estar2       == approx(self.soln_gen.ex2, abs=1.e-5)
        assert self.astar1       == approx(self.soln_gen.ax1, abs=1.e-5)
        assert self.astar2       == approx(self.soln_gen.ax2, abs=1.e-5)

    def test_riem2ig_region_boundaries(self):
        """Using the ideal-gas EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_ig.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_ig.Xregs[1], abs=1.e-12)
        assert self.Xregs[2] == approx(self.soln_ig.Xregs[2], abs=1.e-12)
        assert self.Xregs[3] == approx(self.soln_ig.Xregs[3], abs=1.e-12)
        assert self.Vregs[0] == approx(self.soln_ig.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_ig.Vregs[1], abs=1.e-12)
        assert self.Vregs[2] == approx(self.soln_ig.Vregs[2], abs=1.e-12)
        assert self.Vregs[3] == approx(self.soln_ig.Vregs[3], abs=1.e-12)

    def test_riem2gen_region_boundaries(self):
        """Using the general EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_gen.Xregs[0], abs=1.e-5)
        assert self.Xregs[1] == approx(self.soln_gen.Xregs[1], abs=1.e-5)
        assert self.Xregs[2] == approx(self.soln_gen.Xregs[2], abs=1.e-5)
        assert self.Xregs[3] == approx(self.soln_gen.Xregs[3], abs=1.e-6)
        assert self.Vregs[0] == approx(self.soln_gen.Vregs[0], abs=1.e-5)
        assert self.Vregs[1] == approx(self.soln_gen.Vregs[1], abs=1.e-5)
        assert self.Vregs[2] == approx(self.soln_gen.Vregs[2], abs=1.e-5)
        assert self.Vregs[3] == approx(self.soln_gen.Vregs[3], abs=1.e-5)

    def test_riem2ig_state0(self):
        """Using the ideal-gas EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x=diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem2gen_state0(self):
        """Using the general EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x=diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem2ig_state1(self):
        """Using the ideal-gas EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_ig.x - x))
        r, p, u = rho_p_u_rarefaction(self.pl, self.rl, self.ul, self.gl, x,
                                      self.xd0, self.t, self.soln_ig)

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-7)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-7)

    def test_riem2gen_state1(self):
        """Using the general EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_gen.x - x))
        r, p, u = rho_p_u_rarefaction(self.pl, self.rl, self.ul, self.gl, x,
                                      self.xd0, self.t, self.soln_gen)

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-6)
        assert u == approx(ul, abs=1.e-9)
        assert p == approx(pl, abs=1.e-6)

    def test_riem2ig_state2(self):
        """Using the ideal-gas EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem2gen_state2(self):
        """Using the general EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-6)
        assert u == approx(ul, abs=1.e-9)
        assert p == approx(pl, abs=1.e-6)

    def test_riem2ig_state3(self):
        """Using the ideal-gas EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:4])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem2gen_state3(self):
        """Using the general EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:4])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-6)
        assert u == approx(ul, abs=1.e-9)
        assert p == approx(pl, abs=1.e-6)

    def test_riem2ig_state4(self):
        """Using the ideal-gas EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],Xregs[4]) returns correct values.
        x = diff(self.Xregs[3:])[0] * numpy.random.rand() + self.Xregs[3]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pr, self.rr, self.ur, self.gr, x,
                                      self.xd0, self.t, self.soln_ig)

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem2gen_state4(self):
        """Using the general EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],Xregs[4]) returns correct values.
        x = diff(self.Xregs[3:])[0] * numpy.random.rand() + self.Xregs[3]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pr, self.rr, self.ur, self.gr, x,
                                      self.xd0, self.t, self.soln_gen)

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-6)
        assert u == approx(ul, abs=1.e-9)
        assert p == approx(pl, abs=1.e-6)

    def test_riem2ig_state5(self):
        """Using the ideal-gas EOS solver, test that a random point in region 5 gives the correct state values.
        """
        # Test that any point in (Xregs[4],xmax) returns correct values.
        x = diff([self.Xregs[4], self.xmax])[0] * numpy.random.rand() + self.Xregs[4]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)

    def test_riem2gen_state5(self):
        """Using the general EOS solver, test that a random point in region 5 gives the correct state values.
        """
        # Test that any point in (Xregs[4],xmax) returns correct values.
        x = diff([self.Xregs[4], self.xmax])[0] * numpy.random.rand() + self.Xregs[4]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)


class Test_Riemann2_reversed():
    """Tests ideal-gas EOS (IGEOS) and general EOS (GenEOS) solutions produced by :class:`exactpack.solvers.riemann.riemann` on the second Riemann problem (Einfeldt) are, with left and right states reversed, are correct.
    """

    # Riemann Problem 2 reversed
    xmin, xd0, xmax, t = 0.0, 0.5, 1.0, 0.15
    rl, ul, pl, gl = 1., -2.0, 0.4, 1.4
    rr, ur, pr, gr = 1.,  2.0, 0.4, 1.4
    A, B, R1, R2, r0, e0  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    problem = 'igeos'
    num_x_pts, num_int_pts, int_tol = 10001, 10001, 1.e-12

    soln_ig = RiemannIGEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                           rl=rl, ul=ul, pl=pl, gl=gl,
                           rr=rr, ur=ur, pr=pr, gr=gr)
    soln_ig.driver()

    soln_gen = RiemannGenEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                             rl=rl, ul=ul, pl=pl, gl=gl,
                             rr=rr, ur=ur, pr=pr, gr=gr)
    soln_gen.driver()


    # Test that star state values are computed correctly.
    pstar  = 0.0018938734192488482
    ustar  = 1.0587708487719283e-10
    rstar1 = 0.021852118200170755
    rstar2 = 0.021852118200170755
    estar1 = 0.21666931803824513
    estar2 = 0.21666931803824513
    astar1 = 0.3483314773336128
    astar2 = 0.3483314773336128

    # Test that spatial region boundaries are computed correctly.
    # Xregs = Vregs * t + xd0
    Xregs = array([0.08775027839678179, 0.44775027841583964,
                   0.5000000000158815,  0.5522497216159235,
                   0.9122497216032182])
    Vregs = array([-2.748331477354788,     -0.3483314772277357,
                    1.0587708487719283e-10, 0.3483314774394899,
                    2.748331477354788])

    def test_riem2revig_star_states(self):
        """Using the ideal-gas EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_ig.px,  abs=1.e-12)
        assert self.ustar  == approx(self.soln_ig.ux,  abs=1.e-12)
        assert self.rstar1 == approx(self.soln_ig.rx1, abs=1.e-12)
        assert self.rstar2 == approx(self.soln_ig.rx2, abs=1.e-12)
        assert self.estar1 == approx(self.soln_ig.ex1, abs=1.e-12)
        assert self.estar2 == approx(self.soln_ig.ex2, abs=1.e-12)
        assert self.astar1 == approx(self.soln_ig.ax1, abs=1.e-12)
        assert self.astar2 == approx(self.soln_ig.ax2, abs=1.e-12)

    def test_riem2revgen_star_states(self):
        """Using the general EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar        == approx(self.soln_gen.px,  abs=1.e-4)
        assert self.soln_gen.ux1 == approx(self.soln_gen.ux2, abs=1.e-9)
        assert self.ustar        == approx(self.soln_gen.ux1, abs=1.e-9)
        assert self.ustar        == approx(self.soln_gen.ux2, abs=1.e-9)
        assert self.rstar1       == approx(self.soln_gen.rx1, abs=1.e-6)
        assert self.rstar2       == approx(self.soln_gen.rx2, abs=1.e-6)
        assert self.estar1       == approx(self.soln_gen.ex1, abs=1.e-5)
        assert self.estar2       == approx(self.soln_gen.ex2, abs=1.e-5)
        assert self.astar1       == approx(self.soln_gen.ax1, abs=1.e-5)
        assert self.astar2       == approx(self.soln_gen.ax2, abs=1.e-5)

    def test_riem2revig_region_boundaries(self):
        """Using the ideal-gas EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_ig.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_ig.Xregs[1], abs=1.e-12)
        assert self.Xregs[2] == approx(self.soln_ig.Xregs[2], abs=1.e-12)
        assert self.Xregs[3] == approx(self.soln_ig.Xregs[3], abs=1.e-12)
        assert self.Vregs[0] == approx(self.soln_ig.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_ig.Vregs[1], abs=1.e-12)
        assert self.Vregs[2] == approx(self.soln_ig.Vregs[2], abs=1.e-12)
        assert self.Vregs[3] == approx(self.soln_ig.Vregs[3], abs=1.e-12)

    def test_riem2revgen_region_boundaries(self):
        """Using the general EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_gen.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_gen.Xregs[1], abs=1.e-5)
        assert self.Xregs[2] == approx(self.soln_gen.Xregs[2], abs=1.e-10)
        assert self.Xregs[3] == approx(self.soln_gen.Xregs[3], abs=1.e-5)
        assert self.Vregs[0] == approx(self.soln_gen.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_gen.Vregs[1], abs=5.e-5)
        assert self.Vregs[2] == approx(self.soln_gen.Vregs[2], abs=1.e-9)
        assert self.Vregs[3] == approx(self.soln_gen.Vregs[3], abs=1.e-4)

    def test_riem2revig_state0(self):
        """Using the ideal-gas EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x=diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem2revgen_state0(self):
        """Using the general EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x=diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem2revig_state1(self):
        """Using the ideal-gas EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pl, self.rl, self.ul, self.gl, x,
                                      self.xd0, self.t, self.soln_ig)

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem2revgen_state1(self):
        """Using the general EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pl, self.rl, self.ul, self.gl, x,
                                      self.xd0, self.t, self.soln_gen)

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-6)
        assert u == approx(ul, abs=1.e-9)
        assert p == approx(pl, abs=1.e-6)

    def test_riem2revig_state2(self):
        """Using the ideal-gas EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem2revgen_state2(self):
        """Using the general EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-6)
        assert u == approx(ul, abs=1.e-9)
        assert p == approx(pl, abs=1.e-6)

    def test_riem2revig_state3(self):
        """Using the ideal-gas EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:4])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem2revgen_state3(self):
        """Using the general EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:4])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-6)
        assert u == approx(ul, abs=1.e-10)
        assert p == approx(pl, abs=1.e-6)

    def test_riem2revig_state4(self):
        """Using the ideal-gas EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],Xregs[4]) returns correct values.
        x = diff(self.Xregs[3:])[0] * numpy.random.rand() + self.Xregs[3]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pr, self.rr, self.ur, self.gr, x,
                                      self.xd0, self.t, self.soln_ig)

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem2revgen_state4(self):
        """Using the general EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],Xregs[4]) returns correct values.
        x = diff(self.Xregs[3:])[0] * numpy.random.rand() + self.Xregs[3]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pr, self.rr, self.ur, self.gr, x,
                                      self.xd0, self.t, self.soln_gen)

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-6)
        assert u == approx(ul, abs=1.e-9)
        assert p == approx(pl, abs=1.e-6)

    def test_riem2revig_state5(self):
        """Using the ideal-gas EOS solver, test that a random point in region 5 gives the correct state values.
        """
        # Test that any point in (Xregs[4],xmax) returns correct values.
        x = diff([self.Xregs[4], self.xmax])[0] * numpy.random.rand() + self.Xregs[4]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)

    def test_riem2revgen_state5(self):
        """Using the general EOS solver, test that a random point in region 5 gives the correct state values.
        """
        # Test that any point in (Xregs[4],xmax) returns correct values.
        x = diff([self.Xregs[4], self.xmax])[0] * numpy.random.rand() + self.Xregs[4]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)


class Test_Riemann3():
    """Tests ideal-gas EOS (IGEOS) and general EOS (GenEOS) solutions produced by :class:`exactpack.solvers.riemann.riemann` on the third Riemann problem (stationary contact).
    """

    # Riemann Problem 3
    xmin, xd0, xmax, t = 0.0, 0.8, 1.0, 0.012
    rl, ul, pl, gl = 1., -19.59745, 1000.,   1.4
    rr, ur, pr, gr = 1., -19.59745,    0.01, 1.4
    A, B, R1, R2, r0, e0  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    problem = 'igeos'
    num_x_pts, num_int_pts, int_tol = 10001, 10001, 1.e-12

    soln_ig = RiemannIGEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                           rl=rl, ul=ul, pl=pl, gl=gl,
                           rr=rr, ur=ur, pr=pr, gr=gr)
    soln_ig.driver()

    soln_gen = RiemannGenEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                             rl=rl, ul=ul, pl=pl, gl=gl,
                             rr=rr, ur=ur, pr=pr, gr=gr)
    soln_gen.driver()

    # Test that star state values are computed correctly.
    pstar  = 460.8937874913832
    ustar  = 1.388723067208275e-06
    rstar1 = 0.5750622984765552
    rstar2 = 5.999240704796234
    estar1 = 2003.6689447055342
    estar2 = 192.06338358907274
    astar1 = 33.4970835899948
    astar2 = 10.370896528742378

    # Test that spatial region boundaries are computed correctly.
    # Xregs = Vregs * t + xd0
    Xregs = array([0.11583171358712707, 0.3980350135847392,
                   0.8000000166646769,  0.8470410436028388])
    Vregs = array([-57.01402386773941, -33.49708220127174,
                   1.388723067208275e-06, 3.920086966903227])

    def test_riem3ig_star_states(self):
        """Using the ideal-gas EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_ig.px,  abs=1.e-12)
        assert self.ustar  == approx(self.soln_ig.ux,  abs=1.e-12)
        assert self.rstar1 == approx(self.soln_ig.rx1, abs=1.e-12)
        assert self.rstar2 == approx(self.soln_ig.rx2, abs=1.e-12)
        assert self.estar1 == approx(self.soln_ig.ex1, abs=1.e-12)
        assert self.estar2 == approx(self.soln_ig.ex2, abs=1.e-12)
        assert self.astar1 == approx(self.soln_ig.ax1, abs=1.e-12)
        assert self.astar2 == approx(self.soln_ig.ax2, abs=1.e-12)

    def test_riem3gen_star_states(self):
        """Using the general EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar        == approx(self.soln_gen.px,  abs=1.e-5)
        assert self.soln_gen.ux1 == approx(self.soln_gen.ux2, abs=1.e-12)
        assert self.ustar        == approx(self.soln_gen.ux1, abs=1.e-6)
        assert self.ustar        == approx(self.soln_gen.ux2, abs=1.e-6)
        assert self.rstar1       == approx(self.soln_gen.rx1, abs=1.e-8)
        assert self.rstar2       == approx(self.soln_gen.rx2, abs=1.e-9)
        assert self.estar1       == approx(self.soln_gen.ex1, abs=1.e-4)
        assert self.estar2       == approx(self.soln_gen.ex2, abs=1.e-5)
        assert self.astar1       == approx(self.soln_gen.ax1, abs=1.e-6)
        assert self.astar2       == approx(self.soln_gen.ax2, abs=1.e-6)

    def test_riem3ig_region_boundaries(self):
        """Using the ideal-gas EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_ig.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_ig.Xregs[1], abs=1.e-12)
        assert self.Xregs[2] == approx(self.soln_ig.Xregs[2], abs=1.e-12)
        assert self.Xregs[3] == approx(self.soln_ig.Xregs[3], abs=1.e-12)
        assert self.Vregs[0] == approx(self.soln_ig.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_ig.Vregs[1], abs=1.e-12)
        assert self.Vregs[2] == approx(self.soln_ig.Vregs[2], abs=1.e-12)
        assert self.Vregs[3] == approx(self.soln_ig.Vregs[3], abs=1.e-12)

    def test_riem3gen_region_boundaries(self):
        """Using the general EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_gen.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_gen.Xregs[1], abs=1.e-8)
        assert self.Xregs[2] == approx(self.soln_gen.Xregs[2], abs=1.e-8)
        assert self.Xregs[3] == approx(self.soln_gen.Xregs[3], abs=1.e-8)
        assert self.Vregs[0] == approx(self.soln_gen.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_gen.Vregs[1], abs=1.e-6)
        assert self.Vregs[2] == approx(self.soln_gen.Vregs[2], abs=1.e-6)
        assert self.Vregs[3] == approx(self.soln_gen.Vregs[3], abs=1.e-6)

    def test_riem3ig_state0(self):
        """Using the ideal-gas EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x=diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem3gen_state0(self):
        """Using the general EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x = diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem3ig_state1(self):
        """Using the ideal-gas EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pl, self.rl, self.ul, self.gl, x,
                                      self.xd0, self.t, self.soln_ig)

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem3gen_state1(self):
        """Using the general EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pl, self.rl, self.ul, self.gl, x,
                                      self.xd0, self.t, self.soln_gen)

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-8)
        assert u == approx(ul, abs=1.e-9)
        assert p == approx(pl, abs=1.e-5)

    def test_riem3ig_state2(self):
        """Using the ideal-gas EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem3gen_state2(self):
        """Using the general EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-8)
        assert u == approx(ul, abs=1.e-6)
        assert p == approx(pl, abs=1.e-5)

    def test_riem3ig_state3(self):
        """Using the ideal-gas EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem3gen_state3(self):
        """Using the general EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-9)
        assert u == approx(ul, abs=1.e-6)
        assert p == approx(pl, abs=1.e-5)

    def test_riem3ig_state4(self):
        """Using the ideal-gas EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],xmax) returns correct values.
        x = diff([self.Xregs[3], self.xmax])[0] * numpy.random.rand() + self.Xregs[3]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)

    def test_riem3gen_state4(self):
        """Using the general EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],xmax) returns correct values.
        x = diff([self.Xregs[3], self.xmax])[0] * numpy.random.rand() + self.Xregs[3]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)


class Test_Riemann3_reversed():
    """Tests ideal-gas EOS (IGEOS) and general EOS (GenEOS) solutions produced by :class:`exactpack.solvers.riemann.riemann` on the third Riemann problem (stationary contact), with left and right states reversed.
    """

    # Riemann Problem 3 reversed
    xmin, xd0, xmax, t = 0.0, 0.2, 1.0, 0.012
    rl, ul, pl, gl = 1., 19.59745,    0.01, 1.4
    rr, ur, pr, gr = 1., 19.59745, 1000.,   1.4
    A, B, R1, R2, r0, e0  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    problem = 'igeos'
    num_x_pts, num_int_pts, int_tol = 10001, 10001, 1.e-12

    soln_ig = RiemannIGEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                           rl=rl, ul=ul, pl=pl, gl=gl,
                           rr=rr, ur=ur, pr=pr, gr=gr)
    soln_ig.driver()

    soln_gen = RiemannGenEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                           rl=rl, ul=ul, pl=pl, gl=gl,
                           rr=rr, ur=ur, pr=pr, gr=gr)
    soln_gen.driver()


    # Test that star state values are computed correctly.
    pstar  =  460.8937874913832
    ustar  = -1.388723045891993e-06
    rstar1 =  5.999240704796234
    rstar2 =  0.5750622984765552
    estar1 =  192.06338358907274
    estar2 =  2003.6689447055342
    astar1 =  10.370896528742378
    astar2 =  33.4970835899948

    # Test that spatial region boundaries are computed correctly.
    # Xregs = Vregs * t + xd0
    Xregs = array([0.15295895639716128, 0.19999998333532346,
                   0.601964986415261, 0.884168286412873])
    Vregs = array([-3.920086966903227, -1.388723045891993e-06,
                   33.49708220127175, 57.01402386773941])

    def test_riem3revig_star_states(self):
        """Using the ideal-gas EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_ig.px,  abs=1.e-12)
        assert self.ustar  == approx(self.soln_ig.ux,  abs=1.e-12)
        assert self.rstar1 == approx(self.soln_ig.rx1, abs=1.e-12)
        assert self.rstar2 == approx(self.soln_ig.rx2, abs=1.e-12)
        assert self.estar1 == approx(self.soln_ig.ex1, abs=1.e-12)
        assert self.estar2 == approx(self.soln_ig.ex2, abs=1.e-12)
        assert self.astar1 == approx(self.soln_ig.ax1, abs=1.e-12)
        assert self.astar2 == approx(self.soln_ig.ax2, abs=1.e-12)

    def test_riem3revgen_star_states(self):
        """Using the general EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar        == approx(self.soln_gen.px,  abs=1.e-5)
        assert self.soln_gen.ux1 == approx(self.soln_gen.ux2, abs=1.e-12)
        assert self.ustar        == approx(self.soln_gen.ux1, abs=1.e-6)
        assert self.ustar        == approx(self.soln_gen.ux2, abs=1.e-6)
        assert self.rstar1       == approx(self.soln_gen.rx1, abs=1.e-9)
        assert self.rstar2       == approx(self.soln_gen.rx2, abs=1.e-8)
        assert self.estar1       == approx(self.soln_gen.ex1, abs=1.e-5)
        assert self.estar2       == approx(self.soln_gen.ex2, abs=1.e-4)
        assert self.astar1       == approx(self.soln_gen.ax1, abs=1.e-6)
        assert self.astar2       == approx(self.soln_gen.ax2, abs=1.e-6)

    def test_riem3revig_region_boundaries(self):
        """Using the ideal-gas EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_ig.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_ig.Xregs[1], abs=1.e-12)
        assert self.Xregs[2] == approx(self.soln_ig.Xregs[2], abs=1.e-12)
        assert self.Xregs[3] == approx(self.soln_ig.Xregs[3], abs=1.e-12)
        assert self.Vregs[0] == approx(self.soln_ig.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_ig.Vregs[1], abs=1.e-12)
        assert self.Vregs[2] == approx(self.soln_ig.Vregs[2], abs=1.e-12)
        assert self.Vregs[3] == approx(self.soln_ig.Vregs[3], abs=1.e-12)

    def test_riem3revgen_region_boundaries(self):
        """Using the general EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_gen.Xregs[0], abs=1.e-8)
        assert self.Xregs[1] == approx(self.soln_gen.Xregs[1], abs=1.e-8)
        assert self.Xregs[2] == approx(self.soln_gen.Xregs[2], abs=1.e-8)
        assert self.Xregs[3] == approx(self.soln_gen.Xregs[3], abs=1.e-12)
        assert self.Vregs[0] == approx(self.soln_gen.Vregs[0], abs=1.e-6)
        assert self.Vregs[1] == approx(self.soln_gen.Vregs[1], abs=1.e-6)
        assert self.Vregs[2] == approx(self.soln_gen.Vregs[2], abs=1.e-6)
        assert self.Vregs[3] == approx(self.soln_gen.Vregs[3], abs=1.e-12)

    def test_riem3revig_state0(self):
        """Using the ideal-gas EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x=diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem3revgen_state0(self):
        """Using the general EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x=diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem3revig_state1(self):
        """Using the ideal-gas EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem3revgen_state1(self):
        """Using the general EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-9)
        assert u == approx(ul, abs=1.e-6)
        assert p == approx(pl, abs=1.e-5)

    def test_riem3revig_state2(self):
        """Using the ideal-gas EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem3revig_state2(self):
        """Using the general EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem3revig_state3(self):
        """Using the ideal-gas EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pr, self.rr, self.ur, self.gr, x,
                                      self.xd0, self.t, self.soln_ig)

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem3revgen_state3(self):
        """Using the general EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pr, self.rr, self.ur, self.gr, x,
                                      self.xd0, self.t, self.soln_gen)

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-8)
        assert u == approx(ul, abs=1.e-9)
        assert p == approx(pl, abs=1.e-5)

    def test_riem3revig_state4(self):
        """Using the ideal-gas EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],xmax) returns correct values.
        x=diff([self.Xregs[3], self.xmax])[0] *numpy.random.rand()+self.Xregs[3]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)

    def test_riem3revgen_state4(self):
        """Using the general EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],xmax) returns correct values.
        x=diff([self.Xregs[3], self.xmax])[0] *numpy.random.rand()+self.Xregs[3]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)


class Test_Riemann4():
    """Tests ideal-gas EOS (IGEOS) and general EOS (GenEOS) solutions produced by :class:`exactpack.solvers.riemann.riemann` on the fourth Riemann problem (slow shock).
    """

    # Riemann Problem 4
    xmin, xd0, xmax, t = 0.0, 0.5, 1.0, 1.0
    rl, ul, pl, gl = 3.857143, -0.810631, 31./3., 1.4
    rr, ur, pr, gr = 1.      , -3.44,      1.,    1.4
    A, B, R1, R2, r0, e0  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    problem = 'igeos'
    num_x_pts, num_int_pts, int_tol = 10001, 10001, 1.e-12

    soln_ig = RiemannIGEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                           rl=rl, ul=ul, pl=pl, gl=gl,
                           rr=rr, ur=ur, pr=pr, gr=gr)
    soln_ig.driver()

    soln_gen = RiemannGenEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                             rl=rl, ul=ul, pl=pl, gl=gl,
                             rr=rr, ur=ur, pr=pr, gr=gr)
    soln_gen.driver()


    # Test that star state values are computed correctly.
    pstar  = 10.333334047951963
    ustar  = -0.8106310956659113
    ustar1 = -0.8118752612474093
    ustar2 = -0.8106311343790868
    rstar1 = 3.8571431905336095
    rstar2 = 3.8571429508974844
    estar1 = 6.697530748477618
    estar2 = 6.697531164581023
    astar1 = 1.9366510318452999
    astar2 = 1.936651092005313

    # Test that spatial region boundaries are computed correctly.
    # Xregs = Vregs * t + xd0
    Xregs = array([-2.2472820701116656, -0.3106310956659113,
                   0.6096479906523622])
    Vregs = array([-2.7472820701116656, -0.8106310956659113,
                   0.10964799065236219])

    def test_riem4ig_star_states(self):
        """Using the ideal-gas EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_ig.px,  abs=1.e-12)
        assert self.ustar  == approx(self.soln_ig.ux,  abs=1.e-12)
        assert self.rstar1 == approx(self.soln_ig.rx1, abs=1.e-12)
        assert self.rstar2 == approx(self.soln_ig.rx2, abs=1.e-12)
        assert self.estar1 == approx(self.soln_ig.ex1, abs=1.e-12)
        assert self.estar2 == approx(self.soln_ig.ex2, abs=1.e-12)
        assert self.astar1 == approx(self.soln_ig.ax1, abs=1.e-12)
        assert self.astar2 == approx(self.soln_ig.ax2, abs=1.e-12)

    def test_riem4gen_star_states(self):
        """Using the general EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_gen.px,  abs=1.e-6)
        assert self.ustar1 == approx(self.soln_gen.ux1, abs=1.e-2)
        assert self.ustar2 == approx(self.soln_gen.ux2, abs=1.e-10)
        assert self.rstar1 == approx(self.soln_gen.rx1, abs=1.e-6)
        assert self.rstar2 == approx(self.soln_gen.rx2, abs=1.e-6)
        assert self.estar1 == approx(self.soln_gen.ex1, abs=1.e-6)
        assert self.estar2 == approx(self.soln_gen.ex2, abs=1.e-6)
        assert self.astar1 == approx(self.soln_gen.ax1, abs=1.e-7)
        assert self.astar2 == approx(self.soln_gen.ax2, abs=1.e-6)

    def test_riem4ig_region_boundaries(self):
        """Using the ideal-gas EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        assert self.Xregs[0] == approx(self.soln_ig.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_ig.Xregs[1], abs=1.e-12)
        assert self.Xregs[2] == approx(self.soln_ig.Xregs[2], abs=1.e-12)
        assert self.Vregs[0] == approx(self.soln_ig.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_ig.Vregs[1], abs=1.e-12)
        assert self.Vregs[2] == approx(self.soln_ig.Vregs[2], abs=1.e-12)

    def test_riem4gen_region_boundaries(self):
        """Using the general EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        assert self.Xregs[0] == approx(self.soln_gen.Xregs[0], abs=1.e-3)
        assert self.Xregs[1] == approx(self.soln_gen.Xregs[1], abs=1.e-7)
        assert self.Xregs[2] == approx(self.soln_gen.Xregs[2], abs=1.e-6)
        assert self.Vregs[0] == approx(self.soln_gen.Vregs[0], abs=1.e-3)
        assert self.Vregs[1] == approx(self.soln_gen.Vregs[1], abs=1.e-7)
        assert self.Vregs[2] == approx(self.soln_gen.Vregs[2], abs=1.e-6)

    def test_riem4ig_state0(self):
        """Using the ideal-gas EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x = diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() +self.xmin

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rl == approx(rl, abs=1.e-6)
        assert self.ul == approx(ul, abs=1.e-6)
        assert self.pl == approx(pl, abs=1.e-5)

    def test_riem4gen_state0(self):
        """Using the general EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x = diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() +self.xmin

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rl == approx(rl, abs=1.e-6)
        assert self.ul == approx(ul, abs=1.e-6)
        assert self.pl == approx(pl, abs=1.e-5)

    def test_riem4ig_state1(self):
        """Using the ideal-gas EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem4gen_state1(self):
        """Using the general EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-6)
        assert u == approx(ul, abs=1.e-7)
        assert p == approx(pl, abs=1.e-6)

    def test_riem4ig_state2(self):
        """Using the ideal-gas EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem4gen_state2(self):
        """Using the general EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-6)
        assert u == approx(ul, abs=1.e-7)
        assert p == approx(pl, abs=1.e-6)

    def test_riem4ig_state3(self):
        """Using the ideal-gas EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],xmax) returns correct values.
        x = diff([self.Xregs[2], self.xmax])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)

    def test_riem4gen_state3(self):
        """Using the general EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],xmax) returns correct values.
        x = diff([self.Xregs[2], self.xmax])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)


class Test_Riemann4_reversed():
    """Tests ideal-gas EOS (IGEOS) and general EOS (GenEOS) solutions produced by :class:`exactpack.solvers.riemann.riemann` on the fourth Riemann problem (slow shock), with left and right states reversed.
    """

    # Riemann Problem 4
    xmin, xd0, xmax, t = 0.0, 0.5, 1.0, 1.0
    rl, ul, pl, gl = 1.      , 3.44,      1.,    1.4
    rr, ur, pr, gr = 3.857143, 0.810631, 31./3., 1.4
    A, B, R1, R2, r0, e0  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    problem = 'igeos'
    num_x_pts, num_int_pts, int_tol = 10001, 10001, 1.e-12

    soln_ig = RiemannIGEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                           rl=rl, ul=ul, pl=pl, gl=gl,
                           rr=rr, ur=ur, pr=pr, gr=gr)
    soln_ig.driver()

    soln_gen = RiemannGenEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                            rl=rl, ul=ul, pl=pl, gl=gl,
                            rr=rr, ur=ur, pr=pr, gr=gr)
    soln_gen.driver()

    # Test that star state values are computed correctly.
    pstar  = 10.333334047951963
    ustar  = 0.8106310956662885
    rstar1 = 3.8571429508974844
    rstar2 = 3.8571431905336095
    estar1 = 6.697531164581023
    estar2 = 6.697530748477618
    astar1 = 1.936651092005313
    astar2 = 1.9366510318452999

    # Test that spatial region boundaries are computed correctly.
    # Xregs = Vregs * t + xd0
    Xregs = array([0.3903520093476378, 1.3106310956662885,
                   3.2472820701116656])
    Vregs = array([-0.10964799065236219, 0.8106310956662885,
                    2.7472820701116656,])

    def test_riem4revig_star_states(self):
        """Using the ideal-gas EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_ig.px,  abs=1.e-12)
        assert self.ustar  == approx(self.soln_ig.ux,  abs=1.e-12)
        assert self.rstar1 == approx(self.soln_ig.rx1, abs=1.e-12)
        assert self.rstar2 == approx(self.soln_ig.rx2, abs=1.e-12)
        assert self.estar1 == approx(self.soln_ig.ex1, abs=1.e-12)
        assert self.estar2 == approx(self.soln_ig.ex2, abs=1.e-12)
        assert self.astar1 == approx(self.soln_ig.ax1, abs=1.e-12)
        assert self.astar2 == approx(self.soln_ig.ax2, abs=1.e-12)

    def test_riem4revgen_star_states(self):
        """Using the general EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_gen.px,  abs=1.e-6)
        assert self.ustar  == approx(self.soln_gen.ux1, abs=1.e-7)
        assert self.ustar  == approx(self.soln_gen.ux2, abs=1.e-7)
        assert self.rstar1 == approx(self.soln_gen.rx1, abs=1.e-6)
        assert self.rstar2 == approx(self.soln_gen.rx2, abs=1.e-6)
        assert self.estar1 == approx(self.soln_gen.ex1, abs=1.e-6)
        assert self.estar2 == approx(self.soln_gen.ex2, abs=1.e-6)
        assert self.astar1 == approx(self.soln_gen.ax1, abs=1.e-6)
        assert self.astar2 == approx(self.soln_gen.ax2, abs=1.e-7)

    def test_riem4revig_region_boundaries(self):
        """Using the ideal-gas EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_ig.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_ig.Xregs[1], abs=1.e-12)
        assert self.Xregs[2] == approx(self.soln_ig.Xregs[2], abs=1.e-12)
        assert self.Vregs[0] == approx(self.soln_ig.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_ig.Vregs[1], abs=1.e-12)
        assert self.Vregs[2] == approx(self.soln_ig.Vregs[2], abs=1.e-12)

    def test_riem4revgen_region_boundaries(self):
        """Using the general EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_gen.Xregs[0], abs=1.e-6)
        assert self.Xregs[1] == approx(self.soln_gen.Xregs[1], abs=1.e-7)
        assert self.Xregs[2] == approx(self.soln_gen.Xregs[2], abs=1.e-3)
        assert self.Vregs[0] == approx(self.soln_gen.Vregs[0], abs=1.e-6)
        assert self.Vregs[1] == approx(self.soln_gen.Vregs[1], abs=1.e-7)
        assert self.Vregs[2] == approx(self.soln_gen.Vregs[2], abs=1.e-3)

    def test_riem4revig_state0(self):
        """Using the ideal-gas EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x=diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem4revgen_state0(self):
        """Using the general EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x=diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem4revig_state1(self):
        """Using the ideal-gas EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem4revgen_state1(self):
        """Using the general EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-6)
        assert u == approx(ul, abs=1.e-7)
        assert p == approx(pl, abs=1.e-6)

    def test_riem4revig_state2(self):
        """Using the ideal-gas EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem4revgen_state2(self):
        """Using the general EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-6)
        assert u == approx(ul, abs=1.e-7)
        assert p == approx(pl, abs=1.e-6)

    def test_riem4revig_state3(self):
        """Using the ideal-gas EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],xmax) returns correct values.
        x = diff([self.Xregs[2], self.xmax])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rr == approx(rl, abs=1.e-6)
        assert self.ur == approx(ul, abs=1.e-6)
        assert self.pr == approx(pl, abs=1.e-5)

    def test_riem4revgen_state3(self):
        """Using the general EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],xmax) returns correct values.
        x = diff([self.Xregs[2], self.xmax])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rr == approx(rl, abs=1.e-6)
        assert self.ur == approx(ul, abs=1.e-6)
        assert self.pr == approx(pl, abs=1.e-5)


class Test_Riemann5():
    """Tests ideal-gas EOS (IGEOS) and general EOS (GenEOS) solutions produced by :class:`exactpack.solvers.riemann.riemann` on the fifth Riemann problem (shock contact shock).
    """

    # Riemann Problem 5
    xmin, xd0, xmax, t = 0.0, 0.5, 1.0, 0.3
    rl, ul, pl, gl = 1,     0.5, 1., 1.4
    rr, ur, pr, gr = 1.25, -0.5, 1., 1.4
    A, B, R1, R2, r0, e0  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    problem = 'igeos'
    num_x_pts, num_int_pts, int_tol = 10001, 10001, 1.e-12

    soln_ig = RiemannIGEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                           rl=rl, ul=ul, pl=pl, gl=gl,
                           rr=rr, ur=ur, pr=pr, gr=gr)
    soln_ig.driver()

    soln_gen = RiemannGenEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                             rl=rl, ul=ul, pl=pl, gl=gl,
                             rr=rr, ur=ur, pr=pr, gr=gr)
    soln_gen.driver()


    # Test that star state values are computed correctly.
    pstar  = 1.8137499744295837
    ustar  = -0.02786404500006001
    rstar1 = 1.5207166706719388
    rstar2 = 1.9008958383399235
    estar1 = 2.9817355353054795
    estar2 = 2.385388428244384
    astar1 = 1.2921965406899478
    astar2 = 1.1557757221091187

    # Test that spatial region boundaries are computed correctly.
    # Xregs = Vregs * t + xd0
    Xregs = array([0.18752297652921718, 0.491640786499982,
                   0.7636520250049744])
    Vregs = array([-1.0415900782359429, -0.02786404500006001,
                   0.8788400833499148])

    def test_riem5ig_star_states(self):
        """Using the ideal-gas EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_ig.px,  abs=1.e-12)
        assert self.ustar  == approx(self.soln_ig.ux,  abs=1.e-12)
        assert self.rstar1 == approx(self.soln_ig.rx1, abs=1.e-12)
        assert self.rstar2 == approx(self.soln_ig.rx2, abs=1.e-12)
        assert self.estar1 == approx(self.soln_ig.ex1, abs=1.e-12)
        assert self.estar2 == approx(self.soln_ig.ex2, abs=1.e-12)
        assert self.astar1 == approx(self.soln_ig.ax1, abs=1.e-12)
        assert self.astar2 == approx(self.soln_ig.ax2, abs=1.e-12)

    def test_riem5gen_star_states(self):
        """Using the general EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_gen.px,  abs=1.e-7)
        assert self.ustar  == approx(self.soln_gen.ux1,  abs=1.e-12)
        assert self.ustar  == approx(self.soln_gen.ux2,  abs=1.e-12)
        assert self.rstar1 == approx(self.soln_gen.rx1, abs=1.e-7)
        assert self.rstar2 == approx(self.soln_gen.rx2, abs=1.e-7)
        assert self.estar1 == approx(self.soln_gen.ex1, abs=1.e-7)
        assert self.estar2 == approx(self.soln_gen.ex2, abs=1.e-7)
        assert self.astar1 == approx(self.soln_gen.ax1, abs=1.e-7)
        assert self.astar2 == approx(self.soln_gen.ax2, abs=1.e-7)

    def test_riem5ig_region_boundaries(self):
        """Using the ideal-gas EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_ig.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_ig.Xregs[1], abs=1.e-12)
        assert self.Xregs[2] == approx(self.soln_ig.Xregs[2], abs=1.e-12)
        assert self.Vregs[0] == approx(self.soln_ig.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_ig.Vregs[1], abs=1.e-12)
        assert self.Vregs[2] == approx(self.soln_ig.Vregs[2], abs=1.e-12)

    def test_riem5gen_region_boundaries(self):
        """Using the general EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_gen.Xregs[0], abs=1.e-7)
        assert self.Xregs[1] == approx(self.soln_gen.Xregs[1], abs=1.e-12)
        assert self.Xregs[2] == approx(self.soln_gen.Xregs[2], abs=1.e-7)
        assert self.Vregs[0] == approx(self.soln_gen.Vregs[0], abs=1.e-7)
        assert self.Vregs[1] == approx(self.soln_gen.Vregs[1], abs=1.e-12)
        assert self.Vregs[2] == approx(self.soln_gen.Vregs[2], abs=1.e-7)

    def test_riem5ig_state0(self):
        """Using the ideal-gas EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x = diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem5gen_state0(self):
        """Using the general EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x = diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem5ig_state1(self):
        """Using the ideal-gas EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem5gen_state1(self):
        """Using the general EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-7)
        assert u == approx(ul, abs=1.e-7)
        assert p == approx(pl, abs=1.e-7)

    def test_riem5ig_state2(self):
        """Using the ideal-gas EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem5gen_state2(self):
        """Using the general EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-7)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-7)

    def test_riem5ig_state3(self):
        """Using the ideal-gas EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],xmax) returns correct values.
        x = diff([self.Xregs[2], self.xmax])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)

    def test_riem5gen_state3(self):
        """Using the general EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],xmax) returns correct values.
        x = diff([self.Xregs[2], self.xmax])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)


class Test_Riemann5_reversed():
    """Tests ideal-gas EOS (IGEOS) and general EOS (GenEOS) solutions produced by :class:`exactpack.solvers.riemann.riemann` on the fifth Riemann problem (shock contact shock), with left and right states reversed.
    """

    # Riemann Problem 5 reversed
    xmin, xd0, xmax, t = 0.0, 0.5, 1.0, 0.3
    rl, ul, pl, gl = 1.25,  0.5, 1., 1.4
    rr, ur, pr, gr = 1,    -0.5, 1., 1.4
    A, B, R1, R2, r0, e0  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    problem = 'igeos'
    num_x_pts, num_int_pts, int_tol = 10001, 10001, 1.e-12

    soln_ig = RiemannIGEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                           rl=rl, ul=ul, pl=pl, gl=gl,
                           rr=rr, ur=ur, pr=pr, gr=gr)
    soln_ig.driver()

    soln_gen = RiemannGenEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                             rl=rl, ul=ul, pl=pl, gl=gl,
                             rr=rr, ur=ur, pr=pr, gr=gr)
    soln_gen.driver()


    # Test that star state values are computed correctly.
    pstar  = 1.8137499744295837
    ustar  = 0.027864045000743076
    rstar1 = 1.9008958383399235
    rstar2 = 1.5207166706719388
    estar1 = 2.385388428244384
    estar2 = 2.9817355353054795
    astar1 = 1.1557757221091187
    astar2 = 1.2921965406899478

    # Test that spatial region boundaries are computed correctly.
    # Xregs = Vregs * t + xd0
    Xregs = array([0.23634797499502558, 0.508359213500223,
                   0.8124770234707828])
    Vregs = array([-0.8788400833499148, 0.027864045000743076,
                   1.0415900782359429])

    def test_riem5revig_star_states(self):
        """Using the ideal-gas EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_ig.px,  abs=1.e-12)
        assert self.ustar  == approx(self.soln_ig.ux,  abs=1.e-12)
        assert self.rstar1 == approx(self.soln_ig.rx1, abs=1.e-12)
        assert self.rstar2 == approx(self.soln_ig.rx2, abs=1.e-12)
        assert self.estar1 == approx(self.soln_ig.ex1, abs=1.e-12)
        assert self.estar2 == approx(self.soln_ig.ex2, abs=1.e-12)
        assert self.astar1 == approx(self.soln_ig.ax1, abs=1.e-12)
        assert self.astar2 == approx(self.soln_ig.ax2, abs=1.e-12)

    def test_riem5revgen_star_states(self):
        """Using the general EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_gen.px,  abs=1.e-7)
        assert self.ustar  == approx(self.soln_gen.ux1, abs=1.e-12)
        assert self.ustar  == approx(self.soln_gen.ux2, abs=1.e-12)
        assert self.rstar1 == approx(self.soln_gen.rx1, abs=1.e-7)
        assert self.rstar2 == approx(self.soln_gen.rx2, abs=1.e-7)
        assert self.estar1 == approx(self.soln_gen.ex1, abs=1.e-7)
        assert self.estar2 == approx(self.soln_gen.ex2, abs=1.e-7)
        assert self.astar1 == approx(self.soln_gen.ax1, abs=1.e-7)
        assert self.astar2 == approx(self.soln_gen.ax2, abs=1.e-7)

    def test_riem5revig_region_boundaries(self):
        """Using the ideal-gas EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_ig.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_ig.Xregs[1], abs=1.e-12)
        assert self.Xregs[2] == approx(self.soln_ig.Xregs[2], abs=1.e-12)
        assert self.Vregs[0] == approx(self.soln_ig.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_ig.Vregs[1], abs=1.e-12)
        assert self.Vregs[2] == approx(self.soln_ig.Vregs[2], abs=1.e-12)

    def test_riem5revgen_region_boundaries(self):
        """Using the general EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_gen.Xregs[0], abs=1.e-7)
        assert self.Xregs[1] == approx(self.soln_gen.Xregs[1], abs=1.e-12)
        assert self.Xregs[2] == approx(self.soln_gen.Xregs[2], abs=1.e-7)
        assert self.Vregs[0] == approx(self.soln_gen.Vregs[0], abs=1.e-7)
        assert self.Vregs[1] == approx(self.soln_gen.Vregs[1], abs=1.e-12)
        assert self.Vregs[2] == approx(self.soln_gen.Vregs[2], abs=1.e-7)

    def test_riem5revig_state0(self):
        """Using the ideal-gas EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x = diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem5revgen_state0(self):
        """Using the general EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x = diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem5revig_state1(self):
        """Using the ideal-gas EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem5revgen_state1(self):
        """Using the general EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-7)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-7)

    def test_riem5revig_state2(self):
        """Using the ideal-gas EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem5revgen_state2(self):
        """Using the general EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-7)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-7)

    def test_riem5revig_state3(self):
        """Using the ideal-gas EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],xmax) returns correct values.
        x = diff([self.Xregs[2], self.xmax])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)

    def test_riem5revgen_state3(self):
        """Using the general EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],xmax) returns correct values.
        x = diff([self.Xregs[2], self.xmax])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)


class Test_Riemann6():
    """Tests ideal-gas EOS (IGEOS) and general EOS (GenEOS) solutions produced by :class:`exactpack.solvers.riemann.riemann` on the sixth Riemann problem (LeBlanc).
    """

    # Riemann Problem 6
    xmin, xd0, xmax, t = 0.0, 0.3, 1.0, 0.5
    rl, ul, pl, gl = 1.,   0., 1./15., 5./3.
    rr, ur, pr, gr = 0.01, 0., 2./(3.e10), 5./3.
    A, B, R1, R2, r0, e0  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    problem = 'igeos'
    num_x_pts, num_int_pts, int_tol = 10001, 10001, 1.e-12

    soln_ig = RiemannIGEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                           rl=rl, ul=ul, pl=pl, gl=gl,
                           rr=rr, ur=ur, pr=pr, gr=gr)
    soln_ig.driver()

    soln_gen = RiemannGenEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                             rl=rl, ul=ul, pl=pl, gl=gl,
                             rr=rr, ur=ur, pr=pr, gr=gr)
    soln_gen.driver()


    # Test that star state values are computed correctly.
    pstar  = 0.0028952132330838745
    ustar  = 0.4659838851123562
    rstar1 = 0.1522870901867298
    rstar2 = 0.03999999654602329
    estar1 = 0.02851732109596931
    estar2 = 0.10857050561564521
    astar1 = 0.1780053716292146
    astar2 = 0.34732390520736506

    # Test that spatial region boundaries are computed correctly.
    # Xregs = Vregs * t + xd0
    Xregs = array([0.1333333333333333, 0.4439892567415708,
                   0.5329919425561781, 0.6106559323834297])
    Vregs = array([-0.33333333333333337, 0.28797851348314163,
                   0.4659838851123562, 0.6213118647668594])

    def test_riem6ig_star_states(self):
        """Using the ideal-gas EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_ig.px,  abs=1.e-12)
        assert self.ustar  == approx(self.soln_ig.ux,  abs=1.e-12)
        assert self.rstar1 == approx(self.soln_ig.rx1, abs=1.e-12)
        assert self.rstar2 == approx(self.soln_ig.rx2, abs=1.e-12)
        assert self.estar1 == approx(self.soln_ig.ex1, abs=1.e-12)
        assert self.estar2 == approx(self.soln_ig.ex2, abs=1.e-12)
        assert self.astar1 == approx(self.soln_ig.ax1, abs=1.e-12)
        assert self.astar2 == approx(self.soln_ig.ax2, abs=1.e-12)

    def test_riem6gen_star_states(self):
        """Using the general EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_gen.px,  abs=1.e-6)
        assert self.ustar  == approx(self.soln_gen.ux1, abs=1.e-5)
        assert self.ustar  == approx(self.soln_gen.ux2, abs=1.e-5)
        assert self.rstar1 == approx(self.soln_gen.rx1, abs=1.e-5)
        assert self.rstar2 == approx(self.soln_gen.rx2, abs=1.e-11)
        assert self.estar1 == approx(self.soln_gen.ex1, abs=1.e-6)
        assert self.estar2 == approx(self.soln_gen.ex2, abs=1.e-5)
        assert self.astar1 == approx(self.soln_gen.ax1, abs=1.e-5)
        assert self.astar2 == approx(self.soln_gen.ax2, abs=1.e-5)

    def test_riem6ig_region_boundaries(self):
        """Using the ideal-gas EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_ig.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_ig.Xregs[1], abs=1.e-12)
        assert self.Xregs[2] == approx(self.soln_ig.Xregs[2], abs=1.e-12)
        assert self.Xregs[3] == approx(self.soln_ig.Xregs[3], abs=1.e-12)
        assert self.Vregs[0] == approx(self.soln_ig.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_ig.Vregs[1], abs=1.e-12)
        assert self.Vregs[2] == approx(self.soln_ig.Vregs[2], abs=1.e-12)
        assert self.Vregs[3] == approx(self.soln_ig.Vregs[3], abs=1.e-12)

    def test_riem6gen_region_boundaries(self):
        """Using the general EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_gen.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_gen.Xregs[1], abs=1.e-5)
        assert self.Xregs[2] == approx(self.soln_gen.Xregs[2], abs=1.e-5)
        assert self.Xregs[3] == approx(self.soln_gen.Xregs[3], abs=1.e-5)
        assert self.Vregs[0] == approx(self.soln_gen.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_gen.Vregs[1], abs=1.e-5)
        assert self.Vregs[2] == approx(self.soln_gen.Vregs[2], abs=1.e-5)
        assert self.Vregs[3] == approx(self.soln_gen.Vregs[3], abs=1.e-4)

    def test_riem6ig_state0(self):
        """Using the ideal-gas EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x=diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem6gen_state0(self):
        """Using the general EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x=diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem6ig_state1(self):
        """Using the ideal-gas EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pl, self.rl, self.ul, self.gl, x,
                                      self.xd0, self.t, self.soln_ig)

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem6gen_state1(self):
        """Using the general EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pl, self.rl, self.ul, self.gl, x,
                                      self.xd0, self.t, self.soln_gen)

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-7)
        assert u == approx(ul, abs=1.e-9)
        assert p == approx(pl, abs=1.e-8)

    def test_riem6ig_state2(self):
        """Using the ideal-gas EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem6gen_state2(self):
        """Using the general EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-5)
        assert u == approx(ul, abs=1.e-5)
        assert p == approx(pl, abs=1.e-6)

    def test_riem6ig_state3(self):
        """Using the ideal-gas EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem6gen_state3(self):
        """Using the general EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-11)
        assert u == approx(ul, abs=1.e-5)
        assert p == approx(pl, abs=1.e-6)

    def test_riem6ig_state4(self):
        """Using the ideal-gas EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],xmax) returns correct values.
        x = diff([self.Xregs[3], self.xmax])[0] * numpy.random.rand() + self.Xregs[3]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)

    def test_riem6gen_state4(self):
        """Using the general EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],xmax) returns correct values.
        x = diff([self.Xregs[3], self.xmax])[0] * numpy.random.rand() + self.Xregs[3]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)


class Test_Riemann6_reversed():
    """Tests ideal-gas EOS (IGEOS) and general EOS (GenEOS) solutions produced by :class:`exactpack.solvers.riemann.riemann` on the sixth Riemann problem (LeBlanc), with the left and right states reversed.
    """

    # Riemann Problem 6 reversed
    xmin, xd0, xmax, t = 0.0, 0.7, 1.0, 0.5
    rl, ul, pl, gl = 0.01, 0., 2./(3.e10), 5./3.
    rr, ur, pr, gr = 1.,   0., 1./15., 5./3.
    A, B, R1, R2, r0, e0  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    problem = 'igeos'
    num_x_pts, num_int_pts, int_tol = 10001, 10001, 1.e-12

    soln_ig = RiemannIGEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                           rl=rl, ul=ul, pl=pl, gl=gl,
                           rr=rr, ur=ur, pr=pr, gr=gr)
    soln_ig.driver()

    soln_gen = RiemannGenEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                             rl=rl, ul=ul, pl=pl, gl=gl,
                             rr=rr, ur=ur, pr=pr, gr=gr)
    soln_gen.driver()


    # Test that star state values are computed correctly.
    pstar  = 0.0028952132330838745
    ustar  = -0.46598388516266387
    rstar1 = 0.03999999654602329
    rstar2 = 0.1522870901867298
    estar1 = 0.10857050561564521
    estar2 = 0.02851732109596931
    astar1 = 0.34732390520736506
    astar2 = 0.1780053716292146

    # Test that spatial region boundaries are computed correctly.
    # Xregs = Vregs * t + xd0
    Xregs = array([0.38934406761657026, 0.467008057418668,
                   0.5560107432332753, 0.8666666666666667])
    Vregs = array([-0.6213118647668594, -0.46598388516266387,
                   -0.2879785135334493, 0.33333333333333337])

    def test_riem6revig_star_states(self):
        """Using the ideal-gas EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_ig.px,  abs=1.e-12)
        assert self.ustar  == approx(self.soln_ig.ux,  abs=1.e-12)
        assert self.rstar1 == approx(self.soln_ig.rx1, abs=1.e-12)
        assert self.rstar2 == approx(self.soln_ig.rx2, abs=1.e-12)
        assert self.estar1 == approx(self.soln_ig.ex1, abs=1.e-12)
        assert self.estar2 == approx(self.soln_ig.ex2, abs=1.e-12)
        assert self.astar1 == approx(self.soln_ig.ax1, abs=1.e-12)
        assert self.astar2 == approx(self.soln_ig.ax2, abs=1.e-12)

    def test_riem6revgen_star_states(self):
        """Using the general EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_gen.px,  abs=1.e-6)
        assert self.ustar  == approx(self.soln_gen.ux1, abs=1.e-5)
        assert self.ustar  == approx(self.soln_gen.ux2, abs=1.e-5)
        assert self.rstar1 == approx(self.soln_gen.rx1, abs=1.e-11)
        assert self.rstar2 == approx(self.soln_gen.rx2, abs=1.e-5)
        assert self.estar1 == approx(self.soln_gen.ex1, abs=1.e-5)
        assert self.estar2 == approx(self.soln_gen.ex2, abs=1.e-6)
        assert self.astar1 == approx(self.soln_gen.ax1, abs=1.e-5)
        assert self.astar2 == approx(self.soln_gen.ax2, abs=1.e-5)

    def test_riem6revig_region_boundaries(self):
        """Using the ideal-gas EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_ig.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_ig.Xregs[1], abs=1.e-12)
        assert self.Xregs[2] == approx(self.soln_ig.Xregs[2], abs=1.e-12)
        assert self.Xregs[3] == approx(self.soln_ig.Xregs[3], abs=1.e-12)
        assert self.Vregs[0] == approx(self.soln_ig.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_ig.Vregs[1], abs=1.e-12)
        assert self.Vregs[2] == approx(self.soln_ig.Vregs[2], abs=1.e-12)
        assert self.Vregs[3] == approx(self.soln_ig.Vregs[3], abs=1.e-12)

    def test_riem6revgen_region_boundaries(self):
        """Using the general EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_gen.Xregs[0], abs=1.e-5)
        assert self.Xregs[1] == approx(self.soln_gen.Xregs[1], abs=1.e-5)
        assert self.Xregs[2] == approx(self.soln_gen.Xregs[2], abs=1.e-5)
        assert self.Xregs[3] == approx(self.soln_gen.Xregs[3], abs=1.e-12)
        assert self.Vregs[0] == approx(self.soln_gen.Vregs[0], abs=1.e-4)
        assert self.Vregs[1] == approx(self.soln_gen.Vregs[1], abs=1.e-5)
        assert self.Vregs[2] == approx(self.soln_gen.Vregs[2], abs=1.e-5)
        assert self.Vregs[3] == approx(self.soln_gen.Vregs[3], abs=1.e-12)

    def test_riem6revig_state0(self):
        """Using the ideal-gas EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x = diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem6revgen_state0(self):
        """Using the general EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x = diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem6revig_state1(self):
        """Using the ideal-gas EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem6revgen_state1(self):
        """Using the general EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-11)
        assert u == approx(ul, abs=1.e-5)
        assert p == approx(pl, abs=1.e-6)

    def test_riem6revig_state2(self):
        """Using the ideal-gas EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem6revgen_state2(self):
        """Using the general EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-5)
        assert u == approx(ul, abs=1.e-5)
        assert p == approx(pl, abs=1.e-6)

    def test_riem6revig_state3(self):
        """Using the ideal-gas EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pr, self.rr, self.ur, self.gr, x,
                                      self.xd0, self.t, self.soln_ig)

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem6revgen_state3(self):
        """Using the general EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pr, self.rr, self.ur, self.gr, x,
                                      self.xd0, self.t, self.soln_gen)

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-7)
        assert u == approx(ul, abs=1.e-9)
        assert p == approx(pl, abs=1.e-8)

    def test_riem6revig_state4(self):
        """Using the ideal-gas EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],xmax) returns correct values.
        x = diff([self.Xregs[3], self.xmax])[0] * numpy.random.rand() + self.Xregs[3]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)

    def test_riem6revgen_state4(self):
        """Using the general EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],xmax) returns correct values.
        x = diff([self.Xregs[3], self.xmax])[0] * numpy.random.rand() + self.Xregs[3]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)


class Test_Riemann7():
    """Tests ideal-gas EOS (IGEOS) and general EOS (GenEOS) solutions produced by :class:`exactpack.solvers.riemann.riemann` on the seventh Riemann problem (modified Sod).
    """

    # Riemann Problem 7
    xmin, xd0, xmax, t = 0.0, 0.5, 1.0, 0.2
    rl, ul, pl, gl = 1.0,   0.0, 2.0, 2.0
    rr, ur, pr, gr = 0.125, 0.0, 0.1, 1.4
    A, B, R1, R2, r0, e0  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    problem = 'igeos'
    num_x_pts, num_int_pts, int_tol = 10001, 10001, 1.e-12
 
    soln_ig = RiemannIGEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                           rl=rl, ul=ul, pl=pl, gl=gl,
                           rr=rr, ur=ur, pr=pr, gr=gr)
    soln_ig.driver()

    soln_gen = RiemannGenEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                             rl=rl, ul=ul, pl=pl, gl=gl,
                             rr=rr, ur=ur, pr=pr, gr=gr)
    soln_gen.driver()

    # Test that star state values are computed correctly.
    pstar  = 0.4303319371967973
    ustar  = 1.2757096812803406
    rstar1 = 0.46385985879185393
    rstar2 = 0.3253795605032907
    estar1 = 0.9277197175837077
    estar2 = 3.3063842157999144
    astar1 = 1.3621451593598295
    astar2 = 1.3607259683154251

    # Test that spatial region boundaries are computed correctly.
    # Xregs = Vregs * t + xd0
    Xregs = array([0.09999999999999998, 0.4827129043841022,
                   0.7551419362560681,  0.9143035890305202])
    Vregs = array([-2.0, -0.08643547807948893,
                    1.2757096812803406, 2.0715179451526007])

    def test_riem7ig_star_states(self):
        """Using the ideal-gas EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_ig.px,  abs=1.e-12)
        assert self.ustar  == approx(self.soln_ig.ux,  abs=1.e-12)
        assert self.rstar1 == approx(self.soln_ig.rx1, abs=1.e-12)
        assert self.rstar2 == approx(self.soln_ig.rx2, abs=1.e-12)
        assert self.estar1 == approx(self.soln_ig.ex1, abs=1.e-12)
        assert self.estar2 == approx(self.soln_ig.ex2, abs=1.e-12)
        assert self.astar1 == approx(self.soln_ig.ax1, abs=1.e-12)
        assert self.astar2 == approx(self.soln_ig.ax2, abs=1.e-12)

    def test_riem7gen_star_states(self):
        """Using the general EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar        == approx(self.soln_gen.px,  abs=1.e-6)
        assert self.soln_gen.ux1 == approx(self.soln_gen.ux2, abs=1.e-10)
        assert self.ustar        == approx(self.soln_gen.ux1, abs=1.e-6)
        assert self.ustar        == approx(self.soln_gen.ux2, abs=1.e-6)
        assert self.rstar1       == approx(self.soln_gen.rx1, abs=1.e-6)
        assert self.rstar2       == approx(self.soln_gen.rx2, abs=1.e-6)
        assert self.estar1       == approx(self.soln_gen.ex1, abs=1.e-6)
        assert self.estar2       == approx(self.soln_gen.ex2, abs=1.e-5)
        assert self.astar1       == approx(self.soln_gen.ax1, abs=1.e-6)
        assert self.astar2       == approx(self.soln_gen.ax2, abs=1.e-6)

    def test_riem7ig_region_boundaries(self):
        """Using the ideal-gas EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_ig.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_ig.Xregs[1], abs=1.e-12)
        assert self.Xregs[2] == approx(self.soln_ig.Xregs[2], abs=1.e-12)
        assert self.Xregs[3] == approx(self.soln_ig.Xregs[3], abs=1.e-12)
        assert self.Vregs[0] == approx(self.soln_ig.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_ig.Vregs[1], abs=1.e-12)
        assert self.Vregs[2] == approx(self.soln_ig.Vregs[2], abs=1.e-12)
        assert self.Vregs[3] == approx(self.soln_ig.Vregs[3], abs=1.e-12)

    def test_riem7ig_region_boundaries(self):
        """Using the general EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_gen.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_gen.Xregs[1], abs=1.e-7)
        assert self.Xregs[2] == approx(self.soln_gen.Xregs[2], abs=1.e-7)
        assert self.Xregs[3] == approx(self.soln_gen.Xregs[3], abs=1.e-7)
        assert self.Vregs[0] == approx(self.soln_gen.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_gen.Vregs[1], abs=1.e-5)
        assert self.Vregs[2] == approx(self.soln_gen.Vregs[2], abs=1.e-6)
        assert self.Vregs[3] == approx(self.soln_gen.Vregs[3], abs=1.e-6)

    def test_riem7ig_state0(self):
        """Using the ideal-gas EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x = diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() +self.xmin

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem7gen_state0(self):
        """Using the general EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x = diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() +self.xmin

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riem7ig_state1(self):
        """Using the ideal-gas EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pl, self.rl, self.ul, self.gl, x,
                                      self.xd0, self.t, self.soln_ig)

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem7gen_state1(self):
        """Using the general EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pl, self.rl, self.ul, self.gl, x,
                                      self.xd0, self.t, self.soln_gen)

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-8)
        assert u == approx(ul, abs=1.e-8)
        assert p == approx(pl, abs=1.e-7)

    def test_riem7ig_state2(self):
        """Using the ideal-gas EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem7gen_state2(self):
        """Using the general EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-6)
        assert u == approx(ul, abs=1.e-6)
        assert p == approx(pl, abs=1.e-6)

    def test_riem7ig_state3(self):
        """Using the ideal-gas EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem7gen_state3(self):
        """Using the general EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-6)
        assert u == approx(ul, abs=1.e-6)
        assert p == approx(pl, abs=1.e-6)

    def test_riem7ig_state4(self):
        """Using the ideal-gas EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],Xregs[4]) returns correct values.
        x = diff([self.Xregs[3], self.xmax])[0] * numpy.random.rand() + self.Xregs[3]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)

    def test_riem7gen_state4(self):
        """Using the general EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],Xregs[4]) returns correct values.
        x = diff([self.Xregs[3], self.xmax])[0] * numpy.random.rand() + self.Xregs[3]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)


class Test_Riemann7_reversed():
    """Tests ideal-gas EOS (IGEOS) and general EOS (GenEOS) solutions produced by :class:`exactpack.solvers.riemann.riemann` on the seventh Riemann problem (modified Sod), modified to have different left and right values for gamma, with left and right states reversed, is correct.
    """

    # Riemann Problem 1 modified, reversed
    xmin, xd0, xmax, t = 0.0, 0.5, 1.0, 0.2
    rl, ul, pl, gl = 0.125, 0.0, 0.1, 1.4
    rr, ur, pr, gr = 1.0,   0.0, 2.0, 2.0
    A, B, R1, R2, r0, e0  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    problem = 'igeos'
    num_x_pts, num_int_pts, int_tol = 10001, 10001, 1.e-12

    soln_ig = RiemannIGEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                           rl=rl, ul=ul, pl=pl, gl=gl,
                           rr=rr, ur=ur, pr=pr, gr=gr)
    soln_ig.driver()

    soln_gen = RiemannGenEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                             rl=rl, ul=ul, pl=pl, gl=gl,
                             rr=rr, ur=ur, pr=pr, gr=gr)
    soln_gen.driver()

    # Test that star state values are computed correctly.
    pstar  =  0.4303319371967973
    ustar  = -1.2757096812790123
    rstar1 =  0.3253795605032907
    rstar2 =  0.46385985879185393
    estar1 =  3.3063842157999144
    estar2 =  0.9277197175837077
    astar1 =  1.3607259683154251
    astar2 =  1.3621451593598295

    # Test that spatial region boundaries are computed correctly.
    # Xregs = Vregs * t + xd0
    Xregs = array([0.08569641096947983, 0.24485806374419755,
                   0.5172870956161635,  0.9])
    Vregs = array([-2.0715179451526007, -1.2757096812790123,
                    0.0864354780808172,  2.0])

    def test_riem7revig_star_states(self):
        """Using the ideal-gas EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_ig.px,  abs=1.e-12)
        assert self.ustar  == approx(self.soln_ig.ux,  abs=1.e-12)
        assert self.rstar1 == approx(self.soln_ig.rx1, abs=1.e-12)
        assert self.rstar2 == approx(self.soln_ig.rx2, abs=1.e-12)
        assert self.estar1 == approx(self.soln_ig.ex1, abs=1.e-12)
        assert self.estar2 == approx(self.soln_ig.ex2, abs=1.e-12)
        assert self.astar1 == approx(self.soln_ig.ax1, abs=1.e-12)
        assert self.astar2 == approx(self.soln_ig.ax2, abs=1.e-12)

    def test_riem7revgen_star_states(self):
        """Using the general EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar        == approx(self.soln_gen.px,  abs=1.e-6)
        assert self.soln_gen.ux1 == approx(self.soln_gen.ux2, abs=1.e-11)
        assert self.ustar        == approx(self.soln_gen.ux1, abs=1.e-6)
        assert self.ustar        == approx(self.soln_gen.ux2, abs=1.e-6)
        assert self.rstar1       == approx(self.soln_gen.rx1, abs=1.e-7)
        assert self.rstar2       == approx(self.soln_gen.rx2, abs=1.e-7)
        assert self.estar1       == approx(self.soln_gen.ex1, abs=1.e-5)
        assert self.estar2       == approx(self.soln_gen.ex2, abs=1.e-6)
        assert self.astar1       == approx(self.soln_gen.ax1, abs=1.e-6)
        assert self.astar2       == approx(self.soln_gen.ax2, abs=1.e-7)

    def test_riem7revig_region_boundaries(self):
        """Using the ideal-gas EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_ig.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_ig.Xregs[1], abs=1.e-12)
        assert self.Xregs[2] == approx(self.soln_ig.Xregs[2], abs=1.e-12)
        assert self.Xregs[3] == approx(self.soln_ig.Xregs[3], abs=1.e-12)
        assert self.Vregs[0] == approx(self.soln_ig.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_ig.Vregs[1], abs=1.e-12)
        assert self.Vregs[2] == approx(self.soln_ig.Vregs[2], abs=1.e-12)
        assert self.Vregs[3] == approx(self.soln_ig.Vregs[3], abs=1.e-12)

    def test_riem7revgen_region_boundaries(self):
        """Using the general EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_gen.Xregs[0], abs=1.e-6)
        assert self.Xregs[1] == approx(self.soln_gen.Xregs[1], abs=1.e-7)
        assert self.Xregs[2] == approx(self.soln_gen.Xregs[2], abs=1.e-7)
        assert self.Xregs[3] == approx(self.soln_gen.Xregs[3], abs=1.e-12)
        assert self.Vregs[0] == approx(self.soln_gen.Vregs[0], abs=1.e-6)
        assert self.Vregs[1] == approx(self.soln_gen.Vregs[1], abs=1.e-6)
        assert self.Vregs[2] == approx(self.soln_gen.Vregs[2], abs=1.e-5)
        assert self.Vregs[3] == approx(self.soln_gen.Vregs[3], abs=1.e-12)

    def test_riem7revig_state0(self):
        """Using the ideal-gas EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x=diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert rl == approx(self.soln_ig.rl, abs=1.e-12)
        assert ul == approx(self.soln_ig.ul, abs=1.e-12)
        assert pl == approx(self.soln_ig.pl, abs=1.e-12)

    def test_riem7revgen_state0(self):
        """Using the general EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x=diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() + self.xmin

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert rl == approx(self.soln_gen.rl, abs=1.e-12)
        assert ul == approx(self.soln_gen.ul, abs=1.e-12)
        assert pl == approx(self.soln_gen.pl, abs=1.e-12)

    def test_riem7revig_state1(self):
        """Using the ideal-gas EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem7revgen_state1(self):
        """Using the general EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar1

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-7)
        assert u == approx(ul, abs=1.e-6)
        assert p == approx(pl, abs=1.e-6)

    def test_riem7revig_state2(self):
        """Using the ideal-gas EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem7revgen_state2(self):
        """Using the general EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar, self.rstar2

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-7)
        assert u == approx(ul, abs=1.e-6)
        assert p == approx(pl, abs=1.e-6)

    def test_riem7revig_state3(self):
        """Using the ideal-gas EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pr, self.rr, self.ur, self.gr, x,
                                      self.xd0, self.t, self.soln_ig)

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riem7revgen_state3(self):
        """Using the general EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        r, p, u = rho_p_u_rarefaction(self.pr, self.rr, self.ur, self.gr, x,
                                      self.xd0, self.t, self.soln_gen)

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-8)
        assert u == approx(ul, abs=1.e-9)
        assert p == approx(pl, abs=1.e-7)

    def test_riem7revig_state4(self):
        """Using the ideal-gas EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],Xregs[4]) returns correct values.
        x = diff([self.Xregs[3], self.xmax])[0] * numpy.random.rand() + self.Xregs[3]
        _argmin = argmin(abs(self.soln_ig.x - x))
        x = self.soln_ig.x[_argmin]

        rl = interp(x, self.soln_ig.x, self.soln_ig.r)
        ul = interp(x, self.soln_ig.x, self.soln_ig.u)
        pl = interp(x, self.soln_ig.x, self.soln_ig.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)

    def test_riem7revgen_state4(self):
        """Using the general EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],Xregs[4]) returns correct values.
        x = diff([self.Xregs[3], self.xmax])[0] * numpy.random.rand() + self.Xregs[3]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)


class Test_RiemannJWL_Shyue():
    """Tests general EOS (GenEOS) solutions produced by :class:`exactpack.solvers.riemann.riemann` on Shyue's JWL Riemann problem [Shyue2001]_.
    """

    # Shyue's JWL Riemann Problem
    xmin, xd0, xmax, t = 0.0, 50.0, 100.0, 12.0
    rl, ul, pl, gl = 1.7, 0., 10.0, 1.25
    rr, ur, pr, gr = 1.0, 0.,  0.5, 1.25
    A, B, R1, R2, r0, e0  = 8.545, 0.205, 4.6, 1.35, 1.84, 0.0
    problem = 'JWL'
    num_x_pts, num_int_pts, int_tol = 10001, 10001, 1.e-12

    soln_gen = RiemannGenEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                             rl=rl, ul=ul, pl=pl, gl=gl,
                             rr=rr, ur=ur, pr=pr, gr=gr,
                             A=A, B=B, R1=R1, R2=R2, r0=r0, e0=e0,
                             problem='JWL')
    soln_gen.driver()

    # Test that star state values are computed correctly.
    pstar  = 4.409360935472674
    ustar1 = 1.6945593277042843
    ustar2 = 1.6945593277033082
    rstar1 = 0.8882712086127579
    rstar2 = 3.7668623157768932
    estar1 = 19.798338129896145
    estar2 = 3.7296576856663672
    astar1 = 2.4963291337873743
    astar2 = 1.2954310985969357

    # Test that spatial region boundaries are computed correctly.
    # Xregs = Vregs * t + xd0
    Xregs = array([17.17071528464863, 40.37876232700292,
                   70.33471193245141, 77.68408853918551])
    Vregs = array([-2.7357737262792807, -0.80176980608309,
                    1.6945593277042843, 2.3070073782654594])

    def test_riemShyuegen_star_states(self):
        """Using the general EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_gen.px,  abs=1.e-12)
        assert self.ustar1 == approx(self.soln_gen.ux1, abs=1.e-12)
        assert self.ustar2 == approx(self.soln_gen.ux2, abs=1.e-12)
        assert self.rstar1 == approx(self.soln_gen.rx1, abs=1.e-12)
        assert self.rstar2 == approx(self.soln_gen.rx2, abs=1.e-12)
        assert self.estar1 == approx(self.soln_gen.ex1, abs=1.e-12)
        assert self.estar2 == approx(self.soln_gen.ex2, abs=1.e-12)
        assert self.astar1 == approx(self.soln_gen.ax1, abs=1.e-12)
        assert self.astar2 == approx(self.soln_gen.ax2, abs=1.e-12)

    def test_riemShyuegen_region_boundaries(self):
        """Using the general EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_gen.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_gen.Xregs[1], abs=1.e-12)
        assert self.Xregs[2] == approx(self.soln_gen.Xregs[2], abs=1.e-12)
        assert self.Xregs[3] == approx(self.soln_gen.Xregs[3], abs=1.e-12)
        assert self.Vregs[0] == approx(self.soln_gen.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_gen.Vregs[1], abs=1.e-12)
        assert self.Vregs[2] == approx(self.soln_gen.Vregs[2], abs=1.e-12)
        assert self.Vregs[3] == approx(self.soln_gen.Vregs[3], abs=1.e-12)

    def test_riemShyuegen_state0(self):
        """Using the general EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x = diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() +self.xmin

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riemShyuegen_state2(self):
        """Using the general EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar1, self.rstar1

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riemShyuegen_state3(self):
        """Using the general EOS solver, test that a random point in region 3 gives the correct state values.
        """
        # Test that any point in (Xregs[2],Xregs[3]) returns correct values.
        x = diff(self.Xregs[2:])[0] * numpy.random.rand() + self.Xregs[2]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar2, self.rstar2

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riemShyuegen_state4(self):
        """Using the general EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],xmax) returns correct values.
        x = diff([self.Xregs[3], self.xmax])[0] * numpy.random.rand() + self.Xregs[3]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)


class Test_RiemannJWL_Lee():
    """Tests general EOS (GenEOS) solutions produced by :class:`exactpack.solvers.riemann.riemann` on Lee's JWL Riemann problem [Lee2013]_.
    """

    # Lee's JWL Riemann Problem
    xmin, xd0, xmax, t = 0.0, 50.0, 100.0, 20.0
    rl, ul, pl, gl = 0.9525, 0., 1.0, 1.8938
    rr, ur, pr, gr = 3.81,   0., 2.0, 1.8938
    A, B, R1, R2, r0, e0  = 632.1, -0.04472, 11.3, 1.13, 1.905, 0.0
    problem = 'JWL'
    num_x_pts, num_int_pts, int_tol = 10001, 10001, 1.e-12

    soln_gen = RiemannGenEOS(xmin=xmin, xd0=xd0, xmax=xmax, t=t,
                             rl=rl, ul=ul, pl=pl, gl=gl,
                             rr=rr, ur=ur, pr=pr, gr=gr,
                             A=A, B=B, R1=R1, R2=R2, r0=r0, e0=e0,
                             problem='JWL')
    soln_gen.driver()

    # Test that star state values are computed correctly.
    pstar  = 1.1906972944417038
    ustar1 = -0.13281939826625017
    ustar2 = -0.1328193982662211
    rstar1 = 1.0445383101056884
    rstar2 = 3.5165512209201375
    estar1 = 1.2812058376690656
    estar2 = 0.008405040874636158
    astar1 = 1.469701835538411
    astar2 = 1.525557446779137

    # Test that spatial region boundaries are computed correctly.
    # Xregs = Vregs * t + xd0
    Xregs = array([19.852754491950297, 47.343612034675,
                   77.85476097025833, 85.80228712253455])
    Vregs = array([-1.5073622754024851, -0.13281939826625017,
                    1.392738048512916, 1.7901143561267274])

    def test_riemLeegen_star_states(self):
        """Using the general EOS solver, test star-state values adjacent to the contact discontinuity.
        """
        # Test that star state values are computed correctly.
        assert self.pstar  == approx(self.soln_gen.px,  abs=1.e-12)
        assert self.ustar1 == approx(self.soln_gen.ux1, abs=1.e-12)
        assert self.ustar2 == approx(self.soln_gen.ux2, abs=1.e-12)
        assert self.rstar1 == approx(self.soln_gen.rx1, abs=1.e-12)
        assert self.rstar2 == approx(self.soln_gen.rx2, abs=1.e-12)
        assert self.estar1 == approx(self.soln_gen.ex1, abs=1.e-12)
        assert self.estar2 == approx(self.soln_gen.ex2, abs=1.e-12)
        assert self.astar1 == approx(self.soln_gen.ax1, abs=1.e-12)
        assert self.astar2 == approx(self.soln_gen.ax2, abs=1.e-12)

    def test_riemLeegen_region_boundaries(self):
        """Using the general EOS solver, test that velocities separating boundaries are computed correctly, and that spatial boundaries from these are computed correctly.
        """
        # Test that spatial region boundaries are computed correctly.
        # Xregs = Vregs * t + xd0
        assert self.Xregs[0] == approx(self.soln_gen.Xregs[0], abs=1.e-12)
        assert self.Xregs[1] == approx(self.soln_gen.Xregs[1], abs=1.e-12)
        assert self.Xregs[2] == approx(self.soln_gen.Xregs[2], abs=1.e-12)
        assert self.Xregs[3] == approx(self.soln_gen.Xregs[3], abs=1.e-12)
        assert self.Vregs[0] == approx(self.soln_gen.Vregs[0], abs=1.e-12)
        assert self.Vregs[1] == approx(self.soln_gen.Vregs[1], abs=1.e-12)
        assert self.Vregs[2] == approx(self.soln_gen.Vregs[2], abs=1.e-12)
        assert self.Vregs[3] == approx(self.soln_gen.Vregs[3], abs=1.e-12)

    def test_riemLeegen_state0(self):
        """Using the general EOS solver, test that a random point in region 0 gives the correct state values.
        """
        # Test than any point in (xmin,Xregs[0]) returns left state values.
        x = diff([self.xmin, self.Xregs[0]])[0] * numpy.random.rand() +self.xmin

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rl == approx(rl, abs=1.e-12)
        assert self.ul == approx(ul, abs=1.e-12)
        assert self.pl == approx(pl, abs=1.e-12)

    def test_riemLeegen_state1(self):
        """Using the general EOS solver, test that a random point in region 1 gives the correct state values.
        """
        # Test that any point in (Xregs[0],Xregs[1]) returns correct values.
        x = diff(self.Xregs[:2])[0] * numpy.random.rand() + self.Xregs[0]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar1, self.rstar1

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riemLeegen_state2(self):
        """Using the general EOS solver, test that a random point in region 2 gives the correct state values.
        """
        # Test that any point in (Xregs[1],Xregs[2]) returns correct values.
        x = diff(self.Xregs[1:3])[0] * numpy.random.rand() + self.Xregs[1]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]
        p, u, r = self.pstar, self.ustar2, self.rstar2

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert r == approx(rl, abs=1.e-12)
        assert u == approx(ul, abs=1.e-12)
        assert p == approx(pl, abs=1.e-12)

    def test_riemLeegen_state4(self):
        """Using the general EOS solver, test that a random point in region 4 gives the correct state values.
        """
        # Test that any point in (Xregs[3],xmax) returns correct values.
        x = diff([self.Xregs[3], self.xmax])[0] * numpy.random.rand() + self.Xregs[3]
        _argmin = argmin(abs(self.soln_gen.x - x))
        x = self.soln_gen.x[_argmin]

        rl = interp(x, self.soln_gen.x, self.soln_gen.r)
        ul = interp(x, self.soln_gen.x, self.soln_gen.u)
        pl = interp(x, self.soln_gen.x, self.soln_gen.p)
        assert self.rr == approx(rl, abs=1.e-12)
        assert self.ur == approx(ul, abs=1.e-12)
        assert self.pr == approx(pl, abs=1.e-12)
