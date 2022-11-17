"""Unittests for the 1D Riemann solvers.
"""

import unittest
import pytest
import warnings
from pytest import approx
from numpy import pi, sqrt, array, arcsin

from numpy.random import rand

from exactpack.solvers.oblique_shocks.oblique_shocks import TwoShockTheory

warnings.simplefilter('ignore', RuntimeWarning)


class TestObliqueShockSetup():
    """Tests problem setup of :class:`exactpack.solvers.oblique_shock.oblique_shock`.

       These tests confirm proper assignment of variables, including default
       values.
    """

    def test_defaults(self):
        """Test that default values are set accurately and problem is initialized corretly.
        """

        # here are the defaults
        p0 = 1.0
        rho0 = 0.001
        temp0 = 0.025
        gamma = 1.4
        M0_normal = 2.0
        theta_wedge_deg = 80.
        chi0_deg = 0.

        solver = TwoShockTheory()

        assert solver.p0 == p0
        assert solver.rho0 == rho0
        assert solver.temp0 == temp0
        assert solver.gamma == gamma
        assert solver.M0_normal == M0_normal
        assert solver.theta_wedge_deg == theta_wedge_deg
        assert solver.chi0_deg == chi0_deg

        theta_wedge_rad = theta_wedge_deg * pi / 180.
        assert solver.theta_wedge_rad == theta_wedge_rad
        assert solver.chi0_rad == chi0_deg * pi / 180.
        assert solver.phi0_rad == pi / 2. - theta_wedge_rad
        assert solver.c0 == sqrt(gamma * p0 * 1.e6 / rho0)


    def test_NOTdefaults(self):
        """Test that non-default values are set accurately and problem is initialized corretly.
        """

        # here are the defaults
        p0 = 2.0
        rho0 = 1.
        temp0 = 0.25
        gamma = 5./3.
        M0_normal = 1.3628
        theta_wedge_deg = 70.
        chi0_deg = 1.

        solver = TwoShockTheory(p0=2.0, rho0=1., temp0=0.25, gamma=5./3.,
                              M0_normal=1.3628, theta_wedge_deg=70.,
                              chi0_deg=1.)

        assert solver.p0 == p0
        assert solver.rho0 == rho0
        assert solver.temp0 == temp0
        assert solver.gamma == gamma
        assert solver.M0_normal == M0_normal
        assert solver.theta_wedge_deg == theta_wedge_deg
        assert solver.chi0_deg == chi0_deg

        assert solver.theta_wedge_rad == theta_wedge_deg * pi / 180.
        assert solver.chi0_rad == chi0_deg * pi / 180.
        assert solver.phi0_rad == pi / 2. - theta_wedge_deg * pi / 180.
        assert solver.c0 == sqrt(gamma * p0 * 1.e6 / rho0)


class TestEquations():
      """Tests shock oblique shock jump conditions.
      """

      solver = TwoShockTheory(M0_normal = 1.)
      rnum = rand()

      def test_normal_velocity_jump(self):
          """Test the normal_velocity_jump(self, u, Mn) function.
          """

          assert self.solver.normal_velocity_jump(self.rnum, 1.) == approx(self.rnum, abs=1.e-10)


      def test_normal_pressure_jump(self):
          """Test the normal_pressure_jump(self, p, Mn) function.
          """

          assert self.solver.normal_pressure_jump(self.rnum, 1.) == approx(self.rnum, abs=1.e-10)


      def test_normal_density_jump(self):
          """Test the normal_density_jump(self, r, Mn) function.
          """

          assert self.solver.normal_density_jump(self.rnum, 1.) == approx(self.rnum, abs=1.e-10)


      def test_normal_temperature_jump(self):
          """Test the normal_temperature_jump(self, T, Mn) function.
          """

          assert self.solver.normal_temperature_jump(self.rnum, 1.) == approx(self.rnum, abs=1.e-10)


      def test_normal_soundspeed_jump(self):
          """Test the normal_soundspeed_jump(self, c, Mn) function.
          """

          assert self.solver.normal_soundspeed_jump(self.rnum, 1.) == approx(self.rnum, abs=1.e-10)


      def test_normal_mach_jump(self):
          """Test the normal_mach_jump(self, Mn) function.
          """

          assert self.solver.normal_mach_jump(1.) == 1.


      def test_streaming_mach_number_from_normalMach(self):
          """Test the streaming_mach_number_from_normalMach(self, Mn, rad_angle) function.
          """

          assert self.solver.streaming_mach_number_from_normalMach(self.rnum, pi/2.) == approx(self.rnum, abs=1.e-10)


      def test_streaming_mach_number(self):
          """Test the streaming_mach_number(self, Ms, rad_angle) function.
          """

          assert self.solver.streaming_mach_number(1., pi/2.) == 1.


      def test_deflection_angle(self):
          """Test the deflection_angle(self, Ms, rad_angle) function.
          """

          assert self.solver.deflection_angle(1., pi/2.) == 0.


      def test_phi_sonic(self):
          """Test the phi_sonic(self, Ms, phi_left, phi_right) function.
          """

          assert self.solver.phi_sonic(1., pi/2.-1.e-10, pi/2.+1.e-10) == approx(pi/2, abs=1.e-9)


      def test_postshock_states(self):
          """Test the post_shock_states(self, c, r, T, M) function.
          """

          c, r, T = array([1., 2., 3.]) * self.rnum
          assert self.solver.postshock_states(c, r, T, 1.) == approx([c, r, T, 1., c], abs=1.e-9)


class Test_FirstShockJump():
      """Tests the first_shock_jump method.
      """

      solver = TwoShockTheory(M0_normal = 1., theta_wedge_deg = 0.)
      solver.first_shock_jump()


      def test_chi0_rad(self):
          """Test the value of chi0_rad.
          """

          assert self.solver.chi0_rad == 0.


      def test_phi0_rad(self):
          """Test the value of phi0_rad.
          """

          assert self.solver.phi0_rad == approx(pi/2., abs=1.e-10)


      def test_M0_normal(self):
          """Test the value of M0_normal.
          """

          assert self.solver.M0_normal == 1.


      def test_M0_streaming(self):
          """Test the value of M0_streaming.
          """

          assert self.solver.M0_streaming == 1.


      def test_phi0s_rad(self):
          """Test the values of phi0s_rad.
          """

          assert list(self.solver.phi0s_rad) == list(0.*self.solver.phi0s_rad+pi/2.)


      def test_M1_streaming(self):
          """Test the value of M1_streaming.
          """

          assert self.solver.M1_streaming == 1.


      def test_p1s(self):
          """Test the values of p1s.
          """

          assert list(self.solver.p1s) == list(0.*self.solver.p1s+1.)


      def test_theta1s_rad(self):
          """Test the values of theta1s_rad.
          """

          assert list(self.solver.theta1s_rad) == list(0.*self.solver.theta1s_rad)


      def test_p1(self):
          """Test the value of p1.
          """

          assert self.solver.p1 == 1.


      def test_theta1_rad(self):
          """Test the value of theta1_rad
          """

          assert self.solver.theta1_rad == 0.


      def test_p1_at_theta1_max(self):
          """Test the value of p1_at_theta1_max.
          """

          assert self.solver.p1_at_theta1_max == 1.


class Test_SecondShockJump():
      """Tests the second_shock_jump method.
      """

      solver = TwoShockTheory(p0=1.0, rho0=0.001, temp0=0.025, gamma=1.4,
                              M0_normal=2.0, theta_wedge_deg=80., chi0_deg=0.)
      solver.first_shock_jump()
      solver.second_shock_jump()


      def test_M1_streaming(self):
          """Test the value of M1_streaming.
          """

          assert self.solver.M1_streaming == approx(8.750576697528098, abs=1.e-12)


      def test_M1_mach_angle(self):
          """Test the value of arcsin(1. / M1_streaming), which is the M1 mach angle.
          """

          assert arcsin(1./self.solver.M1_streaming) == approx(0.1145283916812132, abs=1.e-12)


      def test_phi1s_rad(self):
          """Test the values of phi1s_rad.
          """

          assert self.solver.phi1s_rad[0] == approx(0.1145283916812132, abs=1.e-12)
          assert self.solver.phi1s_rad[-1] == approx(pi/2., abs=1.e-12)


      def test_p2s(self):
          """Test the values of p2s.
          """

          assert self.solver.p2s[0] == 4.5
          assert self.solver.p2s[-1] == approx(401.2561108314393, abs=1.e-12)


      def test_theta2s_rad(self):
          """Test the values of theta2s_rad.
          """

          assert self.solver.theta2s_rad[0] == 0.
          assert self.solver.theta2s_rad[-1] == approx(0., abs=1.e-12)


      def test_p2(self):
          """Test the value of p2.
          """

          assert self.solver.p2 == approx(14.823952517142363, abs=1.e-12)


      def test_phi1_rad(self):
          """Test the value of phi1_rad.
          """

          assert self.solver.phi1_rad == approx(0.19811982387305882, abs=1.e-12)


      def test_theta2_rad(self):
          """Test the value of theta2_rad.
          """

          assert self.solver.theta2_rad == approx(0.10850640431355532, abs=1.e-12)


      def test_omegaR_rad(self):
          """Test the value of omegaR_rad.
          """

          assert self.solver.omegaR_rad == approx(0.0896134195595035, abs=1.e-12)


      def test_M2_streaming(self):
          """Test the value of M2_streaming.
          """

          assert self.solver.M2_streaming == approx(7.093978958230159, abs=1.e-12)


class Test_CollectRemainingFlowValues():
      """Teste the collect_remaining_flow_values method.
      """

      solver = TwoShockTheory(p0=1.0, rho0=0.001, temp0=0.025, gamma=1.4,
                              M0_normal=2.0, theta_wedge_deg=80., chi0_deg=0.)
      solver._run()


      def test_phi0_sonic(self):
          """Test the value of phi0_sonic.
          """

          assert self.solver.phi0_sonic == approx(1.1771497560228688, abs=1.e-12)


      def test_theta1_sonic(self):
          """Test the value of theta1_sonic.
          """

          assert self.solver.theta1_sonic == approx(0.7803485156802776, abs=1.e-12)


      def test_p1_sonic(self):
          """Test the value of p1_sonic.
          """

          assert self.solver.p1_sonic == approx(131.82778162232452, abs=1.e-12)


      def test_phi1_sonic(self):
          """Test the value of phi1_sonic.
          """

          assert self.solver.phi1_sonic == approx(1.1728500090962333, abs=1.e-12)


      def test_theta2_sonic(self):
          """Test the value of theta2_sonic.
          """

          assert self.solver.theta2_sonic == approx(-0.6608348084792485, abs=1.e-12)


      def test_p2_sonic(self):
          """Test the value of p2_sonic.
          """

          assert self.solver.p2_sonic == approx(340.88429244506574, abs=1.e-12)


      def test_c1(self):
          """Test the value of c1.
          """

          assert self.solver.c1 == approx(48605.55523805894, abs=1.e-12)


      def test_rho1(self):
          """Test the value of rho1.
          """

          assert self.solver.rho1 == approx(0.0026666666666666666, abs=1.e-12)


      def test_temp1(self):
          """Test the value of temp1.
          """

          assert self.solver.temp1 == approx(0.042187499999999996, abs=1.e-12)


      def test_M1_normal(self):
          """Test the value of M1_normal.
          """

          assert self.solver.M1_normal == approx(0.5773502691896257, abs=1.e-12)


      def test_u1_streaming(self):
          """Test the value of u1_streaming.
          """

          assert self.solver.u1_streaming == approx(28062.430400804555, abs=1.e-12)


      def test_c2(self):
          """Test the value of c2.
          """

          assert self.solver.c2 == approx(59019.99187480064, abs=1.e-12)


      def test_rho2(self):
          """Test the value of rho2.
          """

          assert self.solver.rho2 == approx(0.005957907414456937, abs=1.e-12)


      def test_temp2(self):
          """Test the value of temp2.
          """

          assert self.solver.temp2 == approx(0.06220284715895598, abs=1.e-12)


      def test_M2_normal(self):
          """Test the value of M2_normal.
          """

          assert self.solver.M2_normal == approx(0.6348651855384337, abs=1.e-12)


      def test_u2_streaming(self):
          """Test the value of u2_streaming.
          """

          assert self.solver.u2_streaming == approx(37469.73809207216, abs=1.e-12)
