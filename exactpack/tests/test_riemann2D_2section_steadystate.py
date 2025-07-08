"""Unit tests for the steady-state 2-section 2D Riemann solver.
"""

import pytest
import warnings
from pytest import approx
from numpy import array, interp, diff, sqrt, abs, argmin, linspace, pi, append, nan
from numpy.random import rand

import numpy.random

from exactpack.solvers.riemann2D_2section_steadystate.ep_riemann2D_2section_steadystate import IGEOS_Solver
from exactpack.solvers.riemann2D_2section_steadystate.riemann2D_2section_steadystate import *

warnings.simplefilter('ignore', RuntimeWarning)

# MOST methods in SetupRiemannProblem are tested in this class. The REST need to
# be added.
class TestRiemannSetup():
    """Tests problem setup of :class:`exactpack.solvers.riemann2D_2section_steadystate.ep_riemann2D_2section_steadystate`.

       These tests confirm proper assignment of variables, including default
       values.
    """

    # here are the defaults
    bottom_state = [1., 1., 2.4, 0., 1.4]
    top_state = [0.25, 0.5, 7.0, 0., 1.4]
    pB, rB, MB, thetaB_deg, gB = bottom_state
    pT, rT, MT, thetaT_deg, gT = top_state
    thetaB_rad, thetaT_rad = array([thetaB_deg, thetaT_deg]) / 180. * pi
    muB_rad = thetaB_rad - arcsin(1. / MB)
    muT_rad = thetaT_rad + arcsin(1. / MT)
    cB = sqrt(gB * pB / rB)
    uB, vB = cB * MB * array([cos(thetaB_rad), sin(thetaB_rad)])
    cT = sqrt(gT * pT / rT)
    uT, vT = cT * MT * array([cos(thetaT_rad), sin(thetaT_rad)])
    t    = 0.25

    N=51
    xlim=(0., 0.02, 1.02)
    ylim=(0., 0.5, 0.8)
    ys = linspace(ylim[0], ylim[2], int(1e3)) - ylim[1]
    xs = 0. * ys + xlim[2] - 2. * xlim[1]

    r_state = [[x, y] for x, y in zip(xs, ys)]
    t = 1.
    prob = SetupRiemannProblem(bottom_state=bottom_state, top_state=top_state)

    solver = IGEOS_Solver(bottom_state=bottom_state,
                          top_state   =top_state)

    def test_defaults(self):
        """Test that default values are set accurately.
        """

        assert self.prob.bottom_state == self.bottom_state
        assert self.prob.top_state    == self.top_state

        assert self.solver.bottom_state == self.bottom_state
        assert self.solver.top_state == self.top_state


    soln = solver._run(r_state, t)
 
    def test_set_intial_state_values(self):
        """Test that initial state values are set correctly.
        """
        assert self.prob.pB         == self.pB
        assert self.prob.rB         == self.rB
        assert self.prob.MB         == self.MB
        assert self.prob.thetaB_deg == self.thetaB_deg
        assert self.prob.thetaB_rad == self.thetaB_deg / 180. * pi
        assert self.prob.gB         == self.gB
        assert self.prob.muB_rad    == self.muB_rad
        assert self.prob.uB         == self.uB
        assert self.prob.vB         == self.vB
        assert self.prob.rT         == self.rT
        assert self.prob.MT         == self.MT
        assert self.prob.thetaT_deg == self.thetaT_deg
        assert self.prob.thetaT_rad == self.thetaT_deg / 180. * pi
        assert self.prob.gT         == self.gT
        assert self.prob.muT_rad    == self.muT_rad
        assert self.prob.uT         == self.uT
        assert self.prob.vT         == self.vT


    def test_compression_states(self):
        bottom_result = self.prob.compression_states(self.prob.pB,
                                                     self.prob.bottom_state)
        assert bottom_result[0] == approx(0.0, abs=1.e-12)
        assert bottom_result[1] == approx(self.rB, abs=1.e-12)
        assert bottom_result[2] == approx(self.MB, abs=1.e-12)

        top_result = self.prob.compression_states(self.prob.pT,
                                                  self.prob.top_state)
        assert top_result[0] == approx(0.0, abs=1.e-12)
        assert top_result[1] == approx(self.rT, abs=1.e-12)
        assert top_result[2] == approx(self.MT, abs=1.e-12)


    def test_PrandtlMeyer_function(self):
        assert self.prob.PrandtlMeyer_function(1., 2. * self.gB) == 0.
        assert self.prob.PrandtlMeyer_function(1., 10. * self.gT) == 0.

        answer = 0.25856546088103793
        assert self.prob.PrandtlMeyer_function(2., 1.4) == approx(answer,
                                                                  abs=1.e-12)


    def test_expansion_states(self):
        bottom_result = self.prob.expansion_states(self.prob.pB,
                                                   self.prob.bottom_state)
        assert bottom_result[0] == approx(0.0, abs=1.e-12)
        assert bottom_result[1] == approx(self.rB, abs=1.e-12)
        assert bottom_result[2] == approx(self.MB, abs=1.e-12)

        top_result=self.prob.expansion_states(self.prob.pT,self.prob.top_state)
        assert top_result[0] == approx(0.0, abs=1.e-12)
        assert top_result[1] == approx(self.rT, abs=1.e-12)
        assert top_result[2] == approx(self.MT, abs=1.e-12)


    def test_test_for_nans(self):
        answer = linspace(0., 1., 3)

        ps = append(answer, 1.)
        ds = append(answer, nan)
        rs = append(answer, 2.)
        Ms = append(answer, 3.)
        result = self.prob.test_for_nans(ps, ds, rs, Ms)
        assert all(result[0] == answer)
        assert all(result[1] == answer)
        assert all(result[2] == answer)

        ps = append(answer, 1.)
        ds = append(answer, 2.)
        rs = append(answer, 3.)
        Ms = append(answer, nan)
        result = self.prob.test_for_nans(ps, ds, rs, Ms)
        assert all(result[0] == answer)
        assert all(result[1] == answer)
        assert all(result[2] == answer)


    def test_remove_subsonic_compression(self):
        answer = linspace(1.01, 1.02, 11)

        vec = append(answer, answer[-2])
        result = self.prob.remove_subsonic_compression([vec, vec, vec, vec])
        assert all(result[0] == result[1])
        assert all(result[0] == result[2])
        assert all(result[0] == result[3])
        assert all(result[0] == answer)

        vec = append(append(answer, 2.), 1.)
        result = self.prob.remove_subsonic_compression([vec, vec, vec, vec])
        assert all(result[0] == result[1])
        assert all(result[0] == result[2])
        assert all(result[0] == result[3])
        assert all(result[0] == answer)


    def test_setup_initial_arrays(self):
        result = self.prob.setup_initial_arrays(self.prob.bottom_state, num=10)
        assert len(result) == 2
        assert len(result[0]) == 4
        assert len(result[0][0]) == 4
        assert len(result[0][1]) == 4
        assert len(result[0][2]) == 4
        assert len(result[0][3]) == 4
        assert len(result[1]) == 4
        assert len(result[1][0]) == 9
        assert len(result[1][1]) == 9
        assert len(result[1][2]) == 9
        assert len(result[1][3]) == 9
        assert result[0][0][0] == approx(1., abs=1.e-12)
        assert result[0][0][1] == approx(2., abs=1.e-12)
        assert result[0][0][2] == approx(3., abs=1.e-12)
        assert result[0][0][3] == approx(4., abs=1.e-12)
        assert result[0][1][0] == approx(0., abs=1.e-12)
        assert result[0][1][1] == approx(0.20240894313819316, abs=1.e-12)
        assert result[0][1][2] == approx(0.3361148907311689, abs=1.e-12)
        assert result[0][1][3] == approx(0.43423205969044076, abs=1.e-12)
        assert result[0][2][0] == approx(1., abs=1.e-12)
        assert result[0][2][1] == approx(1.625, abs=1.e-12)
        assert result[0][2][2] == approx(2.111111111111111, abs=1.e-12)
        assert result[0][2][3] == approx(2.5, abs=1.e-12)
        assert result[0][3][0] == approx(2.4, abs=1.e-12)
        assert result[0][3][1] == approx(1.93455421221531, abs=1.e-12)
        assert result[0][3][2] == approx(1.6036994269038858, abs=1.e-12)
        assert result[0][3][3] == approx(1.3133925536563698, abs=1.e-12)
        assert result[1][0][0] == approx(1.00000000e-10, abs=1.e-12)
        assert result[1][0][1] == approx(0.11111111119999999, abs=1.e-12)
        assert result[1][0][2] == approx(0.2222222223, abs=1.e-12)
        assert result[1][0][3] == approx(0.3333333334, abs=1.e-12)
        assert result[1][0][4] == approx(0.4444444445, abs=1.e-12)
        assert result[1][0][5] == approx(0.5555555556, abs=1.e-12)
        assert result[1][0][6] == approx(0.6666666667, abs=1.e-12)
        assert result[1][0][7] == approx(0.7777777777999999, abs=1.e-12)
        assert result[1][0][8] == approx(0.8888888889, abs=1.e-12)
        assert result[1][1][0] == approx(-1.7901457681300053, abs=1.e-12)
        assert result[1][1][1] == approx(-0.5152011327711128, abs=1.e-12)
        assert result[1][1][2] == approx(-0.3693605693223827, abs=1.e-12)
        assert result[1][1][3] == approx(-0.27660990476812786, abs=1.e-12)
        assert result[1][1][4] == approx(-0.20745536552257193, abs=1.e-12)
        assert result[1][1][5] == approx(-0.15199905022697768, abs=1.e-12)
        assert result[1][1][6] == approx(-0.10562655732471304, abs=1.e-12)
        assert result[1][1][7] == approx(-0.06578975546654364, abs=1.e-12)
        assert result[1][1][8] == approx(-0.03092487368929775, abs=1.e-12)
        assert result[1][2][0] == approx(7.196856730011517e-08, abs=1.e-12)
        assert result[1][2][1] == approx(0.2081604450716687, abs=1.e-12)
        assert result[1][2][2] == approx(0.3415227455193763, abs=1.e-12)
        assert result[1][2][3] == approx(0.45624603561257854, abs=1.e-12)
        assert result[1][2][4] == approx(0.5603263659706345, abs=1.e-12)
        assert result[1][2][5] == approx(0.6571469497297188, abs=1.e-12)
        assert result[1][2][6] == approx(0.7485495080224344, abs=1.e-12)
        assert result[1][2][7] == approx(0.8356793384756593, abs=1.e-12)
        assert result[1][2][8] == approx(0.9193110577440997, abs=1.e-12)
        assert result[1][3][0] == approx(87.97055099004656, abs=1.e-12)
        assert result[1][3][1] == approx(3.8933606928479914, abs=1.e-12)
        assert result[1][3][2] == approx(3.3965469718907775, abs=1.e-12)
        assert result[1][3][3] == approx(3.1189135971726616, abs=1.e-12)
        assert result[1][3][4] == approx(2.9266877726285343, abs=1.e-12)
        assert result[1][3][5] == approx(2.7798600902468062, abs=1.e-12)
        assert result[1][3][6] == approx(2.661125524825541, abs=1.e-12)
        assert result[1][3][7] == approx(2.5614501232050633, abs=1.e-12)
        assert result[1][3][8] == approx(2.475532337468696, abs=1.e-12)


    def test_determine_state_functions(self):
        result = self.prob.determine_state_functions(self.prob.pressure_solution)
        assert result[0](1.) == approx(0.16167897721289134, abs=1.e-12)
        assert result[1](1.) == approx(0., abs=1.e-12)


    def test_find_overlap(self):
        self.prob.pressure_solution = 0.
        self.prob.deflection_angle_solution = 0.
        self.prob.find_overlap()
        assert self.prob.pressure_solution == approx(0.6614470466220885, abs=1.e-12)
        assert self.prob.deflection_angle_solution == approx(0.10764352262440605, abs=1.e-12)


    def test_determine_shock_angle(self):
        assert self.prob.determine_shock_angle(self.bottom_state) == approx(0.5152533137025468, abs=1.e-12)
        assert self.prob.determine_shock_angle(self.top_state) == approx(0.22366519804004067, abs=1.e-12)


    def test_set_starstate_values(self):
        self.prob.angles = 0
        self.prob.rB_star, self.prob.MB_star = 0., 0.
        self.prob.rT_star, self.prob.MT_star = 0., 0.
        self.prob.uB_star, self.prob.vB_star = 0., 0.
        self.prob.uT_star, self.prob.vT_star = 0., 0.
        self.prob.bottom_star_vals, self.prob.bottom_star_vals = 0., 0.
        self.prob.set_starstate_values()
        assert self.prob.rB_star == approx(0.744358584230489, abs=1.e-12)
        assert self.prob.rT_star == approx(0.9758930449685297, abs=1.e-12)
        assert self.prob.MB_star == approx(2.666224329994313, abs=1.e-12)
        assert self.prob.MT_star == approx(5.902165537923447, abs=1.e-12)
        assert self.prob.uB_star == approx(2.956624410526687, abs=1.e-12)
        assert self.prob.uT_star == approx(5.716113533924569, abs=1.e-12)
        assert self.prob.vB_star == approx(0.319496436289493, abs=1.e-12)
        assert self.prob.vT_star == approx(0.6176901932530923, abs=1.e-12)
        
 

class Test_example1():
    """Tests ideal-gas EOS (IGEOS) solutions produced on the first example of these steadystsate 2-section 2D Riemann problem.
    """

    # Riemann Problem 1: states = [p, r, M, theta, g]
    top_state = [0.25, 0.5, 7.0, 0., 1.4]
    bottom_state = [1., 1., 2.4, 0., 1.4]
    N=101
    xlim=(0., 0.02, 1.02)
    ylim=(0., 0.5, 0.8)
    ys = linspace(ylim[0], ylim[2], int(1e4)) - ylim[1]
    xs = 0. * ys + xlim[2] - 2. * xlim[1]
    r_state = [[x, y] for x, y in zip(xs, ys)]
    t = 1.

    pB, rB, MB, thetaB_deg, gB = bottom_state
    pT, rT, MT, thetaT_deg, gT = top_state

    thetaB_rad, thetaT_rad = array([thetaB_deg, thetaT_deg]) / 180. * pi
    
    eB = pB / rB / (gB - 1.)
    cB = sqrt(gB * pB / rB)
    uB, vB = MB * cB * array([cos(thetaB_rad), sin(thetaB_rad)])
    qB = sqrt(uB**2 + vB**2)
    eT = pT / rT / (gT - 1.)
    cT = sqrt(gT * pT / rT)
    uT, vT = MT * cT * array([cos(thetaT_rad), sin(thetaT_rad)])
    qT = sqrt(uT**2 + vT**2)

    prob = SetupRiemannProblem(bottom_state=bottom_state, top_state = top_state)
    solver = IGEOS_Solver(bottom_state=bottom_state, top_state=top_state)
    soln = solver._run(r_state, t)

    # Test that star state values are computed correctly: [p, r, M, u, v].
    bottom_star_vals = array([0.6614470466220885,0.744358584230489,
                              2.666224329994313, 2.956624410526687,
                              0.319496436289493])
    top_star_vals = array([0.6614470466220885, 0.9758930449685297,
                           5.902165537923447,  5.716113533924569,
                           0.6176901932530923])
    deflection_angle_solution = 0.10764352262440605
    pressure_solution = 0.6614470466220885

    # Test that angles and morphology are computed corrected.
    angles = {'BR': array([-0.4297754313045277, -0.32213190868012165]),
              'CD': 0.10764352262440605,
              'TS': 0.22366519804004067}
    morphology = 'R-C-S'

    def test_ex1_star_states(self):
        """Test star-state values.
        """
        # Test that star-state values are computed correctly.
        assert self.bottom_star_vals[0] == approx(self.solver.bottom_star_vals[0], abs=1.e-12)
        assert self.bottom_star_vals[1] == approx(self.solver.bottom_star_vals[1], abs=1.e-12)
        assert self.bottom_star_vals[2] == approx(self.solver.bottom_star_vals[2], abs=1.e-12)
        assert self.bottom_star_vals[3] == approx(self.solver.bottom_star_vals[3], abs=1.e-12)
        assert self.bottom_star_vals[4] == approx(self.solver.bottom_star_vals[4], abs=1.e-12)
        assert self.top_star_vals[0] == approx(self.solver.top_star_vals[0], abs=1.e-12)
        assert self.top_star_vals[1] == approx(self.solver.top_star_vals[1], abs=1.e-12)
        assert self.top_star_vals[2] == approx(self.solver.top_star_vals[2], abs=1.e-12)
        assert self.top_star_vals[3] == approx(self.solver.top_star_vals[3], abs=1.e-12)
        assert self.top_star_vals[4] == approx(self.solver.top_star_vals[4], abs=1.e-12)


    def test_ex1_velocity_angles(self):
        """Test that velocity angles are set correctly.
        """
        assert self.thetaB_rad == approx(self.solver.thetaB_rad, abs=1.e-12)
        assert self.thetaT_rad == approx(self.solver.thetaT_rad, abs=1.e-12)

    def test_ex1_deflection_angle(self):
        """Test deflection angle values.
        """
        # Test that star-state values are consistent with deflection angle.
        assert self.deflection_angle_solution == approx(self.solver.deflection_angle_solution, abs=1.e-12)

    def test_ex1_pressure_solution(self):
        """Test pressure solution value.
        """
        # Test that the pressure solution is computed correctly.
        assert self.pressure_solution == approx(self.solver.pressure_solution, abs=1.e-12)

    def test_ex1_angles(self):
        """Test angle values.
        """
        # Test that angle values are computed correctly.
        assert self.angles['BR'][0] == approx(self.solver.angles['BR'][0], abs=1.e-12)
        assert self.angles['BR'][1] == approx(self.solver.angles['BR'][1], abs=1.e-12)
        assert self.angles['CD']    == approx(self.solver.angles['CD'], abs=1.e-12)
        assert self.angles['TS']    == approx(self.solver.angles['TS'], abs=1.e-12)

    def test_ex1_morphology(self):
        """Test morphology string.
        """
        # Test that morphology is set correctly.
        assert self.morphology == self.solver.morphology

    def test_ex1_state0(self):
        """Test that a random angle value less than angles['BR'][0] takes bottom-state values.
        """
        # Test that any angle < angles['BR'][0] returns bottom-state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [-pi/2., self.angles['BR'][0]])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        e = interp(val, arctan(y/x), self.soln['specific_internal_energy'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        q = interp(val, arctan(y/x), self.soln['speed'])
        assert p == approx(self.bottom_state[0], abs=1.e-12)
        assert r == approx(self.bottom_state[1], abs=1.e-12)
        assert M == approx(self.bottom_state[2], abs=1.e-12)
        assert e == approx(self.eB, abs=1.e-12)
        assert u == approx(self.uB, abs=1.e-12)
        assert v == approx(self.vB, abs=1.e-12)
        assert q == approx(self.qB, abs=1.e-12)
    

    def test_ex1_state2(self):
        """Test that a random angle value between angles['BR'][1] and angles['CD'] takes bottom-star state values.
        """
        # Test that any angles['BR'][1] < angle < angles['CD'] returns
        # bottom-star state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['BR'][1], self.angles['CD']])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        assert p == approx(self.bottom_star_vals[0], abs=1.e-12)
        assert r == approx(self.bottom_star_vals[1], abs=1.e-12)
        assert M == approx(self.bottom_star_vals[2], abs=1.e-12)
        assert u == approx(self.bottom_star_vals[3], abs=1.e-12)
        assert v == approx(self.bottom_star_vals[4], abs=1.e-12)
    

    def test_ex1_state3(self):
        """Test that any angles['CD'] < angles < angles['TS'] returns top-star state values.
        """
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['CD'], self.angles['TS']])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        assert p == approx(self.top_star_vals[0], abs=1.e-12)
        assert r == approx(self.top_star_vals[1], abs=1.e-12)
        assert M == approx(self.top_star_vals[2], abs=1.e-12)
        assert u == approx(self.top_star_vals[3], abs=1.e-12)
        assert v == approx(self.top_star_vals[4], abs=1.e-12)
    

    def test_ex1_state4(self):
        """Test that any angle > angles['TS'] returns top-state values.
        """
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['TS'], pi/2.])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        e = interp(val, arctan(y/x), self.soln['specific_internal_energy'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        q = interp(val, arctan(y/x), self.soln['speed'])
        assert p == approx(self.top_state[0], abs=1.e-12)
        assert r == approx(self.top_state[1], abs=1.e-12)
        assert M == approx(self.top_state[2], abs=1.e-12)
        assert e == approx(self.eT, abs=1.e-12)
        assert u == approx(self.uT, abs=1.e-12)
        assert v == approx(self.vT, abs=1.e-12)
        assert q == approx(self.qT, abs=1.e-12)
    

class Test_example1_reversed():
    """Tests ideal-gas EOS (IGEOS) solutions produced on the first example of these steadystsate 2-section 2D Riemann problem, but with top and bottom states reversed.
    """

    # Riemann Problem 1: states = [p, r, M, theta, g]
    top_state = [1., 1., 2.4, 0., 1.4]
    bottom_state = [0.25, 0.5, 7.0, 0., 1.4]
    N=101
    xlim=(0., 0.02, 1.02)
    ylim=(0., 0.3, 0.8)
    ys = linspace(ylim[0], ylim[2], int(1e4)) - ylim[1]
    xs = 0. * ys + xlim[2] - 2. * xlim[1]
    r_state = [[x, y] for x, y in zip(xs, ys)]
    t = 1.

    pB, rB, MB, thetaB_deg, gB = bottom_state
    pT, rT, MT, thetaT_deg, gT = top_state

    thetaB_rad, thetaT_rad = array([thetaB_deg, thetaT_deg]) / 180. * pi
    
    eB = pB / rB / (gB - 1.)
    cB = sqrt(gB * pB / rB)
    uB, vB = MB * cB * array([cos(thetaB_rad), sin(thetaB_rad)])
    qB = sqrt(uB**2 + vB**2)
    eT = pT / rT / (gT - 1.)
    cT = sqrt(gT * pT / rT)
    uT, vT = MT * cT * array([cos(thetaT_rad), sin(thetaT_rad)])
    qT = sqrt(uT**2 + vT**2)

    prob = SetupRiemannProblem(bottom_state=bottom_state, top_state = top_state)
    solver = IGEOS_Solver(bottom_state=bottom_state, top_state=top_state)
    soln = solver._run(r_state, t)

    # Test that star state values are computed correctly: [p, r, M, u, v].
    bottom_star_vals = array([0.6614470466220885, 0.9758930449685297,
                              5.902165537923447,  5.716113533924569,
                              -0.6176901932530915])
    top_star_vals = array([0.6614470466220885,0.744358584230489,
                           2.666224329994313, 2.956624410526687,
                           -0.31949643628949254])
    deflection_angle_solution = -0.10764352262440605
    pressure_solution = 0.6614470466220885

    # Test that angles and morphology are computed correctly.
    angles = {'BS': -0.22366519804004054,
              'CD': -0.10764352262440591,
              'TR': array([0.3221319086801218, 0.4297754313045277])}

    morphology = 'S-C-R'

    def test_ex1rev_star_states(self):
        """Test star-state values.
        """
        # Test that star-state values are computed correctly.
        assert self.bottom_star_vals[0] == approx(self.solver.bottom_star_vals[0], abs=1.e-12)
        assert self.bottom_star_vals[1] == approx(self.solver.bottom_star_vals[1], abs=1.e-12)
        assert self.bottom_star_vals[2] == approx(self.solver.bottom_star_vals[2], abs=1.e-12)
        assert self.bottom_star_vals[3] == approx(self.solver.bottom_star_vals[3], abs=1.e-12)
        assert self.bottom_star_vals[4] == approx(self.solver.bottom_star_vals[4], abs=1.e-12)
        assert self.top_star_vals[0] == approx(self.solver.top_star_vals[0], abs=1.e-12)
        assert self.top_star_vals[1] == approx(self.solver.top_star_vals[1], abs=1.e-12)
        assert self.top_star_vals[2] == approx(self.solver.top_star_vals[2], abs=1.e-12)
        assert self.top_star_vals[3] == approx(self.solver.top_star_vals[3], abs=1.e-12)
        assert self.top_star_vals[4] == approx(self.solver.top_star_vals[4], abs=1.e-12)


    def test_ex1rev_velocity_angles(self):
        """Test that velocity angles are set correctly.
        """
        assert self.thetaB_rad == approx(self.solver.thetaB_rad, abs=1.e-12)
        assert self.thetaT_rad == approx(self.solver.thetaT_rad, abs=1.e-12)

    def test_ex1rev_deflection_angle(self):
        """Test deflection angle values.
        """
        # Test that star-state values are consistent with deflection angle.
        assert self.deflection_angle_solution == approx(self.solver.deflection_angle_solution, abs=1.e-12)

    def test_ex1rev_pressure_solution(self):
        """Test pressure solution value.
        """
        # Test that the pressure solution is computed correctly.
        assert self.pressure_solution == approx(self.solver.pressure_solution, abs=1.e-12)

    def test_ex1rev_angles(self):
        """Test angle values.
        """
        # Test that angle values are computed correctly.
        assert self.angles['BS']    == approx(self.solver.angles['BS'], abs=1.e-12)
        assert self.angles['CD']    == approx(self.solver.angles['CD'], abs=1.e-12)
        assert self.angles['TR'][0] == approx(self.solver.angles['TR'][0], abs=1.e-12)
        assert self.angles['TR'][1] == approx(self.solver.angles['TR'][1], abs=1.e-12)

    def test_ex1rev_morphology(self):
        """Test morphology string.
        """
        # Test that morphology is set correctly.
        assert self.morphology == self.solver.morphology

    def test_ex1rev_state0(self):
        """Test that a random point with angle value less than angles['BS'] takes bottom-state values.
        """
        # Test that any angle < angles['BS'] returns bottom-state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [-pi/2., self.angles['BS']])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        e = interp(val, arctan(y/x), self.soln['specific_internal_energy'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        q = interp(val, arctan(y/x), self.soln['speed'])
        assert p == approx(self.bottom_state[0], abs=1.e-12)
        assert r == approx(self.bottom_state[1], abs=1.e-12)
        assert M == approx(self.bottom_state[2], abs=1.e-12)
        assert e == approx(self.eB, abs=1.e-12)
        assert u == approx(self.uB, abs=1.e-12)
        assert v == approx(self.vB, abs=1.e-12)
        assert q == approx(self.qB, abs=1.e-12)
    
    def test_ex1rev_state1(self):
        """Test that any angles['BS'] < angles < angles['CD'] returns bottom-star state values.
        """
        # Test that any angles['BS'] < angles < angles['CD'] returns bottom-star
        # state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['BS'], self.angles['CD']])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        assert p == approx(self.bottom_star_vals[0], abs=1.e-12)
        assert r == approx(self.bottom_star_vals[1], abs=1.e-12)
        assert M == approx(self.bottom_star_vals[2], abs=1.e-12)
        assert u == approx(self.bottom_star_vals[3], abs=1.e-12)
        assert v == approx(self.bottom_star_vals[4], abs=1.e-12)
    

    def test_ex1rev_state2(self):
        """Test that a random angle value between angles['CD'] and angles['TR'][0] takes top-star state values.
        """
        # Test that any angles['CD'] < angle < angles['TR'][0] returns top-star
        # state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['CD'], self.angles['TR'][0]])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        assert p == approx(self.top_star_vals[0], abs=1.e-12)
        assert r == approx(self.top_star_vals[1], abs=1.e-12)
        assert M == approx(self.top_star_vals[2], abs=1.e-12)
        assert u == approx(self.top_star_vals[3], abs=1.e-12)
        assert v == approx(self.top_star_vals[4], abs=1.e-12)
    

    def test_ex1rev_state4(self):
        """Test that a random angle value greater than angles['TR'][1] takes
           bottom-state values.
        """
        # Test that any angle < angles['TR'][1] returns bottom-state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['TR'][1], pi/2.])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        e = interp(val, arctan(y/x), self.soln['specific_internal_energy'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        q = interp(val, arctan(y/x), self.soln['speed'])
        assert p == approx(self.top_state[0], abs=1.e-12)
        assert r == approx(self.top_state[1], abs=1.e-12)
        assert M == approx(self.top_state[2], abs=1.e-12)
        assert e == approx(self.eT, abs=1.e-12)
        assert u == approx(self.uT, abs=1.e-12)
        assert v == approx(self.vT, abs=1.e-12)
        assert q == approx(self.qT, abs=1.e-12)
    

class Test_example2():
    """Tests ideal-gas EOS (IGEOS) solutions produced on the second example of these steadystsate 2-section 2D Riemann problem.
    """

    # Riemann Problem 2: states = [p, r, M, theta, g]
    top_state = [0.25, 0.5, 4.0, 0., 1.4]
    bottom_state = [1., 1., 2.4, 0., 1.4]
    N=101
    xlim=(0., 0.02, 0.42)
    ylim=(0., 0.22, 0.42)
    ys = linspace(ylim[0], ylim[2], int(1e4)) - ylim[1]
    xs = 0. * ys + xlim[2] - 2. * xlim[1]
    r_state = [[x, y] for x, y in zip(xs, ys)]
    t = 1.

    pB, rB, MB, thetaB_deg, gB = bottom_state
    pT, rT, MT, thetaT_deg, gT = top_state

    thetaB_rad, thetaT_rad = array([thetaB_deg, thetaT_deg]) / 180. * pi
    
    eB = pB / rB / (gB - 1.)
    cB = sqrt(gB * pB / rB)
    uB, vB = MB * cB * array([cos(thetaB_rad), sin(thetaB_rad)])
    qB = sqrt(uB**2 + vB**2)
    eT = pT / rT / (gT - 1.)
    cT = sqrt(gT * pT / rT)
    uT, vT = MT * cT * array([cos(thetaT_rad), sin(thetaT_rad)])
    qT = sqrt(uT**2 + vT**2)

    prob = SetupRiemannProblem(bottom_state=bottom_state, top_state = top_state)
    solver = IGEOS_Solver(bottom_state=bottom_state, top_state=top_state)
    soln = solver._run(r_state, t)

    # Test that star state values are computed correctly: [p, r, M, u, v].
    bottom_star_vals = array([0.5585718504665425, 0.659693454916581,
                              2.7763189991896016, 2.9885175344819777,
                              0.45361126482162734])
    top_star_vals = array([0.5585718504665425, 0.8747401996153419,
                           3.382796710295294,  3.1622331542798694,
                           0.4799786396509851])

    deflection_angle_solution = 0.15063492581686963
    pressure_solution = 0.5585718504665425

    # Test that angles and morphology are computed corrected.
    angles = {'BR': array([-0.4297754313045277, -0.2791405054876581]),
              'CD': 0.15063492581686963,
              'TS': 0.36681040009943555}
    morphology = 'R-C-S'

    def test_ex2_star_states(self):
        """Test star-state values.
        """
        # Test that star-state values are computed correctly.
        assert self.bottom_star_vals[0] == approx(self.solver.bottom_star_vals[0], abs=1.e-12)
        assert self.bottom_star_vals[1] == approx(self.solver.bottom_star_vals[1], abs=1.e-12)
        assert self.bottom_star_vals[2] == approx(self.solver.bottom_star_vals[2], abs=1.e-12)
        assert self.bottom_star_vals[3] == approx(self.solver.bottom_star_vals[3], abs=1.e-12)
        assert self.bottom_star_vals[4] == approx(self.solver.bottom_star_vals[4], abs=1.e-12)
        assert self.top_star_vals[0] == approx(self.solver.top_star_vals[0], abs=1.e-12)
        assert self.top_star_vals[1] == approx(self.solver.top_star_vals[1], abs=1.e-12)
        assert self.top_star_vals[2] == approx(self.solver.top_star_vals[2], abs=1.e-12)
        assert self.top_star_vals[3] == approx(self.solver.top_star_vals[3], abs=1.e-12)
        assert self.top_star_vals[4] == approx(self.solver.top_star_vals[4], abs=1.e-12)


    def test_ex2_velocity_angles(self):
        """Test that velocity angles are set correctly.
        """
        assert self.thetaB_rad == approx(self.solver.thetaB_rad, abs=1.e-12)
        assert self.thetaT_rad == approx(self.solver.thetaT_rad, abs=1.e-12)


    def test_ex2_deflection_angle(self):
        """Test deflection angle values.
        """
        # Test that star-state values are consistent with deflection angle.
        assert self.deflection_angle_solution == approx(self.solver.deflection_angle_solution, abs=1.e-12)


    def test_ex2_pressure_solution(self):
        """Test pressure solution value.
        """
        # Test that the pressure solution is computed correctly.
        assert self.pressure_solution == approx(self.solver.pressure_solution, abs=1.e-12)


    def test_ex2_angles(self):
        """Test angle values.
        """
        # Test that angle values are computed correctly.
        assert self.angles['BR'][0] == approx(self.solver.angles['BR'][0], abs=1.e-12)
        assert self.angles['BR'][1] == approx(self.solver.angles['BR'][1], abs=1.e-12)
        assert self.angles['CD']    == approx(self.solver.angles['CD'], abs=1.e-12)
        assert self.angles['TS']    == approx(self.solver.angles['TS'], abs=1.e-12)


    def test_ex2_morphology(self):
        """Test morphology string.
        """
        # Test that morphology is set correctly.
        assert self.morphology == self.solver.morphology

    def test_ex2_state0(self):
        """Test that a random angle value less than angles['BR'][0] takes
           bottom-state values.
        """
        # Test that any angle < angles['BR'][0] returns bottom-state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [-pi/2., self.angles['BR'][0]])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        e = interp(val, arctan(y/x), self.soln['specific_internal_energy'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        q = interp(val, arctan(y/x), self.soln['speed'])
        assert p == approx(self.bottom_state[0], abs=1.e-12)
        assert r == approx(self.bottom_state[1], abs=1.e-12)
        assert M == approx(self.bottom_state[2], abs=1.e-12)
        assert e == approx(self.eB, abs=1.e-12)
        assert u == approx(self.uB, abs=1.e-12)
        assert v == approx(self.vB, abs=1.e-12)
        assert q == approx(self.qB, abs=1.e-12)
    

    def test_ex2_state2(self):
        """Test that a random angle value between angles['BR'][1] and
           angles['CD'] takes bottom-star state values.
        """
        # Test that any angles['BR'][1] < angle < angles['CD'] returns
        # bottom-star state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['BR'][1], self.angles['CD']])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        assert p == approx(self.bottom_star_vals[0], abs=1.e-12)
        assert r == approx(self.bottom_star_vals[1], abs=1.e-12)
        assert M == approx(self.bottom_star_vals[2], abs=1.e-12)
        assert u == approx(self.bottom_star_vals[3], abs=1.e-12)
        assert v == approx(self.bottom_star_vals[4], abs=1.e-12)
    

    def test_ex2_state3(self):
        """Test that any angles['CD'] < angles < angles['TS'] returns                   top-star state values.
        """
        # Test that any angles['CD'] < angles < angles['TS'] returns
        # top-star state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['CD'], self.angles['TS']])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        assert p == approx(self.top_star_vals[0], abs=1.e-12)
        assert r == approx(self.top_star_vals[1], abs=1.e-12)
        assert M == approx(self.top_star_vals[2], abs=1.e-12)
        assert u == approx(self.top_star_vals[3], abs=1.e-12)
        assert v == approx(self.top_star_vals[4], abs=1.e-12)
    

    def test_ex2_state4(self):
        """Test that a random point with angle value less than angles['BR'][0]
           takes bottom-state values.
        """
        # Test that any angle < angles['BR'][0] returns bottom-state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['TS'], pi/2.])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        e = interp(val, arctan(y/x), self.soln['specific_internal_energy'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        q = interp(val, arctan(y/x), self.soln['speed'])
        assert p == approx(self.top_state[0], abs=1.e-12)
        assert r == approx(self.top_state[1], abs=1.e-12)
        assert M == approx(self.top_state[2], abs=1.e-12)
        assert e == approx(self.eT, abs=1.e-12)
        assert u == approx(self.uT, abs=1.e-12)
        assert v == approx(self.vT, abs=1.e-12)
        assert q == approx(self.qT, abs=1.e-12)
    

class Test_example2_reversed():
    """Tests ideal-gas EOS (IGEOS) solutions produced on the second example of these steadystsate 2-section 2D Riemann problem.
    """

    # Riemann Problem 2: states = [p, r, M, theta, g]
    top_state = [1., 1., 2.4, 0., 1.4]
    bottom_state = [0.25, 0.5, 4.0, 0., 1.4]
    N=101
    xlim=(0., 0.02, 0.42)
    ylim=(0., 0.22, 0.42)
    ys = linspace(ylim[0], ylim[2], int(1e4)) - ylim[1]
    xs = 0. * ys + xlim[2] - 2. * xlim[1]
    r_state = [[x, y] for x, y in zip(xs, ys)]
    t = 1.

    pB, rB, MB, thetaB_deg, gB = bottom_state
    pT, rT, MT, thetaT_deg, gT = top_state

    thetaB_rad, thetaT_rad = array([thetaB_deg, thetaT_deg]) / 180. * pi
    
    eB = pB / rB / (gB - 1.)
    cB = sqrt(gB * pB / rB)
    uB, vB = MB * cB * array([cos(thetaB_rad), sin(thetaB_rad)])
    qB = sqrt(uB**2 + vB**2)
    eT = pT / rT / (gT - 1.)
    cT = sqrt(gT * pT / rT)
    uT, vT = MT * cT * array([cos(thetaT_rad), sin(thetaT_rad)])
    qT = sqrt(uT**2 + vT**2)

    prob = SetupRiemannProblem(bottom_state=bottom_state, top_state = top_state)
    solver = IGEOS_Solver(bottom_state=bottom_state, top_state=top_state)
    soln = solver._run(r_state, t)

    # Test that star state values are computed correctly: [p, r, M, u, v].
    bottom_star_vals = array([0.5585718504665426, 0.8747401996153422,
                              3.382796710295294, 3.1622331542798685,
                              -0.4799786396509855])
    top_star_vals = array([0.5585718504665426, 0.6596934549165812,
                           2.7763189991896016, 2.9885175344819777,
                           -0.4536112648216277])

    deflection_angle_solution = -0.15063492581686963
    pressure_solution = 0.5585718504665425

    # Test that angles and morphology are computed corrected.
    angles = {'BS': -0.3668104000994357,
              'CD': -0.15063492581686977,
              'TR': array([0.27914050548765795, 0.4297754313045277])}
    morphology = 'S-C-R'

    def test_ex2rev_star_states(self):
        """Test star-state values.
        """
        # Test that star-state values are computed correctly.
        assert self.bottom_star_vals[0] == approx(self.solver.bottom_star_vals[0], abs=1.e-12)
        assert self.bottom_star_vals[1] == approx(self.solver.bottom_star_vals[1], abs=1.e-12)
        assert self.bottom_star_vals[2] == approx(self.solver.bottom_star_vals[2], abs=1.e-12)
        assert self.bottom_star_vals[3] == approx(self.solver.bottom_star_vals[3], abs=1.e-12)
        assert self.bottom_star_vals[4] == approx(self.solver.bottom_star_vals[4], abs=1.e-12)
        assert self.top_star_vals[0] == approx(self.solver.top_star_vals[0], abs=1.e-12)
        assert self.top_star_vals[1] == approx(self.solver.top_star_vals[1], abs=1.e-12)
        assert self.top_star_vals[2] == approx(self.solver.top_star_vals[2], abs=1.e-12)
        assert self.top_star_vals[3] == approx(self.solver.top_star_vals[3], abs=1.e-12)
        assert self.top_star_vals[4] == approx(self.solver.top_star_vals[4], abs=1.e-12)


    def test_ex2rev_velocity_angles(self):
        """Test that velocity angles are set correctly.
        """
        assert self.thetaB_rad == approx(self.solver.thetaB_rad, abs=1.e-12)
        assert self.thetaT_rad == approx(self.solver.thetaT_rad, abs=1.e-12)


    def test_ex2rev_deflection_angle(self):
        """Test deflection angle values.
        """
        # Test that star-state values are consistent with deflection angle.
        assert self.deflection_angle_solution == approx(self.solver.deflection_angle_solution, abs=1.e-12)


    def test_ex2rev_pressure_solution(self):
        """Test pressure solution value.
        """
        # Test that the pressure solution is computed correctly.
        assert self.pressure_solution == approx(self.solver.pressure_solution, abs=1.e-12)


    def test_ex2rev_angles(self):
        """Test angle values.
        """
        # Test that angle values are computed correctly.
        assert self.angles['BS']    == approx(self.solver.angles['BS'], abs=1.e-12)
        assert self.angles['CD']    == approx(self.solver.angles['CD'], abs=1.e-12)
        assert self.angles['TR'][0] == approx(self.solver.angles['TR'][0], abs=1.e-12)
        assert self.angles['TR'][1] == approx(self.solver.angles['TR'][1], abs=1.e-12)


    def test_ex2rev_morphology(self):
        """Test morphology string.
        """
        # Test that morphology is set correctly.
        assert self.morphology == self.solver.morphology


    def test_ex2rev_state0(self):
        """Test that a random point with angle value less than angles['BS']
           takes bottom-state values.
        """
        # Test that any angle < angles['BS'] returns bottom-state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [-pi/2., self.angles['BS']])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        e = interp(val, arctan(y/x), self.soln['specific_internal_energy'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        q = interp(val, arctan(y/x), self.soln['speed'])
        assert p == approx(self.bottom_state[0], abs=1.e-15)
        assert r == approx(self.bottom_state[1], abs=1.e-15)
        assert M == approx(self.bottom_state[2], abs=1.e-15)
        assert e == approx(self.eB, abs=1.e-15)
        assert u == approx(self.uB, abs=1.e-15)
        assert v == approx(self.vB, abs=1.e-15)
        assert q == approx(self.qB, abs=1.e-15)
    

    def test_ex2rev_state1(self):
        """Test that any angles['BS'] < angles < angles['CD'] returns bottom-star state values.
        """
        # Test that any angles['BS'] < angles < angles['CD'] returns bottom-star
        # state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['BS'], self.angles['CD']])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        assert p == approx(self.bottom_star_vals[0], abs=1.e-15)
        assert r == approx(self.bottom_star_vals[1], abs=1.e-15)
        assert M == approx(self.bottom_star_vals[2], abs=1.e-15)
        assert u == approx(self.bottom_star_vals[3], abs=1.e-15)
        assert v == approx(self.bottom_star_vals[4], abs=1.e-15)
    

    def test_ex2rev_state2(self):
        """Test that a random angle value between angles['CD'] and angles['TR'][0] takes top-star state values.
        """
        # Test that any angles['CD'] < angle < angles['TR'][0] returns
        # bottom-star state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['CD'], self.angles['TR'][0]])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        assert p == approx(self.top_star_vals[0], abs=1.e-15)
        assert r == approx(self.top_star_vals[1], abs=1.e-15)
        assert M == approx(self.top_star_vals[2], abs=1.e-15)
        assert u == approx(self.top_star_vals[3], abs=1.e-15)
        assert v == approx(self.top_star_vals[4], abs=1.e-15)
    

    def test_ex2rev_state4(self):
        """Test that a random angle value greater than angles['TR'][1] takes top-state values.
        """
        # Test that any angle > angles['TR'][1] returns top-state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['TR'][1], pi/2.])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        e = interp(val, arctan(y/x), self.soln['specific_internal_energy'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        q = interp(val, arctan(y/x), self.soln['speed'])
        assert p == approx(self.top_state[0], abs=1.e-15)
        assert r == approx(self.top_state[1], abs=1.e-15)
        assert M == approx(self.top_state[2], abs=1.e-15)
        assert e == approx(self.eT, abs=1.e-15)
        assert u == approx(self.uT, abs=1.e-15)
        assert v == approx(self.vT, abs=1.e-15)
        assert q == approx(self.qT, abs=1.e-15)

    
class Test_example3():
    """Tests ideal-gas EOS (IGEOS) solutions produced on the third example of these steadystsate 2-section 2D Riemann problem.
    """

    # Riemann Problem 3: states = [p, r, M, theta, g]
    top_state = [0.1, 0.2, 4.0, 0., 1.4]
    bottom_state = [1., 1., 6., 15., 1.4]
    N=101
    xlim=(0., 0.02, 0.32)
    ylim=(0., 0.01, 0.21)
    ys = linspace(ylim[0], ylim[2], int(1e4)) - ylim[1]
    xs = 0. * ys + xlim[2] - 2. * xlim[1]
    r_state = [[x, y] for x, y in zip(xs, ys)]
    t = 1.

    pB, rB, MB, thetaB_deg, gB = bottom_state
    pT, rT, MT, thetaT_deg, gT = top_state

    thetaB_rad, thetaT_rad = array([thetaB_deg, thetaT_deg]) / 180. * pi
    
    eB = pB / rB / (gB - 1.)
    cB = sqrt(gB * pB / rB)
    uB, vB = MB * cB * array([cos(thetaB_rad), sin(thetaB_rad)])
    qB = sqrt(uB**2 + vB**2)
    eT = pT / rT / (gT - 1.)
    cT = sqrt(gT * pT / rT)
    uT, vT = MT * cT * array([cos(thetaT_rad), sin(thetaT_rad)])
    qT = sqrt(uT**2 + vT**2)

    prob = SetupRiemannProblem(bottom_state=bottom_state, top_state = top_state)
    solver = IGEOS_Solver(bottom_state=bottom_state, top_state=top_state)
    soln = solver._run(r_state, t)

    # Test that star state values are computed correctly: [p, r, M, u, v].
    bottom_star_vals = [0.5168196501238399, 0.6240827168186247,
                        6.671530841696756,  6.75585028782287,
                        2.4416385864197707]
    top_star_vals = [0.5168196501238399, 0.5732201972606951,
                     2.577957598623868,  2.7238966426853453,
                     0.9844462006784749]


    deflection_angle_solution = 0.3468041303387218
    pressure_solution = 0.5168196501238399

    # Test that angles and morphology are computed corrected.
    angles = {'BR': array([0.09435130857946009, 0.1793560511190325]),
              'CD': 0.3468041303387218,
              'TS': 0.5640333454902834}
    morphology = 'R-C-S'

    def test_ex3_star_states(self):
        """Test star-state values.
        """
        # Test that star-state values are computed correctly.
        assert self.bottom_star_vals[0] == approx(self.solver.bottom_star_vals[0], abs=1.e-12)
        assert self.bottom_star_vals[1] == approx(self.solver.bottom_star_vals[1], abs=1.e-12)
        assert self.bottom_star_vals[2] == approx(self.solver.bottom_star_vals[2], abs=1.e-12)
        assert self.bottom_star_vals[3] == approx(self.solver.bottom_star_vals[3], abs=1.e-12)
        assert self.bottom_star_vals[4] == approx(self.solver.bottom_star_vals[4], abs=1.e-12)
        assert self.top_star_vals[0] == approx(self.solver.top_star_vals[0], abs=1.e-12)
        assert self.top_star_vals[1] == approx(self.solver.top_star_vals[1], abs=1.e-12)
        assert self.top_star_vals[2] == approx(self.solver.top_star_vals[2], abs=1.e-12)
        assert self.top_star_vals[3] == approx(self.solver.top_star_vals[3], abs=1.e-12)
        assert self.top_star_vals[4] == approx(self.solver.top_star_vals[4], abs=1.e-12)


    def test_ex3_velocity_angles(self):
        """Test that velocity angles are set correctly.
        """
        assert self.thetaB_rad == approx(self.solver.thetaB_rad, abs=1.e-12)
        assert self.thetaT_rad == approx(self.solver.thetaT_rad, abs=1.e-12)


    def test_ex3_deflection_angle(self):
        """Test deflection angle values.
        """
        # Test that star-state values are consistent with deflection angle.
        assert self.deflection_angle_solution == approx(self.solver.deflection_angle_solution, abs=1.e-12)


    def test_ex3_pressure_solution(self):
        """Test pressure solution value.
        """
        # Test that the pressure solution is computed correctly.
        assert self.pressure_solution == approx(self.solver.pressure_solution, abs=1.e-12)

    def test_ex3_angles(self):
        """Test angle values.
        """
        # Test that angle values are computed correctly.
        assert self.angles['BR'][0] == approx(self.solver.angles['BR'][0], abs=1.e-12)
        assert self.angles['BR'][1] == approx(self.solver.angles['BR'][1], abs=1.e-12)
        assert self.angles['CD']    == approx(self.solver.angles['CD'], abs=1.e-12)
        assert self.angles['TS']    == approx(self.solver.angles['TS'], abs=1.e-12)


    def test_ex3_morphology(self):
        """Test morphology string.
        """
        # Test that morphology is set correctly.
        assert self.morphology == self.solver.morphology


    def test_ex3_state0(self):
        """Test that a random angle value less than angles['BR'][0] takes
           bottom-state values.
        """
        # Test that any angle < angles['BR'][0] returns bottom-state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [-1, self.angles['BR'][0]])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        e = interp(val, arctan(y/x), self.soln['specific_internal_energy'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        q = interp(val, arctan(y/x), self.soln['speed'])
        assert p == approx(self.bottom_state[0], abs=1.e-12)
        assert r == approx(self.bottom_state[1], abs=1.e-12)
        assert M == approx(self.bottom_state[2], abs=1.e-12)
        assert e == approx(self.eB, abs=1.e-12)
        assert u == approx(self.uB, abs=1.e-12)
        assert v == approx(self.vB, abs=1.e-12)
        assert q == approx(self.qB, abs=1.e-12)
    

    def test_ex3_state2(self):
        """Test that a random angle value between angles['BR'][1] and
           angles['CD'] takes bottom-star state values.
        """
        # Test that any angles['BR'][1] < angle < angles['CD'] returns
        # bottom-star state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['BR'][1], self.angles['CD']])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        assert p == approx(self.bottom_star_vals[0], abs=1.e-12)
        assert r == approx(self.bottom_star_vals[1], abs=1.e-12)
        assert M == approx(self.bottom_star_vals[2], abs=1.e-12)
        assert u == approx(self.bottom_star_vals[3], abs=1.e-12)
        assert v == approx(self.bottom_star_vals[4], abs=1.e-12)
    

    def test_ex3_state3(self):
        """Test that any angles['CD'] < angles < angles['TS'] returns                   top-star state values.
        """
        # Test that any angles['CD'] < angles < angles['TS'] returns
        # top-star state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['CD'], self.angles['TS']])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        assert p == approx(self.top_star_vals[0], abs=1.e-12)
        assert r == approx(self.top_star_vals[1], abs=1.e-12)
        assert M == approx(self.top_star_vals[2], abs=1.e-12)
        assert u == approx(self.top_star_vals[3], abs=1.e-12)
        assert v == approx(self.top_star_vals[4], abs=1.e-12)
    

    def test_ex3_state4(self):
        """Test that a random point with angle value less than angles['BR'][0]
           takes bottom-state values.
        """
        # Test that any angle < angles['BR'][0] returns bottom-state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['TS'], 1])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        e = interp(val, arctan(y/x), self.soln['specific_internal_energy'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        q = interp(val, arctan(y/x), self.soln['speed'])
        assert p == approx(self.top_state[0], abs=1.e-12)
        assert r == approx(self.top_state[1], abs=1.e-12)
        assert M == approx(self.top_state[2], abs=1.e-12)
        assert e == approx(self.eT, abs=1.e-12)
        assert u == approx(self.uT, abs=1.e-12)
        assert v == approx(self.vT, abs=1.e-12)
        assert q == approx(self.qT, abs=1.e-12)
    

class Test_example3_reversed():
    """Tests ideal-gas EOS (IGEOS) solutions produced on the third example of these steadystsate 2-section 2D Riemann problem.
    """

    # Riemann Problem 3: states = [p, r, M, theta, g]
    top_state = [1., 1., 6., -15., 1.4]
    bottom_state = [0.1, 0.2, 4.0, 0., 1.4]
    N=101
    xlim=(0., 0.02, 0.32)
    ylim=(0., 0.2, 0.21)
    ys = linspace(ylim[0], ylim[2], int(1e4)) - ylim[1]
    xs = 0. * ys + xlim[2] - 2. * xlim[1]
    r_state = [[x, y] for x, y in zip(xs, ys)]
    t = 1.

    pB, rB, MB, thetaB_deg, gB = bottom_state
    pT, rT, MT, thetaT_deg, gT = top_state

    thetaB_rad, thetaT_rad = array([thetaB_deg, thetaT_deg]) / 180. * pi
    
    eB = pB / rB / (gB - 1.)
    cB = sqrt(gB * pB / rB)
    uB, vB = MB * cB * array([cos(thetaB_rad), sin(thetaB_rad)])
    qB = sqrt(uB**2 + vB**2)
    eT = pT / rT / (gT - 1.)
    cT = sqrt(gT * pT / rT)
    uT, vT = MT * cT * array([cos(thetaT_rad), sin(thetaT_rad)])
    qT = sqrt(uT**2 + vT**2)

    prob = SetupRiemannProblem(bottom_state=bottom_state, top_state = top_state)
    solver = IGEOS_Solver(bottom_state=bottom_state, top_state=top_state)
    soln = solver._run(r_state, t)

    # Test that star state values are computed correctly: [p, r, M, u, v].
    bottom_star_vals = [0.516819650123839,  0.5732201972606946,
                        2.5779575986238696, 2.7238966426853324,
                        -0.9844462006785117]
    top_star_vals = [0.516819650123839, 0.6240827168186239,
                     6.671530841696758, 6.755850287822839,
                     -2.441638586419862]


    deflection_angle_solution = -0.3468041303387353
    pressure_solution = 0.516819650123839

    # Test that angles and morphology are computed corrected.
    angles = {'BS': -0.5640333454902988,
              'CD': -0.3468041303387353,
              'TR': array([-0.17935605111904598, -0.09435130857946009])}
    morphology = 'S-C-R'

    def test_ex3rev_star_states(self):
        """Test star-state values.
        """
        # Test that star-state values are computed correctly.
        assert self.bottom_star_vals[0] == approx(self.solver.bottom_star_vals[0], abs=1.e-12)
        assert self.bottom_star_vals[1] == approx(self.solver.bottom_star_vals[1], abs=1.e-12)
        assert self.bottom_star_vals[2] == approx(self.solver.bottom_star_vals[2], abs=1.e-12)
        assert self.bottom_star_vals[3] == approx(self.solver.bottom_star_vals[3], abs=1.e-12)
        assert self.bottom_star_vals[4] == approx(self.solver.bottom_star_vals[4], abs=1.e-12)
        assert self.top_star_vals[0] == approx(self.solver.top_star_vals[0], abs=1.e-12)
        assert self.top_star_vals[1] == approx(self.solver.top_star_vals[1], abs=1.e-12)
        assert self.top_star_vals[2] == approx(self.solver.top_star_vals[2], abs=1.e-12)
        assert self.top_star_vals[3] == approx(self.solver.top_star_vals[3], abs=1.e-12)
        assert self.top_star_vals[4] == approx(self.solver.top_star_vals[4], abs=1.e-12)


    def test_ex3rev_velocity_angles(self):
        """Test that velocity angles are set correctly.
        """
        assert self.thetaB_rad == approx(self.solver.thetaB_rad, abs=1.e-12)
        assert self.thetaT_rad == approx(self.solver.thetaT_rad, abs=1.e-12)


    def test_ex3rev_deflection_angle(self):
        """Test deflection angle values.
        """
        # Test that star-state values are consistent with deflection angle.
        assert self.deflection_angle_solution == approx(self.solver.deflection_angle_solution, abs=1.e-12)


    def test_ex3rev_pressure_solution(self):
        """Test pressure solution value.
        """
        # Test that the pressure solution is computed correctly.
        assert self.pressure_solution == approx(self.solver.pressure_solution, abs=1.e-12)


    def test_ex3rev_angles(self):
        """Test angle values.
        """
        # Test that angle values are computed correctly.
        assert self.angles['BS']    == approx(self.solver.angles['BS'], abs=1.e-12)
        assert self.angles['CD']    == approx(self.solver.angles['CD'], abs=1.e-12)
        assert self.angles['TR'][0] == approx(self.solver.angles['TR'][0], abs=1.e-12)
        assert self.angles['TR'][1] == approx(self.solver.angles['TR'][1], abs=1.e-12)


    def test_ex3rev_morphology(self):
        """Test morphology string.
        """
        # Test that morphology is set correctly.
        assert self.morphology == self.solver.morphology


    def test_ex3rev_state0(self):
        """Test that a random point with angle value less than angles['BS'] takes bottom-state values.
        """
        # Test that any angle < angles['BS'] returns bottom-state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [-pi/2., self.angles['BS']])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        e = interp(val, arctan(y/x), self.soln['specific_internal_energy'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        q = interp(val, arctan(y/x), self.soln['speed'])
        assert p == approx(self.bottom_state[0], abs=1.e-12)
        assert r == approx(self.bottom_state[1], abs=1.e-12)
        assert M == approx(self.bottom_state[2], abs=1.e-12)
        assert e == approx(self.eB, abs=1.e-12)
        assert u == approx(self.uB, abs=1.e-12)
        assert v == approx(self.vB, abs=1.e-12)
        assert q == approx(self.qB, abs=1.e-12)
    
    def test_ex3rev_state1(self):
        """Test that any angles['BS'] < angles < angles['CD'] returns bottom-star state values.
        """
        # Test that any angles['BS'] < angles < angles['CD'] returns bottom-star
        # state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['BS'], self.angles['CD']])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        assert p == approx(self.bottom_star_vals[0], abs=1.e-12)
        assert r == approx(self.bottom_star_vals[1], abs=1.e-12)
        assert M == approx(self.bottom_star_vals[2], abs=1.e-12)
        assert u == approx(self.bottom_star_vals[3], abs=1.e-12)
        assert v == approx(self.bottom_star_vals[4], abs=1.e-12)
    

    def test_ex3rev_state2(self):
        """Test that a random angle value between angles['CD'] and angles['TR'][0] takes top-star state values.
        """
        # Test that any angles['CD'] < angle < angles['TR'][0] returns top-star
        # state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['CD'], self.angles['TR'][0]])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        assert p == approx(self.top_star_vals[0], abs=1.e-12)
        assert r == approx(self.top_star_vals[1], abs=1.e-12)
        assert M == approx(self.top_star_vals[2], abs=1.e-12)
        assert u == approx(self.top_star_vals[3], abs=1.e-12)
        assert v == approx(self.top_star_vals[4], abs=1.e-12)
    

    def test_ex3rev_state4(self):
        """Test that a random angle value greater than angles['TR'][1] takes top-state values.
        """
        # Test that any angle > angles['TR'][1] returns top-state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['TR'][1], pi/2.])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        e = interp(val, arctan(y/x), self.soln['specific_internal_energy'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        q = interp(val, arctan(y/x), self.soln['speed'])
        assert p == approx(self.top_state[0], abs=1.e-12)
        assert r == approx(self.top_state[1], abs=1.e-12)
        assert M == approx(self.top_state[2], abs=1.e-12)
        assert e == approx(self.eT, abs=1.e-12)
        assert u == approx(self.uT, abs=1.e-12)
        assert v == approx(self.vT, abs=1.e-12)
        assert q == approx(self.qT, abs=1.e-12)
    

class Test_example4():
    """Tests ideal-gas EOS (IGEOS) solutions produced on the fourth example of these steadystsate 2-section 2D Riemann problem.
    """

    # Riemann Problem 4: states = [p, r, M, theta, g]
    top_state = [0.01, 0.05, 3.5, 0., 1.4]
    bottom_state = [1., 1., 10., 23., 1.4]
    N=101
    xlim=(0., 0.02, 0.17)
    ylim=(0., 0.01, 0.21)
    ys = linspace(ylim[0], ylim[2], int(1e4)) - ylim[1]
    xs = 0. * ys + xlim[2] - 2. * xlim[1]
    r_state = [[x, y] for x, y in zip(xs, ys)]
    t = 1.

    pB, rB, MB, thetaB_deg, gB = bottom_state
    pT, rT, MT, thetaT_deg, gT = top_state

    thetaB_rad, thetaT_rad = array([thetaB_deg, thetaT_deg]) / 180. * pi
    
    eB = pB / rB / (gB - 1.)
    cB = sqrt(gB * pB / rB)
    uB, vB = MB * cB * array([cos(thetaB_rad), sin(thetaB_rad)])
    qB = sqrt(uB**2 + vB**2)
    eT = pT / rT / (gT - 1.)
    cT = sqrt(gT * pT / rT)
    uT, vT = MT * cT * array([cos(thetaT_rad), sin(thetaT_rad)])
    qT = sqrt(uT**2 + vT**2)

    prob = SetupRiemannProblem(bottom_state=bottom_state, top_state = top_state)
    solver = IGEOS_Solver(bottom_state=bottom_state, top_state=top_state)
    soln = solver._run(r_state, t)

    # Test that star state values are computed correctly: [p, r, M, u, v].
    bottom_star_vals = [ 0.08807958854071007, 0.1763353904126041,
                        14.32515495753201,   10.074633407093081,
                         6.481146448737798]
    top_star_vals = [0.08807958854071007, 0.18182030911580432,
                     1.4566199098321666,  1.0088456167290194,
                     0.6490038815292812]


    deflection_angle_solution = 0.5716602362289064
    pressure_solution = 0.08807958854071007

    # Test that angles and morphology are computed corrected.
    angles = {'BR': array([0.30125830679713594, 0.4714928150673466]),
              'CD': 0.5716602362289064,
              'TS': 0.9148000109379752}
    morphology = 'R-C-S'

    def test_ex4_star_states(self):
        """Test star-state values.
        """
        # Test that star-state values are computed correctly.
        assert self.bottom_star_vals[0] == approx(self.solver.bottom_star_vals[0], abs=1.e-12)
        assert self.bottom_star_vals[1] == approx(self.solver.bottom_star_vals[1], abs=1.e-12)
        assert self.bottom_star_vals[2] == approx(self.solver.bottom_star_vals[2], abs=1.e-12)
        assert self.bottom_star_vals[3] == approx(self.solver.bottom_star_vals[3], abs=1.e-12)
        assert self.bottom_star_vals[4] == approx(self.solver.bottom_star_vals[4], abs=1.e-12)
        assert self.top_star_vals[0] == approx(self.solver.top_star_vals[0], abs=1.e-12)
        assert self.top_star_vals[1] == approx(self.solver.top_star_vals[1], abs=1.e-12)
        assert self.top_star_vals[2] == approx(self.solver.top_star_vals[2], abs=1.e-12)
        assert self.top_star_vals[3] == approx(self.solver.top_star_vals[3], abs=1.e-12)
        assert self.top_star_vals[4] == approx(self.solver.top_star_vals[4], abs=1.e-12)


    def test_ex4_velocity_angles(self):
        """Test that velocity angles are set correctly.
        """
        assert self.thetaB_rad == approx(self.solver.thetaB_rad, abs=1.e-12)
        assert self.thetaT_rad == approx(self.solver.thetaT_rad, abs=1.e-12)


    def test_ex4_deflection_angle(self):
        """Test deflection angle values.
        """
        # Test that star-state values are consistent with deflection angle.
        assert self.deflection_angle_solution == approx(self.solver.deflection_angle_solution, abs=1.e-12)


    def test_ex4_pressure_solution(self):
        """Test pressure solution value.
        """
        # Test that the pressure solution is computed correctly.
        assert self.pressure_solution == approx(self.solver.pressure_solution, abs=1.e-12)


    def test_ex4_angles(self):
        """Test angle values.
        """
        # Test that angle values are computed correctly.
        assert self.angles['BR'][0] == approx(self.solver.angles['BR'][0], abs=1.e-12)
        assert self.angles['BR'][1] == approx(self.solver.angles['BR'][1], abs=1.e-12)
        assert self.angles['CD']    == approx(self.solver.angles['CD'], abs=1.e-12)
        assert self.angles['TS']    == approx(self.solver.angles['TS'], abs=1.e-12)


    def test_ex4_morphology(self):
        """Test morphology string.
        """
        # Test that morphology is set correctly.
        assert self.morphology == self.solver.morphology


    def test_ex4_state0(self):
        """Test that a random angle value less than angles['BR'][0] takes bottom-state values.
        """
        # Test that any angle < angles['BR'][0] returns bottom-state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [-pi/2., self.angles['BR'][0]])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        e = interp(val, arctan(y/x), self.soln['specific_internal_energy'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        q = interp(val, arctan(y/x), self.soln['speed'])
        assert p == approx(self.bottom_state[0], abs=1.e-12)
        assert r == approx(self.bottom_state[1], abs=1.e-12)
        assert M == approx(self.bottom_state[2], abs=1.e-12)
        assert e == approx(self.eB, abs=1.e-12)
        assert u == approx(self.uB, abs=1.e-12)
        assert v == approx(self.vB, abs=1.e-12)
        assert q == approx(self.qB, abs=1.e-12)
    

    def test_ex4_state2(self):
        """Test that a random angle value between angles['BR'][1] and angles['CD'] takes bottom-star state values.
        """
        # Test that any angles['BR'][1] < angle < angles['CD'] returns
        # bottom-star state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['BR'][1], self.angles['CD']])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        assert p == approx(self.bottom_star_vals[0], abs=1.e-12)
        assert r == approx(self.bottom_star_vals[1], abs=1.e-12)
        assert M == approx(self.bottom_star_vals[2], abs=1.e-12)
        assert u == approx(self.bottom_star_vals[3], abs=1.e-12)
        assert v == approx(self.bottom_star_vals[4], abs=1.e-12)
    

    def test_ex4_state3(self):
        """Test that any angles['CD'] < angles < angles['TS'] returns top-star state values.
        """
        # Test that any angles['CD'] < angles < angles['TS'] returns
        # top-star state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['CD'], self.angles['TS']])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        assert p == approx(self.top_star_vals[0], abs=1.e-12)
        assert r == approx(self.top_star_vals[1], abs=1.e-12)
        assert M == approx(self.top_star_vals[2], abs=1.e-12)
        assert u == approx(self.top_star_vals[3], abs=1.e-12)
        assert v == approx(self.top_star_vals[4], abs=1.e-12)
    

    def test_ex4_state4(self):
        """Test that a random point with angle value less than angles['TS'] takes top-state values.
        """
        # Test that any angle > angles['TS'] returns top-state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['TS'], pi/2.])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        e = interp(val, arctan(y/x), self.soln['specific_internal_energy'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        q = interp(val, arctan(y/x), self.soln['speed'])
        assert p == approx(self.top_state[0], abs=1.e-12)
        assert r == approx(self.top_state[1], abs=1.e-12)
        assert M == approx(self.top_state[2], abs=1.e-12)
        assert e == approx(self.eT, abs=1.e-12)
        assert u == approx(self.uT, abs=1.e-12)
        assert v == approx(self.vT, abs=1.e-12)
        assert q == approx(self.qT, abs=1.e-12)
    


class Test_example4_reversed():
    """Tests ideal-gas EOS (IGEOS) solutions produced on the fourth example of these steadystsate 2-section 2D Riemann problem.
    """

    # Riemann Problem 4: states = [p, r, M, theta, g]
    top_state = [1., 1., 10., -23., 1.4]
    bottom_state = [0.01, 0.05, 3.5, 0., 1.4]
    N=101
    xlim=(0., 0.02, 0.17)
    ylim=(0., 0.2, 0.21)
    ys = linspace(ylim[0], ylim[2], int(1e4)) - ylim[1]
    xs = 0. * ys + xlim[2] - 2. * xlim[1]
    r_state = [[x, y] for x, y in zip(xs, ys)]
    t = 1.

    pB, rB, MB, thetaB_deg, gB = bottom_state
    pT, rT, MT, thetaT_deg, gT = top_state

    thetaB_rad, thetaT_rad = array([thetaB_deg, thetaT_deg]) / 180. * pi
    
    eB = pB / rB / (gB - 1.)
    cB = sqrt(gB * pB / rB)
    uB, vB = MB * cB * array([cos(thetaB_rad), sin(thetaB_rad)])
    qB = sqrt(uB**2 + vB**2)
    eT = pT / rT / (gT - 1.)
    cT = sqrt(gT * pT / rT)
    uT, vT = MT * cT * array([cos(thetaT_rad), sin(thetaT_rad)])
    qT = sqrt(uT**2 + vT**2)

    prob = SetupRiemannProblem(bottom_state=bottom_state, top_state = top_state)
    solver = IGEOS_Solver(bottom_state=bottom_state, top_state=top_state)
    soln = solver._run(r_state, t)

    # Test that star state values are computed correctly: [p, r, M, u, v].
    bottom_star_vals = [0.0880795885407101, 0.18182030911580435,
                        1.456619909832166,  1.0088456167290185,
                        -0.6490038815292816]
    top_star_vals = [ 0.0880795885407101, 0.17633539041260413,
                     14.32515495753201, 10.074633407093081,
                     -6.481146448737807]


    deflection_angle_solution = -0.5716602362289069
    pressure_solution = 0.0880795885407101

    # Test that angles and morphology are computed corrected.
    angles = {'BS': -0.9148000109379761,
              'CD': -0.5716602362289069,
              'TR': array([-0.47149281506734714, -0.30125830679713594])}
    morphology = 'S-C-R'

    def test_ex4rev_star_states(self):
        """Test star-state values.
        """
        # Test that star-state values are computed correctly.
        assert self.bottom_star_vals[0] == approx(self.solver.bottom_star_vals[0], abs=1.e-12)
        assert self.bottom_star_vals[1] == approx(self.solver.bottom_star_vals[1], abs=1.e-12)
        assert self.bottom_star_vals[2] == approx(self.solver.bottom_star_vals[2], abs=1.e-12)
        assert self.bottom_star_vals[3] == approx(self.solver.bottom_star_vals[3], abs=1.e-12)
        assert self.bottom_star_vals[4] == approx(self.solver.bottom_star_vals[4], abs=1.e-12)
        assert self.top_star_vals[0] == approx(self.solver.top_star_vals[0], abs=1.e-12)
        assert self.top_star_vals[1] == approx(self.solver.top_star_vals[1], abs=1.e-12)
        assert self.top_star_vals[2] == approx(self.solver.top_star_vals[2], abs=1.e-12)
        assert self.top_star_vals[3] == approx(self.solver.top_star_vals[3], abs=1.e-12)
        assert self.top_star_vals[4] == approx(self.solver.top_star_vals[4], abs=1.e-12)


    def test_ex4rev_velocity_angles(self):
        """Test that velocity angles are set correctly.
        """
        assert self.thetaB_rad == approx(self.solver.thetaB_rad, abs=1.e-12)
        assert self.thetaT_rad == approx(self.solver.thetaT_rad, abs=1.e-12)


    def test_ex4rev_deflection_angle(self):
        """Test deflection angle values.
        """
        # Test that star-state values are consistent with deflection angle.
        assert self.deflection_angle_solution == approx(self.solver.deflection_angle_solution, abs=1.e-12)


    def test_ex4rev_pressure_solution(self):
        """Test pressure solution value.
        """
        # Test that the pressure solution is computed correctly.
        assert self.pressure_solution == approx(self.solver.pressure_solution, abs=1.e-12)


    def test_ex4rev_angles(self):
        """Test angle values.
        """
        # Test that angle values are computed correctly.
        assert self.angles['BS']    == approx(self.solver.angles['BS'], abs=1.e-12)
        assert self.angles['CD']    == approx(self.solver.angles['CD'], abs=1.e-12)
        assert self.angles['TR'][0] == approx(self.solver.angles['TR'][0], abs=1.e-12)
        assert self.angles['TR'][1] == approx(self.solver.angles['TR'][1], abs=1.e-12)


    def test_ex4rev_morphology(self):
        """Test morphology string.
        """
        # Test that morphology is set correctly.
        assert self.morphology == self.solver.morphology


    def test_ex4rev_state0(self):
        """Test that a random point with angle value less than angles['BS'] takes bottom-state values.
        """
        # Test that any angle > angles['TS'] returns top-state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [-pi/2., self.angles['BS']])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        e = interp(val, arctan(y/x), self.soln['specific_internal_energy'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        q = interp(val, arctan(y/x), self.soln['speed'])
        assert p == approx(self.bottom_state[0], abs=1.e-12)
        assert r == approx(self.bottom_state[1], abs=1.e-12)
        assert M == approx(self.bottom_state[2], abs=1.e-12)
        assert e == approx(self.eB, abs=1.e-12)
        assert u == approx(self.uB, abs=1.e-12)
        assert v == approx(self.vB, abs=1.e-12)
        assert q == approx(self.qB, abs=1.e-12)
    

    def test_ex4rev_state1(self):
        """Test that any angles['BS'] < angles < angles['CD'] returns bottom-star state values.
        """
        # Test that any angles['BS'] < angles < angles['CD'] returns bottom-star
        # state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['BS'], self.angles['CD']])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        assert p == approx(self.bottom_star_vals[0], abs=1.e-12)
        assert r == approx(self.bottom_star_vals[1], abs=1.e-12)
        assert M == approx(self.bottom_star_vals[2], abs=1.e-12)
        assert u == approx(self.bottom_star_vals[3], abs=1.e-12)
        assert v == approx(self.bottom_star_vals[4], abs=1.e-12)
    

    def test_ex4rev_state2(self):
        """Test that a random angle value between angles['CD'] and angles['TR'][0] takes top-star state values.
        """
        # Test that any angles['CD'] < angle < angles['TR'][0] returns top-star
        # state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['CD'], self.angles['TR'][0]])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        assert p == approx(self.top_star_vals[0], abs=1.e-12)
        assert r == approx(self.top_star_vals[1], abs=1.e-12)
        assert M == approx(self.top_star_vals[2], abs=1.e-12)
        assert u == approx(self.top_star_vals[3], abs=1.e-12)
        assert v == approx(self.top_star_vals[4], abs=1.e-12)
    

    def test_ex4rev_state4(self):
        """Test that a random angle value greater than angles['TR'][1] takes top-state values.
        """
        # Test that any angle > angles['TR'][1] returns top-state values.
        x = self.soln['x_position']
        y = self.soln['y_position']
        val = interp(rand(), [0,1], [self.angles['TR'][1], pi/2.])
        p = interp(val, arctan(y/x), self.soln['pressure'])
        r = interp(val, arctan(y/x), self.soln['density'])
        e = interp(val, arctan(y/x), self.soln['specific_internal_energy'])
        M = interp(val, arctan(y/x), self.soln['Mach'])
        u = interp(val, arctan(y/x), self.soln['x_velocity'])
        v = interp(val, arctan(y/x), self.soln['y_velocity'])
        q = interp(val, arctan(y/x), self.soln['speed'])
        assert p == approx(self.top_state[0], abs=1.e-12)
        assert r == approx(self.top_state[1], abs=1.e-12)
        assert M == approx(self.top_state[2], abs=1.e-12)
        assert e == approx(self.eT, abs=1.e-12)
        assert u == approx(self.uT, abs=1.e-12)
        assert v == approx(self.vT, abs=1.e-12)
        assert q == approx(self.qT, abs=1.e-12)
