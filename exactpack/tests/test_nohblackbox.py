"""Unit tests for the Black Box EoS Noh solver.

Unit tests include tests for the Newton Solver, the residual function, the coupling between the two, and 
"""

import pytest
import numpy as np
from exactpack.solvers.nohblackboxeos.solution_tools.residual_functions import noh_residual, simplified_noh_residual, simplified_pressure_noh_residual, pressure_noh_residual, ZeroDensityError, ZeroDeterminantError
from exactpack.solvers.nohblackboxeos.solution_tools.newton_solvers import newton_solver, IterationError
from exactpack.solvers.nohblackboxeos import NohBlackBoxEos, PlanarNohBlackBox, CylindricalNohBlackBox, SphericalNohBlackBox

class ideal_gas_eos: 
    def __init__(self, gamma = 5/3): 
        self.gamma = gamma
        assert(gamma != 1), "Gamma cannot be equal to 1"

    def e(self, rho, P): 
        if(rho == 0.0):
            raise ValueError("Error: rho = 0. Please do not break math and divide by zero.")
        return P/(rho*(self.gamma -1.))     
    
    def de_dP(self, rho, P): 
        if(rho == 0.0):
            raise ValueError("Error: rho = 0. Please do not break math and divide by zero.")
        return 1./(rho*(self.gamma-1))
    
    def de_drho(self, rho,P): 
        if(rho == 0.0):
            raise ValueError("Error: rho = 0. Please do not break math and divide by zero.")
        return -P/(rho**2*(self.gamma-1))
    
    def P(self,rho,e): 
        return rho*e*(self.gamma-1)
    
    def dP_drho(self, rho,e): 
        return e*(self.gamma -1)
    
    def dP_de(self, rho,e): 
        return rho*(self.gamma-1)

class TestResidualFunction:
    r"""Tests for :class: `exactpack.solver.nohblackboxeos.solution_tools.residual_functions.noh_residual'. 

    The test consists of several parts. 
     
    First, ensuring that the class accepts only admissible initial conditions (positive density, pressure; 
    negative velocity, and symmetry (:math: 'm'), :math:'m \in {0,1,2}'). This includes enforcing restrictions on the initial conditions when the 
    symmtery is not 0. 

    Second, it tests array sizes--the function should only accept arrays of size 3 and that the determinant only computes for 3x3 matrices. 

    Third, it tests error handling for zero density and zero determinant. 

    Finally, it checks functions at a specific to ensure functions are producing correct output. 
    """

    @classmethod
    def setup_class(cls):
        cls.wrong_eos = None
        cls.eos = ideal_gas_eos()
        cls.wrong_velocity_ic = {'density': 1, 'velocity': 1, 'pressure': 0, 'symmetry': 0}
        cls.wrong_density_ic = {'density': -1, 'velocity': -1, 'pressure': 0, 'symmetry': 0}
        cls.wrong_pressure_ic = {'density': 1, 'velocity': -1, 'pressure': -1, 'symmetry': 0}
        cls.wrong_symmetry_ic = {'density': 1, 'velocity': -1, 'pressure': 0, 'symmetry': 10}
        cls.value_test_ic = {'density': 1, 'velocity': -1, 'pressure': 1, 'symmetry': 0}
        cls.wrong_pressure_symmetry_ic = {'density': 1, 'velocity': -1, 'pressure': 1, 'symmetry': 1}
        cls.good_ic = {'density': 1, 'velocity': -1, 'pressure': 0, 'symmetry': 0}
        cls.wrong_array = [10,10,10,10]
        cls.zero_density_array = [0,1,1]
        cls.wrong_matrix = np.zeros((3,2))
        cls.zero_determinant_array = [1, 0, 1]

    def test_wrong_velocity(self):
        with pytest.raises(ValueError): 
            bad_velocity_function = noh_residual(self.wrong_velocity_ic, self.eos)
        """Noh Problem--Residual Function: Test for valid value of velocity"""
    
    def test_wrong_density(self):
        with pytest.raises(ValueError):
            bad_density_function = noh_residual(self.wrong_density_ic, self.eos)
        """Noh Problem--Residual Function: Test for valid value of density"""

    def test_wrong_pressure(self):
        with pytest.raises(ValueError): 
            bad_pressure_function = noh_residual(self.wrong_pressure_ic, self.eos)
        """Noh Problem--Residual Function: Test for valid value of pressure"""
    
    def test_wrong_symmetry(self):
        with pytest.raises(ValueError): 
            bad_symmetry_funciton = noh_residual(self.wrong_symmetry_ic, self.eos)
        """Noh Problem--Residual Function: Test for valid value of symmetry"""
    
    def test_wrong_eos(self):
        with pytest.raises(ValueError): 
            bad_eos_function = noh_residual(self.good_ic, self.wrong_eos)
        """Noh Problem--Residual Function: Test for valid eos object"""
    
    def test_pressure_symmetry_error(self):
        with pytest.raises(ValueError):
            bad_pressure_symmetry_function = noh_residual(self.wrong_pressure_symmetry_ic, self.eos)
        """Noh Problem--Residual Function: Test for admissability of symmetry and initial pressure"""

    def test_wrong_array_size_F(self): 
        function = noh_residual(self.good_ic, self.eos)
        with pytest.raises(ValueError):
            function.F(self.wrong_array)
        """Noh Problem--Residual Function: Test for valid array size for F"""
    
    def test_zero_density_F(self): 
        function = noh_residual(self.good_ic, self.eos)
        with pytest.raises(ZeroDensityError):
            function.F_prime(self.zero_density_array)
        """Noh Problem--Residual Function: Test for zero density for F"""
    
    def test_wrong_array_size_F_prime(self): 
        function = noh_residual(self.good_ic, self.eos)
        with pytest.raises(ValueError):
            function.F_prime_inv(self.wrong_array)
        """Noh Problem--Residual Function: Test for valid array size for F_prime"""

    def test_zero_density_F_prime(self): 
        function = noh_residual(self.good_ic, self.eos)
        with pytest.raises(ZeroDensityError):
            function.F_prime(self.zero_density_array)
        """Noh Problem--Residual Function: Test for zero density for F_prime"""
    
    def test_matrix_size(self):
        function = noh_residual(self.good_ic, self.eos)
        with pytest.raises(np.linalg.LinAlgError):
            function.determinant(self.wrong_matrix)
        """Noh Problem--Residual Function: Test for value size of matrix"""
    
    def test_wrong_array_size_F_prime_inv(self):
        function = noh_residual(self.good_ic, self.eos)
        with pytest.raises(ValueError):
            function.F_prime_inv(self.wrong_array)
        """Noh Problem--Residual Function: Test for valid array size for F_prime_inv"""
    
    def test_zero_density_F_prime_inv(self):
        function = noh_residual(self.good_ic, self.eos)
        with pytest.raises(ZeroDensityError):
            function.F_prime_inv(self.zero_density_array)
        """Noh Problem--Residual Function: Test for zero density for F_prime_inv"""
    
    def test_zero_determinant_in_F_prime_inv(self):
        function = noh_residual(self.good_ic, self.eos)
        with pytest.raises(ZeroDeterminantError): 
            function.F_prime_inv(self.zero_determinant_array)
        """Noh Problem--Residual Function: Test for zero determinant in F_prime_inv"""

    def test_F_value(self): 
        function = noh_residual(self.value_test_ic, self.eos)
        np.testing.assert_equal(function.F([1,1,1]), [-1., -1., -1.5])
        """Noh Problem--Residual Function: residual function value"""

    def test_F_prime_value(self): 
        function = noh_residual(self.value_test_ic, self.eos)
        np.testing.assert_almost_equal(function.F_prime([1,1,1]), [[1,0,1], [-1,1,-1], [-0.5, 1.5, 1]], decimal=15)
        """Noh Problem--Residual Function: Jacobian value"""

    def test_det_value(self): 
        function = noh_residual(self.value_test_ic, self.eos)
        np.testing.assert_almost_equal(function.determinant(function.F_prime([1,1,1])), 1.5 , decimal=14)
        """Noh Problem--Residual Function: determinant value"""

    def test_F_prime_inv_value(self): 
        function = noh_residual(self.value_test_ic, self.eos)
        np.testing.assert_almost_equal(function.F_prime_inv([1,1,1]), [[5./3., 1, -2./3.], [1,1,0], [-2./3., -1, 2./3.]], decimal=12)
        """Noh Problem--Residual Function: Jacobian Inverse value"""

class TestSimplifiedResidualFunction:
    r"""Tests for :class: `exactpack.solver.nohblackboxeos.solution_tools.residual_functions.simplified_noh_residual'. 

    The test consists of several parts. 
     
    First, ensuring that the class accepts only admissible initial conditions (positive density, pressure; 
    negative velocity, and symmetry (:math: 'm'), :math:'m \in {0,1,2}'). This includes enforcing that P_0 = symmetry = 0. (This is assumption 
    for this particular residual.)

    Second, it tests array sizes--the function should only accept arrays of size 3 and that the determinant only computes for 3x3 matrices. 

    Third, it tests error handling for zero density and zero determinant. 

    Finally, it checks functions at a specific to ensure functions are producing correct output. 
    """

    @classmethod
    def setup_class(cls):
        cls.wrong_eos = None
        cls.eos = ideal_gas_eos()
        cls.wrong_velocity_ic = {'density': 1, 'velocity': 1, 'pressure': 0, 'symmetry': 0}
        cls.wrong_density_ic = {'density': -1, 'velocity': -1, 'pressure': 0, 'symmetry': 0}
        cls.wrong_pressure_ic = {'density': 1, 'velocity': -1, 'pressure': 1, 'symmetry': 0}
        cls.wrong_symmetry_ic = {'density': 1, 'velocity': -1, 'pressure': 0, 'symmetry': 1}
        cls.value_test_ic = {'density': 1, 'velocity': -1, 'pressure': 0, 'symmetry': 0}
        cls.wrong_pressure_symmetry_ic = {'density': 1, 'velocity': -1, 'pressure': 1, 'symmetry': 1}
        cls.good_ic = {'density': 1, 'velocity': -1, 'pressure': 0, 'symmetry': 0}
        cls.wrong_array = [10,10,10]
        cls.zero_density_array = [0,1]
        cls.zero_determinant_array = [1, 0]

    def test_wrong_velocity(self):
        with pytest.raises(ValueError): 
            bad_velocity_function = simplified_noh_residual(self.wrong_velocity_ic, self.eos)
        """Noh Problem--Residual Function: Test for valid value of velocity"""
    
    def test_wrong_density(self):
        with pytest.raises(ValueError):
            bad_density_function = simplified_noh_residual(self.wrong_density_ic, self.eos)
        """Noh Problem--Residual Function: Test for valid value of density"""

    def test_wrong_pressure(self):
        with pytest.raises(ValueError): 
            bad_pressure_function = simplified_noh_residual(self.wrong_pressure_ic, self.eos)
        """Noh Problem--Residual Function: Test for valid value of pressure"""
    
    def test_wrong_symmetry(self):
        with pytest.raises(ValueError): 
            bad_symmetry_funciton = simplified_noh_residual(self.wrong_symmetry_ic, self.eos)
        """Noh Problem--Residual Function: Test for valid value of symmetry"""
    
    def test_wrong_eos(self):
        with pytest.raises(ValueError): 
            bad_eos_function = simplified_noh_residual(self.good_ic, self.wrong_eos)
        """Noh Problem--Residual Function: Test for valid eos object"""
    
    def test_pressure_symmetry_error(self):
        with pytest.raises(ValueError):
            bad_pressure_symmetry_function = simplified_noh_residual(self.wrong_pressure_symmetry_ic, self.eos)
        """Noh Problem--Residual Function: Test for admissability of symmetry and initial pressure"""

    def test_wrong_array_size_F(self): 
        function = simplified_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ValueError):
            function.F(self.wrong_array)
        """Noh Problem--Residual Function: Test for valid array size for F"""
    
    def test_zero_density_F(self): 
        function = simplified_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ZeroDensityError):
            function.F_prime(self.zero_density_array)
        """Noh Problem--Residual Function: Test for zero density for F"""
    
    def test_wrong_array_size_F_prime(self): 
        function = simplified_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ValueError):
            function.F_prime_inv(self.wrong_array)
        """Noh Problem--Residual Function: Test for valid array size for F_prime"""

    def test_zero_density_F_prime(self): 
        function = simplified_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ZeroDensityError):
            function.F_prime(self.zero_density_array)
        """Noh Problem--Residual Function: Test for zero density for F_prime"""

    def test_wrong_array_size_det(self): 
        function = simplified_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ValueError):
            function.determinant(self.wrong_array)
        """Noh Problem--Residual Function: Test for valid array size for F_prime"""

    def test_zero_density_det(self): 
        function = simplified_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ZeroDensityError):
            function.determinant(self.zero_density_array)
        """Noh Problem--Residual Function: Test for zero density for F_prime"""

    def test_wrong_array_size_F_prime_inv(self):
        function = simplified_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ValueError):
            function.F_prime_inv(self.wrong_array)
        """Noh Problem--Residual Function: Test for valid array size for F_prime_inv"""
    
    def test_zero_density_F_prime_inv(self):
        function = simplified_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ZeroDensityError):
            function.F_prime_inv(self.zero_density_array)
        """Noh Problem--Residual Function: Test for zero density for F_prime_inv"""
    
    def test_zero_determinant_in_F_prime_inv(self):
        function = simplified_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ZeroDeterminantError): 
            function.F_prime_inv(self.zero_determinant_array)
        """Noh Problem--Residual Function: Test for zero determinant in F_prime_inv"""

    def test_F_value(self): 
        function = simplified_noh_residual(self.value_test_ic, self.eos)
        np.testing.assert_almost_equal(function.F([1,1]), [-1., 1.], decimal= 15)
        """Noh Problem--Residual Function: residual function value"""

    def test_F_prime_value(self): 
        function = simplified_noh_residual(self.value_test_ic, self.eos)
        np.testing.assert_almost_equal(function.F_prime([1,1]), [[1.,0], [-1.5,1.5]], decimal=15)
        """Noh Problem--Residual Function: Jacobian value"""

    def test_det_value(self): 
        function = simplified_noh_residual(self.value_test_ic, self.eos)
        np.testing.assert_almost_equal(function.determinant([1,1]), 1.5, decimal=14)
        """Noh Problem--Residual Function: determinant value"""

    def test_F_prime_inv_value(self): 
        function = simplified_noh_residual(self.value_test_ic, self.eos)
        np.testing.assert_almost_equal(function.F_prime_inv([1,1]), [[1., 0], [1., 2./3.]], decimal=12)
        """Noh Problem--Residual Function: Jacobian Inverse value"""        

class test_newton_solver: 
    r"""Tests for :class: `exactpack.solver.nohblackboxeos.solution_tools.newton_solvers.newton_solver'. 

    The test consists of several parts.

    First, it tests error handling. This test includes ensuring that any residual function object has the required member, 
    that set tolerances, residual functions, and initial guesses are appropriate. Finally, it tests that an Iteration error is 
    thrown when the max number of iterations is exceeded. 

    Second, it tests the solver on a member residual function (self.function) and solves :math:'F(\mathbf{x}) = 0)'. 
    """

    @classmethod
    def setup_class(cls):
        cls.solver = newton_solver()

    class function: 
        def __init__(self): 
            self.result = np.zeros(2)
            self.DF = np.zeros((2,2))
            self.DF_inv = np.zeros((2,2))

        def F(self, x):
            self.result[0] = x[0]**2 + x[1]**2 - 1 
            self.result[1] = x[0] - x[1]
            return self.result
            
        def F_prime(self, x):
            self.DF[0,0] = 2*x[0]
            self.DF[0,1] = 2*x[1]
            self.DF[1,0] = 1
            self.DF[1,1] = -1
            return self.DF

        def determinant(self, x):
            det_result = -(2*x[0] + 2*x[1])
            return det_result
        
        def F_prime_inv(self, x):
            det = self.determinant(x)
            self.DF_inv[0,0] = -1
            self.DF_inv[0,1] = -2*x[1]
            self.DF_inv[1,0] = -1
            self.DF_inv[1,1] = 2*x[0]
            self.DF_inv = (1/det)*self.DF_inv
            return self.DF_inv    

    class fake_function: 
        def __init__(self): 
            return None
    
    def test_wrong_tolerance(self): 
        with pytest.raises(ValueError): 
            self.solver.set_new_tolerance(100)
        """Noh Problem--Newton Solver: Test for valid value of tolerance"""

    def test_no_function(self): 
        with pytest.raises(ValueError):
            self.solver.solve(verbose=False)            
        """Noh Problem--Newton Solver: Test for no function object"""
    
    def test_wrong_function(self): 
        with pytest.raises(ValueError): 
            self.solver.set_function(self.fake_function)
        """Noh Problem--Newton Solver: Test for valid function object"""
    
    def test_no_initial_guess(self):
        self.solver.set_function(self.function)
        with pytest.raises(ValueError):
            self.solver.solve(verbose=False)
        """Noh Problem--Newton Solver: Test for no initial guess"""
    
    def test_max_iteration_violation(self):
        self.solver.set_function(self.function)
        self.solver.set_new_tolerance(1.0e-08)
        self.solver.set_new_max_iteration(5)
        self.solver.set_new_initial_guess([100,100])
        with pytest.raises(IterationError):
            self.solver.solve(verbose=False)
        """Noh Problem--Newton Solver: Test for max iteration violation"""

    def test_solution(self): 
        self.solver.set_function(self.function)
        self.solver.set_new_tolerance(1.0e-08)
        self.solver.set_new_max_iteration(100)
        self.solver.set_new_initial_guess([1,1])
        solution_info = self.solver.solve(verbose = False)
        np.testing.assert_array_almost_equal(solution_info['solution'], [np.sqrt(2)/2, np.sqrt(2)/2],decimal=8)
        """Noh Problem--Newton Solver: Test solution of an analytic function."""

class test_noh_solver:
    r"""Tests for coupling between :class: `exactpack.solver.nohblackboxeos.solution_tools.newton_solvers.newton_solver' and
    :class: 'exactpack.solver.nohblackboxeos.solution_tools.residual_functions.noh_residual'. 

    The tests ensures that the coupling between these two functions solves the jump condition for the Noh Problem--Residual Function for an easy 
    case. Namely, it uses a standard set of initial conditions, the ideal gas equation of state (encoded as a member class), and in
    planar geometry. 

    """
    @classmethod
    def setup_class(cls):
        cls.initial_conditions = {'density': 1, 'velocity': -1, 'pressure': 0, 'symmetry': 0}       
        cls.solver = newton_solver()
        cls.eos = ideal_gas_eos()
        cls.initial_guess = [5, 3, 0.5]
        cls.noh_residual_function = noh_residual(cls.initial_conditions, cls.eos)
        cls.solver.set_new_initial_guess(cls.initial_guess)

    def test_solution(self): 
        result = self.solver.solve(verbose=False)
        np.testing.assert_almost_equal([result['density'], result['pressure'], result['shock_speed']], [4.0, 4./3., 1./3.], decimal=self.solver.tolerance)
        """Noh Problem--Noh Solver: Test for known solution values."""


"""Unit tests for the NohBlackBox solver.
"""

class TestNohBlackBox():
    r"""Tests for :class:`exactpack.solvers.noh.noh1.Noh`.

    The tests consist of comparing the values at two points, one in
    front of the shock (:math:`r=0.3`) and one behind the shock
    (:math:`r=0.1`), to the analytic solutions at a fixed time
    (:math:`t=0.6`) and :math:`\gamma=5/3`.
    """

    @classmethod
    def setup_method(self):
        self.ideal_gas = ideal_gas_eos()
        self.solver = NohBlackBoxEos(self.ideal_gas)
        self.solver.solve_jump_conditions()
        self.soln = self.solver(np.array([0.1, 0.3]), 0.6)

    def test_velocity_error(self):
        """Noh Problem: Test for valid value of velocity"""

        with pytest.raises(ValueError):
            NohBlackBoxEos(self.ideal_gas, u0=+1)

    def test_preshock_density(self):
        """Noh problem: Pre-shock density"""

        np.testing.assert_array_equal(self.soln.density[1], 9.0)

    def test_preshock_energy(self):
        """Noh problem: Pre-shock internal energy"""

        np.testing.assert_array_equal(self.soln.specific_internal_energy[1], 0.0)

    def test_preshock_velocity(self):
        """Noh problem: Pre-shock velocity"""

        np.testing.assert_array_equal(self.soln.velocity[1], -1.0)

    def test_preshock_pressure(self):
        """Noh problem: Pre-shock pressure"""

        np.testing.assert_array_equal(self.soln.pressure[1], 0.0)

    def test_postshock_density(self):
        """Noh problem: Post-shock density"""

        np.testing.assert_array_equal(self.soln.density[0], pytest.approx(64.0))

    def test_postshock_energy(self):
        """Noh problem: Post-shock internal energy"""

        np.testing.assert_array_equal(self.soln.specific_internal_energy[0], 0.5)

    def test_postshock_velocity(self):
        """Noh problem: Post-shock velocity"""

        np.testing.assert_array_equal(self.soln.velocity[0], 0.0)

    def test_postshock_pressure(self):
        """Noh problem: Post-shock pressure"""

        np.testing.assert_allclose(self.soln.pressure[0], 64.0/3.0)

    def test_geometry_error(self):
        """Noh Problem: Test for valid value of geometry"""

        with pytest.raises(ValueError):
            NohBlackBoxEos(self.ideal_gas, geometry=-1)


class TestNohWrappers():
    """Test wrappers for Noh in specific geometries.

    Test the wrapper functions for specific geometries from
    :mod:`exactpack.solvers.noh.noh1`, by comparing the results computed via
    the wrappers to those from the general solver.
    """
    @classmethod
    def setup_method(self):
        self.ideal_gas = ideal_gas_eos(gamma=1.4)

    def test_planar(self):
        """Planar Noh wrapper"""
        base_solver = NohBlackBoxEos(self.ideal_gas, geometry=1)
        base_solver.solve_jump_conditions()
        planar_solver = PlanarNohBlackBox(self.ideal_gas)
        planar_solver.solve_jump_conditions()

        np.testing.assert_array_equal(
            base_solver(np.linspace(0.1, 1), 0.6),
            planar_solver(np.linspace(0.1, 1), 0.6))

    def test_cylindrical(self):
        """Cylindrical Noh wrapper"""
        base_solver = NohBlackBoxEos(self.ideal_gas, geometry=2)
        base_solver.solve_jump_conditions()
        cylin_solver = CylindricalNohBlackBox(self.ideal_gas)
        cylin_solver.solve_jump_conditions()

        np.testing.assert_array_equal(
            base_solver(np.linspace(0.1, 1), 0.6),
            cylin_solver(np.linspace(0.1, 1), 0.6))

    def test_spherical(self):
        """Spherical Noh wrapper"""
        base_solver = NohBlackBoxEos(self.ideal_gas, geometry=3)
        base_solver.solve_jump_conditions()
        sph_solver = SphericalNohBlackBox(self.ideal_gas)
        sph_solver.solve_jump_conditions()

        np.testing.assert_array_equal(
            base_solver(np.linspace(0.1, 1), 0.6),
            sph_solver(np.linspace(0.1, 1), 0.6))



class TestPressureResidualFunction:
    r"""Tests for :class: `exactpack.solver.nohblackboxeos.solution_tools.residual_functions.noh_pressure_residual'. 

    The test consists of several parts. 
     
    First, ensuring that the class accepts only admissible initial conditions (positive density, pressure; 
    negative velocity, and symmetry (:math: 'm'), :math:'m \in {0,1,2}'). This includes enforcing restrictions on the initial conditions when the 
    symmtery is not 0. 

    Second, it tests array sizes--the function should only accept arrays of size 3 and that the determinant only computes for 3x3 matrices. 

    Third, it tests error handling for zero density and zero determinant. 

    Finally, it checks functions at a specific to ensure functions are producing correct output. 
    """

    @classmethod
    def setup_class(cls):
        cls.wrong_eos = None
        cls.eos = ideal_gas_eos()
        cls.wrong_velocity_ic = {'density': 1, 'velocity': 1, 'pressure': 0, 'symmetry': 0}
        cls.wrong_density_ic = {'density': -1, 'velocity': -1, 'pressure': 0, 'symmetry': 0}
        cls.wrong_pressure_ic = {'density': 1, 'velocity': -1, 'pressure': -1, 'symmetry': 0}
        cls.wrong_symmetry_ic = {'density': 1, 'velocity': -1, 'pressure': 0, 'symmetry': 10}
        cls.value_test_ic = {'density': 1, 'velocity': -1, 'pressure': 1, 'symmetry': 0}
        cls.wrong_pressure_symmetry_ic = {'density': 1, 'velocity': -1, 'pressure': 1, 'symmetry': 1}
        cls.good_ic = {'density': 1, 'velocity': -1, 'pressure': 0, 'symmetry': 0}
        cls.wrong_array = [10,10,10,10]
        cls.zero_density_array = [0,1,1]
        cls.wrong_matrix = np.zeros((3,2))
        cls.zero_determinant_array = [1, 0, 1] 

    def test_wrong_velocity(self):
        with pytest.raises(ValueError): 
            bad_velocity_function = pressure_noh_residual(self.wrong_velocity_ic, self.eos)
        """Noh Problem--Residual Function: Test for valid value of velocity"""
    
    def test_wrong_density(self):
        with pytest.raises(ValueError):
            bad_density_function = pressure_noh_residual(self.wrong_density_ic, self.eos)
        """Noh Problem--Residual Function: Test for valid value of density"""

    def test_wrong_pressure(self):
        with pytest.raises(ValueError): 
            bad_pressure_function = pressure_noh_residual(self.wrong_pressure_ic, self.eos)
        """Noh Problem--Residual Function: Test for valid value of pressure"""
    
    def test_wrong_symmetry(self):
        with pytest.raises(ValueError): 
            bad_symmetry_funciton = pressure_noh_residual(self.wrong_symmetry_ic, self.eos)
        """Noh Problem--Residual Function: Test for valid value of symmetry"""
    
    def test_wrong_eos(self):
        with pytest.raises(ValueError): 
            bad_eos_function = pressure_noh_residual(self.good_ic, self.wrong_eos)
        """Noh Problem--Residual Function: Test for valid eos object"""
    
    def test_pressure_symmetry_error(self):
        with pytest.raises(ValueError):
            bad_pressure_symmetry_function = pressure_noh_residual(self.wrong_pressure_symmetry_ic, self.eos)
        """Noh Problem--Residual Function: Test for admissability of symmetry and initial pressure"""

    def test_wrong_array_size_F(self): 
        function = pressure_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ValueError):
            function.F(self.wrong_array)
        """Noh Problem--Residual Function: Test for valid array size for F"""
    
    def test_zero_density_F(self): 
        function = pressure_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ZeroDensityError):
            function.F_prime(self.zero_density_array)
        """Noh Problem--Residual Function: Test for zero density for F"""
    
    def test_wrong_array_size_F_prime(self): 
        function = pressure_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ValueError):
            function.F_prime_inv(self.wrong_array)
        """Noh Problem--Residual Function: Test for valid array size for F_prime"""

    def test_zero_density_F_prime(self): 
        function = pressure_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ZeroDensityError):
            function.F_prime(self.zero_density_array)
        """Noh Problem--Residual Function: Test for zero density for F_prime"""
    
    def test_matrix_size(self):
        function = pressure_noh_residual(self.good_ic, self.eos)
        with pytest.raises(np.linalg.LinAlgError):
            function.determinant(self.wrong_matrix)
        """Noh Problem--Residual Function: Test for value size of matrix"""
    
    def test_wrong_array_size_F_prime_inv(self):
        function = pressure_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ValueError):
            function.F_prime_inv(self.wrong_array)
        """Noh Problem--Residual Function: Test for valid array size for F_prime_inv"""
    
    def test_zero_density_F_prime_inv(self):
        function = pressure_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ZeroDensityError):
            function.F_prime_inv(self.zero_density_array)
        """Noh Problem--Residual Function: Test for zero density for F_prime_inv"""
    
    def test_zero_determinant_in_F_prime_inv(self):
        function = pressure_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ZeroDeterminantError): 
            function.F_prime_inv(self.zero_determinant_array)
        """Noh Problem--Residual Function: Test for zero determinant in F_prime_inv"""

    def test_F_value(self): 
        function = pressure_noh_residual(self.value_test_ic, self.eos)
        np.testing.assert_almost_equal(function.F([1,1,1]), [-1., -4./3., -2], decimal=15)
        """Noh Problem--Residual Function: residual function value"""

    def test_F_prime_value(self): 
        function = pressure_noh_residual(self.value_test_ic, self.eos)
        np.testing.assert_almost_equal(function.F_prime([1,1,1]), [[-1,0,1], [-1./3., 2./3. ,-1], [-1, 1, 1]], decimal=15)
        """Noh Problem--Residual Function: Jacobian value"""

    def test_det_value(self): 
        function = pressure_noh_residual(self.value_test_ic, self.eos)
        np.testing.assert_almost_equal(function.determinant(function.F_prime([1,1,1])), 2 , decimal=14)
        """Noh Problem--Residual Function: determinant value"""

    def test_F_prime_inv_value(self): 
        function = pressure_noh_residual(self.value_test_ic, self.eos)
        np.testing.assert_almost_equal(function.F_prime_inv([1,1,1]), [[5./6., 1./2., -1./3.], [2./3.,1,1./3.], [1./6., -1./2., 1./3.]], decimal=12)
        """Noh Problem--Residual Function: Jacobian Inverse value"""


class TestSimplifiedPressureResidualFunction:
    r"""Tests for :class: `exactpack.solver.nohblackboxeos.solution_tools.residual_functions.simplified_pressure_noh_residual'. 

    The test consists of several parts. 
     
    First, ensuring that the class accepts only admissible initial conditions (positive density, pressure; 
    negative velocity, and symmetry (:math: 'm'), :math:'m \in {0,1,2}'). This includes enforcing that P_0 = symmetry = 0. (This is assumption 
    for this particular residual.)

    Second, it tests array sizes--the function should only accept arrays of size 3 and that the determinant only computes for 3x3 matrices. 

    Third, it tests error handling for zero density and zero determinant. 

    Finally, it checks functions at a specific to ensure functions are producing correct output. 
    """

    @classmethod
    def setup_class(cls):
        cls.wrong_eos = None
        cls.eos = ideal_gas_eos()
        cls.wrong_velocity_ic = {'density': 1, 'velocity': 1, 'pressure': 0, 'symmetry': 0}
        cls.wrong_density_ic = {'density': -1, 'velocity': -1, 'pressure': 0, 'symmetry': 0}
        cls.wrong_pressure_ic = {'density': 1, 'velocity': -1, 'pressure': 1, 'symmetry': 0}
        cls.wrong_symmetry_ic = {'density': 1, 'velocity': -1, 'pressure': 0, 'symmetry': 1}
        cls.value_test_ic = {'density': 1, 'velocity': -1, 'pressure': 0, 'symmetry': 0}
        cls.wrong_pressure_symmetry_ic = {'density': 1, 'velocity': -1, 'pressure': 1, 'symmetry': 1}
        cls.good_ic = {'density': 1, 'velocity': -1, 'pressure': 0, 'symmetry': 0}
        cls.wrong_array = [10,10,10]
        cls.zero_density_array = [0,1]
        cls.zero_determinant_array = [1, 0]

    def test_wrong_velocity(self):
        with pytest.raises(ValueError): 
            bad_velocity_function = simplified_noh_residual(self.wrong_velocity_ic, self.eos)
        """Noh Problem--Residual Function: Test for valid value of velocity"""
    
    def test_wrong_density(self):
        with pytest.raises(ValueError):
            bad_density_function = simplified_noh_residual(self.wrong_density_ic, self.eos)
        """Noh Problem--Residual Function: Test for valid value of density"""

    def test_wrong_pressure(self):
        with pytest.raises(ValueError): 
            bad_pressure_function = simplified_noh_residual(self.wrong_pressure_ic, self.eos)
        """Noh Problem--Residual Function: Test for valid value of pressure"""
    
    def test_wrong_symmetry(self):
        with pytest.raises(ValueError): 
            bad_symmetry_funciton = simplified_noh_residual(self.wrong_symmetry_ic, self.eos)
        """Noh Problem--Residual Function: Test for valid value of symmetry"""
    
    def test_wrong_eos(self):
        with pytest.raises(ValueError): 
            bad_eos_function = simplified_noh_residual(self.good_ic, self.wrong_eos)
        """Noh Problem--Residual Function: Test for valid eos object"""
    
    def test_pressure_symmetry_error(self):
        with pytest.raises(ValueError):
            bad_pressure_symmetry_function = simplified_noh_residual(self.wrong_pressure_symmetry_ic, self.eos)
        """Noh Problem--Residual Function: Test for admissability of symmetry and initial pressure"""

    def test_wrong_array_size_F(self): 
        function = simplified_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ValueError):
            function.F(self.wrong_array)
        """Noh Problem--Residual Function: Test for valid array size for F"""
    
    def test_zero_density_F(self): 
        function = simplified_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ZeroDensityError):
            function.F_prime(self.zero_density_array)
        """Noh Problem--Residual Function: Test for zero density for F"""
    
    def test_wrong_array_size_F_prime(self): 
        function = simplified_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ValueError):
            function.F_prime_inv(self.wrong_array)
        """Noh Problem--Residual Function: Test for valid array size for F_prime"""

    def test_zero_density_F_prime(self): 
        function = simplified_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ZeroDensityError):
            function.F_prime(self.zero_density_array)
        """Noh Problem--Residual Function: Test for zero density for F_prime"""

    def test_wrong_array_size_det(self): 
        function = simplified_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ValueError):
            function.determinant(self.wrong_array)
        """Noh Problem--Residual Function: Test for valid array size for F_prime"""

    def test_zero_density_det(self): 
        function = simplified_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ZeroDensityError):
            function.determinant(self.zero_density_array)
        """Noh Problem--Residual Function: Test for zero density for F_prime"""

    def test_wrong_array_size_F_prime_inv(self):
        function = simplified_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ValueError):
            function.F_prime_inv(self.wrong_array)
        """Noh Problem--Residual Function: Test for valid array size for F_prime_inv"""
    
    def test_zero_density_F_prime_inv(self):
        function = simplified_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ZeroDensityError):
            function.F_prime_inv(self.zero_density_array)
        """Noh Problem--Residual Function: Test for zero density for F_prime_inv"""
    
    def test_zero_determinant_in_F_prime_inv(self):
        function = simplified_noh_residual(self.good_ic, self.eos)
        with pytest.raises(ZeroDeterminantError): 
            function.F_prime_inv(self.zero_determinant_array)
        """Noh Problem--Residual Function: Test for zero determinant in F_prime_inv"""

    def test_F_value(self): 
        function = simplified_noh_residual(self.value_test_ic, self.eos)
        np.testing.assert_almost_equal(function.F([1,1]), [-1., 0.5], decimal= 15)
        """Noh Problem--Residual Function: residual function value"""

    def test_F_prime_value(self): 
        function = simplified_noh_residual(self.value_test_ic, self.eos)
        np.testing.assert_almost_equal(function.F_prime([1,1]), [[2./3.,0], [0,1.]], decimal=15)
        """Noh Problem--Residual Function: Jacobian value"""

    def test_det_value(self): 
        function = simplified_noh_residual(self.value_test_ic, self.eos)
        np.testing.assert_almost_equal(function.determinant([1,1]), 2./3., decimal=14)
        """Noh Problem--Residual Function: determinant value"""

    def test_F_prime_inv_value(self): 
        function = simplified_noh_residual(self.value_test_ic, self.eos)
        np.testing.assert_almost_equal(function.F_prime_inv([1,1]), [[3./2., 0], [0, 1.]], decimal=12)
        """Noh Problem--Residual Function: Jacobian Inverse value"""   