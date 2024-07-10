"""Unit tests for the Black Box EoS Noh solver.

Unit tests include tests for the Newton Solver, the residual function, and the coupling between the two. 
"""

import pytest
import numpy as np
from exactpack.solvers.nohblackboxeos.solution_tools.residual_functions import noh_residual, ZeroDensityError, ZeroDeterminantError
from exactpack.solvers.nohblackboxeos.solution_tools.newton_solvers import newton_solver, IterationError

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

    class eos:  # Ideal Gas EoS 
        def __init__(self, gamma = 5/3): 
            self.gamma = gamma

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
        with pytest.raises(ValueError):
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
        np.testing.assert_equal(function.F([1,1,1]), [1., 1., -1.5])
        """Noh Problem--Residual Function: residual function value"""

    def test_F_prime_value(self): 
        function = noh_residual(self.value_test_ic, self.eos)
        np.testing.assert_equal(function.F_prime([1,1,1]), [[1,0,1], [-1,1,-1], [-0.5, 1.5, 1]])
        """Noh Problem--Residual Function: Jacobian value"""

    def test_det_value(self): 
        function = noh_residual(self.value_test_ic, self.eos)
        np.testing.assert_equal(function.determinant(function.F_prime([1,1,1])), 1.5)
        """Noh Problem--Residual Function: determinant value"""

    def test_F_prime_inv_value(self): 
        function = noh_residual(self.value_test_ic, self.eos)
        np.testing.assert_almost_equal(function.F_prime_inv([1,1,1]), [[5./3., 1, -2./3.], [1,1,0], [-2./3., -1, 2./3.]], decimal=15)
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
        cls.initial_guess = [5, 3, 0.5]
        cls.noh_residual_function = noh_residual(cls.initial_conditions, cls.eos)
        cls.solver.set_new_initial_guess(cls.initial_guess)

    class eos:  # Ideal Gas EoS 
        def __init__(self, gamma = 5/3): 
            self.gamma = gamma

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

    def test_solution(self): 
        result = self.solver.solve(verbose=False)
        np.testing.assert_almost_equal([result['density'], result['pressure'], result['shock_speed']], [4.0, 4./3., 1./3.], decimal=self.solver.tolerance)
        """Noh Problem--Noh Solver: Test for known solution values."""


