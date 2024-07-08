
import numpy as np

# Parent class for functions F that will be solved for F(x) = 0. There is the function `F`, its derivative `F_prime`, the determinant of the derivative `determinant`, and the derivatives inverse `F_prime_inv`
class newton_solver_residual_function:
    def F(self, *args, **kwargs):
        pass
    
    def F_prime(self, *args, **kwargs):
        pass 

    def determinant(self, *args, **kwargs):
        pass

    def F_prime_inv(self, *args, **kwargs): 
        pass     

class ZeroDeterminantError(Exception): 
    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return f"Error: Zero Determinant computed; {self.message}"

class ZeroDensityError(Exception):
    def __init__(self, message, code):
        self.message = message
        super().__init__(message)
    
    def __str__(self):
        return f"Error: zero density computed; {self.message}"


# Class for the (most) modular/generic Noh function. This function is the 3D function that solves for rho, P, and D. This is the class that MUST be used when P_0 =/= 0. 
class noh_residual(newton_solver_residual_function): 
    def __init__(self, initial_conditions, equation_of_state): # Initializer. Requires the initial conditions as a dictionary of the Noh problem and the desired eos (eos should come from the eos_library.py file).
        self.valid_instance = True
        self.required_eos_methods = ['e', 'de_drho', 'de_dP']
        self.u_0 = initial_conditions['velocity'] # Sets initial velocity
        if(self.u_0>=0):
            raise ValueError("Error: initial velocity must be negative by assumptions of the Noh Problem.")
        self.rho_0 = initial_conditions['density'] # Sets initial density 
        if(self.rho_0 <= 0): 
            raise ValueError("Error: initial density must be postive and nonzero.")
        self.P_0 = initial_conditions['pressure'] # Set initial pressure 
        if(self.P_0 <0):
            raise ValueError("Error: initial pressure must be nonnegative for the Noh Problem.")
        self.symmetry = initial_conditions['symmetry'] # Sets the geometry of the problem, ie, 0 (planar), 1 (cylindrical), 2 (sphereical)
        if self.symmetry not in [0,1,2]: 
            raise ValueError("Error: Symmetry must be 0, 1, or 2.")
        if( self.symmetry != 0 and self.P_0 != 0):
            raise ValueError("Error: if `symmetry' != 0, then the initial pressure must be 0.")
        for method in self.required_eos_methods:
            if not hasattr(equation_of_state, method):
                raise ValueError(f"Error: The equation of state class does not have the required member method: {method}.")
        self.equation_of_state = equation_of_state # Sets the equation of state 
        self.e_0 = self.equation_of_state.e(self.rho_0, self.P_0) # Computes the initial energy
        self.result = np.zeros(3) # Initialize the array for F(x) = result 
        self.DF = np.zeros((3,3)) # Initialize the matrix for F_prime(x) = DF
        self.DF_inv = np.zeros((3,3)) # Initialize the matrix F_prime_inv(x) = DF_inv

    def set_new_equation_of_state(self, new_eos):
        for method in self.required_eos_methods:
            if not hasattr(new_eos, method):
                raise ValueError(f"Error: The equation of state class does not have the required member method: {method}.")
        self.equation_of_state = new_eos

    def set_new_initial_conditions(self, new_initial_conditions): 
        self.u_0 = new_initial_conditions['velocity'] # Sets initial velocity
        if(self.u_0>=0):
            raise ValueError("Error: initial velocity must be negative by assumptions of the Noh Problem.")
        self.rho_0 = new_initial_conditions['density'] # Sets initial density 
        if(self.rho_0 <= 0): 
            raise ValueError("Error: initial density must be postive and nonzero.")
        self.P_0 = new_initial_conditions['pressure'] # Set initial pressure 
        if(self.P_0 <0):
            raise ValueError("Error: initial pressure must be nonnegative for the Noh Problem.")
        self.symmetry = new_initial_conditions['symmetry'] # Sets the geometry of the problem, ie, 0 (planar), 1 (cylindrical), 2 (sphereical)
        if self.symmetry not in [0,1,2]: 
            raise ValueError("Error: Symmetry must be 0, 1, or 2.")
        if( self.symmetry != 0 and self.P_0 != 0):
            raise ValueError("Error: if `symmetry' != 0, then the initial pressure must be 0.")
        try: 
            self.e_0 = self.equation_of_state.e(self.rho_0, self.P_0)
        except TypeError as e: 
            message = "Error: equation of state not initialized in grid_solver."
            raise TypeError(message) from e

    def F(self, state, *args, **kwargs): # Input must be an array--this makes this compatible with the generic Newton solver class, newton_solver, in newton_solver.py
        try:
            rho, P, D = state # Extract values from the array
        except ValueError as e: 
            message = "Must have three components to the vector/array."
            raise ValueError(message) from e
        if (rho ==0): # Make sure density isn't 0
            raise ZeroDensityError("Error: rho cannot be zero.")
        self.result[0] = rho - self.rho_0*( 1 - (self.u_0/D))**(self.symmetry+1) # First component, ie, F(rho, P, D)_1
        self.result[1] = P - self.P_0 + rho*self.u_0*D # Second component, ie, F(rho, P, D)_2
        self.result[2] = self.equation_of_state.e(rho, P) -self.e_0 - 0.5*(self.u_0)**2 + (self.u_0/rho)*(self.P_0/D) # Third component, ie, F(rho, P, D)_3
        return self.result
    
    def F_prime(self, state, *args, **kwargs): # Input must be an array--this makes this compatible with the generic Newton solver class, newton_solver, in newton_solver.py
        try:
            rho, P, D = state # Extract values from the array
        except ValueError as e: 
            message = "Must have three components to the vector/array."
            raise ValueError(message) from e
        if (rho ==0): # Make sure density isn't 0
            raise ZeroDensityError("Error: rho cannot be zero.")
        self.DF[0,0] = 1 
        self.DF[0,1] = 0 
        self.DF[0,2] = -(self.symmetry+1)*self.rho_0*(1 - (self.u_0/D))**(self.symmetry)*(self.u_0/(D**2))
        self.DF[1,0] = self.u_0*D
        self.DF[1,1] = 1
        self.DF[1,2] = rho*self.u_0
        self.DF[2,0] = self.equation_of_state.de_drho(rho,P) - (self.u_0*self.P_0)/(rho**2*D)
        self.DF[2,1] = self.equation_of_state.de_dP(rho,P)
        self.DF[2,2] = -(self.u_0*self.P_0)/(rho*D**2)
        return self.DF

    def determinant(self, mat, *args, **kwargs): # Accepts a matrix as an input 
        try: 
            matrix = mat
        except ValueError as e: 
            message = "Input must be a square matrix"
            raise ValueError(message) from e
        det = np.linalg.det(matrix) # Use numpy to compute determinant
        return det

    def F_prime_inv(self, state, *args, **kwargs): # Input must be an array--this makes this compatible with the generic Newton solver class, newton_solver, in newton_solver.py
        try:
            rho, P, D = state # Extract values from the array
        except ValueError as e: 
            message = "Must have three components to the vector/array."
            raise ValueError(message) from e
        if (rho ==0): 
            raise ZeroDensityError("Error: rho cannot be zero.")
        temp = self.F_prime([rho, P,D]) # Computes F_prime 
        det = self.determinant(temp) # Computes the determinant of F_prime
        if(det == 0): # Check that determinant isn't 0; helps to gurantee that the inverse exists
            raise ValueError("Zero determinant; inverse does not exist. Exiting iteration.")
        self.DF_inv = np.linalg.inv(temp) # Invert F_prime 
        return self.DF_inv

# Class for the an eos modular. This function is a 2D function that solves for rho, P. If this classed is used, then P_0 = 0 MUST BE THE INITIAL PRESSURE. 
class eos_noh_residual(newton_solver_residual_function): 
    def __init__(self, initial_conditions, equation_of_state): # Initializer. Requires the initial conditions as a dictionary of the Noh problem and the desired eos (eos should come from the eos_library.py file).
        self.valid_instance = True
        self.required_eos_methods = ['e', 'de_drho', 'de_dP']
        self.u_0 = initial_conditions['velocity'] # Sets initial velocity
        if(self.u_0>=0):
            raise ValueError("Error: initial velocity must be negative by assumptions of the Noh Problem.")
        self.rho_0 = initial_conditions['density'] # Sets initial density 
        if(self.rho_0 <= 0): 
            raise ValueError("Error: initial density must be postive and nonzero.")
        self.P_0 = initial_conditions['pressure'] # Set initial pressure 
        if(self.P_0 <0):
            raise ValueError("Error: initial pressure must be nonnegative for the Noh Problem.")
        self.symmetry = initial_conditions['symmetry'] # Sets the geometry of the problem, ie, 0 (planar), 1 (cylindrical), 2 (sphereical)
        if self.symmetry not in [0,1,2]: 
            raise ValueError("Error: Symmetry must be 0, 1, or 2.")
        if( self.symmetry != 0):
            raise ValueError("Error: This alogrithm assumes symmetry = 0.")
        for method in self.required_eos_methods:
            if not hasattr(equation_of_state, method):
                raise ValueError(f"Error: The equation of state class does not have the required member method: {method}.")
        self.equation_of_state = equation_of_state # Sets the equation of state 
        self.e_0 = self.equation_of_state.e(self.rho_0, self.P_0) # Computes the initial energy
        self.result = np.zeros(2) # Initialize the array for F(x) = result 
        self.DF = np.zeros((2,2)) # Initialize the matrix for F_prime(x) = DF
        self.DF_inv = np.zeros((2,2)) # Initialize the matrix F_prime_inv(x) = DF_inv

    def F(self, state, *args, **kwargs): # Input must be an array--this makes this compatible with the generic Newton solver class, newton_solver, in newton_solver.py
        try:
            rho, P = state # Extract values from the array
        except ValueError as e: 
            message = "Must have two components to the vector/array."
            raise ValueError(message) from e
        if (rho ==0): # Make sure density isn't 0
            raise ZeroDensityError("rho cannot be zero.")
        self.result[0] = P - self.u_0**2*self.rho_0 - P/rho*self.rho_0 # First component, ie, F(rho, P)_1
        self.result[1] = self.equation_of_state.e(rho, P) - self.e_0 - 0.5*self.u_0**2 # Second component, ie, F(rho, P)_2
        return self.result
    
    def F_prime(self, state, *args, **kwargs): # Input must be an array--this makes this compatible with the generic Newton solver class, newton_solver, in newton_solver.py
        try:
            rho, P = state # Extract values from the array
        except ValueError as e: 
            message = "Must have two components to the vector/array."
            raise ValueError(message) from e
        if (rho ==0): # Make sure density isn't 0
            raise ZeroDensityError("rho cannot be zero.")      
        self.DF[0,0] = P/(rho**2)*self.rho_0
        self.DF[0,1] = 1 - (1/rho)*self.rho_0
        self.DF[1,0] = self.equation_of_state.de_drho(rho,P)
        self.DF[1,1] = self.equation_of_state.de_dP(rho,P)
        return self.DF

    def determinant(self, state, *args, **kwargs): #Computes determinant DIRECTLY. Not using numpy; accepts an array as input
        try:
            rho, P = state # Extract values from the array
        except ValueError as e: 
            message = "Must have two components to the vector/array."
            raise ValueError(message) from e
        if (rho ==0): # Make sure density isn't 0
            raise ZeroDensityError("rho cannot be zero.")
        det_result = self.equation_of_state.de_dP(rho,P)*(P/rho**2)*self.rho_0 - self.equation_of_state.de_drho(rho,P)*(1 - (1/rho)*self.rho_0)
        if(det_result == 0.0): 
            print("Warning, 0 determinant, and F_prime is singular. Inverse does not exist.")
        return det_result

    def F_prime_inv(self, state, *args, **kwargs): # Input must be an array--this makes this compatible with the generic Newton solver class, newton_solver, in newton_solver.py. Computes the inverse DIRECTLY. 
        try:
            rho, P = state # Extract values from the array
        except ValueError as e: 
            message = "Must have two components to the vector/array."
            raise ValueError(message) from e
        if (rho ==0): 
            raise ZeroDensityError("rho cannot be zero.")
        det = self.determinant(state) # Computes the determinant
        if (det == 0.0): # Check that determinant isn't 0; helps to gurantee that the inverse exists
            raise ZeroDeterminantError("Error: zero determinant. Cannot construct inverse: it does not exist.")
        self.DF_inv[0,0] = self.equation_of_state.de_dP(rho,P)
        self.DF_inv[0,1] = - (1 - (1/rho)*self.rho_0)
        self.DF_inv[1,0] = -(self.equation_of_state.de_drho(rho,P))
        self.DF_inv[1,1] = P/(rho**2)*self.rho_0
        self.DF_inv = (1/det)*self.DF_inv
        return self.DF_inv

