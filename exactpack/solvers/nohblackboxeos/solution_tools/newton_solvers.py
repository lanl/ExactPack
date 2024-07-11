import numpy as np

class IterationError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return f"{self.message}"

# Generic Newton Solver. This is extremely modular and can be used to solve problems in arbitrary dimensions. There are no mathematical assumptions about the function F. However, the class for F needs to contain F (itself), 
# the Jacobian of F (F_prime) and the inverse of the Jacobian (F_prime_inv). This class requires no parameters to be constructed, however there are default values for robustness. The user can set the tolerance (to test for convergence) via set_new_tolerance()
# the initial guess (here called initial guess) via set_new_initial_guess(), and the maximum number of iterations via set_new_max_iteration(). The important function is the set_function(). This accepts a function from the noh_functions.py file (though it can accept 
# any function with the newton_solver_function base class found therein). The solve() function does the actual Newton iteration; it has options for outputing data. `verbose = True` will output real-time iteration data (current solution, residual, etc) and `output_file = file.txt`
# will put that same information into .txt file in the current directory. As mentioned, this class is meant for arbitrary dimensions, so `x_old` and `x_new` are viewed as n-tuples (though scalar values can be used as well in a 1D setting). 
class newton_solver:
        def __init__(self):
                self.function = None # Empty function 
                self.tolerance = 1.0e-06 # Set default tolerance 
                self.initial_guess = None # Empty initial guess
                self.residual = 10 # Set default residual 
                self.error = 10 # Set default error
                self.x_new = None
                self.F_x = None
                self.x_old = None
                self.max_iterations = 10000 # Set default max_iterations
                self.required_methods = ['F', 'F_prime_inv'] # The bare minimum member methods for the class `function` to have. 
                self.external_log_function = None

        def set_function(self, function): # Pass the function object into the solver
                self.function = function
                for method in self.required_methods:
                       if not hasattr(function, method):
                              raise ValueError(f"Error: function object does not have required methtod: {method}")

        def set_new_tolerance(self, epsilon): # Set the new tolerance
                if(epsilon > 1.0e-2):
                        raise ValueError("Error in newton_solver: tolerance is too low. Please set a smaller tolerance.") 
                self.tolerance = epsilon 
        
        def set_new_max_iteration(self, N): # Set the new max iterations
                self.max_iterations = N

        def set_new_initial_guess(self, x_0): # Set the new initial guess/guess
                self.initial_guess = x_0
                self.residual = 10
                self.error = 10

        def set_external_log_function(self, log_function): # Allows iteration to print to external files. 
               self.external_log_function = log_function
        
        def solve(self, verbose = True, output_file = None): # Run the Newton iteration until residual is below tolerance or maximum iterations are reached. 

                file = None # Empty file 
                if output_file: # If output_file is provided, intialize and prepare to write. 
                       file = open(output_file, 'w')
                
                # Internal function for outputing data in real time and into the output.txt file. 
                def log(message): 
                        if(verbose):
                            print(message)
                        if file:
                               file.write(message + "\n")
                        if self.external_log_function:
                               self.external_log_function(message)

                if(self.function == None): # Check that a function has been provided.
                       raise ValueError("Please provide a function")
                if(self.initial_guess == None): # Check that an initial guess has been provided. 
                       raise ValueError("Please provide an initial guess/guess")
                log(f"Beginning Newton iteration with initial guess = {self.initial_guess} \n Tolerance is set to {self.tolerance} \n Number of iterations allowed = {self.max_iterations} \n") # Output information at the beginning of the iteration.    
                iteration_counter = 0
                self.x_old = self.initial_guess # Set x_old to the initial guess.
                while(self.residual > self.tolerance or self.error > self.tolerance): # Enter while loop; checking if residual is larger than tolerance 
                      if(iteration_counter >= self.max_iterations): # Checking if the current iteration is larger than the max iteration allowed.
                            # log(f"Exceeded number of iterations. Exiting iteration.")
                            raise IterationError("Exceeded number of iterations. Exiting iteration.")                 
                      J_inv = self.function.F_prime_inv(self.x_old) # Compute F_prime_inv at current guess
                      self.F_x = self.function.F(self.x_old) # Compute F at current guess
                      self.x_new = self.x_old - np.dot(J_inv, self.F_x) # Newton algorithm 
                      self.residual = np.linalg.norm(self.x_new - self.x_old) # Compute residual (this is checking how far the old and new guesses are for convergence)
                      self.error = np.linalg.norm(self.function.F(self.x_new)) # Compute error (this is seeing how fart F(x_new) is from 0 since we are solving F(x) = 0)
                      self.x_old = self.x_new # Set new guess to old guess
                      iteration_counter += 1 # Up the iteration counter
                      log(f"At iteration number: {iteration_counter} \n solution = {self.x_old} \n residual = {self.residual} \n Error = {self.error} \n")

                # Add final solution information to output and output_file.txt if toggled for
                if (verbose):         
                        log(f"The solution is {self.x_old} \n with residual = {self.residual} \n and error = {self.error} \n ------------------------------------------------ \n ------------------------------------------------ \n")
                if file: 
                        file.close()
                result = {'solution': self.x_old.copy(), 'initial_guess': self.initial_guess.copy(), 'number_of_iterations': iteration_counter, 'residual_achieved': self.residual.copy(), 'error_achieved': self.error.copy()}
                return result



