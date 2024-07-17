import numpy as np
from .solution_tools import newton_solver, noh_residual
from ...base import ExactSolver, ExactSolution, Jump, JumpCondition

class NohBlackBoxEos(ExactSolver):
        parameters = {
        'geometry': "1=planar, 2=cylindrical, 3=spherical",
        'u0': "incident velocity (negative)",
        'rho0': "density"
        }

        solver = newton_solver()
        geometry = 3 # Default to sphere
        solver_tolerance = 1.0e-10 # Default tolerance 
        solver_max_iterations = 100 # Default iterations
        rho0 = 1 # Maybe refactor
        u0 = -1 # Maybe refactor 
        p0 = 0 # Maybe refactor 
        buffer = 0.5 # Buffer to help pick an initial guess for the Newton solver
        initial_guess = [rho0 + buffer, p0 + buffer, buffer] # Initial guess for the Newton solver (self.solver())   
        solution_data = None
        shocked_density = None
        shocked_pressure = None
        shock_speed = None

        def __init__(self, equation_of_state, **kwargs): # EoS object (as of now) is designed to be object from the eos_library.py file. 
            super(NohBlackBoxEos, self).__init__(**kwargs)
            self.eos = equation_of_state
            self.symmetry = self.geometry-1            
            self.initial_conditions = {'density': self.rho0, 'velocity': self.u0, 'pressure': self.p0, 'symmetry': self.symmetry} # Maybe refactor this later so users can change initial conditions. For now focus on black box eos interaction.             
            self.residual_funciton = noh_residual(self.initial_conditions, self.eos)

            if self.geometry not in [1, 2, 3]:
                raise ValueError("geometry must be 1, 2, or 3")

        def set_new_solver_initial_guess(self, new_initial_guess): # Perhaps put all of these together into one function? Or just leave independent?
              new_initial_guess

        def set_new_solver_tolerance(self, new_tolerance):
              self.solver_tolerance = new_tolerance
              self.solver.set_new_tolerance(new_tolerance)

        def set_new_solver_max_iterations(self, new_max_iteration):
              self.solver_max_iterations = new_max_iteration

        def solve_jump_conditions(self): # Solve jump conditions outside of _run() to avoid iterating every time _run() is called. 
                self.residual_funciton.set_new_initial_conditions(self.initial_conditions)
                self.solver.set_function(self.residual_funciton)
                self.solver.set_new_initial_guess(self.initial_guess)
                self.solution_data = self.solver.solve(verbose=False)
                self.shocked_density = self.solution_data['solution'][0]
                self.shocked_pressure = self.solution_data['solution'][1]
                self.shock_speed = self.solution_data['solution'][2]   

        def _run(self, r,t):
                if(self.shock_speed == None or self.shocked_density == None or self.shocked_pressure == None):
                     self.solve_jump_conditions()
                shock_location = self.shock_speed*t

                with np.errstate(all="ignore"): 
                        density = np.where(r<shock_location, self.shocked_density, self.rho0*(1 - (self.u0)*(t/r))**(self.symmetry))

                pressure = np.where(r<shock_location, self.shocked_pressure, self.p0)

                sie = np.where(r<shock_location, self.eos.e(self.shocked_density, self.shocked_pressure), self.eos.e(self.rho0, self.p0))

                velocity = np.where(r<shock_location, 0, self.u0)

                return ExactSolution([r, density, pressure, sie, velocity],
                                    names=['position',
                                            'density',
                                            'pressure',
                                            'specific_internal_energy',
                                            'velocity'],
                                    jumps=[JumpCondition(shock_location,
                                            "Shock",
                                            density=(self.shocked_density, 
                                            self.rho0*(1 - (self.u0)*self.shock_speed)**(self.symmetry)),
                                            pressure=(self.shocked_pressure, self.p0),
                                            sie=(self.eos.e(self.shocked_density, self.shocked_pressure), self.eos.e(self.rho0, self.p0)),
                                            velocity=(0, self.u0))
                                            ]
                )


class PlanarNohBlackBox(NohBlackBoxEos): 
    parameters = NohBlackBoxEos.parameters
    geometry = 1

class CylindricalNohBlackBox(NohBlackBoxEos): 
    parameters = NohBlackBoxEos.parameters
    geometry = 2

class SphericalNohBlackBox(NohBlackBoxEos): 
    parameters = NohBlackBoxEos.parameters
    geometry = 3

