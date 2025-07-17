import numpy as np

# This file serves an equation of state library. Contained within EoS classes that are meant to interface with
# the residual functions found in the `functions` directory. All classes are based on the `equation_of_state` base class,
# which provides the skeleton for all classes. The base class is meant to caputure the "black box" nature of our method:
# all we require is a relation between e, rho, and P. How that relationship is structured is open. For our purposes, we consider
# e = e(rho,P) and P = P(rho,e) formulations, and so our structure is biased with those formulations in mind.
#
# There are a few other classses of interest. The first is the `derived_singularity_eos` class. This class is the wrapper for the singularity-eos
# library, which allows the user to find a solution to the Noh Problem using singularity as its equation of state object. The class accepts a python
# singularity eos object. Note that, however, if the user is utilizing the e = e(rho, P) formulation, singularity does not have a call for this formulation.
# Thus, we use an "energy inverter" to compute e; see the `energy_solver.py` file for more information on this piece.
#
# The second class of interest is the `generic_mie_gruneisen` class. It is a base class (inheriting from the 'equation_of_state' class) for Mie-Gruneisen
# equations of state. See the singularity-eos documentation to find the specific form we chose. The basic idea is the following: we utilized the chain rule
# to calculate the requiste partial derivatives. All the user needs to implement is the specifcs of their EoS, such as the Gruneisen parameter.
#
# Finally, we highlight the `steinberg` class. This class is the EoS object for the Steinberg equations of state. The math is completely internal; all the
# user needs to provide is the parameters from the Steinberg. We have provided an example class for aluminum using the values from the Steinberg paper.
# These EoS are of particular interest as they model practical materials of interest, and thus are quite valuable for verification purposes.
#
# In total: we have classes for Ideal Gas, Stiffened Gas, Noble Able, Carnahan-Starling, Generic-Mie-Gruneisen, Steinberg, and singulariy-eos wrapper EoSs.
# Additional EoS classes can certainly be implemented; they need only follow the basic structure given the `equation_of_state` base class, and they should be able
# to interface with the residual functions referred to previously.


# Custom Error to check that the density is not zero. This is a common check for EoSs.
class EosZeroDensityError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return f"Error: zero density computed; {self.message}"

### equation_of_state base class. For EoS that can viewed as e = e(rho, P). Also supports P = P(rho, e) and rho = rho(P, e), but partial derivatives still need to be implemented.
class equation_of_state:
    def e(self, rho,P): #e = e(rho, P)
        return NotImplemented

    def de_drho(self, rho,P): # partial derivative of e wrt rho (constant P)
        return NotImplemented

    def P(self, rho, e): # P = P(rho, e)
        return NotImplemented

    def de_dP(self, rho, P): # partial derivative of e wrt P (constant rho)
        return NotImplemented

    def dP_drho(self, rho, e): # partial derivative of P wrt rho (constant e)
        return NotImplemented

    def dP_de(self, rho, e): # partial derivative of P wrt e (constant rho)
        raise NotImplementedError

## Implementation of the ideal gas equation of state using `equation_of_state` base class.
class ideal_gas_eos(equation_of_state):
    def __init__(self, gamma = 5/3):
        self.gamma = gamma
        assert(gamma != 1), "Gamma cannot be equal to 1"

    def P(self, rho, e):
        return rho*e*(self.gamma -1)

    def dP_drho(self, rho, e):
        return e*(self.gamma -1)

    def dP_de(self, rho, e):
        return rho*(self.gamma -1)

    def e(self, rho, P):
        if(rho == 0.0):
            raise EosZeroDensityError("Error: rho = 0. Please do not break math and divide by zero.")
        return P/(rho*(self.gamma -1.))

    def de_dP(self, rho, P):
        if(rho == 0.0):
            raise EosZeroDensityError("Error: rho = 0. Please do not break math and divide by zero.")
        return 1./(rho*(self.gamma-1))

    def de_drho(self, rho,P):
        if(rho == 0.0):
            raise EosZeroDensityError("Error: rho = 0. Please do not break math and divide by zero.")
        return -P/(rho**2*(self.gamma-1))


### Implementation of the stiffened gas equation of state using `equation_of_state` base class.
class stiffened_gas_eos(equation_of_state):
    def __init__(self, gamma = 5./3., c_s = np.sqrt(5./3.), rho_inf = 1.):
        self.gamma = gamma
        self.c_s = c_s
        self.rho_inf = rho_inf

    def e(self, rho,P):
        if(rho == 0.0):
            raise EosZeroDensityError("Error: rho = 0. Please do not break math and divide by zero.")
        return (P - self.c_s**2*(rho - self.rho_inf))/(rho*(self.gamma -1))

    def de_dP(self, rho, P):
        if(rho == 0.0):
            raise EosZeroDensityError("Error: rho = 0. Please do not break math and divide by zero.")
        return 1/(rho*(self.gamma-1))

    def de_drho(self, rho,P):
        if(rho == 0.0):
            raise EosZeroDensityError("Error: rho = 0. Please do not break math and divide by zero.")
        return (-P-self.c_s**2*self.rho_inf)/(rho**2*(self.gamma-1))

    def P(self, rho, e):
        return e*rho*(self.gamma -1) + self.c_s**2*(rho -self.rho_inf)

    def dP_drho(self, rho, e):
        return e*(self.gamma-1) + self.c_s**2

    def dP_de(self, rho, e):
        return rho*(self.gamma -1)

    def set_new_sound_speed(self, c_s_new):
        self.c_s = c_s_new

    def set_new_reference_density(self, rho_inf_new):
        self.rho_inf = rho_inf_new

# Implementation of the Noble-Able equation of state
class noble_abel_eos(equation_of_state):
    def __init__(self, gamma = 5./3., b = 0.01):
        self.gamma = gamma
        self.b = b

    def e(self, rho,P): # e = e(rho, P)
        if(rho == 0.0):
            raise EosZeroDensityError("Error: rho = 0. Please do not break math and divide by zero.")
        return (P*(1 - self.b*rho))/(rho*(self.gamma -1))

    def de_drho(self, rho,P): # partial derivative of e wrt rho (constant P)
        if(rho == 0.0):
            raise EosZeroDensityError("Error: rho = 0. Please do not break math and divide by zero.")
        return -P/(rho**2*(self.gamma -1))

    def de_dP(self, rho, P): # partial derivative of e wrt P (constant rho)
        if(rho == 0.0):
            raise EosZeroDensityError("Error: rho = 0. Please do not break math and divide by zero.")
        return (1 - self.b*rho)/(rho*(self.gamma -1))

    def set_new_co_volume(self,b):
        self.b = b

    def P(self, rho, e):
        if(1 - self.b*rho == 0):
            raise ValueError("Error: Division by zero.")
        return rho*e*(self.gamma - 1)/(1 -self.b*rho)

    def dP_drho(self, rho, e):
        if(1 - self.b*rho == 0):
            raise ValueError("Error: Division by zero.")
        return (e*(self.gamma -1)*(1 - self.b*rho) + self.b*rho*e*(self.gamma-1))/((1 - self.b*rho)**2)

    def dP_de(self, rho,e):
        if(1 - self.b*rho == 0):
            raise ValueError("Error: Division by zero.")
        return rho*(self.gamma -1)/(1 - self.b*rho)

# Implementation of the Carnahan-Starling Equation of State
class carnahan_starling_eos(equation_of_state):
    def __init__(self, gamma = 5./3., b = 1):
        self.gamma = gamma
        self.b = b

    def eta(self, rho):
        if(rho == 0.0):
            raise EosZeroDensityError("Error: rho = 0. Please do not break math and divide by zero.")
        return self.b*rho

    def deta_drho(self, rho):
        return self.b

    def Z(self, eta):
        if(eta == 1):
            raise ValueError("Error: eta = 0. Please do not break math and divide by zero.")
        return (1 + eta + eta**2 - eta**3)/((1 - eta)**3)

    def dZ_deta(self, eta):
        if(eta == 1):
            raise ValueError("Error: eta = 0. Please do not break math and divide by zero.")
        return ((1 + 2*eta - 3*eta**2)*(1 -eta) + 3*(1 + eta + eta**2 - eta**3))/((1 -eta)**4)

    def e(self, rho,P): #e = e(rho, P)
        if(rho == 0.0):
            raise EosZeroDensityError("Error: rho = 0. Please do not break math and divide by zero.")
        return P/(self.Z(self.eta(rho))*rho*(self.gamma -1))

    def de_drho(self, P, rho):
        if(rho == 0.0):
            raise EosZeroDensityError("Error: rho = 0. Please do not break math and divide by zero.")
        return -(self.dZ_deta(self.eta(rho))*self.deta_drho(rho)*rho + self.Z(self.eta(rho)))/((self.gamma -1)*(self.Z(self.eta(rho))**2*rho**2))

    def de_dP(self, rho, P):
        return 1/(self.Z(self.eta(rho))*rho*(self.gamma-1))

    def set_new_co_volume(self, b):
        self.b = b

    def P(self, rho, e):
        return e * rho * self.Z(self.eta(rho))*(self.gamma -1)

    def dP_drho(self, rho,e):
        return e*(self.gamma -1)*(self.Z(self.eta(rho)) + rho*self.dZ_deta(self.eta(rho))*self.deta_drho(rho))

    def dP_de(self, rho, e):
        return rho*self.Z(self.eta(rho))*(self.gamma -1)


# Base class for Mie-Gruneisen equations of state
class generic_mie_gruneisen(equation_of_state):
    def e(self, rho,P): #e = e(rho, P)
        return (P - self.P_inf(rho))/(rho*self.gruneisen(rho)) + self.e_inf(rho)

    def P(self, rho, e): # P = P(rho, e)
        return rho*self.gruneisen(rho)*(e - self.e_inf(rho)) + self.P_inf(rho)

    def rho(self, P, e): # rho = rho(P, e)
        return NotImplemented

    def P_inf(self, rho):
        raise NotImplementedError

    def e_inf(self, rho):
        raise NotImplementedError

    def gruneisen(self, rho):
        raise NotImplementedError

    def dPinf_drho(self, rho):
        raise NotImplementedError

    def deinf_drho(self, rho):
        raise NotImplementedError

    def dgru_drho(self, rho):
        raise NotImplementedError

    def dP_drho(self, rho, e):
        raise NotImplementedError

    def dP_de(self, rho, e):
        raise NotImplementedError

    def de_dP(self, rho, P): # partial derivative of e wrt P (constant rho)
        return 1/(rho*self.gruneisen(rho))

    def de_drho(self, rho,P): # partial derivative of e wrt rho (constant P)
        if(rho ==0):
            raise EosZeroDensityError("Error: rho = 0 causes a division by zero.")
        return (-self.dPinf_drho(rho)*rho*self.gruneisen(rho) - (P - self.P_inf(rho))*(self.gruneisen(rho) + self.dgru_drho(rho)*rho))/(rho**2*self.gruneisen(rho)**2) + self.deinf_drho(rho)

# Base class for Steinberg equations of state implementations
### Should only use this class for solid materials. The reference density should not be zero. ###
class steinberg(generic_mie_gruneisen):
    def __init__(self, reference_density, reference_pressure, reference_gruneisen, b, c_0, s_1, s_2, s_3):
        self.reference_density = reference_density
        self.reference_pressure = reference_pressure
        self.reference_gruneisen = reference_gruneisen
        self.b = b
        self.c_0 = c_0
        self.s_1 = s_1
        self.s_2 = s_2
        self.s_3 = s_3

    def P(self, rho, e):
        return self.P_inf(rho) + rho*self.gruneisen(rho)*(e - self.e_inf(rho))

    def dP_de(self, rho, e):
        return rho*self.gruneisen(rho)

    def dP_drho(self, rho, e):
        return self.dPinf_drho(rho) + e*(self.gruneisen(rho) + rho*self.dgru_drho(rho)) - (self.gruneisen(rho)*self.e_inf(rho) + rho*(self.e_inf(rho)*self.dgru_drho(rho) + self.gruneisen(rho)*self.deinf_drho(rho)))

    def eta(self, rho):
        if(rho ==0):
            raise EosZeroDensityError("Error: rho = 0 causes a division by zero.")
        return 1 - self.reference_density/rho

    def gruneisen(self, rho):
        if(self.eta(rho) <= 0):
            return self.reference_gruneisen
        if(self.eta(rho) > 0 and self.eta(rho)<1):
            return self.reference_gruneisen*(1 - self.eta(rho)) +  self.b*self.eta(rho)
        else:
            raise ValueError("Invalid value for gruneisen")

    def deinf_drho(self, rho):
        if(rho< self.reference_density):
            return 0
        elif(rho >= self.reference_density):
            return (self.deta_drho(rho)* self.P_inf(rho) + self.eta(rho)*self.dPinf_drho(rho) + self.reference_pressure*self.deta_drho(rho))/(2*self.reference_density)

    def poly(self, eta):
        return ( 1 - self.s_1*eta - self.s_2*eta**2 - self.s_3*eta**3)**2

    def dpoly_deta(self, eta):
        return 2*(1 - self.s_1*eta - self.s_2* eta**2 - self.s_3*eta**3)*(-self.s_1 - 2*self.s_2* eta - 3*self.s_3*eta**2)

    def P_inf(self, rho):
        if(rho< self.reference_density):
            return self.reference_pressure + self.c_0**2*self.eta(rho)*rho
        if(rho>=self.reference_density):
            return self.reference_pressure + self.c_0**2*self.eta(rho)*(self.reference_density/ self.poly(self.eta(rho)))

    def e_inf(self, rho):
        if(rho<self.reference_density):
            return 0
        if(rho >= self.reference_density):
            return self.eta(rho)*(self.P_inf(rho) + self.reference_pressure)/(2*self.reference_density)

    def deta_drho(self, rho):
        return self.reference_density/(rho**2)

    def dgru_drho(self, rho):
        if(self.eta(rho)<=0):
            return 0
        if(self.eta(rho)>0 and self.eta(rho)<1):
            return (-self.reference_gruneisen + self.b)*self.deta_drho(rho)

    def dPinf_drho(self, rho):
        if(rho < self.reference_density):
            return self.c_0**2*(self.eta(rho) + rho*self.deta_drho(rho))
        elif(rho>=self.reference_density):
            return self.c_0**2*self.reference_density*(self.poly(self.eta(rho)) + self.eta(rho)* self.dpoly_deta(self.eta(rho)))/(self.poly(self.eta(rho))**2)*self.deta_drho(rho)



# Implementation of the aluminum EoS using the Steinberg formulation
class aluminum_eos(steinberg):
    def __init__(self):
        reference_density = 2.703
        reference_pressure = 0
        reference_gruneisen = 1.97
        b = 0.48
        c_0 = 0.524*1.0e6
        s_1 = 1.40
        s_2 = 0.0
        s_3 = 0.0

        super().__init__(reference_density, reference_pressure, reference_gruneisen, b, c_0, s_1, s_2, s_3)


