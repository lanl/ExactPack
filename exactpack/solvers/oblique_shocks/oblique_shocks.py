r"""Exact solution for oblique shock problems.
"""

from exactpack.base import ExactSolver, ExactSolution

import matplotlib.pyplot
from numpy import cos, sin, tan, pi, arcsin, arctan, sqrt, linspace, argmax, max, argmin, abs, interp, array, average, append, diff
from scipy.optimize import bisect, fsolve

### R  E  F  E  R  E  N  C  E  S

# [1] Chapman, C.J. High Speed Flow. 2000 Cambridge University Press
# [2] Anderson, J.D. Modern Compressible Flow with Historical Perspective. 2020
#     McGraw-Hill
# [3] Ben-Dor, G. Shockwave Reflection Phenomena. 2007 Springer


class TwoShockTheory(ExactSolver):
    r"""For computing pressure-deflection shock polars for reflected shocks.
    """
  
    parameters = {
        'geometry': '1=planar',
        'p0': 'reference upstream, unshocked pressure',
        'rho0': 'reference upstream, unshocked density',
        'temp0': 'reference upstream, unshocked temperature',
        'gamma': 'adiabatic index of the gas traversing the wedge',
        'M0_normal': 'normal flow Mach number streaming into the flow',
        'theta_wedge_deg': 'wedge angle (degrees) with respect to the horizontal',
        'chi0_deg': 'angle between wedge and line connecting triple-point to wedge corner'
        }
  
  
    geometry = 1
    p0 = 1.0
    rho0 = 0.001
    temp0 = 0.025
    gamma = 1.4
    M0_normal = 2.0
    theta_wedge_deg = 80.
    chi0_deg=0.
  
    def __init__(self, **kwargs):
        super(TwoShockTheory, self).__init__(**kwargs)
  
        self.theta_wedge_rad = self.theta_wedge_deg * pi / 180.
        self.chi0_rad = self.chi0_deg / 180. * pi
        self.phi0_rad = pi / 2. - self.theta_wedge_rad
        self.c0 = sqrt(self.gamma * self.p0 * 1.e6 / self.rho0)
  
  
    def _run(self):
        self.first_shock_jump()
        self.second_shock_jump()
        self.collect_remaining_flow_values()
  
        return ExactSolution([self.p0, self.p1, self.p2],
                             names=['p0',
                                    'p1',
                                    'p2'])
  
  
    def plot_pressure_deflection(self):
        matplotlib.pyplot.axhline(self.p1, color = 'g', linestyle = '--', linewidth = 1.)
        matplotlib.pyplot.axvline(self.theta1_rad, color = 'g', linestyle = '--', linewidth = 1.)
        matplotlib.pyplot.axvline(- self.theta2_rad + self.theta1_rad, color = 'r', linestyle = '--', linewidth = 1.)
        matplotlib.pyplot.axhline(self.p2, color = 'r', linestyle = '--', linewidth = 1.)
        matplotlib.pyplot.semilogy(  self.theta1s_rad, self.p1s, 'b')
        matplotlib.pyplot.semilogy(- self.theta2s_rad + self.theta1_rad, self.p2s, 'b')
        matplotlib.pyplot.semilogy(self.theta1_sonic, self.p1_sonic, 'om')
        matplotlib.pyplot.semilogy(self.theta2_sonic, self.p2_sonic, 'om')
        matplotlib.pyplot.xlabel('flow deflection angle [radians]')
        matplotlib.pyplot.ylabel('ratio of pressures, $p_1 / p_0$')
        matplotlib.pyplot.suptitle('Pressure-Deflection Shock Polar Diagram')
        matplotlib.pyplot.title('M0 = ' + str(self.M0_normal) + '; Wedge angle = ' + str(self.theta_wedge_deg) + ' degrees')
        matplotlib.pyplot.show()
  
  
    # T H I S   I S
    # Eq. 6.4.5a in [1]
    #   which is Eq. 6.3.16a for 1D flow;
    # Eq. 4.9 in [2]                   - C O R R E C T   T H I S
    #   which is Eq. 3.57 for 1D flow; - C O R R E C T   T H I S 
    def normal_velocity_jump(self, u, Mn):
        gamma = self.gamma
        return u * (1. + (gamma - 1.) * Mn**2 / 2.) / (gamma + 1.) / Mn**2 * 2.
   
   
    # T H I S   I S
    # Eq. 6.4.5b in [1]
    #   which is Eq. 6.3.16b for 1D flow;
    # Eq. 4.9 in [2]
    #   which is Eq. 3.57 for 1D flow;
    def normal_pressure_jump(self, p, Mn):
        gamma = self.gamma
        return p * (1. + 2. * gamma / (gamma + 1.) * (Mn**2 - 1.))
   
   
    # T H I S   I S
    # Eq. 6.4.5c in [1]
    #   which is Eq. 6.3.16c for 1D flow;
    # Eq. 4.8 in [2]
    #   which is Eq. 3.53 for 1D flow;
    def normal_density_jump(self, r, Mn):
        gamma = self.gamma
        return r * (gamma + 1.) * Mn**2 / (2. + (gamma - 1.) * Mn**2)
  
  
    def normal_temperature_jump(self, T, Mn):
        gamma = self.gamma
        val  = T * (1. + (gamma - 1.) * Mn**2 / 2.)
        val *= (gamma * Mn**2 - (gamma - 1.) / 2.) / ((gamma + 1.) * Mn)**2 * 4.
        return val
  
  
    # T H I S   I S
    # Eq. 6.4.5e in [1]
    #   which is Eq. 6.3.16e for 1D flow;
    def normal_soundspeed_jump(self, c, Mn):
        gamma = self.gamma
        val = (1. + (gamma-1.) * Mn**2 / 2.) * (gamma * Mn**2 - (gamma-1.) / 2.)
        val = c * sqrt(val) / (gamma + 1.) / Mn * 2.
        return val
   
   
    # T H I S   I S
    # I can't find this in [1]
    # Eq. 4.10 in [2]
    #   which is Eq. 3.51 for 1D flow;
    def normal_mach_jump(self, Mn):
        gamma = self.gamma
        return sqrt((2. + (gamma-1.) * Mn**2) / (2.*gamma*Mn**2 - gamma + 1.))
  
  
    def streaming_mach_number_from_normalMach(self, Mn, rad_angle):
        return Mn / sin(rad_angle)
  
  
    # This is Eq. 6.4.7 in [1], which has the form sqrt((a**2 + b**2) / a / c)
    def streaming_mach_number(self, Ms, rad_angle):
        gamma = self.gamma
        a = 1. + (gamma - 1.) * (Ms * sin(rad_angle))**2 / 2.
        b = (gamma + 1.) * Ms**2 * sin(rad_angle) * cos(rad_angle) / 2.
        c = gamma * (Ms * sin(rad_angle))**2 - (gamma - 1.) / 2.
        return sqrt((a**2 + b**2) / a / c)
  
  
    # T H I S   I S
    # Eq. 6.4.15 in [1]
    # Eq. 4.17 in [2]
    def deflection_angle(self, Ms, rad_angle):
        val  = 2. / tan(rad_angle) * (Ms**2 * sin(rad_angle)**2 - 1.)
        val /= (2. + Ms**2 * (self.gamma + cos(2. * rad_angle)))
        return arctan(val)
  
  
    # T H I S   I S
    # Eq. 6.4.10 in [1]
    def phi_sonic(self, Ms, phi_left, phi_right):
        return bisect(lambda phi: self.streaming_mach_number(Ms, phi) - 1.,
                      phi_left, phi_right)
  
  
    def postshock_states(self, c, r, T, M):
        c = self.normal_soundspeed_jump(c, M)
        r = self.normal_density_jump(r, M)
        T = self.normal_temperature_jump(T, M)
        M = self.normal_mach_jump(M)
        u = c * M
        return c, r, T, M, u
  
  
    def first_shock_jump(self):
        chi = self.chi0_rad
        phi0_rad = self.phi0_rad
        M0_normal = self.M0_normal
        M0s = self.streaming_mach_number_from_normalMach(M0_normal,
                                                         phi0_rad - chi)
        self.M0_streaming = M0s

        # create array of possible phi values ranging from the Mach angle for a
        # weak shock to a planar shock for a strong shock
        phi0s_rad = linspace(arcsin(1. / M0s), pi/2., int(1.e4))
        self.phi0s_rad = phi0s_rad
        self.M1_streaming = self.streaming_mach_number(M0s, phi0_rad - chi)

        # create the array of this field by ranging over the phi0s_rad array
        p1s = self.normal_pressure_jump(self.p0, M0s * sin(phi0s_rad))
        self.p1s = p1s
        theta1s_rad = self.deflection_angle(M0s, phi0s_rad)
        self.theta1s_rad = theta1s_rad

        # determine the deflection angle of the post-shock pressure
        self.p1 = self.normal_pressure_jump(self.p0, M0_normal)
        self.theta1_rad = interp(self.p1, p1s, theta1s_rad)
 
        # determine theta_max and the pressure there
        self.theta1s_argmax = argmax(theta1s_rad)
        self.p1_at_theta1_max = p1s[self.theta1s_argmax]
  
  
    def second_shock_jump(self):
        chi = self.chi0_rad
        M1s = self.M1_streaming
        theta1_rad = self.theta1_rad
        theta1s_rad = self.theta1s_rad

        phi1s_rad = linspace(arcsin(1./M1s), pi/2., int(1.e4))
        self.phi1s_rad = phi1s_rad

        p2s = self.normal_pressure_jump(self.p1, M1s * sin(phi1s_rad))
        self.p2s = p2s

        theta2s_rad = self.deflection_angle(M1s, phi1s_rad)
        self.theta2s_rad = theta2s_rad
    
        # I think these maxima can be made more exact based on expressions in
        # [1], but this isn't yet a concern, only something that should be done
        # "later"
        theta2s_argmax = argmax(theta2s_rad)
        self.theta2s_argmax = theta2s_argmax

        # determine whether pressure-deflection curves cross/overlap
        # ask where "theta min" is for the theta2s_rad curve,
        # but that's theta2s_argmax, even better, the 0deg crossing argument for
        # theta2s is arg_theta2_min, and theta1s_argmax is the right-most
        # point of that curve; really want argument of strong pressure at theta1
        # T H I S   I S   C O M P L I C A T E D
        # 1) focus on state2 curve from beginning until max deflection angle,
        #    which is the weak shock region (check strong shock region later)
        vals_weak = - theta2s_rad[:theta2s_argmax] + theta1_rad
        arg_theta2_min_weak = argmin(abs(vals_weak))
        vals_strong = - theta2s_rad[theta2s_argmax:] + theta1_rad
        arg_theta2_min_strong = argmin(abs(vals_strong)) + theta2s_argmax
        # 2) focus on state1 curve in subsonic region above max deflection angle
        #    to find array index at p1
        vals = theta1s_rad[self.theta1s_argmax:] - theta1_rad
        arg_strong_p1_at_theta1 = argmin(abs(vals)) + self.theta1s_argmax
        # 3) interpolate state2 thetas onto state1 thetas to pull state1
        # pressures
        vals_weak=interp((- theta2s_rad+theta1_rad)[:arg_theta2_min_weak][::-1],
                            theta1s_rad[arg_strong_p1_at_theta1:][::-1],
                            self.p1s[arg_strong_p1_at_theta1:][::-1])
        vals_strong=interp((- theta2s_rad + theta1_rad)[arg_theta2_min_strong:],
                              theta1s_rad[arg_strong_p1_at_theta1:][::-1],
                              self.p1s[arg_strong_p1_at_theta1:][::-1])
        # 4) compare projected state1 pressures from above with state2
        # pressures, subtracting to find index of nearest overlap; if the
        # pressures don't actually cross and there's no overlap then val (below)
        # will be zero
        val_weak   = argmin(abs(vals_weak   - p2s[:arg_theta2_min_weak][::-1])) 
        val_strong = argmin(abs(vals_strong - p2s[arg_theta2_min_strong:])) 
        if (val_weak > 0):
            arg_theta23_cross = arg_theta2_min_weak - val_weak
        elif (val_strong > 0):
            arg_theta23_cross = arg_theta2_min_strong + val_strong
        else:
            arg_theta23_cross = arg_theta2_min_weak
    
        if (arg_theta23_cross != arg_theta2_min_weak):
            self.morphology = 'SMR'
            p2 = p2s[arg_theta23_cross]
            self.p2 = p2
            p3 = p2
            self.p3 = p3

            phi1_rad = phi1s_rad[arg_theta23_cross]
            self.phi1_rad = phi1_rad

            theta2_rad = theta2s_rad[arg_theta23_cross]
            self.theta2_rad = theta2_rad

            theta3_rad = - theta2_rad + theta1_rad
            self.theta3_rad = theta3_rad

            # given p3 and the shock jump across the Mach stem from state0 to
            # state3 determine the shock angle, phi3_rad, that would produce p3,
            # and use it to determine flow values in state3
            phi3_rad_press = bisect(lambda phi:
                                    self.normal_pressure_jump(self.p0,
                                    self.M0_streaming*sin(phi - chi)) - p3,
                                    self.phi0s_rad[self.theta1s_argmax],
                                    self.phi0s_rad[-1])
            self.phi3_rad_press = phi3_rad_press
            phi3_rad = phi3_rad_press
      
            # given theta3 and the shock jump across the Mach stem from state0
            # to state3 determine the shock angle, phi3_rad, that would produce
            # theta and use it to determine flow values in state3
            phi3_rad_defl = bisect(lambda phi:
                                   self.deflection_angle(self.M0_streaming,
                                   phi - chi) - theta3_rad,
                                   self.phi0s_rad[self.theta1s_argmax],
                                   self.phi0s_rad[-1])
            self.phi3_rad_defl = phi3_rad_defl
      
            vals = self.postshock_states(self.c0, self.rho0, self.temp0,
                                         self.M0_streaming * sin(phi3_rad))

            self.c3, self.rho3, self.temp3 = vals[:3]
            self.M3_streaming, self.u1_streaming = vals[3:]
        else:
            self.morphology = 'RR'
            p2 = interp(0., (- theta2s_rad[:theta2s_argmax] + theta1_rad)[::-1],
                        p2s[:theta2s_argmax][::-1])
            self.p2 = p2

            phi1_rad = interp(0.,
                              (- theta2s_rad[:theta2s_argmax]+theta1_rad)[::-1],
                              phi1s_rad[:theta2s_argmax][::-1])
            self.ph1_rad = phi1_rad

            theta2_rad = theta1_rad
            self.theta2_rad = theta2_rad

            # the reflected shock angle
            self.omegaR_rad = phi1_rad - theta2_rad
    
        phi1_rad = min(phi1_rad, pi - phi1_rad)
        self.phi1_rad = phi1_rad
    
        M2_streaming = self.streaming_mach_number(M1s, phi1_rad)
        self.M2_streaming = M2_streaming
  
  
    def collect_remaining_flow_values(self):
        self.phi0_sonic = self.phi_sonic(self.M0_streaming, self.phi0s_rad[0],
                                         self.phi0s_rad[self.theta1s_argmax])
    
        self.theta1_sonic = interp(self.phi0_sonic, self.phi0s_rad,
                                   self.theta1s_rad)
    
        self.p1_sonic = interp(self.phi0_sonic, self.phi0s_rad, self.p1s)
    
        self.phi1_sonic = self.phi_sonic(self.M1_streaming, self.phi1s_rad[0],
                                         self.phi1s_rad[self.theta2s_argmax])
    
        self.theta2_sonic = - interp(self.phi1_sonic, self.phi1s_rad,
                                     self.theta2s_rad) + self.theta1_rad
    
        self.p2_sonic = interp(self.phi1_sonic, self.phi1s_rad, self.p2s)
    
        vals = self.postshock_states(self.c0, self.rho0, self.temp0,
                                     self.M0_normal)
    
        self.c1, self.rho1, self.temp1, self.M1_normal, self.u1_streaming = vals
    
        vals = self.postshock_states(self.c1, self.rho1, self.temp1,
                                     self.M1_streaming * sin(self.phi1_rad))
    
        self.c2, self.rho2, self.temp2, self.M2_normal, self.u2_streaming = vals
