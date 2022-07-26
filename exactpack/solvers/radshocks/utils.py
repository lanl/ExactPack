"""
Because this is the first item in the file, this is a docstring.
The (!) difference is that 'help' on this file will grab this docstring, and return it to the terminal (useful!).  The comments below go into overhead, but otherwise are unaccessible (useless!).  Use docstrings to explain how to use file/function/class/method...'  Use comments for longterm maintenance to describe internal methods of code.
"""

# TOC:
# BasicShockProfile(object)
#   def __init__(self, incoming)
#
# RadShockProfile(BasicShockProfile)
#   def downstream_equilibrium(self)
#
# IEShockProfile(BasicShockProfile)
#   def downstream_equilibrium(self)
#
# ShockMethods_2T(object)
#   def make_2T_solution(self)
#   def linearize_away_from_equilibrium(self)
#   def LHopital_near_equilibrium(self)
#   def make_mach_arrays(self)
#   def integrate_ddM(self) !!! this behemoth needs to be split
#   def check_solns_overlap(self)
#   def splice_precursor_and_relaxation(self)
#
# ED_ShockProfiles(RadShockProfile)
#   def __init__(self, incoming, **kwargs)
#   def make_ED_solution(self)
#
# nED_ShockProfiles(ED_ShockProfiles, ShockMethods_2T)
#   def __init__(self, incoming)
#
# Sn_ShockProfiles(nED_ShockProfiles)
#   def __init__(self, incoming)
#   def fill_in_xs(self)
#   def make_Ims(self)
#   def linearize_transport_equation(self)
#   def integrate_Sn(self)
#   def Sn_angular_moments(self)
#   def make_RT_solution(self)
#   def make_dictionaries(self)
#   def update_dictionaries(self)
#   def compute_errors(self)
#   def print_errors(self, i_iters)
#   def continue_running(self)
#
# IEShockProfiles(IEShockProfile, ShockMethods_2T)
#   def __init__(self, incoming)

import os, copy, numpy, scipy.optimize, scipy.integrate
import matplotlib.pyplot

class BasicShockProfile(object):
    '''
    '''
    def __init__(self, incoming):
        self.M0 = incoming.M0
        self.T0 = incoming.T0
        self.rho0 = incoming.rho0
        self.gamma = incoming.gamma
        self.num_pts = int(1e3)
        self.left_pts = int(self.num_pts * self.M0 + 1)
        self.max_steps = int(1e4)
        self.int_tol = incoming.int_tol
        self.eps_precursor_equil  = incoming.eps_precursor_equil
        self.eps_relaxation_equil = incoming.eps_relaxation_equil
        self.problem = incoming.problem
        self.use_jac = incoming.use_jac

class RadShockProfile(BasicShockProfile):
      '''
      '''
      def downstream_equilibrium(self):
          M0 = self.M0
          M02 = M0 * M0
          gamma = self.gamma
          P0 = self.P0
  
          def a2(T):
              return T / gamma
  
          def a1(T):
              return P0 * (T * T * T * T - 1.) / 3. - M02 - 1. / gamma
  
          def a0(T):
              return M02
  
          def b2(T):
              return (T - 1.) / (gamma - 1.) - M02 / 2. - 4. * P0 / 3.
  
          def b1(T):
              return 4. * P0 * T * T * T * T / 3.
  
          def b0(T):
              return M02 / 2.
  
          def discriminant_a(T):
              return a1(T) * a1(T) - 4. * a2(T) * a0(T)
  
          def discriminant_b(T):
              return b1(T) * b1(T) - 4. * b2(T) * b0(T)
  
          zero_discriminant_a = scipy.optimize.fsolve(discriminant_a, 1.,
                                                      xtol=1.e-13)
          zero_discriminant_b = scipy.optimize.fsolve(discriminant_b, 1.,
                                                      xtol=1.e-13)
  
          def momentum_and_energy(x):
              rho, T = x
              rho2 = rho * rho
              T4 = T * T * T * T
              momentum = M02 + rho2 * T / gamma + P0 * rho * T4 / 3. \
                       - rho * (M02 + 1. / gamma + P0 / 3.)
              energy   = M02 / 2. + rho2 * T / (gamma - 1.) \
                       + 4. * P0 * rho * T4 / 3. \
                       - rho2 * (M02 / 2. + 1. / (gamma - 1.) + 4. * P0 / 3.)
              return [momentum, energy]
          rho1, T1 = scipy.optimize.fsolve(momentum_and_energy,
                                          [zero_discriminant_b,
                                           zero_discriminant_a],
                                           xtol = 1.e-13)
          self.Pr1    = T1 * T1 * T1 * T1 / 3.
          self.Er1    = T1 * T1 * T1 * T1
          self.M1     = M0 / rho1 / T1**(0.5)
          self.speed1 = M0 / rho1
          self.rho1   = rho1
          self.T1     = T1
  
class IEShockProfile(BasicShockProfile):
      '''
      '''
      def downstream_equilibrium(self):
          print_stmnt  = '\n'
          print_stmnt += 'entered downstream_equilibrium' + '\n'
          M0 = self.M0
          M02 = M0 * M0
          gamma = self.gamma
          a = gamma * (2. * M02 - 1.) + 1.
          b = - 2. * (gamma * M02 * M02 + 1.)
          c = M02 * ((gamma - 1.) * M02 + 2.)
#         the positive value returns M0 and is a good unittest
#         M0_test = numpy.sqrt((- b + numpy.sqrt(b * b - 4. * a * c)) / 2. / a)
#         M0_test == M0; if not exit gracefully on warning
          M1 = numpy.sqrt((- b - numpy.sqrt(b * b - 4. * a * c)) / 2. / a)
          M12 = M1 * M1
          rho1 = self.rho0 * M02 * (gamma * M12 + 1.) / M12 / (gamma * M02 + 1.)
          T1 = M02 / M12 / rho1 / rho1
          self.M1     = M0 / rho1 / T1**(0.5)
          self.speed1 = M0 / rho1
          self.rho1   = rho1
          self.T1     = T1
          print_stmnt += 'M1     = ' + str(self.M1) + '\n'
          print_stmnt += 'speed1 = ' + str(self.speed1) + '\n'
          print_stmnt += 'rho1   = ' + str(self.rho1) + '\n'
          print_stmnt += 'T1     = ' + str(self.T1) + '\n'
          print_stmnt += 'leaving downstream_equilibrium' + '\n'
        
class ShockMethods_2T(object):
    '''
    '''
    def make_2T_solution(self):
        print_stmnt  = '\n'
        print_stmnt += 'entered make_RH_solution\n'
        self.false_continuity = 0.
        self.integrate_ddM_again = 2
        while self.integrate_ddM_again:
            self.integrate_ddM_again -= 1
            if (self.integrate_ddM_again == 1):
                self.linearize_away_from_equilibrium()
            self.make_mach_arrays()
            self.integrate_ddM()
            self.check_solns_overlap()
            print_stmnt += 'in make_RH_solution, self.continuous_shock = '
            print_stmnt += str(self.continuous_shock)
            if (self.continuous_shock > 0.):
                print_stmnt += 'in make_RH_solution, '
                print_stmnt += 'checking validity of continuous shock\n'
                self.make_mach_arrays()
                self.integrate_ddM()
                self.check_solns_overlap()
            if (self.integrate_ddM_again == 0):
                self.splice_precursor_and_relaxation()
        print_stmnt += 'leaving make_RH_solution\n'

    def linearize_away_from_equilibrium(self):
        print_stmnt  = '\n'
        print_stmnt += 'entered linearize_away_from_equilibrium' + '\n'
        print('entered linearize_away_from_equilibrium')
        if ('f' in self.__dict__):
            d_M = self.d_M[-1][::-1]
            d_Prt = self.d_Prt[-1][::-1]
            d_f = self.d_f[-1][::-1]
            Pr_eps0 = numpy.interp(self.M_eps0, d_M, d_Prt)
            f_at_eps0 = numpy.interp(self.M_eps0, d_M, d_f)
            self.Pr_eps0 = max(Pr_eps0, self.Pr_eps0)
            self.Pr_eps1 = numpy.interp(self.M_eps1, d_M, d_Prt)
            f_at_eps1 = numpy.interp(self.M_eps1, d_M, d_f)
            f_argmin = numpy.argmin(d_f[-1])
            val_arg  = numpy.argmin(numpy.abs(d_f[-1] - 1.e-4 - d_f[f_argmin:]))
            val_arg += f_argmin
            M_eps1 = d_M[val_arg]
            Pr_eps1 = d_Prt[val_arg]
            print('\n')
            print('f at eps0 = ', f_at_eps0)
            print('f at eps0 - f_eq = ', f_at_eps0 - self.f[0])
            print('\n')
            print('f at eps1 = ', f_at_eps1)
            print('f_eq - f at eps1 = ', self.f[-1] - f_at_eps1)
            print('\n')
            print('delta M for (delta f = 1.e-4) is ', M_eps1 - self.M1)
            print('\n')
            print('self.Pr_eps1 = ', self.Pr_eps1)
            print('but Pr_eps1 for (delta f = 1.e-4) is ', Pr_eps1)
            print('and abs(Pr_eps1 - self.Pr_eps1) / Pr_eps1 = ', abs(Pr_eps1 - self.Pr_eps1) / Pr_eps1)
            print('\n')
            if (M_eps1 < self.M_eps1):
              self.M_eps1 = M_eps1
              self.Pr_eps1 = Pr_eps1
              print('\n')
              print("This looks better so we're usin' it.")
              print('\n')
              print('self.M_eps1 = ', self.M_eps1)
        else:
            T1 = self.T1
            if ('ED' in self.problem):
                T14 = T1 * T1 * T1 * T1
                eq0_val = 1. / 3.
                eq1_val = T14 / 3.
                y_lin = fnctn.dPdM
            elif ('FLD' in self.problem):
                T14 = T1 * T1 * T1 * T1
                eq0_val = 1.
                eq1_val = T14
                y_lin = fnctn.dEdM
            elif ('ie' in self.problem):
                eq0_val = 1.
                eq1_val = T1
                y_lin = fnctn.dTedM
            eps0 = self.eps_precursor_equil
            eps1 = self.eps_relaxation_equil
            M_eps0 = self.M0 - eps0
            self.M_eps0 = M_eps0
            init_val = eq0_val + eps0
            val0 = scipy.optimize.fsolve(
                   lambda y: y - eq0_val + eps0 * y_lin(y, M_eps0, self) / self.epsilon**2,
                   init_val, xtol = 1.e-13)[0]
            print_stmnt += 'init_val = ' + str(init_val) + '\n'
            print_stmnt += 'val = ' + str(val0) + '\n'
            print('val0 = ', val0)
            print('val0 - eq0_val = ', val0 - eq0_val)
            if (val0 < eq0_val):
                print_stmnt += "if (val0 < eq0_val):"
                print("if (val0 < eq0_val):")
                print('eq0_val = ', eq0_val)
                print('val0 = ', val0)
                print('eq0_val - val0 = ', eq0_val - val0)
                val0 += 20. * numpy.abs(eq0_val - val0)
                val0 = max(8.e-7 + eq0_val, val0)
                print('val0 = ', val0)
                print('val0 - eq0_val= ', val0 - eq0_val)
                print_stmnt += 'val = ' + str(val0) + '\n'
            init_val = eq1_val - eps1
            M_eps1 = self.M1 + eps1
            self.M_eps1 = M_eps1
            val1 = scipy.optimize.fsolve(
                  lambda y: y - eq1_val - eps1 * y_lin(y, M_eps1, self) / self.epsilon,
                  init_val, xtol = 1.e-13)[0]
            print_stmnt += 'val = ' + str(val1) + '\n'
            if (val1 > eq1_val):
                print_stmnt += "if (val1 > eq1_val):"
                print("if (val1 > eq1_val):")
                print('val1 = ', val1)
                print('eq1_val = ', eq1_val)
                print('eq1_val - val1 = ', eq1_val - val1)
                val1 -= 10. * numpy.abs(eq1_val - val1)
                print('val1 = ', val1)
                print('eq1_val = ', eq1_val)
                print('eq1_val - val1 = ', eq1_val - val1)
                print_stmnt += 'val = ' + str(val1) + '\n'
            if ('ED' in self.problem):
                self.Pr_init = [val0, val1]
                self.Pr_eps0 = val0
                self.Pr_eps1 = val1
            if ('FLD' in self.problem):
                self.Er_init = [val0, val1]
                self.Er_eps0 = val0
                self.Er_eps1 = val1
                self.Lambda = 1./3.
                self.R = 0.
            elif ('ie' in self.problem):
                self.Te_init[0] = val0
                self.Te_eps0 = val0
                self.Te_init[1] = val1
                self.Te_eps1 = val1
        print_stmnt += 'leaving linearize_away_from_equilibrium' + '\n'

    def LHopital_near_equilibrium(self):
        print('entered LHopital_near_equilibrium')
        def get_vals(eps, Peq, Meq, self):
            a  = fnctn.ddM_dPdx_eq(Peq0, Meq0, self)
            b  = fnctn.ddP_dPdx_eq(Peq0, Meq0, self)
            b -= fnctn.ddM_dMdx_eq(Peq0, Meq0, self)
            c  = fnctn.ddP_dMdx_eq(Peq0, Meq0, self)
            b2 = b * b
            mult_sign = numpy.sign(1. - Meq)
            val = (b + mult_sign * numpy.sqrt(b2 + 4. * a * c)) / 2. / a * eps
            P = Peq + val
            print(f'val = {val}')
            print(f'Peq + val = {P}')
            return Peq + val
        eps0 = self.eps_precursor_equil
        Meq0 = self.M0
        Peq0 = self.T0**4 / 3.
        eps1 = self.eps_relaxation_equil
        Meq1 = self.M1
        Peq1 = self.Pr1
        get_vals(eps0, Peq0, Meq0, self)
        get_vals(eps1, Peq1, Meq1, self)

    def make_mach_arrays(self):
        print_stmnt = '\n'
        print_stmnt += 'entered make_mach_arrays' + '\n'
        print_stmnt += 'self.eps_precursor_equil = '
        print_stmnt += str(self.eps_precursor_equil) + '\n'
        print_stmnt += 'self.eps_precursor_ASP = '
        print_stmnt += str(self.eps_precursor_ASP) + '\n'
        print_stmnt += 'self.eps_relaxation_ASP = '
        print_stmnt += str(self.eps_relaxation_ASP) + '\n'
        print_stmnt += 'self.eps_relaxation_equil = '
        print_stmnt += str(self.eps_relaxation_equil) + '\n'
        M_eps0 = self.M_eps0
        M_eps1 = self.M_eps1
        left_pts = int(self.left_pts)
        right_pts = int(self.right_pts)
        print_stmnt += 'self.eps_precursor_ASP_hold = '
        print_stmnt += str(self.eps_precursor_ASP_hold)
        print_stmnt += 'self.eps_relaxation_ASP_hold = '
        print_stmnt += str(self.eps_relaxation_ASP_hold)
        print_stmnt += 'self.eps_precursor_ASP = '
        print_stmnt += str(self.eps_precursor_ASP)
        print_stmnt += 'self.eps_relaxation_ASP = '
        print_stmnt += str(self.eps_relaxation_ASP)
        if (self.continuous_shock > 0):
            self.eps_precursor_ASP = self.eps_precursor_ASP_hold
            self.eps_relaxation_ASP = self.eps_relaxation_ASP_hold
        print('self.eps_precursor_ASP = ', self.eps_precursor_ASP)
        print('self.eps_relaxation_ASP = ', self.eps_relaxation_ASP)
        Mp = numpy.linspace(M_eps0, 1. + self.eps_precursor_ASP,
                            int(left_pts), endpoint = True)
        Mp_index = int(left_pts / 10.)
        Mp_linfill_eq  = numpy.linspace(M_eps0, Mp[Mp_index], int(left_pts),
                                        endpoint = False)
        Mp_linfill_ASP = numpy.linspace(Mp[-Mp_index], Mp[-1], int(left_pts),
                                        endpoint = True)
        Mr = numpy.linspace(M_eps1, 1. - self.eps_relaxation_ASP,
                            int(right_pts), endpoint = True)
        Mr_index = int(right_pts / 10.)
        Mr_linfill_eq  = numpy.linspace(M_eps1, Mr[Mr_index], int(right_pts),
                                        endpoint = False)
        Mr_linfill_ASP = numpy.linspace(Mr[-Mr_index], Mr[-1], int(right_pts),
                                        endpoint = True)
        Mp = numpy.append(Mp_linfill_eq, Mp[Mp_index + 1:-Mp_index])
        Mr = numpy.append(Mr_linfill_eq, Mr[Mr_index + 1:-Mr_index])
        Mp = numpy.append(Mp, Mp_linfill_ASP)
        Mr = numpy.append(Mr, Mr_linfill_ASP)
        self.Mach_precursor = Mp
        self.Mach_relaxation = Mr
        print_stmnt += 'leaving make_mach_arrays' + '\n'

    def integrate_ddM(self):
        problem = self.problem
        print_stmnt  = '\n'
        print_stmnt += 'entered integrate_ddM\n'
        print('use_jac = ', self.use_jac)
        if ('f' in self.__dict__):
#             i = scipy.integrate.ode(fnctn.ddM_ode)
#             i.set_f_params(self)
            i = scipy.integrate.ode(fnctn.ddM_ode, fnctn.ddM_jac)
            i.set_f_params(self).set_jac_params(self)
            d_M = self.d_M[-1][::-1]
            d_f = self.d_f[-1][::-1]
            d_Prt = self.d_Prt[-1][::-1]
            def check_for_spurious_soln():
                not_spurious = True
                P = result_arrays[k - 1,0]
                M = integ_array[k - 1]
                P_lt_fMT4 = True
                P_lt_prev = True
                P_is_zero = False
                P_gt_onethird = True
                if (self.dxdM_val > 0):
                    if (side == 'precursor'):
                        M_near_equil_bool = bool(M > M_equil_boundary)
                    else:
                        M_near_equil_bool = bool(M < M_equil_boundary)
                        self.dxdM_relax_negative = \
                        bool(bool(M_near_equil_bool) |
                             bool(self.dxdM_relax_negative))
                    not_spurious = not M_near_equil_bool
                if ((side == 'relaxation') & self.dxdM_relax_negative):
                    rho = fnctn.mat_density(P, M, self)
                    T4 = (self.M0 / M / rho)**8
                    fM = numpy.interp(M, d_M, d_f)
                    P_lt_fMT4 = bool(P < fM * T4)
                    prev = result_arrays[k - 2, 0]
                    P_lt_prev = bool(P < prev)
                    P_is_zero = bool(P == 0.)
                    P_lt_onethird = bool(P < 1./3.)
                    P_gt_onethird = not P_lt_onethird
# the not_spurious boolean is there to ensure that the radiation temperature
# is properly bounded by the material temperature
                    not_spurious = (P_lt_fMT4 & P_lt_prev & P_gt_onethird)
                return not_spurious, P_lt_fMT4, P_lt_prev, P_is_zero, P_gt_onethird
        else:
            if ('ED' in problem):
                if self.use_jac:
                    i = scipy.integrate.ode(fnctn.ddM_ode, fnctn.ddM_jac)
                    i.set_f_params(self).set_jac_params(self)
                else:
                    i = scipy.integrate.ode(fnctn.ddM_ode)
                    i.set_f_params(self)
            elif ('FLD' in problem):
                if self.use_jac:
                    i = scipy.integrate.ode(fnctn.ddM_ode, fnctn.ddM_jac)
                    i.set_f_params(self).set_jac_params(self)
                else:
                    i = scipy.integrate.ode(fnctn.ddM_ode)
                    i.set_f_params(self)
            elif ('ie' in problem):
                i = scipy.integrate.ode(fnctn.ddM_ode)
                i.set_f_params(self)
        i.set_integrator('vode', atol = self.int_tol, rtol = self.int_tol,
                         method = 'bdf', nsteps = self.max_steps)
        for side in ('relaxation', 'precursor'):
            if ('FLD' in problem):
              Lambda_vals = []
              R_vals = []
            if (side == 'precursor'):
                M_equil_boundary = (self.M0 + 1.) / 2.
                integ_array = self.Mach_precursor
                if ('ED' in problem):
                    cont0 = self.Pr0
                    cont_init = self.Pr_eps0
                elif ('FLD' in problem):
                    cont0 = self.Er0
                    cont_init = self.Er_eps0
                    fnctn.dEdx(cont_init, integ_array[0], self)
                elif ('ie' in problem):
                    cont0 = self.Te0
                    cont_init = self.Te_eps0
                init_vals = [cont_init, 0]
                print_stmnt += 'self.Mach_precursor = '
                print_stmnt += str(self.Mach_precursor) + '\n'
            elif (side == 'relaxation'):
                M_equil_boundary = (self.M1 + 1.) / 2.
                integ_array = self.Mach_relaxation
                if ('ED' in problem):
                    cont0 = self.Pr1
                    cont_init = self.Pr_eps1
                elif ('FLD' in problem):
                    cont0 = self.Er1
                    cont_init = self.Er_eps1
                    fnctn.dEdx(cont_init, integ_array[0], self)
                elif ('ie' in problem):
                    cont0 = self.T1
                    cont_init = self.Te_eps1
                init_vals = [cont_init, 0]
                print_stmnt += 'self.Mach_relaxation = '
                print_stmnt += str(self.Mach_relaxation) + '\n'
            integ_array_diffed = numpy.diff(integ_array)
            integ_array_size = numpy.size(integ_array)
            result_arrays = numpy.zeros((integ_array_size, 2))
            result_arrays[0,:] = init_vals
            mult_exp = 0
            mult_val = 1.
            Mach_precursor_baseline = 0.
            Mach_relaxation_baseline = 0.
#             while ((mult_exp < 10) & (mult_val > 0)):
            while ((mult_exp < 8) & (mult_val > 0)):
                print('mult_exp = ' + str(mult_exp))
                print('mult_val = ' + str(mult_val))
                print_stmnt += 'mult_exp = ' + str(mult_exp) + '\n'
                print_stmnt += 'mult_val = ' + str(mult_val) + '\n'
                print_stmnt += 'cont0           = ' + str(cont0) + '\n'
                print_stmnt += 'cont_init       = ' + str(cont_init) + '\n'
                print_stmnt += 'init_vals[0]  = ' + str(init_vals[0]) + '\n'
                print_stmnt += 'integrating ' + side + ' region' + '\n'
                i.set_initial_value(init_vals, integ_array[0])
                k = 1
                got_zeros = False
                not_spurious = True
                while (i.successful() & not_spurious & (k < integ_array_size)):
                    dt = integ_array_diffed[k - 1]
                    i.integrate(i.t + dt)
                    result_arrays[k, 0] = i.y[0]
                    result_arrays[k, 1] = i.y[1]
                    if ('FLD' in problem):
                      Lambda_vals.append(self.Lambda)
                      R_vals.append(self.R)
                    k += 1
                    if ('f' in self.__dict__):
                        unpackem = check_for_spurious_soln()
                        not_spurious = unpackem[0]
                        if not unpackem[1]:
                          print('P   gt fM T^4')
                        if not unpackem[2]:
                          print('P   gt prev')
                        if unpackem[3]:
                          print('P   is 0')
                        if not unpackem[4]:
                          print('P   lt 1/3')
                print_stmnt += 'finished integrating ' + side + ' region' + '\n'
                print('finished integrating ' + side + ' region' + '\n')
                y = result_arrays[:, 0]
                x = result_arrays[:, 1]
                M = integ_array
                print_stmnt += 'y = ' + str(y) + '\n'
                print_stmnt += 'x = ' + str(x) + '\n'
                print_stmnt += 'M = ' + str(M) + '\n'
                got_zeros = numpy.any(y == 0.)
                print_stmnt += 'got_zeros = ' + str(got_zeros) + '\n'
                if (got_zeros & not_spurious):
                    print('\n')
                    print('got zeroes')
                    print('side = ', side)
                    print('got zeroes')
                    print('side = ', side)
                    print('got zeroes')
                    print('\n')
                    while got_zeros:
                        argmin_y = numpy.argmin(y)
                        x = numpy.delete(x, argmin_y)
                        M = numpy.delete(M, argmin_y)
                        y = numpy.delete(y, argmin_y)
                        got_zeros = numpy.any(y == 0)
                        if ('f' in self.__dict__):
                          not_spurious = False
                    print_stmnt += 'got_zeros = ' + str(got_zeros) + '\n'
                print_stmnt += 'not_spurious          = '
                print_stmnt += str(not_spurious) + '\n'
                if (not not_spurious):
                    print_stmnt += 'got a spurious solution' + '\n'
                    print_stmnt += 'spurious solution = ' + str(y) + '\n'
                    if (mult_exp == 0):
                        mult_exp += 1
                        mult_val -= 10**(- mult_exp)
                        init_vals[0] = mult_val * cont_init
                    else:
                        mult_val -= 10**(- mult_exp)
                        init_vals[0] = mult_val * cont_init
                elif (mult_exp != 0):
                    mult_val += 10**(- mult_exp)
                    mult_exp += 1
                    mult_val -= 10**(- mult_exp)
                    print_stmnt += 'good solution = ' + str(y) + '\n'
                    init_vals[0] = mult_val * cont_init
                else:
                    break
                print_stmnt += 'mult_exp = ' + str(mult_exp) + '\n'
                print_stmnt += 'mult_val = ' + str(mult_val) + '\n'
            if (side == 'precursor'):
                self.x_precursor = x - x[-1]
                self.Mach_precursor = M
                if ('ED' in problem):
                    self.Pr_precursor = y
                if ('FLD' in problem):
                    self.Er_precursor = y
                    self.Lambda_precursor = Lambda_vals
                    self.R_precursor = R_vals
                elif ('ie' in problem):
                    self.Te_precursor = y
            else:
                self.x_relaxation = x - x[-1]
                self.Mach_relaxation = M
                if ('ED' in problem):
                    self.Pr_relaxation = y
                if ('FLD' in problem):
                    self.Er_relaxation = y
                    self.Lambda_relaxation = Lambda_vals
                    self.R_relaxation = R_vals
                elif ('ie' in problem):
                    self.Te_relaxation = y
                self.mult_vals.append(mult_val)
        print_stmnt += 'self.Mach_precursor = ' + str(self.Mach_precursor)+ '\n'
        print_stmnt += 'self.Mach_relaxation = ' + str(self.Mach_relaxation)
        print_stmnt += '\n'
        print_stmnt += 'leaving integrate_ddM' + '\n'

    def check_solns_overlap(self):
        print_stmnt = '\n'
        print_stmnt += 'entered check_solns_overlap' + '\n'
        if ('ED' in self.problem):
            _pre = self.Pr_precursor
            _rel = self.Pr_relaxation
            cont_fnctn = fnctn.rad_flux
        if ('FLD' in self.problem):
            _pre = self.Er_precursor
            _rel = self.Er_relaxation
            cont_fnctn = fnctn.rad_flux
        elif ('ie' in self.problem):
            _pre = self.Te_precursor
            _rel = self.Te_relaxation
            cont_fnctn = fnctn.Se
            print(f'_pre = {_pre}\n')
            print(f'_rel = {_rel}\n')
            print('\n')
        M_pre = self.Mach_precursor
        M_rel = self.Mach_relaxation
        if ('f' in self.__dict__):
          if (self.f_iters == 1):
            self._pre = _pre
            self._rel = _rel
            self.M_pre = M_pre
            self.M_rel = M_rel
        print('M_pre[-1] = ', M_pre[-1])
        print('M_rel[-1] = ', M_rel[-1])
        print('_pre[-1] = ', _pre[-1])
        print('_rel[-1] = ', _rel[-1])
        proper_overlap  = (_pre[-1] > _rel[-1] + self.int_tol)
        print('proper_overlap = ', proper_overlap)
        proper_cont_shock  = (numpy.abs(_rel[-1] - _pre[-1]) / _rel[-1]
                              < 400. * (self.eps_precursor_ASP_initially
                                     + self.eps_relaxation_ASP_initially))
        proper_cont_shock *= (M_pre[-1] - M_rel[-1]
                              < 10. * (self.eps_precursor_ASP_initially
                                     + self.eps_relaxation_ASP_initially))
        print('proper_cont_shock = ', proper_cont_shock)
        if proper_overlap:
            print_stmnt += 'seeing an embedded hydrodynamic shock\n'
            self.continuous_shock = 0
            left_flux  = cont_fnctn(_rel[-1],
                           numpy.interp(_rel[-1], _pre, M_pre), self)
            left_flux -= cont_fnctn(_rel[-1], M_rel[-1], self)
            right_flux  = cont_fnctn(_pre[-1], M_pre[-1], self)
            right_flux -= cont_fnctn(_pre[-1],
                                     numpy.interp(_pre[-1], _rel[::-1],
                                                  M_rel[::-1]), self)
            print('left_flux = ', left_flux)
            print('right_flux = ', right_flux)
            avg_flux = - (left_flux + right_flux) / 2.
            cont_val = scipy.optimize.bisect(lambda y:
                       cont_fnctn(y, numpy.interp(y, _pre, M_pre), self)
                     - cont_fnctn(y, numpy.interp(y, _rel[::-1], M_rel[::-1]),
                                  self)
                     , _rel[-1], _pre[-1])
            print('cont_val = ', cont_val)
            left1 = sum(_pre <= cont_val)
            right1 = sum(_rel >= cont_val)
            if (('ED' in self.problem) | ('FLD' in self.problem)):
                self.Pcont = cont_val
            elif ('ie' in self.problem):
                self.Tecont = cont_val
        elif proper_cont_shock:
            print_stmnt += 'seeing a continuous shock\n'
            self.continuous_shock = 1
            print_stmnt += 'seeing a continuous shock' + '\n'
            cont_val = numpy.interp(1., numpy.array([M_rel[-1], M_pre[-1]]), \
                                        numpy.array([_rel[-1], _pre[-1]]))
            _pre[-1] = cont_val
            _rel[-1] = cont_val
            left1 = sum(_pre <= cont_val)
            right1 = sum(_rel >= cont_val)
            if (('ED' in self.problem) | ('FLD' in self.problem)):
                self.Pcont = cont_val
            elif ('ie' in self.problem):
                self.Tecont = cont_val
            self.eps_precursor_ASP = self.eps_precursor_ASP_hold
            self.eps_relaxation_ASP = self.eps_relaxation_ASP_hold
        else:
          left1 = len(_pre)
          right1 = len(_rel)
          self.eps_precursor_ASP /= 2.
          self.eps_relaxation_ASP /= 2.
          self.eps_relaxation_ASP = max(self.eps_relaxation_ASP, self.eps_relaxation_ASP_initially)
          self.integrate_ddM_again += 2
        if self.make_RT_solution_bool:
            self.left1 = left1
            self.right1 = right1
            print_stmnt += 'self.left1 = ' + str(self.left1) + '\n'
            print_stmnt += 'self.right1 = ' + str(self.right1) + '\n'
            if ((left1 == self.left_pts) | (right1 == self.right_pts) |
                (self.continuous_shock == 1)):
                self.integrate_ddM_again = 0
                print_stmnt += 'self.left1 = ' + str(left1) + '\n'
                print_stmnt += 'self.right1 = ' + str(right1) + '\n'
            if (proper_overlap | proper_cont_shock):
              print_stmnt += 'cont_val = ' + str(cont_val) + '\n'
              self.eps_precursor_ASP = M_pre[:left1 + 2][-1] - 1.
              self.eps_relaxation_ASP = 1. - M_rel[:right1 + 2][-1]
              if (self.integrate_ddM_again == 2):
                  self.eps_precursor_ASP = self.eps_precursor_ASP_hold
                  self.eps_relaxation_ASP = self.eps_relaxation_ASP_hold
            print_stmnt += 'self.eps_precursor_ASP = '
            print_stmnt += str(self.eps_precursor_ASP) + '\n'
            print_stmnt += 'self.eps_relaxation_ASP = '
            print_stmnt += str(self.eps_relaxation_ASP) + '\n'
            print_stmnt += 'leaving check_solns_overlap' + '\n'

    def splice_precursor_and_relaxation(self):
        print_stmnt = '\n'
        print_stmnt += 'entered splice_precursor_and_relaxation' + '\n'
        if ('ED' in self.problem):
            cont_val = self.Pcont
            _pre = self.Pr_precursor
            _rel = self.Pr_relaxation[::-1]
        if ('FLD' in self.problem):
            cont_val = self.Pcont
            _pre = self.Er_precursor
            _rel = self.Er_relaxation[::-1]
        elif ('ie' in self.problem):
            cont_val = self.Tecont
            _pre = self.Te_precursor
            _rel = self.Te_relaxation[::-1]
        Mach_precursor_last = numpy.interp(cont_val, _pre, self.Mach_precursor)
        Mach_relaxation_last = numpy.interp(cont_val, _rel,
                                            self.Mach_relaxation[::-1])
        x_precursor_last = numpy.interp(cont_val, _pre, self.x_precursor)
        x_relaxation_last = numpy.interp(cont_val, _rel,
                                         self.x_relaxation[::-1])
        self.Mach_precursor = self.Mach_precursor[:self.left1]
        self.Mach_relaxation = self.Mach_relaxation[:self.right1]
        self.Mach_precursor[-1] = Mach_precursor_last
        self.Mach_relaxation[-1] = Mach_relaxation_last
        self.x_precursor = self.x_precursor[:self.left1] - x_precursor_last
        self.x_relaxation = self.x_relaxation[:self.right1] - x_relaxation_last
        self.x_precursor[-1] = 0.
        self.x_relaxation[-1] = 0.
        self.Mach = numpy.append(
                    numpy.append(self.M0, self.Mach_precursor),
                    numpy.append(self.Mach_relaxation[::-1], self.M1))
        x0 = 5. * self.x_precursor[0]
        x1 = max(numpy.abs(x0), 5. * self.x_relaxation[0])
        x0 = - max(numpy.abs(x0), x1)
        self.x = numpy.append(numpy.append(x0, self.x_precursor),
                              numpy.append(self.x_relaxation[::-1], x1))
        ASP_multiplier = 1. / 10.
        if ('ED' in self.problem):
            self.Pr_precursor = self.Pr_precursor[:self.left1]
            self.Pr_relaxation = self.Pr_relaxation[:self.right1]
            self.Pr_precursor[-1] = cont_val
            self.Pr_relaxation[-1] = cont_val
            self.Pr = numpy.append(numpy.append(self.Pr0, self.Pr_precursor),
                                   numpy.append(self.Pr_relaxation[::-1],
                                                self.Pr1))
            self.Tm = fnctn.mat_temp(self.Pr, self.Mach, self)
            self.Er = fnctn.rad_energy_density(self.Pr, self.Mach, self)
            self.Tr = fnctn.rad_temp(self.Pr, self.Mach, self)
            self.Speed = fnctn.mat_speed(self.Pr, self.Mach, self)
            self.Density = fnctn.mat_density(self.Pr, self.Mach, self)
            self.Pressure = fnctn.mat_pres(self.Pr, self.Mach, self)
            self.Fr = fnctn.rad_flux(self.Pr, self.Mach, self)
            if ('f_iters' in self.__dict__):
                ASP_multiplier = 1. / 10.
        elif ('FLD' in self.problem):
            self.Er_precursor = self.Er_precursor[:self.left1]
            self.Er_relaxation = self.Er_relaxation[:self.right1]
            self.Er_precursor[-1] = cont_val
            self.Er_relaxation[-1] = cont_val
            self.Er = numpy.append(numpy.append(self.Er0, self.Er_precursor),
                                   numpy.append(self.Er_relaxation[::-1],
                                                self.Er1))
            print('self.continuous_shock = ', self.continuous_shock)
            self.Lambda_precursor = self.Lambda_precursor[:self.left1]
            self.R_precursor = self.R_precursor[:self.left1]
            self.Lambda_relaxation = self.Lambda_relaxation[:self.right1]
            self.R_relaxation = self.R_relaxation[:self.right1]
            if self.continuous_shock:
              self.Lambda_precursor.append(self.Lambda_precursor[-1])
              self.R_precursor.append(self.R_precursor[-1])
              self.Lambda_relaxation.append(self.Lambda_relaxation[-1])
              self.R_relaxation.append(self.R_relaxation[-1])
            fnctn.dEdx(cont_val, Mach_precursor_last, self)
            self.Lambda_precursor[-1] = self.Lambda
            self.R_precursor[-1] = self.R
            fnctn.dEdx(cont_val, Mach_relaxation_last, self)
            self.Lambda_relaxation[-1] = self.Lambda
            self.R_relaxation[-1] = self.R
            self.Lambda = numpy.append(
                                     numpy.append(1./3.,
                                                  self.Lambda_precursor),
                                     numpy.append(self.Lambda_relaxation[::-1],
                                                  1./3.))
            self.R = numpy.append(numpy.append(0., self.R_precursor),
                                  numpy.append(self.R_relaxation[::-1], 0.))
            self.Tm = fnctn.mat_temp(self.Er, self.Mach, self)
            self.Pr = (self.Lambda + (self.Lambda * self.R)**2) * self.Er
            self.Tr = fnctn.rad_temp(self.Er, self.Mach, self)
            self.Speed = fnctn.mat_speed(self.Er, self.Mach, self)
            self.Density = fnctn.mat_density(self.Er, self.Mach, self)
            self.Pressure = fnctn.mat_pres(self.Er, self.Mach, self)
            self.Fr = fnctn.rad_flux(self.Er, self.Mach, self)
        elif ('ie' in self.problem):
            self.Te_precursor = self.Te_precursor[:self.left1]
            self.Te_relaxation = self.Te_relaxation[:self.right1]
            self.Te_precursor[-1] = self.Tecont
            self.Te_relaxation[-1] = self.Tecont
            self.Te = numpy.append(numpy.append(self.Te0, self.Te_precursor),
                                   numpy.append(self.Te_relaxation[::-1],
                                                self.T1))
            self.Ti = fnctn.ion_temp(self.Te, self.Mach, self)
            self.Tm = fnctn.mat_temp(self.Te, self.Mach, self)
            self.Density = fnctn.mat_density(self.Te, self.Mach, self)
            self.Speed = self.M0 / self.Density
            self.Pressure = self.Density * self.Tm / self.gamma
            Fe  = - fnctn.kappa_e(self.Tm, self.Density, self)
            Fe *= fnctn.dTedx(self.Te, self.Mach, self)
            self.Fe = Fe
        self.SIE = self.Tm / self.gamma / (self.gamma - 1.)
        self.eps_precursor_ASP  = Mach_precursor_last - 1.
        self.eps_precursor_ASP -= self.Mach_precursor_baseline
        self.eps_precursor_ASP *= ASP_multiplier
        self.eps_precursor_ASP += self.Mach_precursor_baseline
        self.eps_relaxation_ASP  = 1. - Mach_relaxation_last
        self.eps_relaxation_ASP -= self.Mach_relaxation_baseline
        self.eps_relaxation_ASP *= ASP_multiplier
        self.eps_relaxation_ASP += self.Mach_relaxation_baseline
        self.eps_relaxation_ASP = self.eps_relaxation_ASP_initially
        print_stmnt += 'leaving splice_precursor_and_relaxation' + '\n'

    def plot_TiTe(self):
        x = self.x
        Ti = self.Ti
        fig = matplotlib.pyplot.figure()
        ax1 = fig.add_subplot(111)
        Ti_max = numpy.max(Ti)
        delta_T = (Ti_max - 1.) / 100.
        ax1.plot(x, Ti, 'r', label = r'Ti')
        ax1.plot(x, self.Te, 'k', label = r'Te')
        x0 = x[numpy.argmin(numpy.abs(Ti - (1. + delta_T / 100.)))]
        x1 = Ti - (Ti[-1] + numpy.sign(Ti[-2] - Ti[-1]) * delta_T / 100.)
        x1 = x[numpy.argmin(numpy.abs(x1))]
        if (x1 < 0.4 * numpy.abs(x0)):
          x1 = 0.4 * numpy.abs(x0)
        matplotlib.pyplot.xlim((x0, x1))
        matplotlib.pyplot.ylim((1. - delta_T, Ti_max + delta_T))
        matplotlib.pyplot.legend(loc = 'upper left')
        matplotlib.pyplot.show()

    def plot_Tetas(self):
        etas = self.rho0 / self.Density
        Tm = self.Tm
        Ti = self.Ti
        Te = self.Te
        fig = matplotlib.pyplot.figure()
        ax1 = fig.add_subplot(111)
        Ti_max = numpy.max(Ti)
        delta_T = (Ti_max - 1.) / 100.
        ax1.plot(etas, Ti, 'o', label = r'Ti')
        ax1.plot(etas, Tm, 'o', label = r'Tm')
        ax1.plot(etas, fnctn.TeC2(etas, Te, self), 'o', label = r'TeC2')
        ax1.plot(etas, Te, label = r'Te')
        ax1.plot(etas, fnctn.TeC1(etas, Te, self), label = r'TeC1')
        matplotlib.pyplot.xlim((etas[-1], etas[0]))
        matplotlib.pyplot.ylim((1. - delta_T, Ti_max + delta_T))
        matplotlib.pyplot.legend(loc = 'upper right')
        matplotlib.pyplot.show()

class ED_ShockProfiles(RadShockProfile):
    '''
    '''
    def __init__(self, incoming, **kwargs):
        super(ED_ShockProfiles, self).__init__(incoming)
        print_stmnt  = '\n'
        print_stmnt += 'entered AnalyticShockProfiles __init__ for M0 = '
        print_stmnt += str(incoming.M0) + '\n'
        self.P0 = incoming.P0
        self.C0 = incoming.C0
        self.sigA = incoming.sigA
        self.sigS = incoming.sigS
        self.expDensity_abs = incoming.expDensity_abs
        self.expTemp_abs = incoming.expTemp_abs
        self.expDensity_scat = incoming.expDensity_scat
        self.expTemp_scat = incoming.expTemp_scat
        try:
            import fnctn_ED as fnctn
        except ImportError:
            from exactpack.solvers.radshocks import fnctn_ED as fnctn
        if (self.problem == 'ED'):
            try:
                import fnctn_ED as fnctn
            except ImportError:
                from exactpack.solvers.radshocks import fnctn_ED as fnctn
        elif ('nED' in self.problem):
            try:
                import fnctn_nED as fnctn
            except ImportError:
                from exactpack.solvers.radshocks import fnctn_nED as fnctn
        elif ('FLD' in self.problem):
            try:
                import fnctn_FLD as fnctn
            except ImportError:
                from exactpack.solvers.radshocks import fnctn_FLD as fnctn
            self.use_jac = False
            self.Lambda = 1./3.
            self.R = 0.
            self.FLD_type = 'LP'
            if (('_1' in self.problem) | ('_2' in self.problem)):
                self.FLD_type = self.problem[-1]
                self.use_jac = True
            elif ('poly' in self.problem):
                self.FLD_type = 'poly'
                self.use_jac = True
        global fnctn
        print_stmnt += 'leaving AnalyticShockProfiles __init__ for M0 = '
        print_stmnt += str(incoming.M0)
        print_stmnt += '\n'

    def make_ED_solution(self):
        print_stmnt  = '\n'
        print_stmnt += 'entered make_ED_solution\n'
        M0 = self.M0
        Ts = numpy.linspace(1. + self.eps_precursor_equil,
                            self.T1 - self.eps_relaxation_equil,
                            int(1000. * self.left_pts))
        print('use_jac = ', self.use_jac)
        if (self.use_jac):
          xs = scipy.integrate.odeint(fnctn.dxdT, 0., Ts, args=(self,),
                                      Dfun = fnctn.dxdT_jac)
        else:
          xs = scipy.integrate.odeint(fnctn.dxdT, 0., Ts, args=(self,))
        xs = xs[:, 0]
        xs = numpy.append(- 5. * xs[-1], xs)
        xs = numpy.append(xs, 5. * xs[-1])
        Ts = numpy.append(1., Ts)
        Ts = numpy.append(Ts, self.T1)
        rhos = fnctn.rho(Ts, self)
        rhos[-1] = self.rho1
        speed = M0 / rhos
        Fr  = - 4. * Ts**3 / 3. / fnctn.sigma_t(Ts, self)
        Fr /= fnctn.dxdT(0., Ts, self)
        Fr += 4. / 3. * speed / self.C0 * Ts**4
        Ms = M0 / rhos / numpy.sqrt(Ts)
        Ms[-1] = self.M1
        x_shift = numpy.interp(1., Ms[::-1], xs[::-1])
        if (self.M1 < numpy.sqrt(1. / self.gamma)):
            x_shift = xs[-2]
        xs -= x_shift
        M_precursor = numpy.where(Ms >= 1., Ms, 0.)
        M_relaxation = numpy.where(Ms < 1., Ms, 0.)
        num_precursor_zeros = sum(M_precursor == 0.)
        num_relaxation_zeros = sum(M_relaxation == 0.)
        M_precursor = numpy.delete(M_precursor,
                                   numpy.s_[-num_precursor_zeros:])
        M_relaxation = numpy.delete(M_relaxation,
                                    numpy.s_[:num_relaxation_zeros])
        self.x = xs
        self.Tm = Ts
        self.Fr = Fr
        self.Mach = Ms
        self.Mach_precursor = M_precursor
        self.Mach_relaxation = M_relaxation
        self.Density = rhos
        self.Speed = speed
        self.Pressure = rhos * Ts / self.gamma
        self.SIE = Ts / self.gamma / (self.gamma - 1.)
        print_stmnt += 'leaving make_ED_solution\n'


class nED_ShockProfiles(ED_ShockProfiles, ShockMethods_2T):
    '''
    '''
    def __init__(self, incoming):
        super(nED_ShockProfiles, self).__init__(incoming)
        print_stmnt  = '\n'
        print_stmnt += 'entered AnalyticShockProfiles __init__ for M0 = '
        print_stmnt += str(self.M0) + '\n'
        self.right_pts = 2. * self.num_pts + 1
        self.eps_precursor_ASP = copy.deepcopy(incoming.eps_precursor_ASP)
        self.eps_precursor_ASP_hold = copy.deepcopy(incoming.eps_precursor_ASP)
        self.eps_precursor_ASP_initially = copy.deepcopy(incoming.eps_precursor_ASP_initially)
        self.eps_relaxation_ASP = copy.deepcopy(incoming.eps_relaxation_ASP)
        self.eps_relaxation_ASP_hold = copy.deepcopy(incoming.eps_relaxation_ASP)
        self.eps_relaxation_ASP_initially = copy.deepcopy(incoming.eps_relaxation_ASP_initially)
        self.make_RT_solution_bool = 1
        self.Mach_precursor_baseline = 0.
        self.Mach_relaxation_baseline = 0.
        self.left1 = 0
        self.right1 = 0
        self.continuous_shock = 1
        self.epsilon = copy.deepcopy(incoming.epsilon)
        self.sigS *= self.epsilon
        if ('ED' in self.problem):
          self.Pr0 = self.T0**4 / 3.
          self.Pr_init = numpy.zeros(2)
        elif ('FLD' in self.problem):
          self.Pr0 = self.T0**4 / 3.
          self.Er0 = self.T0**4
          self.Er_init = numpy.zeros(2)
          self.Lambda = 1./3.
          self.R = 0.
        self.mult_vals = []
        print_stmnt += 'leaving AnalyticShockProfiles __init__ for M0 = '
        print_stmnt += str(self.M0)
        print_stmnt += '\n'


class Sn_ShockProfiles(nED_ShockProfiles):
    '''
    '''
    def __init__(self, incoming):
        super(Sn_ShockProfiles, self).__init__(incoming)
        print_stmnt  = '\n'
        print_stmnt += 'entered AnalyticShockProfiles __init__ for M0 = '
        print_stmnt += str(self.M0) + '\n'
        incoming_list = incoming.__dict__.keys()
        self_list = self.__dict__.keys()
        get_list = list(set(incoming_list) - set(self_list))
        for key in get_list:
            exec('self.' + key + ' = copy.deepcopy(incoming.' + key + ')')
        self.dxdM_relax_negative = False
        print_stmnt += 'leaving AnalyticShockProfiles __init__ for M0 = '
        print_stmnt += str(self.M0)
        print_stmnt += '\n'

    def fill_in_xs(self):
        print_stmnt = '\n'
        print_stmnt += 'entered fill_in_xs' + '\n'
        M0 = self.M0
        M02 = M0 * M0
        x0 = self.x[0]
        x1 = self.x[-1] * M0
        self.x0 = x0
        self.x1 = x1
        left_pts = self.left_pts
        right_pts = self.right_pts
        x_precursor = self.x_precursor
        x_relaxation = self.x_relaxation
        x_linfill_left = numpy.linspace(x0, x_precursor[9],
                                        int(left_pts), endpoint=True)
        x_logfill_left = numpy.logspace(numpy.log10(numpy.abs(x0)),
                                        numpy.log10(x_precursor[9]
                                                    + 100. * numpy.abs(x0)),
                                        int(left_pts), endpoint=True)
        x_logfill_left_shock = ((x_logfill_left - x_logfill_left[0])
                              / (x_logfill_left - x_logfill_left[0])[-1]
                              * (x_precursor[-10] - x_precursor[-1])
                              + x_precursor[-1])[::-1]
        x_logfill_left_equil = ((x_logfill_left - x_logfill_left[0])
                              / (x_logfill_left - x_logfill_left[0])[-1]
                              * (- x0 + x_precursor[9]) + x0)
        x_left_RT = numpy.append(
                    numpy.append(
                    numpy.append(x_logfill_left_equil,
                                 x_logfill_left_shock),
                                 x_linfill_left),
                                 x_precursor)
        x_linfill_right = numpy.linspace(x1, x_relaxation[9],
                                         int(M02 * right_pts), endpoint=True)
        x_logfill_right = numpy.logspace(numpy.log10(x1),
                                         numpy.log10(100. * x1),
                                         int(M02 * right_pts),
                                         endpoint=True)[::-1]
        x_logfill_right_shock = ((x_logfill_right[-1] - x_logfill_right)
                               / (x_logfill_right[-1] - x_logfill_right)[0]
                               * (x_relaxation[-10] - x_relaxation[-1])
                               + x_relaxation[-1])
        x_logfill_right_equil = ((x_logfill_right[-1] - x_logfill_right)
                               / (x_logfill_right[-1] - x_logfill_right)[0]
                               * (x_relaxation[9] - x1) + x1)[::-1]
        x_right_RT = numpy.append(
                     numpy.append(
                     numpy.append(x_logfill_right_equil,
                                  x_logfill_right_shock),
                                  x_linfill_right),
                                  x_relaxation)
        x_left_RT.sort()
        x_right_RT.sort()
        x_left_RT = x_left_RT[1:-1]
        x_right_RT = x_right_RT[1:-1]
        x_RT = numpy.append(x_left_RT, x_right_RT)
        x_RT.sort()
        self.x_RT = x_RT
        self.d_x.append(x_RT)
        print_stmnt += 'self.x_RT = ' + str(self.x_RT) + '\n'
        x_RT_zero = numpy.argmin(numpy.abs(x_RT))
        Mach_RT = numpy.interp(x_RT, self.x, self.Mach)
        Mach_RT[x_RT_zero] = self.Mach_precursor[-1]
        Mach_RT[x_RT_zero + 1] = self.Mach_relaxation[-1]
        self.Mach_RT = Mach_RT
        self.Pr_RT = numpy.interp(Mach_RT, self.Mach[::-1], self.Pr[::-1])
        Pr_RT = self.Pr_RT
        self.d_Prh.append(Pr_RT)
        Er = fnctn.rad_energy_density(Pr_RT, Mach_RT, self)
        self.d_Erh.append(Er)
        Fr = fnctn.rad_flux(Pr_RT, Mach_RT, self)
        self.d_Frh.append(Fr)
        Ip = numpy.argmin(numpy.abs(x_RT))
        if (Mach_RT[Ip] < 1.):
            Ip -= 1
        Is = Ip + 1
        Tm = fnctn.mat_temp(Pr_RT, Mach_RT, self)
        Tm_p = Tm[Ip]
        Tm_s = Tm[Is]
        Tm_max = numpy.max(Tm)
        Tr = fnctn.rad_temp(Pr_RT, Mach_RT, self)
        Tr_cont = Tr[Ip]
        Tr_max = numpy.max(Tr)
        xFWHM = 0
        Tf = self.T1
        if (Tm_max > Tf + self.int_tol):
            Tm_half = (Tm_max + Tf) / 2.
            Im = numpy.argmin(numpy.abs(Tm - Tm_max))
            If = numpy.argmin(numpy.abs(Tm - Tm_half))
            if (Tm_half < Tm_s):
                Ileft = Ip
                Iright = If
                Iright = min(Iright, len(x_RT) - 1)
            else:
                Ileft = numpy.argmin(numpy.abs(Tm[Is:Im] - Tm_half)) + Is
                Iright = numpy.argmin(numpy.abs(Tm[Im:-1] - Tm_half)) + Im
                Iright = min(Iright, len(x_RT) - 1)
            xFWHM = x_RT[Iright] - x_RT[Ileft]
        print_stmnt += 'leaving fill_in_xs' + '\n'

    def make_Ims(self):
        print_stmnt = '\n'
        print_stmnt += 'entered make_Ims' + '\n'
        pi  = numpy.pi
        Sn  = self.Sn
        M0  = self.M0
        beta0 = M0 / self.C0 * self.epsilon
        beta1 = beta0 / self.rho1 * self.epsilon
        T04 = 1.
        Tf  = M0 / self.M1 / self.rho1
        Tf *= Tf
        Tf4 = Tf * Tf * Tf * Tf
        x_RT_size = numpy.size(self.x_RT)
        if (self.f_iters == 0):
            self.mus, self.weights = numpy.polynomial.legendre.leggauss(Sn)
        Im = numpy.zeros((Sn, x_RT_size))
        mus = self.mus
        weights = self.weights
        Im[:, 0] = T04 / 4. / pi * (1. + 4. * mus * beta0)
        Im[:,-1] = Tf4 / 4. / pi * (1. + 4 * mus * beta1)
        self.Im = Im
        self.mus = mus
        self.weights = weights
        print('\n')
        print('Im[:, 0] - 1. / (4 * pi) = ')
        print(Im[:, 0] - 1. / 4 / numpy.pi)
        print('\n')
        print('Im[:, -1] - Tf4 / (4 * pi) = ')
        print(Im[:, -1] - Tf4 / 4 / numpy.pi)
        print('\n')
        print_stmnt += 'leaving make_Ims' + '\n'

    def linearize_transport_equation(self):
        print_stmnt = '\n'
        print_stmnt += 'entered linearize_transport_equation' + '\n'
        x = self.x_RT
        Sn = self.Sn
        mus = self.mus
        for mu_j, self.mu in enumerate(mus[:Sn]):
            sgn = int(numpy.sign(self.mu))
            ii=0 if (sgn > 0) else -1
            print_stmnt += 'self.mu = ' + str(self.mu) + '\n'
            eps = 1.e-4 * sgn
            init_val = self.Im[mu_j, ii]
            x1 = x[sgn]
            diff = x1 - x[ii]
            val = scipy.optimize.fsolve(
                  lambda I:
                  I - init_val
                    - diff * fnctn.DIm_Dx_ode(x1, I, self) / self.epsilon,
                  init_val + eps, xtol = 1.e-13)[0]
            self.Im[mu_j, ii + 1 * sgn] = val
            print_stmnt += 'init_val = ' + str(init_val) + '\n'
            print_stmnt += 'val =      ' + str(val) + '\n'
            print('\n')
            print('mu_j = ', mu_j)
            print('init_val = ', init_val)
            print('val =      ', val)
            print('\n')
        print_stmnt += 'leaving linearize_transport_equation' + '\n'

    def Lhopital_near_equilibrium(self):
        print_stmnt  = '\n'

    def integrate_Sn(self):
        print_stmnt = '\n'
        print_stmnt += 'entered integrate_Sn' + '\n'
        print_stmnt += 'Sn = ' + str(self.Sn) + '\n'
        pi = numpy.pi
        Im = self.Im
        C0 = self.C0
        T0 = self.T0
        T04 = T0 * T0 * T0 * T0
        u0 = self.M0 * self.epsilon
        T1 = self.T1
        T14 = T1 * T1 * T1 * T1
        u1 = self.speed1 * self.epsilon
        i = scipy.integrate.ode(fnctn.DIm_Dx_ode, fnctn.DIm_Dx_jac)
        i.set_integrator('vode', atol=self.int_tol,
                         rtol=self.int_tol, method='bdf',
                         nsteps=self.max_steps)
        i.set_f_params(self)
        i.set_jac_params(self)
        mus = self.mus
        x_RT_fwd = self.x_RT[1:]
        x_RT_rev = self.x_RT[::-1][1:]
        result_array = numpy.zeros(len(x_RT_fwd))
        integration_array_size = numpy.size(x_RT_fwd)
        absval0_keep = 0
        absval1_keep = 0
        for mu_j, mu in enumerate(mus):
            sgn = numpy.sign(mu)
            self.mu = mu
            print('mu_j = ', mu_j)
            print_stmnt += 'i = ' + str(mu_j) + ' and mu = ' + str(mu) + '\n'
            [ii, ia] = [1,x_RT_fwd] if (sgn > 0) else [-2,x_RT_rev]
            integration_array = ia
            integration_array_diffed = numpy.diff(integration_array)
            i.set_initial_value(Im[mu_j,ii], integration_array[0])
            result_array[0] = Im[mu_j,ii]
            k = 1
            while (i.successful() & (k < integration_array_size)):
                dt = integration_array_diffed[k - 1]
                i.integrate(i.t + dt)
                result_array[k] = i.y[0]
                k += 1
            if (sgn > 0):
                self.Im[mu_j,1:] = result_array
            else:
                self.Im[mu_j,:-1] = result_array[::-1]
            print_stmnt += 'self.Im[mu_j, :]  = ' + str(self.Im[mu_j, :]) + '\n'
            print_stmnt += 'self.Im[mu_j, 1]  = ' + str(self.Im[mu_j, 1]) + '\n'
            print_stmnt += 'self.Im[mu_j, 0]  = ' + str(self.Im[mu_j, 0]) + '\n'
            print('\n')
            val0 = (1. + 4 * mu * u0 / C0) * T04 / 4. / pi
            val1 = (1. + 4 * mu * u1 / C0) * T14 / 4. / pi
            print('Im[mu_j, 0] - val0 = ', Im[mu_j, 0] - val0)
            print('Im[mu_j, -1] - val1 = ', Im[mu_j, -1] - val1)
            print('\n')
            print_stmnt += 'first elem should  = ' + str(val0) + '\n'
            absval0 = abs(val0 - self.Im[mu_j, 0]) / val0
            print_stmnt += 'first elem |rel diff|  = ' + str(absval0) + '\n'
            print_stmnt += 'self.Im[mu_j,-1] = ' + str(self.Im[mu_j,-1]) + '\n'
            print_stmnt += 'last elem should   = ' + str(val1) + '\n'
            absval1 = abs(val1 - self.Im[mu_j, -1]) / val1
            print_stmnt += 'last elem |rel diff|   = ' + str(absval1) + '\n'
        print_stmnt += 'absval0_keep = ' + str(absval0_keep) + '\n'
        print_stmnt += 'absval1_keep = ' + str(absval1_keep) + '\n'
        print_stmnt += '\n'
        print_stmnt += 'leaving integrate_Sn' + '\n'

    def Sn_angular_moments(self):
        print_stmnt = '\n'
        print_stmnt += 'entered Sn_angular_moments' + '\n'
        x_size = numpy.size(self.x_RT)
        E_RT = numpy.zeros(x_size)
        F_RT = numpy.zeros(x_size)
        P_RT = numpy.zeros(x_size)
        pi = numpy.pi
        Sn = self.Sn
        mus = self.mus
        mus2 = mus * mus
        Im = self.Im
        weights = self.weights
        for i in range(Sn):
            e_rt = weights[i] * Im[i,:]
            E_RT += e_rt
            F_RT += e_rt * mus[i]
            P_RT += e_rt * mus2[i]
        E_RT *= 2. * pi
        F_RT *= 2. * pi
        P_RT *= 2. * pi
        f = P_RT / E_RT
        self.E_RT = E_RT
        self.F_RT = F_RT
        self.P_RT = P_RT
        self.f = f
        x_RT = self.x_RT
        x = numpy.delete(x_RT, numpy.argmin(numpy.abs(x_RT)))
        E_RT = numpy.delete(E_RT, numpy.argmin(numpy.abs(x_RT)))
        x = numpy.append(x, 2. * x[-1] - x[-2])
        x = numpy.append(x, 2. * x[0] - x[1])
        T0 = self.T0
        T04 = T0 * T0 * T0 * T0
        T1 = self.T1
        T14 = T1 * T1 * T1 * T1
        M0 = self.M0
        C0 = self.C0
        Pr0 = self.Pr0
        Pr1 = self.Pr1
        print_stmnt += 'upstream equilibrium radiation values:\n'
        print_stmnt += 'Er        = ' + str(E_RT[0]) + '\n'
        print_stmnt += 'should    = ' + str(T04) + '\n'
        val  = numpy.abs(E_RT[0] - T04)
        val /= min(E_RT[0], T04)
        print_stmnt += '|rel err| = ' + str(val) + '\n'
        print_stmnt += 'Fr        = ' + str(F_RT[0]) + '\n'
        Fval = 4. / 3. * M0 / C0 * T04
        print_stmnt += 'should    = ' + str(Fval) + '\n'
        val = numpy.abs(F_RT[0] - Fval) / min(F_RT[0], Fval)
        print_stmnt += '|rel err| = ' + str(numpy.abs(val)) + '\n'
        print_stmnt += 'Pr        = ' + str(P_RT[0]) + '\n'
        print_stmnt += 'should    = ' + str(1./3.) + '\n'
        val = numpy.abs(P_RT[0] - Pr0) / min(P_RT[0], Pr0)
        print_stmnt += '|rel err| = ' + str(val) + '\n'
        print_stmnt += 'f         = ' + str(f[0]) + '\n'
        print_stmnt += 'should    = ' + str(1./3.) + '\n'
        val = numpy.abs(f[0] - 1./3.) / min(f[0], 1./3.)
        print_stmnt += '|rel err| = ' + str(val) + '\n'
        print_stmnt += '\n'
        print_stmnt += 'downstream equilibrium radiation values:\n'
        print_stmnt += 'Er        = ' + str(E_RT[-1]) + '\n'
        print_stmnt += 'should    = ' + str(T14) + '\n'
        val  = numpy.abs(E_RT[-1] - T14)
        val /= min(E_RT[-1], T14)
        print_stmnt += '|rel err| = ' + str(val) + '\n'
        print_stmnt += 'Fr        = ' + str(F_RT[-1]) + '\n'
        Fval = 4. / 3. * M0 / self.rho1 / C0 * T14
        print_stmnt += 'should    = ' + str(Fval) + '\n'
        val = numpy.abs(F_RT[-1] - Fval) / min(F_RT[-1], Fval)
        print_stmnt += '|rel err| = ' + str(numpy.abs(val)) + '\n'
        print_stmnt += 'Pr        = ' + str(P_RT[-1]) + '\n'
        print_stmnt += 'should    = ' + str(T14 / 3.) + '\n'
        val = numpy.abs(P_RT[-1] - Pr1) / min(P_RT[-1], Pr1)
        print_stmnt += '|rel err| = ' + str(val) + '\n'
        print_stmnt += 'f         = ' + str(f[-1]) + '\n'
        print_stmnt += 'should    = ' + str(1./3.) + '\n'
        val = numpy.abs(f[-1] - 1./3.) / min(f[-1], 1./3.)
        print_stmnt += '|rel err| = ' + str(val) + '\n'
        print_stmnt += 'leaving Sn_angular_moments' + '\n'

    def make_RT_solution(self):
        print_stmnt  = '\n'
        print_stmnt += 'entered make_RT_solution\n'
        if ('f_iters' not in self.__dict__):
            self.make_dictionaries()
        self.fill_in_xs()
        self.make_Ims()
        self.linearize_transport_equation()
        self.integrate_Sn()
        self.Sn_angular_moments()
        self.update_dictionaries()
        self.compute_errors()
        print_stmnt += 'leaving make_RT_solution\n'

    def make_dictionaries(self):
        print_stmnt = '\n'
        print_stmnt += 'entered make_dictionaries' + '\n'
        self.f_iters = 0
        self.d_f = []
        self.d_x = []
        self.d_M = []
        self.d_Erh = []
        self.d_Frh = []
        self.d_Prh = []
        self.d_Ert = []
        self.d_Frt = []
        self.d_Prt = []
        self.d_Im = []
        self.d_errs = []
        self.f_err = []
        self.f_L1 = []
        self.f_L2 = []
        self.Er_err = []
        self.Er_L1 = []
        self.Er_L2 = []
        self.Fr_err = []
        self.Fr_L1 = []
        self.Fr_L2 = []
        self.Pr_err = []
        self.Pr_L1 = []
        self.Pr_L2 = []
        self.Im_err = []
        self.Machs_to_add = []
        print_stmnt += 'leaving make_dictionaries' + '\n'

    def update_dictionaries(self):
        print_stmnt = '\n'
        print_stmnt += 'entered update_dictionaries' + '\n'
        self.d_M.append(self.Mach_RT)
        self.d_f.append(self.f)
        self.d_Prt.append(self.P_RT)
        self.d_Ert.append(self.E_RT)
        self.d_Frt.append(self.F_RT)
        self.d_Im.append(self.Im)
        self.d_errs.append([self.f_err,  self.f_L1,  self.f_L2,
                            self.Er_err, self.Er_L1, self.Er_L2,
                            self.Fr_err, self.Fr_L1, self.Fr_L2,
                            self.Pr_err, self.Pr_L1, self.Pr_L2])
        print_stmnt += 'leaving update_dictionaries' + '\n'

    def compute_errors(self):
        print_stmnt = '\n'
        print_stmnt += 'entered compute_errors' + '\n'
        f = self.f
        x = self.x_RT
        Sn = self.Sn
        Erh = self.d_Erh[-1]
        Frh = self.d_Frh[-1]
        Prh = self.Pr_RT
        Ert = self.E_RT
        Frt = self.F_RT
        Prt = self.P_RT
        Im = self.Im
        f_iters = self.f_iters
        Im_err = []
        bool_Im_err = 0
        if (f_iters == 0):
            f_OldOnNewx = numpy.ones(numpy.shape(f)) / 3.
            for i in range(Sn):
                Im_err.append(numpy.max(Im[i]))
        else:
            f_OldOnNewx = numpy.interp(
                x, self.d_x[f_iters - 1], self.d_f[f_iters - 1])
            for i in range(Sn):
                Im_OldOnNewx = numpy.interp(
                    x, self.d_x[f_iters - 1], self.d_Im[f_iters - 1][i])
                val = numpy.abs(Im[i] - Im_OldOnNewx) / Im[i]
                Im_err.append(numpy.max(val))
            print_stmnt += 'Im_err = ' + str(Im_err) + '\n'
        print_stmnt += 'Im_err = ' + str(Im_err) + '\n'
        f_err = numpy.max(numpy.abs(f - f_OldOnNewx) / f)
        val1  = (f[2:-1] + f[1:-2]) / 2.
        val1 -= (f_OldOnNewx[2:-1] + f_OldOnNewx[1:-2]) / 2.
        val2  = val1 * val1
        val1 *= x[2:-1] - x[1:-2] 
        val2 *= x[2:-1] - x[1:-2]
        f_L1  = numpy.sum(numpy.abs(val1)) / (x[-1] - x[0])
        f_L2  = numpy.sqrt(numpy.sum(val2)) / (x[-1] - x[0])
        Er_err = numpy.max(numpy.abs(Erh - Ert) / Erh)
        val1  = (Erh[2:-1] + Erh[1:-2]) / 2. - (Ert[2:-1] + Ert[1:-2]) / 2.
        val2  = val1 * val1
        val1 *= (x[2:-1] - x[1:-2])
        val2 *= (x[2:-1] - x[1:-2])
        Er_L1  = numpy.sum(numpy.abs(val1)) / (x[-1] - x[0])
        Er_L2  = numpy.sqrt(numpy.sum(val2)) / (x[-1] - x[0])
        Fr_err = numpy.max(numpy.abs((Frh - Frt) / Frh))
        val1  = (Frh[2:-1] + Frh[1:-2]) / 2. - (Frt[2:-1] + Frt[1:-2]) / 2.
        val2  = val1 * val1
        val1 *= x[2:-1] - x[1:-2]
        val2 *= x[2:-1] - x[1:-2]
        Fr_L1 = numpy.sum(numpy.abs(val1)) / (x[-1] - x[0])
        Fr_L2 = numpy.sqrt(numpy.sum(val2)) / (x[-1] - x[0])
        Pr_err = numpy.max(numpy.abs(Prh - Prt) / Prh)
        val1  = (Prh[2:-1] + Prh[1:-2]) / 2. - (Prt[2:-1] + Prt[1:-2]) / 2.
        val2  = val1 * val1
        val1 *= x[2:-1] - x[1:-2]
        val2 *= x[2:-1] - x[1:-2]
        Pr_L1 = numpy.sum(numpy.abs(val2)) / (x[-1] - x[0])
        Pr_L2 = numpy.sqrt(numpy.sum(val2)) / (x[-1] - x[0])
        print_bool = "The following errors are new lows:\n"
        if (self.f_iters == 0):
            self.f_err_low = f_err
            self.f_L1_low = f_L1
            self.f_L2_low = f_L2
            self.Er_err_low = Er_err
            self.Er_L1_low = Er_L1
            self.Er_L2_low = Er_L2
            self.Fr_err_low = Fr_err
            self.Fr_L1_low = Fr_L1
            self.Fr_L2_low = Fr_L2
            self.Pr_err_low = Pr_err
            self.Pr_L1_low = Pr_L1
            self.Pr_L2_low = Pr_L2
            self.Im_err_low = Im_err
            bool_f_err = 1
            bool_f_L1 = 1
            bool_f_L2 = 1
            bool_Er_err = 1
            bool_Er_L1 = 1
            bool_Er_L2 = 1
            bool_Fr_err = 1
            bool_Fr_L1 = 1
            bool_Fr_L2 = 1
            bool_Pr_err = 1
            bool_Pr_L1 = 1
            bool_Pr_L2 = 1
            bool_Im_err = 1
            all_bools = 1
        else:
            all_bools = 0
            for i in range(Sn):
                if (int(Im_err[i] < self.Im_err_low[i])):
                    self.Im_err_low[i] = Im_err[i]
                    bool_Im_err = 1
                    print_bool += "Im_err[" + str(i) + "]\n"
            i = self.f_iters
            prev_f_err = self.d_errs[0][0][i - 2]
            if (int(f_err < self.f_err_low) | int(f_err < prev_f_err)):
                self.f_err_low = f_err
                bool_f_err = 1
                print_bool += "f_err\n"
            else:
                bool_f_err = 0
            all_bools += bool_f_err
            if (int(f_L1 < self.f_L1_low) & int(f_L1 > 0)):
                self.f_L1_low = f_L1
                bool_f_L1 = 1
                print_bool += "f_L1\n"
            else:
                bool_f_L1 = 0
            all_bools += bool_f_L1
            if (int(f_L2 < self.f_L2_low) & int(f_L2 > 0)):
                self.f_L2_low = f_L2
                bool_f_L2 = 1
                print_bool += "f_L2\n"
            else:
                bool_f_L2 = 0
            all_bools += bool_f_L2
            if (int(Er_err < self.Er_err_low) & int(Er_err > 0)):
                self.Er_err_low = Er_err
                bool_Er_err = 1
                print_bool += "Er_err\n"
            else:
                bool_Er_err = 0
            all_bools += bool_Er_err
            if (int(Er_L1 < self.Er_L1_low) & int(Er_L1 > 0)):
                self.Er_L1_low = Er_L1
                bool_Er_L1 = 1
                print_bool += "Er_L1\n"
            else:
                bool_Er_L1 = 0
            all_bools += bool_Er_L1
            if (int(Er_L2 < self.Er_L2_low) & int(Er_L2 > 0)):
                self.Er_L2_low = Er_L2
                bool_Er_L2 = 1
                print_bool += "Er_L2\n"
            else:
                bool_Er_L2 = 0
            all_bools += bool_Er_L2
            if (int(Fr_err < self.Fr_err_low) & int(Fr_err > 0)):
                self.Fr_err_low = Fr_err
                bool_Fr_err = 1
                print_bool += "Fr_err\n"
            else:
                bool_Fr_err = 0
            all_bools += bool_Fr_err
            if (int(Fr_L1 < self.Fr_L1_low) & int(Fr_L1 > 0)):
                self.Fr_L1_low = Fr_L1
                bool_Fr_L1 = 1
                print_bool += "Fr_L1\n"
            else:
                bool_Fr_L1 = 0
            all_bools += bool_Fr_L1
            if (int(Fr_L2 < self.Fr_L2_low) & int(Fr_L2 > 0)):
                self.Fr_L2_low = Fr_L2
                bool_Fr_L2 = 1
                print_bool += "Fr_L2\n"
            else:
                bool_Fr_L2 = 0
            all_bools += bool_Fr_L2
            if (int(Pr_err < self.Pr_err_low) & int(Pr_err > 0)):
                self.Pr_err_low = Pr_err
                bool_Pr_err = 1
                print_bool += "Pr_err\n"
            else:
                bool_Pr_err = 0
            all_bools += bool_Pr_err
            if (int(Pr_L1 < self.Pr_L1_low) & int(Pr_L1 > 0)):
                self.Pr_L1_low = Pr_L1
                bool_Pr_L1 = 1
                print_bool += "Pr_L1\n"
            else:
                bool_Pr_L1 = 0
            all_bools += bool_Pr_L1
            if (int(Pr_L2 < self.Pr_L2_low) & int(Pr_L2 > 0)):
                self.Pr_L2_low = Pr_L2
                bool_Pr_L2 = 1
                print_bool += "Pr_L2\n"
            else:
                bool_Pr_L2 = 0
            all_bools += bool_Pr_L2
        self.make_RT_solution_bool = 0
        print_stmnt += 'self.make_RT_solution_bool = '
        print_stmnt += str(bool(self.make_RT_solution_bool))
        print_stmnt += '\n'
        if (bool(all_bools) & (f_err > self.f_tol)):
            self.make_RT_solution_bool = 1
            print_stmnt += 'self.make_RT_solution_bool = '
            print_stmnt += str(self.make_RT_solution_bool) + '\n'
        self.f_err.append(f_err)
        self.f_L1.append(f_L1)
        self.f_L2.append(f_L2)
        self.Er_err.append(Er_err)
        self.Er_L1.append(Er_L1)
        self.Er_L2.append(Er_L2)
        self.Fr_err.append(Fr_err)
        self.Fr_L1.append(Fr_L1)
        self.Fr_L2.append(Fr_L2)
        self.Pr_err.append(Pr_err)
        self.Pr_L1.append(Pr_L1)
        self.Pr_L2.append(Pr_L2)
        if (self.f_iters != 0):
            self.Im_err.append(Im_err)
        if self.f_iters < 5:
            for i_iters in range(self.f_iters + 1):
                self.print_errors(i_iters)
        else:
            for i_iters in range(self.f_iters - 5, self.f_iters + 1):
                self.print_errors(i_iters)
        print_stmnt += print_bool
        print_stmnt += 'leaving compute_errors' + '\n'
        print('\n')
        print('self.f_err = ', self.f_err)
        print('\n')
        print('self.f_L1 = ', self.f_L1)
        print('\n')
        print('self.f_L2 = ', self.f_L2)
        print('\n')
        print('\n')
        print('self.Er_err = ', self.Er_err)
        print('\n')
        print('self.Er_L1 = ', self.Er_L1)
        print('\n')
        print('self.Er_L2 = ', self.Er_L2)
        print('\n')
        print('\n')
        print('self.Fr_err = ', self.Fr_err)
        print('\n')
        print('self.Fr_L1 = ', self.Fr_L1)
        print('\n')
        print('self.Fr_L2 = ', self.Fr_L2)
        print('\n')
        print('\n')
        print('self.Pr_err = ', self.Pr_err)
        print('\n')
        print('self.Pr_L1 = ', self.Pr_L1)
        print('\n')
        print('self.Pr_L2 = ', self.Pr_L2)
        print('\n')
        print('\n')
        print('leaving compute_errors')

    def print_errors(self, i_iters):
        print_stmnt = '\n'
        print_stmnt += 'entered print_errors\n'
        print_stmnt += 'For iteration ' + str(i_iters) + '\n'
        print_stmnt += 'The VEF relative error              is '
        print_stmnt += str(self.f_err[i_iters])
        if (i_iters == 0):
            ith  = numpy.abs(self.d_f[i_iters] - 1. / 3.)
            ith /= self.d_f[i_iters]
            ith  = numpy.argmax(ith)
        else:
            ith  = numpy.interp(self.d_x[i_iters],
                                self.d_x[i_iters - 1],
                                self.d_f[i_iters - 1])
            ith -= self.d_f[i_iters]
            ith  = numpy.argmax(numpy.abs(ith) / self.d_f[i_iters])
        print_stmnt += "This ocurrs at the " + str(ith) + " value of f"
        len_f = len(self.d_f[i_iters])
        print_stmnt += "and f contains " + str(len(self.d_f[i_iters]))
        print_stmnt += " elements\n"
        if (i_iters == 0):
            f_old_ith = 1./3.
        else:
            f_old_ith = numpy.interp(self.d_x[i_iters][ith],
                                     self.d_x[i_iters - 1],
                                     self.d_f[i_iters - 1])
        print_stmnt += 'The value of f_old at that position is '
        print_stmnt += str(f_old_ith) + '\n'
        print_stmnt += 'The value of f     at that position is '
        print_stmnt += str(self.d_f[i_iters][ith]) + '\n'
        print_stmnt += 'and the value of x at that position is '
        print_stmnt += str(self.d_x[i_iters][ith]) + '\n'
        print_stmnt += 'The VEF L2 error (x normalized)     is '
        print_stmnt += str(self.f_L2[i_iters]) + '\n'
        if (i_iters == self.f_iters):
            self.Machs_to_add.append(self.d_M[-1][ith])
        print_stmnt += 'The Er relative error               is '
        print_stmnt += str(self.Er_err[i_iters]) + '\n'
        ith  = abs(self.d_Erh[i_iters] - self.d_Ert[i_iters])
        ith /= self.d_Erh[i_iters]
        ith  = numpy.argmax(ith)
        print_stmnt += "This is the " + str(ith) + " value of Er,"
        len_RED_RT = len(self.d_Erh[i_iters])
        print_stmnt += "and Er contains " + str(len_RED_RT) + " elements."
        print_stmnt += 'The value of Erh at that position   is '
        print_stmnt += str(self.d_Erh[i_iters][ith]) + '\n'
        print_stmnt += 'The value of Ert at that position   is '
        print_stmnt += str(self.d_Ert[i_iters][ith]) + '\n'
        print_stmnt += 'and the value of x at that position is '
        print_stmnt += str(self.d_x[i_iters][ith]) + '\n'
        print_stmnt += 'The Er L2 error (x normalized)      is '
        print_stmnt += str(self.Er_L2[i_iters]) + '\n'
        print_stmnt += 'The Fr relative error               is '
        print_stmnt += str(self.Fr_err[i_iters]) + '\n'
        ith  = abs(self.d_Frh[i_iters] - self.d_Frt[i_iters]) 
        ith /= self.d_Frh[i_iters]
        ith  = numpy.argmax(ith)
        print_stmnt += "This is the " + str(ith) + " value of Fr,"
        len_radFlux_RT = len(self.d_Frh[i_iters])
        print_stmnt += "and Fr contains " + str(len_radFlux_RT) + " elements."
        print_stmnt += 'The value of Frh at that position   is '
        print_stmnt += str(self.d_Frh[i_iters][ith]) + '\n'
        print_stmnt += 'The value of Frt at that position   is '
        print_stmnt += str(self.d_Frt[i_iters][ith]) + '\n'
        print_stmnt += 'and the value of x at that position is '
        print_stmnt += str(self.d_x[i_iters][ith]) + '\n'
        print_stmnt += 'The Fr L2 error (x normalized)      is '
        print_stmnt += str(self.Fr_L2[i_iters]) + '\n'
        print_stmnt += 'The P_RT relative error             is '
        print_stmnt += str(self.Pr_err[i_iters]) + '\n'
        if (len(self.d_Prh[i_iters]) == len(self.d_Prt[i_iters])):
            ith  = abs(self.d_Prh[i_iters] - self.d_Prt[i_iters])
            ith /= self.d_Prh[i_iters]
            ith  = numpy.argmax(ith)
            print_stmnt += "This is the " + str(ith) + " value of Pr,"
            len_fRED_RT = len(self.d_Prh[i_iters])
            print_stmnt += "and Pr contains " + str(len_fRED_RT) + " elements."
            print_stmnt += 'The value of Prh at that position   is '
            print_stmnt += str(self.d_Prh[i_iters][ith]) + '\n'
            print_stmnt += 'The value of Prt at that position   is '
            print_stmnt += str(self.d_Prt[i_iters][ith]) + '\n'
            print_stmnt += 'and the value of x at that position is '
            print_stmnt += str(self.d_x[i_iters][ith]) + '\n'
        print_stmnt += 'The Pr L2 error (x normalized)      is '
        print_stmnt += str(self.Pr_L2[i_iters]) + '\n'
        print_stmnt += 'leaving print_errors' + '\n'

    def continue_running(self):
        print_stmnt  = '\n'
        print_stmnt += 'entered continue_running\n'
        while self.make_RT_solution_bool:
            self.f_iters += 1
            self.make_2T_solution()
            self.make_RT_solution()
        print_stmnt += 'leaving continue_running'


class IE_discontinuousShockProfiles(IEShockProfile, ShockMethods_2T):
    '''
    '''
    def __init__(self, incoming):
        super(IE_discontinuousShockProfiles, self).__init__(incoming)
        print_stmnt  = '\n'
        print_stmnt += 'entered AnalyticShockProfiles __init__ for M0 = '
        print_stmnt += str(incoming.M0) + '\n'
#         from exactpack.solvers.radshocks import fnctn_2Tie as fnctn
        import fnctn_2Tie as fnctn
        global fnctn
        self.Z = incoming.Z
        self.right_pts = 2. * self.num_pts + 1
        self.eps_precursor_ASP      = incoming.eps_precursor_ASP
        self.eps_precursor_ASP_hold = incoming.eps_precursor_ASP
        self.eps_relaxation_ASP      = incoming.eps_relaxation_ASP
        self.eps_relaxation_ASP_hold = incoming.eps_relaxation_ASP
        self.make_RT_solution_bool = 1
        self.Mach_precursor_baseline = 0.
        self.Mach_relaxation_baseline = 0.
        self.left1 = 0
        self.right1 = 0
        self.continuous_shock = 1
        self.Te0 = incoming.Te0
        self.Te_init = numpy.zeros(2)
        self.epsilon = incoming.epsilon
        self.mult_vals = []
        self.eps_precursor_ASP_initially = self.eps_precursor_ASP
        self.eps_relaxation_ASP_initially = self.eps_relaxation_ASP
        print_stmnt += 'leaving AnalyticShockProfiles __init__ for M0 = '
        print_stmnt += str(incoming.M0) + '\n'


class IE_continuousShockProfiles(IEShockProfile, ShockMethods_2T):
    '''
    '''
    def __init__(self, incoming):
        super(IE_continuousShockProfiles, self).__init__(incoming)
        print_stmnt  = '\n'
        print_stmnt += 'entered AnalyticShockProfiles __init__ for M0 = '
        print_stmnt += str(incoming.M0) + '\n'
#         from exactpack.solvers.radshocks import fnctn_2Tie as fnctn
        import fnctn_2Tie as fnctn
        global fnctn
        self.Z = incoming.Z
        self.right_pts = 2. * self.num_pts + 1
        self.eps_precursor_ASP      = incoming.eps_precursor_ASP
        self.eps_precursor_ASP_hold = incoming.eps_precursor_ASP
        self.eps_relaxation_ASP      = incoming.eps_relaxation_ASP
        self.eps_relaxation_ASP_hold = incoming.eps_relaxation_ASP
        self.make_RT_solution_bool = 1
        self.Mach_precursor_baseline = 0.
        self.Mach_relaxation_baseline = 0.
        self.left1 = 0
        self.right1 = 0
        self.continuous_shock = 1
        self.Te0 = incoming.Te0
        self.Te_init = numpy.zeros(2)
        self.epsilon = incoming.epsilon
        self.mult_vals = []
        self.eps_precursor_ASP_initially = self.eps_precursor_ASP
        self.eps_relaxation_ASP_initially = self.eps_relaxation_ASP
        print_stmnt += 'leaving AnalyticShockProfiles __init__ for M0 = '
        print_stmnt += str(incoming.M0) + '\n'

    def make_continuous_solution(self):
        print_stmnt  = '\n'
        print_stmnt += 'entered make_cont_solution\n'
        self.linearize_away_from_equilibrium()
        self.make_mach_arrays()
        M_pre = self.Mach_precursor
        M_rel = self.Mach_relaxation[::-1]
        Mach = numpy.append(numpy.append(M_pre, 1.), M_rel)
        print('use_jac = ', self.use_jac)
        if self.use_jac:
          i = scipy.integrate.ode(fnctn.ddM_ode, fnctn.ddM_jac)
          i.set_f_params(self).set_jac_params(self)
        else:
          i = scipy.integrate.ode(fnctn.ddM_ode)
          i.set_f_params(self)
        i.set_integrator('vode', atol = self.int_tol, rtol = self.int_tol,
                         method = 'bdf', nsteps = self.max_steps)
        init_vals = [self.Te_eps0, 0.]
        Mach_diffed = numpy.diff(Mach)
        Mach_size = numpy.size(Mach)
        result_arrays = numpy.zeros((Mach_size, 2))
        result_arrays[0,:] = init_vals
        i.set_initial_value(init_vals, Mach[0])
        k = 1
        while (i.successful() & (k < Mach_size)):
            dt = Mach_diffed[k - 1]
            i.integrate(i.t + dt)
            result_arrays[k, 0] = i.y[0]
            result_arrays[k, 1] = i.y[1]
            k += 1
        Te = result_arrays[:, 0]
        x = result_arrays[:, 1]
        x_shift = numpy.interp(1., Mach[::-1], x[::-1])
        x -= x_shift
        self.x = numpy.append(numpy.append(-5. * x[-1], x), 5. * x[-1])
        self.Te = numpy.append(numpy.append(self.T0, Te), self.T1)
        self.Mach = numpy.append(numpy.append(self.M0, Mach), self.M1)
        self.Ti = fnctn.ion_temp(self.Te, self.Mach, self)
        self.Tm = fnctn.mat_temp(self.Te, self.Mach, self)
        self.Density = fnctn.mat_density(self.Te, self.Mach, self)
        self.Speed = self.M0 / self.Density
        self.Pressure = self.Density * self.Tm / self.gamma
        Fe  = - fnctn.kappa_e(self.Tm, self.Density, self)
        Fe *= fnctn.dTedx(self.Te, self.Mach, self)
        self.Fe = Fe
        self.SIE = self.Tm / self.gamma / (self.gamma - 1.)
