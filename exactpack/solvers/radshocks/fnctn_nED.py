import numpy, scipy.integrate, scipy.interpolate

def Srie(P, M, self):
    Er = rad_energy_density(P, M, self)
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    beta = M0 / rho / self.C0
    T  = M0 / rho / M
    T *= T
    T4 = T * T * T * T
    siga = self.sigA * rho**self.expDensity_abs * T**self.expTemp_abs
    epsilon = self.epsilon
    epsilon2 = epsilon * epsilon
    problem = self.problem
    if (problem == 'LM_nED'):
      return siga * (T4 - Er) / epsilon2
    elif (problem == 'nED'):
      Meq = numpy.where(M > 1, M0, self.M1)
      Preq = numpy.where(M > 1, self.Pr0, self.Pr1)
      Teq = numpy.where(M > 1, self.T0, self.T1)
      rhoeq = numpy.where(M > 1, self.rho0, self.rho1)
      beta0 = M0 / mat_density(Preq, Meq, self) / self.C0
      sigs = self.sigS * rho**self.expDensity_scat * T**self.expTemp_scat
      sigt = siga + sigs
      siga_eq = self.sigA * rhoeq**self.expDensity_abs * Teq**self.expTemp_abs
      sigs_eq = self.sigS * rhoeq**self.expDensity_scat * Teq**self.expTemp_scat
      F  = - dPdx(P, M, self) + beta * (sigt * P + sigs * Er + siga * T4)
      F /= sigt
      val  = siga * (T4 - Er) / epsilon2 + 2. * siga * beta * F
      val -= beta * beta * (sigt * P + sigs * Er + siga * T4)
      val += 4. * beta0 * beta0 * (sigs_eq - siga_eq) * Preq
      return val

def rad_flux2(P, M, self):
    problem = self.problem
    if (problem == 'LM_nED'):
      return 4. * P
    elif (problem == 'nED'):
      Er = P / f_interp(P, M, self)
      Er = rad_energy_density(P, M, self)
      M0 = self.M0
      M02 = M0 * M0
      M2 = M * M
      gamma = self.gamma
      rho  = M02 * (gamma * M2 + 1.) / M2
      rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
      T  = M0 / rho / M
      T *= T
      T4 = T * T * T * T
      siga = self.sigA * rho**self.expDensity_abs * T**self.expTemp_abs
      sigs = self.sigS * rho**self.expDensity_scat * T**self.expTemp_scat
      sigt = siga + sigs
      return (sigt * P + sigs * Er + siga * T4) / sigt

def dPdx(P, M, self):
#     Er = P / f_interp(P, M, self)
    Er = rad_energy_density(P, M, self)
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    speed = M0 / rho
    beta = speed / self.C0
    T  = M0 / rho / M
    T *= T
    T4 = T * T * T * T
    e = T / gamma / (gamma - 1.)
    p = rho * T / gamma
    Em = 0.5 * rho * speed * speed + rho * e + p
    siga = self.sigA * rho**self.expDensity_abs * T**self.expTemp_abs
    sigs = self.sigS * rho**self.expDensity_scat * T**self.expTemp_scat
    sigt = siga + sigs
    problem = self.problem
    if (problem == 'LM_nED'):
      F2 = 4. * P
    elif (problem == 'nED'):
      F2 = (sigt * P + sigs * Er + siga * T4) / sigt
    Meq = numpy.where(M > 1, M0, self.M1)
    Preq = numpy.where(M > 1, self.Pr0, self.Pr1)
    Em0 = mat_total_energy(Preq, Meq, self)
    F20 = rad_flux2(Preq, Meq, self)
    # beta0 defined as M0 / rho1 / C0 causes Srp(Pr1, M1, self) != 0
    beta0 = mat_beta(Preq, Meq, self)
    return sigt / P0 * (beta * (Em + P0 * F2) - beta0 * (Em0 + P0 * F20))

def dMdx(P, M, self):
    C0 = self.C0
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    T  = M0 / rho / M
    T *= T
    dPdx_val = dPdx(P, M, self)
    srie = Srie(P, M, self)
    drhodx_val  = P0 * (M0 * dPdx_val - (gamma - 1.) * rho * C0 * srie)
    drhodx_val /= M0 * T * (M2 - 1.)
    dTdx_val  = (T * (gamma * M2 - 1.) * drhodx_val - gamma * P0 * dPdx_val)
    dTdx_val /= rho
    return - M * (drhodx_val / rho + dTdx_val / 2. / T)

def dPdM(P, M, self):
    return dPdx(P, M, self) / dMdx(P, M, self)

def ddM_ode(M, vals, self):
    P, x = vals
    dxdM_val = 1. / dMdx(P, M, self)
    dPdM_val = dPdx(P, M, self) * dxdM_val
    self.dxdM_val = dxdM_val
    return [dPdM_val, dxdM_val]

def sigma_a(P, M, self):
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    T  = M0 / rho / M
    T *= T
    return self.sigA * rho**self.expDensity_abs * T**self.expTemp_abs

def sigma_s(P, M, self):
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    T  = M0 / rho / M
    T *= T
    return self.sigS * rho**self.expDensity_scat * T**self.expTemp_scat

def sigma_t(P, M, self):
    return sigma_a(P, M, self) + sigma_s(P, M, self)

def dsigA_dx(P, M, self):
    C0 = self.C0
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * P0 * (self.Pr0 - P))
    T  = M0 / rho / M
    T *= T
    rho_pow = self.expDensity_abs
    T_pow   = self.expTemp_abs
    sigA = self.sigA * rho**rho_pow * T**T_pow
    dPdx_val    = dPdx(P, M, self)
    srie        = Srie(P, M, self)
    drhodx_val  = dPdx_val - (gamma - 1.) / M0 * rho * C0 * srie
    drhodx_val *= P0 / T / (M2 - 1.)
    dTdx_val    = dPdx_val - (gamma * M2 - 1.) / M0 * rho * C0 * srie
    dTdx_val   *= P0 * (gamma - 1.) / rho / (M2 - 1.)
    return sigA * (rho_pow / rho * drhodx_val + T_pow / T * dTdx_val)

def dsigA_dM(P, M, self):
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * P0 * (self.Pr0 - P))
    T  = M0 / rho / M
    T *= T
    rho_pow = self.expDensity_abs
    T_pow   = self.expTemp_abs
    sigA    = self.sigA * rho**rho_pow * T**T_pow
    drhodM_val = -2. * rho / M / (gamma * M2 + 1.)
    dTdM_val   = -2. * (gamma * M2 - 1.) * T / M / (gamma * M2 + 1.)
    return sigA * (rho_pow / rho * drhodM_val + T_pow / T * dTdM_val)

def dsigA_dP(P, M, self):
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * P0 * (self.Pr0 - P))
    T  = M0 / rho / M
    T *= T
    rho_pow = self.expDensity_abs
    T_pow   = self.expTemp_abs
    sigA = self.sigA * rho**rho_pow * T**T_pow
    hP = gamma * M02 + 1. + gamma * P0 * (1./3. - P)
    drhodP_val = gamma * P0 * rho / hP
    dTdP_val   = - 2. * gamma * P0 * T / hP
    return sigA * (rho_pow / rho * drhodP_val + T_pow / T * dTdP_val)

def dsigS_dx(P, M, self):
    C0 = self.C0
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * P0 * (self.Pr0 - P))
    T  = M0 / rho / M
    T *= T
    rho_pow = self.expDensity_scat
    T_pow   = self.expTemp_scat
    sigS = self.sigS * rho**rho_pow * T**T_pow
    dPdx_val    = dPdx(P, M, self)
    srie        = Srie(P, M, self)
    drhodx_val  = dPdx_val - (gamma - 1.) / M0 * rho * C0 * srie
    drhodx_val *= P0 / T / (M2 - 1.)
    dTdx_val    = dPdx_val - (gamma * M2 - 1.) / M0 * rho * C0 * srie
    dTdx_val   *= P0 * (gamma - 1.) / rho / (M2 - 1.)
    return sigS * (rho_pow / rho * drhodx_val + T_pow / T * dTdx_val)

def dsigS_dM(P, M, self):
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * P0 * (self.Pr0 - P))
    T  = M0 / rho / M
    T *= T
    rho_pow = self.expDensity_scat
    T_pow   = self.expTemp_scat
    sigS    = self.sigS * rho**rho_pow * T**T_pow
    drhodM_val = - 2. * rho / M / (gamma * M2 + 1.)
    dTdM_val   = - 2. * (gamma * M2 - 1.) * T / M / (gamma * M2 + 1.)
    return sigS * (rho_pow / rho * drhodM_val + T_pow / T * dTdM_val)

def dsigS_dP(P, M, self):
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * P0 * (self.Pr0 - P))
    T  = M0 / rho / M
    T *= T
    rho_pow = self.expDensity_scat
    T_pow   = self.expTemp_scat
    sigS = self.sigS * rho**rho_pow * T**T_pow
    hP = gamma * M02 + 1. + gamma * P0 * (1./3. - P)
    drhodP_val = gamma * P0 * rho / hP
    dTdP_val   = - 2. * gamma * P0 * T / hP
    return sigS * (rho_pow / rho * drhodP_val + T_pow / T * dTdP_val)

def dsigT_dx(P, M, self):
    return dsigA_dx(P, M, self) + dsigS_dx(P, M, self)

def dsigT_dM(P, M, self):
    return dsigA_dM(P, M, self) + dsigS_dM(P, M, self)

def dsigT_dP(P, M, self):
    return dsigA_dP(P, M, self) + dsigS_dP(P, M, self)

def mat_density(P, M, self):
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    val  = M02 * (gamma * M2 + 1.) / M2
    val /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    return val

def mat_temp(P, M, self):
    val  = self.M0 / M / mat_density(P, M, self)
    val *= val 
    return val

def mat_speed(P, M, self):
    return self.M0 / mat_density(P, M, self)

def mat_beta(P, M, self):
    return self.M0 / mat_density(P, M, self) / self.C0

def mat_pres(P, M, self):
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    T  = M0 / rho / M
    T *= T
    return rho * T / gamma

def mat_internal_energy(P, M, self):
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    T  = M0 / rho / M
    T *= T
    return T / gamma / (gamma - 1.)

def mat_total_energy(P, M, self):
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    speed = M0 / rho
    T  = speed / M
    T *= T
    e = T / gamma / (gamma - 1.)
    p = rho * T / gamma
    return 0.5 * rho * speed * speed + rho * e + p

def momentum_flux(P, M, self):
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    speed = M0 / rho
    T  = speed / M
    T *= T
    p = rho * T / gamma
    return rho * speed * speed + p

def energy_flux(P, M, self):
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    speed = M0 / rho
    T  = speed / M
    T *= T
    e = T / gamma / (gamma - 1.)
    p = rho * T / gamma
    beta = M0 / rho / M
    return beta * (0.5 * rho * speed * speed + rho * e + p)

def rad_energy_density(P, M, self):
    return P / f_interp(P, M, self)

def rad_temp(P, M, self):
    return (P / f_interp(P, M, self))**(1./4.)

def rad_flux(P, M, self):
    dPdx_val = dPdx(P, M, self)
#     Er = P / f_interp(P, M, self)
    Er = rad_energy_density(P, M, self)
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    T  = M0 / rho / M
    T *= T
    T4 = T * T * T * T
    beta = M0 / rho / self.C0
    siga = self.sigA * rho**self.expDensity_abs * T**self.expTemp_abs
    sigs = self.sigS * rho**self.expDensity_scat * T**self.expTemp_scat
    sigt = siga + sigs
    problem = self.problem
    if (problem == 'LM_nED'):
        F2 = 4. * P
    elif (problem == 'nED'):
        F2 = (sigt * P + sigs * Er + siga * T4) / sigt
    return -dPdx_val / sigt + beta * F2

def ddM_jac(M, vals, self):
    P, x = vals
    C0 = self.C0
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    gMp1  = gamma * M2 + 1.
    gMm1  = gamma * M2 - 1.
    rho  = M02 * gMp1 / M2
    rho /= (gamma * M02 + 1. + gamma * P0 * (self.Pr0 - P))
    beta = M0 / rho / C0
    beta2 = beta * beta
    T  = M0 / rho / M
    T *= T
    T3 = T * T * T
    T4 = T3 * T
    epsilon = self.epsilon
    epsilon2 = epsilon * epsilon
    siga = self.sigA * rho**self.expDensity_abs * T**self.expTemp_abs
    sigs = self.sigS * rho**self.expDensity_scat * T**self.expTemp_scat
    sigt = siga + sigs
    dsiga_dx_val = dsigA_dx(P, M, self)
    dsigs_dx_val = dsigS_dx(P, M, self)
    dsigt_dx_val = dsiga_dx_val + dsigs_dx_val
    dsiga_dP_val = dsigA_dP(P, M, self)
    dsigs_dP_val = dsigS_dP(P, M, self)
    dsigt_dP_val = dsiga_dP_val + dsigs_dP_val
    dsiga_dM_val = dsigA_dM(P, M, self)
    dsigs_dM_val = dsigS_dM(P, M, self)
    dsigt_dM_val = dsiga_dM_val + dsigs_dM_val
    dPdx_val = dPdx(P, M, self)
    dEdx_val = dPdx_val * 3.
    f = 1./3.
#     Er = P / f
    Er = rad_energy_density(P, M, self)
    srie = Srie(P, M, self)
    drhodx_val  = dPdx_val - (gamma - 1.) / M0 * C0 * rho * srie
    drhodx_val *= P0 / T / (M2 - 1.)
    dTdx_val  = dPdx_val - gMm1 / M0 * C0 * rho * srie 
    dTdx_val *= P0 * (gamma - 1.) / rho / (M2 - 1.)
    dMdx_val = - M * (drhodx_val / rho + dTdx_val / 2. / T)
    dMdx_val2 = dMdx_val**2
    hP = gamma * M02 + 1. + gamma * P0 * (1./3. - P)
    drhodP_val = gamma * P0 * rho / hP
    drhodM_val = -2. * rho / M / gMp1
    dTdP_val = -2. * gamma * P0 * T / hP
    dTdM_val = -2. * gMm1 * T / M / gMp1
    MCP     = sigt * M0 / C0 / P0
    cT      = (2. + (gamma - 1.) * M2) / 2. / (gamma - 1.)
    Prs     = P0 / rho / sigt
    Prrss   = Prs / rho / sigt
    sT3     = 4. * siga * T3
    F1      = sigt * P + sigs * Er + siga * T4
    F       = (- dPdx_val + beta * F1) / sigt
    problem = self.problem
    if (problem == 'nED'):
      ddP_dPdx_val  = P * dsigt_dP_val + sigt + Er * dsigs_dP_val
      ddP_dPdx_val += sigs / f + T4 * dsiga_dP_val + sT3 * dTdP_val
      dSrie_dP_val  = - beta * ddP_dPdx_val
      ddP_dPdx_val *= - Prs
      ddP_dPdx_val -= Prrss * F1 * (sigt * drhodP_val + rho * dsigt_dP_val)
      ddP_dPdx_val += cT * dTdP_val
      ddP_dPdx_val *= MCP
      ddP_dPdx_val += dPdx_val * dsigt_dP_val / sigt
      ddP_dxdx_val  = P * dsigt_dx_val + sigt * dPdx_val
      ddP_dxdx_val += Er * dsigs_dx_val + sigs * dEdx_val
      ddP_dxdx_val += T4 * dsiga_dx_val + sT3 * dTdx_val
      dSrie_dx_val  = - beta * ddP_dxdx_val
      ddP_dxdx_val *= - Prs
      ddP_dxdx_val -= Prrss * F1 * (sigt * drhodx_val + rho * dsigt_dx_val)
      ddP_dxdx_val += T * M * dMdx_val
      ddP_dxdx_val += cT * dTdx_val
      ddP_dxdx_val *= MCP
      ddP_dxdx_val += dPdx_val * dsigt_dx_val / sigt
      dSrie_dP_val += ddP_dPdx_val
      dSrie_dP_val *= - 2. * beta * siga / sigt
      hold_val  = siga / rho * drhodP_val - dsiga_dP_val
      hold_val += siga / sigt * dsigt_dP_val
      dSrie_dP_val -= 2. * beta * F * hold_val / sigt
      hold_val = (T4 - Er) * dsiga_dP_val + siga * (4. * T3 * dTdP_val - 1. / f)
      dSrie_dP_val += hold_val / epsilon2
      dSrie_dx_val += ddP_dxdx_val
      dSrie_dx_val *= - 2. * beta * siga / sigt
      hold_val  = siga / rho * drhodx_val - dsiga_dx_val
      hold_val += siga / sigt * dsigt_dx_val
      dSrie_dx_val -= 2. * beta * F * hold_val / sigt
      hold_val  = (T4 - Er) * dsiga_dx_val
      hold_val += siga * (4. * T3 * dTdx_val - dEdx_val)
      dSrie_dx_val += hold_val / epsilon2
    elif (problem == 'LM_nED'):
      ddP_dPdx_val  = cT * dTdP_val - 4. * P0 * Er * drhodP_val / 3. / rho / rho
      ddP_dPdx_val += 4. * P0 / rho
      ddP_dPdx_val *= MCP
      ddP_dPdx_val += dPdx_val * dsigt_dP_val / sigt
      ddP_dxdx_val  = cT * dTdx_val - 4. * P0 * Er * drhodx_val / 3. / rho / rho
      ddP_dxdx_val += T * M * dMdx_val + 4. * P0 * dPdx_val / rho
      ddP_dxdx_val *= MCP
      ddP_dxdx_val += dPdx_val * dsigt_dx_val / sigt
      dSrie_dx_val  = (T4 - Er) * dsiga_dx_val
      dSrie_dx_val += siga * (4. * T3 * dTdx_val - dEdx_val)
      dSrie_dx_val /= epsilon2
      dSrie_dP_val  = (T4 - Er) * dsiga_dP_val
      dSrie_dP_val += siga * (4. * T3 * dTdP_val - 1. / f)
      dSrie_dP_val /= epsilon2
### everything below here is common to 'nED' and 'LM_nED'
    k1 = P0 * (gamma + 1.) / 2. / rho / T / (M2 - 1.)
    k2 = (M2 + 1.) / (M2 - 1)
    k = M * k1
    dkdx_val = - k1 * (k2 * dMdx_val + M * (drhodx_val / rho + dTdx_val / T))
    dkdM_val = - k1 * (k2 + M * (drhodM_val / rho + dTdM_val / T))
    dkdP_val = - k1 * M * (drhodP_val / rho + dTdP_val / T)
    l1 = (gamma - 1.) * C0 / (gamma + 1.) / M0
    l2 = 2. * gamma * rho * M
    l = gMp1 * rho * l1
    dldx_val = l1 * (l2 * dMdx_val + gMp1 * drhodx_val)
    dldM_val = l1 * (l2 + gMp1 * drhodM_val)
    dldP_val = l1 * gMp1 * drhodP_val
    ddM_dPdx_val  = - dkdP_val * (dPdx_val - l * srie)
    ddM_dPdx_val -= k * (ddP_dPdx_val - srie * dldP_val - l * dSrie_dP_val)
    ddM_dxdx_val  = - dkdx_val * (dPdx_val - l * srie)
    ddM_dxdx_val -= k * (ddP_dxdx_val - srie * dldx_val - l * dSrie_dx_val)
#     ddM_dxdx_val /= epsilon
#     ddP_dxdx_val /= epsilon
    jac00 = - dPdx_val * ddM_dPdx_val / dMdx_val2 + ddP_dPdx_val / dMdx_val
    jac10 = - ddM_dPdx_val / dMdx_val2
    jac01 = - dPdx_val * ddM_dxdx_val / dMdx_val2 + ddP_dxdx_val / dMdx_val
    jac11 = - ddM_dxdx_val / dMdx_val2
    return [[jac00, jac01], [jac10, jac11]]

def DIm_Dx_ode(x_in, Im, self):
    x_RT = self.x_RT
    M = numpy.interp(x_in, x_RT, self.Mach_RT)
    P = numpy.interp(x_in, x_RT, self.Pr_RT)
    pi = numpy.pi
#     Er = P / f_interp(P, M, self)
    Er = rad_energy_density(P, M, self)
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    T  = M0 / rho / M
    T *= T
    T4 = T * T * T * T
    beta = M0 / rho / self.C0
    epsilon = self.epsilon
    siga = self.sigA * rho**self.expDensity_abs * T**self.expTemp_abs
    sigs = self.sigS * rho**self.expDensity_scat * T**self.expTemp_scat
    sigt = siga + sigs
    siga_4pi = siga / 4. / pi
    sigs_4pi = sigs / 4. / pi
    Meq = numpy.where(M > 1, M0, self.M1)
    Preq = numpy.where(M > 1, self.Pr0, self.Pr1)
    Teq = numpy.where(M > 1, self.T0, self.T1)
    rhoeq = numpy.where(M > 1, self.rho0, self.rho1)
    beta0 = M0 / mat_density(Preq, Meq, self) / self.C0
    siga_eq  = self.sigA * rhoeq**self.expDensity_abs * Teq**self.expTemp_abs
    sigs_eq  = self.sigS * rhoeq**self.expDensity_scat * Teq**self.expTemp_scat
    sigt_eq  = siga_eq + sigs_eq
    problem = self.problem
    mu = self.mu
    dPdx_val = dPdx(P, M, self)
    if (problem == 'LM_nED'):
      F2 = 4. * P
      F = - dPdx_val / sigt + beta * F2
      sigt_4pi = siga_4pi + sigs_4pi
      val  = - sigt_4pi * (F - 4. / 3. * Er * beta) * beta * epsilon / mu
      val += 4. * sigt_4pi * Er * beta
    elif (problem == 'nED'):
      F2 = (sigt * P + sigs * Er + siga * T4) / sigt
      F = - dPdx_val / sigt + beta * F2
      val = beta0 * beta0 * Preq / pi * (2. * sigs_eq / mu - 3. * mu * sigt_eq)
      val -= 2. * sigs_4pi * beta * F / mu
      val *= epsilon
      val += beta * (sigt * Im + 3. * sigs_4pi * Er + 3. * siga_4pi * T4)
    val += (- sigt * Im + sigs_4pi * Er + siga_4pi * T4) / mu / epsilon
    return val

def DIm_Dx_jac(x_in, Im, self):
    x_RT = self.x_RT
    M = numpy.interp(x_in, x_RT, self.Mach_RT)
    P = numpy.interp(x_in, x_RT, self.Pr_RT)
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    T  = M0 / rho / M
    T *= T
    beta = M0 / rho / self.C0
    epsilon = self.epsilon
    siga = self.sigA * rho**self.expDensity_abs * T**self.expTemp_abs
    sigs = self.sigS * rho**self.expDensity_scat * T**self.expTemp_scat
    sigt = siga + sigs
    val = -sigt / self.mu / epsilon
    if (self.problem == 'nED'):
      val += beta * sigt
    return val

def f_interp(P, M, self):
    if ('f' in self.__dict__):
### the line below does not converge Erh and Ert at M0 = 3
#         return numpy.interp(P, self.d_Prt[-1], self.d_f[-1])
        return numpy.interp(M, self.d_M[-1][::-1], self.d_f[-1][::-1])
    else:
        return 1. / 3.

def Q_mu_x(mu, x_in, self): 
    M = numpy.interp(x_in, self.x_RT, self.Mach_RT)
    P = numpy.interp(x_in, self.x_RT, self.Pr_RT)
    pi = numpy.pi
    dPdx_val = dPdx(P, M, self)
    siga_4pi = siga / 4. / pi
    sigs_4pi = sigs / 4. / pi
#     Er = P / f_interp(P, M, self)
    Er = rad_energy_density(P, M, self)
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    T  = M0 / rho / M
    T *= T
    T4 = T * T * T * T
    beta = M0 / rho / self.C0
    mu = self.mu
    siga = self.sigA * rho**self.expDensity_abs * T**self.expTemp_abs
    sigs = self.sigS * rho**self.expDensity_scat * T**self.expTemp_scat
    sigt = siga + sigs
    problem = self.problem
    if (problem == 'LM_nED'):
        F2 = 4. * P
    elif (problem == 'nED'):
        F2 = (sigt * P + sigs * Er + siga * T4) / sigt
    F = -dPdx_val / sigt + beta * F2
    val  = (sigs_4pi * Er + siga_4pi * T4) / mu
    val -= 2. * sigs_4pi * beta * F / mu
    val += beta * (3. * sigs_4pi * Er + 3. * siga_4pi * T4)
    val += beta * beta * P / pi * (2. * sigs / mu - 3. * mu * sigt)
    return val

def Ieq_x(mu, self):
    pi = numpy.pi
    beta = self.M0 / self.C0
    if (mu >= 0.):
        val  = 1. / 4. / pi * (1. + 16. / 3. * mu * beta)
    else:
        beta /= self.rho1
        val  = self.d_Ert[-1][0] / 4. / pi * (1. + 16. / 3. * mu * beta)
    return val 

def exp_factor(mu, x_in, self): 
    M = numpy.interp(x_in, self.x_RT, self.Mach_RT)
    P = numpy.interp(M, self.Mach_RT[::-1], self.Pr_RT[::-1])
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    siga = self.sigA * rho**self.expDensity_abs * T**self.expTemp_abs
    sigs = self.sigS * rho**self.expDensity_scat * T**self.expTemp_scat
    sigt = siga + sigs
    val  = (1. / mu - M0 / rho / self.C0) * sigt * x_in
    return val

def I_mu_x(mus, x_in, self): 
    vals = numpy.zeros(numpy.size(mus))
    for i in range(numpy.size(vals)):
        print('i = ', i)
        mu = mus[i]
        Ieq = Ieq_x(mu, self)
        if ((i == 0) | (i == len(mus) / 2)):
            if (mu >= 0.):
                x_eq = self.d_x[-1][0]
                rho_eq = 1.
                sigt_eq = sigma_t(self.Pr0, self.M0, self)
            else:
                x_eq = self.d_x[-1][-1]
                rho_eq = self.rho1
                sigt_eq = sigma_t(self.Pr1, self.M1, self)
                exp_fac = scipy.integrate.quad(lambda y: sigma_t(numpy.interp(y, x, Pr), numpy.interp(y, x, Mach), self) * (1. / mu - M0 / mat_density(numpy.interp(y, x, Pr), numpy.interp(y, x, Mach), self) / C0), x_in, x_eq)[0]
        vals[i] = Ieq * numpy.exp(exp_fac) + scipy.integrate.quad(lambda z: Q_mu_x(mu, z, self) / mu * numpy.exp(scipy.integrate.quad(lambda y: sigma_t(numpy.interp(y, x, Pr), numpy.interp(y, x, Mach), self) * (1. / mu - M0 / mat_density(numpy.interp(y, x, Pr), numpy.interp(y, x, Mach), self) / C0), x_in, z)[0]), x_eq, x_in)[0]
    return vals

def ddP_dMdx_eq(P, M, self):
    C0 = self.C0
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    T  = M0 / rho / M
    T *= T
    siga = self.sigA * rho**self.expDensity_abs * T**self.expTemp_abs
    sigs = self.sigS * rho**self.expDensity_scat * T**self.expTemp_scat
    sigt = siga + sigs
    dTdM_val = - 2. * (gamma * M2 - 1.) * T / M / (gamma * M2 + 1.)
    drhodM_val = - 2. * rho / M / (gamma * M2 + 1.)
    val  = (2. + (gamma - 1.) * M2) / 2. / (gamma - 1.)
    val -= 12. * P0 * siga * P / sigt / rho / T
    val *= dTdM_val
    val += - 4. * P0 * P / rho**2 * drhodM_val + T * M
    val -= 8. * P0 * P / rho / sigt * dsigT_dM(P, M, self)
    val *= sigt * M0 / C0 / P0
    return val

def ddP_dPdx_eq(P, M, self):
    C0 = self.C0
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    T  = M0 / rho / M
    T *= T
    siga = self.sigA * rho**self.expDensity_abs * T**self.expTemp_abs
    sigs = self.sigS * rho**self.expDensity_scat * T**self.expTemp_scat
    sigt = siga + sigs
    hP = gamma * M02 + 1. + gamma * P0 * (1./3. - P)
    dTdP_val = - 2. * gamma * P0 * T / hP
    drhodP_val = gamma * P0 * rho / hP
    val  = (2. + (gamma - 1) * M2) / 2. / (gamma - 1.)
    val -= 12. * P0 * siga * P / sigt / rho / T
    val *= dTdP_val
    val -= 4. * P0 * P / rho**2 * drhodP_val
    val -= 8. * P0 * P / rho / sigt * dsigT_dP(P, M, self)
    val -= P0 / rho / sigt * (siga + 4. * sigs)
    val *= sigt * M0 / C0 / P0
    return val

def dSrie_dM_eq(P, M, self):
    C0 = self.C0
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    beta = M0 / rho / C0
    beta2 = beta * beta
    T  = M0 / rho / M
    T *= T
    siga = self.sigA * rho**self.expDensity_abs * T**self.expTemp_abs
    sigs = self.sigS * rho**self.expDensity_scat * T**self.expTemp_scat
    sigt = siga + sigs
    dTdM_val = - 2. * (gamma * M2 - 1.) * T / M / (gamma * M2 + 1.)
    drhodM_val = - 2. * rho / M / (gamma * M2 + 1.)
    val  = 12. * siga * P * (1. + beta2 * (siga - sigs) / sigt) / T * dTdM_val
    val -= 8. * beta2 * (siga - sigs) * P / rho * drhodM_val
    val += 4. * beta2 * P * (dsigA_dM(P, M, self) - dsigS_dM(P, M, self))
    val -= beta * (siga - sigs) / sigt * ddP_dMdx_eq(P, M, self)
    return val

def dSrie_dP_eq(P, M, self):
    C0 = self.C0
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    beta = M0 / rho / C0
    beta2 = beta * beta
    T  = M0 / rho / M
    T *= T
    siga = self.sigA * rho**self.expDensity_abs * T**self.expTemp_abs
    sigs = self.sigS * rho**self.expDensity_scat * T**self.expTemp_scat
    sigt = siga + sigs
    hP = gamma * M02 + 1. + gamma * P0 * (1./3. - P)
    dTdP_val = - 2. * gamma * P0 * T / hP
    drhodP_val = gamma * P0 * rho / hP
    val  = 12. * siga * P * (1. + beta2 * (siga - sigs) / sigt) / T * dTdP_val
    val -= 8. * beta2 * (siga - sigs) * P / rho * drhodP_val
    val += 4. * beta2 * P * (dsigA_dP(P, M, self) - dsigS_dP(P, M, self))
    val -= beta * (siga - sigs) / sigt * ddP_dPdx_eq(P, M, self)
    val += -3. * siga + beta2 * (siga - sigs) * (1. + 3. * sigs / sigt)
    return val

def ddM_dMdx_eq(P, M, self):
    C0 = self.C0
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    T  = M0 / rho / M
    T *= T
    val  = - (gamma - 1.) * C0 / (gamma + 1.) / M0 * (gamma * M2 + 1.) * rho
    val *= dSrie_dM_eq(P, M, self)
    val += ddP_dMdx_eq(P, M, self)
    val *= - P0 * (gamma + 1.) * M / 2. / rho / T / (M2 - 1.)
    return val

def ddM_dPdx_eq(P, M, self):
    C0 = self.C0
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    T  = M0 / rho / M
    T *= T
    val  = - (gamma - 1.) * C0 / (gamma + 1.) / M0 * (gamma * M2 + 1.) * rho
    val *= dSrie_dP_eq(P, M, self)
    val += ddP_dPdx_eq(P, M, self)
    val *= - P0 * (gamma + 1.) * M / 2. / rho / T / (M2 - 1.)
    return val
