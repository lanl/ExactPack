import numpy, scipy.optimize, scipy.integrate, scipy.interpolate

def Srie(E, M, self):
    P = (self.Lambda + (self.Lambda * self.R)**2) * E
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
    epsilon = self.epsilon
    epsilon2 = epsilon * epsilon
    siga = self.sigA * rho**self.expDensity_abs * T**self.expTemp_abs
    sigs = self.sigS * rho**self.expDensity_scat * T**self.expTemp_scat
    sigt = siga + sigs
    F = (- dPdx(E, M, self) + beta * (sigt * P + sigs * E + siga * T4)) / sigt
    return siga * (T4 - E) / epsilon2 + 2. * siga * beta * F

def rad_flux2(E, M, self):
    P = (self.Lambda + (self.Lambda * self.R)**2) * E
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
    return (sigt * P + sigs * E + siga * T4) / sigt

def dPdx(E, M, self):
    P = (self.Lambda + (self.Lambda * self.R)**2) * E
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
    F2 = (sigt * P + sigs * E + siga * T4) / sigt
    Meq = numpy.where(M > 1, M0, self.M1)
    Ereq = numpy.where(M > 1, self.Er0, self.Er1)
    Em0 = mat_total_energy(Ereq, Meq, self)
    F20 = rad_flux2(Ereq, Meq, self)
    # beta0 defined as M0 / rho1 / C0 causes Srp(Pr1, M1, self) != 0
    beta0 = mat_beta(Ereq, Meq, self)
    return sigt / P0 * (beta * (Em + P0 * F2) - beta0 * (Em0 + P0 * F20))

def dEdx(E, M, self):
    P = (self.Lambda + (self.Lambda * self.R)**2) * E
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
    epsilon = self.epsilon
    siga = self.sigA * rho**self.expDensity_abs * T**self.expTemp_abs
    sigs = self.sigS * rho**self.expDensity_scat * T**self.expTemp_scat
    sigt = siga + sigs
    F2 = (sigt * P + sigs * E + siga * T4) / sigt
    Meq = numpy.where(M > 1, M0, self.M1)
    Ereq = numpy.where(M > 1, self.Er0, self.Er1)
    Em0 = mat_total_energy(Ereq, Meq, self)
    F20 = rad_flux2(Ereq, Meq, self)
    # beta0 defined as M0 / rho1 / C0 causes Srp(Pr1, M1, self) != 0
    beta0 = mat_beta(Ereq, Meq, self)
    val = sigt / P0 * (beta * (Em + P0 * F2) - beta0 * (Em0 + P0 * F20))
    if (self.FLD_type is 'LP'):
### R is "positive" for Wilson's sum theory and Larsen's sqrt limiter,
### but "negative" for the Levermore-Pomraning limiter,
### but this "negative" cancels out, which I have checked, "neat"
      def lambda_def(x):
        R = max(epsilon * numpy.abs(x) / E / sigt, 1.e-4)
        return (1. / numpy.tanh(R) - 1. / R) / R, R
###     as a reminder, val == dPdx = d (fE) dx ~ dEdx / 3. so dEdx ~ 3.*val
      val = scipy.optimize.fsolve(lambda x:
                                  lambda_def(x)[0] * x - val, 3. * val)[0]
      self.Lambda, self.R = lambda_def(val)
    elif (self.FLD_type is 'poly'):
      a  = (val / sigt / E * epsilon * epsilon - epsilon) / sigt / E
      b  = 3. * val / sigt / E * epsilon - 2.
      b2 = b * b
      c  = 6. * val
### below, only the (-) of the discriminant returns dEdx = 0 in equilibrium
      val = (- b - numpy.sqrt(b2 - 4. * a * c)) / 2. / a
### setting R = 0 gives P/E = Lambda
      R = val / E / sigt
      self.Lambda, self.R = (2. + R) / (6. + 3. * R + R**2), R
    else:
      n = float(self.FLD_type)
      val = 3. * val / (1. - (epsilon * val / sigt / E)**n)**(1. / n)
### setting R = 0 gives P/E = Lambda
      R = 0.
      if (n == 2):
        R = val / E / sigt
      self.Lambda, self.R = (3.**n + R**n)**(- 1. / n), R
    return val

def dMdx(E, M, self):
    P = (self.Lambda + (self.Lambda * self.R)**2) * E
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
    dPdx_val = dPdx(E, M, self)
    srie = Srie(E, M, self)
    drhodx_val  = P0 * (M0 * dPdx_val - (gamma - 1.) * rho * C0 * srie)
    drhodx_val /= M0 * T * (M2 - 1.)
    dTdx_val  = (T * (gamma * M2 - 1.) * drhodx_val - gamma * P0 * dPdx_val)
    dTdx_val /= rho
    return - M * (drhodx_val / rho + dTdx_val / 2. / T)

def dEdM(E, M, self):
    return dEdx(E, M, self) / dMdx(E, M, self)

def ddM_ode(M, vals, self):
    E, x = vals
    dxdM_val = 1. / dMdx(E, M, self)
    dEdM_val = dEdx(E, M, self) * dxdM_val
    self.dxdM_val = dxdM_val
    return [dEdM_val, dxdM_val]

def dsigA_dx(E, M, self):
    P = (self.Lambda + (self.Lambda * self.R)**2) * E
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
    dPdx_val    = dPdx(E, M, self)
    srie        = Srie(E, M, self)
    drhodx_val  = dPdx_val - (gamma - 1.) / M0 * rho * C0 * srie
    drhodx_val *= P0 / T / (M2 - 1.)
    dTdx_val    = dPdx_val - (gamma * M2 - 1.) / M0 * rho * C0 * srie
    dTdx_val   *= P0 * (gamma - 1.) / rho / (M2 - 1.)
    return sigA * (rho_pow / rho * drhodx_val + T_pow / T * dTdx_val)

def dsigA_dM(E, M, self):
    P = (self.Lambda + (self.Lambda * self.R)**2) * E
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

def dsigA_dE(E, M, self):
    Lambda = self.Lambda
    P = (Lambda + (Lambda * self.R)**2) * E
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
    drhodE_val = Lambda * gamma * P0 * rho / hP
    dTdE_val   = - 2. * Lambda * gamma * P0 * T / hP
    return sigA * (rho_pow / rho * drhodE_val + T_pow / T * dTdE_val)

def dsigS_dx(E, M, self):
    P = (self.Lambda + (self.Lambda * self.R)**2) * E
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
    sigS = self.sigS * rho**rho_pow * T**T_pow * self.epsilon
    dPdx_val    = dPdx(E, M, self)
    srie        = Srie(E, M, self)
    drhodx_val  = dPdx_val - (gamma - 1.) / M0 * rho * C0 * srie
    drhodx_val *= P0 / T / (M2 - 1.)
    dTdx_val    = dPdx_val - (gamma * M2 - 1.) / M0 * rho * C0 * srie
    dTdx_val   *= P0 * (gamma - 1.) / rho / (M2 - 1.)
    return sigS * (rho_pow / rho * drhodx_val + T_pow / T * dTdx_val)

def dsigS_dM(E, M, self):
    P = (self.Lambda + (self.Lambda * self.R)**2) * E
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
    sigS    = self.sigS * rho**rho_pow * T**T_pow * self.epsilon
    drhodM_val = - 2. * rho / M / (gamma * M2 + 1.)
    dTdM_val   = - 2. * (gamma * M2 - 1.) * T / M / (gamma * M2 + 1.)
    return sigS * (rho_pow / rho * drhodM_val + T_pow / T * dTdM_val)

def dsigS_dE(E, M, self):
    Lambda = self.Lambda
    P = (Lambda + (Lambda * self.R)**2) * E
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
    sigS = self.sigS * rho**rho_pow * T**T_pow * self.epsilon
    hP = gamma * M02 + 1. + gamma * P0 * (1./3. - P)
    drhodE_val = Lambda * gamma * P0 * rho / hP
    dTdE_val   = - 2. * Lambda * gamma * P0 * T / hP
    return sigS * (rho_pow / rho * drhodE_val + T_pow / T * dTdE_val)

def dsigT_dx(E, M, self):
    return dsigA_dx(E, M, self) + dsigS_dx(E, M, self)

def dsigT_dM(E, M, self):
    return dsigA_dM(E, M, self) + dsigS_dM(E, M, self)

def dsigT_dE(E, M, self):
    return dsigA_dE(E, M, self) + dsigS_dE(E, M, self)

def mat_density(E, M, self):
    P = (self.Lambda + (self.Lambda * self.R)**2) * E
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    val  = M02 * (gamma * M2 + 1.) / M2
    val /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    return val

def mat_temp(E, M, self):
    val  = self.M0 / M / mat_density(E, M, self)
    val *= val 
    return val

def mat_speed(E, M, self):
    return self.M0 / mat_density(E, M, self)

def mat_beta(E, M, self):
    return self.M0 / mat_density(E, M, self) / self.C0

def mat_pres(E, M, self):
    P = (self.Lambda + (self.Lambda * self.R)**2) * E
    M0 = self.M0
    M02 = M0 * M0
    M2 = M * M
    gamma = self.gamma
    rho  = M02 * (gamma * M2 + 1.) / M2
    rho /= (gamma * M02 + 1. + gamma * self.P0 * (self.Pr0 - P))
    T  = M0 / rho / M
    T *= T
    return rho * T / gamma

def mat_total_energy(E, M, self):
    P = (self.Lambda + (self.Lambda * self.R)**2) * E
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

def rad_temp(E, M, self):
    return E**(1./4.)

def rad_flux(E, M, self):
    dPdx_val = dPdx(E, M, self)
    P = (self.Lambda + (self.Lambda * self.R)**2) * E
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
    F2 = (sigt * P + sigs * E + siga * T4) / sigt
    return -dPdx_val / sigt + beta * F2

def ddM_jac(M, vals, self):
    E, x = vals
    Lambda = self.Lambda
    P = (Lambda + (Lambda * self.R)**2) * E
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
    dsiga_dx_val = dsigA_dx(E, M, self)
    dsigs_dx_val = dsigS_dx(E, M, self)
    dsigt_dx_val = dsiga_dx_val + dsigs_dx_val
    dsiga_dE_val = dsigA_dE(E, M, self)
    dsigs_dE_val = dsigS_dE(E, M, self)
    dsigt_dE_val = dsiga_dE_val + dsigs_dE_val
    dsiga_dM_val = dsigA_dM(E, M, self)
    dsigs_dM_val = dsigS_dM(E, M, self)
    dsigt_dM_val = dsiga_dM_val + dsigs_dM_val
    dPdx_val = dPdx(E, M, self)
    dEdx_val = dEdx(E, M, self)
    f = 1./3.
    srie = Srie(E, M, self)
    drhodx_val  = dPdx_val - (gamma - 1.) / M0 * C0 * rho * srie
    drhodx_val *= P0 / T / (M2 - 1.)
    dTdx_val  = dPdx_val - gMm1 / M0 * C0 * rho * srie
    dTdx_val *= P0 * (gamma - 1.) / rho / (M2 - 1.)
    dMdx_val = - M * (drhodx_val / rho + dTdx_val / 2. / T)
    dMdx_val2 = dMdx_val**2
    hP = gamma * M02 + 1. + gamma * P0 * (1./3. - P)
    drhodE_val = Lambda * gamma * P0 * rho / hP
    drhodM_val = -2. * rho / M / gMp1
    dTdE_val = -2. * Lambda * gamma * P0 * T / hP
    dTdM_val = -2. * gMm1 * T / M / gMp1
    MCP     = sigt * M0 / C0 / P0
    cT      = (2. + (gamma - 1.) * M2) / 2. / (gamma - 1.)
    Prs     = P0 / rho / sigt
    Prrss   = Prs / rho / sigt
    sT3     = 4. * siga * T3
    F1      = sigt * P + sigs * E + siga * T4
    F       = (- dPdx_val + beta * F1) / sigt
    sigfrac = (siga - sigs) / sigt
    bs2     = 2. * beta / sigt**2
    cM      = (M2 + 1.) / M / (M2 - 1.)
    PMT     = P0 * (gamma + 1.) * M / 2. / rho / T / (M2 - 1.)
    gCM     = (gamma - 1.) * C0 / (gamma + 1.) / M0
    ddP_dEdx_val  = P * dsigt_dE_val + Lambda * sigt + E * dsigs_dE_val
    ddP_dEdx_val += sigs + T4 * dsiga_dE_val + sT3 * dTdE_val
    dSrie_dE_val  = - beta * ddP_dEdx_val
    ddP_dEdx_val *= - Prs
    ddP_dEdx_val -= Prrss * F1 * (sigt * drhodE_val + rho * dsigt_dE_val)
    ddP_dEdx_val += cT * dTdE_val
    ddP_dEdx_val *= MCP
    ddP_dEdx_val += dPdx_val * dsigt_dE_val / sigt
    ddP_dxdx_val  = P * dsigt_dx_val + sigt * dPdx_val
    ddP_dxdx_val += E * dsigs_dx_val + sigs * dEdx_val
    ddP_dxdx_val += T4 * dsiga_dx_val + sT3 * dTdx_val
    dSrie_dx_val  = - beta * ddP_dxdx_val
    ddP_dxdx_val *= - Prs
    ddP_dxdx_val -= Prrss * F1 * (sigt * drhodx_val + rho * dsigt_dx_val)
    ddP_dxdx_val += T * M * dMdx_val
    ddP_dxdx_val += cT * dTdx_val
    ddP_dxdx_val *= MCP
    ddP_dxdx_val += dPdx_val * dsigt_dx_val / sigt
    dSrie_dE_val += ddP_dEdx_val
    dSrie_dE_val *= - 2. * beta * siga / sigt
    hold_val  = siga / rho * drhodE_val - dsiga_dE_val
    hold_val += siga / sigt * dsigt_dE_val
    dSrie_dE_val -= 2. * beta * F * hold_val / sigt
    hold_val = (T4 - E) * dsiga_dE_val + siga * (4. * T3 * dTdE_val - 1.)
    dSrie_dE_val += hold_val / epsilon2
    dSrie_dx_val += ddP_dxdx_val
    dSrie_dx_val *= - 2. * beta * siga / sigt
    hold_val  = siga / rho * drhodx_val - dsiga_dx_val
    hold_val += siga / sigt * dsigt_dx_val
    dSrie_dx_val -= 2. * beta * F * hold_val / sigt
    hold_val = (T4 - E) * dsiga_dx_val + siga * (4. * T3 * dTdx_val - dEdx_val)
    dSrie_dx_val += hold_val / epsilon2
    k1 = P0 * (gamma + 1.) / 2. / rho / T / (M2 - 1.)
    k2 = (M2 + 1.) / (M2 - 1)
    k = P0 * M * (gamma + 1.) / 2. / rho / T / (M2 - 1.)
    dkdx_val = - k1 * (k2 * dMdx_val + M * (drhodx_val / rho + dTdx_val / T))
    dkdM_val = - k1 * (k2 + M * (drhodM_val / rho + dTdM_val / T))
    dkdE_val = - k1 * Lambda * M * (drhodE_val / rho + dTdE_val / T)
    l1 = (gamma - 1.) * C0 / (gamma + 1.) / M0
    l2 = 2. * gamma * rho * M
    l = gMp1 * rho * l1
    dldx_val = l1 * (l2 * dMdx_val + gMp1 * drhodx_val)
    dldM_val = l1 * (l2 + gMp1 * drhodM_val)
    dldE_val = l1 * Lambda * gMp1 * drhodE_val
    ddM_dEdx_val  = - dkdE_val * (dPdx_val - l * srie)
    ddM_dEdx_val -= k * (ddP_dEdx_val - srie * dldE_val - l * dSrie_dE_val)
    ddM_dxdx_val  = - dkdx_val * (dPdx_val - l * srie)
    ddM_dxdx_val -= k * (ddP_dxdx_val - srie * dldx_val - l * dSrie_dx_val)
    if ((self.FLD_type == '1') | (self.FLD_type == '2')):
      n = float(self.FLD_type)
      LAMBDA = 1. - (epsilon * dPdx_val / sigt / E)**n
      const  = (epsilon * dPdx_val / sigt / E)**n * dPdx_val
      const *= LAMBDA**(- (n + 1) / n)
      ddE_dxdx_val  = ddP_dxdx_val / dPdx_val - dsigt_dx_val / sigt
      ddE_dxdx_val -= dEdx_val / E
      ddE_dxdx_val *= const
      ddE_dxdx_val += ddP_dxdx_val / LAMBDA**(1. / n)
      ddE_dEdx_val  = ddP_dEdx_val / dPdx_val - dsigt_dE_val / sigt - 1. / E
      ddE_dEdx_val *= const
      ddE_dEdx_val += ddP_dEdx_val / LAMBDA**(1. / n)
    elif (self.FLD_type is 'poly'):
      EsE = dPdx_val / sigt / E
      a_x = 2 - 3. * epsilon * EsE
      b_x = numpy.sqrt(4. + 12. * epsilon * EsE - 15. * epsilon2 * EsE * EsE)
      c_x = 2. * (epsilon2 * EsE - epsilon) / sigt / E
      c1 = - (3. * epsilon + (6. * epsilon - 15. * epsilon2 * EsE) / b_x) / c_x
      c2 = (a_x - b_x) / c_x / (epsilon2 * EsE - epsilon)
      dETsigEdx_val  = ddP_dxdx_val / dPdx_val - dsigt_dx_val / sigt
      dETsigEdx_val -= dEdx_val / E
      dETsigEdx_val *= EsE
      dETsigEdE_val  = ddP_dEdx_val / dPdx_val - dsigt_dE_val / sigt
      dETsigEdE_val -= 1. / E
      dETsigEdE_val *= EsE
      ddE_dxdx_val  = 2. * dsigt_dx_val / sigt + 2. * dEdx_val / E
      ddE_dxdx_val -= ddP_dxdx_val / dPdx_val
      ddE_dxdx_val *= epsilon2 * EsE
      ddE_dxdx_val -= epsilon * (dsigt_dx_val / sigt + dEdx_val / E)
      ddE_dxdx_val *= c2
      ddE_dxdx_val += c1 * dETsigEdx_val
      ddE_dEdx_val  = 2. * dsigt_dE_val / sigt + 2. / E
      ddE_dEdx_val -= ddP_dEdx_val / dPdx_val
      ddE_dEdx_val *= epsilon2 * EsE
      ddE_dEdx_val -= epsilon * (dsigt_dE_val / sigt + 1. / E)
      ddE_dEdx_val *= c2
      ddE_dEdx_val += c1 * dETsigEdE_val
    jac00 = - dEdx_val * ddM_dEdx_val / dMdx_val2 + ddE_dEdx_val / dMdx_val
    jac10 = - ddM_dEdx_val / dMdx_val2
    jac01 = - dEdx_val * ddM_dxdx_val / dMdx_val2 + ddE_dxdx_val / dMdx_val
    jac11 = - ddM_dxdx_val / dMdx_val2
    return [[jac00, jac01], [jac10, jac11]]
