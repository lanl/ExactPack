import numpy

def gamma_ei(T, rho, self):
#     return 0.5292 # for R =   1.
#     return 5.292  # for R =  10.
    return 52.92  # for R = 100.

def kappa_e(T, rho, self):
    return 1.

def mat_density(Te, M, self):
    M2 = M * M
    M0 = self.M0
    M02 = M0 * M0
    rho0 = self.rho0
    gamma = self.gamma
    val  = rho0 * M02 * (gamma * M2 + 1.) / (gamma * M02 + 1.) / M2
    return val

def mat_temp(Te, M, self):
    M2 = M * M
    M0 = self.M0
    M02 = M0 * M0
    rho0 = self.rho0
    gamma = self.gamma
    rho = rho0 * M02 * (gamma * M2 + 1.) / (gamma * M02 + 1.) / M2
    val = M02 / M2 / rho / rho
    return val

def ion_temp(Te, M, self):
    M2 = M * M
    M0 = self.M0
    M02 = M0 * M0
    rho0 = self.rho0
    gamma = self.gamma
    rho = rho0 * M02 * (gamma * M2 + 1.) / (gamma * M02 + 1.) / M2
    T = M02 / M2 / rho / rho
    Z = self.Z
    val = (Z + 1.) * T - Z * Te
    return val

def dTedx(Te, M, self):
    M2 = M * M
    M0 = self.M0
    M02 = M0 * M0
    rho0 = self.rho0
    gamma = self.gamma
    rho = rho0 * M02 * (gamma * M2 + 1.) / (gamma * M02 + 1.) / M2
    eta = rho0 / rho
    eta1 = rho0 / self.rho1
#     energy_flux   = M0 * M02 * ((gamma  - 1.) * M2 + 2.)
#     energy_flux  /= 2. * rho * rho * (gamma - 1.) * M2
#     energy_flux0  = M0 * ((gamma  - 1.) * M02 + 2.)
#     energy_flux0 /= 2. * rho0 * rho0 * (gamma - 1.)
    T = mat_temp(Te, M, self)
    k_e = kappa_e(T, rho, self)
#     val = (energy_flux - energy_flux0) / k_e
    val = rho0 * M0**3 * (gamma + 1.) / (gamma - 1.) * (1. - eta) * (eta - eta1)
    return val / k_e / 2.

def dMdx(Te, M, self):
    M2 = M * M
    M0 = self.M0
    M02 = M0 * M0
    rho0 = self.rho0
    gamma = self.gamma
    rho = rho0 * M02 * (gamma * M2 + 1.) / (gamma * M02 + 1.) / M2
    rho2 = rho * rho
    eta = rho0 / rho
    eta1 = rho0 / self.rho1
    T = M02 / M2 / rho2
    g_ei = gamma_ei(T, rho, self)
    Z = self.Z
    Cve = Z / (Z + 1.) / gamma / (gamma - 1.)
    dTedx_val = dTedx(Te, M, self)
    val  = g_ei * (Z + 1.) * (T - Te) - M0 * Cve * dTedx_val
    val *= M * (gamma * M2 + 1.) / 2. / M0
    val /= M02 * (M2 - 1.) / (gamma - 1.) / rho2 / M2 + Cve * Te * (gamma - 1.)
#     TeC1_val = TeC1(eta, Te, self)
#     TeC2_val = TeC2(eta, Te, self)
#     val  = 2. * gamma * M02 * eta**2 - 2. * (2. * gamma * M02 + 1.) * eta
#     val += gamma * M02 + 1.
# #     val *= - M0 / 2. / T * eta * (Z + 1.) / (gamma - 1.) / rho0 / M0 / Cve
#     val *= M0 / 2. / T * eta * g_ei * (Z + 1.) / (gamma - 1.) / rho0 / M0 / Cve
#     val *= (Te - TeC2_val) / (Te - TeC1_val)
    return val

def dTedM(Te, M, self):
    val = dTedx(Te, M, self) / dMdx(Te, M, self)
    return val

def ddM_ode(M, vals, self):
    Te, x = vals
    val_dxdM = 1. / dMdx(Te, M, self)
    val_dTedM = dTedx(Te, M, self) * val_dxdM
    return [val_dTedM, val_dxdM]

def Se(Te, M, self):
    M2 = M * M
    Z = self.Z
    M0 = self.M0
    M02 = M0 * M0
    rho0 = self.rho0
    gamma = self.gamma
    rho = rho0 * M02 * (gamma * M2 + 1.) / (gamma * M02 + 1.) / M2
    Cve = Z / (Z + 1.) / gamma / (gamma - 1.)
    pe = rho * Cve * Te * (gamma - 1.)
    se = Cve * numpy.log(pe / rho**gamma)
    dTedx_val = dTedx(Te, M, self)
    T = mat_temp(Te, M, self)
    k_e = kappa_e(T, rho, self)
    val = M0 * Te * se - k_e * dTedx_val
    return val

def TeC1(eta, Te, self):
    Z = self.Z
    M0 = self.M0
    gamma = self.gamma
    eta1 = self.rho0 / self.rho1
    val  = M0**2 * eta * gamma * (Z + 1) / 2. * (gamma + 1.) / (gamma - 1.)
    val *= 1. + eta1 - 2. * eta
    return val

def TeC2(eta, Te, self):
    Z = self.Z
    M0 = self.M0
    rho0 = self.rho0
    gamma = self.gamma
    eta1 = rho0 / self.rho1
    T = self.T0 * eta * (gamma * M0**2 * (1. - eta) + 1.)
    g_ei = gamma_ei(T, rho0 / eta, self)
    k_e = kappa_e(T, rho0 / eta, self)
    val  = - (gamma + 1.) / (gamma - 1.)**2 * (1. - eta) * (eta - eta1)
    val *= rho0**2 * M0**4 * Z / 2. / k_e / g_ei / (Z + 1.)**2 / gamma
    val += self.T0 * eta * (gamma * M0**2 * (1. - eta) + 1.)
    return val
