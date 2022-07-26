import numpy
import scipy.integrate
import scipy.interpolate

def sigma_a(T, self):
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    gamma = self.gamma
    a = T / gamma
    b = P0 * T * T * T * T / 3. - M02 - 1./gamma - P0 / 3.
    c = M02
    rho = - (b + numpy.sqrt(b**2 - 4. * a * c)) / 2. / a
    return self.sigA * rho**self.expDensity_abs * T**self.expTemp_abs

def sigma_s(T, self):
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    gamma = self.gamma
    a = T / gamma
    b = P0 * T * T * T * T / 3. - M02 - 1./gamma - P0 / 3.
    c = M02
    rho = - ( b + numpy.sqrt(b**2 - 4. * a * c)) / 2. / a
    return self.sigS * rho**self.expDensity_scat * T**self.expTemp_scat

def sigma_t(T, self):
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    gamma = self.gamma
    a = T / gamma
    b = P0 * T * T * T * T / 3. - M02 - 1./gamma - P0 / 3.
    c = M02
    rho = - ( b + numpy.sqrt(b**2 - 4. * a * c)) / 2. / a
    siga = self.sigA * rho**self.expDensity_abs  * T**self.expTemp_abs
    sigs = self.sigS * rho**self.expDensity_scat * T**self.expTemp_scat
    return siga + sigs

def dsigma_dx(T, self):
    eDa = self.expDensity_abs
    eDs = self.expDensity_scat
    eTa = self.expTemp_abs
    eTs = self.expTemp_scat
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    gamma = self.gamma
    a = T / gamma
    b = P0 * T * T * T * T / 3. - M02 - 1./gamma - P0 / 3.
    c = M02
    rho = - ( b + numpy.sqrt(b**2 - 4. * a * c)) / 2. / a
    dTdx_val = 1. / dxdT_val(0., T, self)
    drhodx_val  = (rho / gamma + 4. * P0 * T3 / 3.) * dTdx_val
    drhodx_val /= M02 / rho / rho - T / gamma
    sigA = self.sigA * rho**eDa * T**eTa
    sigS = self.sigS * rho**eDs * T**eTs
    dsiga_dx = sigA * (eDa / rho * drhodx_val + eTa / T * dTdx_val)
    dsigs_dx = sigS * (eDs / rho * drhodx_val + eTs / T * dTdx_val)
    return dsiga_dx + dsigs_dx

def rho(T, self):
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    gamma = self.gamma
    a = T / gamma
    b = P0 * T * T * T * T / 3. - M02 - 1./gamma - P0 / 3.
    c = M02
    return - (b + numpy.sqrt(b**2 - 4. * a * c)) / 2. / a

def dxdT(x, T, self):
    C0 = self.C0
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    gamma = self.gamma
    T3 = T * T * T
    T4 = T3 * T
    a = T / gamma
    b = P0 * T4 / 3. - M02 - 1./gamma - P0 / 3.
    c = M02
    rho = - (b + numpy.sqrt(b**2 - 4. * a * c)) / 2. / a
    speed = M0 / rho
    beta = speed / C0
    e = T / gamma / (gamma - 1.)
    p = rho * T / gamma
    Em  = 0.5 * rho * speed * speed + rho * e + p + 4. * P0 * T4 / 3.
    beta0_Em0 = M0 / C0 * (0.5 * M02 + 1. / (gamma - 1.) + 4. * P0 / 3.)
    siga = self.sigA * rho**self.expDensity_abs  * T**self.expTemp_abs
    sigs = self.sigS * rho**self.expDensity_scat * T**self.expTemp_scat
    sigt = siga + sigs
    return 1. / (3. * sigt * (beta * Em - beta0_Em0) / 4. / P0 / T3)

def dxdT_jac(x, T, self):
    P0 = self.P0
    M0 = self.M0
    M02 = M0 * M0
    gamma = self.gamma
    T3 = T * T * T
    T4 = T3 * T
    a = T / gamma
    b = P0 * T4 / 3. - M02 - 1./gamma - P0 / 3.
    c = M02
    rho = - (b + numpy.sqrt(b**2 - 4. * a * c)) / 2. / a
    siga = self.sigA * rho**self.expDensity_abs  * T**self.expTemp_abs
    sigs = self.sigS * rho**self.expDensity_scat * T**self.expTemp_scat
    sigt = siga + sigs
    dTdx_val = 1. / dxdT(x, T, self)
    val  = - (M02 / rho + 4. * P0 * T4 / 3.) * (rho / gamma + 4. * P0 * T3 / 3.)
    val /= rho * (M02 / rho - T / gamma)
    val += rho / (gamma - 1.) + 16. * P0 * T3 / 3.
    val *= 3. * sigt * beta / 4. / P0 / T3
    val += dsigma_dx(T, self) / sigt - 3. / T
    val /= - dTdx_val
    return val
