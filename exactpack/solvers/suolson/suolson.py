"""Python Su-Olson solver.

This is a Python re-implementation of the Riemann solution code suo02.f from
`Frank Timmes' website <http://cococubed.asu.edu/research_pages/su_olson.shtml>`_.
This code is released under LA-CC-05-101.
"""

import numpy as np
from math import sqrt, sin, acos, exp
from scipy.integrate import quad
from scipy.optimize import brentq


# common bloack variables
posx = None
tau = None
epsilon = None
jwant = None


def suolson(t, x, trad_bc_ev, opac, alpha):
    """Compute the solution to the Su-Olson problem over an array of x-values.

    Args:
        t (float): Time at which to compute the solution.
        x (ndarray): An array of positions to compute the solution at.
        trad_bc_ev (float): boundary condition temperature in electron volts
        opac (float): the opacity in cm**2/g
        alpha (float): coefficient of the material equation of state c_v = alpha T_mat**3

    Returns:
        tuple: A 2-tuple containing:
            - *ndarray*: trad_ev, An array temperatures of the radiation field in eV
            - *ndarray*: tmat_ev, An array temperatures of the material field in eV
    """
    nstep = len(x)
    trad_ev = np.zeros(nstep)
    tmat_ev = np.zeros(nstep)
    
    for i in range(nstep):
        zpos = x[i]
        _, _, trad_ev_out, _, tmat_ev_out = so_wave(t, zpos, trad_bc_ev, opac, alpha)
        trad_ev[i] = trad_ev_out
        tmat_ev[i] = tmat_ev_out

    return trad_ev, tmat_ev


def so_wave(time,zpos,trad_bc_ev,opac,alpha):
    """Provides solution to the Su-Olson problem.
    
    Args:
        time (float): time point where solution is desired
        zpos (float): spaatial point where solution is desired
        trad_bc_ev (float): boundary condition temperature in electron volts
        opac (float): the opacity in cm**2/g
        alpha (float): coefficient of the material equation of state c_v = alpha T_mat**3

    Returns:
        tuple: A 5-tuple containing the following:
           - *float*: erad, energy desnity of radiation field erg/cm**3
           - *float*: trad, temperature of radiation field kelvin
           - *float*: trad_ev, temperature of radiation field electron volts
           - *float*: trad, temperature of material field kelvin
           - *float*: trad_ev, temperature of material field electron volts 
    """
    clight = 2.99792458e10
    ssol = 5.67051e-5
    asol = 4.0 * ssol / clight
    kev = 8.617385e-5
    rt3 = 1.7320508075688772
    a4 = 4.0*asol
    a4c = a4 * clight

    # derived parameters and conversion factors
    trad_bc = trad_bc_ev / kev
    ener_in = asol * trad_bc**4
    xpos    = rt3 * opac * zpos
    ialpha  = 1.0 / alpha
    tau     = a4c * opac * ialpha * time
    epsilon = a4 * ialpha
    
    # get the dimensionless solutions
    uans = usolution(xpos, tau, epsilon)
    vans = vsolution(xpos, tau, epsilon, uans)

    # compute the physical solution
    erad = uans * ener_in
    trad = (erad / asol)**0.25
    trad_ev = trad * kev

    tmat = (vans * ener_in / asol)**0.25
    tmat_ev = tmat * kev

    return erad, trad, trad_ev, tmat, tmat_ev


def usolution(posx_in, tau_in, epsilon_in):
    """computes the u solution for the su-olson problem

    Args:
        posx_in (float): X-position
        tau_in (float):
        epsilon_in (float):

    Returns:
        float: usolution_out, the value of the solution at the given x-position.
    """
    tol = 1.0e-6
    eps = 1.0e-10
    eps2 = 1.0e-8
    pi = 3.1415926535897932384
    rt3 = 1.7320508075688772
    rt3opi = rt3/pi

    # transfer input to common block
    global posx
    global tau
    global epsilon
    global jwant 
    posx = posx_in
    tau = tau_in
    epsilon = epsilon_in

    # integrand may not oscillate for small values of posx
    eta_lo = 0.0
    eta_hi = 1.0
    sum1 = 0.0
    jwant = 1
    bracket = (gamma_one_root(eta_lo) * gamma_one_root(eta_hi)) <= 0.0
    if not bracket:
        sum1 = quad(upart1, eta_lo, eta_hi, epsabs=eps)[0]
    # integrate over each oscillitory piece
    else:
        for i in range(100):
            jwant = i + 1
            eta_int = brentq(gamma_one_root, eta_lo, eta_hi, xtol=tol, maxiter=100)
            xi1 = quad(upart1, eta_lo, eta_int, epsabs=eps)[0]
            sum1  = sum1 + xi1
            eta_lo = eta_int
            if abs(xi1) <= eps2:
                break

    # integrand may not oscillate for small values of posx
    eta_lo = 0.0
    eta_hi = 1.0
    sum2 = 0.0
    jwant = 1
    bracket = (gamma_two_root(eta_lo) * gamma_two_root(eta_hi)) <= 0.0
    if not bracket:
        sum2 = quad(upart2, eta_lo, eta_hi, epsabs=eps)[0]
    # integrate from hi to lo on this piece
    else:
        for i in range(100):
            jwant = i + 1
            eta_int = brentq(gamma_two_root, eta_lo, eta_hi, xtol=tol, maxiter=100)
            xi2 = quad(upart2, eta_hi, eta_int, epsabs=eps)[0]
            sum2  = sum2 + xi2
            eta_hi = eta_int
            if abs(xi2) <= eps2:
                break
        sum2 = -sum2

    return 1.0 - 2.0 * rt3opi * sum1 - rt3opi * exp(-tau) * sum2


def vsolution(posx_in, tau_in, epsilon_in, uans):
    """computes the v solution for the su-olson problem"""
    tol = 1.0e-6
    eps = 1.0e-10
    eps2 = 1.0e-8
    pi = 3.1415926535897932384
    rt3 = 1.7320508075688772
    rt3opi = rt3/pi

    # transfer input to common block
    global posx
    global tau
    global epsilon
    global jwant 
    posx = posx_in
    tau = tau_in
    epsilon = epsilon_in

    # integrand may not oscillate for small values of posx
    eta_lo = 0.0
    eta_hi = 1.0
    sum1 = 0.0
    jwant = 1
    bracket = gamma_three_root(eta_lo) * gamma_three_root(eta_hi) <= 0.0
    if not bracket:
        sum1 = quad(vpart1, eta_lo, eta_hi, epsabs=eps)[0]
    # integrate over each oscillitory piece
    # from 1 to 0 on this part; this one really oscillates
    else:
        for i in range(100):
            jwant = i + 1
            eta_int = brentq(gamma_three_root, eta_lo, eta_hi, xtol=tol, maxiter=100)
            xi1 = quad(vpart1, eta_hi, eta_int, epsabs=eps)[0]
            sum1 = sum1 + xi1
            eta_hi = eta_int
            if abs(xi1) <= eps2:
                break
        sum1 = -sum1

    # integrand may not oscillate for small values of posx
    eta_lo = 0.0
    eta_hi = 1.0
    sum2 = 0.0
    jwant = 1
    bracket = gamma_two_root(eta_lo) * gamma_two_root(eta_hi) <= 0.0
    if not bracket:
        sum2 = quad(vpart2, eta_lo, eta_hi, epsabs=eps)[0]
    # integrate over each oscillitory piece
    # from 1 to 0 on this part; this one really oscillates
    else:
        for i in range(100):
            jwant = i + 1
            eta_int = brentq(gamma_two_root, eta_lo, eta_hi, xtol=tol, maxiter=100)
            xi2 = quad(vpart2, eta_hi, eta_int, epsabs=eps)[0]
            sum2 = sum2 + xi2
            eta_hi = eta_int
            if abs(xi2) <= eps2:
                break
        sum2 = -sum2

    # done
    return uans - 2.0 * rt3opi * sum1 + rt3opi * exp(-tau) * sum2


def upart1(eta):
    """equation 36 of su & olson jqsrt 1996, first integrand"""
    tiny = 1.0e-14
    numer = sin(posx * gamma_one(eta, epsilon) + theta_one(eta, epsilon))

    denom = eta * sqrt(3.0 + 4.0 * gamma_one(eta, epsilon)**2)
    denom = max(tiny, denom)

    return exp(-tau * eta * eta) * numer / denom


def upart2(eta):
    """equation 36 of su & olson jqsrt 1996, second integrand"""
    tiny = 1.0e-14
    numer = sin(posx * gamma_two(eta, epsilon) + theta_two(eta, epsilon))

    denom = eta * (1.0 + epsilon * eta)
    denom = denom * sqrt(3.0 + 4.0 * gamma_two(eta, epsilon)**2)
    denom = max(tiny, denom)

    return exp(-tau / (max(tiny, eta * epsilon))) * numer / denom


def vpart1(eta):
    """equation 42 of su & olson jqsrt 1996, first integrand"""
    tiny = 1.0e-14
    eta2 = eta * eta

    numer = sin(posx * gamma_three(eta, epsilon) + theta_three(eta, epsilon))

    denom  = sqrt(4.0 - eta2 + 4.0 * epsilon * eta2 * (1.0 - eta2))
    denom  = max(tiny, denom)

    return exp(-tau * (1.0 - eta2)) * numer / denom


def vpart2(eta):
    """equation 42 of su & olson jqsrt 1996, second integrand"""
    tiny = 1.0e-14

    numer = sin(posx * gamma_two(eta, epsilon) + theta_two(eta, epsilon))

    denom = eta * sqrt(3.0 + 4.0 * gamma_two(eta, epsilon)**2)
    denom = max(tiny, denom)

    return exp(-tau / (max(tiny, eta * epsilon))) * numer / denom


def gamma_one_root(eta_in):
    """used by a root finder to determine the integration inveral"""
    pi = 3.141592653589793238
    twopi = 2.0 * pi

    root = gamma_one(eta_in, epsilon) * posx
    root += theta_one(eta_in, epsilon)
    root -= jwant * twopi
    return root


def gamma_two_root(eta_in):
    """used by a root finder to determine the integration inveral"""
    pi = 3.141592653589793238
    twopi = 2.0 * pi

    root = gamma_two(eta_in, epsilon) * posx
    root += theta_two(eta_in, epsilon)
    root -= jwant * twopi
    return root


def gamma_three_root(eta_in):
    """used by a root finder to determine the integration inveral"""
    pi = 3.141592653589793238
    twopi = 2.0 * pi

    root = gamma_three(eta_in, epsilon) * posx
    root += theta_three(eta_in, epsilon)
    root -= jwant * twopi
    return root


def theta_one(eta, epsilon):
    """equation 38 of su & olson jqsrt 1996"""
    return acos(sqrt(3.0 / (3.0 + (4.0 * gamma_one(eta, epsilon)**2))))


def theta_two(eta, epsilon):
    """equation 38 of su & olson jqsrt 1996"""
    return acos(sqrt(3.0 / (3.0 + (4.0 * gamma_two(eta, epsilon)**2))))


def theta_three(eta, epsilon):
    """equation 43 of su & olson jqsrt 1996"""
    return acos(sqrt(3.0 / (3.0 + (4.0 * gamma_three(eta, epsilon)**2))))


def gamma_one(eta, epsilon):
    """equation 37 of su & olson jqsrt 1996"""
    tiny = 1.0e-14

    ein = max(tiny, min(eta, 1.0 - tiny))
    return ein * sqrt(epsilon + (1.0 / (1.0 - ein * ein)))


def gamma_two(eta, epsilon):
    """equation 37 of su & olson jqsrt 1996"""
    tiny = 1.0e-14

    ein = max(tiny, min(eta, 1.0 - tiny))
    return sqrt((1.0 - ein) * (epsilon + (1.0 / ein)))


def gamma_three(eta, epsilon):
    """equation 43 of su & olson jqsrt 1996"""
    tiny = 1.0e-14

    ein = max(tiny, min(eta, 1.0 - tiny))
    return sqrt((1.0 - ein * ein) * (epsilon + (1.0 / (ein * ein))))
