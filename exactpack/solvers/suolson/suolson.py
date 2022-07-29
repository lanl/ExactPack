"""Python Su-Olson solver.

This is a Python re-implementation of the Riemann solution code suo02.f from
`Frank Timmes' website <http://cococubed.asu.edu/research_pages/su_olson.shtml>`_.
This code is released under LA-CC-05-101.
"""

import numpy as np
from math import sqrt, sin, acos, exp

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
        sum1 = qromo(upart1, eta_lo, eta_hi, eps, midpnt)
    # integrate over each oscillitory piece
    else:
        for i in range(100):
            jwant = i + 1
            eta_int = zbrent(gamma_one_root, eta_lo, eta_hi, tol)
            xi1 = qromo(upart1, eta_lo, eta_int, eps, midpnt)
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
        sum2 = qromo(upart2, eta_lo, eta_hi, eps, midpnt)
    # integrate from hi to lo on this piece
    else:
        for i in range(100):
            jwant = i + 1
            eta_int = zbrent(gamma_two_root, eta_hi, eta_lo, tol)
            xi2 = qromo(upart2, eta_hi, eta_int, eps, midpnt)
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
        sum1 = qromo(vpart1, eta_lo, eta_hi, eps, midpnt)

    # integrate over each oscillitory piece
    # from 1 to 0 on this part; this one really oscillates
    else:
        for i in range(100):
            jwant = i + 1
            eta_int = zbrent(gamma_three_root, eta_hi, eta_lo, tol)
            xi1 = qromo(vpart1, eta_hi, eta_int, eps, midpnt)
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
        sum2 = qromo(vpart2, eta_lo, eta_hi, eps, midpnt)

    # integrate over each oscillitory piece
    # from 1 to 0 on this part; this one really oscillates
    else:
        for i in range(100):
            jwant = i + 1
            eta_int = zbrent(gamma_two_root, eta_hi, eta_lo, tol)
            xi2 = qromo(vpart2, eta_hi, eta_int, eps, midpnt)
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


def zbrent(func, x1, x2, tol):
    """using brent's method this routine finds the root of a function func  
    between the limits x1 and x2. the root is when accuracy is less than tol. 
 
    Note:
        eps the the machine floating point precision
    """
    itmax = 100
    eps = 3.0e-15
    niter = 0
    
    a = x1 
    b = x2 
    fa = func(a)   
    fb = func(b)
    
    if (fa > 0.0 and fb > 0.0) or (fa < 0.0 and fb < 0.0):
        msg = f'\nx1={x1}, f(x1)={fa}'
        msg += f'\nx2={x2}, f(x2)={fb}'
        msg += '\nroot not bracketed in routine zbrent'
        raise ValueError(msg)

    c = b 
    fc = fb    

    # rename a,b,c and adjusting bound interval d
    for i in range(itmax):
        niter = niter + 1
        if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):            
            c  = a    
            fc = fa  
            d = b - a  
            e = d    

        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        tol1 = 2.0 * eps * abs(b) + 0.5 * tol 
        xm = 0.5 * (c - b)  

        # convergence check
        if abs(xm) < tol1 or fb == 0.0:
            return b

        # attempt quadratic interpolation
        if (abs(e) > tol1) and (abs(fa) > abs(fa)):
            s = fb / fa    
            if a == c:
                p = 2.0 * xm * s    
                q = 1.0 - s  
            else:
                q = fa / fc   
                r = fb / fc   
                p = s * (2.0 * xm * q *(q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            # check if in bounds
            if p > 0.0:
                q = -q
            p = abs(p)   

            # accept interpolation
            if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                 e = d
                 d = p / q

            # or bisect
            else:
                d = xm  
                e = d   

        # bounds decreasing to slowly use bisection 
        else:
            d = xm   
            e = d    

        # move best guess to a  
        a  = b 
        fa = fb   
        if abs(d) > tol1:
            b = b + d    
        else:
            b = b + sign(tol1, xm)
        fb = func(b)

    msg = 'too many iterations in routine zbrent'
    raise ValueError(msg)


def midpnt(func, a, b, s, n):
    """this routine computes the n'th stage of refinement of an extended midpoint 
    rule. func is input as the name of the function to be integrated between  
    limits a and b. when called with n=1, the routine returns as s the crudest 
    estimate of the integral of func from a to b. subsequent calls with n=2,3... 
    improve the accuracy of s by adding 2/3*3**(n-1) addtional interior points.
    """
    if n == 1:
        arg = 0.5 * (a + b)
        s = (b - a) * func(arg) 
    else:
        it = 3**(n - 1)
        tnm = it
        delta = (b - a) / (3.0 * tnm) 
        ddelta = delta + delta
        x = a + (0.5 * delta)
        x_sum = 0.0
        for j in range(it):
            x_sum = x_sum + func(x)
            x = x + ddelta
            x_sum = x_sum + func(x)
            x = x + delta

        s = (s + ((b - a) * x_sum / tnm)) / 3.0

    return s


def polint(xa, ya, n, x):
    """given arrays xa and ya of length n and a value x, this routine returns a 
    value y and an error estimate dy. if p(x) is the polynomial of degree n-1
    such that ya = p(xa) ya then the returned value is y = p(x)
    """
    nmax = 10
    # find the index ns of the closest table entry; initialize the c and d
    # tables
    ns = 1
    dif = abs(x - xa[0])
    c = np.zeros(n)
    d = np.zeros(n)
    
    for i in range(n):
        dift = abs(x - xa[i])
        if dift < dif:
            ns = i
            dif = dift
        c[i] = ya[i]
        d[i] = ya[i]

    # first guess for y
    y = ya[ns]

    # for each column of the table, loop over the c's and d's and update them
    ns = ns - 1
    for m in range(1, n):
        for i in range(n - m):
            ho = xa[i] - x
            hp = xa[i + m] - x
            w = c[i + 1] - d[i]
            den = ho - hp
            if den == 0.0:
                raise ValueError('2 xa entries are the same in polint')
            den  = w / den
            d[i] = hp * den
            c[i] = ho * den

        # after each column is completed, decide which correction c or d, to add
        # to the accumulating value of y, that is, which path to take in the
        # table by forking up or down. ns is updated as we go to keep track of
        # where we are. the last dy added is the error indicator.
        if 2 * ns < n - m:
            dy = c[ns + 1]
        else:
            dy = d[ns]
            ns = ns - 1
        y = y + dy
    return y, dy


def qromo(func, a, b, eps, choose):
    """this routine returns as s the integral of the function func from a to b 
    with fractional accuracy eps. 
    jmax limits the number of steps; nsteps = 3**(jmax-1)  
    integration is done via romberg algorithm.    

    it is assumed the call to choose triples the number of steps on each call  
    and that its error series contains only even powers of the number of steps. 
    the external choose may be any of the above drivers, i.e midpnt,midinf... 
    """
    jmax = 14
    jmaxp = jmax + 1
    k = 5
    # km = k - 1
    s = np.zeros(jmaxp)
    h = np.zeros(jmaxp)

    h[0] = 1.0
    for j in range(jmax):
        s[j] = choose(func, a, b, s[j], j+1)
        if j >= k:
            ss, dss = polint(h[j - k:j], s[j - k:j], k, 0.0)    
            if abs(dss) <= eps * abs(ss):
                return ss

        s[j + 1] = s[j]
        h[j + 1] = h[j] / 9.0
    # print('too many steps in qromo')
    return ss
