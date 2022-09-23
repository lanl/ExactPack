r"""
This program generates physical variable (i.e. density, velocity pressure, sound
speed) solution data at a specified point in space and time for the converging
shock wave problem first solved by G. Guderley.

It follows the notation given in [Lazarus1981]_

'guderley_1D' makes use of several subprograms and functions in order to
calculate the two nonlinear eigenvalues that appear in the problem: the
similarity exponent 'lambda' and the reflected shock position (in similarity
variables) 'B.' Once these values are calculated the self-similar equations
governing the flow are solved for the dimensionless velocity V, sound speed C,
and density R, as a function of the similarity variable:

.. math::

  x = \frac{t}{r \lambda}

The program  also computes the result starting from x = infinity (Lazarus,
p. 330 ff.). The subroutine that transforms the similarity variable data is
self-contained in the subroutine "state."

This code is based on the driver code 'guderley' first developed
by J. Bolstad of LLNL.

2007.08.01  S. Ramsey     Code reproduces correct results for gamma = 1.4

Code translated from Fortran to Python by J. Thrussell, 2022.09.23.
"""
import numpy as np
from math import sqrt


def guderley_1d(t, r, ngeom, gamma, rho0):
    """Solve the Guderley problem at a given time over an array of positions.

    Args:
        t (float): The time to solve at.
        r (array): An array of poistions to solve at.
        ngeom (int): 1=planar, 2=cylindrical, 3=spherical.
        gamma (float): Specific heat ratio.
        rho0 (float): Initial density.

    Returns:
        tuple: A 5-tuple containing:
            *array*: An array of density values.
            *array*: An array of velocity values.
            *array*: An array of pressure values.
            *array*: An array of sound speed values.
            *array*: An array of Specific Internal Energy values.
    """
    nstep = len(r)
    den = np.zeros(nstep)
    vel = np.zeros(nstep)
    pres = np.zeros(nstep)
    snd = np.zeros(nstep)
    sie = np.zeros(nstep)
    factorC = 0.750024322

    #.... The input time is a Caramana/Whalen time, defined by:
    #
    #       t_C = 0.750024322*(t_L + 1)
    #
    #.... Here, the time is converted to Lazarus time.
    #
    tee = (t / factorC) - 1.0
    #
    #.... The value of the similarity exponent "lambda" is calulated using the
    #       "exp" function. See documentation appearing in "exp" for an 
    #       explanation of how this value is calculated.
    #
    lambda_ = eexp(ngeom, gamma)
    #
    #.... If a position in both space and time are specified, this data can be
    #       converted into an appropriate value of the similarity variable x
    #       defined above. This value of x is where we desire to know the values
    #       of the similarity variables. 
    for i in range(nstep):
        rpos = r[i]
        targetx = tee / (rpos**lambda_)
    #
    #.... As is the case with lambda, the reflected shock space-time position
    #       "B" is not known a priori (though it is known that B lies in
    #       the range (0 < B < 1)). Lazarus was the first to determine the
    #       value of B to 6 significant figures (appearing in Tables 6.4 and
    #       6.5). This precision can be improved upon using the "zeroin"
    #       routine, as will be explained below.
    #
        Bmaxg = interp_laz(ngeom, gamma, lambda_)
        Bming = 0.34  # use this value for gamma=3.0 and rho0=1.0

        if (Bming <= 0.0):
            raise ValuError('guderley_1D error: Bmin < 0')
        elif (Bmaxg >= 1.0):
            raise ValueError('guderley_1D error: Bmax > 1')

        Bmin = ((gamma + 1.0) / (gamma - 1.0)) * Bming
    #
    #.... The maximum allowable value for B (Bmax) is determined by using
    #       the function INTERP_LAZ. This function interpolates in Lazarus
    #       Tables 6.4 and 6.5 for general polytropic index for B. This
    #       interpolated value of B is used as an upper bound for a more
    #       precise value of B.
    #
        Bmax = ((gamma + 1.0) / (gamma - 1.0)) * Bmaxg + 0.001
        tol = d1mach(4)
    #
    #.... Below, a more precise value of B for a given polytropic index and
    #       geometry type than is given by Lazarus can be computed by using
    #       the "zeroin" routine, which here finds the B-zero of a function
    #       called "Guderley," which is defined below. 
    #
        # Possibly this can be replace with a call to brentq?
        B = zeroin(Bmin, Bmax, Guderley, tol, ngeom, gamma, lambda_)
        # B = brentq(Guderley, Bmin, Bmax, xtol=tol, args=(gamma, lambda_s))
    #
    #.... The ultimate output of the program is generated through the "state"
    #       subroutine, which computes the solution of the similarity variable
    #       equations at the target value of x and then transforms this
    #       solution back to physical variable space.
    #
        deni, veli, presi, sndi, siei = state(rpos, rho0, ngeom, gamma,
                                              lambda_, B, targetx)
        den[i] = deni
        vel[i] = veli
        pres[i] = presi
        snd[i] = sndi
        sie[i] = siei

    return den, vel, pres, snd, sie


def Guderley(n, gamma_d, lambda_d, B):
    """This function computes a difference in similarity variable phase space.

    In particular, It computes the result of two numerical integrations:
        (1) The final C-value found by integrating between x = -1  and x = B
            (namely, at the space-time position of the reflected shock).
        (2) The final C-value found by integrating between x = infinity and
            x = B.

    Boundary conditions are available at both x = -1 and x = infinity, but NOT
    in numerical form at x = B. Therefore, integration to x = B must be
    performed starting from both x = -1 and x = infinity, and the result
    compared.

    It should be noted that the generalized Rankine-Hugoniot jump conditions
    must be executed upon one (but not the other) of the final integration
    points, so that the comparison of C at x = B is consistent (i.e. not
    comparing the variable C evaluated on one side of the shock wave to its
    value on the other side of the shock wave).

    Args:
        n (int): Dimensionality:  2 for cylindrical, 3 for spherical   
        gamma (float): ratio of specific heats                               
        lambda (float): similarity exponent                                   
        B (float): Estimate of the x-coordinate of the location of the reflected
        shock.

    Returns:
        float: The difference between C1 (value of C behind the reflected shock
            obtained by integrating in increasing x) and y(2) (that from
            integrating in increasing w (decreasing x)).  The smaller the 
            absolute value of this difference, the better the choice of B.
    """
    global gamma
    global lambda_
    global sigma
    global intno
    global nu
    global V1
    #.... The following parameters are adjustable, but it is not recommended
    #       that they be adjusted unless error messages are returned by 
    #       the function.
    abserr = 6.0e-11
    doublefreq = 50
    relerr = 5.0e-10
    aeroot = 8.0e-16
    reroot = 8.0e-16
    neqn = 3
    
    # pressure(C, R) = R*C*C/gamma
    #.... nu = n - 1; it is 1 for cylindrical symmetry and 2 for spherical
    nu = n - 1
    gamma = gamma_d
    lambda_ = lambda_d

    #.... When final is false, the integration fro x = B to x = infinity
    #       is skipped.
    final = False
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0

    #.... y(1) = V, y(2) = C, y(3) = R
    #       The initial conditions starting at the incoming shock wave
    #       are set here, along with the parameters necessary for a call
    #       to "ode."
    y = np.zeros(neqn)
    y[0] = -2.0 / gp1
    y[1] = sqrt(2.0 * gamma * gm1) / gp1
    y[2] = gp1 / gm1
    iflag = 1
    x = -1.0
    xout = x
    intno = 1

    #.... An energy integral is used as a consistency check during the 
    #       integration. The definition of this parameter is set by 
    #       a function defined below.
    energymax = np.zeros(2)
    energymin = np.zeros(2)
    energy0 = energy(x, y, gamma, lambda_, nu, 0.0)
    energymax[0] = 0.0
    energymin[0] = 0.0

    #.... Slightly perturb the following if stuck on a singularity.
    dx = 0.00125

    #.... Now begin the integration, starting from x = -1 and continuing to 
    #       x = B. B is the x coordinate of the reflected shock. Only if the
    #       integration returns the error message below should the parameters
    #       abserr and relerr be adjusted.
    #
    j = 0
    while (xout < B):
        j = j + 1
        xout = min(-1.0 + dx * j, B)
        # Replace this with a scipy ode solver
        iflag, work, iwork = ode(f, neqn, y, x, xout, relerr, abserr)
        
        e = energy(x, y, gamma, lambda_, nu, energy0)
        energymin[0] = min(energymin[0], e)
        energymax[0] = max(energymax[0], e)

        if (iflag #= 2 and iflag #= 4):
            err_str = f'Error during B-search for B = {B}\n'
            err_str += f'iflag: {iflag:3}, abserr: {abserr:14.7e}, '
            err_str += f'relerr: {relerr:14.7e}'
            print(err_str)
            return 0.0

    #
    #.... If the integration up until x = B is completed, apply the shock
    #       jump conditiosn given by Lazarus Eq. (2.6); afterwards
    #       continue the integration from x = B to some large x (set here
    #       at x = 10^6).
    # 
    iflag = 1
    C2 = y[1]**2
    V1 = gm1 * (1.0 + y[0]) / gp1 + 2.0 * C2 / (gp1 * (1.0 + y[0])) - 1.0
    y[1] = np.sign(sqrt(C2 + 0.5 * gm1 * ((1.0 + y[0])**2 - (1.0 + V1)**2)), y[1])
    C1 = y[1]
    y[2] = y[2] * (1.0 + y[0]) / (1.0 + V1)
    y[0] = V1

    energy0 = energy(x, y, gamma, lambda_, nu, 0.0)
    energymax[1] = 0.0
    energymin[1] = 0.0

    dx = 0.05
    xlast = B
    j = 0
    while x < 1.0e6 and final:
        j = j + 1
        jmod = np.mod(j-1, doublefreq) + 1

        #.... jmod goes 1, 2, ..., doublefreq

        xout = xlast + dx * jmod
        # Replace this with a scipy ode solver
        iflag, work, iwork = ode(f, neqn, y, x, xout, relerr, abserr)
        e = energy(x, y, gamma, lambda_, nu, energy0)

        #.... If the energy check goes bad, quit the integration.
        energymin[1] = min(energymin[1], e)
        energymax[1] = max(energymax[1], e)
        if iflag #= 2:
            err_str = f'Error during integration\n'
            err_str += f'iflag: {iflag:3}, abserr: {abserr:14.7e}, '
            err_str += f'relerr: {relerr:14.7e}'
            raise ValueError(err_str)

        if jmod == doublefreq:
            #.... Double the spacing of the output.
            xlast = xout
            dx = 2.0 * dx

    #.... Now switch variables to w = k*x^(-sigma) and integrate from
    #       w near zero to some positive w. This (redundant) integration
    #       is necessary to determine the parameter B to more than the
    #       six digits given by Lazarus (through the procedure described
    #       in the function description space.
    iflag = 1
    nuz = (lambda_ - 1.0) / gamma
    V0 = - 2.0 * nuz / (nu + 1)
    sigma = (1.0 + nuz / (1.0 + V0)) / lambda_
    w = 1.0e-10

    #.... Initial w is chosen as follows:
    #       The first neglected term is the asymptotic expansion for
    #       V = y(1) is w*w*V2, and in C = y(2) is w*c1, which are
    #       O(w*w) less than the first terms.  Thus the neglected terms
    #       are O(10^(-18)), less than the O(10^(-16)) machine precision.
    y[0] = V0
    y[1] = -1.0 / w
    intno = 2
    neq = 2

    #.... We integrate only 2 differential equations for V and C.
    #       Trying to integrate the R equation requires knowing k, which
    #       we are trying to determine.  Thus no energy check is possible
    #       here.  But it is not needed since no singularities arise.
    dw = w
    wlast = w
    j = 0
    phi = np.zeros((neqn, 16))
    while True:
        j = j + 1
        jmod = np.mod(j-1, doublefreq) + 1

        #.... jmod goes 1, 2, ..., doublefreq
        wout = wlast + dw * jmod
        
        y, w, iflag = deroot(f, neq, y, w, wout, relerr, abserr, iflag, Vdiff,
                             reroot, aeroot, phi)
        
        if iflag #= 2 and iflag #= 7:
            raise ValueError

        if iflag == 7:
            #.... Root found.  Let D be the number of correct digits in
            #       lambda.  Then min(D, -log_10(|y(1) - V1|) or -log_10(
            #       |y(2) - C1|)) is roughly the number of correct digits in B.
            raise ValueError

        if jmod == doublefreq:
            wlast = wout
            #.... double the spacing of the output
            if dw < .0025:
                dw = 2.0 * dw

#.... The phase space is calculated here and returned as output of
#       the function Guderley.
    return y[1] - C1


def energy(x, y, gamma, lambda_, nu, energy0):
    """The function energy is used during numerical integrations as a
    consistency check. It computes the difference between the adiabatic energy
    integral given by Lazarus Eq. (2.7) and its initial value energy0. Its
    constancy is a good check on the accuracy of the integration.

    Args:
        x (float): Independent similarity variable; space-time position
        y (array): V, C, or R for i = 1, 2, or 3
        lambda_ (float): Similarity exponent
        nu (int): n - 1; space index
        energy0 (float): Initial value of the adiabatic energy integral
    
    Returns:
        float: The difference between the adiabatic energy integral evaluated at
            a particular space-time combination of x, V,C, and R.                  
    """
    q = 2.0 * (lambda_ - 1.0) / (nu + 1)

    if abs(x) >= 1.0e-8:
        return (y[1] / x)**2 * (1.0 + y[0])**q * y[2]**(q - gamma + 1.0) - energy0
    else:
        #.... It is impossible to compute C/x = dC/dx = 0/0 at x = 0,
        #       since the differential equations are singular there. We punt
        #       to avoid a machine infinity or NaN.
        return 0.0


def f(xorw, y):
    """This subroutine evaluates the differential equations given by Lazarus
    Eqs. (2.8), (2.9) and the R-equation.

    This subroutine (as opposed to the subroutine g) is for use with  the
    "Guderley" function. The difference between this subroutine and "g" is the
    inclusion of diagnostic write statements appearing in "g." Since this
    subroutine is used to evaluate Eqs. (2.8), (2.9) and the R-equation for
    choices of B that are incorrect (or inprecise), the corresponding diagnostic
    statements have been commented out.

    Args:
        xorw (float): The x or w values.
        y (array): Length-3 array of floats.

    Returns:
        array: The length-3 array of energy values.
    """
    V = y[0]
    C = y[1]
    Vp1 = V + 1.0
    C2 = C * C
    denom = (C2 - Vp1**2) * xorw * lambda_
    factor = (lambda_ - 1.0) / gamma

    num = np.zeros(3)
    num[0] = ((nu + 1)*V + 2.d0*factor)*C2 - V*Vp1*(V + lambda)
    num[1] = (1.0 + factor / Vp1) * C2 - 0.5 * nu * (gamma - 1.0) * V * Vp1 \
            - Vp1**2 - 0.5 * (lambda_ - 1.0)*((3.0 - gamma) * V + 2.0)
    #
    #.... The next equation is redundant, as the density can be found
    #       from energy conservation (2.7).  But we compute it so that we
    #       can use (2.7) as a consistency/accuracy check.
    #
    num[2] = - 2.0 * factor * C2 / Vp1 + V * (V + lambda_) - (nu + 1) * V * Vp1
    #
    #.... The diagnostic statements have been commented out.
    #
    #      if (abs(denom) .le. 1.d-6) then
    #        Near a singular point such as x = 0, dV/dx = dC/dx = 0/0.
    #        If this message is triggered, the calculation may eventually
    #        terminate prematurely.  The remedy is to very slightly loosen
    #        the tolerances abserr or relerr.
    #         write (*, 20) label(intno), xorw, denom
    #   20    format ('*** Warning, ', a, ' =', 1p, e23.15, '  denom =',
    #     &   e10.2)
    #      endif
    if intno == 2:
    #
    #.... Here df/dw = df/dx / dw/dx with dw/dx = -sigma*w/x.  The 1/x
    #       cancels the 1/x in df/dx, so the x's wash out.
    #
        denom = -denom * sigma

    yp = np.zeros(3)
    yp[0] = num[0] / denom
    yp[1] = C * num[1] / denom
    yp[2] = y[2] * num[2] / denom

    return yp


def g(t, y):
    """This subroutine evaluates the differential equations given by Lazarus
    Eqs. (2.8), (2.9) and the R-equation.

    This subroutine (as opposed to the subroutine f) is for use with the "sim"
    subroutine. The diagnostic statements have been left in here.
    """
    labsl = ['x', 'w']
    intno = 1
    V = y[0]
    C = y[1]
    Vp1 = V + 1.0
    C2 = C * C
    denom = (C2 - Vp1**2) * t * lambda_
    factor = (lambda_ - 1.0) / gamma

    num = np.zeros(3)
    num[0] = ((nu + 1) * V + 2.0 * factor) * C2 - V * Vp1 * (V + lambda_)
    num[1] = (1.0 + factor / Vp1) * C2 - 0.5 * nu * (gamma - 1.0) * V * Vp1 \
             - Vp1**2 - 0.5 * (lambda_ - 1.0) * ((3.0 - gamma) * V + 2.0)
    #
    #.... The next equation is redundant, as the density can be gotten
    #       from energy conservation (2.7).  But we compute it so that we
    #       can use (2.7) as a consistency/accuracy check.
    #
    num[2] = - 2.0 * factor * C2 / Vp1 + V * (V + lambda_) - (nu + 1) * V * Vp1

    if abs(denom) <= 1.0e-8:
        #
        #.... Near a singular point such as x = 0, dV/dx = dC/dx = 0/0.
        #       If this message is triggered, the calculation may eventually
        #       terminate prematurely.  The remedy is to very slightly loosen
        #       the tolerances abserr or relerr.
        #
        print(f'*** Warning, {label[intno-1]} = {t}, denom = {denom}')
    #	if (intno .eq. 2) then
    #
    #.... Here df/dw = df/dx / dw/dx with dw/dx = -sigma*w/x.  The 1/x
    #       cancels the 1/x in df/dx, so the x's wash out.
    #
    #	   denom = -denom*sigma
    #	endif
    yp = np.zeros(3)
    yp[0] = num[0] / denom
    yp[1] = C * num[1] / denom
    yp[2] = y[2] * num[2] / denom

    return yp


def Vdiff(w, y, yp):
    """This function computes the difference between V1 (value of V behind the
    reflected shock obtained by integrating in increasing x) and y(1) (that is
    obtained from integrating in increasing w (decreasing x)). The smaller the
    absolute value of this difference and the corresponding C difference are,
    the better the choice of B.

    Args:
        w (float): Transformed independent similarity variable
        y (array): y(i) = V, C, or R for i = 1, 2, or 3
        yp (array): RHS of Eqs. (2.8), (2.9), and the R-equation

    returns:
         float: Vdiff difference between V1 (from x) and corresponding  value
            from integrating in w.
    """
    return y[0] - V1


def state(r, rho0, n, gamma_d, lambda_d, B, targetxd):
    """This subroutine, given the various parameters computed in other
    parts of the driver program guderley_1D, integrates the governing ODEs up to
    a pre-specified point (targetx, which is computed in the guderley_1D driver
    program. It then transforms the similarity variable data at the targetx
    point to physical data at a particular space-time point.

    Lazarus Eqs. (2.5) are used to transform the similarity variables back to
    physical variable space in this subroutine. The results will be in terms of
    the "Lazarus Time," as opposed to "Caramana and Whalen" Time.
    """
    global gamma
    global lambda_
    global nu
    abserr = 6.0e-14
    relerr = 5.0e-13
    neqn = 3
    y = np.zeros(neqn)
    
    # pressure(C, R) = R*C*C/gamma
    nu = n - 1
    gamma = gammad
    lambda_ = lambdad
    targetx = targetxd
    #
    #.... Factors gamma + 1 and gamma - 1.
    #
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    #
    #.... Initializing the similarity variables at the position of the 
    #       converging shock wave. This is the starting point of all
    #       integrations of the governing equations.
    #
    y = np.zeros(3)
    y[0] = -2.0 / gp1
    y[1] = sqrt(2.0 * gamma * gm1) / gp1
    y[2] = gp1 / gm1
    t = -1.0
    #
    iflag = 1
    #
    #.... x < -1 represents the unshocked state (interior to hte converging
    #       shock wave), where the physical variables have constant values
    #       given by:
    #
    #          density = constant (specified by user)
    #          velocity = 0 (required)
    #          pressure = sound speed = 0 (required)
    #
    #       When a combination of space and time variables are specified
    #       such that x < -1, the constant state data is returned as output.
    #
    if targetx < -1.0:
        den  = rho0
        vel  = 0.0
        pres = 0.0
        snd  = 0.0
        sie  = 0.0
    #.... If -1 < x < 0, then we are behind the converging shock wave, and
    #       reflection has yet to occur. The integration of the governing
    #       ODEs is initiated at the position of the converging shock
    #       (x = -1) and carried through to whatever negative value of x
    #       that results from the specification of the space and time 
    #       variables.
    #
    elif targetx < 0.0 and targetx >= -1.0:
        while t < targetx:
            t, iflag = ode(g, neqn, y, t, targetx, relerr, abserr)
        #
        #.... Definition of the PHYSICAL pressure variable, as a function of the
        #       dimensionless similarity variables.
        #
        p = (((y[1] * r**(1.0 - lambda_)) \
            / (targetx * (-1.0) * lambda_))**2) \
            / (gamma * (1.0 / rho0) * (1.0 / y[2]))
        #
        #.... Writing of solution data.
        #
        den  = y[2] * rho0
        vel  = (y[0] * r**(1.0 - lambda_)) / (targetx * (-1.0) * lambda_)
        pres = p
        snd  = (y[1] * r**(1.0 - lambda_)) / (targetx * (-1.0) * lambda_)
        sie  = p / (gm1 * rho0 * y[2])
    #.... If 0 < x < B (the space-time position of the reflected shock
    #       wave), then we are upstream of the reflected shock wave. The 
    #       integration  of the governing ODEs is again started at position
    #       of the  convergent shock wave and integrated through x = 0 into
    #       a portion of the phase space representing the flow ahead of the 
    #       reflected shock wave. The integration terminates before the
    #       position x = B is reached.
    #
    elif targetx > 0.0 and targetx < B:
        while t < targetx:
            t, iflag = ode(g, neqn, y, t, targetx, relerr, abserr)
        #
        #.... Physical pressure variable.
        #
        p = (((y[1] * r**(1.0 - lambda_)) \
            / (targetx * (-1.0) * lambda_))**2) \
            / (gamma * (1.0/rho0) * (1.0 / y[2]))
        #
        #.... Writing of solution data.
        #
        den  = y[2] * rho0
        vel  = (y[0] * r**(1.0 - lambda_)) / (targetx * (-1.0) * lambda_)
        pres = p
        snd  = (y[1] * r**(1.0 - lambda_)) / (targetx * (-1.0) * lambda_)
        sie  = p / (gm1 * rho0 * y[2])
    #.... If B < x < infinity, then we are behind the reflected shock wave.
    #       The numerical integration starts at the position of the
    #       convergent shock wave as before and is carred through x = 0
    #       until x = B.
    #
    elif targetx > B:
        while t < B:
            t, iflag = ode(g, neqn, y, t, B, relerr, abserr)

        iflag = 1
        #
        #.... At x = B, the general-strength Rankine-Hugoniot conditions are
        #       applied, and we move to the other side of the reflected shock
        #       wave (just downstream).
        #
        C2 = y[1]**2
        V1 = gm1 * (1.0 + y[0]) / gp1 + 2.0 * C2 / (gp1 * (1.0 + y[0])) - 1.0
        y[1] = np.sign(sqrt(C2 + 0.5 * gm1 * ((1.0 + y[0])**2 - (1.0 + V1)**2)),
                       y[1])
        C1 = y[1]
        y[2] = y[2] * (1.0 + y[0]) / (1.0 + V1)
        y[0] = V1
        #
        #.... Numerical integration of the governing ODEs continues from x = B
        #       (with the similarity variables taking their shocked values)
        #       until the targetx point is reached.
        #
        while t < targetx:
            t, iflag = ode(g, neqn, y, t, targetx, relerr, abserr)

        #
        #.... Physical pressure variable.
        #
        p = (((y[1] * r**(1.0 - lambda_)) \
            / (targetx * (-1.0) * lambda_))**2) \
            / (gamma * (1.0 / rho0) * (1.0 / y[2]))
        #
        #....Writing of solution data.
        #
        den  = y[2] * rho0
        vel  = (y[0] * r**(1.0 - lambda_)) / (targetx * (-1.0) * lambda_)
        pres = p
        snd  = (y[1] * r**(1.0 - lambda_)) / (targetx * (-1.0) * lambda_)
        sie  = p / (gm1 * rho0 * y[2])

    return den, vel, pres, snd, sie
