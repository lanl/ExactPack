"""
This program generates physical variable (i.e. density, velocity pressure, sound
speed) solution data at a specified point in space and time for the converging
shock wave problem first solvedby G. Guderley.

It follows the notation given in [Lazarus1981]_

"guderley_1D" makes use of several subprograms and functions in order to
calculate the two nonlinear eigenvalues that appear in the problem: the
similarity exponent "lambda" and the reflected shock position (in similarity
variables) "B." Once these values are calculated the self-similar equations
governing the flow are solved for the dimensionless velocity V, sound speed C,
and density R, as a function of the similarity variable:

                 t
          x = -------- .
                lambda
               r

The program  also computes the result starting from x = infinity (Lazarus,
p. 330 ff.). The subroutine that transforms the similarity variable data is
self-contained in the subroutine "state."

This code is based on the driver code "guderley" first developed
by J. Bolstad of LLNL.

2007.08.01  S. Ramsey     Code reproduces correct results for gamma = 1.4

Code translated from Fortran to Python by J. Thrussell, 2022.09.23.
"""
import numpy as np
from math import sqrt
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

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

    It should be noted that the generalized Rankine-Hugonoit jump conditions
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
    global V1
    #.... The following parameters are adjustable, but it is not recommended
    #       that they be adjusted unless error messages are returned by 
    #       the function.
    abserr = 6.0e-11
    doublefreq 50
    relerr 5.0e-10
    aeroot 8.0e-16
    reroot 8.0e-16
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

        if (iflag != 2 and iflag != 4):
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
	energymax(2) = 0.0
	energymin(2) = 0.0

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
        if iflag != 2:
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
        
        if iflag != 2 and iflag != 7:
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