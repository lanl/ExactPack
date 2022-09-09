r"""A pure Python Riemann solver based on Toro. [Toro2009]_"""
from ...base import ExactSolver, ExactSolution
from scipy.optimize import root
from math import sqrt
from functools import partial
from numpy import array


class Riemann(ExactSolver):
    """ Computes the solution to the Riemann problem.
    """

    parameters = {
        'interface_loc': 'initial interface location :math:`r_0`',
        'gammal': r'left specific heat ratio :math:`\gamma \equiv c_p/c_v`',
        'gammar': r'right specific heat ratio :math:`\gamma \equiv c_p/c_v`',
        'rhol': 'density on left in Eq. :eq:`riemannL`',
        'pl': 'pressure on left in Eq. :eq:`riemannL`',
        'ul': 'velocity on left in Eq. :eq:`riemannL`',
        'rhor': 'density on right in Eq. :eq:`riemannR`',
        'pr': 'pressure on right in Eq. :eq:`riemannR`',
        'ur': 'velocity on right in Eq. :eq:`riemannR`',
        }

    geometry = 1
    gammal = 1.4
    gammar = 1.4
    interface_loc = 0.5
    rhol = 1.0
    pl = 1.0
    ul = 0.0
    rhor = 0.125
    pr = 0.1
    ur = 0.0
                  
    def __init__(self, **kwargs):
        """Initialize the solver with left and right initial states.
        
        For the ideal gas equation of state with a constant specific heat
        ratio, the Riemann problem is uniquely defined by left and right
        initial states. The entire solution may then be constructed as part
        of the initialization of the Riemann object, such that only queries of
        the exact solution at specific positions and times must be handled at
        run-time. 
        
        Note that, unlike the more sophisticated Riemann solvers appearing in
        the exactpack-fortran package, this solver permits only the ideal gas
        equation of state with constant specific heat ratio."""
        super(Riemann, self).__init__(**kwargs)
        left = RiemannState(self.pl,self.ul,self.rhol)
        right = RiemannState(self.pr,self.ur,self.rhor)
        if self.gammal == self.gammar:
            self.gamma = self.gammal
        else:
            raise Exception("Multiple values for gamma unsupported! Additional functionality provided in exactpack-fortran package.")
        self.g = self.gamma
        self.left = left
        self.right = right
        a = self.sound_speed
        # check for formed vacuum
        if left.d > 0 and right.d > 0 and self.pressure_positivity():
            pstar = self.pressure_star().x[0]
            ustar = right.u + self.du(pstar, right)
            dstarL = self.density_f(pstar, left)
            dstarR = self.density_f(pstar, right)
        else:
            pstar = 0.
            ustar = float('nan') # best I can come up with
            dstarL = 0.
            dstarR = 0.
        starL = RiemannState(pstar, ustar, dstarL)
        starR = RiemannState(pstar, ustar, dstarR)

        # compute wave speeds
        self.wave_speeds, self.state_functions = self.nonlinear_waves(
            left, starL, 'left')
        self.state_functions.append(starL)
        if pstar == 0.: # Cavitation vaccuum
            pass
        else: 
            self.wave_speeds.append(ustar) # Contact discontinuity
            self.state_functions.append(starR)
        ws2, sf2 = self.nonlinear_waves(right, starR, 'right')
        self.wave_speeds += reversed(ws2)
        self.state_functions += reversed(sf2)

    def _run(self, r, t):
        """Compute the state for an array of positions at some time."""
        d = 0*r
        u = 0*r
        p = 0*r
        for ind, pos in enumerate(r):
            state = self.query(pos, t, r0=self.interface_loc)
            d[ind] = state.d
            u[ind] = state.u
            p[ind] = state.p
        return ExactSolution([r, d, u, p,], names=['speed', 'density',
                                                   'velocity', 'pressure'])
                             

    def query(self, r, t=1, r0=0):
        """Compute the state of the solution at a given point and time."""
        s = (r-r0)/t
        for ws, sf in zip(self.wave_speeds, self.state_functions):
            if s < ws: return sf(s)
        return self.state_functions[-1](s)

    def nonlinear_waves(self, state0, state_star, leftright):
        """Compute the wave speeds and state functions for a nonlinear wave.
        
        The portion of the solution associated with a given nonlinear wave
        is determined by an ordered set of wave speeds representing the
        discontinuities of the wave (i.e. the shock, or the head and tail
        of the rarefaction wave) and a corresponding set of functions that
        return the state of the solution between these discontinuities.
        
        In order to minimize duplicated code and simplify testing of the
        solver, it is convenient to compute these all with a single routine.
        Using a plus/minus operation, the formulas for these states and wave
        speeds can be represented with a single expression, and this is 
        represented in code by the leftright argument."""
        if leftright == 'left': pm = -1
        if leftright == 'right': pm = 1
        a = self.sound_speed
        gamma = self.g
        wave_speeds = []
        state_functions = [state0]
        if state0.p == 0.:
            pass # No wave from this side
        elif state_star.p < state0.p: # Rarefaction wave
            wave_speeds.append(state0.u + pm*a(state0))
            state_functions.append(partial(
                self.fan_state, state=state0, pm=pm))
            if state_star.p == 0.: # Emergent vaccuum state
                wave_speeds.append(state0.u - pm*2/(gamma-1)*a(state0))
            else: # Normal rarefaction
                wave_speeds.append(state_star.u + pm*a(state_star))
        else: # Shock
            wave_speeds.append(
                state0.u + pm*self.shock_speed_diff(state_star.p, state0))
        return wave_speeds, state_functions

    def pressure_star(self):
        """Compute the pressure star region.
        
        The pressure and velocity between the two nonlinear waves are 
        defined in terms of a nonlinear, transcendental equation. This
        routine solves for pressure such that velocity is constant
        across the contact discontinuity."""

        p = self.guess_p()
        def fun(p):
            return (self.du(p, self.left) + self.du(p, self.right) + 
                    self.right.u - self.left.u)
        out = root(fun, p)
        return out

    def guess_p(self):
        """Compute initial guess for pressure between the nonlinear waves."""
        left = self.left
        right = self.right
        g = self.g
        a = self.sound_speed
        # PVRS Riemann solver
        quser = 2.
        cup = .25*(left.d + right.d)*(a(left) + a(right))
        ppv = .5*(left.p + right.p) + .5*(left.u - right.u)*cup
        ppv = max(0., ppv)
        pmin = min(left.p, right.p)
        pmax = max(left.p, right.p)
        qmax = pmax/pmin
        if qmax <= quser and (pmin <= ppv and ppv <= pmax): # Use PVRS result
            out = ppv
        else: 
            if ppv < pmin: # Use Two-rarefaction solver
                pq = (left.p/right.p)**((g-1)/(2*g))
                um = (pq*left.u/a(left) + right.u/a(right) + 
                      2/(g-1)*(pq-1))/(pq/a(left)+1/a(right))
                ptl = 1 + .5*(g-1)*(left.u-um)/a(left)
                ptr = 1 + .5*(g-1)*(um - right.u)/a(right)
                out = .5*(left.p*ptl**(2*g/(g-1)) + right.p*ptr**(2*g/(g-1)))
            else: # Use two-shock solver with PVRS as guess
                gel = ((2/(g+1)/left.d)/((g-1)/(g+1)*left.p + ppv))**.5
                ger = ((2/(g+1)/right.d)/((g-1)/(g+1)*right.p + ppv))**.5
                out = (gel*left.p + ger*right.p - (right.u - left.u))/(gel+ger)
        return out

    def pressure_positivity(self):
        """Check if the initial conditions generate a vaccuum state."""
        left = self.left
        right = self.right
        g = self.g
        out = right.u - left.u <= 2./(g - 1) * (
            self.sound_speed(left) + self.sound_speed(right))
        return out

    def sound_speed(self, state):
        return sqrt(self.g*state.p/state.d)

    def density_f(self, p, state):
        """Compute the density based on 'upstream' state and pressure."""
        if p > state.p: # Shock
            gamma_fac = (self.g - 1)/(self.g + 1)
            out = state.d*(p/state.p+gamma_fac)/(gamma_fac*p/state.p + 1)
        else: # Rarefaction
            out = state.d*(p/state.p)**(1./self.g)
        return out

    def du(self, p, state):
        """Compute the velocity change across a nonlinear wave."""
        g = self.g
        if p > state.p: # Shock
            a = 2/(state.d*(g+1))
            b = state.p*(g-1)/(g+1)
            out = (p - state.p)*sqrt(a/(p+b))
        else: # Rarefaction
            out = 2 * self.sound_speed(state)/(g - 1)*(
                (p/state.p)**((g - 1)/(2*g)) - 1)
        return out

    def shock_speed_diff(self, p, state):
        """Compute difference between velocity and shock speed."""
        g = self.g
        out = self.sound_speed(state)*sqrt(
            (g+1)/(2*g)*p/state.p+(g-1)/(2*g))
        return out

    def fan_state(self, speed, state, pm):
        """Compute the state inside a rarefaction wave."""
        g = self.g
        a = self.sound_speed
        d = state.d*(
            2/(g+1) - pm*(g-1)/((g+1)*a(state))*(state.u-speed))**(2/(g-1))
        u = 2/(g+1)*(-pm*a(state)+(g-1)*.5*state.u+speed)
        p = state.p*(
            2/(g+1) - pm*(g-1)/((g+1)*a(state))*(state.u-speed))**(2*g/(g-1))
        return RiemannState(p, u, d)


class RiemannState(object):
    def __init__(self, p, u, d):
        self.p = p
        self.u = u
        self.d = d

    def __call__(self, speed):
        return self
