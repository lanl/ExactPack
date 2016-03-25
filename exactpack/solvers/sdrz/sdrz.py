r"""Exact solution for the Steady Detonation Reaction Zone problem

"""
from ...base import ExactSolver, ExactSolution, Jump, JumpCondition
import warnings
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint


class SteadyDetonationReactionZone(ExactSolver):
    r"""
    Computes the solution to the Steady Deonation Reaction Zone (SDRZ)
    problem.  The problem default values are selected to be consistent
    with the original problem definition in Fickett and Rivard [Fickett1974].

    The problem has a rate law defined as :math:`\frac{d\lambda}{dt}
    = 2\sqrt{1-\lambda}`

    Default values are: :math:`D=0.85 cm/\mu s, \rho=1.6,
    \gamma=3.0`
    """

    parameters = {
        'geometry' : '1=planar',
        'D': "Detonation velocity of the HE (cm/us)",
        'rho_0': "Initial density of the HE (g/cc)",
        'gamma' : 'adiabatic index'
        }

    #
    #  Define default values
    #

    geometry = 1 # Planar, 1D (axial)
    D = 0.85     # Detonation velocity of HE
    rho_0 = 1.6  # Initial density of HE
    gamma = 3.0  # Adiabatic index

    def __init__(self, **kwargs):
        """Set default values if necessary, check for valid inputs,
        and compute polygon vertices for the parameter values.

        """
        super(SteadyDetonationReactionZone, self).__init__(**kwargs)

        # Assign default values if necessary, then
        # check for valid input parameter values.

        if self.D <= 0:
            raise ValueError('Detonation velocity must be >=0')

        if self.rho_0 <= 0:
            raise ValueError('Initial density must be >=0')

        if self.gamma <= 0:
            raise ValueError('Adiabatic index must be >=0')

        if self.geometry != 1:
            raise ValueError('Problem is axial only, geometry must be set to 1')


        #  Define time-independent solution quantities

        self.Dj = self.D      # Test problem assumes Dj=D
        self.f = (self.D/self.Dj)**2
        self.Pj = self.rho_0 * self.Dj**2 / (self.gamma+1)
        self.rhoj = self.rho_0 * (self.gamma+1)/self.gamma

        return

    def run_tvec(self, tvec, useExactLambda=True):
        ''' Evaluates the solution to the Steady Detonation Reaction Zone problem
        at a given vector of times tvec. If useExactLambda is True, the solution
        uses an exact formulation of the reaction progress variable, otherwise
        the rate law is integrated numerically.'''

        # Evaluate the reaction progress at vector of times [t]

        if useExactLambda:

            # use exact fomulation of the reaction progress variable
    
            lamvec = tvec * (2.0 - tvec)

            # Correct reaction progress to be monotonic

            lamvec = np.maximum.accumulate(lamvec)

        else:

            # numerically integrate the reaction progress variable

            # define the reaction rate equation as a function

            def reactionrate(lam, tvec):
                if lam<1.0 and lam>=0.0:
                    dldt = 2*np.sqrt(1-lam)
                else:
                    dldt = 0.0
                return dldt

            # perform the integration at the vector of time points tvec

            lam0 = 0. # set initial condition for reaction progress at t=0

            lamvec = odeint(reactionrate, lam0, tvec)

            lamvec = lamvec[:,0]

        # Correct reaction progress to stay at 1.0 after it first hits 1.0

        if np.any(lamvec >= 1.0):
            ibf1 = np.where(lamvec>=1.0)[0][0]
            for i in range(ibf1,len(lamvec)):
                lamvec[i] = 1.0

        # Evaluate the solution variables at vector of times [t]

        gvec = np.sqrt(1 - lamvec / self.f)
        pvec = self.f * self.Pj * (1.0 + gvec)
        rhovec = self.rhoj * self.gamma / (self.gamma - gvec)
        uvec = (1.0 - self.rho_0 / rhovec)*self.D
        csvec = np.sqrt(self.gamma * pvec / rhovec)

        xvec_rel = np.empty_like(tvec) # initialize xvec_rel (position relative to shock front)

        # Find index where t first equals or exceeds 1.0
        if np.any(tvec >= 1.0):
            it1 = np.where(tvec>=1.0)[0][0]

        for i,t in enumerate(tvec):
            if t <= 1.0:
                xvec_rel[i] = self.rho_0 * self.Dj / self.rhoj *\
                ((1.0 - 1.0/self.gamma)*t + t**2 /
                        (2.0 * self.gamma))
            else:
                xvec_rel[i] = xvec_rel[it1] + (self.D - uvec[it1]) * (t-1.0)

        xvec_abs = self.D * tvec[-1] - xvec_rel   # Particle position in absolute coordinates

        return ExactSolution(
            [xvec_abs, pvec, uvec, rhovec, csvec, lamvec, xvec_rel],
            names=['position','pressure','velocity',
                'density','sound_speed','reaction_progress','position_relative'],
            jumps=None)

    def _run(self, xvec, t, NP=201):
        ''' Evaluates the solution to the Steady Detonation Reaction Zone problem
        at a given time t at a vector of absolute positions xvec. Defaults to
        201 time points unless NP is specified otherwise.

        '''

        # Create a vector of times t

        tvec = np.linspace(0., t, NP)

        tsolution = self.run_tvec(tvec)

        xsolution = dict()

        varnames = ['pressure','velocity','density','sound_speed',
                        'reaction_progress','position_relative']

        # interpolate / extrapolate solution to xvec

        # initialize variables

        for var in varnames:
            xsolution[var] = np.zeros_like(xvec)

        # find the indices of xvec that lie above the upper end of xvec_abs
        #  (highest value of position is first entry in array)


        imask = xvec > tsolution['position'][0]

        # assign the variables to have the undisturbed physical state at 
        #  these indices

        xsolution['density'][imask] = np.ones(sum(imask))*self.rho_0
        #    all other variables stay at the initialized state of zero

        #  find the indices of xvec that lie below the lower end of 
        #   xvec_abs

        kmask = xvec < tsolution['position'][-1]

        # assign the variables to have the leftmost physical state at 
        #  these indices

        for var in varnames:
            xsolution[var][kmask] = np.ones(sum(kmask))*tsolution[var][-1]
        
        #
        # assign all other variables by interpolation from tsolution
        # Need to reverse order of variables in tsolution, as they are
        #  in descending order of absolute position
        #

        jmask = (imask==False) & (kmask==False)

        for var in varnames:
            interpfcn = interp1d(tsolution['position'][::-1],tsolution[var][::-1])
            xsolution[var][jmask] = interpfcn(xvec[jmask])

        #
        # assign xvec into the solution object
        #

        xsolution['position'] = xvec

        return xsolution

