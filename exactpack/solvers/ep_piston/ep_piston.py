from ...base import ExactSolver, ExactSolution, Jump, JumpCondition
import warnings
import math
import scipy.optimize as sci_opt
import numpy as np


class EPpiston(ExactSolver):
    r""" Computes the solution to the Elastic-Plastic piston problem.
    The problem default values are selected to be consistent with the
    problem definition in Lieberman et. al. [Lieberman2019]_ for the infinitesimal
    hyperelasticity model.

    Default values are: G = 0.286 Mbar, Y = 0.0026 Mbar, :math:`\rho` = 2.79 g/cc,
    :math:`u_p` = 0.01 cm/us, :math:`\Gamma` = 2.0, c0 = 0.533 cm/us, s0 = 1.34, model = 'hyperIfin'
    """

    parameters = {
        'gamma': 'Gruneisen Gamma',
        'c0': 'Gruneisen parameter (cm/us)',
        's0': 'Gruneisen parameter',
        'model': 'hypo=Hypoelastic Model, hyperIfin=Hyperelastic Infinitesimal strain, hyperFin=Hyperelastic Finite strain',
        'G': "Shear Modulus of the Material (Mbars)",
        'Y': "Yield stress of the Material (Mbars)",
        'rho0': "Initial density of the material (g/cc)",
        'up': "Velocity of the piston (cm/us)",
        #'xmax': "maximum value of x allowed for exact solution (cm)",
        #'tmax': "maximum value of t allowed for exact solution (us)"
        }

    # Default parameter values

    gamma = 2.
    c0 = 0.533
    s0 = 1.34
    model = 'hyperIfin'
    G = 0.286
    Y = 0.0026
    rho0 = 2.79
    up = 0.01
    #xmax = 2.
    #tmax = 2


    def __init__(self, **kwargs):
        """Set default values if necessary, check for valid inputs,
        and compute general analytic solution for the parameter values.

        """
        super(EPpiston, self).__init__(**kwargs)

        # check for illegal input values

        if self.G <= 0:
            raise ValueError('Shear modulus must be > 0')

        if self.Y <= 0:
            raise ValueError('Yield Stress must be > 0')

        if self.rho0 <= 0:
            raise ValueError('Initial density must be > 0')

        if self.up < 0:
            raise ValueError('Piston velocity must be >= 0')
        #if self.up >= self.D/(self.gamma+1):
        #    raise ValueError(('Piston velocity must be less than ',
        #                      'C-J particle velocity'))

        #if self.tmax <= 0:
        #    raise ValueError('tmax must be >0 ')

        #Check for one of three elastic model types
        if self.model not in ('hypo','hyperIfin','hyperFin'):
            raise ValueError("Elastic model not correctly specified, must be 'hypo', 'hyperIfin' or 'hyperFin'.")

        rho0 = self.rho0
        G = self.G
        Y = self.Y
        gamma = self.gamma
        c0 = self.c0
        s0 = self.s0

        ### Solve values for the elastic wave ###

        #solve for deviatoric stress at yield
        self.sdev_y = (-2./3.)*Y
        #variation of model takes effect through calc of density at yield
        if self.model=='hypo':
            self.rho_y = self.rho_hypoYield(G,Y,rho0)
        elif self.model=='hyperIfin':
            self.rho_y = self.rho_hyperIfinYield(G,Y,rho0)
        elif self.model=='hyperFin':
            self.rho_y = self.rho_hyperFinYield(G,Y,rho0)

        #the rest is solving jump conditions with Hugoniot and EOS
        eta_y = 1.-(rho0/self.rho_y)
        Ph_y = (rho0*c0**2.*eta_y)/((1.-s0*eta_y)**2.)
        Eh_y = (eta_y*Ph_y)/(rho0*2.)
        self.e_y = ( (Ph_y-self.rho_y*gamma*Eh_y+(2./3.)*Y)*(self.rho_y-rho0) )\
        / ( 2.*rho0*self.rho_y-self.rho_y*gamma*(self.rho_y-rho0) )

        self.p_y = self.Gruneisen(rho0,gamma,c0,s0,self.rho_y,self.e_y)
        sg_y = self.sdev_y-self.p_y
        self.wv_el = math.sqrt( (self.rho_y*sg_y)/(rho0*(rho0-self.rho_y)))
        self.vel_y = self.wv_el*(self.rho_y-rho0)/self.rho_y

        ### Solve values for plastic wave ###

        #purely a jump condition solve
        #print(Plastic_Residual(wv_el,rho0,gamma,c0,s0,up,rho_y,p_y,sdev_y,e_y,vel_y))
        self.wv_pl = float(sci_opt.fsolve(self.Plastic_Residual,self.wv_el)[0])

        self.p2 = self.p_y + self.rho_y*(self.wv_pl-self.vel_y)*(self.up-self.vel_y)
        self.rho2 = self.rho_y*( (self.wv_pl-self.vel_y)/(self.wv_pl-self.up) )
        self.e2 = self.e_y + (1./(2.*self.rho_y*self.rho2))*\
        (self.p_y+self.p2-2*self.sdev_y)*(self.rho2-self.rho_y)

    def _run(self, xvec, t, xmax=None):
        r''' Evaluate the physical variables at (x,t).
        '''
        if xmax is None:
            xmax = max(xvec)

        # Initialize the physical variables
        vel_x = np.empty_like(xvec)
        p_x = np.empty_like(xvec)
        e_x = np.empty_like(xvec)
        rho_x = np.empty_like(xvec)
        sdev_x = np.empty_like(xvec)

        tmax = xmax/self.wv_el
        wv_el_x = self.wv_el*t
        wv_pl_x = self.wv_pl*t
        if t > tmax:
            raise ValueError('Elastic Wave went beyond xmax by',wv_el_x-xmax,', reduce time or increase xmax')
            #print('Elastic Wave went beyond xmax by ',wv_el_x-xmax)
            #print('Either reduce time or increase xmax for these parameters.')

        # Loop over x
        for i, x in enumerate(xvec):
            if x < wv_pl_x:
                vel = self.up
                p = self.p2
                e = self.e2
                rho = self.rho2
                sdev = self.sdev_y
            elif x > wv_pl_x and x < wv_el_x:
                vel = self.vel_y
                p = self.p_y
                e = self.e_y
                rho = self.rho_y
                sdev = self.sdev_y
            else:
                vel = 0.
                p = 0.
                e = 0.
                rho = self.rho0
                sdev = 0.

            vel_x[i] = vel
            p_x[i] = p
            e_x[i] = e
            rho_x[i] = rho
            sdev_x[i] = sdev
        return ExactSolution([xvec, rho_x, p_x, e_x, vel_x, sdev_x],
                             names=['position',
                                    'density',
                                    'pressure',
                                    'specific_internal_energy',
                                    'velocity',
                                    'deviatoric stress'
                                    ],
                             jumps=None)

    #Calculate density at yield for hypoelastic model
    def rho_hypoYield(self,G,Y,rho0):
    #def rho_hypoYield(G,Y,rho0):
        rho_y = rho0*math.exp(Y/(2.*G))
        return rho_y

    #Calculate density at yield for infinitesimal hyperelasitc model
    def rho_hyperIfinYield(self,G,Y,rho0):
        rho_y = rho0*(1.-Y/(2.*G))**(-1.)
        return rho_y

    #Calculate density at yield for finite hyperelastic model
    def rho_hyperFinYield(self,G,Y,rho0):
        def finite_yield(F,G,Y):
            R=(2./3.)*Y+(2./3.)*G*(F**(7./3.)-F**(-5./3.)+F**(-1.)-F)
            return R
        F = float(sci_opt.fsolve(finite_yield,1,(G,Y))[0])
        rho_y = rho0/F
        return rho_y

    #Calculate pressure from Gruneisen equation of state
    def Gruneisen(self,rho0,gamma,c0,s0,rho,e):
        eta = 1.-(rho0/rho)
        Ph = (rho0*c0**2.*eta)/((1.-s0*eta)**2.)
        Eh = (eta*Ph)/(rho0*2.)
        p = Ph + gamma*rho*(e-Eh)
        return p

    #Solve residual of the plastic shock wave Hugoniot
    def Plastic_Residual(self,wv_pl):
        p2 = self.p_y + self.rho_y*(wv_pl-self.vel_y)*(self.up-self.vel_y)
        rho2 = self.rho_y*( (wv_pl-self.vel_y)/(wv_pl-self.up) )
        e2 = self.e_y + (1./(2.*self.rho_y*rho2))*(self.p_y+p2-2*self.sdev_y)*(rho2-self.rho_y)

        p_eos = self.Gruneisen(self.rho0,self.gamma,self.c0,self.s0,rho2,e2)
        #eta = 1.-(rho0/rho2)
        #Ph = (rho0*c0**2.*eta)/((1.-s0*eta)**2.)
        #Eh = (eta*Ph)/(rho0*2.)
        #p_eos = Ph + gamma*rho2*(e2-Eh)

        R = p2-p_eos
        return R
