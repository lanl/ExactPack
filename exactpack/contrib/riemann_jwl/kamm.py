
"""A Fortran based Riemann solver from Jim Kamm.
"""

from ...base import ExactSolver, ExactSolution
from _kamm import riemann_kamm_jwl


class RiemannJWL(ExactSolver):
    """ Computes the solution to the Riemann problem with a JWL EOS.
    """

    parameters = {
        'interface_loc': 'initial interface location :math:`r_0`',
        'rhol': 'initial left mass density :math:`\rho_{\rm l}`',
        'pl': 'initial left pressure :math:`p_{\rm l}`',
        'ul': 'initial left fluid velocityy :math:`u_{\rm l}`',
        'rhor': 'initial right mass density :math:`\rho_{\rm r}`',
        'pr': 'initial right pressure :math:`p_{\rm r}`',
        'ur': 'initial right fluid velocity :math:`u_{\rm r}`',
        'rho0l': 'initial left X :math:`X_{\rm l}',
        'sie0l': 'initial left X :math:`X_{\rm l}',
        'gammal': 'initial left X :math:`X_{\rm l}',
        'bigal': 'initial left X :math:`X_{\rm l}',
        'bigbl': 'initial left X :math:`X_{\rm l}',
        'r1l': 'initial left X :math:`X_{\rm l}',
        'r2l': 'initial left X :math:`X_{\rm l}',
        'rho0r': 'initial left X :math:`X_{\rm r}',
        'sie0r': 'initial left X :math:`X_{\rm r}',
        'gammar': 'initial left X :math:`X_{\rm r}',
        'bigar': 'initial left X :math:`X_{\rm r}',
        'bigbr': 'initial left X :math:`X_{\rm r}',
        'r1r': 'initial left X :math:`X_{\rm r}',
        'r2r': 'initial left X :math:`X_{\rm r}'
        }

    # default parameters: the Lee problem
    interface_loc = 50.0
    rhol = 0.9525
    pl = 1.
    ul = 0.
    rhor = 3.810
    pr = 2.
    ur = 0.
    rho0l = 1.905
    sie0l = 0.
    gammal = 0.8938
    bigal = 6.321e2
    bigbl = -4.472e-2
    r1l = 1.13e1
    r2l = 1.13
    rho0r = 1.905
    sie0r = 0.
    gammar = 0.8938
    bigar = 6.321e2
    bigbr = -4.472e-2
    r1r = 1.13e1
    r2r = 1.13

    def _run(self, r, t):

        rho, p, u, sound, sie, entropy = riemann_kamm_jwl(time=t, x=r,
                                         xd0=self.interface_loc,
                                         rhol=self.rhol,
                                         pl=self.pl,
                                         ul=self.ul,
                                         rhor=self.rhor,
                                         pr=self.pr,
                                         ur=self.ur,
                                         rho0l=self.rho0l,
                                         sie0l=self.sie0l,
                                         gammal=self.gammal,
                                         bigal=self.bigal,
                                         bigbl=self.bigbl,
                                         r1l=self.r1l,
                                         r2l=self.r2l,
                                         rho0r=self.rho0r,
                                         sie0r=self.sie0r,
                                         gammar=self.gammar,
                                         bigar=self.bigar,
                                         bigbr=self.bigbr,
                                         r1r=self.r1r,
                                         r2r=self.r2r)

        return ExactSolution([r, rho, p, u, sound, sie, entropy],
                             names=['position',
                                    'density',
                                    'pressure',
                                    'velocity',
                                    'sound',
                                    'energy',
                                    'entropy'])
# Lee
# xmin   = 0.d0
# xmax   = 100.d0
# xd0    = 50.d0
# time   = 20.d0


class RiemannJWLLee(RiemannJWL):
    """JWL Riemann for Shyue problem
    """

    parameters = {
        'interface_loc': 'initial interface location :math:`r_0`',
        'rhol': 'initial left mass density :math:`\rho_{\rm l}`',
        'pl': 'initial left pressure :math:`p_{\rm l}`',
        'ul': 'initial left fluid velocityy :math:`u_{\rm l}`',
        'rhor': 'initial right mass density :math:`\rho_{\rm r}`',
        'pr': 'initial right pressure :math:`p_{\rm r}`',
        'ur': 'initial right fluid velocity :math:`u_{\rm r}`',
        'rho0l': 'initial left X :math:`X_{\rm l}',
        'sie0l': 'initial left X :math:`X_{\rm l}',
        'gammal': 'initial left X :math:`X_{\rm l}',
        'bigal': 'initial left X :math:`X_{\rm l}',
        'bigbl': 'initial left X :math:`X_{\rm l}',
        'r1l': 'initial left X :math:`X_{\rm l}',
        'r2l': 'initial left X :math:`X_{\rm l}',
        'rho0r': 'initial left X :math:`X_{\rm r}',
        'sie0r': 'initial left X :math:`X_{\rm r}',
        'gammar': 'initial left X :math:`X_{\rm r}',
        'bigar': 'initial left X :math:`X_{\rm r}',
        'bigbr': 'initial left X :math:`X_{\rm r}',
        'r1r': 'initial left X :math:`X_{\rm r}',
        'r2r': 'initial left X :math:`X_{\rm r}'
        }

    # parameters for the Lee problem
    interface_loc = 50.0
    rhol = 0.9525
    pl = 1.
    ul = 0.
    rhor = 3.810
    pr = 2.
    ur = 0.
    rho0l = 1.905
    sie0l = 0.
    gammal = 0.8938
    bigal = 6.321e2
    bigbl = -4.472e-2
    r1l = 1.13e1
    r2l = 1.13
    rho0r = 1.905
    sie0r = 0.
    gammar = 0.8938
    bigar = 6.321e2
    bigbr = -4.472e-2
    r1r = 1.13e1
    r2r = 1.13

# Shyue
# xmin   = 0.d0
# xmax   = 100.d0
# xd0    = 50.d0
# time   = 12.d


class RiemannJWLShyue(RiemannJWL):
    """JWL Riemann for Shyue problem
    """

    parameters = {
        'interface_loc': 'initial interface location :math:`r_0`',
        'rhol': 'initial left mass density :math:`\rho_{\rm l}`',
        'pl': 'initial left pressure :math:`p_{\rm l}`',
        'ul': 'initial left fluid velocityy :math:`u_{\rm l}`',
        'rhor': 'initial right mass density :math:`\rho_{\rm r}`',
        'pr': 'initial right pressure :math:`p_{\rm r}`',
        'ur': 'initial right fluid velocity :math:`u_{\rm r}`',
        'rho0l': 'initial left X :math:`X_{\rm l}',
        'sie0l': 'initial left X :math:`X_{\rm l}',
        'gammal': 'initial left X :math:`X_{\rm l}',
        'bigal': 'initial left X :math:`X_{\rm l}',
        'bigbl': 'initial left X :math:`X_{\rm l}',
        'r1l': 'initial left X :math:`X_{\rm l}',
        'r2l': 'initial left X :math:`X_{\rm l}',
        'rho0r': 'initial left X :math:`X_{\rm r}',
        'sie0r': 'initial left X :math:`X_{\rm r}',
        'gammar': 'initial left X :math:`X_{\rm r}',
        'bigar': 'initial left X :math:`X_{\rm r}',
        'bigbr': 'initial left X :math:`X_{\rm r}',
        'r1r': 'initial left X :math:`X_{\rm r}',
        'r2r': 'initial left X :math:`X_{\rm r}'
        }

    # parameters for the Shuye problem
    interface_loc = 50.0
    rhol = 0.9525
    pl = 1.
    ul = 0.
    rhor = 3.810
    pr = 2.
    ur = 0.
    rho = 1.7
    pl = 10.
    ul = 0.
    rhor = 1.
    pr = 0.5
    ur = 0.
    rho0l = 1.84
    sie0l = 0.
    gamma0l = 0.25
    bigal = 8.545
    bigbl = 0.205
    r1l = 4.6
    r2l = 1.35
    rho0r = 1.84
    sie0r = 0.
    gamma0r = 0.25
    bigar = 8.545
    bigbr = 0.205
    r1r = 4.6
    r2r = 1.35
