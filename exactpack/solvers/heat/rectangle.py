r"""The Single Component Rectangle. This test problem consists of a rectangular region of length a along the
x-direction and length b along the y-direction. The boundary conditions are given by keeping the bottom of
the rectangle at zero temperature and the top at a fixed temperature. The sides of the rectangle use zero
heat flux conditions. The initial temperature of the rectangle is taken to vanish.

"""

import numpy as np

from ...base import ExactSolver, ExactSolution


class Rectangle(ExactSolver):

    r"""Computes the solution to a rectangular heat flow problem.
    """

    parameters = {
        'kappa': "Thermal diffusivity",
        'Nsum': "Number of terms to include in the sums.",
        'a': "Length of the rectangle in the x-direction",
        'b': "Length of the rectangle in the y-direction",
        'Ttop': "Temperature BC on the top of the rectangle at y=b",
        'NonHomogeneousOnly': "If True, then compute only the static nonhomogeneous solution. "
        }
# NonHomogeneousOnly = False is the default, i.e. plot the dynamic solution on top of the static.
    kappa = 1.0
    Nsum = 100
    a = 2.0
    b = 2.0
    Ttop = 1.0
    NonHomogeneousOnly = False

    def __init__(self, **kwargs):

        super(Rectangle, self).__init__(**kwargs)

    def _run(self, xylist, t):

        x = xylist[0]
        y = xylist[1]

        # specific nonhomogeneous contribution \bar T(x,y)
        tempnonhom = 0
        temperature = 0
        for n in xrange(1, self.Nsum):
            kn = n * np.pi / self.a
            Ttopn = 2 * self.Ttop * (1 - (-1)**n) / (n * np.pi)
            tmp = Ttopn * np.sin(kn * x) * np.sinh(kn * y) / np.sinh(kn * self.b)
            tempnonhom += tmp
        # general homogeneous contribution \tilde T(x, y, t)
        if self.NonHomogeneousOnly == False:
            for n in xrange(0, self.Nsum):
                kn = (2 * n + 1) * np.pi / self.a
                for m in xrange(1, self.Nsum):
                    km = m * np.pi / self.b
                    alpha2 = kn**2 + km**2
                    Anm = 4 * self.Ttop * 2 * (-1)**m * (m / (2 * n + 1)) / alpha2 / self.b**2
                    tmp = Anm * np.sin(kn * x) * np.sin(km * y) * np.exp(-self.kappa * alpha2 * t)
                    temperature += tmp

        # add homogeneous and nonhomogeneous
        temperature = temperature + tempnonhom

        return ExactSolution([x, y, temperature],
                    names=['position_x',
                    'position_y',
                    'temperature'],
                    jumps=[]
                    )
