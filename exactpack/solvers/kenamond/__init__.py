r"""Python based solvers for high explosive burn times when using programmed
burn models.

This suite of solvers calculates high explosive (HE) burn times for programmed burn
models [Kenamond2011]_. The Kenamond HE problems are a series of problems
designed to test the burn table solution (HE light times) generated for programmed
burn simulations. The suite of test problems has exact solutions in 2D and 3D.

It should be understood that these burn time calculations are purely
geometry-based solutions. They do not account for HE behaviors such as
shock formation time, inert boundary behaviors, or behavior at boundaries
between two high explosives.


"""

from .kenamond1 import Kenamond1
from .kenamond2 import Kenamond2
from .kenamond3 import Kenamond3
