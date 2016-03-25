r"""Python based solvers for HE burn time when using programmed burn models.

This suite of solvers calculate HE burn times for programmed burn models.

The Kenamond High Explosive Problem Set is a series of three problems
designed to test the burn table solution (HE light times) generated
for programmed burn simulations. The suite of test problems has exact
solutions in 2D and 3D [Kenamond]_.

It should be understood that these burn time calculations are purely
geometry-based solutions. They do not account for HE behaviors such as
shock formation time, inert boundary behaviors or behavior at boundaries
between two HEs.

.. [Kenamond] Kenamond, M. A., HE Burn Table Verification Problems,
   LA-UR 11-03096, 2011.

"""

from kenamond1 import Kenamond1
from kenamond2 import Kenamond2
from kenamond3 import Kenamond3
