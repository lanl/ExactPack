"""Unittests for the Guderley solver.
"""

import pytest

import numpy

from exactpack.solvers.guderley.eexp import eexp
from exactpack.solvers.guderley.ramsey import get_shock_position
from exactpack.solvers.guderley.guderley import Guderley


class TestGuderleyLambda():
    """Test the calculation of the Lambda value by the eexp routine.
    Results taken from [Guderley2012]_
    """
    # gamma = 5/3 case not tested as it takes several minutes to run. 
    test_data = [
        (2, 1.4, 0.835323192),
        # (2, 5.0/3.0, 0.815624901),
        (2, 2.0, 0.800112351),
        (2, 3.0, 0.775666619),
        (2, 6.0, 0.751561684),
        (3, 1.4, 0.717174501),
        # (3, 5.0/3.0, 0.688376823),
        (3, 2.0, 0.667046070),
        (3, 3.0, 0.636410594),
        (3, 6.0, 0.610339148),
    ]
    
    @pytest.mark.parametrize("n, gamma, alpha", test_data)
    def test_eexp(self, n, gamma, alpha):
        lambda_ = eexp(n, gamma)
        assert 1.0 / lambda_ == pytest.approx(alpha)


class TestGuderleyShockPosition():
    """Test the calculation of B by the get_shock_position routine.
    Alpha values and Results taken from [Guderley2012]_
    """
    test_data = [
        (2, 1.4, 0.835323192, 2.815610935),
        (2, 5.0/3.0, 0.815624901, 1.694792696),
        (2, 2.0, 0.800112351, 1.199630409),
        (2, 3.0, 0.775666619, 0.763159927),
        (2, 6.0, 0.751561684, 0.540791267),
        (3, 1.4, 0.717174501, 2.688492680),
        (3, 5.0/3.0, 0.688376823, 1.547894929),
        (3, 2.0, 0.667046070, 1.077253818),
        (3, 3.0, 0.636410594, 0.693969704),
        (3, 6.0, 0.610339148, 0.531821969),
    ]
    
    @pytest.mark.parametrize("n, gamma, alpha, posn", test_data)
    def test_shock_position(self, n, gamma, alpha, posn):
        lambda_ = 1.0 / alpha
        B = get_shock_position(n, gamma, lambda_)
        assert B == pytest.approx(posn, rel=5.0e-5)


class TestGuderleyRamseyGamma3():
    """Tests for the Guderley problem 
    :class:`exactpack.solvers.guderley.ramsey.Guderley`.
    """
    solver = Guderley(gamma=3.0)
    solution = solver(numpy.array([1.0]), 0.0)

    def test_density(self):
        """Regression test for density."""
        assert self.solution.density[0] == pytest.approx(2.0)

    def test_velocity(self):
        """Regression test for velocity."""
        assert self.solution.velocity[0] == pytest.approx(-0.3182052970545358)

    def test_pressure(self):
        """Regression test for pressure."""
        assert self.solution.pressure[0] == pytest.approx(0.20250922214713074)

    def test_speed_of_sound(self):
        """Regression test for the speed of sound."""
        assert self.solution.sound[0] == pytest.approx(0.5511477417360032)

    def test_sie(self):
        """Regression test for specific internal energy."""
        assert self.solution.sie[0] == pytest.approx(0.050627305536782685)
