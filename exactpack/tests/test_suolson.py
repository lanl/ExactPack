"""Unit tests for the Su-Olson solver.
"""

import pytest
import numpy as np

from exactpack.solvers.suolson.suolson import SuOlson
from exactpack.solvers.suolson.timmes import usolution, vsolution


class TestSuOlsonTimmes():
    r"""Regresstion tests for :class:`exactpack.solvers.suolson.timmes.SuOlson`.

    Comparisions are made at :math:`r=0.1` and :math:`t=10^{-9}`.
    """
    solver = SuOlson(trad_bc_ev=1.0e3, opac=1.0)
    data = np.linspace(0, 20.0, 4)
    soln = solver(data, 1.e-9)

    def test_invalid_time(self):
        """There is no valid solution at :math:`t=0`"""
        soln = self.solver(self.data, 0.0)
        for quant in ['temperature_mat', 'temperature_rad']:
            assert np.all(np.isnan(soln[quant]))

    def test_mat_temperature(self):
        """SuOlson problem: mat temperature"""
        expected = [955.9875670217054, 388.14357235758376,
                    72.16540048091636, 7.158933356184126]
        result = self.soln.temperature_mat
        np.testing.assert_allclose(result, expected, rtol=1.0e-2)

    def test_radiation_temperature(self):
        """SuOlson problem: radiation temperature"""
        expected = [956.7434855255531, 396.89673904429804,
                    76.45509373805481, 7.802002997462728]
        result = self.soln.temperature_rad
        np.testing.assert_allclose(result, expected, rtol=1.0e-2)


class TestSuOlsonDimensionlessUEps0p1():
    """Compare the dimensionless u-solution of the Python solver with eps = 0.1

    Results are compared against table 1 in [Su1996]_
    """
    xpos = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0])
    test_data = [
        (0.001, np.array([0.09039, 0.03241, 0.00360, 0.00001, 0., 0., 0., 0., 0., 0., 0., 0.])),
        (0.003, np.array([0.14768, 0.08521, 0.03042, 0.00293, 0.00012, 0., 0., 0., 0., 0., 0., 0.])),
        (0.01, np.array([0.23997, 0.17979, 0.11006, 0.04104, 0.01214, 0.00268, 0., 0., 0., 0., 0., 0.])),
        (0.03, np.array([0.34328, 0.28962, 0.22063, 0.13345, 0.07545, 0.03958, 0.00014, 0., 0., 0., 0., 0.])),
        (0.1, np.array([0.43876, 0.39240, 0.33075, 0.24629, 0.18087, 0.13089, 0.01274, 0.00005, 0., 0., 0., 0.])),
        (0.3, np.array([0.48556, 0.44289, 0.38544, 0.30500, 0.24062, 0.18922, 0.04167, 0.00238, 0.00008, 0., 0., 0.])),
        (1, np.array([0.55182, 0.51412, 0.46198, 0.38541, 0.32046, 0.26564, 0.08147, 0.00961, 0.00097, 0.00009, 0., 0.])),
        (3, np.array([0.66334, 0.63458, 0.59295, 0.52771, 0.46773, 0.41298, 0.18266, 0.03844, 0.00678, 0.00105, 0.00003, 0.])),
        (10, np.array([0.79420, 0.77644, 0.75004, 0.70679, 0.66458, 0.62353, 0.40703, 0.17142, 0.06123, 0.01909, 0.00135, 0.00007])),
        (30, np.array([0.87731, 0.86669, 0.85082, 0.82451, 0.79839, 0.77252, 0.62363, 0.40954, 0.24817, 0.13893, 0.03473, 0.00660])),
        (100, np.array([0.93202, 0.92613, 0.91731, 0.90263, 0.88799, 0.87338, 0.78683, 0.64880, 0.52202, 0.40952, 0.23303, 0.11915])),
    ]

    @pytest.mark.parametrize("tau, expected", test_data)
    def test_usolution(self, tau, expected):
        result = np.array([usolution(pos, tau, 0.1) for pos in self.xpos])
        np.testing.assert_allclose(result, expected, atol=1.5e-4)


class TestSuOlsonDimensionlessVEps0p1():
    """Compare the dimensionless v-solution of the Python solver with eps = 0.1

    Results are compared against table 2 in [Su1996]_
    """
    xpos = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0])
    test_data = [
        (0.001, np.array([0.00006, 0.00002, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])),
        (0.003, np.array([0.00030, 0.00014, 0.00004, 0., 0., 0., 0., 0., 0., 0., 0., 0.])),
        (0.01, np.array([0.00170, 0.00110, 0.00055, 0.00012, 0.00003, 0., 0., 0., 0., 0., 0., 0.])),
        (0.03, np.array([0.00762, 0.00592, 0.00398, 0.00196, 0.00088, 0.00027, 0., 0., 0., 0., 0., 0.])),
        (0.1, np.array([0.03446, 0.02955, 0.02339, 0.01566, 0.01030, 0.00672, 0.00035, 0.00001, 0., 0., 0., 0.])),
        (0.3, np.array([0.11322, 0.10124, 0.08551, 0.06437, 0.04830, 0.03612, 0.00584, 0.00020, 0.00001, 0., 0., 0.])),
        (1, np.array([0.32030, 0.29429, 0.25915, 0.20925, 0.16862, 0.13563, 0.03539, 0.00334, 0.00028, 0.00003, 0., 0.])),
        (3, np.array([0.58906, 0.55843, 0.51488, 0.44845, 0.38930, 0.33690, 0.13377, 0.02432, 0.00381, 0.00054, 0.00002, 0.])),
        (10, np.array([0.78318, 0.76448, 0.73673, 0.69139, 0.64730, 0.60461, 0.38320, 0.15285, 0.05166, 0.01527, 0.00098, 0.00005])),
        (30, np.array([0.87523, 0.86443, 0.84829, 0.82154, 0.79500, 0.76871, 0.61768, 0.40167, 0.24046, 0.13273, 0.03213, 0.00589])),
        (100, np.array([0.93167, 0.92576, 0.91689, 0.90214, 0.88743, 0.87275, 0.78578, 0.64715, 0.51993, 0.40717, 0.23071, 0.11735])),
    ]

    @pytest.mark.parametrize("tau, expected", test_data)
    def test_vsolution(self, tau, expected):
        usolutions = np.array([usolution(pos, tau, 0.1) for pos in self.xpos])
        result = np.array([vsolution(pos, tau, 0.1, u) \
                          for pos, u in zip(self.xpos, usolutions)])
        np.testing.assert_allclose(result, expected, atol=1.5e-4)


class TestSuOlsonDimensionlessUEps1():
    """Compare the dimensionless u-solution of the Python solver with eps = 1.0

    Results are compared against table 3 in [Su1996]_
    """
    xpos = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0])
    test_data = [
        (0.001, np.array([0.03016, 0.00034, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])),
        (0.003, np.array([0.0513, 0.00605, 0.00003, 0., 0., 0., 0., 0., 0., 0., 0., 0.])),
        (0.01, np.array([0.0904, 0.03241, 0.00361, 0.00001, 0., 0., 0., 0., 0., 0., 0., 0.])),
        (0.03, np.array([0.14769, 0.08522, 0.03043, 0.00294, 0.00012, 0., 0., 0., 0., 0., 0., 0.])),
        (0.1, np.array([0.24023, 0.18003, 0.11024, 0.04111, 0.01217, 0.0028, 0., 0., 0., 0., 0., 0.])),
        (0.3, np.array([0.34619, 0.29261, 0.22334, 0.13531, 0.07653, 0.04016, 0.00014, 0., 0., 0., 0., 0.])),
        (1, np.array([0.46599, 0.42133, 0.3602, 0.27323, 0.20332, 0.14837, 0.01441, 0.00005, 0.00001, 0., 0., 0.])),
        (3, np.array([0.58965, 0.55471, 0.50462, 0.42762, 0.35891, 0.29847, 0.08222, 0.00505, 0.00015, 0.00001, 0., 0.])),
        (10, np.array([0.73611, 0.71338, 0.67978, 0.62523, 0.57274, 0.52255, 0.27705, 0.07075, 0.01271, 0.00167, 0.00002, 0.])),
        (30, np.array([0.83793, 0.82392, 0.80302, 0.76849, 0.73442, 0.70087, 0.51363, 0.27157, 0.12325, 0.04803, 0.00469, 0.00027])),
        (100, np.array([0.90895, 0.90107, 0.88926, 0.86965, 0.85011, 0.83067, 0.71657, 0.5408, 0.38982, 0.26789, 0.10906, 0.03624])),
    ]    

    @pytest.mark.parametrize("tau, expected", test_data)
    def test_usolution(self, tau, expected):
        result = np.array([usolution(pos, tau, 1.0) for pos in self.xpos])
        np.testing.assert_allclose(result, expected, atol=1.0e-4)


class TestSuOlsonDimensionlessVEps1():
    """Compare the dimensionless v-solution of the Python solver with eps = 1.0

    Results are compared against table 4 in [Su1996]_
    """
    xpos = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0])
    test_data = [
        (0.001, np.array([0.00002, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])),
        (0.003, np.array([0.0001, 0.00001, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])),
        (0.01, np.array([0.00062, 0.00014, 0.00001, 0., 0., 0., 0., 0., 0., 0., 0., 0.])),
        (0.03, np.array([0.00302, 0.00135, 0.00034, 0.00002, 0., 0., 0., 0., 0., 0., 0., 0.])),
        (0.1, np.array([0.01641, 0.01068, 0.00532, 0.00143, 0.00032, 0.00005, 0., 0., 0., 0., 0., 0.])),
        (0.3, np.array([0.06844, 0.05353, 0.03639, 0.01822, 0.00854, 0.00367, 0.00001, 0., 0., 0., 0., 0.])),
        (1, np.array([0.24762, 0.21614, 0.1753, 0.12182, 0.08306, 0.05556, 0.00324, 0.00001, 0., 0., 0., 0.])),
        (3, np.array([0.51337, 0.47651, 0.42483, 0.3481, 0.28252, 0.22719, 0.05123, 0.00226, 0.00005, 0., 0., 0.])),
        (10, np.array([0.72328, 0.69946, 0.66432, 0.60749, 0.55308, 0.50134, 0.25413, 0.05936, 0.00968, 0.00115, 0.00001, 0.])),
        (30, np.array([0.83529, 0.82105, 0.79981, 0.76474, 0.73015, 0.69611, 0.5066, 0.26374, 0.11733, 0.04465, 0.00413, 0.00023])),
        (100, np.array([0.90849, 0.90057, 0.88871, 0.869, 0.84937, 0.82983, 0.71521, 0.53877, 0.38745, 0.26551, 0.10732, 0.03534])),
    ]

    @pytest.mark.parametrize("tau, expected", test_data)
    def test_vsolution(self, tau, expected):
        usolutions = np.array([usolution(pos, tau, 1.0) for pos in self.xpos])
        result = np.array([vsolution(pos, tau, 1.0, u) \
                          for pos, u in zip(self.xpos, usolutions)])
        np.testing.assert_allclose(result, expected, atol=1.0e-4)
            