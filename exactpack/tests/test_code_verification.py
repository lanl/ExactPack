import unittest

import numpy

import exactpack.analysis.code_verification as verif


class TestStudy(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        dxs = [ 0.01, 0.02, 0.04, 0.08, 0.16 ]
        datasets = [ numpy.ones(10)+dx for dx in dxs ]
        self.study = verif.Study(datasets, dxs, abscissa="radius")

    def test_index(self):
        numpy.testing.assert_equal(self.study[1,2,4].study_parameters,
                                   [ 0.02, 0.04, 0.16 ])

    def test_slice(self):
        numpy.testing.assert_equal(self.study[1:4].study_parameters,
                                   [ 0.02, 0.04, 0.08 ])
