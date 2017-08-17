import unittest

import numpy

import exactpack.analysis.code_verification as verif
from exactpack.analysis import *
from exactpack.analysis.readers import VTKReader, generic_VTK_name_mapping
from exactpack.solvers.riemann import Sod
import os


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


# TODO: Need to test norm=CellNorm() in convergence rate fit calls


    def test_conv1(self):

        base = os.path.join(os.path.dirname(__file__),"../examples")

        study = Study([os.path.join(base, "data/shockTube_COARSE.vtk"),
               os.path.join(base, "data/shockTube_MED.vtk"),
               os.path.join(base, "data/shockTube_FINE.vtk"),],
              reference=Sod(pl=1e5, pr=1e4, interface_loc=0.0),
              study_parameters=[0.02, 0.01, 0.005],
              time=0.007,
              reader=VTKReader(name_mapping=generic_VTK_name_mapping)
              )

        fiducials = { 'pressure': 1 , 'density': 1}

        fitstudy = FitConvergenceRate(study, fiducials=fiducials)
        roachestudy = RoacheConvergenceRate(study, fiducials=fiducials)
        linregstudy = RegressionConvergenceRate(study, fiducials=fiducials)
