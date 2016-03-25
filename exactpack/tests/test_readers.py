"""Testing for ExactPack data readers.
"""

import os
import unittest

from exactpack.analysis.readers import TextReader, VTKReader


class TestTextReader(unittest.TestCase):
    """Tests for the text based reader."""
    
    def test_read(self):
        """Test that reader reads column data correctly."""
        reader = TextReader(
            names=['x_position', 'velocity_x', 'pressure', 'temperature']
        )
        data = reader(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                   "../examples/data/coarse.dat"))
        self.assertAlmostEqual(data['x_position'][5], -4.89)

    def test_missing_file(self):
        """Test that missing files return None."""
        reader = TextReader(
            names=['x_position', 'velocity_x', 'pressure', 'temperature']
        )
        data = reader("this_file_does_not_exist")
        self.assertEqual(data, None)


class TestVTKReader(unittest.TestCase):
    """Tests for the VTK based reader."""
    
    def test_read(self):
        """Test that VTK files read point data correctly."""
        reader = VTKReader(name_mapping={'U': 'velocity'})
        data = reader(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                   "../examples/data/shockTube_COARSE.vtk"))
        # Round to number of places which are used in the *.vtk (ASCII) file.
        self.assertAlmostEqual(data['velocity_x'][120], 12.5429, places=4)
