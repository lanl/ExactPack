import os.path

import numpy
import vtk
from vtk.numpy_interface.dataset_adapter import WrapDataObject


#: A dictionary containing mappings for many commonly used naming conventions
#: in code output to the ExactPack naming convention.  This can be passed to
#: the *name_mapping* argument of the :class:`VTKReader` constructor.
generic_VTK_name_mapping = {
    'pressure': 'pressure', 'pres': 'pressure', 'p': 'pressure',
    'temperature': 'temperature', 'temp': 'temperature', 'T':'temperature',
    'velocity': 'velocity', 'vel': 'velocity', 'U': 'velocity',
}


class VTKReader(object):
    r"""Read data from a file using VTK.

    When using the :mod:`exactpack.analysis` tools, it is necessary to
    read data output of many different codes, which may be in a
    variety of formats.  The :class:`VTKReader` class is a generic
    interface which uses the VTK library to read data and translate
    it so that ExactPack understands it.  The VTK library provides
    readers for a range of formats.  :class:`VTKReader` can, in
    principle, use any of these.

    The user can also specify a specific VTK object to use to read the
    data.  If no object is specified (the default), then choice is
    made automatically based on the file extension.

    Currently the following limitations apply to reading data with
    :class:`VTKReader`

        * Only point data, not cell data.
        * No support for time dependent data sets.
        * No support for multi-block grids.
    """
    #: A dictionary which maps file extenstions to VTK reader objects,
    #: for automatic file type detection.
    readers = { '.vtk': vtk.vtkGenericDataObjectReader,
                '.case': vtk.vtkGenericEnSightReader,
              }
    def __init__(self, name_mapping, geometry=1, reader=None):
        r"""Set up a :class:`VTKReader`.

        *name_mapping* is a dictionary of mappings from the
        VTK field names to ExactPack canonical names.  The keys are
        names of VTK fields, and the values are the canonical
        ExactPack names they should map to.

        The *geometry* is used by the analysis tools to determine the
        problem type, 1 for planar, 2 for cylindrical, and 3 for
        spherical.

        *reader* is an optional VTK data object which is a reader
        type.  This will override automatic type detection based on
        file extension.
        """
        self.geometry = geometry
        self.reader = reader
        self.name_mapping = name_mapping

    def __call__(self, filename):
        r"""Read data from a file."""
        if self.reader==None:
            extension = os.path.splitext(filename)[1]
            reader = self.readers[extension]()
        else:
            reader = self.reader()

        print("Reading {} using file format {}.".format(os.path.basename(filename),
                                                        repr(reader)))
            
        if isinstance(reader, vtk.vtkGenericEnSightReader):
            reader.SetCaseFileName(filename)
        else:
            reader.SetFileName(filename)
        reader.Update()
        data = WrapDataObject(reader.GetOutput())
    
        arrays = [ data.Points[:,0], data.Points[:,1], data.Points[:,2] ]
        names = [ 'x_position', 'y_position', 'z_position' ]

        print("The following mapping were used for variable names:")
        print("")
        print(" Name in VTK File   | Name in ExactPack  ")
        print("--------------------+--------------------")
        for key in data.PointData:
            if key in self.name_mapping:
                if data.PointData[key].ndim==1:
                    names.append(self.name_mapping[key])
                    print("{:20s}|{:20s}".format(key, self.name_mapping[key]))
                    arrays.append(data.PointData[key])
                elif data.PointData[key].ndim==2:
                    names.extend([self.name_mapping[key]+comp for comp in ["_x", "_y", "_z"]])
                    print("{:20s}|{:20s}".format(key, self.name_mapping[key]+"_{xyz}"))
                    arrays.extend([data.PointData[key][:,i] for i in range(3)])
            else:
                print("{:20s}|*NOT MAPPED*        ".format(key))

        if self.geometry == 2:
            r_position = sqrt(data.Points[:,0]**2 + data.Points[:,1]**2)
            names.append('r_position')
            arrays.append(r_position)
        elif self.geometry == 3:
            radius = sqrt(data.Points[:,0]**2 + data.Points[:,1]**2
                          + data.Points[:,0]**2)
            names.append('radius')
            arrays.append(radius)
                    
        dump = numpy.core.records.fromarrays(arrays, names=names)
        dump.dims = 3
    
        return dump
