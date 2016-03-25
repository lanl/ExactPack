r"""This module provides tools to perform code verification analysis.
"""


import glob
import math
from warnings import warn

import matplotlib.pyplot as plt
import numpy
import scipy.optimize


class CodeVerificationResult(object):
    """Holds results for verification analysis.

    Objects of class :class:`CodeVerificationResult` are created by
    invoking the :meth:`convergence` of an instance of
    :class:`CodeVerificationStudy`.  Class instances contain the
    verification data, and have methods for producing plots of
    convergence rates.
    """

    @staticmethod
    def fit(n, p, a):
        """The fitting function

        This is the function used to fit the error curve,

        .. math::

           \mathrm{Error} = a n^p
        """
        return a*n**p

    def __init__(self, study, var, norm):
        dxs, errs = study.errors(var, norm)
        #: An array containing the accuracy metric for each simulation
        #: (typically :math:`\Delta x` or number of cells for spatial
        #: errors, or :math:`\Delta t` for time step convergence).
        self.dxs = dxs
        #: An array containing the solution error for each simulation.
        self.errs = errs
        
        try:
            popt, pcov = scipy.optimize.curve_fit(self.fit, self.dxs, self.errs)
            #: The fit values of the parameters of the fitting function.
            self.popt = popt
            #: The covariance matrix for the parameter fit.
            self.pcov = pcov
            #: A label describing the fitting results, used for plot legends.
            self.label = var+": p={:4.3f}, std={:4.3f}".format(self.popt[0], math.sqrt(self.pcov[0,0]))
        except:
            self.label = var+" fit failed."

    def plot(self, fiducial=False):
        """Plot the convergence.

        Plot the error versus fitting parameter along with the curve
        fit, and an optional fiducial curve with slope *fiducial*
        (otherwise *fiducial* should be ``False``).
        """
        line, = plt.loglog(self.dxs, self.errs, 's', label=self.label)
        # Fit line, in matching color
        if hasattr(self, 'popt'):
            plt.plot(self.dxs, self.fit(self.dxs, *self.popt),
                     color=line.get_color())
        lsize = 18
        plt.xlabel('initial zone size', fontsize=lsize)
        plt.ylabel('L1 error norm', fontsize=lsize)
        plt.grid(b=True, which='both')
        plt.tick_params(axis='both', labelsize=lsize)
        plt.subplots_adjust(bottom=0.15)

        if fiducial:
            self.plot_fiducial(fiducial)

        plt.legend(loc='upper left')

    def plot_fiducial(self, slope):
        """Plot the fiducual curve.

        Plot a line with slope (in log-log coordinates) *slope*, for
        comparision to the fitted data.
        """
        n = len(self.errs)

        if n == 0:
            bottom = 1
            left = 0.01
            right = 0.1
        else:
            bottom = sorted(self.errs)[(n-1)/2]/2
            left = sorted(self.dxs)[(n-1)/2]
            right = 2*left

        plt.plot([left, right, right, left],
                 [bottom, bottom, bottom*(right/left)**slope, bottom],
                 'k-',
                 label='convergence fiducial: slope = {:4.3f}'.format(slope))


class PointNorm(object):
    r"""Pointwise error norm.
        
    The pointwise error norm is computed as

    .. math::

       L_n (d1 - d2) = \frac{1}{N} \left[ \sum_i |d1_i - d2_i|^n \right]^{1/n}

    where :math:`N` is the number of points, and :math:`n` is the
    order of the norm, *ord*.
    """

    def __init__(self, ord=1):
        """*ord* is the order of the norm."""
        self.ord = ord
    
    def __call__(self, d1, d2, var):
        return numpy.linalg.norm(d1[var]-d2[var], self.ord)/len(d1)
        

class CellNorm(object):
    r"""Volume weighted cell error norm.

    The volume weighted cell error norm is computed as

    ..  math::

        L_n (d1 - d2) = \frac{1}{N} \left[ \sum_i |(d1_i - d2_i) w_i|^n \right]^{1/n}
    
    where :math:`n` is the order of the norm, *ord*.  The weights
    :math:`w_i` are chosen based on the  dimensionality of the problem:
    For 1D Cartesian, they are the zone lengths, for 2D Cartesian, they
    are the zone areas, and for 2D axisymmetric and 3D they are the zone
    volumes.
    """

    def __init__(self, ord=1):
        """*ord* is the order of the norm."""
        self.ord = ord

    def __call__(self, d1, d2, var):
        return numpy.linalg.norm((d1[var]-d2[var])*d1.cell_volume, self.ord)/ \
               numpy.linalg.norm(d1.cell_volume, self.ord)


class CodeVerificationStudy(object):
    """An object which defines a code verification study.

    The :class:`CodeVerificationStudy` object holds the basic
    information used to define a code verification study.  This
    includes the output data read produced by the simulation code, as
    well as the exact solver.  It has a :meth:`convergence` method
    which is used to perform the actual convergence analysis, and
    which returns a :class:`CodeVerificationResult` instance.
    """

    def __init__(self, dumpfiles, exact_solver, transform=None,
                 domain=(-numpy.inf, numpy.inf), reader=None, dxs=None, time=0):
        """*dumpfiles* is a list of files to use for the verification
        study.  There are two modes: if the *reader* parameter is
        ``None``, then the elements of the list are actual data
        (:class:`numpy.core.recarray`), otherwise the list contains
        filenames from which the data should be read.

        *exact_solver* is an instance of a solver class which should
        be used to compute the exact solution for comparision.

        *transform* is a transformation which should be applied to the
        computational data before doing the comparision.  Typically
        this will be a rotation or translation, since most solutions
        are not invariant under other transformations.

        .. note::

           This option is currently not implemented, but in future
           releases will be done using a VTK object.

        *domain* is the domain in which the comparision should be performed.
        The domain is a 2-tuple consisting of the end-points of the comparision
        region.

        .. note::

           The transform is applied before the domain is selected, so
           the domain should be specified in the space of the
           exact solver, not the simulation code.

        .. note::

           In upcoming releases, *domain* will be a VTK object
           defining the region of comparision, an instance of a child
           class of :class:`vtk.vtkImplicitFunction`, and will be an
           argument to the :meth:`CodeVerificationStudy.convergence`
           method.

        *reader* is a callable data file reader which supports
        whatever data file format is used by the data files in the
        *dumpfiles* list, or, ``None`` if the *dumpfiles* list
        contains actual data.

        *dxs* are optional values to be used for the independent
        variable in the convergence study (number of points or
        :math:`\Delta x` for a grid convergence study, or
        :math:`\Delta t` for a timestep convergence study.  The *dxs*
        parameter is provided for cases in which ExactPack cannot
        determine the necessary information from the dump files.

        *time* is an optional value to be used when the solution time
        cannot be determined from the dump files.
        """
        self.exact_solver = exact_solver

        if transform:
            raise NotImplemented("Transforms are not yet supported.")

        self.domain = domain

        if reader:
            if isinstance(dumpfiles, str):
                dumpfiles = sorted(glob.glob(dumpfiles))
            self.dumpfiles = [reader(fn) for fn in dumpfiles]
        else:
            self.dumpfiles = dumpfiles
            
        if dxs:
            for dumpfile, dx in zip(self.dumpfiles, dxs):
                if dumpfile is not None:
                    dumpfile.dx = dx

        if time:
            for dumpfile in self.dumpfiles:
                if dumpfile is not None:
                    dumpfile.time = time

        self.dumpfiles = [dump for dump in self.dumpfiles
                          if dump is not None]

        if len(self.dumpfiles) <= 1:
            warn(RuntimeWarning("Only found {} runs for  {}".
                 format(len(self.dumpfiles), dumpfiles)))

        self.abscissa = ['x_position', 'r_position', 'radius'][exact_solver.
                                                               geometry-1]

    def errors(self, var, norm):
        """
        *norm* is a callable which will be used to perform the error
        norm.

        .. note::

           In the current release, the norm expects a 1-d Numpy array
           arguement.  In future releases, it will take a VTK data
           object.
        """
        dxs = [ dumpfile.dx for dumpfile in self.dumpfiles ]
        errs = []
        for dump in self.dumpfiles:
            masked_dump = dump[numpy.logical_and(dump[self.abscissa]>self.domain[0],
                                                 dump[self.abscissa]<self.domain[1])]
            errs.append(norm(masked_dump,
                             self.exact_solver(masked_dump[self.abscissa], dump.time),
                             var))
        return dxs, errs

    def convergence(self, var, norm):
        """Perform a convergence study
        """
        return CodeVerificationResult(self, var, norm)

    def plot(self, var, maxpts=20000):
        """Plot the simulation and exact solution profiles.

        Plot the exact solution and the simulation data for the
        variable *var*.  For very large data sets, *maxpts* is used to
        cut the data set down by using only every n-th point, where n
        is chosen such that only *maxpts* points are plotted.
        """
        if not self.dumpfiles:
            plt.text(0, 0.5, "No data found to plot.")
            return

        for dumpfile in self.dumpfiles:
            stride = max(len(dumpfile[self.abscissa])/maxpts, 1)
            plt.plot(dumpfile[self.abscissa][::stride],
                     getattr(dumpfile, var)[::stride],
                     '.', markersize=10, markeredgewidth=0,
                     label="h={:.3g}".format(dumpfile.dx))

        if self.domain[0] != -numpy.inf:
            plt.xlim(xmin=self.domain[0])

        if self.domain[1] != numpy.inf:
            plt.xlim(xmax=self.domain[1])

        rmin, rmax = plt.xlim()

        self.exact_solver(numpy.linspace(rmin, rmax, 1000),
                          self.dumpfiles[0].time).plot(var, color='k',
                                                       label='exact')
        plt.ylabel(var, fontsize=18)
