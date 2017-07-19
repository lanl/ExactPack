r"""This module provides tools to perform code verification analysis.
"""

import glob
import math
from warnings import warn

import matplotlib.pyplot as plt
import numpy
import numpy.lib.recfunctions
import scipy.optimize
import scipy.stats


class PointNorm(object):
    r"""Pointwise error norm.
        
    Returns a callable which computes the pointwise error norm according to
    the formula:

    .. math::

       L_n (d) = \left[ \frac{1}{N} \sum_i | d_i |^n \right]^{1/n}

    where :math:`N` is the number of points, and :math:`n` is the
    order of the norm, *ord*.
    """
    def __init__(self, ord=1):
        """*ord* is the order of the norm."""
        self.ord = ord
    
    def __call__(self, data):
        return numpy.linalg.norm(data, self.ord)/(len(data)**(1.0/self.ord))

        
class CellNorm(object):
    r"""Volume weighted cell error norm.

    Returns a callable which computes the volume weighted cell error norm
    according to the formula:

    ..  math::

        L_n (d) = \left[ \frac{\sum_i | w_i d_i |^n}{\sum_i w_i^n} \right]^{1/n}
    
    where :math:`n` is the order of the norm, *ord*.  The weights
    :math:`w_i` are chosen based on the  dimensionality of the problem:
    For 1D Cartesian, they are the zone lengths, for 2D Cartesian, they
    are the zone areas, and for 2D axisymmetric and 3D they are the zone
    volumes.
    """
    def __init__(self, ord=1):
        """*ord* is the order of the norm."""
        self.ord = ord

    def __call__(self, data):
        return numpy.linalg.norm(data[var]*data.cell_volume, self.ord)/ \
               numpy.linalg.norm(data.cell_volume, self.ord)


class Study(object):

    """Container object for building parameter studies.

    The :class:`Study` is a base class which loads data from multiple runs to
    be used in a parameter study, typically a solution convergence study.  Child
    classes can be defined to implement specific types of analysis.
    """
    def __init__(self, datasets, study_parameters,
                 reference=None, time=None, reader=None, abscissa=None):
        """Create a study object.

        If the *reader* option is not ``None``, then *datasets* is a list of
        filenames to read data from.  Otherwise, it is a list of actual data
        objects.  The *study_parameters* are the numerical values which are
        used as the independent variable in the :class:`Study`, typically the
        cell size or grid spacing.

        The reference solution to compare to is supplied in the form of an
        :class:`ExactSolver` instance passed to *reference*.  The exact solution
        will be compared at *time*.
        
        .. note::
            If a file :func:`~glob.glob` is used to create the list of filenames,
            it is advisable that it be :func:`sorted`, so that the order of
            the cases is predictable for consistency with the *study_parameters*
            list.
        """
        if reader:
            #: A sequence of :class:`numpy.recarray`\s, one for each data set.
            self.datasets = [ reader(dataset)
                              for dataset in datasets ]
        else:
            self.datasets = datasets
        self.study_parameters = numpy.array(study_parameters)
        self.reference = reference
        self.time = time
        #: The name of the variable which is the independent space variable for
        #: comparisons.
        self.abscissa = abscissa \
          or ['x_position', 'r_position', 'radius'][self.reference.geometry-1]

    def __getitem__(self, index):
        try:
            datasets = [ self.datasets[i] for i in index ]
            study_parameters = [ self.study_parameters[i] for i in index ]
        except TypeError:
            datasets = self.datasets[index]
            study_parameters = self.study_parameters[index]
        return Study(datasets=datasets,
                     study_parameters=study_parameters,
                     reference=self.reference,
                     time=self.time,
                     abscissa=self.abscissa)
          
    def plot(self, var, maxpts=10000):
        """Plot the study.

        Plot the spatial profiles of the variable *var* from each data set in
        the study and the exact reference solution (if one is defined).  If the
        data contains more than *maxpts* points to plot, sample the data at a
        fixed interval to plot no more than *maxpts* points.
        """
        for dataset, param in zip(self.datasets,
                                  self.study_parameters):
            stride = max(len(dataset[self.abscissa])/maxpts, 1)
            plt.plot(dataset[self.abscissa][::stride],
                     dataset[var][::stride],
                     '.', markersize=10, markeredgewidth=0,
                     label="h={:.3g}".format(param))

        rmin, rmax = plt.xlim()
        self.reference(numpy.linspace(rmin, rmax, 1000),
                       self.time).plot(var, color='k', label="Exact Solution")
                                       
        plt.xlabel(self.abscissa)
        plt.ylabel(var)
        plt.legend()
        

class FilterStudy(Study):
    """A study built from another study using a filter.

    A filter study is a class derived from :class:`FilterStudy`, which takes
    one study and transforms it by applying a filter function to every data
    set in that study, to create a new study.  To create a filter study, 
    necessary to subclass :class:`FilterStudy` and override the :meth:`filter`
    method with a function which takes a single data set as an argument and
    returns a new data set.
    """
    def __init__(self, source):
        """Create a filtered study.

        Filter the data in the *source* study using the :meth:`filter` method
        and construct a new study that contains the filtered data fields.
        """
        super(FilterStudy, self).__init__(
            datasets = source.datasets,
            study_parameters = source.study_parameters,
            reference = source.reference,
            time = source.time
            )
        self.datasets = [ self.filter(dataset)
                          for dataset in self.datasets ]

    def filter(self, dataset):
        """Data set filter.

        The filter function is applied to each data set when constructing the
        new filtered study.  The default :meth:`filter` method raises an
        exception, and must be over-ridden in child classes.
        """
        raise NotImplemented
    

class ExtractRegion(FilterStudy):
    """A study that extracts data in a particular region.

    This filter creates a new study which only includes data within a specific
    region of space.
    """
    def __init__(self, source, domain=(-numpy.inf, numpy.inf)):
        self.domain = domain
        super(ExtractRegion, self).__init__(source)
        
    def filter(self, dataset):
        if self.domain==(-numpy.inf, numpy.inf):
            return dataset
        else:
            mask = numpy.logical_and(dataset[self.abscissa]>self.domain[0],
                                     dataset[self.abscissa]<self.domain[1])
            return dataset[mask]
    
        
class ComputeExact(FilterStudy):
    """A study that adds exact solution data.

    This filter creates a new study which contains additional fields in each
    data set corresponding to the exact solution on the grid from that
    data set.  Exact solution fields are added for each variable which
    is defined in both the source data set and the exact solution.  The
    variable names in the new data set are the same as in the source,
    except that the string "Exact:" is pre-pended to each variable name.
    """
    def filter(self, dataset):
        """Compute the exact solution corresponding to a data set.

        Compute the exact solution on the same grid as the data set.  Then,
        return a new data set containing the exact solution on that grid for
        each variable which is defined in both the source data set and the
        exact solution.  The variable names in the new data set are the same
        as in the source, except that the string "Exact:" is prepended to
        each variable name.
        """
        soln = self.reference(dataset[self.abscissa], self.time)
        var_list = [ varname for varname in dataset.dtype.names
                     if varname in soln.dtype.names ]
        field_list = [ soln[varname] for varname in var_list ]
        field_name_list = [ "Exact:{}".format(varname)
                            for varname in var_list ]
        return numpy.lib.recfunctions.rec_append_fields(dataset,
                                                        field_name_list,
                                                        field_list)


class ComputeErrors(FilterStudy):
    """A study object that adds error fields.

    This filter creates a new study which contains additional fields in each
    data set corresponding to the error field.  :class:`ComputeErrors` assumes
    that the source study contains both code output data and exact solution data
    for one or more variables, the latter being indicated by a variable name
    pre-pended with the string "Exact:".  Typically this is obtained as the output
    from a :class:`ComputeExact` study.  The error is just the difference between
    the two fields, and is indicated in the output by the string "Error:" being
    prepended to the variable name.
    """
    def filter(self, dataset):
        var_list = [ varname for varname in dataset.dtype.names
                    if varname.startswith("Exact:") ]
        field_list = [ dataset[varname[6:]]-dataset[varname]
                    for varname in var_list ]
        field_name_list = [ "Error:{}".format(varname[6:])
                            for varname in var_list ]
        return numpy.lib.recfunctions.rec_append_fields(dataset,
                                                        field_name_list,
                                                        field_list)
    

class ComputeNorms(FilterStudy):
    """A study that computes error norms.

    This filter creates a new study whose datasets have an :attr:`errors`
    attribute, which is a dictionary with keys that are variable names, and
    values that are the norms of the corresponding error variable.
    """
    def __init__(self, study, norm=PointNorm()):
        """Create a :class:`ComputeNorms` object.

        Create a new instance of *study* with the error norms added to each
        data set.  The specific norm to use can be chosen with the *norm*
        parameter.

        .. note::
           It is extremely important to use the correct type of norm:
           a :class:`PointNorm` for intrinsic quantities, and a
           :class:`CellNorm` for extrinsic ones.
        """
        self.norm = norm
        super(ComputeNorms, self).__init__(study)
    
    def filter(self, dataset):
        dataset.errors = { varname[6:]: self.norm(dataset[varname])
                           for varname in dataset.dtype.names
                           if varname.startswith("Error:") }
        return dataset

    def plot(self, var, label=None, **kwargs):
        """Plot the norms.

        Plot the error norms of *var* versus the :attr:`study_parameters`.
        """
        plotret = plt.loglog(
            self.study_parameters,
            [ dataset.errors[var] for dataset in self.datasets ],
            's',
            label=label or var,
            **kwargs
            )
        lsize = 18
        plt.xlabel('Study Parameter', fontsize=lsize)
        plt.ylabel('Error Norm', fontsize=lsize)
        plt.grid(b=True, which='both')
        plt.tick_params(axis='both', labelsize=lsize)
        plt.subplots_adjust(bottom=0.15)
        plt.legend(loc='upper left')
        return plotret


class GlobalConvergenceRate(object):
    """Base class for convergence analaysis studies.

    To perform convergence analysis in ExactPack, it is necessary first to pipe
    the data through a series of filters, and then to compute the convergence
    rate.  This class is intended as a base class for convergence analysis; it
    performs all the requisite filtering, but does not compute the convergence
    rate.  That computation should be done in a child class.
    """
    def __init__(self, study, norm=PointNorm(), domain=(-numpy.inf, numpy.inf),
                 fiducials={}):
        self.fiducials = fiducials
        self.study = ExtractRegion(study, domain)
        self.exact = ComputeExact(self.study)
        self.errors = ComputeErrors(self.exact)
        self.norms = ComputeNorms(self.errors, norm)

    def __getitem__(self, index):
        return self.__class__(
            self.study[index],
            norm = self.norm,
            domain = self.domain,
            fiducials = self.fiducials,
            )
    
    def plot_fiducial(self, var):
        """Plot the fiducual curve.

        Plot a fiducial line with slope (in log-log coordinates) specified in
        the :attr:`fiducials` dictionary, for comparision to the fitted data.
        """
        n = len(self.norms.study_parameters)

        if n == 0:
            bottom = 1
            left = 0.01
            right = 0.1
        else:
            bottom = sorted([ dataset.errors[var]
                              for dataset in self.norms.datasets ])[(n-1)/2]/2
            left = sorted(self.norms.study_parameters)[(n-1)/2]
            right = 2*left

        plt.plot([left, right, right, left],
                 [bottom, bottom, bottom*(right/left)**self.fiducials[var], bottom],
                 'k-',
                 label='convergence fiducial: slope = {:4.3f}'.format(self.fiducials[var]))
                
    def plot(self, var, label=None, fiducial=True):
        if fiducial and var in self.fiducials:
            self.plot_fiducial(var)
        return self.norms.plot(var, label=label)

        
class FitConvergenceRate(GlobalConvergenceRate):
    """A class which performs a convergence analysis using a curve fit.

    When initialized with a :class:`Study` instance, the
    :class:`FitConvergenceRate` performs a convergence analysis using
    a optimization procedure to fit the study data to a specified fitting
    function.
    """
    def __init__(self, study, fitting_function=lambda n, p, a: a*n**p,
                 *args, **kwargs):
        """Create a :class:`FitConvergenceRate` object.

        Use the data sets in *study* to compute the convergence rate,
        by fitting the data to the *fitting_function*.  Any other
        arguments are passed to :class:`GlobalConvergenceRate`.
        """
        super(FitConvergenceRate, self).__init__(study, *args, **kwargs)
        #: The function to fit the convergence data to.
        self.fitting_function = fitting_function
        #: A dictionary whose keys are the variables for which
        #: convergence analysis is avaiable, and values are the fit
        #: parameters.  Each fit value is a tuple (*popt*, *pcov*) returned by
        #: :func:`scipy.optimize.curve_fit`, where *popt* is the optimal fit
        #: values for :attr:`fitting_function`, and *pcov* is the estimated
        #: covarience of *popt*.  For more information, see the documentation
        #: for :func:`scipy.optimize.curve_fit`.
        self.fits = {}
        for varname in self.norms.datasets[0].errors:
            self.fits[varname] = scipy.optimize.curve_fit(
                self.fitting_function,
                self.norms.study_parameters,
                [ dataset.errors[varname] for dataset in self.norms.datasets ]
                )

    def plot(self, var, label="{var}: p={popt[0]:4.3f}, std={pcov[0][0]:4.3f}", **kwargs):
        """Create a standard convergence rate plot.

        This method uses :func:`matplotlib.pyplot.plot` to plot the error norms and the best
        fit converegence rate.  It is necessary to do an explicit
        :func:`~matplotlib.pyplot.show` to display the plot.
        """
        line, = super(FitConvergenceRate, self).plot(var)
        # Remove the label, since the fit line will be labeled
        line.set_label("")
        # Draw the fit line, in matching color
        plt.plot(self.norms.study_parameters,
                 self.fitting_function(self.norms.study_parameters, *self.fits[var][0]),
                 color=line.get_color(),
                 label=label.format(var=var, popt=self.fits[var][0], pcov=self.fits[var][1]),
                 **kwargs
                 )
        plt.legend()

    def p(self, var):
        """The convergence rate for *var*."""
        return self.fits[var][0][0]

    def goodness(self, var):
        """The goodness of fit for *var*."""
        return self.fits[var][1][0,0]
        
    def __str__(self):
        """Return a table with the fits as a string.

        The special method :meth:`__str__` defines a standard string
        representation of the convergence study, which will be used when
        a :class:`FitConvergenceRate` instance is printed using the
        :func:`print` function.  The standard format is a table with the
        fits, presented using RST markup, to be both human readable and
        simple to incorporate in documents.
        """
        separator = 16*'='+" "+4*"="
        lines = [ separator, "{:16s} {:4s}".format("Variable", "p"), separator ]
        lines.extend(
            [ "{:16s} {:4.2g}".format(varname, self.fits[varname][0][0])
            for varname in self.fits
            ] )
        lines.append(separator)
        return "\n".join(lines)


class RoacheConvergenceRate(GlobalConvergenceRate):
    """A class which performs a convergence analysis using the generalized
    Richardson extrapolation.

    When initialized with a :class:`Study` instance, the
    :class:`RoacheConvergenceRate` performs a convergence analysis using
    the generalized Richardson extrapolation, also known as Roache's
    formula.

    .. note::
       Convergence values are evaluated using pairs of data sets in
       the order in which they are passed to the study routine.  If
       the grids are not in order, unexpected results may be
       obtained.
    """
    def __init__(self,  *args, **kwargs):
        """Create a :class:`RoacheConvergenceRate` object."""
        super(RoacheConvergenceRate, self).__init__(*args, **kwargs)
        #: A dictionary whose keys are the variables for which
        #: convergence analysis is avaiable, and values are the fit
        #: parameters.  Since the Roache formula is applied to
        #: each pair of grids in sequence, this is a list of
        #: tuples, each containing a convergence rate and a
        #: slope.
        self.fits = {}
        for varname in self.norms.datasets[0].errors:
            self.fits[varname] = [ self.roache(self.norms.datasets[i].errors[varname],
                                               self.norms.datasets[i+1].errors[varname],
                                               self.norms.study_parameters[i],
                                               self.norms.study_parameters[i+1])
                                   for i in range(len(self.norms.datasets)-1) ]

    @staticmethod
    def roache(e1, e2, h1, h2):
        """Return rate of convergence and pre-factor.
        """
        ratio = h1/h2

        p = math.log(e1/e2)/math.log(ratio)
        A = e1/h1**p
        return p, A

    def p(self, var):
        """An array of convergence rates for *var*.

        As the Roache formula operates on solutions on exactly two
        grids, this function returns an array of p values.
        """
        return numpy.array([ fit[0] for fit in self.fits[var] ])
    
    def plot(self, var):
        p = self.p(var)
        line, = super(RoacheConvergenceRate, self).plot(
            var, label="{}: p={:4.2g}-{:4.2g}".format(var, min(p), max(p))
            )
        
    def __str__(self):
        """Return a table with the fits as a string.

        The format is the same as for :meth:`FitConvergenceRate.__str__`
        """
        separator = 16*'='+" "+6*"="+" "+6*"="+" "+4*"="
        lines = [separator,
                 "{:16s} {:6s} {:6s} {:4s}".format(
                     "Variable", "Grid 1", "Grid 2", "p"
                     ),
                 separator
                ]
        lines.extend(
            ["{:16s} {:6.2g} {:6.2g} {:4.2g}".format(
                    varname if i==0 else "",
                    self.norms.study_parameters[i],
                    self.norms.study_parameters[i+1],
                    self.fits[varname][i][0]
                    )
             for i in range(len(self.norms.datasets)-1)
             for varname in self.norms.datasets[0].errors
            ])
        lines.append(separator)
        return "\n".join(lines)


class RegressionConvergenceRate(GlobalConvergenceRate):
    """A class which performs a convergence analysis using linear regression.

    When initialized with a :class:`Study` instance, the
    :class:`RegressionConvergenceRate` performs a convergence analysis
    based on the error ansatz

    .. math::
       \epsilon = A h^p

    by applying linear regression to the logarithm of the errors and
    deltas.
    """
    def __init__(self, *args, **kwargs):
        """Create a :class:`RegressionConvergenceRate` object."""
        super(RegressionConvergenceRate, self).__init__(*args, **kwargs)
        #: A dictionary whose keys are the variables for which
        #: convergence analysis is avaiable, and values are the fit
        #: parameters.  Each fit value is a tuple (*slope*,
        #: *intercept*, *rvalue*, *pvalue*, *stderr*) returned by
        #: :func:`scipy.stats.linregress`, where *slope* is the
        #: convergence rate.  For more information, see the documentation
        #: for :func:`scipy.stats.linregress`.
        self.fits = {}
        for varname in self.norms.datasets[0].errors:
            self.fits[varname] = scipy.stats.linregress(
                numpy.log(numpy.array(self.norms.study_parameters)),
                numpy.log(numpy.array([dataset.errors[varname] for dataset in self.norms.datasets]))
                )

    def plot(self, var, label="{var}: p={slope:4.3f}, cor={rvalue:4.3f}", **kwargs):
        line, = super(RegressionConvergenceRate, self).plot(var)
        # Remove the label, since the fit line will be labeled
        line.set_label("")
        # Draw the fit line, in matching color
        plt.plot(self.norms.study_parameters,
                 math.exp(self.fits[var][1])
                 *self.norms.study_parameters**self.fits[var][0],
                 color=line.get_color(),
                 label=label.format(var=var,
                                    slope=self.fits[var][0],
                                    intercept=self.fits[var][1],
                                    rvalue=self.fits[var][2],
                                    pvalue=self.fits[var][3],
                                    stderr=self.fits[var][4]),
                 **kwargs)
        plt.legend()

    def p(self, var):
        """The convergence rate for *var*."""
        return self.fits[var][0]

    def goodness(self, var):
        """The goodness of fit for *var*."""
        return self.fits[var][2]
        
    def __str__(self):
        """Return a table with the fits as a string.

        The format is the same as for :meth:`FitConvergenceRate.__str__`
        """
        separator = 16*'='+" "+4*"="
        lines = [separator, "{:16s} {:4s}".format("Variable", "p"), separator]
        lines.extend(
            ["{:16s} {:4.2g}".format(varname, self.fits[varname][0])
             for varname in self.fits
            ])
        lines.append(separator)
        return "\n".join(lines)
