import inspect
import importlib
import os.path

from . import base


def discover_solvers():
    """Return a list of available solvers.

    Solver discovery is performed by walking the directory tree
    containing the :mod:`exactpack` module, and looking for classes
    which inherit from :class:`exactpack.base.ExactSolver`.  The return
    value is a list of names of solver classes.

    Since the directory walk is somewhat inefficient, it is suggested
    that the results be cached by the caller, if they are likely to be
    needed again.

    .. todo::

       Add a search directory list to :func:`exactpack.utils.discover_solvers`,
       so that users can develop other solvers outside the library.
    """

    solvers = []

    # Find the root path of the exactpack module
    modpath = os.path.dirname(os.path.abspath(__file__))

    # Walk the directory tree
    for path, dirs, files in os.walk(modpath):
        # Make sure this directory is a package
        if "__init__.py" in files:
            # Get package name
            pkgname = os.path.relpath(path, modpath[:-10]).split('/')
            if len(pkgname)==1:
                # Ignore the following in the package root
                for d in ['analysis', 'tests']:
                    dirs.remove(d)
                for f in ['cmdline.py', 'utils.py', 'gui.py', 'base.py']:
                    files.remove(f)
            for file in files:
                if file[-3:]==".py" and file!="__init__.py":
                    mod = importlib.import_module('.'.join(pkgname + [ file[:-3] ]))
                    for name, cls in inspect.getmembers(mod, inspect.isclass):
                        if issubclass(cls, base.ExactSolver):
                            solvers.append('.'.join(pkgname + [ file[:-3], name ]))
        else:
            # Prune sub-directories that are not in packages
            del dirs[:]

    return solvers
