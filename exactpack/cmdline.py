import argparse
import importlib
import sys

import numpy

import exactpack


def main():
    """A command line interface for ExactPack

    For usage information, run::

        exactpack --help
    """
    
    epilog = """
notes:

The solver name is the full name of a Python class with
"exactpack" omitted, e.g. "noh.Noh".

The PARAMS are keys and values separated by an equal sign (no
spaces), e.g. "gamma=1.4".

examples:
To plot the spherical Noh problem with gamma=5/3, you can use the
wrapper class:
  exactpack --plot noh.SphericalNoh
or the general class:
  exactpack noh.Noh --plot --params gamma=1.6667 geometry=3
Note that in this case, solver argument must go first so it is not
mistaken for an argument to --params.
"""

    parser = argparse.ArgumentParser(description="A command line interface to the Exactpack Python library",
                                     epilog=epilog,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-p', '--plot', action='store_true', help="Plot results on-screen using matplotlib")
    parser.add_argument('-d', '--dump', action='store', nargs=1, help="Dump results to a CSV file")
    parser.add_argument('--params', action='store', nargs='*', help="A series of parameter settings")
    parser.add_argument('-i', '--info', action='store_true', help="Print information about the case")
    parser.add_argument('--doc', action='store_true', help="Open ExactPack documentation in a viewer")
    parser.add_argument('--list-solvers', action='store_true', help="List all the available solvers")
    parser.add_argument('solver', nargs='?', help="Name of the solver to use")
    args =  parser.parse_args()

    if args.doc:
        # Delay the imports until here for efficiency
        import os
        import webbrowser

        # We are not sure where the documentation is, depends on if we are running
        # from a install directory or a source directory
        path = os.path.dirname(__file__)
        for fn in [ "../../../../../share/doc/html/index.html", "../doc/build/html/index.html", 
                    "../../../../../share/doc/ExactPack.pdf", "../doc/build/latex/ExactPack.pdf" ]:
            if os.access("{}/{}".format(path, fn), os.R_OK):
                webbrowser.open("file://{}/{}".format(path, fn), new=1, autoraise=True)
                sys.exit()

    if args.list_solvers:
        for solver in exactpack.discover_solvers():
            print solver[10:]
        sys.exit()

    # Get the solver name
    try:
        name = args.solver.split('.')
        mod = importlib.import_module('.'.join([''] + name[:-1]), 'exactpack')
        solver = getattr(mod, name[-1])
    except:
        print "Can't find requested module {}".format(args.solver)
        sys.exit(-1)

    # Run the requested command
    kwargs = {}
    if args.params:
        for p in args.params:
            key, val = p.split("=")
            kwargs[key] = float(val)
    soln = solver(**kwargs)(numpy.linspace(0, 1.0, 500), 0.6)

    if args.info:
        print solver.__doc__

    if args.plot:
        import matplotlib.pyplot as plt
        soln.plot()
        plt.show()
        
    if args.dump:

        soln.dump(args.dump[0])

if __name__=='__main__':
    main()            
