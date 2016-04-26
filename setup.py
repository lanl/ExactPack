from setuptools import setup, find_packages
from numpy.distutils.core import setup, Extension
from sphinx.setup_command import BuildDoc



setup(
    name = "ExactPack",
    version = "1.3",
    packages = find_packages(),
#    install_requires = [ 'importlib', 'numpy', 'vtk', 'scipy', 'matplotlib', 'sphinx'],
    ext_modules = [ Extension(name = 'exactpack.solvers.noh._timmes',
                              sources = ['src/timmes/noh/noh.f'] ),
                    Extension(name = 'exactpack.solvers.sedov._timmes',
                              sources = ['src/timmes/sedov/sedov3.f'],
                              f2py_options = (['only:'] + [ 'sed_1d' ] + [':'] +
                                              ['free-form'] + [':'])), 
                    Extension(name = 'exactpack.solvers.riemann._timmes',
                              sources = ['src/timmes/riemann/exact_riemann.f'],
                              f2py_options = ['only:'] + [ 'riemann' ] + [':']),
                    Extension(name = 'exactpack.solvers.suolson._timmes',
                              sources = ['src/timmes/suolson/suo02.f'],
                              f2py_options = ['only:'] + [ 'suolson' ] + [':']),
                    Extension(name = 'exactpack.solvers.mader._timmes',
                              sources = ['src/timmes/mader/rarefaction.f'],
                              f2py_options = ['only:'] + [ 'mader' ] + [':']),
                    Extension(name = 'exactpack.solvers.rmtv._timmes',
                              sources = ['src/timmes/rmtv/rmtv.f'],
                              f2py_options = ['only:'] + [ 'rmtv' ] + [':']),
                    Extension(name = 'exactpack.solvers.cog._timmes',
                              sources = ['src/timmes/cog8/cog8.f'],
                              f2py_options = ['only:'] + [ 'cog8_timmes' ] + [':']),
                    Extension(name = 'exactpack.solvers.riemann._kamm',
                              sources = ['src/kamm/riemann/shktub.f90',
                                         'src/kamm/riemann/param.h'],
                              f2py_options = ['only:'] + [ 'riemann_kamm' ] + [':']),
                    Extension(name = 'exactpack.solvers.guderley._ramsey',
                              sources = ['src/ramsey/guderley/guderley_1D.f',
                                         'src/ramsey/guderley/d1mach.f',
                                         'src/ramsey/guderley/deroot.f',
                                         'src/ramsey/guderley/exp.f',
                                         'src/ramsey/guderley/interp_laz.f',
                                         'src/ramsey/guderley/ode.f',
                                         'src/ramsey/guderley/zeroin_a.f',
                                         'src/ramsey/guderley/zeroin.f'],
                              f2py_options = (['only:'] + [ 'guderley_1d' ] +
                                              [':'])),
                    Extension(name = 'exactpack.contrib.riemann_jwl._kamm',
                              sources = ['src/kamm/riemann_jwl/riemjwl.f90',
                                         'src/kamm/riemann_jwl/param.h'],
                              f2py_options = (['only:'] +
                                              [ 'riemann_kamm_jwl'] + [':'])),
                    Extension(name = 'exactpack.solvers.sedov._kamm',
                              sources = ['src/kamm/sedov/sedov.f90',
                                         'src/kamm/sedov/slatec.f90',
                                         'src/kamm/sedov/param.h'],
                              f2py_options = (['only:'] +
                                              [ 'sedov_kamm_1d' ] + [':'])),
                  ],
    entry_points = {
        'console_scripts' : [
            'exactpack = exactpack.cmdline:main',
        ],
        'gui_scripts' : [
            'epgui = exactpack.gui:main',
        ],
    },
    cmdclass = { 'build_sphinx' : BuildDoc },
    test_suite = "exactpack.tests"
    )

