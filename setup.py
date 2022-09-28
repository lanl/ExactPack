import os

from setuptools import find_packages
from numpy.distutils.core import setup, Extension
from sphinx.setup_command import BuildDoc
from exactpack import __version__

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "ExactPack",
    version = __version__,
    description = "Exact solution API for physics code verification",
    long_description = read('README.md'),
    classifiers = ['Development Status :: 4 - Beta',
                   'Intended Audience :: Science/Research',
                   'Programming Language :: Python :: 3 :: Only',
                   'Programming Language :: Fortran',
                   'Topic :: Scientific/Engineering :: Physics'],
    keywords = 'verification',
    url = 'https://github.com/lanl/exactpack',
    author = 'Los Alamos National Laboratory',
    author_email = '',
    license = read('LICENSE.txt'),
    packages = find_packages(),
    ext_modules = [ Extension(name = 'exactpack.solvers.noh._timmes',
                              sources = ['src/timmes/noh/noh.f90'] ),
                    Extension(name = 'exactpack.solvers.sedov._timmes',
                              sources = ['src/timmes/sedov/sedov3.f90'],
                              f2py_options = (['only:'] + [ 'sed_1d' ] + [':'])
                              ),
                    Extension(name = 'exactpack.solvers.riemann._timmes',
                              sources = ['src/timmes/riemann/exact_riemann.f90'],
                              f2py_options = ['only:'] + [ 'riemann' ] + [':']
                              ),
                    Extension(name = 'exactpack.solvers.cog._timmes',
                              sources = ['src/timmes/cog8/cog8.f90'],
                              f2py_options = ['only:'] + [ 'cog8_timmes' ] + [':']
                              ),
                    Extension(name = 'exactpack.solvers.riemann._kamm',
                              sources = ['src/kamm/riemann/shktub.f90',
                                         'src/kamm/riemann/param.h'],
                              f2py_options = ['only:'] + [ 'riemann_kamm' ] + [':']
                              ),
                    Extension(name = 'exactpack.contrib.riemann_jwl._kamm',
                              sources = ['src/kamm/riemann_jwl/riemjwl.f90',
                                         'src/kamm/riemann_jwl/param.h'],
                              f2py_options = (['only:'] +
                                              [ 'riemann_kamm_jwl'] + [':'])
                              ),
                    Extension(name = 'exactpack.solvers.sedov._kamm',
                              sources = ['src/kamm/sedov/sedov.f90',
                                         'src/kamm/sedov/slatec.f90',
                                         'src/kamm/sedov/param.h'],
                              f2py_options = (['only:'] +
                                              [ 'sedov_kamm_1d' ] + [':'])
                              ),
                    Extension(name = 'exactpack.solvers.heat._dawes',
                              sources = ['src/dawes/planar_sandwich.f90'],
                              f2py_options = (['only:'] +
                                              [ 'planar_sand' ] + [':'] + ['free-form'] + [':'])
                              ),
                            ],
    entry_points = {
        'console_scripts' : [
            'exactpack = exactpack.cmdline:main',
        ],
    },
    cmdclass = { 'build_sphinx' : BuildDoc },
    test_suite = "exactpack.tests",
    )

