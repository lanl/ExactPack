import os

from setuptools import find_packages, setup
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
                   'Topic :: Scientific/Engineering :: Physics'],
    keywords = 'verification',
    url = 'https://github.com/lanl/exactpack',
    author = 'Los Alamos National Laboratory',
    author_email = '',
    license = read('LICENSE.txt'),
    packages = find_packages(),
    cmdclass = { 'build_sphinx' : BuildDoc },
    test_suite = "exactpack.tests",
    )

