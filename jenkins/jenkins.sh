echo "============================================================"
echo "======== Begin running exactpack/jenkins/jenkins.sh ========"
echo "============================================================"
# Print specifics about server environment
echo "Host:"
echo $HOSTNAME
echo "User:"
echo $USER
echo "Working directory:"
echo `pwd`
echo "============================================================"
echo "=============== Building Conda Evironment =================="
echo "============================================================"
# Build base conda environment
conda create -n testEP python=2.7 numpy scipy sphinx matplotlib \
      vtk pytest pytest-cov wxpython

# Activate conda environment
source activate testEP

# Install gcc in anaconda. For some reason the gcc package in
# anaconda is 'touchy'. We must specify the version of some packages
# in order for gcc to install correctly.
# WARNING: This will most likely be machine dependent for any linux
# build.
conda install isl=0.12.2 mpfr mpc gmp cloog=0.18.0
conda install gcc

echo "============================================================"
echo "================== Building Exactpack ======================"
echo "============================================================"
# Exactpack build
python setup.py install

# Change out of source directory for testing
cd jenkins

echo "============================================================"
echo "=============== Running Exactpack Tests ===================="
echo "============================================================"
# pytest with junit testing and coverage reports in xml format
python -m pytest --cov=exactpack --cov-report=xml \
       --junitxml=./junittest_out.xml --pyargs exactpack

echo "============================================================"
echo "================== Testing Sphinx Build ===================="
echo "============================================================"
# First build the *.so extension libraries in the source tree so
# the documentation can import all modules from the source tree.
cd ..
python setup.py build_ext --inplace

cd doc
sphinx-build -W -b html -d build/doctrees source build/html

echo "============================================================"
echo "============== Cleaning Conda Environment =================="
echo "============================================================"
# Deactivate conda environment
source deactivate

# Remove conda environment
conda remove --name testEP --all


