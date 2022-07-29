# ExactPack: An Open-Source Software Package for Code Verification

ExactPack (LA-CC-14-047) is an open-source software package that has been
developed for the verification & validation community. The package has two
major capabilities:

1. The generation of exact solutions for common benchmark problems in computational physics.
2. The analysis of computational physics code output to assess accuracy and convergence rates.

ExactPack is designed to be an open project, readily expandable to include new test problems for new physical models. At present, it contains the following modules and solutions (see reference list and documentation for details on these test problems):

* Hydrodynamics
  * Noh problem 
  * Uniform collapse problem (a.k.a. Noh2, Shockless Noh)
  * Sedov problem 
  * Several common shock tube problems (Sod, Einfeldt, Stationary Contact, Slow Shock, Shock Contact Shock, LeBlanc)
  * Shock tube problem with JWL equation of state
  * Guderley problem
* Reactive flows
  * Escape of high-explosive products
  * Programmed burn timing
  * Steady-detonation reaction-zone
  * Mader problem
  * Detonation Shock Dynamics problems (beta)
    * Rate stick
    * Cylindrical expansion
    * Explosive arc
* Inviscid hydrodynamics with heat transfer (radiation, conduction)
  * Coggeshall problems
  * Reinicke Meyer-ter-Vehn problem
  * Su-Olson problem
* Solid mechanics
  * Blake problem
* Heat Conduction
  * Planar sandwich
  * Cylindrical sandwich

The analysis capabilities of ExactPack include the import of data from computational physics codes, calculation of error norms between exact solutions and numerical results, and the calculation of spatial convergence rates. These tools follow the standards defined by the American Society of Mechanical Engineers (ASME).

Installation
------------

The repository may be cloned from the GitHub repository located at https://github.com/lanl/ExactPack, and installed using `python setup.py install`. The following dependencies are required:

### Dependencies

Python dependencies:
* Python 3.6
* setuptools
* numpy (version 1.12.0 is known to be broken)
* sphinx
* matplotlib
* scipy

Other dependencies:
* C compiler
* Fortran compiler (must support quad-precision arithmetic, i.e. REAL*16)

The built-in test suite may be run in a similar way: `python setup.py test`.

The online documentation may be compiled via Sphinx by moving to the doc directory within the Exactpack repository and using the command `make html`. This creates a build directory. The root file for this online documentation is located in `$EXACTPACK_ROOT/doc/build/html/index.html`, which may be opened in a web browser.

References
----------
#### The Noh Problem
W. F. Noh, *Errors for Calculations of Strong Shocks Using an Artificial Viscosity and an Artificial Heat Flux*, Journal of Computational Physics, **72** (1978), pp 78-120.

Francis X. Timmes, Galen Gisler, George Hrbek, *Automated analyses of the Tri-Lab verification test suite on uniform and adaptive grids for Code Project A*, Los Alamos report, LA-UR-05-6865 (2005).

M. Gehmeyr, B. Cheng, D. Mihalas, *Noh's Constant-Velocity Shock Problem Revisited*, Shock Waves, **7** 255 (1997).

#### Uniform collapse problem (a.k.a. Noh2, Shockless Noh)
W. F. Noh, *Errors for Calculations of Strong Shocks Using an Artificial Viscosity and an Artificial Heat Flux*, Journal of Computational Physics, **72** (1978), pp 78-120.

#### The Sedov Problem
L. I. Sedov, *Similarity and Dimensional Methods in Mechanics*, Academic Press, New York, NY, p. 146 ff. (1959).

#### The Riemann Problem
E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, Springer, 2009.

#### The Guderley Problem
S. D. Ramsey, J. R. Kamm, J. H. Bolstad, *The Guderley Problem Revisited*, International Journal of Computational Fluid Dynamics **26**, no. 2, p. 79-99 (2012).

#### Escape of HE Products
W. Fickett and C. Rivard, *Test Problems for Hydrocodes*, LASL Report, LA-5479 (1974, Rev 1981).

P. Dykema, S. Brandon, J. Bolstad, T. Woods, R. Klein, *Level 1 V&V, Test Problem 10: Escape of High Explosive Products, LLNL Report, UCRL-ID-150418 (2002).

S. Doebling, *The Escape of High Explosive Products: An Exact-Solution Problem for Verification of Hydrodynamics Codes*, LA-UR-15-22547

#### Programmed Burn Timing
M. A. Kenamond, *HE Burn Table Verification Problems*, Los Alamos report, LA-UR-11-03096 (2011).

#### Steady-Detonation Reaction-Zone
W. Fickett and W. C. Davis, *Detonation: Theory and Experiment*, University of California Press, 1979.

W. Fickett and C. Rivard, *Test Problems for Hydrocodes*, LASL Report, LA-5479 (1974, Rev 1981).

#### Coggeshall Problems
S. V. Coggeshall, *Analytic Solutions of Hydrodynamics Equations*, Physics of FLuids A **3**, no. 5, p. 757 (1991).

#### The Reinicke Meyer-ter-Vehn Problem
P. Reinicke and J. Meyer-ter-Vehn, *The Point Explosion with Heat Conduction*, Physics of Fluids A **3** no. 7 p. 1807 (1991).

J. R. Kamm, *Investigation of the Reinicke & Meyer-ter-Vehn Equations: I. The Strong Conduction Case*, Los Alamos report, LA-UR-00-4304 (2000).

#### Su-Olson Problem
B. Su, G. L. Olson, *Benchmark Results for the Non-equilibrium Marshak Diffusion Problem*, Journal of Quantitative Spectroscopic Radiative Transfer **56** p. 337 (1996).

#### Blake Problem
D. F. Aldridge, *Elastic Wave Radiation from a Pressurized Spherical Cavity*, Sandia National Laboratories Report, SAND2002-1882 (2002).

F. G. Blake, *Spherical Wave Propagation in Solid Media*, The Journal of the Acoustical Society of America **24**, no. 2, p. 211 (1952).

G. J. Hutchens, *An Analysis of the Blake Problem*, Los Alamos report, LA-UR-05-8737 (2005).

#### Detonation Shock Dynamics Problems
J.B. Bdzil, R. J. Henninger, and J. W. Walter, *Test Problems for DSD2D*, LA-14277 (2006).

#### Heat Conduction Problems
A Dawes, *3D Multi-Material Polyhedral Methods for Diffusion*, MultiMat Conference, Warzberg, Germany (2015).    

A. Dawes, C. Malone, M. Shashkov, *Some New Verification Test Problems for Multimaterial Diffusion on Meshes that are Non-Aligned with Material Boundaries.*, LA-UR-16-24696, Los Alamos National Laboratory (2016).

G J Hutchens, *A Generalized Set of Heat Conduction Test Problems*, LA-UR-09-01692, Los Alamos National Laboratory (2009).   
