# ExactPack: An Open-Source Software Package for Code Verification

ExactPack (LA-CC-14-047, C23033) is an open-source software package that has been developed for the verification & validation community. The package is capable of the generation of exact solutions for common benchmark problems in computational physics.

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
  * Radshock problem
* Solid mechanics
  * Blake problem
  * Elastic-Plastic Piston problem
* Heat Conduction
  * Planar sandwich
  * Cylindrical sandwich

Installation
------------

The repository may be cloned from the GitHub repository located at https://github.com/lanl/ExactPack, and installed using `pip install ./`. The minimum set of required dependencies should be installed automatically. If you wish to run the test suite or build the documentation then run `pip install -r requirements.txt` before installing Exactpack.

### Dependencies

Python dependencies:
* Python >=3.6
* setuptools
* numpy
* sphinx
* matplotlib
* scipy
* pytest

The built-in test suite may be run by using: `pytest`.

The online documentation may be compiled via Sphinx by moving to the doc directory within the Exactpack repository and using the command `make html`. This creates a build directory. The root file for this online documentation is located in `$EXACTPACK_ROOT/doc/build/html/index.html`, which may be opened in a web browser.

References
----------
#### The Noh Problem
W. F. Noh, *Errors for Calculations of Strong Shocks Using an Artificial Viscosity and an Artificial Heat Flux*, Journal of Computational Physics, **72** (1987), pp 78-120.

Francis X. Timmes, Galen Gisler, George Hrbek, *Automated analyses of the Tri-Lab verification test suite on uniform and adaptive grids for Code Project A*, Los Alamos report, LA-UR-05-6865 (2005).

M. Gehmeyr, B. Cheng, D. Mihalas, *Noh's Constant-Velocity Shock Problem Revisited*, Shock Waves, **7** 255 (1997).

#### Uniform collapse problem (a.k.a. Noh2, Shockless Noh)
W. F. Noh, *Errors for Calculations of Strong Shocks Using an Artificial Viscosity and an Artificial Heat Flux*, Journal of Computational Physics, **72** (1987), pp 78-120.

#### The Sedov Problem
L. I. Sedov, *Similarity and Dimensional Methods in Mechanics*, Academic Press, New York, NY, p. 146 ff. (1959).

#### The 1D Riemann Problem
J. J. Gottlieb and C. P. T. Groth, *Assessment of Riemann Solvers for Unsteady One-Dimensional Inviscid Flows of Perfect Gases*, Journal of Computational Physics, **78** (1988) pp 437-458.

R. Menikoff and B. J. Plohr, *The Riemann problem for fluid flow of real materials*, Reviews of Modern Physics, **61**, no. 1, p. 75 (1989).

R. J. LeVeque, *Finite Volume Methods for Hyperbolic Problems*, Cambridge University Press, 2002.

E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, Springer, 2009.

F. D. Lora-Clavijo, J. P. Cruz-Perez, F. Siddhartha Guzman, J. A. Gonzalez, *Exact solution of the 1D riemann problem in Newtonian and relativistic hydrodynamics*, Revista Mexicana de Fisica, **59** (2013), pp 28-50.

#### The Supersonic Steady-State 2-section 2D Riemann Problem
H.M. Glaz and A.B. Wardlaw, *A High-Order Godunov Scheme for Steady Supersonic Gas Dynamics*, Journal of Computational Physics **58**, 157-187 (1985).

C.Y. Loh and W.H. Hui, *New Lagrangian Method for Steady Supersonic Flow Computation I: Godunov Scheme*, Journal of Computational Physics **89**, 207-240 (1990).

C.Y. Loh and M.S. Liou, *New Lagrangian Method for Three-Dimensional Steady Supersonic Flows*, Journal of Computational Physics **113**, 224-248 (1994).

W.H. Hui, P.Y. Li and Z.W. Li, *Unified Coordinate System for Solving the Two-Dimensional Euler Equations*, Journal of Computational Physics **153**, 596-637 (1999).

W.H. Hui and K. Xu, *Computational Fluid Dynamics Based on the Unified Coordinates*, Springer, 2012.

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

#### Mader Problem
W. Fickett and W. C. Davis, *Detonation: Theory and Experiment*, University of California Press, Berkeley, 1979.

Francis X. Timmes, Galen Gisler, and George M. Hrbek, *Automated Analyses of the Tri-Lab Verification Test Suite on Uniform and Adaptive Grids for Code Project A*, LA-UR-05-6865, Los Alamos National Laboratory (2005).

R. Kirkpatrick, C. Wingate, and J.R. Kamm, *HE Burn Test Problem*, X-3-19U (2004).

#### Coggeshall Problems
S. V. Coggeshall, *Analytic Solutions of Hydrodynamics Equations*, Physics of FLuids A **3**, no. 5, p. 757 (1991).

#### The Reinicke Meyer-ter-Vehn Problem
P. Reinicke and J. Meyer-ter-Vehn, *The Point Explosion with Heat Conduction*, Physics of Fluids A **3** no. 7 p. 1807 (1991).

J. R. Kamm, *Investigation of the Reinicke & Meyer-ter-Vehn Equations: I. The Strong Conduction Case*, Los Alamos report, LA-UR-00-4304 (2000).

#### Su-Olson Problem
B. Su, G. L. Olson, *Benchmark Results for the Non-equilibrium Marshak Diffusion Problem*, Journal of Quantitative Spectroscopic Radiative Transfer **56** p. 337 (1996).

#### Radshock Problems
R. Lowrie and R. Rauenzahn, *Radiative shock solutions in the equilibrium diffusion limit* Shock Waves **18** (2007) 445.

R. Lowrie and J. Edwards, *Radiative shock solutions with grey nonequilibrium diffusion* Shock Waves **18** (2008) 129.

J. Ferguson, J. Morel and R. Lowrie *Nonrelativistic grey Sn-transport radiative-shock solutions*, High Energy Density Physics, 23 (2017) 95-114.

J. Ferguson, J. Morel and R. Lowrie *The equilibrium-diffusion limit for radiation hydrodynamics*, Journal of Quantitative Spectroscopy and Radiative Transfer, 202 (2017) 176-186

#### Blake Problem
D. F. Aldridge, *Elastic Wave Radiation from a Pressurized Spherical Cavity*, Sandia National Laboratories Report, SAND2002-1882 (2002).

F. G. Blake, *Spherical Wave Propagation in Solid Media*, The Journal of the Acoustical Society of America **24**, no. 2, p. 211 (1952).

G. J. Hutchens, *An Analysis of the Blake Problem*, Los Alamos report, LA-UR-05-8737 (2005).

#### Elastic-Plastic Piston Problem
Evan J. Lieberman, Xiaodong Liu, Nathaniel R. Morgan, Darby J. Luscher, and Donald E. Burton, *A higher-order Lagrangian discontinuous Galerkin hydrodynamic method for solid dynamics*, CMAME **353** (2019) 467-490.

#### Detonation Shock Dynamics Problems
J.B. Bdzil, R. J. Henninger, and J. W. Walter, *Test Problems for DSD2D*, LA-14277 (2006).

#### Heat Conduction Problems
A Dawes, *3D Multi-Material Polyhedral Methods for Diffusion*, MultiMat Conference, Warzberg, Germany (2015).    

A. Dawes, C. Malone, M. Shashkov, *Some New Verification Test Problems for Multimaterial Diffusion on Meshes that are Non-Aligned with Material Boundaries.*, LA-UR-16-24696, Los Alamos National Laboratory (2016).

G J Hutchens, *A Generalized Set of Heat Conduction Test Problems*, LA-UR-09-01692, Los Alamos National Laboratory (2009).   
