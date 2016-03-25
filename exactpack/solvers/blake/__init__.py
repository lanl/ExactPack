r"""The spherically symmetric, isotropic, linear elastic Blake problem.

The Blake problem concerns the spherically symmetric propagation of radial,
longitudinal waves from a cavity of radius, :math:`a`, in a homogeneous,
isotropic, linear elastic whole space, whose surface is loaded by a
time-dependent normal traction or pressure.  When the pressure history,
:math:`p(t)`, is a step or rapid increase in time followed by a suitable decay,
the solution has application in modeling the behavior of an embedded explosive
energy source.  Indeed, this was the motivation for early analyses of the
problem: see Jeffries [#jef31]_, Sharpe [#sha42]_, Blake [#bla52]_, Denny and
Johnson [#den91]_ and references therein.

===================
Governing equations
===================

Following Aldridge [#ald02]_ (among others) we use spherical coordinates with
origin at the cavity center for our analysis. Symmetry considerations require
that the particle displacement field be purely radial so that,

.. math::
   \mathbf{u}(\mathbf{r},t) = u(r,t)\,{\mathbf{\hat{e}}_r} \,,
   :label: sphDispl

where :math:`\mathbf r` is the position vector and :math:`\mathbf{\hat{e}}_r`
is the radial unit vector.  Thus, :math:`u(r,t)` is the *physical* radial
component of displacement. The elastodynamic equation :eq:`elastoDyn` for
:math:`\mathbf u` is obtained by substituting the isotropic, linear elastic
constitutive model :eq:`strsStrn` into the momentum equation :eq:`momenEq` (in
its body force- and body moment-free form),

.. math::
   \boldsymbol{T} = \lambda\,\mathrm{tr}\,(\boldsymbol{E})\,\boldsymbol{1}
   + 2\mu\,\boldsymbol{E} \,,
   :label: strsStrn

.. math::
   \boldsymbol{\nabla}  \bullet \boldsymbol{T} =
   \frac{{{\partial ^2}\boldsymbol{u}}}{{\partial {t^2}}} \,,
   :label: momenEq

.. math::
   (\lambda  + \mu)\,\boldsymbol{\nabla} (\boldsymbol{\nabla}  \bullet
   \boldsymbol{u}) + \mu \,{\boldsymbol{\nabla} ^2}\boldsymbol{u} =
   \rho_0 \frac{{{\partial ^2}\boldsymbol{u}}}{{\partial {t^2}}} \,\ldotp
   :label: elastoDyn

Here :math:`\boldsymbol T` is the (Cauchy) stress tensor, :math:`\boldsymbol 1`
the unit tensor, :math:`\mathrm{tr}\,(\cdot)` the trace of a 2-tensor,
:math:`\boldsymbol{\nabla}(\cdot)` the gradient operator, and
:math:`\boldsymbol{\nabla} \bullet (\cdot)` the divergence.  The infinitesimal
strain tensor, :math:`\boldsymbol E`, is the symmetric part of
:math:`\boldsymbol{\nabla}\boldsymbol{u}`, :math:`\lambda \textrm{ and } \mu`
are the (constant) Lame moduli and :math:`\rho_0` is the initial mass density.
With use of the identity,

.. math::
   {\boldsymbol{\nabla} ^2}\boldsymbol{u} = \boldsymbol{\nabla}
   (\boldsymbol{\nabla}  \bullet \boldsymbol{u}) - \boldsymbol{\nabla} \times
   (\boldsymbol{\nabla}  \times \boldsymbol{u}) \,,
   :label: GDDGcurlIden

:eq:`elastoDyn` may be written as,

.. math::
   (\lambda  + 2\mu )\,{\boldsymbol{\nabla} ^2}\boldsymbol{u} +
   (\lambda  + \mu )\,\boldsymbol{\nabla}  \times \,(\boldsymbol{\nabla}
   \times \boldsymbol{u}) =
   \rho_0 \frac{{{\partial ^2}\boldsymbol{u}}}{{\partial {t^2}}} \,\ldotp
   :label: elastoDynCurl

The motion described by :eq:`sphDispl` is curl-free so that :eq:`elastoDynCurl`
reduces to the vector wave equation,

.. math::
   {\boldsymbol{\nabla} ^2}\boldsymbol{u} =
   \frac{1}{{{c_L}^2}}\frac{{{\partial ^2}\boldsymbol{u}}}{{\partial {t^2}}}
   :label: vecWaveEq

where :math:`{c_L}^2` is the (squared) longitudinal or compressional wave speed
of the material, see equation :eq:`alphaBetaClCt`.  Introducing :eq:`sphDispl`
into :eq:`vecWaveEq` and expanding in spherical coordinates, we find that the
radial component :math:`u(r,t)` satisfies the *scalar* wave equation in
spherical coordinates,

.. math::
   \frac{{{\partial ^2}u}}{{\partial {r^2}}} +
   \frac{2}{r}\frac{{\partial u}}{{\partial r}} - \frac{{2u}}{{{r^2}}} =
   \frac{1}{{{c_L}^2}}\frac{{{\partial ^2}u}}{{\partial {t^2}}} \ldotp
   :label: sphWaveEq

As noted by (Hutchens 2005), further simplification obtains by introduction
of a displacement potential, :math:`\phi`, so that :eq:`sphWaveEq` may be
written as,

.. math::
   \frac{\partial }{{\partial r}}\left[ {\frac{1}{{{r^2}}}
   \frac{\partial }{{\partial r}}
   \left( {{r^2}\frac{{\partial \phi }}{{\partial r}}} \right) -
   \frac{1}{{{c_L}^2}}\frac{{{\partial ^2}\phi }}{{\partial {t^2}}}}
   \right] = 0\,; \quad u(r,t) = \frac{{\partial \phi }}{{\partial r}}
   \ldotp
   :label: sphPotEq

The outermost derivative in :eq:`sphPotEq` is eliminated by equating the
expression inside the square brackets to an arbitrary function of time,
:math:`C(t)`.  Because we need only :math:`u = {\partial \phi}/{\partial r}`
and its derivatives (but not :math:`\phi` itself) we may, without loss of
generality, set :math:`C(t) = 0`.  Finally, by introducing :math:`(r\,\phi)` as
the field to be solved for, the governing equation reduces to a 1D Cartesian
wave equation,

.. math::
   \frac{{{\partial ^2}(r{\kern 1pt} \phi )}}{{\partial {r^2}}} =
   \frac{1}{{{c_L}^2}}
   \frac{{{\partial ^2}(r{\kern 1pt} \phi )}}{{\partial {t^2}}}\,\ldotp
   :label: rPhiEq

.. _formul-soln:

========================
Formulation and solution
========================

The formulation is completed by augmenting :eq:`rPhiEq` with suitable boundary
and initial conditions (BC and IC).  At the cavity surface we require that the
radial normal stress component equal the imposed normal traction or pressure
history: :math:`{\left. {{T_{rr}}} \right|_{r = a}} = p(t)`, while at infinity
:math:`\boldsymbol u` must vanish for all time. Substituting
:math:`\boldsymbol{E} = sym(\boldsymbol{\nabla} \boldsymbol{u})` into
:eq:`strsStrn` and replacing :math:`u \mathrm{\ with\ }{\partial
\phi}/{\partial r}` these BCs may be written,

.. math::
   {\left. {\left( {(\lambda  + 2\mu )
   \frac{{{\partial ^2}\phi }}{{\partial {r^2}}} +
   \frac{{2\lambda }}{r}\frac{{\partial \phi }}{{\partial r}}} \right)}
   \right|_{(r = a,\;t)}} =  - p(t); \quad \phi(r \to \infty ,\;t) = 0
   \,\ldotp
   :label: bdryCon

At the initial time the material is undistorted and quiescent:
:math:`u(r,t=0) = ({\partial u}/{\partial t})(r,t=0) = 0`.

In terms of :math:`\phi` the ICs become,

.. math::
    \phi (r,{\kern 1pt} \,t = 0) = 0\,;\quad {\left.
    {\frac{{\partial \phi }}{{\partial t}}} \right|_{t = 0}} = 0 \,,
    :label: initCon

where again we have set several arbitrary functions of time to zero.

A general solution to :eq:`rPhiEq`, :eq:`bdryCon`, :eq:`initCon`, may be
constructed conveniently by use of the Laplace transform [#zwi12]_, see
Hutchens [#hut05]_ for details of the calculation.  We introduce a rescaled
time, :math:`t'(r,t) = t - (r - a)/c_L`, and the definitions,

.. math::
   \alpha  = \frac{{2{c_T}^2}}{{a{c_L}}},
   \quad {\beta ^2} = {\alpha ^2}\left( {\frac{{{c_L}^2}}{{{c_T}^2}} - 1}
       \right),
   \quad {c_L}^2 = \frac{{\lambda  + 2\mu }}{{{\rho _0}}},
   \quad {c_T}^2 = \frac{\mu }{{{\rho _0}}} \,,
   :label: alphaBetaClCt

where :math:`c_L \mathrm{\ and\ } c_T` are the longitudinal and shear wave
speeds.  Then, for an arbitrary :math:`p(t)`, the time-domain solution may be
written formally as,

.. math::
   \phi(r,t) =  - \frac{a}{\rho }\frac{1}{{\beta {\kern 1pt} r}}
   \int\limits_0^{t'(r,\,t)} {p\left( {t'(r,t) - \tau }
   \right){e^{ - \alpha \,\tau }}\sin (\beta \tau )\;d\tau }
   :label: timeDomSoln

Provided that this integral is calculable in terms of elementary or tabulated
special functions or a convergent series, the solution can be computed at high
accuracy and can be useful for verification.  All fields relevant for
comparison with hydrocode results are obtained by calculating :math:`\mathbf E`
from :math:`\mathbf{u} = r\,\phi(r,t)\,\mathbf{\hat{e}}_r` and then
substituting into :eq:`strsStrn` with specific values of the Lame and other
parameters.  For the specific loading history, :math:`p(t) = P_0\,H(t)`, where
:math:`\mathit{H}` is the Heaviside or unit-step function and :math:`P_0` the
imposed pressure scale, :eq:`timeDomSoln` takes the form,

.. math::
   \phi (r,t') =  - \frac{a}{\rho }\frac{{{P_0}\operatorname{H} (t')}}
   {{({\alpha ^2} + {\beta ^2}){\kern 1pt} \,r}}\left[ {1 - {e^{ - \alpha t'}}
   \left\{ {\cos (\beta t') + \frac{\alpha }{\beta }\sin (\beta t')}
   \right\}} \right]\ldotp
   :label: heavisideSoln

This is the case implemented in this initial version of the Blake problem
solver in *ExactPack*.

By default, :py:mod:`exactpack.solvers.blake` loads
:py:mod:`exactpack.solvers.blake.blake`.

==========
References
==========

.. [#ald02] Aldridge, David F. *Elastic Wave Radiation from a Pressurized
   Spherical Cavity*. Report SAND2002-1882, Sandia National Laboratories (SNL),
   Albuquerque, NM 87185: SNL (2002).

.. [#bla52] Blake, F. G. *Spherial Wave Propagation in Solid Media*. The
   Journal of the Acoustical Society of America (Acoustical Society of America)
   **24**, no. 2, p. 211 (1952).

.. [#bro08] Brock, Jerry S. *Blake Test Problem Parameters*. Report
   LA-UR-08-3005, Los Alamos National Laboratory (LANL), Los Alamos,
   NM 87545: LANL (2008).

.. [#den91] Denny, Marvin D. and Lane R. Johnson. *The Explosion Seismic Source
   Function: Models and Scaling Laws Reviewed*. In "Explosion Source
   Phenomenology", by Steven R Taylor, Patton Howard J and Paul G Richards,
   pp. 1-24. Washington D.C.: American Geophysical Union (1991).

.. [#gur83] Morton E. Gurtin, "Linear Theory of Elasticity" in Mechanics of
   Solids, ed. C. Truesdell, Springer, 1983.  Reprinted from "Handbuch der
   Physik", vol. VIa/2, ed. S. :math:`\text{Fl}\ddot{\text{u}}\text{gge}`,
   Springer, 1972.

.. [#hut05] Hutchens, Gregory J. *An Analysis of the Blake Problem*. Report
   LA-UR-05-8737, Los Alamos National Laboratory (LANL), Los Alamos, NM 87545:
   LANL (2005).

.. [#jef31] Jeffries, Harold. *On the Cause of Oscillatory Movement in
   Seismograms*. Geophysical Journal International (Oxford University Press)
   **2**, pp. 407-416 (1931).

.. [#kam09] Kamm, James R., and Lee A. Ankeny. *Analysis of the Blake Problem
   with RAGE*. Report LA-UR-09-01255, Los Alamos National Laboratory
   (LANL), Los Alamos, NM 87545: LANL (2009).

.. [#sha42] Sharpe, Joseph A. *The production of elastic waves by explosion
   pressures I, Theory and empirical field observations*. Geophysics (Society
   of Exploration Geophysicists) **7**, no. 2, pp. 144-154 (1942).

.. [#zwi12] Zwillinger, Daniel. CRC Standard Mathematical Tables and Formulae
   (32nd Edition), Taylor & Francis (2012).

"""

from blake import Blake
