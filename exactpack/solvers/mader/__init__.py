r"""The Mader rarefaction burn wave.

The Mader Problem is a one-dimensional piston-driven detonation wave with 
a trailing polytropic rarefaction, in which a slab of high 
explosive (HE) is initiated on one side and a detonation wave propagates 
to the other side.  The Mader Problem tests a code's ability to compute 
the evolution of the rarefaction behind the burn front, and the CJ state 
at the detonation front, assuming the detonation wave propagates through 
a one-dimensional compressible gas.

The Mader Problem is a special case of the detonation wave solution
given on page 24 of [Fickett1979]_.  A detonation wave is
driven by a piston moving in the +x direction in a one-dimensional 5 cm 
slab of gamma-law gas. A rarefaction (i.e., a Taylor wave) follows the 
detonation front.  The Mader Problem solution is given here in the rest 
frame of the piston.  The head of the rarefaction is at the detonation 
front and the tail is half-way between the front and the piston. For
detailed analysis see Timmes, *et al.*, [Timmes2005]_,  and Kirkpatrick, 
*et al.* [Kirkpatrick2004]_,  See also  [Kamm2008]_.

.. figure:: ../../../doc/source/solvers/mader_diagram.png
    :align: center
    :scale: 70 %

    Geometry of the Mader Problem

In the Mader Problem, there is no explicit reaction chemistry, and the reaction zone
has zero length and the reaction energy has a fixed value.  The CJ detonation 
speed is given as a parameter, along with the reaction enthalpy (chemical energy 
released), and the initial state ahead of the burn front. Given these 
parameters and the assumptions of the model, all of the remaining quantities can 
be computed analytically. The Hugoniot and CJ state are easily found from the jump 
conditions and Rayleigh line; the Euler equations yield a self-similar analytic
solution for the Taylor wave behind  the burn front. This is the essence of the 
Mader Problem. 

The structure of the detonation wave in the slab is divided into three regions:  

I.   Ahead of the shock front, the unburned gas is in a uniform initial state (Region I).
II.  The rarefaction region, proceeding from the the moving shock front to a terminal 
     point exactly half-way between the shock front and the piston.  This is a consequence
     of the polytropic assumption with :math:`\gamma=3` (Region II).
III. Between the piston and the rarefaction, the material is in a uniform 
     final state (Region III).

At the final time of :math:`6.25 \mu s`, the detonation front has reached the end of
the slab, at 5 cm, and the terminal point of the rarefaction will be at 2.5 cm. The 
Hugoniot relations and final-state isentrope determine the CJ state, which is the
state of the reaction products as they exit the detonation front in Region II.  

Assuming a :math:`\gamma`-law gas, with equation of state (EOS) :math:`p=(\gamma-1)\rho e`, the CJ 
state is given by,

.. math::
   
   p_{\rm cj} = \frac{\rho_{0} D_{\rm cj}}{(\gamma+1)} 

   \rho_{\rm cj} = \rho_{0} \left( \frac{\gamma+1}{\gamma}\right)   

   c_{\rm cj} = D_{\rm cj} \left(\frac{\gamma}{\gamma+1}\right)

   \upsilon_{\rm cj} = \frac{D_{\rm cj}}{\gamma+1} \ .

The specific reaction enthalpy :math:`q`  and the position of the detonation front at 
time :math:`t` are given by,

.. math::

   q = \frac{D_{\rm CJ}^{2}}{2 (\gamma^2 - 1)}

   x_{\rm det} = D_{\rm cj} t \ .

In Region I, the material is in the constant initial state given above, but, is assumed
to be moving in the frame of the piston in the -x direction with speed,
:math:`\upsilon_{0} = -\upsilon_{\rm piston}`, where :math:`\upsilon_{\rm piston}` is the
speed of the piston in the lab frame. Thus, the initial state in Region I can be chosen to
have the following simple form, consistent with the imposed Hugoniot, detonation speed, and
chemical energy,

.. math:: 

  \upsilon = -\upsilon_{\rm piston}

  p = 0.0

  \rho = 1.857

  c = 0 \ .


The rarefaction fan in Region II consists of the set of characteristics, :math:`\upsilon + c = x/t`.
The flow is self-similar here, determined only by the ratio, :math:`x/t`.
The characteristics are bounded between :math:`x/t = D_{\rm cj}`, at the detonation front  
and, :math:`x/t = D_{\rm cj}/2`, at the tail (for the polytropic case).  So, the transition point 
between tail of the the rarefaction fan and the final state at any time is given 
by :math:`x_{\rm tail} = (1/2) D_{\rm cj} t`.  Then, the self-similar solution for the Taylor
wave in Region II is found to be,

.. math:: 

  \upsilon = \frac{(2 (x/t) - D_{\rm cj})}{(\gamma+1)} 

  p/p_{\rm cj} = \left[ 1 + \frac{(\gamma-1)(\upsilon - \upsilon_{\rm cj})}{2c_{\rm cj}}
  \right]^{2\gamma/(\gamma-1)}    

  \rho = \rho_{\rm cj} (p / p_{\rm cj})^{1/\gamma}  

  c = c_{\rm cj} (p / p_{\rm cj})^{(\gamma - 1)/2\gamma} \ .

The constant final state in Region III, equal to the state at the tail of the rarefaction. Thus we have,

.. math:: 

  \upsilon = 0

  p/p_{\rm cj} = \left[ 1 - \frac{\upsilon_{\rm cj} (\gamma-1)}{2 c_{\rm cj}}
  \right]^{2\gamma/(\gamma-1)}    

  \rho = \rho_{\rm cj} (p/p_{\rm cj})^{1/\gamma}

  c = c_{\rm cj} (p / p_{\rm cj})^{(\gamma - 1)/2\gamma} \ .

To obtain the solutions above in the lab frame, simply subtract :math:`\upsilon_{\rm piston}` 
from all of the velocities, e.g.,

.. math:: 

  \upsilon \rightarrow (\upsilon - \upsilon_{\rm piston}) \ .

One will also need to subtract the initial position of the detonation front from all of the
:math:`x` quantities above if you started the detonation from some position other than the
initial position of the piston.


The following table gives the default parameters for Mader Problem. The specific reaction enthalpy,
:math:`q = D_{cj}^{2}/2 (\gamma^2 - 1)`, is computed from the values of :math:`D_{cj}` and :math:`\gamma`.

.. _mader-table-1:

.. table:: Parameters for the Mader Problem.

   +------------------------+----------------+------------------------+--------------------------------+
   | :math:`t_\mathrm{fin}` | :math:`\gamma` | :math:`D_{cj}`         | :math:`q`                      |
   +------------------------+----------------+------------------------+--------------------------------+
   | [:math:`\mu{\rm s}`]   | [-]            | [cm/:math:`\mu{\rm s}`]| [erg/gm]                       |
   +========================+================+========================+================================+
   | 6.25                   | 3              | 0.8                    |  4.0 x 10 :sup:`10`            |
   +------------------------+----------------+------------------------+--------------------------------+


Initial conditions in Region I are given in the table below.
This table represents a nominal initial state in Region I corresponding to the analytic solution
given here. The rarefaction solution should be somewhat (or entirely) insensitive to the initial
pressure and sound speed as long as a simulation code is able to sufficiently approximate the
given CJ state directly behind the detonation front, so there may be some freedom to deviate
somewhat from this prescription, if necessary, for running the Mader Problem on individual codes.

.. _mader-table-2:

.. table:: Initial Conditions for the Mader Problem.

   +--------------------------------+------------------+-------------------+-------------------------+
   | :math:`\upsilon_{0}`           | :math:`\rho_{0}` | :math:`p_{0}`     | :math:`c_{0}`           |
   +--------------------------------+------------------+-------------------+-------------------------+
   | [cm/:math:`\mu{\rm s}` ]       | [g/cm :sup:`3`]  | [dyn/cm :sup:`2`] | [:math:`{\rm cm/\mu s}`]|
   +================================+==================+===================+=========================+
   | :math:`-\upsilon_{piston}`     | 1.875            | 0.0               | 0.0                     |
   +--------------------------------+------------------+-------------------+-------------------------+

Theory Discussion
-----------------
The Mader Problem is based on the simplest HE detonation theory, as
outlined in section 2A of Fickett and Davis [Fickett1979]_.  This theory
is based on the following assumptions:

1. One-dimensional flow in a simple polytropic gas expansion.
2. The planar detonation front is a jump discontinuity or shock in which the
   thermodynamic path is the Rayleigh line connecting the initial and
   final states, as determined by the shock Hugoniot of the unreacted HE, modified
   by the addition of the full chemical reaction enthalpy.
3. The chemical reaction is assumed to burn to completion instantaneously,
   so the reaction products are emerge, in equilibrium, from the detonation front.
4. The detonation front motion is assumed to to be steady, so the the state
   of the material emerging from the shock front is time-independent.

In the simple piston detonation problem, the fuel is assumed to burn
instantaneously and completely at the detonation front, which may be treated as idealized shock
front, thus the material in the trailing rarefaction is assumed to consist entirely of reaction
products. Therefore, the reaction enthalpy - the total heat produced by burning the fuel -
is simply added to the energy Hugoniot.  Using this to derive the total Hugoniot from
the conservation laws, one finds that the slope of the P-V Hugoniot, which is
proportional to :math:`D_{cj}^2`, is only changed by the linear addition of this
chemical energy. Equating the expression for the slope of the P-V Hugoniot with the
slope of the Rayleigh line one finds that, :math:`(\partial{E}/\partial{v})_{\cal{H}} = -p`,
on the Hugoniot.  But, this relation also obtains on the isentrope of the
reaction products, i.e., :math:`(\partial{E}/\partial{v})_{\cal{S}} = -p`. Therefore,
the CJ state is a special point where the slope is the same for the Rayleigh line,
the Hugoniot *and* the reaction product isentrope. This slope is proportional to
:math:`D_{cj}^2`. This implies that :math:`D_{cj}` is the stable propagation speed for
the detonation wave: If the detonation wave propagates a bit *faster* than :math:`D_{cj}`,
the reaction products will be moving away from the detonation front faster than the local
sound speed, so pressure disturbances due to the energy released from the reaction will
not be able to support the shock wave. Therefore, the detonation wave should slow
down until disturbances from behind "catch up". On the other hand, if the detonation
wave travels a bit *slower* than :math:`D_{cj}`, the sound speed for the reaction
products will be greater the  detonation speed, so pressure disturbances from the
reaction products will overtake it, increasing the shock strength - the pressure
and shock speed - until it reaches :math:`D_{CJ}`, at which point
:math:`u_{cj}+c_{cj} = D_{cj}`, i.e, the detonation is exactly sonic in the frame
moving with the detonation front.
"""

from .timmes import Mader
