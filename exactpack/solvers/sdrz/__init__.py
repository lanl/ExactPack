r"""The Steady Detonation Reaction Zone Problem. 

The steady detonation reaction zone (SDRZ) problem tests reactive high
explosives (HE) burn capability, and was first published by
[Fickett1974]_. This problem describes a steady supported
Chapman-Jouguet (C-J) detonation with finite reaction-zone thickness.
(Supported means that a piston is driving the reaction products from
behind at exactly the C-J velocity.) The problem assumes a single forward
reaction from species :math:`A` to species :math:`B`  of the form
:math:`A \rightarrow B`, where :math:`\lambda` is the mass fraction
of species :math:`B`, so that :math:`\lambda` evolves from 0.0 to 1.0
as the detonation progresses.

For specified detonation velocity :math:`D`, the equations of motion
have a solution that is steady in the frame attached to the shock.
With the equation-of-state :math:`p(\rho,e,\lambda)`, the Rankine-Hugoniot
relations can be solved to obtain :math:`p,\rho, \text{and } u` as
functions of :math:`\lambda` alone. The reaction rate :math:`r(\rho,e,\lambda)`
is then also a function of :math:`\lambda` alone:

..  math::

    \frac{d\lambda}{dt}=r(\lambda) \ .

The value of :math:`x`, which is the distance from the shock for a 
Lagrangian particle that begins at the shock location, is derived from

..  math::

    \frac{dx}{dt} = D - u(\lambda) \ .

Both species :math:`A` species :math:`B`, are described by the polytropic gas
equation of state with heat of reaction :math:`q`,

..  math::

    p = (\gamma - 1) \rho (e + \lambda q)  \ .

General Solution
----------------

There exists a general solution, given :math:`D, \rho_0, \gamma, q, \lambda(t)`.
A set of constants exist (independent of :math:`t`), 

..  math::

    {D_j}^2 &= 2(\gamma^2-1)q
    \\
    f &= (D/D_j)^2
    \\
    p_j &= \rho_0 D^2 / (\gamma+1)
    \\
    \rho_j &= \frac{\rho_0(\gamma + 1)}{\gamma} \ ,

as well as a set of time-dependent functions,

..  math::

    g(t) &= \sqrt{(1-\lambda(t)/f)}
    \\
    p(t) &= f p_j (1+g(t))
    \\
    \rho(t) &= \rho_j /(1-g(t)/\gamma)
    \\
    u(t) &= (1 - \rho_0/\rho(t))D
    \\
    cs(t) &= \sqrt{\gamma p(t)/\rho(t)} 


Special case for SDRZ problem
-----------------------------

Assume :math:`D=D_j` and the rate function

..  math::

    d\lambda/dt = 2 \sqrt{1-\lambda} 


and thus

..  math::

    \lambda = t(2-t) \ .


Then the constants are

..  math::

    p_j &= \rho_0 D^2 / (\gamma+1)
    \\
    \rho_j &= \frac{\rho_0(\gamma + 1)}{\gamma} \ ,

and the time-dependent functions are

..  math::

    g(t) &= 1-t
    \\
    p(t) &= p_j(2-t)
    \\
    \rho(t) &= \frac{\rho_j\gamma}{\gamma+t-1}
    \\
    u(t) &= (1 - \rho_0/\rho(t))D
    \\
    cs(t) &= \sqrt{\gamma p(t)/\rho(t)} \ .

Then, for :math:`t \leq 1`:

..  math::

    x(t) = \frac{\rho_0 D_j}{\rho_j} \left[\left(1-\frac{1}{\gamma}\right)t +
    \frac{t^2}{2\gamma}\right] \ ;

otherwise:
    
..  math::

    x(t) = x|_{t=1} + (t-1)(D-u|_{t=1})

Fickett & Rivard suggest comparison at :math:`t=0.5` s. Note that in the 
computational solution, care must be taken to ensure that :math:`\lambda(t)`
is monotonic and does not exceed a value of 1.0.

Test Case
=========

Use results from Table 10.1 of [Fickett1974]_ with:

..  math::

    0 \leq t \leq & 1.2 ~~~\text{every}~~~ 0.1
    \\
    D & =  0.85
    \\
    \rho_0 & =  1.6
    \\
    \gamma & =  3 \ .

Results should match:

..  Table 10.1 from [Fickett1974]_

=============== ============ ============== ==================  ================ ================== ================== ===============
t               x            p              u                   :math:`\rho`     cs                 D                  :math:`\lambda`
[:math:`\mu s`] [:math:`cm`] [:math:`Mbar`] [:math:`cm/\mu s`]  [:math:`g/cm^3`] [:math:`cm/\mu s`] [:math:`cm/\mu s`] 
=============== ============ ============== ==================  ================ ================== ================== ===============
0.0             0.0000000    0.578000       0.425000            3.200000         0.7361216           0.85                0.00
0.1             0.0435625    0.549100       0.403750            3.047619         0.7352009           0.85                0.19
0.2             0.0892500    0.520200       0.382500            2.909091         0.7324317           0.85                0.36
0.3             0.1370625    0.491300       0.361250            2.782609         0.7277931           0.85                0.51
0.4             0.1870000    0.462400       0.340000            2.666667         0.7212489           0.85                0.64
0.5             0.2390625    0.433500       0.318750            2.560000         0.7127467           0.85                0.75
0.6             0.2932500    0.404600       0.297500            2.461538         0.7022152           0.85                0.84
0.7             0.3495625    0.375700       0.276250            2.370370         0.6895617           0.85                0.91
0.8             0.4080000    0.346800       0.255000            2.285714         0.6746666           0.85                0.96
0.9             0.4685625    0.317900       0.233750            2.206897         0.6573776           0.85                0.99
1.0             0.5312500    0.289000       0.212500            2.133333         0.6375000           0.85                1.00
1.2             0.6587500    0.289000       0.212500            2.133333         0.6375000           0.85                1.00
=============== ============ ============== ==================  ================ ================== ================== ===============

Note: Corrected value 0.70125 to 0.65875 for x value at t=1.2
(Error in original table and formula)
"""

from sdrz import SteadyDetonationReactionZone
