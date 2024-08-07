import scipy.integrate
from scipy.optimize import bisect
from numpy import linspace, array, sqrt, interp, append, where, argmin, argmax, exp, shape, abs

def JWL_f(r, g, inst):
  """The JWL EOS ammends the IGEOS pressure with a term depending on density,
     gamma and empirically determined constants.
  """
  G   = g - 1.
  r0 = inst.r0
  R1r = inst.R1 * r0 / r
  R2r = inst.R2 * r0 / r
  A, B = inst.A, inst.B
  return A * (1. - G / R1r) * exp(- R1r) + B * (1. - G / R2r) * exp(- R2r)

def JWL_dfdr(r, g, inst):
  G  = g - 1.
  r0 = inst.r0
  R1r = inst.R1 * r0 / r
  R2r = inst.R2 * r0 / r
  val  = inst.A * (R1r / r - G / inst.R1 / r0 - G / r) * exp(- R1r)
  val += inst.B * (R2r / r - G / inst.R2 / r0 - G / r) * exp(- R2r)
  return val

def sie(p, r, g, inst):
  JWL_fval = 0 if (inst.problem == 'igeos') else JWL_f(r, g, inst)
  return (p - JWL_fval) / (g - 1.) / r

def dsdr_cP(p, r, g, inst):
  """See [Kamm2015]_ eqs. 5 & 6. The generalized definition for sound speed
     contains the derivative of SIE wrt rho at constant p -> 'dsdr_cP'
  """
  G = g - 1.
  if ('JWL' in inst.problem):
  # the JWL result
    return - JWL_dfdr(r, g, inst) / G / r - (p - JWL_f(r, g, inst)) / G / r**2
  else:
  # the ideal-gas result
    return - p / G / r**2

def dsdp_cR(p, r, g, inst):
  """See [Kamm2015]_ eqs. 5 & 6. The generalized definition for sound speed
     contains the derivative of SIE wrt p at constant rho -> 'dsdp_cR'
  """
  # the ideal-gas and JWL results are the same
  return 1. / r / (g - 1.)

# Copying and modifying the above equations for a JWL EOS, for example, produces
# the associated sound speed.

# This is the sound speed for the generalized EOS.
def sound_speed(p, r, g, inst):
  """See [Kamm2015]_ eqs. 5 & 6. The generalized definition for sound speed
     contains the derivative of SIE wrt p at constant rho |-> dsdp_cR
  """
  if (inst.problem == 'igeos'):
    return sqrt(g * p / r)
  else:
    return sqrt((p / r**2 - dsdr_cP(p, r, g, inst)) / dsdp_cR(p, r, g, inst))

# These are the generalized ODEs for density and velocity wrt pressure.
def drdp_dudp(p, vals, g, wave_sign, inst):
  r, u = vals
  a = sound_speed(p, r, g, inst)
  drdp_val = 1. / a**2
  dudp_val = 1. / r / a * wave_sign
  return [drdp_val, dudp_val]

# Generalized shock jump for the generalized wave speed relation below.
def shock_jump(p0, r0, g0, p, r, inst):
  e0 = sie(p0, r0, g0, inst)
  e  = sie(p,  r,  g0, inst)
  val  = e0 + p0 / r0 + r  / r0 * (p - p0) / (r - r0) / 2.
  val -= e  + p  / r  + r0 / r  * (p - p0) / (r - r0) / 2.
  return val

# Generalized wave speed for the generalized shock jump relation above.
def shock_speed(pa, ra, pb, rb, u, inst):
  sgn = 1
  if (len(shape(array(pa))) == len(shape(array(pb)))):
    sgn = -1 if ((pb == inst.pl) and (rb == inst.rl) and (u == inst.ul)) else 1
  return sgn * sqrt( ra / rb * (pa - pb) / (ra - rb)) + u

# Generalized wave speed for the generalized shock jump relation above.
def star_velocity(p0, r0, u0, p, r, inst):
  sgn = -1 if ((p0 == inst.pl) and (r0 == inst.rl) and (u0 == inst.ul)) else 1
  val = shock_speed(p,r,p0,r0,0,inst) - shock_speed(p0,r0,p,r,0,inst)
  return u0 + val * sgn

# Rarefaction state integrator.
def r_int_call(init_vals, fparams, pmin, inst):
  integ_array = linspace(init_vals[2], pmin, inst.num_int_pts + 2)[1:-1]
  dt = integ_array[1] - integ_array[0]
  i = scipy.integrate.ode(drdp_dudp)
  i.set_initial_value(init_vals[:2], init_vals[2])
  i.set_f_params(fparams[0], fparams[1], inst)
  i.set_integrator('vode', atol = inst.int_tol, rtol = inst.int_tol,
                   method = 'bdf', nsteps = int(1e6))
  k, rs, us = 0, [], []
  while (i.successful() and (k < inst.num_int_pts) & (i.y[0] > 0)):
    i.integrate(i.t + dt)
    if (i.y[0] > 0):
      rs.append(i.y[0])
      us.append(i.y[1])
      k += 1
  return [integ_array[::-1], array(rs[::-1]), array(us[::-1])]

# Shock state match conditions.
def match_shocks(pmax, p, r, u, g, inst):
  sgn = -1 if ((p == inst.pl) and (r == inst.rl) and (u == inst.ul)) else 1
  shock_array = linspace(p, pmax, inst.num_int_pts + 2)
  shock_array[0] = (shock_array[1] - shock_array[0]) * 1.e-8 + shock_array[0]
  rxs, rx0, rxf = [], (1. + inst.int_tol) * r, (g + 1.) / (g - 1.) * r
  for px in shock_array:
    try:
      rxs.append(bisect(lambda rx: shock_jump(p, r, g, px, rx, inst), rx0, rxf))
    except:
      print('failed px = ', px)
      break
  rxs = array(rxs)
  uxs = star_velocity(p, r, u, shock_array[:len(rxs)], rxs, inst)
  return [shock_array[:len(rxs)], rxs, uxs]

def rarefaction(px, p, r, u, g, inst):
  a = sound_speed(p, r, g, inst)
  return 2.*a / (g - 1.) * (1. - (px / p)**((g - 1.) / 2. / g)) + u

def shock(px, p, r, u, g, inst):
  A = 2. / (g + 1.) / r
  B = (g - 1.) / (g + 1.) * p
  return (px - p) * sqrt(A / (px + B)) + u

def RCS_call(p, inst):
  rl, pl, ul, gl = inst.rl, inst.pl, inst.ul, inst.gl
  rr, pr, ur, gr = inst.rr, inst.pr, inst.ur, inst.gr
  return shock(p,pr,rr,ur,gr,inst) - rarefaction(p,pl,rl,ul,gl,inst)

def SCR_call(p, inst):
  rl, pl, ul, gl = inst.rl, inst.pl, inst.ul, inst.gl
  rr, pr, ur, gr = inst.rr, inst.pr, inst.ur, inst.gr
  return rarefaction(p,pr,rr,ur,gr,inst) - shock(p,pl,rl,ul,gl,inst)

def RCR_call(p, inst):
  rl, pl, ul, gl = inst.rl, inst.pl, inst.ul, inst.gl
  rr, pr, ur, gr = inst.rr, inst.pr, inst.ur, inst.gr
  return rarefaction(p,pr,rr,-ur,gr,inst) + rarefaction(p,pl,rl,ul,gl,inst)

def SCS_call(p, inst):
  rl, pl, ul, gl = inst.rl, inst.pl, inst.ul, inst.gl
  rr, pr, ur, gr = inst.rr, inst.pr, inst.ur, inst.gr
  return shock(p, pr, rr, ur, gr, inst) + shock(p, pl, rl, -ul, gl, inst)

def rho_star_shock(px, p, r, g, inst):
  return r * (p * (g-1.) + px * (g+1.)) / (px * (g-1.) + p * (g+1.))

def rho_star_rarefaction(px, p, r, g, inst):
  return r * (px / p)**(1. / g)

def rho_p_u_rarefaction(p, r, u, g, x, xd0, t, inst):
  sgn = 1 if ((p == inst.pl) and (u == inst.ul) and (r == inst.rl)) else -1
  a = sound_speed(p, r, g, inst)
  y = 2. / (g + 1.) + sgn * (g - 1.) / a / (g + 1.) * (u - (x - xd0) / t)
  v = 2. * (sgn * a + (g - 1.) * u / 2. + (x - xd0) / t) / (g + 1.)
  return r * y**(2. / (g - 1.)), p * y**(2. * g / (g - 1.)), v

def shock_velocity(px, p, r, u, g, inst):
  sgn = -1 if ((p == inst.pl) and (u == inst.ul) and (r == inst.rl)) else 1
  a = sound_speed(p, r, g, inst)
  return u + sgn * a * sqrt((g + 1.) * px / 2. / g / p + (g - 1.) / 2. / g)

def arp(p, inst):
  return sound_speed(p, inst.rr, inst.gr, inst)

def u_SCN(px, inst):
  u, a, g, p = inst.ul, inst.al, inst.gl, inst.pl
  return u - a/g* (px/p - 1.) / sqrt((g+1.) / 2. / g * px/p + (g-1.) / 2. / g)

def u_NCS(px, inst):
  u, g, p = inst.ul, inst.gr, inst.pl
  arp = sound_speed(px, inst.rr, inst.gr, inst)
  return u - arp / g * (p / px - 1.) / sqrt((g + 1.) / 2. / g * p / px + (g - 1.) / 2. / g)

def u_NCR(px, inst):
  u, g, p = inst.ul, inst.gr, inst.pl
  arp = sound_speed(px, inst.rr, inst.gr, inst)
  return u + 2. * arp / (g - 1.) * (1. - (p / px)**((g - 1.) / 2. / g))

def u_RCN(px, inst):
  u, a, g, p = inst.ul, inst.al, inst.gl, inst.pl
  return u + 2. * a / (g - 1.) * (1. - (px / p)**((g - 1.) / 2. / g))

def u_a(p, inst):
  return inst.ul + 2. * sound_speed(p, inst.rr, inst.gr, inst) / (inst.gr - 1.)

def u_RCVR(p, inst):
  return inst.ul_tilde + 2.*sound_speed(p,inst.rr,inst.gr,inst) / (inst.gr - 1.)
