import scipy.integrate
from scipy.optimize import bisect
from numpy import linspace, array, sqrt, interp, append, where, argmin, argmax, exp, shape, abs

def JWL_f(r, g, self):
  """The JWL EOS ammends the IGEOS pressure with a term depending on density,
     gamma and empirically determined constants.
  """
  G   = g - 1.
  r0 = self.r0
  R1r = self.R1 * r0 / r
  R2r = self.R2 * r0 / r
  A, B = self.A, self.B
  return A * (1. - G / R1r) * exp(- R1r) + B * (1. - G / R1r) * exp(- R2r)

def JWL_dfdr(r, g, self):
  G  = g - 1.
  r0 = self.r0
  R1r = self.R1 * r0 / r
  R2r = self.R2 * r0 / r
  val  = self.A * (R1r / r - G / self.R1 / r0 - G / r) * exp(- R1r)
  val += self.B * (R2r / r - G / self.R2 / r0 - G / r) * exp(- R2r)
  return val

def sie(p, r, g, self):
  JWL_fval = 0 if (self.problem == 'igeos') else JWL_f(r, g, self)
  return (p - JWL_fval) / (g - 1.) / r

def dsdr_cP(p, r, g, self):
  """See [Kamm2015]_ eqs. 5 & 6. The generalized definition for sound speed
     contains the derivative of SIE wrt rho at constant p -> 'dsdr_cP'
  """
  G = g - 1.
  if ('JWL' in self.problem):
  # the JWL result
    return - JWL_dfdr(r, g, self) / G / r - (p - JWL_f(r, g, self)) / G / r**2
  else:
  # the ideal-gas result
    return - p / G / r**2

def dsdp_cR(p, r, g, self):
  """See [Kamm2015]_ eqs. 5 & 6. The generalized definition for sound speed
     contains the derivative of SIE wrt p at constant rho -> 'dsdp_cR'
  """
  # the ideal-gas and JWL results are the same
  return 1. / r / (g - 1.)

# Copying and modifying the above equations for a JWL EOS, for example, produces
# the associated sound speed.

# This is the sound speed for the generalized EOS.
def sound_speed(p, r, g, self):
  """See [Kamm2015]_ eqs. 5 & 6. The generalized definition for sound speed
     contains the derivative of SIE wrt p at constant rho |-> dsdp_cR
  """
  if (self.problem == 'igeos'):
    return sqrt(g * p / r)
  else:
    return sqrt((p / r**2 - dsdr_cP(p, r, g, self)) / dsdp_cR(p, r, g, self))

# These are the generalized ODEs for density and velocity wrt pressure.
def drdp_dudp(p, vals, g, wave_sign, self):
  r, u = vals
  a = sound_speed(p, r, g, self)
  drdp_val = 1. / a**2
  dudp_val = 1. / r / a * wave_sign
  return [drdp_val, dudp_val]

# Generalized shock jump for the generalized wave speed relation below.
def shock_jump(p0, r0, g0, p, r, self):
  e0 = sie(p0, r0, g0, self)
  e  = sie(p,  r,  g0, self)
  val  = e0 + p0 / r0 + r  / r0 * (p - p0) / (r - r0) / 2.
  val -= e  + p  / r  + r0 / r  * (p - p0) / (r - r0) / 2.
  return val

# Generalized wave speed for the generalized shock jump relation above.
def shock_speed(pa, ra, pb, rb, u, self):
  sgn = 1
  if (len(shape(array(pa))) == len(shape(array(pb)))):
    sgn = -1 if ((pb == self.pl) and (rb == self.rl) and (u == self.ul)) else 1
  return sgn * sqrt( ra / rb * (pa - pb) / (ra - rb)) + u

# Generalized wave speed for the generalized shock jump relation above.
def star_velocity(p0, r0, u0, p, r, self):
  sgn = -1 if ((p0 == self.pl) and (r0 == self.rl) and (u0 == self.ul)) else 1
  val = shock_speed(p,r,p0,r0,0,self) - shock_speed(p0,r0,p,r,0,self)
  return u0 + val * sgn

# Rarefaction state integrator.
def r_int_call(init_vals, fparams, pmin, self):
  integ_array = linspace(init_vals[2], pmin, self.num_int_pts + 2)[1:-1]
  dt = integ_array[1] - integ_array[0]
  i = scipy.integrate.ode(drdp_dudp)
  i.set_initial_value(init_vals[:2], init_vals[2])
  i.set_f_params(fparams[0], fparams[1], self)
  i.set_integrator('vode', atol = self.int_tol, rtol = self.int_tol,
                   method = 'bdf', nsteps = int(1e6))
  k, rs, us = 0, [], []
  while (i.successful() and (k < self.num_int_pts) & (i.y[0] > 0)):
    i.integrate(i.t + dt)
    if (i.y[0] > 0):
      rs.append(i.y[0])
      us.append(i.y[1])
      k += 1
  return [integ_array[::-1], array(rs[::-1]), array(us[::-1])]

# Shock state match conditions.
def match_shocks(pmax, p, r, u, g, self):
  sgn = -1 if ((p == self.pl) and (r == self.rl) and (u == self.ul)) else 1
  shock_array = linspace(p, pmax, self.num_int_pts + 2)[1:]
  rxs, rx0, rxf = [], (1. + self.int_tol) * r, (g + 1.) / (g - 1.) * r
  for px in shock_array:
    try:
      rxs.append(bisect(lambda rx: shock_jump(p, r, g, px, rx, self), rx0, rxf))
    except:
      print('failed px = ', px)
      break
  rxs = array(rxs)
  uxs = star_velocity(p, r, u, shock_array[:len(rxs)], rxs, self)
  return [shock_array[:len(rxs)], rxs, uxs]

def rarefaction(px, p, r, u, g, self):
  a = sound_speed(p, r, g, self)
  return 2.*a / (g - 1.) * (1. - (px / p)**((g - 1.) / 2. / g)) + u

def shock(px, p, r, u, g, self):
  A = 2. / (g + 1.) / r
  B = (g - 1.) / (g + 1.) * p
  return (px - p) * sqrt(A / (px + B)) + u

def RCS_call(p, self):
  rl, pl, ul, gl = self.rl, self.pl, self.ul, self.gl
  rr, pr, ur, gr = self.rr, self.pr, self.ur, self.gr
  return shock(p,pr,rr,ur,gr,self) - rarefaction(p,pl,rl,ul,gl,self)

def SCR_call(p, self):
  rl, pl, ul, gl = self.rl, self.pl, self.ul, self.gl
  rr, pr, ur, gr = self.rr, self.pr, self.ur, self.gr
  return rarefaction(p,pr,rr,ur,gr,self) - shock(p,pl,rl,ul,gl,self)

def RCR_call(p, self):
  rl, pl, ul, gl = self.rl, self.pl, self.ul, self.gl
  rr, pr, ur, gr = self.rr, self.pr, self.ur, self.gr
  return rarefaction(p,pr,rr,-ur,gr,self) + rarefaction(p,pl,rl,ul,gl,self)

def SCS_call(p, self):
  rl, pl, ul, gl = self.rl, self.pl, self.ul, self.gl
  rr, pr, ur, gr = self.rr, self.pr, self.ur, self.gr
  return shock(p, pr, rr, ur, gr, self) + shock(p, pl, rl, -ul, gl, self)

def rho_star_shock(px, p, r, g, self):
  return r * (p * (g-1.) + px * (g+1.)) / (px * (g-1.) + p * (g+1.))

def rho_star_rarefaction(px, p, r, g, self):
  return r * (px / p)**(1. / g)

def rho_p_u_rarefaction(p, r, u, g, x, xd0, t, self):
  sgn = 1 if ((p == self.pl) and (u == self.ul) and (r == self.rl)) else -1
  a = sound_speed(p, r, g, self)
  y = 2. / (g + 1.) + sgn * (g - 1.) / a / (g + 1.) * (u - (x - xd0) / t)
  v = 2. * (sgn * a + (g - 1.) * u / 2. + (x - xd0) / t) / (g + 1.)
  return r * y**(2. / (g - 1.)), p * y**(2. * g / (g - 1.)), v

def shock_velocity(px, p, r, u, g, self):
  sgn = -1 if ((p == self.pl) and (u == self.ul) and (r == self.rl)) else 1
  a = sound_speed(p, r, g, self)
  return u + sgn * a * sqrt((g + 1.) * px / 2. / g / p + (g - 1.) / 2. / g)

def arp(p, self):
  return sound_speed(p, self.rr, self.gr, self)

def u_SCN(px, self):
  u, a, g, p = self.ul, self.al, self.gl, self.pl
  return u - a/g* (px/p - 1.) / sqrt((g+1.) / 2. / g * px/p + (g-1.) / 2. / g)

def u_NCS(px, self):
  u, g, p = self.ul, self.gr, self.pl
  arp = sound_speed(px, self.rr, self.gr, self)
  return u - arp / g * (p / px - 1.) / sqrt((g + 1.) / 2. / g * p / px + (g - 1.) / 2. / g)

def u_NCR(px, self):
  u, g, p = self.ul, self.gr, self.pl
  arp = sound_speed(px, self.rr, self.gr, self)
  return u + 2. * arp / (g - 1.) * (1. - (p / px)**((g - 1.) / 2. / g))

def u_RCN(px, self):
  u, a, g, p = self.ul, self.al, self.gl, self.pl
  return u + 2. * a / (g - 1.) * (1. - (px / p)**((g - 1.) / 2. / g))

def u_a(p, self):
  return self.ul + 2. * sound_speed(p, self.rr, self.gr, self) / (self.gr - 1.)

def u_RCVR(p, self):
  return self.ul_tilde + 2.*sound_speed(p,self.rr,self.gr,self) / (self.gr - 1.)
