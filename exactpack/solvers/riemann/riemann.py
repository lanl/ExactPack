r"""A pure Python, analytic Riemann solver based on the 2013 paper by LoraClavijo, et. al. [LoraClavijo2013]_, and reports by Jim Kamm [Kamm2014, Kamm2015]_.
"""

from scipy.optimize import bisect
from numpy import linspace, array, sqrt, interp, append, where, argmin

from exactpack.solvers.riemann.utils import *

class SetupRiemannProblem(object):
  """ Defines: \\
      1) the 6 classical Riemann problems for an ideal-gas EOS, as well as 1 reversed shocktube problem, and 1 modified Sod shocktube problem; and \\
      2) the Lee and Shyue JWL EOS shocktube problems using the general EOS solver.
  """
  def __init__(self,xmin=0., xd0=0.5, xmax=1., t=0.25,
               rl=1., ul=0., pl=1., gl=1.4, rr=0.125, ur=0., pr=0.1, gr=1.4,
               A=0., B=0., R1=0., R2=0., r0=0., e0=0., problem='igeos',
               num_int_pts=10001, num_x_pts = 10001, int_tol=1.e-12):
      self.problem = problem
      self.num_x_pts = num_x_pts
      self.num_int_pts = num_int_pts
      self.int_tol = int_tol
      self.xmin, self.xd0, self.xmax, self.t = xmin, xd0, xmax, t
      self.pl, self.rl, self.ul, self.gl = pl, rl, ul, gl
      self.pr, self.rr, self.ur, self.gr = pr, rr, ur, gr
      self.A, self.B, self.R1, self.R2, self.r0, self.e0 = A, B, R1, R2, r0, e0

      self.al = sound_speed(pl, rl, gl, self)
      self.ar = sound_speed(pr, rr, gr, self)
      self.el, self.er = sie(pl, rl, gl, self), sie(pr, rr, gr, self)
      self.pmax = 10. * max(pl, pr)


class RiemannIGEOS(SetupRiemannProblem):
  '''
  '''
  def driver(self):
      pl, rl, ul, gl = self.pl, self.rl, self.ul, self.gl
      pr, rr, ur, gr = self.pr, self.rr, self.ur, self.gr
      xmin, xd0, xmax, t = self.xmin, self.xd0, self.xmax, self.t
      num_x_pts = self.num_x_pts
      pmax = self.pmax

      al, ar = sound_speed(pl, rl, gl, self), sound_speed(pr, rr, gr, self)
      el, er = sie(pl, rl, gl, self), sie(pr, rr, gr, self)

      # this if/elif set is based on Fig3 in Gottlieb & Groth
      # determining the solution type considerably aides solution construction
      self.ul_tilde = ul + 2. * al / (gl - 1.) # needed in u_RCVR
      u_RCVR_val = u_RCVR(pr, self)
      u_SCN_val = u_SCN(pr, self)
      u_NCR_val = u_NCR(pr, self)
      u_NCS_val = u_NCS(pr, self)
      u_RCN_val = u_RCN(pr, self)

      # Determine the solution type, star state pressure, px, and shock
      # velocity, ux, if appropriate. These are time-independent.
      z = 1 # this is a sign (+1, -1) that goes in ux, below
      if (((pr >= pl) and (ur <= u_SCN_val)) or
          ((pl > pr) and (ur <= u_NCS_val))):
        soln_type, z = 'shock-contact-shock-SCS', -1
        px = bisect(lambda p: SCS_call(p, self), 0., pmax)
        Vsl = shock_velocity(px, pl, rl, ul, gl, self)
        Vsr = shock_velocity(px, pr, rr, ur, gr, self)
      elif ((pr >= pl) and (u_SCN_val < ur <= u_NCR_val)):
        soln_type, z = 'shock-contact-rarefaction-SCR', -1
        px = bisect(lambda p: SCR_call(p, self), 0., pmax)
        Vs = shock_velocity(px, pl, rl, ul, gl, self)
      elif ((pl > pr) and (u_NCS_val < ur <= u_RCN_val)):
        soln_type = 'rarefaction-contact-shock-RCS'
        px = bisect(lambda p: RCS_call(p, self), 0., pmax)
        Vs = shock_velocity(px, pr, rr, ur, gr, self)
      elif (((pr >= pl) and (u_NCR_val < ur <= u_RCVR_val)) or
            ((pl > pr) and (u_RCN_val < ur <= u_RCVR_val))):
        soln_type = 'rarefaction-contact-rarefaction-RCR'
        px = bisect(lambda p: RCR_call(p, self), 0., pmax)
      elif (ur > u_RCVR_val):
        soln_type = 'R,C,V,C,R'
        print('the solution for this problem is not ready')

      # Determine the star state values for the contact discontinuity, and the
      # left/right star state vals for density, sound speed and internal energy.
      # These values are time-independent.
      ux = ul + z * eval(soln_type.split('-')[0] + "(px,pl,rl,0,gl,self)")
      rx1 = eval('rho_star_'+soln_type.split('-')[0]+'(px,pl,rl,gl,self)')
      rx2 = eval('rho_star_'+soln_type.split('-')[2]+'(px,pr,rr,gr,self)')
      ax1 = sound_speed(px, rx1, gl, self)
      ax2 = sound_speed(px, rx2, gr, self)
      ex1 = sie(px, rx1,  gl, self)
      ex2 = sie(px, rx2,  gr, self)

      # Store velocities that bound regions in time in Vregs. Vh & Vt are the
      # head and tail rarefaction velocities. These values are time-independent.
      # For all solutions, the contact discontinuity travels at Vc = ux.
      if (soln_type == 'shock-contact-shock-SCS'):
        Vregs = array([Vsl, ux, Vsr])
      elif (soln_type == 'shock-contact-rarefaction-SCR'):
        Vregs = array([Vs, ux, ux + ax2, ur + ar])
      elif (soln_type == 'rarefaction-contact-shock-RCS'):
        Vregs = array([ul - al, ux - ax1, ux, Vs])
      elif (soln_type == 'rarefaction-contact-rarefaction-RCR'):
        Vregs = array([ul - al, ux - ax1, ux, ux + ax2, ur + ar])

      # Determine the time-dependent spatial boundaries, append these points
      # to the array 'x', and initialize 'vals' for the physical fields.
      Xregs = xd0 + t * Vregs
      xmin, xmax = min(xmin, 1.1 * min(Xregs)), max(xmax, 1.1 * max(Xregs))
      x = linspace(xmin, xmax, self.num_x_pts)
      x = append(x, Xregs)
      x.sort()
      vals = pl+0.* x, rl+0.*x, ul+0.*x, el+0.*x

      def reg_state(xl, xe, regvals, exactvals):
        p,  r,  u,  e  = regvals
        pe, re, ue, ee = exactvals
        pe = where(xl <= xe, p, pe)
        re = where(xl <= xe, r, re)
        ue = where(xl <= xe, u, ue)
        ee = where(xl <= xe, e, ee)
        return pe, re, ue, ee
    
      def rarefaction_region(p, r, u, g, x, xd0, Xr, t, self):
        rho, prs, vel = rho_p_u_rarefaction(p, r, u, g, x, xd0, t, self)
        e = sie(prs, rho, g, self)
        return reg_state(Xr, x, [prs, rho, vel, e], vals)

      # Determine the time-dependent spatial profiles for the physical fields.
      if (soln_type == 'shock-contact-shock-SCS'):
        vals = reg_state(Xregs[0], x, [px, rx1, ux, ex1], vals)
        vals = reg_state(Xregs[1], x, [px, rx2, ux, ex2], vals)
      elif (soln_type == 'shock-contact-rarefaction-SCR'):
        vals = reg_state(Xregs[0], x, [px, rx1, ux, ex1], vals)
        vals = reg_state(Xregs[1], x, [px, rx2, ux, ex2], vals)
        vals = rarefaction_region(pr, rr, ur, gr, x, xd0, Xregs[2], t, self)
      elif (soln_type == 'rarefaction-contact-shock-RCS'):
        vals = rarefaction_region(pl, rl, ul, gl, x, xd0, Xregs[0], t, self)
        vals = reg_state(Xregs[1], x, [px, rx1, ux, ex1], vals)
        vals = reg_state(Xregs[2], x, [px, rx2, ux, ex2], vals)
      elif (soln_type == 'rarefaction-contact-rarefaction-RCR'):
        vals = rarefaction_region(pl, rl, ul, gl, x, xd0, Xregs[0], t, self)
        vals = reg_state(Xregs[1], x, [px, rx1, ux, ex1], vals)
        vals = reg_state(Xregs[2], x, [px, rx2, ux, ex2], vals)
        vals = rarefaction_region(pr, rr, ur, gr, x, xd0, Xregs[3], t, self)
      p, r, u, e = reg_state(Xregs[-1], x, [pr, rr, ur, er], vals)

      # Storing solution variables
      self.x, self.p, self.r, self.u, self.e = x, p, r, u, e
      self.px,  self.ux,  self.rx1, self.rx2 = px,  ux,  rx1, rx2
      self.ex1, self.ex2, self.ax1, self.ax2 = ex1, ex2, ax1, ax2
      self.ax1, self.ax2 = ax1, ax2
      self.Xregs, self.Vregs = Xregs, Vregs
      self.xmin, self.xmax = xmin, xmax
      self.soln_type = soln_type

      # Storing variables for plotting Fig. 3 in Gottlieb & Groth's 1988 JCP.
      # This is done in the example riemann solutions.
      self.plow = linspace(0., pl, num_x_pts + 1)[1:]
      self.phigh = linspace(pl, pmax, num_x_pts)
      self.ps = array(append(self.plow, self.phigh))
      self.uSCNphigh = u_SCN(self.phigh, self)
      self.uNCRphigh = u_NCR(self.phigh, self)
      self.uNCSplow = u_NCS(self.plow, self)
      self.uRCNplow = u_RCN(self.plow, self)
      self.uaps, self.uRCVRps = u_a(self.ps, self), u_RCVR(self.ps, self)
      

class RiemannGenEOS(SetupRiemannProblem):
  '''
  '''
  def driver(self):
      xmin, xd0, xmax, t = self.xmin, self.xd0, self.xmax, self.t
      pl, rl, ul, gl = self.pl, self.rl, self.ul, self.gl
      pr, rr, ur, gr = self.pr, self.rr, self.ur, self.gr
      al, ar = sound_speed(pl, rl, gl, self), sound_speed(pr, rr, gr, self)
      el, er = sie(pl, rl, gl, self), sie(pr, rr, gr, self)
      pmax = self.pmax

      # Create rarefaction and shock [p, r, u] values as P-U data.
      integ_ps_left,  rls, uls = r_int_call([rl, ul, pl], [gl, -1], 0., self)
      integ_ps_right, rrs, urs = r_int_call([rr, ur, pr], [gr,  1], 0., self)
      shock_ps_left,  rlx, ulx = match_shocks(pmax, pl, rl, ul, gl, self)
      shock_ps_right, rrx, urx = match_shocks(pmax, pr, rr, ur, gr, self)

      # Splice p-u rarefaction and shock values for left & right states.
      ps_left_splice  = append(integ_ps_left,  append(pl, shock_ps_left))
      ps_right_splice = append(integ_ps_right, append(pr, shock_ps_right))
      us_left_splice  = append(uls, append(ul, ulx))
      us_right_splice = append(urs, append(ur, urx))
      
      # Determine star states by seeing where the spliced lines above cross.
      bisect_min = max(ps_left_splice[0], ps_right_splice[0])
      bisect_max = min(ps_left_splice[-1], ps_right_splice[-1])
      px = bisect(lambda px:   interp(px, ps_left_splice, us_left_splice)
                             - interp(px, ps_right_splice, us_right_splice),
                  bisect_min, bisect_max)
      
      # Determine whether px is an element in integ_ps_left, integ_ps_right,
      # shock_ps_left, shock_ps_right. This determines the solution's structure.
      # !!! This set of quantities are time independent.
      soln_type = ['N', 'C', 'N']
      Vregs = []
      if (px < pl):
        soln_type[0] = 'R'
        rx1 = interp(px, integ_ps_left, rls)
        ux1 = interp(px, integ_ps_left, uls)
        ax1 = sound_speed(px, rx1, gl, self)
        ps_left = integ_ps_left[where(integ_ps_left > px)[0]][::-1]
        rs_left = rls[where(rls > rx1)[0]][::-1]
        us_left = uls[where(uls < ux1)[0]][::-1]
        Vregs.append(ul - al)
        Vregs.append(ux1 - ax1)
      elif (px > pl):
        soln_type[0] = 'S'
        rx1 = interp(px, shock_ps_left, rlx)
        ux1 = interp(px, shock_ps_left, ulx)
        ax1 = sound_speed(px, rx1, gl, self)
        ps_left, rs_left, us_left = px, rx1, ux1
        Vregs.append(shock_speed(px, rx1, pl, rl, ul, self))

      Vregs.append(ux1)

      if (px < pr):
        soln_type[-1] = 'R'
        rx2 = interp(px, integ_ps_right, rrs)
        ux2 = interp(px, integ_ps_right, urs)
        ax2 = sound_speed(px, rx2, gr, self)
        ps_right = integ_ps_right[where(integ_ps_right > px)[0]]
        rs_right = rrs[where(rrs > rx2)[0]]
        us_right = urs[where(urs > ux2)[0]]
        Vregs.append(ux2 + ax2)
        Vregs.append(ur + ar)
      elif (px > pr):
        soln_type[-1] = 'S'
        rx2 = interp(px, shock_ps_right, rrx)
        ux2 = interp(px, shock_ps_right, urx)
        ax2 = sound_speed(px, rx2, gr, self)
        ps_right, rs_right, us_right = px, rx2, ux2
        Vregs.append(shock_speed(px, rx2, pr, rr, ur, self))
      
      Vregs = array(Vregs)
      ps_left = append(append(pl, ps_left), px)
      rs_left = append(append(rl, rs_left), rx1)
      us_left = append(append(ul, us_left), ux1)
      ps_right = append(append(px,  ps_right), pr)
      rs_right = append(append(rx2, rs_right), rr)
      us_right = append(append(ux2, us_right), ur)

      ex1, ex2 = sie(px, rx1, gl, self), sie(px, rx2, gr, self)
      es_left  = sie(ps_left,  rs_left,  gl, self)
      es_right = sie(ps_right, rs_right, gr, self)
      left_arrays  = ps_left,  rs_left,  us_left,  es_left
      right_arrays = ps_right, rs_right, us_right, es_right
      
      # !!! Now consider the spatial solutions at a specific time.
      Xregs = xd0 + t * Vregs
      xmin, xmax = min(xmin, 1.1 * min(Xregs)), max(xmax, 1.1 * max(Xregs))
      x = linspace(xmin, xmax, self.num_x_pts)
      x = append(x, Xregs)
      x.sort()
      
      def reg_state_geos(xl, xr, xe, regvals, exactvals):
        p,  r,  u,  e  = regvals
        pe, re, ue, ee = exactvals
        pe = where(xl < xe, interp(xe, xr, p), pe)
        re = where(xl < xe, interp(xe, xr, r), re)
        ue = where(xl < xe, interp(xe, xr, u), ue)
        ee = where(xl < xe, interp(xe, xr, e), ee)
        return pe, re, ue, ee
      
      # Define region1: the constant left state
      vals = pl+0.*x, rl+0.*x, ul+0.*x, el+0.*x
      
      soln_type = soln_type[0] + soln_type[1] + soln_type[2]
      if (soln_type == 'RCS'):
          # Define region2: rarefaction fan adjacent the constant left state
          xr = xd0 + t * (us_left - sound_speed(ps_left, rs_left, gl, self))
          vals = reg_state_geos(Xregs[0], xr, x, left_arrays, vals)
          # Define region3: constant left star-state
          xr_argmin = argmin(abs(x - Xregs[2])) - 1
          xr = append(Xregs[1], array([x[xr_argmin], Xregs[2]]))
          regvals_send = [[px,px,px],[rx1,rx1,rx2],[ux1,ux1,ux2],[ex1,ex1,ex2]]
          vals = reg_state_geos(Xregs[1], xr, x, regvals_send, vals)
          # Define region4: constant right star-state, which is a shock jump
          # from the constant right state
          xr_argmin = argmin(abs(x - Xregs[3])) - 1
          xr = append(Xregs[2], array([x[xr_argmin], Xregs[3]]))
          regvals_send = [[px,px,pr], [rx2,rx2,rr], [ux2,ux2,ur], [ex2,ex2,er]]
          vals = reg_state_geos(Xregs[2], xr, x, regvals_send, vals)
      elif (soln_type == 'SCR'):
          # Define region3: constant left star-state, which is a shock jump
          # from the constant left state
          xl_argmin = argmin(abs(x - Xregs[0])) + 1
          xr = append(Xregs[0], array([x[xl_argmin], Xregs[1]]))
          regvals_send = [[pl,px,px], [rl,rx1,rx1], [ul,ux1,ux1], [el,ex1,ex1]]
          vals = reg_state_geos(Xregs[0], xr, x, regvals_send, vals)
          # Define region4: constant right star-state
          xr = append(Xregs[1], array(Xregs[2]))
          regvals_send = [[px,px], [rx2,rx2], [ux2,ux2], [ex2,ex2]] 
          vals = reg_state_geos(Xregs[1], xr, x, regvals_send, vals)
          # Define region5: rarefaction fan adjacent the constant right state
          xr = xd0 + t * (us_right + sound_speed(ps_right, rs_right, gr, self))
          vals = reg_state_geos(Xregs[2], xr, x, right_arrays, vals)
      elif (soln_type == 'RCR'):
          # Define region2: the rarefaction fan adjacent the constant left state
          xr = xd0 + t * (us_left - sound_speed(ps_left, rs_left, gl, self))
          vals = reg_state_geos(Xregs[0], xr, x, left_arrays, vals)
          # Define region3: the constant left star-state
          xr = append(Xregs[1], array(Xregs[2]))
          regvals_send = [[px, px], [rx1, rx1], [ux1, ux1], [ex1, ex1]] 
          vals = reg_state_geos(Xregs[1], xr, x, regvals_send, vals)
          # Define region4: the constant right star-state
          xr = append(Xregs[2], array(Xregs[3]))
          regvals_send = [[px, px], [rx2, rx2], [ux2, ux2], [ex2, ex2]] 
          vals = reg_state_geos(Xregs[2], xr, x, regvals_send, vals)
          # Define region5: rarefaction fan adjacent the constant right state
          xr = xd0 + t * (us_right + sound_speed(ps_right, rs_right, gr, self))
          vals = reg_state_geos(Xregs[3], xr, x, right_arrays, vals)
      elif (soln_type == 'SCS'):
          # Define region3: the constant left star-state, which is a shock jump
          # from the constant left state
          xl_argmin = argmin(abs(x - Xregs[0])) + 1
          xr = append(Xregs[0], array([x[xl_argmin], Xregs[1]]))
          regvals_send = [[pl,px,px], [rl,rx1,rx1], [ul,ux1,ux1], [el,ex1,ex1]] 
          vals = reg_state_geos(Xregs[0], xr, x, regvals_send, vals)
          # Define region4: the constant right star-state
          xr_argmin = argmin(abs(x - Xregs[2])) - 1
          xr = append(Xregs[1], array([x[xr_argmin], Xregs[2]]))
          regvals_send = [[px,px,pr], [rx2,rx2,rr], [ux2,ux2,ur], [ex2,ex2,er]] 
          vals = reg_state_geos(Xregs[1], xr, x, regvals_send, vals)
      # Define region6: the constant right state
      xl_argmin = argmin(abs(x - Xregs[-1])) - 1
      xr = append(x[xl_argmin], array([Xregs[-1], xmax]))
      regvals_send = [[px, pr, pr], [rx2, rr, rr], [ux2, ur, ur], [ex2, er, er]]
      vals = reg_state_geos(x[xl_argmin], xr, x, regvals_send, vals)
      self.p, self.r, self.u, self.e = vals

      self.al, self.el, self.ar, self.er = al, el, ar, er
      self.integ_ps_left, self.integ_ps_right = integ_ps_left, integ_ps_right
      self.shock_ps_left, self.shock_ps_right = shock_ps_left, shock_ps_right
      self.rls, self.uls, self.rrs, self.urs = rls, uls, rrs, urs
      self.rlx, self.ulx, self.rrx, self.urx = rlx, ulx, rrx, urx
      self.ps_left_splice, self.ps_right_splice = ps_left_splice,ps_right_splice
      self.us_left_splice, self.us_right_splice = us_left_splice,us_right_splice
      self.rx1, self.ux1, self.ax1, self.ex1 = rx1, ux1, ax1, ex1
      self.rx2, self.ux2, self.ax2, self.ex2, self.px = rx2, ux2, ax2, ex2, px
      self.ps_left,  self.rs_left,  self.us_left  = ps_left,  rs_left,  us_left
      self.ps_right, self.rs_right, self.us_right = ps_right, rs_right, us_right
      self.es_left, self.es_right = es_left, es_right
      self.soln_type = soln_type
      self.Vregs, self.Xregs = Vregs, Xregs
      self.xmin, self.xmax, self.x = xmin, xmax, x
