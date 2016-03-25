      subroutine rmtv(r,den,tev,ener,pres,vel,nstep,
     1    aval_in, bval_in, chi0, gamma, bigamma, rf, xif_in, 
     2    xis, beta0_in, g0)

      implicit none
      integer          nstep
      double precision r(nstep),den(nstep),tev(nstep),ener(nstep)
      double precision pres(nstep),vel(nstep)
      double precision aval_in, bval_in, chi0, gamma, bigamma 
      double precision rf, xif_in, xis, beta0_in, g0
cf2py intent(hide)     :: nstep
cf2py intent(out)      :: den, tev, ener, pres, vel
cf2py integer          :: nstep
cf2py double precision :: den(nstep), tev(nstep), ener(nstep)
cf2py double precision :: pres(nstep), vel(nstep)
cf2py double precision :: r(nstep), aval_in, bval_in, chi0, gamma
cf2py double precision :: bigamma, rf, xif_in, xis, beta0_in, g0
      integer          i
      double precision ri, deni, tevi, eneri, presi, veli

      do i=1,nstep
         ri = r(i)
         call rmtv_1d(ri,
     1        aval_in,bval_in,chi0,gamma,bigamma,
     2        rf,xif_in,xis,beta0_in,g0,
     3        deni,tevi,eneri,presi,veli)
         den(i) = deni
         tev(i) = tevi
         ener(i) = eneri
         pres(i) = presi
         vel(i) = veli
         enddo
      end subroutine rmtv


      subroutine rmtv_1d(rpos,
     1                  aval_in,bval_in,chi0,gamma,bigamma,
     2                  rf,xif_in,xis,beta0_in,g0,
     3                  den,tev,ener,pres,vel)
      implicit none
      save

c..solves the rmtv in one-dimension, spherical coordinates
c..this a highly simplified version of kamm's code that solves for 
c..a given point value for a specific tri-lab verification test problem.

c..input: 
c..rpos    = desired radial position for the solution in cm
c..aval_in = power in thermal conductivity chi0 * rho**a * T**b
c..bval_in = power in thermal conductivity chi0 * rho**a * T**b
c..chi0    = coefficient in thermal conductivity chi0 * rho**a * T**b
c..gamma   = ratio of specific heats
c..bigamma = Gruneisen coefficient (gamma-1)*ener = pres/den = G*temp
c..rf      = position of the heat front in cm
c..xif     = dimensionless position of the heat front
c..xis     = dimensionless position of the shock front
c..beta0   = eigenvalue of the problem
c..g0      = heat front scaling parameter

c..output:
c..den  = density  g/cm**3
c..tev  = temperature ev
c..ener = specific internal energy erg/g
c..pres = presssure erg/cm**3
c..vel  = velocity cm/sh


c..declare the pass
      double precision rpos,
     1                 aval_in,bval_in,chi0,gamma,bigamma,
     2                 rf,xif_in,xis,beta0_in,g0,
     3                 den,tev,ener,pres,vel


c..local variables
      external         rmtvfun,fun,derivs
      integer          i,it
      real             flag
      double precision twoa,twob,u_0,u_l,u_r,u_c,atemp,btemp,rmtvfun,
     1                 ustar,zeroin,fun,ans,errest,
     2                 xistar,rstar,gstar,hstar,wstar,tstar,
     3                 zeta,time,rs,xiwant,usub2,hsub2,wsub2,tsub2,
     4                 tol,zero,abserr,relerr
      parameter        (tol = 1.0d-16,    zero = 0.0d0,
     2                  abserr = 1.0d-14, relerr = 1.0d-12)


c..for the ode integration
      integer          nvar,iwork(1:5),jwork
      parameter        (nvar = 4, jwork = 100 + 21*nvar)
      double precision ystart(nvar),ytemp(nvar),eta1,eta2,epsa,epsr,
     1                 xi_end,xi_small,work(jwork) 
      parameter        (epsr = 4.0d-10, epsa = 4.0d-10, xi_small=1.0d-4)


c..common block communication
      double precision aval,bval,amu,alpha,xif,beta0,kappa,sigma,xgeom
      common /rmtv1/   aval,bval,amu,alpha,xif,beta0,kappa,sigma,xgeom


c..transfer passed values to common
c..xgeom is for spherical coordinates

      aval  = aval_in
      bval  = bval_in
      xif   = xif_in
      beta0 = beta0_in
      xgeom = 3.0d0
     

c..initialize work arrays
      do i=1,5
       iwork(i) = 0
      end do
      do i=1,jwork
       work(i) = 0.0d0
      end do


c..frequent factors
      twoa  = 2.0d0 * aval
      twob  = 2.0d0 * bval
      alpha = (twob - twoa + 1.0d0)/(twob - (xgeom + 2.0d0)*aval +xgeom)
      amu   = 2.0d0 / (gamma - 1.0d0)
      kappa = -((twob - 1.0d0)*xgeom + 2.0d0)/(twob - twoa + 1.0d0)
      sigma = (twob - 1.0d0)/(alpha*(1.0d0 - aval))


c..equations 28, 30, 33, 29
c..for the scale factor, the phyiscal time, and the shock front position

      zeta = (((0.5d0 * beta0*bigamma**(bval+1.0d0) * g0**(1.0d0-aval) / 
     1          chi0)**(1.0d0/(twob - 1.0d0)))/alpha)**alpha
      time = (rf/zeta/xif)**(1.0d0/alpha)
      rs   = zeta * 1.0d0 * abs(time)**alpha


c..this section does a root find to obtain the initial conditions
c..bracket the initial zero-value of u
      it  = 0
      u_0 = (amu*bval*xif**(-((2.0d0*bval) - 1.0d0 )/alpha) 
     1       / beta0)**(1.0d0/bval)
      u_l = 0.0d0
      u_r = 0.5d0
      u_c = 0.5d0 * ( u_l + u_r )

 10   continue
      it    = it + 1
      atemp = rmtvfun(u_l)
      btemp = rmtvfun(u_r)
      if (atemp * btemp .lt. 0.0d0 ) go to 20
      if (it .ge. 100) stop 'cannot bracket zero in 100 tries'
      u_l = u_l + 0.1d0 * (u_c - u_l)
      u_r = u_r - 0.1d0 * (u_c - u_r)
      go to 10
 20   continue


c..root bracketed, solve for the zero-value ustar
      ustar = zeroin(u_l,u_r,rmtvfun,tol)


c..form the converged value of the integral
      call quanc8(fun, zero, ustar, abserr, relerr, 
     1            ans, errest, it, flag)


c..equation 11 for the position to start the integration from
      xistar  = xif * exp(-(beta0 * (xif**((twob - 1.0d0)/alpha))*ans))
      rstar   = xistar * zeta * time**alpha


c..equation 11, 13 for the initial values of the other functions
      gstar = 1.0d0/(1.0d0 - ustar )
      hstar = xistar**(-sigma) * gstar
      wstar = 0.5d0 * (amu - (( amu + 1.0d0)*ustar))
      tstar = ustar * (1.0d0 - ustar)


c..now integrate
c..beyond the heat front
      if (rpos .gt. rstar) then
       den  = g0 * rpos**kappa
       vel  = 0.0d0
       ener = 0.0d0
       pres = 0.0d0
       tev  = 0.0d0

c..integrate from the heat front to perhaps the shock front
      else 
       ystart(1) = ustar
       ystart(2) = hstar
       ystart(3) = wstar
       ystart(4) = tstar

       xiwant = rpos/zeta/time**alpha
       xi_end = max(xis,xiwant)
       eta1   = log(xistar)
       eta2   = log(xi_end)
       it     = 1
       call ode(derivs,nvar,ystart,eta1,eta2,epsr,epsa,it,work,iwork)


c..apply equation 15 of kamm 2000 for the post-shock values if we must integrate farther
       if (rpos .le. rs) then
        usub2 = ystart(1)
        hsub2 = ystart(2)
        wsub2 = ystart(3)
        tsub2 = ystart(4)

        ystart(1) = 1.0d0 - (tsub2/(1.0d0 - usub2))
        ystart(2) = (1.0d0 - usub2 )**2 / tsub2 * hsub2
        ystart(3) = (tsub2*wsub2 - 0.5d0*((1.0d0-usub2)**4 - tsub2**2)
     &                / (1.0d0 - usub2)) / (1.0d0 - usub2)**2
        ystart(4) = tsub2

c..and integrate to near the origin if nned be
        eta1   = eta2
        xi_end = max(xi_small,xiwant)
        eta2   = log(xi_end)
        it     = 1
        call ode(derivs,nvar,ystart,eta1,eta2,epsr,epsa,it,work,iwork )
       end if

c..convert the integration variables to physical quantities
c..equations 5, 2 of kamm 2000

       vel  = alpha * rpos * ystart(1) / time
       den  = g0 * rpos**kappa * xi_end**sigma * ystart(2)
       ener = (alpha*rpos/time)**2 * ystart(4) / (gamma - 1.0d0)
       pres = (gamma - 1.0d0) * den * ener
       tev  = (alpha*rpos/time)**2 * ystart(4) / bigamma


c..convert from jerk = 1e16 erg,  kev = 1e3 ev,  sh = 10e-8 s to cgs units
       vel  = vel  * 1.0d8
       ener = ener * 1.0d16
       pres = pres * 1.0d16
       tev  = tev  * 1.0d3
      end if

      return
      end







      double precision function rmtvfun(u)
      implicit none
      save

c..evaluates the expression for the initial integral for a root find

c..declare the pass
      double precision u

c..common block communication
      double precision aval,bval,amu,alpha,xif,beta0,kappa,sigma,xgeom
      common /rmtv1/   aval,bval,amu,alpha,xif,beta0,kappa,sigma,xgeom

c..local variables
      external         fun
      integer          numfun
      real             flag
      double precision fun,zero,abserr,relerr,ans,errest,smallval
      parameter        (zero = 0.0d0,
     1                  abserr = 1.0d-14,
     2                  relerr = 1.0d-12,
     3                  smallval = 1.0d-12)

      call quanc8 (fun, zero, u, abserr, relerr, 
     1             ans, errest, numfun, flag)

      rmtvfun = log(1.0d0 - smallval) + (beta0 * 
     1          (xif**(((2.0d0*bval) - 1.0d0)/alpha))*ans)

      return
      end






      double precision function fun(y)
      implicit none
      save

c.. evaluates the integrand of the initial integral

c..declare the pass
      double precision  y

c..common block communication
      double precision aval,bval,amu,alpha,xif,beta0,kappa,sigma,xgeom
      common /rmtv1/   aval,bval,amu,alpha,xif,beta0,kappa,sigma,xgeom

c..equation 12 of kamm 2000

      fun = ((1.0d0 - (2.0d0 * y)) / (amu - ((amu + 1.0d0)*y))) 
     1       * (y**(bval - 1.0d0)) 
     2       * ((1.0d0 - y)**(bval - aval))

      return
      end





      subroutine derivs ( t, y, yp )
      implicit none
      save

c.. evaluates the rhs of the system of odes 

c..declare the pass
      double precision  t,y(1:4), yp(1:4)

c..common block communication
      double precision aval,bval,amu,alpha,xif,beta0,kappa,sigma,xgeom
      common /rmtv1/   aval,bval,amu,alpha,xif,beta0,kappa,sigma,xgeom


c..local variables
      double precision g1,g2,g3,g4,temp,denom,y1m1,alphainv,omega,
     1                 eps16,eps12
      parameter        (eps16 = 1.0d-16, eps12 = 1.0d-12)


c..some factors
      y1m1  = y(1) - 1.0d0

      if (alpha .eq. 0) stop 'alpha = 0 in routine derivs'
      alphainv = 1.0d0 / alpha

      if (abs(y(2)) .le. eps16  .or. abs(y(4)) .le. eps16) then
        write(6,*) 'derivs:  y(2) or y(4) <  eps16'
        omega  = 1.0d0/eps12 * sign(1.0d0, y(3))* sign(1.0d0, beta0) 
     &           * sign(1.0d0,y(2)) * sign(1.0d0, y(4))
      else
        if (abs(aval-1.0d0) .le. eps16) stop 'aval=1 in routine derivs'
        omega  = y(3) * y(2)**(1.0d0 - aval) * y(4)**(-bval) / beta0
      end if


c..rhs of original (coupled) ode system
c..equation 26

      g1 = sigma -  (xgeom + kappa + sigma)*y(1) 
      g2 = y(1)*(alphainv - y(1)) + y(4)*(2.0d0*omega-kappa-sigma)
      g3 = omega*(amu*y1m1 + 2.0d0*y(3)) + amu*(alphainv - 1.0d0) 
     &       - xgeom*y(1) - (xgeom+kappa+sigma)*y(3)
      g4 = -2.0d0 * (1.0d0 + omega )


c..rhs of uncoupled ode system
c..equations 24, 25

      denom = y(4) - y1m1**2
      if (abs(denom) .le. eps16) stop 'denom=0 in routine derivs'
      temp  = g2 - (y1m1 * g1)
      yp(1) = g1 - ( y1m1 * temp ) / denom
      yp(2) = y(2) * temp / denom
      yp(3) = g3 - ( yp(1) + y(3) * yp(2) / y(2) )
      yp(4) = y(4) * g4

      return
      end








      subroutine quanc8 (fun,a,b,abserr,relerr,result,errest,nofun,flag)
      implicit double precision (a-h,o-z)

c-----------------------------------------------------------------------
c                                                                      c
c this subroutine estimates the integral of fun(x) from a to b         c
c to the user provided tolerance, using an automatic adaptive          c
c routine based on the 8-panel newton-cotes rule.                      c
c                                                                      c
c reference:  computer methods for mathematical computations,          c
c   forsythe, malcolm & moler, pp. 97-105.                             c
c                                                                      c
c input data:                                                          c
c                                                                      c
c     fun       name of the integrand function fun(x)                  c
c     a         lower limit of integration                             c
c     b         upper limit of integration                             c
c     relerr    relative error tolerance (>= 0)                        c
c     abserr    absolute error tolerance (>= 0)                        c
c                                                                      c
c  output data:                                                        c
c                                                                      c
c     result    approximation to the integral satisfying the least     c
c               stringent of the two error tolerances.                 c
c     errest    an estimate of the magnitude of the actual error       c
c     nofun     the number of function values used in the calculation  c
c     flag      reliability indicator:                                 c
c               = 0     => error tolerance satisfied                   c
c               xxx.yyy => xxx = number of intervals not converged     c
c                          yyy = fraction of the interval left to do   c
c                                when nofun was approached.            c
c                                                                      c
c-----------------------------------------------------------------------
c start of subroutine quanc8

c.... include files


      integer nofun

      real  flag
      double precision  fun
      double precision  a
      double precision  b
      double precision  abserr
      double precision  relerr
      double precision  result
      double precision  errest

c.... local variables

      integer levmin
      integer levmax
      integer levout
      integer nomax
      integer nofin
      integer lev
      integer nim
      integer i
      integer j

      double precision  w0
      double precision  w1
      double precision  w2
      double precision  w3
      double precision  w4
      double precision  area
      double precision  x0
      double precision  f0
      double precision  stone
      double precision  step
      double precision  cor11
      double precision  temp
      double precision  qprev
      double precision  qnow
      double precision  qdiff
      double precision  qleft
      double precision  esterr
      double precision  tolerr
      double precision  qright(31)
      double precision  f(16)
      double precision  x(16)
      double precision  fsave(8,30)
      double precision  xsave(8,30)

      external fun

c----------------------------------------------------------------------

c.... general initialization.  set constants.

      levmin = 1
      levmax = 30
      levout = 6
      nomax  = 5000
      nofin  = nomax - 8 * ( levmax - levout + 2**( levout + 1 ) )

c.... trouble when nofun reaches nofin

      w0 =   3956.d0 / 14175.d0
      w1 =  23552.d0 / 14175.d0
      w2 =  -3712.d0 / 14175.d0
      w3 =  41984.d0 / 14175.d0
      w4 = -18160.d0 / 14175.d0

c.... initialize the running sums to zero

      flag   = 0.0
      result = 0.d0
      cor11  = 0.d0
      errest = 0.d0
      area   = 0.d0
      nofun  = 0
      if ( a .eq. b ) return

c.... initialization for the first interval

      lev   = 0
      nim   = 1
      x0    = a
      x(16) = b
      qprev = 0.d0
      f0    = fun(x0)
      stone = 0.0625q0 * (b-a)
      x(8)  = 0.5d0 * ( x0    + x(16) )
      x(4)  = 0.5d0 * ( x0    + x(8)  )
      x(12) = 0.5d0 * ( x(8)  + x(16) )
      x(2)  = 0.5d0 * ( x0    + x(4)  )
      x(6)  = 0.5d0 * ( x(4)  + x(8)  )
      x(10) = 0.5d0 * ( x(8)  + x(12) )
      x(14) = 0.5d0 * ( x(12) + x(16) )
c
      do 25 j = 2, 16, 2
        f(j) = fun(x(j))
25    continue
      nofun=9

c=======================================================================
c.... central calculations
c=======================================================================

c.... requires:   qprev,x0,x2,x4,...,x16,f0,f2,f4,...,f16
c.... calculates: x1,x3,...,x15,f1,f3,...,f15,qleft,qright,qnow,qdiff,area

30    x(1) = 0.5d0 * ( x0 + x(2) )
      f(1) = fun(x(1))
      do 35 j = 3, 15, 2
        x(j) = 0.5d0 * ( x(j-1) + x(j+1) )
        f(j) = fun( x(j) )
35    continue
      nofun = nofun + 8
      step  = 0.0625d0 * ( x(16) - x0 )
      qleft = (w0*(f0  +f(8)) + w1*(f(1)+f(7)) + w2*(f(2)+f(6)) 
     1         + w3*(f(3)+f(5)) + w4*f(4)) * step
      qright(lev+1) = (w0*(f(8) +f(16)) + w1*(f(9) +f(15)) 
     1               + w2*(f(10)+f(14)) + w3*(f(11)+f(13)) 
     2               + w4*f(12)) * step
      qnow  = qleft + qright(lev+1)
      qdiff = qnow  - qprev
      area  = area  + qdiff

c.... interval convergence test

      esterr = abs( qdiff ) / 1023.d0
      tolerr = max( abserr, relerr*abs(area) ) * ( step / stone )
      if ( lev .lt. levmin )    go to 50
      if ( lev .ge. levmax )    go to 62
      if ( nofun .gt. nofin )   go to 60
      if ( esterr .le. tolerr ) go to 70

c.... no convergence => locate next interval

50    nim = 2 * nim
      lev = lev + 1

c.... store right hand elements for future use

      do 52 i = 1, 8
        fsave(i,lev) = f(i+8)
        xsave(i,lev) = x(i+8)
52    continue

c.... assemble left hand elements for immediate use

      qprev = qleft
      do 55 i=1,8
        j = -i
        f(2*j+18) = f(j+9)
        x(2*j+18) = x(j+9)
55    continue
      go to 30

c.... trouble section:  # of function values is about to exceed limit

60    nofin  = 2 * nofin
      levmax = levout
      flag   = flag + ( ( sngl(b) - sngl(x0) ) / ( sngl(b) - sngl(a) ) )
      go to 70

c.... current level is levmax

62    flag = flag + 1.0

c.... interval converged:  add contributions into running sums

70    result = result + qnow
      errest = errest + esterr
      cor11  = cor11  + ( qdiff / 1023.d0 )

c.... locate next interval

72    if ( nim .eq. 2*(nim/2) ) go to 75
      nim = nim / 2
      lev = lev - 1
      go to 72
75    nim = nim + 1
      if ( lev .le. 0 ) go to 80

c.... assemble elements required for the next interval

      qprev = qright(lev)
      x0 = x(16)
      f0 = f(16)
      do 78 i = 1, 8
        f(2*i) = fsave(i,lev)
        x(2*i) = xsave(i,lev)
78    continue
      go to 30

c.... finalize and return

80    result = result + cor11

c.... make sure errest not less than roundoff level

      if ( errest .eq. 0.d0 ) return
82    temp = abs(result) + errest
      if ( temp .ne. abs(result) ) return
      errest = 2.d0 * errest
      go to 82

      end


      double precision function zeroin( ax, bx, f, tol)
      implicit double precision (a-h,o-z)

c-----------------------------------------------------------------------
c
c This subroutine solves for a zero of the function  f(x)  in the
c interval ax,bx.
c
c  input..
c
c  ax     left endpoint of initial interval
c  bx     right endpoint of initial interval
c  f      function subprogram which evaluates f(x) for any x in
c         the interval  ax,bx
c  tol    desired length of the interval of uncertainty of the
c         final result ( .ge. 0.0d0)
c
c
c  output..
c
c  zeroin abcissa approximating a zero of  f  in the interval ax,bx
c
c
c      it is assumed  that   f(ax)   and   f(bx)   have  opposite  signs
c  without  a  check.  zeroin  returns a zero  x  in the given interval
c  ax,bx  to within a tolerance  4*macheps*abs(x) + tol, where macheps
c  is the relative machine precision.
c      this function subprogram is a slightly  modified  translation  of
c  the algol 60 procedure  zero  given in  richard brent, algorithms for
c  minimization without derivatives, prentice - hall, inc. (1973).
c
c-----------------------------------------------------------------------

c.... call list variables

      double precision  ax
      double precision  bx
      double precision  f
      double precision  tol
c
      double precision  a
      double precision  b
      double precision  c
      double precision  d
      double precision  e
      double precision  eps
      double precision  fa
      double precision  fb
      double precision  fc
      double precision  tol1
      double precision  xm
      double precision  p
      double precision  q
      double precision  r
      double precision  s

      external f

c----------------------------------------------------------------------

c
c  compute eps, the relative machine precision
c
      eps = 1.0d0
   10 eps = eps/2.0d0
      tol1 = 1.0d0 + eps
      if (tol1 .gt. 1.0d0) go to 10
c
c initialization
c
      a = ax
      b = bx
      fa = f(a)
      fb = f(b)
c
c begin step
c
   20 c = a
      fc = fa
      d = b - a
      e = d
   30 if ( abs(fc) .ge.  abs(fb)) go to 40
      a = b
      b = c
      c = a
      fa = fb
      fb = fc
      fc = fa
c
c convergence test
c
   40 tol1 = 2.0d0*eps*abs(b) + 0.5d0*tol
      xm = 0.5d0*(c - b)
      if (abs(xm) .le. tol1) go to 90
      if (fb .eq. 0.0d0) go to 90
c
c is bisection necessary?
c
      if (abs(e) .lt. tol1) go to 70
      if (abs(fa) .le. abs(fb)) go to 70
c
c is quadratic interpolation possible?
c
      if (a .ne. c) go to 50
c
c linear interpolation
c
      s = fb/fa
      p = 2.0d0*xm*s
      q = 1.0d0 - s
      go to 60
c
c inverse quadratic interpolation
c
   50 q = fa/fc
      r = fb/fc
      s = fb/fa
      p = s*(2.0d0*xm*q*(q - r) - (b - a)*(r - 1.0d0))
      q = (q - 1.0d0)*(r - 1.0d0)*(s - 1.0d0)
c
c adjust signs
c
   60 if (p .gt. 0.0d0) q = -q
      p = abs(p)
c
c is interpolation acceptable?
c
      if ((2.0d0*p) .ge. (3.0d0*xm*q - abs(tol1*q))) go to 70
      if (p .ge. abs(0.5d0*e*q)) go to 70
      e = d
      d = p/q
      go to 80
c
c bisection
c
   70 d = xm
      e = d
c
c complete step
c
   80 a = b
      fa = fb
      if (abs(d) .gt. tol1) b = b + d
      if (abs(d) .le. tol1) b = b + Sign(tol1, xm)
      fb = f(b)
      if ((fb*(fc/abs(fc))) .gt. 0.0d0) go to 20
      go to 30
c
c done
c
   90 zeroin = b

      return
      end


czzz

      subroutine ode(f,neqn,y,t,tout,relerr,abserr,iflag,work,iwork)
      implicit double precision(a-h,o-z)
c
c   double precision subroutine ode integrates a system of neqn
c   first order ordinary differential equations of the form:
c             dy(i)/dt = f(t,y(1),y(2),...,y(neqn))
c             y(i) given at  t .
c   the subroutine integrates from  t  to  tout .  on return the
c   parameters in the call list are set for continuing the integration.
c   the user has only to define a new value  tout  and call  ode  again.
c
c   the differential equations are actually solved by a suite of codes
c   de ,  step , and  intrp .  ode  allocates virtual storage in the
c   arrays  work  and  iwork  and calls  de .  de  is a supervisor which
c   directs the solution.  it calls on the routines  step  and  intrp
c   to advance the integration and to interpolate at output points.
c   step  uses a modified divided difference form of the adams pece
c   formulas and local extrapolation.  it adjusts the order and step
c   size to control the local error per unit step in a generalized
c   sense.  normally each call to  step  advances the solution one step
c   in the direction of  tout .  for reasons of efficiency  de
c   integrates beyond  tout  internally, though never beyond
c   t+10*(tout-t), and calls  intrp  to interpolate the solution at
c   tout .  an option is provided to stop the integration at  tout  but
c   it should be used only if it is impossible to continue the
c   integration beyond  tout .
c
c   this code is completely explained and documented in the text,
c   computer solution of ordinary differential equations:  the initial
c   value problem  by l. f. shampine and m. k. gordon.
c
c   the parameters represent:
c      f -- double precision subroutine f(t,y,yp) to evaluate
c                derivatives yp(i)=dy(i)/dt
c      neqn -- number of equations to be integrated (integer*4)
c      y(*) -- solution vector at t                 (double precision)
c      t -- independent variable                    (double precision)
c      tout -- point at which solution is desired   (double precision)
c      relerr,abserr -- relative and absolute error tolerances for local
c           error test (double precision).  at each step the code requires
c             dabs(local error) .le. dabs(y)*relerr + abserr
c           for each component of the local error and solution vectors
c      iflag -- indicates status of integration     (integer*4)
c      work(*)  (double precision)  -- arrays to hold information internal to
c      iwork(*) (integer*4)    which is necessary for subsequent calls
c
c   first call to ode --
c
c   the user must provide storage in his calling program for the arrays
c   in the call list,
c      y(neqn), work(100+21*neqn), iwork(5),
c   declare  f  in an external statement, supply the double precision
c   subroutine f(t,y,yp)  to evaluate
c      dy(i)/dt = yp(i) = f(t,y(1),y(2),...,y(neqn))
c   and initialize the parameters:
c      neqn -- number of equations to be integrated
c      y(*) -- vector of initial conditions
c      t -- starting point of integration
c      tout -- point at which solution is desired
c      relerr,abserr -- relative and absolute local error tolerances
c      iflag -- +1,-1.  indicator to initialize the code.  normal input
c           is +1.  the user should set iflag=-1 only if it is
c           impossible to continue the integration beyond  tout .
c   all parameters except  f ,  neqn  and  tout  may be altered by the
c   code on output so must be variables in the calling program.
c
c   output from  ode  --
c
c      neqn -- unchanged
c      y(*) -- solution at  t
c      t -- last point reached in integration.  normal return has
c           t = tout .
c      tout -- unchanged
c      relerr,abserr -- normal return has tolerances unchanged.  iflag=3
c           signals tolerances increased
c      iflag = 2 -- normal return.  integration reached  tout
c            = 3 -- integration did not reach  tout  because error
c                   tolerances too small.  relerr ,  abserr  increased
c                   appropriately for continuing
c            = 4 -- integration did not reach  tout  because more than
c                   500 steps needed
c            = 5 -- integration did not reach  tout  because equations
c                   appear to be stiff
c            = 6 -- invalid input parameters (fatal error)
c           the value of  iflag  is returned negative when the input
c           value is negative and the integration does not reach  tout ,
c           i.e., -3, -4, -5.
c      work(*),iwork(*) -- information generally of no interest to the
c           user but necessary for subsequent calls.
c
c   subsequent calls to  ode --
c
c   subroutine  ode  returns with all information needed to continue
c   the integration.  if the integration reached  tout , the user need
c   only define a new  tout  and call again.  if the integration did not
c   reach  tout  and the user wants to continue, he just calls again.
c   the output value of  iflag  is the appropriate input value for
c   subsequent calls.  the only situation in which it should be altered
c   is to stop the integration internally at the new  tout , i.e.,
c   change output  iflag=2  to input  iflag=-2 .  error tolerances may
c   be changed by the user before continuing.  all other parameters must
c   remain unchanged.
c
c-----------------------------------------------------------------------
c*  subroutines  de  and  step  contain machine dependent constants. *
c*  be sure they are set before using  ode .                          *
c-----------------------------------------------------------------------
c
      logical start,phase1,nornd
      dimension y(neqn),work(1),iwork(5)
      external f
      data ialpha,ibeta,isig,iv,iw,ig,iphase,ipsi,ix,ih,ihold,istart, 
     1      itold,idelsn/1,13,25,38,50,62,75,76,88,89,90,91,92,93/
      iyy = 100
      iwt = iyy + neqn
      ip = iwt + neqn
      iyp = ip + neqn
      iypout = iyp + neqn
      iphi = iypout + neqn
      if(iabs(iflag) .eq. 1) go to 1
      start = work(istart) .gt. 0.0d0
      phase1 = work(iphase) .gt. 0.0d0
      nornd = iwork(2) .ne. -1
    1 call de(f,neqn,y,t,tout,relerr,abserr,iflag,work(iyy), 
     &   work(iwt),work(ip),work(iyp),work(iypout),work(iphi), 
     &   work(ialpha),work(ibeta),work(isig),work(iv),work(iw),work(ig), 
     &   phase1,work(ipsi),work(ix),work(ih),work(ihold),start, 
     &   work(itold),work(idelsn),iwork(1),nornd,iwork(3),iwork(4), 
     &   iwork(5))
      work(istart) = -1.0d0
      if(start) work(istart) = 1.0d0
      work(iphase) = -1.0d0
      if(phase1) work(iphase) = 1.0d0
      iwork(2) = -1
      if(nornd) iwork(2) = 1
      return
      end




      subroutine de(f,neqn,y,t,tout,relerr,abserr,iflag, 
     &   yy,wt,p,yp,ypout,phi,alpha,beta,sig,v,w,g,phase1,psi,x,h,hold, 
     &   start,told,delsgn,ns,nornd,k,kold,isnold)
      implicit double precision(a-h,o-z)
c
c   ode  merely allocates storage for  de  to relieve the user of the
c   inconvenience of a long call list.  consequently  de  is used as
c   described in the comments for  ode .
c
c   this code is completely explained and documented in the text,
c   computer solution of ordinary differential equations:  the initial
c   value problem  by l. f. shampine and m. k. gordon.
c
      logical stiff,crash,start,phase1,nornd
      dimension y(neqn),yy(neqn),wt(neqn),phi(neqn,16),p(neqn),yp(neqn), 
     &  ypout(neqn),psi(12),alpha(12),beta(12),sig(13),v(12),w(12),g(13)
      external f
c
c-----------------------------------------------------------------------
c*  the only machine dependent constant is based on the machine unit   *
c*  roundoff error  u  which is the smallest positive number such that *
c*  1.0+u .gt. 1.0 .  u  must be calculated and  fouru=4.0*u  inserted *
c*  in the following data statement before using  de .  the routine    *
c*  machin  calculates  u .  fouru  and  twou=2.0*u  must also be      *
c*  inserted in subroutine  step  before calling  de .                 *
c
c.... Sun DOUBLE PRECISION
      data fouru/8.88d-16/
c
c.... Sun QUADRUPLE PRECISION
c      data fouru/7.70q-34/
c-----------------------------------------------------------------------
c
c   the constant  maxnum  is the maximum number of steps allowed in one
c   call to  de .  the user may change this limit by altering the
c   following statement
      data maxnum/10000/
c
c            ***            ***            ***
c   test for improper parameters
c
cccc      fouru = 4.0 * d1mach(4)                                           ***
c
      if(neqn .lt. 1) go to 10
      if(t .eq. tout) go to 10
      if(relerr .lt. 0.0d0  .or.  abserr .lt. 0.0d0) go to 10
      eps = dmax1(relerr,abserr)
      if(eps .le. 0.0d0) go to 10
      if(iflag .eq. 0) go to 10
      isn = isign(1,iflag)
      iflag = iabs(iflag)
      if(iflag .eq. 1) go to 20
      if(t .ne. told) go to 10
      if(iflag .ge. 2  .and.  iflag .le. 5) go to 20
   10 iflag = 6
      return
c
c   on each call set interval of integration and counter for number of
c   steps.  adjust input error tolerances to define weight vector for
c   subroutine  step
c
   20 del = tout - t
      absdel = dabs(del)
      tend = t + 10.0d0*del
      if(isn .lt. 0) tend = tout
      nostep = 0
      kle4 = 0
      stiff = .false.
      releps = relerr/eps
      abseps = abserr/eps
      if(iflag .eq. 1) go to 30
      if(isnold .lt. 0) go to 30
      if(delsgn*del .gt. 0.0d0) go to 50
c
c   on start and restart also set work variables x and yy(*), store the
c   direction of integration and initialize the step size
c
   30 start = .true.
      x = t
      do 40 l = 1,neqn
   40   yy(l) = y(l)
      delsgn = dsign(1.0d0,del)
      h = dsign(dmax1(dabs(tout-x),fouru*dabs(x)),tout-x)
c
c   if already past output point, interpolate and return
c
   50 if(dabs(x-t) .lt. absdel) go to 60
      call intrp(x,yy,tout,y,ypout,neqn,kold,phi,psi)
      iflag = 2
      t = tout
      told = t
      isnold = isn
      return
c
c   if cannot go past output point and sufficiently close,
c   extrapolate and return
c
   60 if(isn .gt. 0  .or.  dabs(tout-x) .ge. fouru*dabs(x)) go to 80
      h = tout - x
      call f(x,yy,yp)
      do 70 l = 1,neqn
   70   y(l) = yy(l) + h*yp(l)
      iflag = 2
      t = tout
      told = t
      isnold = isn
      return
c
c   test for too many steps
c
   80 if(nostep .lt. maxnum) go to 100
      iflag = isn*4
      if(stiff) iflag = isn*5
      do 90 l = 1,neqn
   90   y(l) = yy(l)
      t = x
      told = t
      isnold = 1
      return
c
c   limit step size, set weight vector and take a step
c
  100 h = dsign(dmin1(dabs(h),dabs(tend-x)),h)
      do 110 l = 1,neqn
  110   wt(l) = releps*dabs(yy(l)) + abseps
      call step(x,yy,f,neqn,h,eps,wt,start, 
     &   hold,k,kold,crash,phi,p,yp,psi, 
     &   alpha,beta,sig,v,w,g,phase1,ns,nornd)
c
c   test for tolerances too small
c
      if(.not.crash) go to 130
      iflag = isn*3
      relerr = eps*releps
      abserr = eps*abseps
      do 120 l = 1,neqn
  120   y(l) = yy(l)
      t = x
      told = t
      isnold = 1
      return
c
c   augment counter on number of steps and test for stiffness
c
  130 nostep = nostep + 1
      kle4 = kle4 + 1
      if(kold .gt. 4) kle4 = 0
      if(kle4 .ge. 50) stiff = .true.
      go to 50
      end




      subroutine step(x,y,f,neqn,h,eps,wt,start, 
     &   hold,k,kold,crash,phi,p,yp,psi, 
     &   alpha,beta,sig,v,w,g,phase1,ns,nornd)
      implicit double precision(a-h,o-z)
c
c   double precision subroutine  step
c   integrates a system of first order ordinary
c   differential equations one step, normally from x to x+h, using a
c   modified divided difference form of the adams pece formulas.  local
c   extrapolation is used to improve absolute stability and accuracy.
c   the code adjusts its order and step size to control the local error
c   per unit step in a generalized sense.  special devices are included
c   to control roundoff error and to detect when the user is requesting
c   too much accuracy.
c
c   this code is completely explained and documented in the text,
c   computer solution of ordinary differential equations:  the initial
c   value problem  by l. f. shampine and m. k. gordon.
c
c
c   the parameters represent:
c      x -- independent variable             (double precision)
c      y(*) -- solution vector at x          (double precision)
c      yp(*) -- derivative of solution vector at  x  after successful
c           step                             (double precision)
c      neqn -- number of equations to be integrated (integer*4)
c      h -- appropriate step size for next step.  normally determined by
c           code                             (double precision)
c      eps -- local error tolerance.  must be variable  (double precision)
c      wt(*) -- vector of weights for error criterion   (double precision)
c      start -- logical variable set .true. for first step,  .false.
c           otherwise                        (logical*4)
c      hold -- step size used for last successful step  (double precision)
c      k -- appropriate order for next step (determined by code)
c      kold -- order used for last successful step
c      crash -- logical variable set .true. when no step can be taken,
c           .false. otherwise.
c   the arrays  phi, psi  are required for the interpolation subroutine
c   intrp.  the array p is internal to the code.  all are double precision
c
c   input to  step
c
c      first call --
c
c   the user must provide storage in his driver program for all arrays
c   in the call list, namely
c
c     dimension y(neqn),wt(neqn),phi(neqn,16),p(neqn),yp(neqn),psi(12)
c
c   the user must also declare  start  and  crash  logical variables
c   and  f  an external subroutine, supply the subroutine  f(x,y,yp)
c   to evaluate
c      dy(i)/dx = yp(i) = f(x,y(1),y(2),...,y(neqn))
c   and initialize only the following parameters:
c      x -- initial value of the independent variable
c      y(*) -- vector of initial values of dependent variables
c      neqn -- number of equations to be integrated
c      h -- nominal step size indicating direction of integration
c           and maximum size of step.  must be variable
c      eps -- local error tolerance per step.  must be variable
c      wt(*) -- vector of non-zero weights for error criterion
c      start -- .true.
c
c   step  requires the l2 norm of the vector with components
c   local error(l)/wt(l)  be less than  eps  for a successful step.  the
c   array  wt  allows the user to specify an error test appropriate
c   for his problem.  for example,
c      wt(l) = 1.0  specifies absolute error,
c            = dabs(y(l))  error relative to the most recent value of
c                 the l-th component of the solution,
c            = dabs(yp(l))  error relative to the most recent value of
c                 the l-th component of the derivative,
c            = dmax1(wt(l),dabs(y(l)))  error relative to the largest
c                 magnitude of l-th component obtained so far,
c            = dabs(y(l))*relerr/eps + abserr/eps  specifies a mixed
c                 relative-absolute test where  relerr  is relative
c                 error,  abserr  is absolute error and  eps =
c                 dmax1(relerr,abserr) .
c
c      subsequent calls --
c
c   subroutine  step  is designed so that all information needed to
c   continue the integration, including the step size  h  and the order
c   k , is returned with each step.  with the exception of the step
c   size, the error tolerance, and the weights, none of the parameters
c   should be altered.  the array  wt  must be updated after each step
c   to maintain relative error tests like those above.  normally the
c   integration is continued just beyond the desired endpoint and the
c   solution interpolated there with subroutine  intrp .  if it is
c   impossible to integrate beyond the endpoint, the step size may be
c   reduced to hit the endpoint since the code will not take a step
c   larger than the  h  input.  changing the direction of integration,
c   i.e., the sign of  h , requires the user set  start = .true. before
c   calling  step  again.  this is the only situation in which  start
c   should be altered.
c
c   output from  step
c
c      successful step --
c
c   the subroutine returns after each successful step with  start  and
c   crash  set .false. .  x  represents the independent variable
c   advanced one step of length  hold  from its value on input and  y
c   the solution vector at the new value of  x .  all other parameters
c   represent information corresponding to the new  x  needed to
c   continue the integration.
c
c      unsuccessful step --
c
c   when the error tolerance is too small for the machine precision,
c   the subroutine returns without taking a step and  crash = .true. .
c   an appropriate step size and error tolerance for continuing are
c   estimated and all other information is restored as upon input
c   before returning.  to continue with the larger tolerance, the user
c   just calls the code again.  a restart is neither required nor
c   desirable.
c
      logical start,crash,phase1,nornd
      dimension y(neqn),wt(neqn),phi(neqn,16),p(neqn),yp(neqn),psi(12)
      dimension alpha(12),beta(12),sig(13),w(12),v(12),g(13), 
     &   gstr(13),two(13)
      external f
c-----------------------------------------------------------------------
c*  the only machine dependent constants are based on the machine unit *
c*  roundoff error  u  which is the smallest positive number such that *
c*  1.0+u .gt. 1.0  .  the user must calculate  u  and insert          *
c*  twou=2.0*u  and  fouru=4.0*u  in the data statement before calling *
c*  the code.  the routine  machin  calculates  u .                    *
c     data twou,fouru/.444d-15,.888d-15/                                ***
c
c.... Sun DOUBLE PRECISION
      data twou/4.44d-16/
      data fouru/8.88d-16/
c
c.... Sun QUADRUPLE PRECISION
c      data twou/3.85q-34/
c      data fouru/7.70q-34/
c-----------------------------------------------------------------------
      data two/2.0d0,4.0d0,8.0d0,16.0d0,32.0d0,64.0d0,128.0d0,256.0d0, 
     &  512.0d0,1024.0d0,2048.0d0,4096.0d0,8192.0d0/
      data gstr/0.500d0,0.0833d0,0.0417d0,0.0264d0,0.0188d0,0.0143d0, 
     &   0.0114d0,0.00936d0,0.00789d0,0.00679d0,0.00592d0,0.00524d0, 
     &   0.00468d0/
c     data g(1),g(2)/1.0,0.5/,sig(1)/1.0/
c
c
cccc      twou = 2.0 *d1mach(4)                                            ***
cccc      fouru = 2.0 * twou                                                ***
c       ***     begin block 0     ***
c   check if step size or error tolerance is too small for machine
c   precision.  if first step, initialize phi array and estimate a
c   starting step size.
c                   ***
c
c   if step size is too small, determine an acceptable one
c
      crash = .true.
      if(dabs(h) .ge. fouru*dabs(x)) go to 5
      h = dsign(fouru*dabs(x),h)
      return
    5 p5eps = 0.5d0*eps
c
c   if error tolerance is too small, increase it to an acceptable value
c
      round = 0.0d0
      do 10 l = 1,neqn
   10   round = round + (y(l)/wt(l))**2
      round = twou*dsqrt(round)
      if(p5eps .ge. round) go to 15
      eps = 2.0*round*(1.0d0 + fouru)
      return
   15 crash = .false.
      g(1)=1.0d0
      g(2)=0.5d0
      sig(1)=1.0d0
      if(.not.start) go to 99
c
c   initialize.  compute appropriate step size for first step
c
      call f(x,y,yp)
      sum = 0.0d0
      do 20 l = 1,neqn
        phi(l,1) = yp(l)
        phi(l,2) = 0.0d0
   20   sum = sum + (yp(l)/wt(l))**2
      sum = dsqrt(sum)
      absh = dabs(h)
      if(eps .lt. 16.0d0*sum*h*h) absh = 0.25d0*dsqrt(eps/sum)
      h = dsign(dmax1(absh,fouru*dabs(x)),h)
      hold = 0.0d0
      k = 1
      kold = 0
      start = .false.
      phase1 = .true.
      nornd = .true.
      if(p5eps .gt. 100.0d0*round) go to 99
      nornd = .false.
      do 25 l = 1,neqn
   25   phi(l,15) = 0.0d0
   99 ifail = 0
c       ***     end block 0     ***
c
c       ***     begin block 1     ***
c   compute coefficients of formulas for this step.  avoid computing
c   those quantities not changed when step size is not changed.
c                   ***
c
  100 kp1 = k+1
      kp2 = k+2
      km1 = k-1
      km2 = k-2
c
c   ns is the number of steps taken with size h, including the current
c   one.  when k.lt.ns, no coefficients change
c
      if(h .ne. hold) ns = 0
      if(ns.le.kold)   ns=ns+1
      nsp1 = ns+1
      if (k .lt. ns) go to 199
c
c   compute those components of alpha(*),beta(*),psi(*),sig(*) which
c   are changed
c
      beta(ns) = 1.0d0
      realns = ns
      alpha(ns) = 1.0d0/realns
      temp1 = h*realns
      sig(nsp1) = 1.0d0
      if(k .lt. nsp1) go to 110
      do 105 i = nsp1,k
        im1 = i-1
        temp2 = psi(im1)
        psi(im1) = temp1
        beta(i) = beta(im1)*psi(im1)/temp2
        temp1 = temp2 + h
        alpha(i) = h/temp1
        reali = i
  105   sig(i+1) = reali*alpha(i)*sig(i)
  110 psi(k) = temp1
c
c   compute coefficients g(*)
c
c   initialize v(*) and set w(*).  g(2) is set in data statement
c
      if(ns .gt. 1) go to 120
      do 115 iq = 1,k
        temp3 = iq*(iq+1)
        v(iq) = 1.0d0/temp3
  115   w(iq) = v(iq)
      go to 140
c
c   if order was raised, update diagonal part of v(*)
c
  120 if(k .le. kold) go to 130
      temp4 = k*kp1
      v(k) = 1.0d0/temp4
      nsm2 = ns-2
      if(nsm2 .lt. 1) go to 130
      do 125 j = 1,nsm2
        i = k-j
  125   v(i) = v(i) - alpha(j+1)*v(i+1)
c
c   update v(*) and set w(*)
c
  130 limit1 = kp1 - ns
      temp5 = alpha(ns)
      do 135 iq = 1,limit1
        v(iq) = v(iq) - temp5*v(iq+1)
  135   w(iq) = v(iq)
      g(nsp1) = w(1)
c
c   compute the g(*) in the work vector w(*)
c
  140 nsp2 = ns + 2
      if(kp1 .lt. nsp2) go to 199
      do 150 i = nsp2,kp1
        limit2 = kp2 - i
        temp6 = alpha(i-1)
        do 145 iq = 1,limit2
  145     w(iq) = w(iq) - temp6*w(iq+1)
  150   g(i) = w(1)
  199   continue
c       ***     end block 1     ***
c
c       ***     begin block 2     ***
c   predict a solution p(*), evaluate derivatives using predicted
c   solution, estimate local error at order k and errors at orders k,
c   k-1, k-2 as if constant step size were used.
c                   ***
c
c   change phi to phi star
c
      if(k .lt. nsp1) go to 215
      do 210 i = nsp1,k
        temp1 = beta(i)
        do 205 l = 1,neqn
  205     phi(l,i) = temp1*phi(l,i)
  210   continue
c
c   predict solution and differences
c
  215 do 220 l = 1,neqn
        phi(l,kp2) = phi(l,kp1)
        phi(l,kp1) = 0.0d0
  220   p(l) = 0.0d0
      do 230 j = 1,k
        i = kp1 - j
        ip1 = i+1
        temp2 = g(i)
        do 225 l = 1,neqn
          p(l) = p(l) + temp2*phi(l,i)
  225     phi(l,i) = phi(l,i) + phi(l,ip1)
  230   continue
      if(nornd) go to 240
      do 235 l = 1,neqn
        tau = h*p(l) - phi(l,15)
        p(l) = y(l) + tau
  235   phi(l,16) = (p(l) - y(l)) - tau
      go to 250
  240 do 245 l = 1,neqn
  245   p(l) = y(l) + h*p(l)
  250 xold = x
      x = x + h
      absh = dabs(h)
      call f(x,p,yp)
c
c   estimate errors at orders k,k-1,k-2
c
      erkm2 = 0.0d0
      erkm1 = 0.0d0
      erk = 0.0d0
      do 265 l = 1,neqn
        temp3 = 1.0d0/wt(l)
        temp4 = yp(l) - phi(l,1)
        if(km2)265,260,255
  255   erkm2 = erkm2 + ((phi(l,km1)+temp4)*temp3)**2
  260   erkm1 = erkm1 + ((phi(l,k)+temp4)*temp3)**2
  265   erk = erk + (temp4*temp3)**2
      if(km2)280,275,270
  270 erkm2 = absh*sig(km1)*gstr(km2)*dsqrt(erkm2)
  275 erkm1 = absh*sig(k)*gstr(km1)*dsqrt(erkm1)
  280 temp5 = absh*dsqrt(erk)
      err = temp5*(g(k)-g(kp1))
      erk = temp5*sig(kp1)*gstr(k)
      knew = k
c
c   test if order should be lowered
c
      if(km2)299,290,285
  285 if(dmax1(erkm1,erkm2) .le. erk) knew = km1
      go to 299
  290 if(erkm1 .le. 0.5d0*erk) knew = km1
c
c   test if step successful
c
  299 if(err .le. eps) go to 400
c       ***     end block 2     ***
c
c       ***     begin block 3     ***
c   the step is unsuccessful.  restore  x, phi(*,*), psi(*) .
c   if third consecutive failure, set order to one.  if step fails more
c   than three times, consider an optimal step size.  double error
c   tolerance and return if estimated step size is too small for machine
c   precision.
c                   ***
c
c   restore x, phi(*,*) and psi(*)
c
      phase1 = .false.
      x = xold
      do 310 i = 1,k
        temp1 = 1.0d0/beta(i)
        ip1 = i+1
        do 305 l = 1,neqn
  305     phi(l,i) = temp1*(phi(l,i) - phi(l,ip1))
  310   continue
      if(k .lt. 2) go to 320
      do 315 i = 2,k
  315   psi(i-1) = psi(i) - h
c
c   on third failure, set order to one.  thereafter, use optimal step
c   size
c
  320 ifail = ifail + 1
      temp2 = 0.5d0
      if(ifail - 3) 335,330,325
  325 if(p5eps .lt. 0.25d0*erk) temp2 = dsqrt(p5eps/erk)
  330 knew = 1
  335 h = temp2*h
      k = knew
      if(dabs(h) .ge. fouru*dabs(x)) go to 340
      crash = .true.
      h = dsign(fouru*dabs(x),h)
      eps = eps + eps
      return
  340 go to 100
c       ***     end block 3     ***
c
c       ***     begin block 4     ***
c   the step is successful.  correct the predicted solution, evaluate
c   the derivatives using the corrected solution and update the
c   differences.  determine best order and step size for next step.
c                   ***
  400 kold = k
      hold = h
c
c   correct and evaluate
c
      temp1 = h*g(kp1)
      if(nornd) go to 410
      do 405 l = 1,neqn
        rho = temp1*(yp(l) - phi(l,1)) - phi(l,16)
        y(l) = p(l) + rho
  405   phi(l,15) = (y(l) - p(l)) - rho
      go to 420
  410 do 415 l = 1,neqn
  415   y(l) = p(l) + temp1*(yp(l) - phi(l,1))
  420 call f(x,y,yp)
c
c   update differences for next step
c
      do 425 l = 1,neqn
        phi(l,kp1) = yp(l) - phi(l,1)
  425   phi(l,kp2) = phi(l,kp1) - phi(l,kp2)
      do 435 i = 1,k
        do 430 l = 1,neqn
  430     phi(l,i) = phi(l,i) + phi(l,kp1)
  435   continue
c
c   estimate error at order k+1 unless:
c     in first phase when always raise order,
c     already decided to lower order,
c     step size not constant so estimate unreliable
c
      erkp1 = 0.0d0
      if(knew .eq. km1  .or.  k .eq. 12) phase1 = .false.
      if(phase1) go to 450
      if(knew .eq. km1) go to 455
      if(kp1 .gt. ns) go to 460
      do 440 l = 1,neqn
  440   erkp1 = erkp1 + (phi(l,kp2)/wt(l))**2
      erkp1 = absh*gstr(kp1)*dsqrt(erkp1)
c
c   using estimated error at order k+1, determine appropriate order
c   for next step
c
      if(k .gt. 1) go to 445
      if(erkp1 .ge. 0.5d0*erk) go to 460
      go to 450
  445 if(erkm1 .le. dmin1(erk,erkp1)) go to 455
      if(erkp1 .ge. erk  .or.  k .eq. 12) go to 460
c
c   here erkp1 .lt. erk .lt. dmax1(erkm1,erkm2) else order would have
c   been lowered in block 2.  thus order is to be raised
c
c   raise order
c
  450 k = kp1
      erk = erkp1
      go to 460
c
c   lower order
c
  455 k = km1
      erk = erkm1
c
c   with new order determine appropriate step size for next step
c
  460 hnew = h + h
      if(phase1) go to 465
      if(p5eps .ge. erk*two(k+1)) go to 465
      hnew = h
      if(p5eps .ge. erk) go to 465
      temp2 = k+1
      r = (p5eps/erk)**(1.0d0/temp2)
      hnew = absh*dmax1(0.5d0,dmin1(0.9d0,r))
      hnew = dsign(dmax1(hnew,fouru*dabs(x)),h)
  465 h = hnew
      return
c       ***     end block 4     ***
      end




      subroutine intrp(x,y,xout,yout,ypout,neqn,kold,phi,psi)
      implicit double precision(a-h,o-z)
c
c   the methods in subroutine  step  approximate the solution near  x
c   by a polynomial.  subroutine  intrp  approximates the solution at
c   xout  by evaluating the polynomial there.  information defining this
c   polynomial is passed from  step  so  intrp  cannot be used alone.
c
c   this code is completely explained and documented in the text,
c   computer solution of ordinary differential equations:  the initial
c   value problem  by l. f. shampine and m. k. gordon.
c
c   input to intrp --
c
c   all floating point variables are double precision
c   the user provides storage in the calling program for the arrays in
c   the call list
       dimension y(neqn),yout(neqn),ypout(neqn),phi(neqn,16),psi(12)
c   and defines
c      xout -- point at which solution is desired.
c   the remaining parameters are defined in  step  and passed to  intrp
c   from that subroutine
c
c   output from  intrp --
c
c      yout(*) -- solution at  xout
c      ypout(*) -- derivative of solution at  xout
c   the remaining parameters are returned unaltered from their input
c   values.  integration with  step  may be continued.
c
      dimension g(13),w(13),rho(13)
      data g(1)/1.0d0/,rho(1)/1.0d0/
c
      hi = xout - x
      ki = kold + 1
      kip1 = ki + 1
c
c   initialize w(*) for computing g(*)
c
      do 5 i = 1,ki
        temp1 = i
    5   w(i) = 1.0d0/temp1
      term = 0.0d0
c
c   compute g(*)
c
      do 15 j = 2,ki
        jm1 = j - 1
        psijm1 = psi(jm1)
        gamma = (hi + term)/psijm1
        eta = hi/psijm1
        limit1 = kip1 - j
        do 10 i = 1,limit1
   10     w(i) = gamma*w(i) - eta*w(i+1)
        g(j) = w(1)
        rho(j) = gamma*rho(jm1)
   15   term = psijm1
c
c   interpolate
c
      do 20 l = 1,neqn
        ypout(l) = 0.0d0
   20   yout(l) = 0.0d0
      do 30 j = 1,ki
        i = kip1 - j
        temp2 = g(i)
        temp3 = rho(i)
        do 25 l = 1,neqn
          yout(l) = yout(l) + temp2*phi(l,i)
   25     ypout(l) = ypout(l) + temp3*phi(l,i)
   30   continue
      do 35 l = 1,neqn
   35   yout(l) = y(l) + hi*yout(l)
      return
      end


czzz




      double precision function value(string)
      implicit none
      save

c..this routine takes a character string and converts it to a real number. 
c..on error during the conversion, a fortran stop is issued

c..declare
      logical          pflag
      character*(*)    string
      character*1      plus,minus,decmal,blank,se,sd,se1,sd1
      integer          noblnk,long,ipoint,power,psign,iten,j,z,i
      double precision x,sign,factor,rten,temp
      parameter        (plus = '+'  , minus = '-' , decmal = '.'   ,
     1                  blank = ' ' , se = 'e'    , sd = 'd'       ,
     2                  se1 = 'E'   , sd1 = 'D'   , rten =  10.0,
     3                  iten = 10                                   )

c..initialize
      x      =  0.0d0
      sign   =  1.0d0
      factor =  rten
      pflag  =  .false.
      noblnk =  0
      power  =  0
      psign  =  1
      long   =  len(string)


c..remove any leading blanks and get the sign of the number
      do z = 1,7
       noblnk = noblnk + 1
       if ( string(noblnk:noblnk) .eq. blank) then
        if (noblnk .gt. 6 ) goto  30
       else
        if (string(noblnk:noblnk) .eq. plus) then
         noblnk = noblnk + 1
        else if (string(noblnk:noblnk) .eq. minus) then
         noblnk = noblnk + 1
         sign =  -1.0d0
        end if
        goto 10
       end if
      enddo


c..main number conversion loop
 10   continue
      do i = noblnk,long
       ipoint = i + 1


c..if a blank character then we are done
       if ( string(i:i) .eq. blank ) then
        x     = x * sign
        value = x 
        return


c..if an exponent character, process the whole exponent, and return
       else if (string(i:i).eq.se  .or. string(i:i).eq.sd .or.
     1          string(i:i).eq.se1 .or. string(i:i).eq.sd1   ) then
        if (x .eq. 0.0 .and. ipoint.eq.2)     x = 1.0d0
        if (sign .eq. -1.0 .and. ipoint.eq.3) x = 1.0d0
        if (string(ipoint:ipoint) .eq. plus) ipoint = ipoint + 1
        if (string(ipoint:ipoint) .eq. minus) then
         ipoint = ipoint + 1
         psign = -1
        end if
        do z = ipoint,long
         if (string(z:z) .eq. blank)  then
          x = sign * x * rten**(power*psign)
          value = x
          return
         else
          j = ichar(string(z:z)) - 48
          if ( (j.lt.0) .or. (j.gt.9) ) goto 30
          power= (power * iten)  + j
         end if
        enddo


c..if an ascii number character, process ie
       else if (string(i:i) .ne. decmal) then
        j = ichar(string(i:i)) - 48
        if ( (j.lt.0) .or. (j.gt.9) ) goto 30
        if (.not.(pflag) ) then
         x = (x*rten) + j
        else
         temp   = j
         x      = x + (temp/factor)
         factor = factor * rten
         goto 20
        end if

c..must be a decimal point if none of the above
c..check that there are not two decimal points
       else
        if (pflag) goto 30
        pflag = .true.
       end if
 20   continue
      end do

c..if we got through the do loop ok, then we must be done
      x     = x * sign
      value = x 
      return
      

c..error processing the number
 30   write(6,40) long,string(1:long)
 40   format(' error converting the ',i4,' characters ',/,
     1       ' >',a,'< ',/,
     2       ' into a real number in function value')
      stop ' error in routine value'
      end



