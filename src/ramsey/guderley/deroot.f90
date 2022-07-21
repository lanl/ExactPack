      subroutine deroot(f,neqn,y,t,tout,relerr,abserr,iflag,
     *                  g,reroot,aeroot, phi)
      external f
      integer neqn, iflag
      real*8 y(neqn), t, tout, relerr, abserr, g,
     *                 reroot, aeroot
c
c   Subroutine  deroot  integrates a system of up to 20 first order
c   ordinary differential equations of the form
c             dy(i)/dt = f(t,y(1),...,y(neqn))
c             y(i) given at t.
c   The subroutine integrates from  t  in the direction of  tout  until
c   it locates the first root of the nonlinear equation
c         g(t,y(1),...,y(neqn),yp(1),...,yp(neqn)) = 0.
c   Upon finding the root, the code returns with all parameters in the
c   call list set for continuing the integration to the next root or
c   the first root of a new function  g  .  If no root is found, the
c   integration proceeds to  tout .  Again all parameters are set to
c   continue.
c
c   The differential equations are actually solved by a suite of codes,
c   deroot , step , and  intrp .  Deroot  is a supervisor which directs
c   the integration.  It calls on  step  to advance the solution and
c   intrp  to interpolate the solution and its derivative.  Step  uses
c   a modified divided difference form of the adams pece formulas and
c   local extrapolation.  It adjusts the order and step size to control
c   the local error per unit step in a generalized sense.  Normally each
c   call to  step  advances the solution one step in the direction of
c   tout .  For reasons of efficiency  deroot  integrates beyond  tout
c   internally, though never beyond t+10*(tout-t), and calls  intrp  to
c   interpolate the solution and derivative at  tout .  An option is
c   provided to stop the integration at  tout  but it should be used
c   only if it is impossible to continue the integration beyond  tout .
c
c   After each internal step,  deroot  evaluates the function  g  and
c   checks for a change in sign in the function value from the
c   preceding step.  Such a change indicates a root lies in the
c   interval of the step just completed.  Deroot  then calls subroutine
c   root  to reduce the bracketing interval until the root is
c   determined to the desired accuracy.  Subroutine  root  uses a
c   combination of the secant rule and bisection to do this.  The
c   solution and derivative values required are obtained by
c   interpolation with  intrp .  The code locates only those roots
c   for which  g  changes sign in  (t,tout)  and for which a
c   bracketing interval exists.  In particular, it does not locate
c   a root at the initial point  t .
c
c   The codes  step  and  intrp  and that portion of  deroot  which
c   directs the integration are completely explained and documented in
c   the text, "Computer Solution of Ordinary Differential Equations,
c   the Initial Value Problem" by L. F. Shampine and M. K. Gordon.
c   Subroutine  root  is a slightly modified version of the root-solver
c   discussed in the text, "Numerical Computing, an Introduction" by
c   L. F. Shampine and R. C. Allen.
c
c   The parameters for deroot are
c      f -- subroutine f(t,y,yp) to evaluate derivatives yp(i)=dy(i)/dt
c      neqn -- number of equations to be integrated
c      y(*) -- solution vector at  t
c      t -- independent variable
c      tout -- arbitrary point beyond the root desired
c      relerr,abserr -- relative and absolute error tolerances for local
c           error test.  at each step the code requires
c             abs(local error) .le. abs(y)*relerr + abserr
c           for each component of the local error and solution vectors
c      iflag -- indicates status of integration
c      g - function of t, y(*), yp(*) whose root is desired.
c      reroot, aeroot -- relative and absolute error tolerances for
c           accepting the root.  the interval containing the root is
c           reduced until it satisfies
c            0.5*abs(length of interval) .le. reroot*abs(root)+aeroot
c           where root is that endpoint yielding the smaller value of
c           g  in magnitude.  pure relative error is not recommended
c           if the root might be zero.
c
c   First call to  deroot  --
c
c   The user must provide storage in his calling program for the
c   array in the call list,
c              y(neqn)
c   and declare  f , g  in an external statement.  He must supply the
c   subroutine  f(t,y,yp)  to evaluate
c           dy(i)/dt = yp(i) = f(t,y(1),...,y(neqn))
c   and the function  g(t,y,yp)  to evaluate
c           g = g(t,y(1),...,y(neqn),yp(1),...,yp(neqn)).
c   Note that the array yp is an input argument and should not be
c   computed in the function subprogram.  Finally the user must
c   initialize the parameters
c      neqn -- number of equations to be integrated
c      y(*) -- vector of initial conditions
c      t -- starting point of integration
c      tout -- arbitrary point beyond the root desired
c      relerr,abserr -- relative and absolute local error tolerances
c                       for integrating the equations
c      iflag -- +1,-1.  indicator to initialize the code.  normal input
c           is +1.  the user should set iflag=-1 only if it is
c           impossible to continue the integration beyond  tout .
c      reroot,aeroot -- relative and absolute error tolerances for
c                       computing the root of  g
c
c   All parameters except f, g, neqn, tout, reroot and aeroot may be
c   altered by the code on output so must be variables in the calling
c   program.
c
c   Output from  deroot  --
c
c      neqn -- unchanged
c      y(*) -- solution at  t
c      t -- last point reached in integration.  normal return has
c           t = tout or t = root
c      tout -- unchanged
c      relerr,abserr -- normal return has tolerances unchanged.  iflag=3
c           signals tolerances increased
c      iflag = 2 -- normal return.  integration reached  tout
c            = 3 -- integration did not reach  tout  because error
c                   tolerances too small.  relerr ,  abserr  increased
c                   appropriately for continuing
c            = 4 -- integration did not reach  tout  because more than
c                   maxnum  steps needed
c            = 5 -- integration did not reach  tout  because equations
c                   appear to be stiff
c            = 6 -- invalid input parameters (fatal error)
c            = 7 -- normal return.  a root was found which satisfied
c                   the error criterion or had a zero residual
c            = 8 -- abnormal return.  an odd order pole of  g  was
c                   found.
c            = 9 -- abnormal return.  too many evaluations of  g  were
c                   required (as programmed 500 are allowed.)
c           the value of  iflag  is returned negative when the input
c           value is negative and the integration does not reach
c           tout , i.e., -3,-4,-5,-7,-8,-9.
c      reroot,aeroot -- unchanged
c
c   Subsequent calls to  deroot  --
c
c   Subroutine  deroot  returns with all information needed to continue
c   the integration.  If the integration did not reach  tout  and the
c   user wants to continue, he just calls again.  If the integration
c   reached  tout , the user need only define a new  tout  and call
c   again.  The output value of  iflag  is the appropriate input value
c   for subsequent calls.  The only situation in which it should be
c   altered is to stop the integration internally at the new  tout ,
c   i.e., change output  iflag=2  to input  iflag=-2 .  Only the error
c   tolerances and the function  g  may be changed by the user before
c   continuing.  All other parameters must remain unchanged.
c
      logical crash, nornd, phase1, start, stiff
      integer isn, isnold, jflag, kle4, k, kold, l, maxnum, nostep, ns
      real*8 alpha(12), beta(12), sig(13), v(12), w(12), gg(13)
      real*8 fouru, eps, gxold, gx, x, delsgn,
     *                 b, c, gc, del, tend, releps, abseps, troot,
     *                 absdel, h, hold, told, gt
      real*8 yy(20),wt(20),phi(neqn,16),p(20),yp(20),ypout(20),
     *                 psi(12)
      real*8 d1mach
      save delsgn, gx, gxold, h, isnold, nostep, psi, told, troot, x, yy
c
c***********************************************************************
c*  The only machine dependent constant is based on the machine unit   *
c*  roundoff error  u  which is the smallest positive number such that *
c*  1.0+u .gt. 1.0 .  U  must be calculated and  fouru=4.0*u  inserted *
c*  in the following statement before using  deroot .  The subroutine  *
c*  d1mach  calculates  u .  fouru  and  twou=2.0*u  must also be      *
c*  inserted in subroutine  step  before calling  deroot .             *
c     data fouru/8.8d-16/                                               ***
c***********************************************************************
c
c   The constant  maxnum  is the maximum number of steps allowed in one
c   call to  deroot .  The user may change this limit by altering the
c   following statement
      data maxnum /500/
c
c   This version of  deroot  allows a maximum of 20 equations.  To
c   increase this number, only the number 20 in the dimension statement
c   and in the following statement need be changed
c            ***            ***            ***
c   Test for improper parameters
c
c-----------------------------------------------------------------
      fouru = 4.0*d1mach(4)                                             ***
      if (neqn .lt. 1  .or.  neqn .gt. 20) go to 10
      if (t .eq. tout) go to 10
      if (relerr .lt. 0.0d0 .or.  abserr .lt. 0.0d0) go to 10
      eps = max(relerr,abserr)
      if (eps .le. 0.0d0) go to 10
      if (reroot .lt. 0.0d0 .or. aeroot .lt. 0.0d0) go to 10
      if (reroot+aeroot .le. 0.0d0) go to 10
      if (iflag .eq. 0) go to 10
      isn = sign(1,iflag)
      iflag = abs(iflag)
      if (iflag .eq. 1) go to 20
      if (t .ne. told) go to 10
      if (iflag .ge. 2  .and.  iflag .le. 5) go to 15
      if (iflag .ge. 7  .and.  iflag .le. 9) go to 15
   10 iflag = 6
      return
c
c   For a new function g, check for a root in interval of step
c   just completed
c
   15 gxold = gx
      gx = g(x,yy,yp)
      if (gx .eq. gxold) go to 20
      if (iflag .gt. 2  .and.  iflag .le. 5) go to 20
      if (isnold .lt. 0  .or.  delsgn*(tout-t) .lt. 0.0d0) go to 20
      jflag = 1
      b = x
      c = t
      call intrp(x,yy,c,y,ypout,neqn,kold,phi,psi)
      gc = g(c,y,ypout)
      if (sign(1.0d0,gc)*sign(1.0d0,gx) .lt. 0.0d0) go to 140
      if (gc .eq. 0.0d0 .or.  gx .eq. 0.0d0) go to 140
c
c   On each call set interval of integration and counter for number of
c   steps.  Adjust input error tolerances to define weight vector for
c   subroutine  step
c
   20 del = tout - t
      absdel = abs(del)
      tend = t + 10.0d0*del
      if (isn .lt. 0) tend = tout
      nostep = 0
      kle4 = 0
      stiff = .false.
      releps = relerr/eps
      abseps = abserr/eps
      if (iflag .eq. 1) go to 30
      if (isnold .lt. 0) go to 30
      if (delsgn*del .gt. 0.0d0) go to 50
c
c   On start and restart also set work variables x and yy(*), store the
c   direction of integration, initialize the step size, and evaluate  g
c
   30 start = .true.
      x = t
      troot = t
      do 40 l = 1,neqn
   40   yy(l) = y(l)
      delsgn = sign(1.0d0, del)
      h = sign(max(abs(tout-x),fouru*abs(x)),tout-x)
      call f(x,yy,yp)
      gx = g(x,yy,yp)
c
c   If already past output point, interpolate and return
c
   50 gxold = gx
      if (abs(x-t) .lt. absdel) go to 60
      call intrp(x,yy,tout,y,ypout,neqn,kold,phi,psi)
      iflag = 2
      t = tout
      told = t
      isnold = isn
      return
c
c   If cannot go past output point and sufficiently close,
c   extrapolate and return
c
   60 if (isn .gt. 0  .or.  abs(tout-x) .ge. fouru*abs(x)) go to 80
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
c   Test for too much work
c
   80 if (nostep .lt. maxnum) go to 100
      iflag = isn*4
      if (stiff) iflag = isn*5
      do 90 l = 1,neqn
   90   y(l) = yy(l)
      t = x
      told = t
      isnold = 1
      return
c
c   Limit step size, set weight vector and take a step
c
  100 h = sign(min(abs(h), abs(tend-x)),h)
      do 110 l = 1,neqn
  110   wt(l) = releps*abs(yy(l)) + abseps
      call step(x,yy,f,neqn,h,eps,wt,start,
     *  hold,k,kold,crash,phi,p,yp,psi,
     *  alpha,beta,sig,v,w,gg,phase1,ns,nornd)
c
c   Test for tolerances too small.  If so, set the derivative at x
c   before returning.
c
      if (.not.crash) go to 130
      iflag = isn*3
      relerr = eps*releps
      abserr = eps*abseps
      do 120 l = 1,neqn
        yp(l) = phi(l,1)
  120   y(l) = yy(l)
      t = x
      told = t
      isnold = 1
      return
c
c   Augment counter on work and test for stiffness.  Also test for a
c   root in the step just completed.
c
  130 nostep = nostep + 1
      kle4 = kle4 + 1
      if (kold .gt. 4) kle4 = 0
      if (kle4 .ge. 50) stiff = .true.
      gx = g(x,yy,yp)
      if (sign(1.0d0, gxold)*sign(1.0d0, gx) .lt. 0.0d0) go to 135
      if (gx .eq. 0.0d0 .or. gxold .eq. 0.0d0) go to 135
      go to 50
c
c   Locate root of g.  Interpolate with  intrp  for solution and
c   derivative values
c
  135 b = x
      c = x - hold
      jflag = 1
  140 call root(t,gt,b,c,reroot,aeroot,jflag)
      if (jflag .gt. 0) go to 150
      call intrp(x,yy,t,y,ypout,neqn,kold,phi,psi)
      gt = g(t, y, ypout)
      go to 140

  150 continue
      iflag = jflag + 6
      if (jflag .eq. 2 .or. jflag .eq. 4) iflag = 7
      if (jflag .eq. 3) iflag = 8
      if (jflag .eq. 5) iflag = 9
      iflag = iflag*isn
      call intrp(x,yy,b,y,ypout,neqn,kold,phi,psi)
      t = b
      if (abs(t-troot) .le. reroot*abs(t) + aeroot) go to 50
      troot = t
      told = t
      isnold = 1
      return
      end


      subroutine root(t, ft, b, c, relerr, abserr, iflag)
      real*8 t, ft, b, c, relerr, abserr
      integer iflag
c
c  Root computes a root of the nonlinear equation f(x) = 0
c  where f(x) is a continuous real function of a single real
c  variable x.  The method used is a combination of bisection
c  and the secant rule.
c
c  Normal input consists of a continuous function f and an
c  interval (b, c) such that f(b)*f(c) <= 0.0.  Each iteration
c  finds new values of b and c such that the interval (b, c) is
c  shrunk and f(b)*f(c) <= 0.0.  The stopping criterion is
c
c         abs(b-c) <= 2.0*(relerr*abs(b) + abserr),
c
c  where relerr = relative error and abserr = absolute error are
c  input quantities.  Set the flag, iflag, positive to initialize
c  the computation.  As b,c and iflag are used for both input and
c  output, they must be variables in the calling program.
c
c  If 0 is a possible root, one should not choose abserr = 0.0.
c
c  The output value of b is the better approximation to a root
c  as b and c are always redefined so that abs(f(b)) <= abs(f(c)).
c
c  To solve the equation, root must evaluate f(x) repeatedly.  This
c  is done in the calling program.  When an evaluation of f is
c  needed at t, root returns with iflag negative.  Evaluate ft = f(t)
c  and call root again.  Do not alter iflag.
c
c  When the computation is complete, root returns to the calling
c  program with iflag positive:
c
c     iflag=1  if f(b)*f(c) < 0 and the stopping criterion is met.
c
c          =2  if a value b is found such that the computed value
c              f(b) is exactly zero.  The interval (b,c) may not
c              satisfy the stopping criterion.
c
c          =3  if abs(f(b)) exceeds the input values abs(f(b)),
c              abs(f(c)).  In this case it is likely that b is close
c              to a pole of f.
c
c          =4  if no odd order root was found in the interval.  A
c              local minimum may have been obtained.
c
c          =5  if too many function evaluations were made.
c              (as programmed, 500 are allowed.)
c
c  This code is a modification of the code  zeroin  which is completely
c  explained and documented in the text,  "Numerical Computing:  an
c  Introduction"  by L. F. Shampine and R. C. Allen.
c
      real*8 u, re, ae, acbs, a, fa, fb, fc, fx, cmb,
     *                 acmb, tol, p, q
      real*8 d1mach
      integer ic, kount
      save a, acbs, ae, fa, fb, fc, fx, ic, kount, re
c***********************************************************************
c*  The only machine dependent constant is based on the machine unit   *
c*  roundoff error  u  which is the smallest positive number such that *
c*  1.0+u .gt. 1.0 .  U  must be calculated and inserted in the        *
c*  following data statement before using  root .  The routine  d1mach *
c*  calculates  u .                                                    *
c     data u/2.2d-16/
c***********************************************************************
c
      u = d1mach(4)
      if (iflag .ge. 0) go to 100

      iflag = abs(iflag)
      go to (200, 300, 400), iflag

  100 re = max(relerr, u)
      ae = max(abserr, 0.0d0)
      ic = 0
      acbs = abs(b - c)
      a = c
      t = a
      iflag = -1
      return

  200 fa = ft
      t = b
      iflag = -2
      return

  300 fb = ft
      fc = fa
      kount = 2
      fx = max(abs(fb), abs(fc))

    1 if (abs(fc) .lt. abs(fb)) then
c
c  Interchange b and c so that abs(f(b)) <= abs(f(c)).
c
         a = b
         fa = fb
         b = c
         fb = fc
         c = a
         fc = fa
      endif

      cmb = 0.5*(c - b)
      acmb = abs(cmb)
      tol = re*abs(b) + ae
c
c  Test stopping criterion and function count.
c
      if (acmb .le. tol) go to 8
      if (kount .ge. 500) go to 12
c
c  Calculate new iterate implicitly as b+p/q
c  where we arrange p >= 0.  The implicit
c  form is used to prevent overflow.
c
      p = (b - a)*fb
      q = fa - fb
      if (p .lt. 0.0d0) then
         p = -p
         q = -q
      endif
c
c  Update a, check if reduction in the size of bracketing
c  interval is satisfactory.  If not, bisect until it is.
c
      a = b
      fa = fb
      ic = ic + 1
      if (ic .lt. 4) go to 4
      if (8.0*acmb .ge. acbs) go to 6
      ic = 0
      acbs = acmb
c
c  Test for too small a change.
c
    4 if (p .gt. abs(q)*tol) go to 5
c
c  Increment by tolerance.
c
      b = b + sign(tol, cmb)
      go to 7
c
c  Root ought to be between b and (c+b)/2.
c
    5 if (p .ge. cmb*q) go to 6
c
c  Use secant rule.
c
      b = b + p/q
      go to 7
c
c  Use bisection.
c
    6 b = 0.5*(c+b)
c
c  Have completed computation for new iterate b.
c
    7 t = b
      iflag = -3
      return

  400 fb = ft
      if (fb .eq. 0.0d0) go to 9
      kount = kount + 1
      if (sign(1.0d0, fb)*sign(1.0d0, fc) .lt. 0.d0) go to 1
      c = a
      fc = fa
      go to 1
c
c  Finished.  Set iflag.
c
    8 if (sign(1.0d0,fb) .eq. sign(1.0d0,fc)) go to 11
      if (abs(fb) .gt. fx) go to 10
      iflag = 1
      return

    9 iflag = 2
      return

   10 iflag = 3
      return

   11 iflag = 4
      return

   12 iflag = 5
      return
      end
