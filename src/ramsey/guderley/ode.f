      subroutine ode(f,neqn,y,t,tout,relerr,abserr,iflag,work,iwork)
      real*8 abserr, relerr, t, tout
      integer iflag, neqn
c
c   Double precision subroutine ode integrates a system of neqn
c   first order ordinary differential equations of the form:
c             dy(i)/dt = f(t,y(1),y(2),...,y(neqn))
c             y(i) given at  t .
c   The subroutine integrates from  t  to  tout .  On return the
c   parameters in the call list are set for continuing the integration.
c   The user has only to define a new value  tout  and call  ode  again.
c
c   The differential equations are actually solved by a suite of codes
c   de ,  step , and  intrp .  Ode  allocates virtual storage in the
c   arrays  work  and  iwork  and calls  de .  De  is a supervisor which
c   directs the solution.  It calls on the routines  step  and  intrp
c   to advance the integration and to interpolate at output points.
c   step  uses a modified divided difference form of the adams pece
c   formulas and local extrapolation.  It adjusts the order and step
c   size to control the local error per unit step in a generalized
c   sense.  normally each call to  step  advances the solution one step
c   in the direction of  tout .  For reasons of efficiency  de
c   integrates beyond  tout  internally, though never beyond
c   t+10*(tout-t), and calls  intrp  to interpolate the solution at
c   tout .  An option is provided to stop the integration at  tout  but
c   it should be used only if it is impossible to continue the
c   integration beyond  tout .
c
c   This code is completely explained and documented in the text,
c   Computer Solution of Ordinary Differential Equations:  The Initial
c   Value Problem  by L. F. Shampine and M. K. Gordon.
c
c   The parameters represent:
c      f -- double precision subroutine f(t,y,yp) to evaluate
c                derivatives yp(i)=dy(i)/dt
c      neqn -- number of equations to be integrated (integer*4)
c      y(*) -- solution vector at t                 (real*8)
c      t -- independent variable                    (real*8)
c      tout -- point at which solution is desired   (real*8)
c      relerr,abserr -- relative and absolute error tolerances for local
c           error test (real*8).  at each step the code requires
c             abs(local error) .le. abs(y)*relerr + abserr
c           for each component of the local error and solution vectors
c      iflag -- indicates status of integration     (integer*4)
c      work(*)  (real*8)  -- arrays to hold information internal to
c      iwork(*) (integer*4)    which is necessary for subsequent calls
c
c   first call to ode --
c
c   The user must provide storage in his calling program for the arrays
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
c      iflag -- +1,-1.  indicator to initialize the code.  Normal input
c           is +1.  The user should set iflag=-1 only if it is
c           impossible to continue the integration beyond  tout .
c   All parameters except  f ,  neqn  and  tout  may be altered by the
c   code on output so must be variables in the calling program.
c
c   output from  ode  --
c
c      neqn -- unchanged
c      y(*) -- solution at  t
c      t -- last point reached in integration.  Normal return has
c           t = tout .
c      tout -- unchanged
c      relerr,abserr -- normal return has tolerances unchanged.  Iflag=3
c           signals tolerances increased
c      iflag = 2 -- normal return.  Integration reached  tout
c            = 3 -- integration did not reach  tout  because error
c                   tolerances too small.  Relerr ,  abserr  increased
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
c   Subroutine  ode  returns with all information needed to continue
c   the integration.  If the integration reached  tout , the user need
c   only define a new  tout  and call again.  If the integration did not
c   reach  tout  and the user wants to continue, he just calls again.
c   The output value of  iflag  is the appropriate input value for
c   subsequent calls.  The only situation in which it should be altered
c   is to stop the integration internally at the new  tout , i.e.,
c   change output  iflag=2  to input  iflag=-2 .  Error tolerances may
c   be changed by the user before continuing.  All other parameters must
c   remain unchanged.
c
c***********************************************************************
c*  Subroutines  de  and  step  contain machine dependent constants. *
c*  be sure they are set before using  ode .                          *
c***********************************************************************
c
      logical start, phase1, nornd
      real*8 y(neqn), work(*)
      integer iwork(5)
      external f
      integer ialpha, ibeta, idelsn, ig, ih, ihold, ip, iphase, iphi,
     1 ipsi, isig, istart, itold, iv, iw, iwt, ix, iyp, iypout, iyy
      data ialpha,ibeta,isig,iv,iw,ig,iphase,ipsi,ix,ih,ihold,istart,
     1  itold,idelsn/1,13,25,38,50,62,75,76,88,89,90,91,92,93/
      iyy = 100
      iwt = iyy + neqn
      ip = iwt + neqn
      iyp = ip + neqn
      iypout = iyp + neqn
      iphi = iypout + neqn
      if (iabs(iflag) .eq. 1) go to 1
      start = work(istart) .gt. 0.0d0
      phase1 = work(iphase) .gt. 0.0d0
      nornd = iwork(2) .ne. -1
    1 call de(f,neqn,y,t,tout,relerr,abserr,iflag,work(iyy),
     1  work(iwt),work(ip),work(iyp),work(iypout),work(iphi),
     2  work(ialpha),work(ibeta),work(isig),work(iv),work(iw),work(ig),
     3  phase1,work(ipsi),work(ix),work(ih),work(ihold),start,
     4  work(itold),work(idelsn),iwork(1),nornd,iwork(3),iwork(4),
     5  iwork(5))
      work(istart) = -1.0d0
      if (start) work(istart) = 1.0d0
      work(iphase) = -1.0d0
      if (phase1) work(iphase) = 1.0d0
      iwork(2) = -1
      if (nornd) iwork(2) = 1
      return
      end


      subroutine de(f,neqn,y,t,tout,relerr,abserr,iflag,
     1  yy,wt,p,yp,ypout,phi,alpha,beta,sig,v,w,g,phase1,psi,x,h,hold,
     2  start,told,delsgn,ns,nornd,k,kold,isnold)
c
c   Ode  merely allocates storage for  de  to relieve the user of the
c   inconvenience of a long call list.  Consequently  de  is used as
c   described in the comments for  ode .
c
c   This code is completely explained and documented in the text,
c   Computer Solution of Ordinary Differential Equations:  The Initial
c   Value Problem  by L. F. Shampine and M. K. Gordon.
c
      logical stiff,crash,start,phase1,nornd
      integer iflag, isn, isnold, k, kle4, kold, l, maxnum, neqn,
     1 nostep, ns
      real*8 absdel, abseps, abserr, del, delsgn, d1mach, eps, fouru,
     1 h, hold, releps, relerr, t, tend, told, tout, x
      real*8 y(neqn),yy(neqn),wt(neqn),phi(neqn,16),p(neqn),yp(neqn),
     1  ypout(neqn),psi(12),alpha(12),beta(12),sig(13),v(12),w(12),g(13)
      external f
c
c***********************************************************************
c*  The only machine dependent constant is based on the machine unit   *
c*  roundoff error  u  which is the smallest positive number such that *
c*  1.0+u .gt. 1.0 .  U  must be calculated and  fouru=4.0*u  inserted *
c*  in the following data statement before using  de .  The routine    *
c*  d1mach  calculates  u .  Fouru  and  twou=2.0*u  must also be      *
c*  inserted in subroutine  step  before calling  de .                 *
c     data fouru/.888d-15/                                              ***
c***********************************************************************
c
c   The constant  maxnum  is the maximum number of steps allowed in one
c   call to  de .  The user may change this limit by altering the
c   following statement
      data maxnum /2000/
c
c            ***            ***            ***
c   test for improper parameters
c
      fouru = 4.0 * d1mach(4)                                           ***
      if (neqn .lt. 1) go to 10
      if (t .eq. tout) go to 10
      if (relerr .lt. 0.0d0  .or.  abserr .lt. 0.0d0) go to 10
      eps = max(relerr, abserr)
      if (eps .le. 0.0d0) go to 10
      if (iflag .eq. 0) go to 10
      isn = sign(1, iflag)
      iflag = iabs(iflag)
      if (iflag .eq. 1) go to 20
      if (t .ne. told) go to 10
      if (iflag .ge. 2  .and.  iflag .le. 5) go to 20
   10 iflag = 6
      return
c
c   On each call set interval of integration and counter for number of
c   steps.  Adjust input error tolerances to define weight vector for
c   subroutine  step.
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
c   direction of integration and initialize the step size.
c
   30 start = .true.
      x = t
      do 40 l = 1,neqn
   40   yy(l) = y(l)
      delsgn = sign(1.0d0, del)
      h = sign(max(abs(tout-x), fouru*abs(x)), tout-x)
c
c   If already past output point, interpolate and return.
c
   50 if (abs(x-t) .lt. absdel) go to 60
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
c   Test for too many steps
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
  100 h = sign(min(abs(h), abs(tend-x)), h)
      do 110 l = 1,neqn
  110   wt(l) = releps*abs(yy(l)) + abseps
      call step(x,yy,f,neqn,h,eps,wt,start,
     1  hold,k,kold,crash,phi,p,yp,psi,
     2  alpha,beta,sig,v,w,g,phase1,ns,nornd)
c
c   Test for tolerances too small
c
      if (.not. crash) go to 130
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
c   Augment counter on number of steps and test for stiffness
c
  130 nostep = nostep + 1
      kle4 = kle4 + 1
      if (kold .gt. 4) kle4 = 0
      if (kle4 .ge. 50) stiff = .true.
      go to 50
      end


      subroutine step(x,y,f,neqn,h,eps,wt,start,
c    1  hold,k,kold,crash,phi,p,yp,psi)
     1  hold,k,kold,crash,phi,p,yp,psi,
     2  alpha,beta,sig,v,w,g,phase1,ns,nornd)
c
c   Double precision subroutine  step
c   integrates a system of first order ordinary
c   differential equations one step, normally from x to x+h, using a
c   modified divided difference form of the adams pece formulas.  Local
c   extrapolation is used to improve absolute stability and accuracy.
c   The code adjusts its order and step size to control the local error
c   per unit step in a generalized sense.  Special devices are included
c   to control roundoff error and to detect when the user is requesting
c   too much accuracy.
c
c   This code is completely explained and documented in the text,
c   Computer Solution of Ordinary Differential Equations:  The Initial
c   Value Problem  by L. F. Shampine and M. K. Gordon.
c
c
c   The parameters represent:
c      x -- independent variable             (real*8)
c      y(*) -- solution vector at x          (real*8)
c      yp(*) -- derivative of solution vector at  x  after successful
c           step                             (real*8)
c      neqn -- number of equations to be integrated (integer*4)
c      h -- appropriate step size for next step.  Normally determined by
c           code                             (real*8)
c      eps -- local error tolerance.  Must be variable  (real*8)
c      wt(*) -- vector of weights for error criterion   (real*8)
c      start -- logical variable set .true. for first step,  .false.
c           otherwise                        (logical*4)
c      hold -- step size used for last successful step  (real*8)
c      k -- appropriate order for next step (determined by code)
c      kold -- order used for last successful step
c      crash -- logical variable set .true. when no step can be taken,
c           .false. otherwise.
c   the arrays  phi, psi  are required for the interpolation subroutine
c   intrp.  The array p is internal to the code.  All are real*8
c
c   Input to  step
c
c      First call --
c
c   The user must provide storage in his driver program for all arrays
c   in the call list, namely
c
c     real*8 y(neqn),wt(neqn),phi(neqn,16),p(neqn),yp(neqn),psi(12)
c
c   The user must also declare  start  and  crash  logical variables
c   and  f  an external subroutine, supply the subroutine  f(x,y,yp)
c   to evaluate
c      dy(i)/dx = yp(i) = f(x,y(1),y(2),...,y(neqn))
c   and initialize only the following parameters:
c      x -- initial value of the independent variable
c      y(*) -- vector of initial values of dependent variables
c      neqn -- number of equations to be integrated
c      h -- nominal step size indicating direction of integration
c           and maximum size of step.  Must be variable
c      eps -- local error tolerance per step.  must be variable
c      wt(*) -- vector of non-zero weights for error criterion
c      start -- .true.
c
c   Step  requires the l2 norm of the vector with components
c   local error(l)/wt(l)  be less than  eps  for a successful step.  The
c   array  wt  allows the user to specify an error test appropriate
c   for his problem.  For example,
c      wt(l) = 1.0  specifies absolute error,
c            = abs(y(l))  error relative to the most recent value of
c                 the l-th component of the solution,
c            = abs(yp(l))  error relative to the most recent value of
c                 the l-th component of the derivative,
c            = max(wt(l), abs(y(l)))  error relative to the largest
c                 magnitude of l-th component obtained so far,
c            = abs(y(l))*relerr/eps + abserr/eps  specifies a mixed
c                 relative-absolute test where  relerr  is relative
c                 error,  abserr  is absolute error and  eps =
c                 max(relerr, abserr) .
c
c      Subsequent calls --
c
c   Subroutine  step  is designed so that all information needed to
c   continue the integration, including the step size  h  and the order
c   k , is returned with each step.  With the exception of the step
c   size, the error tolerance, and the weights, none of the parameters
c   should be altered.  The array  wt  must be updated after each step
c   to maintain relative error tests like those above.  Normally the
c   integration is continued just beyond the desired endpoint and the
c   solution interpolated there with subroutine  intrp .  If it is
c   impossible to integrate beyond the endpoint, the step size may be
c   reduced to hit the endpoint since the code will not take a step
c   larger than the  h  input.  Changing the direction of integration,
c   i.e., the sign of  h , requires the user set  start = .true. before
c   calling  step  again.  This is the only situation in which  start
c   should be altered.
c
c   Output from  step:
c
c      successful step --
c
c   The subroutine returns after each successful step with  start  and
c   crash  set .false. .  X  represents the independent variable
c   advanced one step of length  hold  from its value on input and  y
c   the solution vector at the new value of  x .  All other parameters
c   represent information corresponding to the new  x  needed to
c   continue the integration.
c
c      unsuccessful step --
c
c   When the error tolerance is too small for the machine precision,
c   the subroutine returns without taking a step and  crash = .true. .
c   an appropriate step size and error tolerance for continuing are
c   estimated and all other information is restored as upon input
c   before returning.  To continue with the larger tolerance, the user
c   just calls the code again.  A restart is neither required nor
c   desirable.
c
      logical start, crash, phase1, nornd
      integer i, ifail, im1, ip1, iq, j, k, kold, km1, km2, knew,
     1 kp1, kp2, l, limit1, limit2, neqn, ns, nsm2, nsp1, nsp2
      real*8 y(neqn),wt(neqn),phi(neqn,16),p(neqn),yp(neqn),psi(12)
      real*8 alpha(12),beta(12),sig(13),w(12),v(12),g(13),
     1  gstr(13),two(13)
      real*8 absh, d1mach, eps, erk, erkm1, erkm2, erkp1, err, fouru,
     1 h, hnew, hold, p5eps, r, reali, realns, rho, round,
     1 sum, tau, temp1, temp2, temp3, temp4, temp5, temp6, twou,
     2 x, xold
      external f
c***********************************************************************
c*  The only machine dependent constants are based on the machine unit *
c*  roundoff error  u  which is the smallest positive number such that *
c*  1.0+u .gt. 1.0  .  The user must calculate  u  and insert          *
c*  twou=2.0*u  and  fouru=4.0*u  in the data statement before calling *
c*  the code.  The routine  d1mach  calculates  u .                    *
c     data twou,fouru/.444d-15,.888d-15/                             ***
c***********************************************************************
      data two/2.0d0,4.0d0,8.0d0,16.0d0,32.0d0,64.0d0,128.0d0,256.0d0,
     1  512.0d0,1024.0d0,2048.0d0,4096.0d0,8192.0d0/
      data gstr/0.500d0,0.0833d0,0.0417d0,0.0264d0,0.0188d0,0.0143d0,
     1  0.0114d0,0.00936d0,0.00789d0,0.00679d0,0.00592d0,0.00524d0,
     2  0.00468d0/
c     data g(1),g(2) /1.0d0, 0.5d0/, sig(1) /1.0d0/
c
c
      twou = 2.0*d1mach(4)                                          
      fouru = 2.0*twou                                              
c       ***     begin block 0     ***
c   Check if step size or error tolerance is too small for machine
c   precision.  If first step, initialize phi array and estimate a
c   starting step size.
c                   ***
c
c   If step size is too small, determine an acceptable one
c
      crash = .true.
      if (abs(h) .ge. fouru*abs(x)) go to 5
      h = sign(fouru*abs(x),h)
      return
    5 p5eps = 0.5d0*eps
c
c   If error tolerance is too small, increase it to an acceptable value
c
      round = 0.0d0
      do 10 l = 1,neqn
   10   round = round + (y(l)/wt(l))**2
      round = twou*sqrt(round)
      if (p5eps .ge. round) go to 15
      eps = 2.0*round*(1.0d0 + fouru)
      return
   15 crash = .false.
      g(1)=1.0d0
      g(2)=0.5d0
      sig(1)=1.0d0
      if (.not. start) go to 99
c
c   Initialize.  Compute appropriate step size for first step
c
      call f(x,y,yp)
      sum = 0.0d0
      do 20 l = 1,neqn
        phi(l,1) = yp(l)
        phi(l,2) = 0.0d0
   20   sum = sum + (yp(l)/wt(l))**2
      sum = sqrt(sum)
      absh = abs(h)
      if (eps .lt. 16.0d0*sum*h*h) absh = 0.25d0*sqrt(eps/sum)
      h = sign(max(absh, fouru*abs(x)), h)
      hold = 0.0d0
      k = 1
      kold = 0
      start = .false.
      phase1 = .true.
      nornd = .true.
      if (p5eps .gt. 100.0d0*round) go to 99
      nornd = .false.
      do 25 l = 1,neqn
   25   phi(l,15) = 0.0d0
   99 ifail = 0
c       ***     End block 0     ***
c
c       ***     Begin block 1     ***
c   Compute coefficients of formulas for this step.  Avoid computing
c   those quantities not changed when step size is not changed.
c                   ***
c
  100 kp1 = k+1
      kp2 = k+2
      km1 = k-1
      km2 = k-2
c
c   Ns is the number of steps taken with size h, including the current
c   one.  When k < ns, no coefficients change
c
      if (h .ne. hold) ns = 0
      if (ns.le.kold)   ns=ns+1
      nsp1 = ns+1
      if (k .lt. ns) go to 199
c
c   Compute those components of alpha(*),beta(*),psi(*),sig(*) which
c   are changed
c
      beta(ns) = 1.0d0
      realns = ns
      alpha(ns) = 1.0d0/realns
      temp1 = h*realns
      sig(nsp1) = 1.0d0
      if (k .lt. nsp1) go to 110
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
c   Compute coefficients g(*)
c
c   Initialize v(*) and set w(*).  g(2) is set in data statement
c
      if (ns .gt. 1) go to 120
      do 115 iq = 1,k
        temp3 = iq*(iq+1)
        v(iq) = 1.0d0/temp3
  115   w(iq) = v(iq)
      go to 140
c
c   If order was raised, update diagonal part of v(*)
c
  120 if (k .le. kold) go to 130
      temp4 = k*kp1
      v(k) = 1.0d0/temp4
      nsm2 = ns-2
      if (nsm2 .lt. 1) go to 130
      do 125 j = 1,nsm2
        i = k-j
  125   v(i) = v(i) - alpha(j+1)*v(i+1)
c
c   Update v(*) and set w(*)
c
  130 limit1 = kp1 - ns
      temp5 = alpha(ns)
      do 135 iq = 1,limit1
        v(iq) = v(iq) - temp5*v(iq+1)
  135   w(iq) = v(iq)
      g(nsp1) = w(1)
c
c   Compute the g(*) in the work vector w(*)
c
  140 nsp2 = ns + 2
      if (kp1 .lt. nsp2) go to 199
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
c   Predict a solution p(*), evaluate derivatives using predicted
c   solution, estimate local error at order k and errors at orders k,
c   k-1, k-2 as if constant step size were used.
c                   ***
c
c   Change phi to phi star
c
      if (k .lt. nsp1) go to 215
      do 210 i = nsp1,k
        temp1 = beta(i)
        do 205 l = 1,neqn
  205     phi(l,i) = temp1*phi(l,i)
  210   continue
c
c   Predict solution and differences
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
      if (nornd) go to 240
      do 235 l = 1,neqn
        tau = h*p(l) - phi(l,15)
        p(l) = y(l) + tau
  235   phi(l,16) = (p(l) - y(l)) - tau
      go to 250
  240 do 245 l = 1,neqn
  245   p(l) = y(l) + h*p(l)
  250 xold = x
      x = x + h
      absh = abs(h)
      call f(x,p,yp)
c
c   Estimate errors at orders k,k-1,k-2
c
      erkm2 = 0.0d0
      erkm1 = 0.0d0
      erk = 0.0d0
      do 265 l = 1,neqn
        temp3 = 1.0d0/wt(l)
        temp4 = yp(l) - phi(l,1)
        if (km2) 265, 260, 255
  255   erkm2 = erkm2 + ((phi(l,km1)+temp4)*temp3)**2
  260   erkm1 = erkm1 + ((phi(l,k)+temp4)*temp3)**2
  265   erk = erk + (temp4*temp3)**2
      if (km2) 280, 275, 270
  270 erkm2 = absh*sig(km1)*gstr(km2)*sqrt(erkm2)
  275 erkm1 = absh*sig(k)*gstr(km1)*sqrt(erkm1)
  280 temp5 = absh*sqrt(erk)
      err = temp5*(g(k)-g(kp1))
      erk = temp5*sig(kp1)*gstr(k)
      knew = k
c
c   Test if order should be lowered
c
      if (km2) 299, 290, 285
  285 if (max(erkm1, erkm2) .le. erk) knew = km1
      go to 299
  290 if (erkm1 .le. 0.5d0*erk) knew = km1
c
c   Test if step successful
c
  299 if (err .le. eps) go to 400
c       ***     end block 2     ***
c
c       ***     begin block 3     ***
c   The step is unsuccessful.  Restore  x, phi(*,*), psi(*) .
c   if third consecutive failure, set order to one.  If step fails more
c   than three times, consider an optimal step size.  double error
c   tolerance and return if estimated step size is too small for machine
c   precision.
c                   ***
c
c   Restore x, phi(*,*) and psi(*)
c
      phase1 = .false.
      x = xold
      do 310 i = 1,k
        temp1 = 1.0d0/beta(i)
        ip1 = i+1
        do 305 l = 1,neqn
  305     phi(l,i) = temp1*(phi(l,i) - phi(l,ip1))
  310   continue
      if (k .lt. 2) go to 320
      do 315 i = 2,k
  315   psi(i-1) = psi(i) - h
c
c   On third failure, set order to one.  Thereafter, use optimal step
c   size
c
  320 ifail = ifail + 1
      temp2 = 0.5d0
      if (ifail - 3) 335, 330, 325
  325 if (p5eps .lt. 0.25d0*erk) temp2 = sqrt(p5eps/erk)
  330 knew = 1
  335 h = temp2*h
      k = knew
      if (abs(h) .ge. fouru*abs(x)) go to 340
      crash = .true.
      h = sign(fouru*abs(x),h)
      eps = eps + eps
      return
  340 go to 100
c       ***     End block 3     ***
c
c       ***     Begin block 4     ***
c   The step is successful.  Correct the predicted solution, evaluate
c   the derivatives using the corrected solution and update the
c   differences.  Determine best order and step size for next step.
c                   ***
  400 kold = k
      hold = h
c
c   Correct and evaluate
c
      temp1 = h*g(kp1)
      if (nornd) go to 410
      do 405 l = 1,neqn
        rho = temp1*(yp(l) - phi(l,1)) - phi(l,16)
        y(l) = p(l) + rho
  405   phi(l,15) = (y(l) - p(l)) - rho
      go to 420
  410 do 415 l = 1,neqn
  415   y(l) = p(l) + temp1*(yp(l) - phi(l,1))
  420 call f(x,y,yp)
c
c   Update differences for next step
c
      do 425 l = 1,neqn
        phi(l,kp1) = yp(l) - phi(l,1)
  425   phi(l,kp2) = phi(l,kp1) - phi(l,kp2)
      do 435 i = 1,k
        do 430 l = 1,neqn
  430     phi(l,i) = phi(l,i) + phi(l,kp1)
  435   continue
c
c   Estimate error at order k+1 unless:
c     in first phase when always raise order,
c     already decided to lower order,
c     step size not constant so estimate unreliable
c
      erkp1 = 0.0d0
      if (knew .eq. km1  .or.  k .eq. 12) phase1 = .false.
      if (phase1) go to 450
      if (knew .eq. km1) go to 455
      if (kp1 .gt. ns) go to 460
      do 440 l = 1,neqn
  440   erkp1 = erkp1 + (phi(l,kp2)/wt(l))**2
      erkp1 = absh*gstr(kp1)*sqrt(erkp1)
c
c   Using estimated error at order k+1, determine appropriate order
c   for next step
c
      if (k .gt. 1) go to 445
      if (erkp1 .ge. 0.5d0*erk) go to 460
      go to 450
  445 if (erkm1 .le. min(erk, erkp1)) go to 455
      if (erkp1 .ge. erk  .or.  k .eq. 12) go to 460
c
c   Here erkp1 .lt. erk .lt. max(erkm1, erkm2) else order would have
c   been lowered in block 2.  Thus order is to be raised
c
c   Raise order
c
  450 k = kp1
      erk = erkp1
      go to 460
c
c   Lower order
c
  455 k = km1
      erk = erkm1
c
c   With new order determine appropriate step size for next step
c
  460 hnew = h + h
      if (phase1) go to 465
      if (p5eps .ge. erk*two(k+1)) go to 465
      hnew = h
      if (p5eps .ge. erk) go to 465
      temp2 = k+1
      r = (p5eps/erk)**(1.0d0/temp2)
      hnew = absh*max(0.5d0, min(0.9d0, r))
      hnew = sign(max(hnew, fouru*abs(x)), h)
  465 h = hnew
      return
c       ***     End block 4     ***
      end


      subroutine intrp(x,y,xout,yout,ypout,neqn,kold,phi,psi)
c
c   The methods in subroutine  step  approximate the solution near  x
c   by a polynomial.  Subroutine  intrp  approximates the solution at
c   xout  by evaluating the polynomial there.  Information defining this
c   polynomial is passed from  step  so  intrp  cannot be used alone.
c
c   This code is completely explained and documented in the text,
c   Computer Solution of Ordinary Differential Equations:  The Initial
c   Value Problem  by L. F. Shampine and M. K. Gordon.
c
c   input to intrp --
c
c   All floating point variables are double precision
c   The user provides storage in the calling program for the arrays in
c   the call list
       integer neqn
       real*8 y(neqn),yout(neqn),ypout(neqn),phi(neqn,16),psi(12)
c   and defines
c      xout -- point at which solution is desired.
c   The remaining parameters are defined in  step  and passed to  intrp
c   from that subroutine.
c
c   Output from  intrp --
c
c      yout(*) -- solution at  xout
c      ypout(*) -- derivative of solution at  xout
c   The remaining parameters are returned unaltered from their input
c   values.  Integration with  step  may be continued.
c
      integer i, j, jm1, ki, kip1, kold, l, limit1
      real*8 eta, gamma, hi, psijm1, temp1, temp2, temp3,
     & term, x, xout
      real*8 g(13), w(13), rho(13)
      data g(1)/1.0d0/, rho(1)/1.0d0/
c
      hi = xout - x
      ki = kold + 1
      kip1 = ki + 1
c
c   Initialize w(*) for computing g(*)
c
      do 5 i = 1,ki
        temp1 = i
    5   w(i) = 1.0d0/temp1
      term = 0.0d0
c
c   Compute g(*)
c
      do 15 j = 2, ki
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
c   Interpolate
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


