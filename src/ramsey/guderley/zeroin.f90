!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                        !
!     This function computes a zero of the function f(..., x) in the     !
!     interval [ax, bx]. It returns a zero x in the given interval       !
!     to within a tolerance 4*macheps*abs(x) + tol, where macheps is the !
!     relative machine precision.                                        !
!                                                                        !
!     This function is a slightly modified translation of the Algol 60   !
!     procedure "zero" given in:                                         !
!                                                                        !
!     Richard Brent, "Algorithms for Minimization without Derivatives,"  !
!     Prentice - Hall, Inc. (1973).                                      !
!                                                                        !
!     In particular, this function is for exclusive use within the       !
!     "guderley_1D" program of finding a more precise value of the       !
!     reflected shock space-time position B (as compared to the function !
!     "zeroin_a," which is used within the context of this code package  !
!     for finding a precise value of the similarity exponent lambda.     !
!                                                                        !
!     Input:                                                             !
!                                                                        !
!         ax      left endpoint of initial interval                      !
!         bx      right endpoint of initial interval                     !
!         f       function which evaluates f(..., x) for any x in the    !
!                 interval [ax, bx]                                      !
!         tol     desired length of the interval of uncertainty of the   !
!                 final result (>= 0.0d0)                                !
!         n       |                                                      !
!         gamma   |    pass-throughs to guderley_1D                      !
!         lambda  |                                                      !
!                                                                        !
!     Output:                                                            !
!                                                                        !
!         zeroin  abscissa approximating a zero of f(..., x) in the      !
!                 interval [ax, bx]                                      !
!                                                                        !
!     2007.07.23   S. Ramsey       Cleaned up code                       !
!                                                                        !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Start of Function "zeroin"
!
      real*8 function zeroin(ax, bx, f, tol, n, gamma, lambda)
!
      real*8 ax, bx, f, gamma, lambda, tol
      integer n
      real*8 a, b, c, d, e, eps, fa, fb, fc, p, q, r, s, tol1, xm
      real*8 d1mach
      integer iter, maxit
      parameter (maxit = 25)
!
!-------------------------------------------------------------------------
!
!.... Compute eps, the relative machine precision.

      eps = d1mach(4)
!
!.... Initialize the working variables.
!
      iter = 0
      a = ax
      b = bx
      fa = f(n, gamma, lambda, a)
      fb = f(n, gamma, lambda, b)
!
!.... The following error is important within the context of determining
!     B. If the "endpoints don't have opposite signs" error is returned,
!     then the range over which B is trying to be determined is either too
!     large or too small. A good place to start in correcting this error
!     if it is returned is to adjust the maximum guess for B.
!
      if (fa*sign(1.d0, fb) .gt. 0.d0) then
         write (*, 15) a, fa, b, fb
 15      format (' Zeroin error:  endpoints don''t have opposite ',
     1   'signs, a, f(a), b, f(b)' / 1p, 4e15.7)
         call exit(1)
      endif
!
!.... Begin step.
!
 20   c = a
      fc = fa
      d = b - a
      e = d
 30   if (abs(fc) .lt. abs(fb)) then
         a = b
         b = c
         c = a
         fa = fb
         fb = fc
         fc = fa
      endif
!
!.... Convergence test.
!
      tol1 = 2.0d0*eps*abs(b) + 0.5d0*tol
      xm = .5*(c - b)
      if (abs(xm) .le. tol1 .or. fb .eq. 0.d0) go to 90
!
!.... Is bisection necessary?
!
      if (abs(e) .lt. tol1 .or. abs(fa) .le. abs(fb)) go to 70
!
!.... Is quadratic interpolation possible?
!
      s = fb/fa
      if (a .eq. c) then
!
!.... Linear interpolation (regula falsi)
!
         p = 2.0d0*xm*s
         q = 1.0d0 - s
      else
!
!.... Inverse quadratic interpolation
!
         q = fa/fc
         r = fb/fc
         p = s*(2.0d0*xm*q*(q - r) - (b - a)*(r - 1.0d0))
         q = (q - 1.0d0)*(r - 1.0d0)*(s - 1.0d0)
      endif
!
!.... Adjust signs.
!
      if (p .gt. 0.0d0) q = -q
      p = abs(p)
!
!.... Is interpolation acceptable?
!
      if (2.0d0*p .lt. 3.0d0*xm*q - abs(tol1*q) .and.
     1 p .lt. abs(0.5d0*e*q)) then
         e = d
         d = p/q
         go to 80
      endif
!
!.... Bisect.
!
 70   d = xm
      e = d
!
!.... Complete step; get new point.
!
 80   a = b
      fa = fb
      if (abs(d) .gt. tol1) then
         b = b + d
      else
         b = b + sign(tol1, xm)
      endif
      fb = f(n, gamma, lambda, b)
      iter = iter + 1
!
!.... The maximum number of iterations has been set at 50. This
!     should be sufficient for determining B, but this parameter
!     can be adjusted as necessary.
!
      if (iter .eq. maxit) then
         write (*, 85) maxit
 85      format (' Zeroin error:  More than', i3, ' iterations')
         call exit(1)
      endif
      if (fb*sign(1.d0, fc) .gt. 0.0d0) then
         go to 20
      else
         go to 30
      endif
!
!.... Done!
!
   90 zeroin = b
      return
      end function zeroin
!
! End of Function "zeroin"
!<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
