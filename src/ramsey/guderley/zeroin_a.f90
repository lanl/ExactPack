!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                            !
!     This function is identical to the "zeroin" function that is used to    !
!     determine a value of B (the space-time position of the reflected       !
!     shock wave) except it does not include "lambda" as a pass-through to   !
!     the "guderley_1D" program.                                             !
!                                                                            !
!     Instead, this function computes a precise value of the reciprocal of   !
!     the similarity exponent alpha and returns it to the "guderley_1D"      !
!     program.                                                               !
!                                                                            !
!     See documentation appearing in the "zeroin" function for a more        !
!     rigorous description of the "zeroin" routine.                          !
!                                                                            !
!     Input:                                                                 !
!                                                                            !
!         ax      left endpoint of initial interval                          !
!         bx      right endpoint of initial interval                         !
!         f       function which evaluates f(..., x) for any x in the        !
!                 interval [ax, bx]                                          !
!         tol     desired length of the interval of uncertainty of the       !
!                 final result (>= 0.0d0)                                    !
!         n       |                                                          !
!         gamma   |    pass-throughs to guderley_1D                          !
!                                                                            !
!     Output:                                                                !
!                                                                            !
!         zeroin  abscissa approximating a zero of f(..., x) in the          !
!                 interval [ax, bx]; in particular a value of alpha,         !
!                 the reciprocal of the similarity exponent lambda           !
!                                                                            !
!     2007.07.23   S. Ramsey       Cleaned up code                           !
!                                                                            !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Start of Function "zeroin_a"
!
      real*8 function zeroin_a(ax, bx, f, tol, n, gamma)
!
      real*8 ax, bx, f, gamma, tol
      integer n
      real*8 a, b, c, d, e, eps, fa, fb, fc, p, q, r, s, tol1, xm
      real*8 d1mach
      integer iter, maxit
      parameter (maxit = 75)
      external f
!
!-----------------------------------------------------------------------------
!
!.... Compute eps, the relative machine precision.
!
      eps = d1mach(4)
!
!.... Initialize
!
      iter = 0
      a = ax
      b = bx
      fa = f(n, gamma, a)
      fb = f(n, gamma, b)
!.... The following warning message is important. If it is triggered,
!     then it is probable that the interval over which we are searching
!     for alpha is mis-defined, and in particular the upper bound of this
!     interval. In all likelihood, adjusting the upper bound of the 
!     search interval (while keeping it less than unity) will fix the
!     problem.
!
      if (fa*sign(1.d0, fb) .gt. 0.d0) then
         write (*, 15) a, fa, b, fb
 15      format (' Zeroin_a error:  endpoints don''t have opposite ',
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
!.... Convergence test
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
!.... Linear interpolation (regula falsi).
!
         p = 2.0d0*xm*s
         q = 1.0d0 - s
      else
!
!.... Inverse quadratic interpolation.
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
!.... Bisect
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
      fb = f(n, gamma, b)
      iter = iter + 1
      if (iter .eq. maxit) then
         write (*, 85) maxit
 85      format (' Zeroin_a error:  More than', i3, ' iterations')
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
 90   zeroin_a = b
      return
      end function zeroin_a
!
! End of Function "zeroin_a"
!<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
