!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                       !
!	This function calculates numerical values of the similarity     !
!	exponent "alpha" for the converging shock problem in            !
!	cylindrical or spherical geometry using the "ode" and           !
!	"zeroin_a" routines.						!
!									!
!	The values of "lambda" can be found in Tables 6.4 (cyl) and	!
!	6.5 (sph) of:					              	!
!									!
!	Lazarus, R.B., "Self-Similar Solutions for Converging Shocks	!	
!	and Collapsing Cavities," SIAM J. Numer Anal. 18, pp. 316-371	!
!	(1981).								!
!									!
!	While the output of this function is the "standard lambda" 	!
!	appearing in the report above, the function itself is actually	!
!	solves for alpha = 1/lambda through the terminology found in:	!
!									!
!	Chisnell, R.F., "An Analytic Description of Converging Shock	!
!	Waves," J. Fluid Mech. 354, pp. 357-375 (1998). 		!
!									!
!	Chisnell's formulation of the problem is more lucid and useful	!
!	for the evaluation of the similarity exponent than the          !
!	exposition appearing in the Lazarus paper, though the ultimate  !
!	output is identical to a large number of significant digits.	!
!									!
!	Calls: zeroin_a		Called by: none (at present)		!
!									!
!	2007.07.17	S. Ramsey	initial development --          !
!	                                seems to work except for gamma  !
!				        < 1.01 ... I can't seem to get  !
!                                        more than 2 of Lazarus'  	!
!					significant figures.            !
!                                                                       !
!       2007.07.23      S. Ramsey       Cleaned up code	         	!
!									!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Start of Function "exp"
!
	real*8 function eexp(nnn, gamm)
!
	implicit none
!	
	real*8 lambda, tol, gamma, zeroin_a, d1mach, Cdiff, gamm
	real*8 alpha, amin, amax, a0num, a0dem, a0, g, gam
	integer en, n, warn, iflag, nn, ierr, nnn
	logical lcyl, lsph
	external Cdiff
	common /param1/ g, n, warn, iflag
!
!------------------------------------------------------------------------
!
!.... Here we read in (or set) the space index (n) and specific heat 
!	ratio (g) from the namelist file.
!
	n = nnn
	g = gamm
!
	if (n .eq. 2) then
	   warn = 0
	else if (n .eq. 3) then
	   warn = 0
	else
	   warn = 2
	   go to 5
	endif
!
!.... Data does not exist for gamma > 9999 or gamma < 1.00001 (in these
!       cases a polytropic gas probably doesn't make sense anyway), so
!       if we try to find the similarity exponent for one of these
!       cases, we punt.
!
	if (g .lt. 1.00001d0) then
	   warn = 5
	   go to 5
	else if (g .gt. 9999.d0) then
	   warn = 5
	   go to 5
	else
	   warn = 0
	endif
!       
	tol = d1mach(4)
!
!.... Next we set the range in which we expect alpha to lie for a given
!	geometry and specific heat ratio. See the README file for a more
!	detailed discussion of the origin of these approximations.
!
	a0num = -2.d0-g-sqrt(2.d0)*g*sqrt(g/(g-1.d0))
	a0dem = -2.d0-sqrt(2.d0)*g*sqrt(g/(g-1.d0))-g*n
	a0 = a0num/a0dem
!
	if (g .gt. 3.732050808d0) then
	   amin = a0
	else
	   amin = (4.d0+2.d0*sqrt(2.d0)*sqrt((g**3)*(-1.d0+n)**2)
     &	   +g*(-6.d0+(2.d0+g)*n))/(4.d0+g*(-8.d0+n*(4.d0+
     &	   g*n)))+.000001d0
	endif
!
	amax = 1.05d0*a0
!
	if (amax .ge. 1.d0) then
	   warn = 1
	   go to 5
	endif
!
!.... The "exact" value of alpha is found through the "zeroin_a" routine.
!	We are attempting to find the value of alpha that zeros the 
!	"Cdiff" function defined below.
!
	alpha = zeroin_a(amin, amax, Cdiff, tol, n, g)
!
!.... Various errors that can occur in the program through the flag 'warn.'
!       0 = normal return
!       1 = maximum alpha is greater than unity. 
!	2 = geometry input is invalid (i.e. not 2 or 3)
!	3 = analytic expression for V0 involves a negative determinant
!	4 = numerical integration through call to "ode" fails in some manner
!       5 = specific heat ratio is less than 1.00001 or greater than 9999
!           analytic results pretty much don't exist for gamma outside this 
!           range.
!       
 5	if (warn .eq. 0) then
	   go to 10
	else if (warn .eq. 1) then
	   write (*,*) 'EXP.F WARNING: Maximum alpha exceeds unity. Adjust
     &		"amax" premultiplier.'
	   go to 15
	else if (warn .eq. 2) then
	   write (*,*) 'EXP.F WARNING: Invalid geometry input.'
	   go to 15
	else if (warn .eq. 3) then
	   write (*,*) 'EXP.F WARNING: Analytic result for V0 is non-real.
     &		Increase "amin."'
	else if (warn .eq. 4) then
	   write (*,*) 'EXP.F WARNING: Numerical integration failed.'
	   write (*,*) 'ode failure mode =', iflag
	   go to 15
	else if (warn .eq. 5) then
	   write (*,*) 'EXP.F WARNING: Invalid polytropic index.'
	   go to 15
	else
	   go to 15
	endif
!       
 10	eexp = 1.d0/alpha
 15	return
!	
	end function eexp
!
! End of function "Exp"
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
!
	real*8 function Cdiff(en, gamma, alpha)
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!									!
!	In principle, the solution curve C(V) of Chisnell Eq. (3.1)	!
!	must pass through a critical point at a prescribed location	!
!	in (V, C) phase space. This location is determined through a	!
!	singular analysis of Eq. (3.1) itself.				!
!                                                                       !
!	The function Cdiff, starting from known initial conditions, 	!
!	numerically integrates Eq. (3.1) to the prescribed location in 	!
!	V and returns the corresponding value of C. This value of C is	!
!	then compared to the prescribed value of C. This difference is	!
!	ostensibly zero for a single, correct choice of alpha. It is 	!
!	made to be zero by the "zeroin" routine.			!
!									!
!	Input:								!
!									!
!	  en = a dummy variable that simulates the geometry index (n)	!
!	  gamma = a dummy variable that simulates the specific heat	!
!			ratio (gamma)					!
!	  (these variables are passed to the function from the driver	!
!	  program)                                                      !
!                                                                       !
!       Output:                                                         !
!                                                                       !
!         Cdiff = The phase space difference evaluated at the critical  !
!                 point. The smallness of this difference is a good     !
!                 measure of the correctness of alpha.                  !
!									!
!	Calls: ode		Called by: none				!
!									!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Start of Function "Cdiff"
!
	implicit none
!
	integer neqn, iflag, iwork(5), n, en, warn
	parameter (neqn = 1)
	real*8 abserr, relerr, work(100+21*neqn), y(neqn), t, tout, V0, C0
	real*8 V0dem, V0num1, V0num2, gamma, a, alpha, disc, g
	real*8 gammacrit, factor 
	external fe
	data abserr /1.d-10/, relerr /1.d-9/
	common /param/ a
	common /param1/ g, n, warn, iflag
!
!------------------------------------------------------------------------
!
!.... The "dummy variables" assume their passed values here.
!
	a = alpha
	en = n
	gamma = g
!
!.... It was determined by Lazarus that the V-coordinate of the critical
!	point through which the solution curve must pass is algebraically
!	distinct for different ranges of the specific heat ratio. The
!	"critical" value at which this distinction is realized is taken as
!	given for each geometry type.
!
	if (n .eq. 3) then
	   gammacrit = 1.8697680d0
	else if (n .eq. 2) then
	   gammacrit = 1.9092084d0
	endif
!
!.... The calculation of the critical (V0,C0) pair follows.
!       
	V0dem = 2.d0*g*(n-1.d0)
	factor = g*n-2.d0
	disc = 8.d0*(a-1.d0)*a*g*(n-1.d0)+(2.d0-g+a*factor)**2
!
	if (disc .lt. 0.d0) then
	   warn = 3
	   go to 50
	endif
!
	if (g .ge. gammacrit) then
	   V0num1 = 2.d0-2.d0*a-g+a*g*n-sqrt(disc)
	   V0 = V0num1/V0dem
	else
	   V0num2 = 2.d0-2.d0*a-g+a*g*n+sqrt(disc)
	   V0 = V0num2/V0dem
	endif
!	
	C0 = (V0-a)**2
!
!.... The coordinate (V0,C0) just calculated analytically must be 
!	matched by a numerical integration of Chisnell Eq. (3.1) starting
!	from the shock point (Vs,Cs) = (t, y(1)) below. This numerical
!	integration is carried through by the "ode" subroutine.
!       
	t = (2.d0*a)/(g+1.d0)
	y(1) = (2.d0*g*(g-1.d0)*a**2)/(g+1.d0)**2	
	tout = V0
	iflag = 1
!
	do while (t .gt. tout)
	   call ode(fe, neqn, y, t, tout, relerr, abserr, 
     &       iflag, work, iwork)
	end do
	if (iflag .ne. 2) then
	   warn = 4
	   go to 50
	endif
!
!.... The difference function between the analytically and numerically
!	obtained values of the coordinate C0 is returned by the function.
!
	Cdiff = C0 - y(1)
!
50	return
!
	end function Cdiff
!
! End of Function "Cdiff"
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
!       
	subroutine fe(t, y, yp)
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!									!
!	"fe" is the RHS of Chisnell Eq. (3.1), and is used in the       !
!	numerical integration of Eq. (3.1) through the call to "ode."	!
!									!
!	Calls: none		Called by: none				!
!									!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Start of Subroutine "fe"
!
	implicit none
!
	real*8 numer, denom, y(1), yp(1), t, delta, Q
	real*8 numer1, numer2, g, a
	integer n, warn
	common /param/ a
	common /param1/ g, n, warn
!
!------------------------------------------------------------------------
!     
!.... Establishment of the various factors appearing in Eq. (3.1)
!  
	delta = (t-a)**2-y(1)
	Q = n*t*(t-a)+(2.d0/g)*(1.d0-a)*(a-t)-t*(t-1.d0)
	numer = y(1)*(2.d0*delta*(a-t+(1.d0-a)*(1.d0/g))+(g-1.d0)*(a-t)*Q)	
	denom = delta*(n*t-2.d0*(1.d0-a)*(1.d0/g))*(a-t)+((a-t)**2.d0)*Q
!
!.... Computation of the RHS of Eq. (3.1)
!
	yp(1) = numer/denom
	return
!
	end subroutine fe
!
! End of Subroutine "fe"
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
