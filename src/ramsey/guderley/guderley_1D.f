!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                           !
!	This program generates physical variable (i.e. density, velocity    !
!       pressure, sound speed) solution data at a specified point in        !
!       space and time for the converging shock wave problem first solved   !
!	by G. Guderley. It follows the notation given in:                   !
!       It follows the notation given in:                                   !
!                                                                           !
!       Lazarus, R.B. "Self-Similar Solutions for Converging Shocks         !
!       and Collapsing Cavities," SIAM J. NUMER. ANAL. 18.2,                !
!       pp. 316-371 (1981).                                                 !
!                                                                           !
!       "guderley_1D" makes use of several subprograms and functions        !
!       in order to calculate the two nonlinear eigenvalues that appear     !
!       in the problem: the similarity exponent "lambda" and the            !
!       reflected shock position (in similarity variables) "B." Once        !
!       these values are calculated the self-similar equations              !
!       governing the flow are solved for the dimensionless velocity        !
!       V, sound speed C, and density R, as a function of the similarity    !
!       variable:                                                           !
!                                                                           !
!                           t                                               !
!                    x = -------- .                                         !
!                          lambda                                           !
!                         r                                                 !
!                                                                           !
!       The program  also computes the result starting from x = infinity    !
!       (Lazarus, p. 330 ff.). The subroutine that transforms the           !
!       similarity variable data is self-contained in the subroutine        !
!       "state."                                                            !
!                                                                           !
!       Calls: GET_PARAMS, state    Called by: None                         !
!                                                                           !
!       This code is based on the driver code "guderley" first developed    !
!       by J. Bolstad of LLNL.                                              !
!                                                                           !
!       2007.08.01  S. Ramsey     Code reproduces correct results for       !
!                                 gamma = 1.4.                              !
!                                                                           !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	subroutine guderley_1d(t,r,nstep,ngeom,gamma,rho0,
     1      den,vel,pres,snd,sie)
	implicit none
	integer          nstep, ngeom
        double precision t
        double precision r(nstep),den(nstep),vel(nstep)
	double precision pres(nstep),snd(nstep),sie(nstep)
        double precision gamma,rho0
cf2py intent(out)      :: den, vel, pres, snd, sie        
cf2py intent(hide)     :: nstep
cf2py integer          :: nstep, ngeom
cf2py double precision :: den(nstep), ener(nstep), pres(nstep)
cf2py double precision :: snd(nstep), sie(nstep), r(nstep)
cf2py double precision :: t, gamma, rho0
	double precision tee,lambda,eexp,lambdad,targetx
        double precision Bmaxg,interp_laz,Bmin,Bming,Bmax,tol,d1mach
        double precision zeroin, B, rpos, deni, veli, presi, sndi, siei
	double precision factorC
	data factorC /.750024322d0/
        integer i
	external Guderley


!.... The input time is a Caramana/Whalen time, defined by:
!
!       t_C = 0.750024322*(t_L + 1)
!
!.... Here, the time is converted to Lazarus time.
!
	tee = (t/factorC)-1.d0
!
!.... The value of the similarity exponent "lambda" is calulated using the
!       "exp" function. See documentation appearing in "exp" for an 
!       explanation of how this value is calculated.
!
	lambda = eexp(ngeom, gamma)
	lambdad = lambda
!
!.... If a position in both space and time are specified, this data can be
!       converted into an appropriate value of the similarity variable x
!       defined above. This value of x is where we desire to know the values
!       of the similarity variables. 
!
	do i=1,nstep
           rpos = r(i)
	   targetx = tee/(rpos**lambda)
!
!.... As is the case with lambda, the reflected shock space-time position
!       "B" is not known a priori (though it is known that B lies in
!       the range (0 < B < 1)). Lazarus was the first to determine the
!       value of B to 6 significant figures (appearing in Tables 6.4 and
!       6.5). This precision can be improved upon using the "zeroin"
!       routine, as will be explained below.
!
	   Bmaxg = interp_laz(ngeom, gamma, lambda)
!
	   Bming = 0.34d0	! use this value for gamma=3.0 and rho0=1.0
	   if (Bming .le. 0.d0) then
	      write (*,*) 'guderley_1D error: Bmin < 0'
	      stop
	   else if (Bmaxg .ge. 1.d0) then
	      write (*,*) 'guderley_1D error: Bmax > 1'
	      stop
	   endif

	   Bmin = ((gamma+1.d0)/(gamma-1.d0))*Bming
!
!.... The maximum allowable value for B (Bmax) is determined by using
!       the function INTERP_LAZ. This function interpolates in Lazarus
!       Tables 6.4 and 6.5 for general polytropic index for B. This
!       interpolated value of B is used as an upper bound for a more
!       precise value of B.
!
	   Bmax = ((gamma+1.d0)/(gamma-1.d0))*Bmaxg + 0.001d0
	   tol = d1mach(4)
!
!.... Below, a more precise value of B for a given polytropic index and
!       geometry type than is given by Lazarus can be computed by using
!       the "zeroin" routine, which here finds the B-zero of a function
!       called "Guderley," which is defined below. 
!
	   B = zeroin(Bmin, Bmax, Guderley, tol, ngeom, gamma, lambda)
	   
!       
!.... The ultimate output of the program is generated through the "state"
!       subroutine, which computes the solution of the similarity variable
!       equations at the target value of x and then transforms this
!       solution back to physical variable space.
!
	   call state(rpos, rho0, ngeom, gamma, lambda, B, targetx,
     1          deni, veli, presi, sndi, siei)
           den(i)  = deni
           vel(i)  = veli
           pres(i) = presi
           snd(i)  = sndi
           sie(i)  = siei	  
	enddo

        end subroutine guderley_1d
c$$$
c$$$ Spherical Guderley Problem, gamma =    3.0000000000000000
c$$$ For r =  1.0000000000100000
c$$$ At  t = -1.0000000000000000
c$$$ den   =  2.0000000000142788
c$$$ vel   = -0.31820529705044343
c$$$ pres  =  0.20250922214684017
c$$$ snd   =  0.55114774173364034
c$$$ sie   =  5.06273055363485949E-002


!
! End of Program "guderley_1D"
!<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
!
	real*8 function Guderley(n, gammad, lambdad, B)
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                           !
!       This function computes a difference in similarity variable phase    !
!	space. In particular, It computes the result of two numerical       !
!	integrations:                                                       !
!                                                                           !
!	  (1) The final C-value found by integrating between x = -1         !
!	      and x = B (namely, at the space-time position of the          !
!	      reflected shock).                                             !
!	  (2) The final C-value found by integrating between x = infinity   !
!	      and x = B.                                                    !
!                                                                           !
!	Boundary conditions are available at both x = -1 and x = infinity,  !
!	but NOT in numerical form at x = B. Therefore, integration to       !
!	x = B must be performed starting from both x = -1 and x =           !
!	infinity, and the result compared.                                  !
!                                                                           !
!	It should be noted that the generalized Rankine-Hugonoit jump       !
!	conditions must be executed upon one (but not the other) of         !
!	the final integration points, so that the comparison of C           !
!	at x = B is consistent (i.e. not comparing the variable C           !
!	evaluated on one side of the shock wave to its value on the         !
!	other side of the shock wave).                                      !
!                                                                           !
!       Calls: ode, deroot      Called by: none                             !
!                                                                           !
!       Inputs (passed in from the driver program)                         !
!             n       dimensionality:  2 for cylindrical, 3 for spherical   !
!             gamma   ratio of specific heats                               !
!             lambda  similarity exponent                                   !
!             B       Estimate of the x-coordinate of the location of the   !
!                     reflected shock.                                      !
!                                                                           !
!       Output                                                             !
!            Guderley the difference between C1 (value of C behind          !
!                     the reflected shock obtained by integrating in        !
!                     increasing x) and y(2) (that from integrating         !
!                     in increasing w (decreasing x)).  The smaller the     !
!                     absolute value of this difference, the better the     !
!                     choice of B.                                          !
!                                                                           !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Start of Function "Guderley"
!
	implicit none
!
	integer doublefreq, iflag, intno, iwork(5), j, jmod, n, neq, neqn,
     & nu
	parameter (neqn = 3)
	real*8 abserr, aeroot, B, C, C1, C2, dw, dx, e, energy,
     & energymax(2), energymin(2), energy0, gamma, gammad, gm1, gp1,
     & lambda, lambdad, nuz, phi(neqn, 16), pressure, R, relerr,
     & reroot, sigma, Vdiff, V0, V1, w, wlast, work(100+21*neqn), wout,
     & x, xlast, xout, y(neqn)
	logical final
	common /param/ gamma, lambda, sigma, intno, nu
	common /V1/ V1
	external f, Vdiff
!
!----------------------------------------------------------------------------!
!
!.... The following parameters are adjustable, but it is not recommended
!       that they be adjusted unless error messages are returned by 
!       the function.
!
	data abserr /6.d-11/, doublefreq /50/, relerr /5.d-10/
	data aeroot /8.d-16/, reroot /8.d-16/
!
	pressure(C, R) = R*C*C/gamma
!
!.... nu = n - 1; it is 1 for cylindrical symmetry and 2 for spherical
!
	nu = n - 1
	gamma = gammad
	lambda = lambdad
!
!.... When final is false, the integration fro x = B to x = infinity
!       is skipped.
!
	final = .false.
	gp1 = gamma + 1.d0
	gm1 = gamma - 1.d0
!
!.... y(1) = V, y(2) = C, y(3) = R
!       The initial conditions starting at the incoming shock wave
!       are set here, along with the parameters necessary for a call
!       to "ode."
!
	y(1) = -2.d0/gp1
	y(2) = sqrt(2.d0*gamma*gm1)/gp1
	y(3) = gp1/gm1
	iflag = 1
	x = -1.d0
	xout = x
	intno = 1
!
!.... An energy integral is used as a consistency check during the 
!       integration. The definition of this parameter is set by 
!       a function defined below.
!
	energy0 = energy(x, y, gamma, lambda, nu, 0.d0)
	energymax(1) = 0.d0
	energymin(1) = 0.d0
!
!.... Slightly perturb the following if stuck on a singularity.
!
!       dx = 0.00131d0
	dx = 0.00125d0
!
! 9      format ('# ', a, ' Guderley problem, gamma', 1p, e22.15 /
!     & '#      x          density       velocity      ',
!     & 'pressure      sound speed energy check')
!	write (*, 10) x, y(3), y(1), pressure(y(2), y(3)), y(2), 0.d0
!
 10	format (1p, 2e14.7, e15.7, e14.7, e15.7, e9.1)
!
!.... Now begin the integration, starting from x = -1 and continuing to 
!       x = B. B is the x coordinate of the reflected shock. Only if the
!       integration returns the error message below should the parameters
!       abserr and relerr be adjusted.
!
      j = 0
      do while (xout .lt. B)
         j = j + 1
         xout = min(-1.d0 + dx*j, B)
         call ode(f, neqn, y, x, xout, relerr, abserr,iflag,work,iwork)
         e = energy(x, y, gamma, lambda, nu, energy0)
         energymin(1) = min(energymin(1), e)
         energymax(1) = max(energymax(1), e)
!         write (*, 10) x, y(3), y(1), pressure(y(2), y(3)), y(2), e
         if (iflag .ne. 2 .and. iflag .ne. 4) then
		write (*,*) 'Error during B-search for B = ', B
            write (*, 20) iflag, abserr, relerr
   20       format ('iflag', i3, '  abserr', 1p,  e14.7, '  relerr',
     &      e14.7)
            go to 98
         endif
      enddo
!
!.... If the integration up until x = B is completed, apply the shock
!       jump conditiosn given by Lazarus Eq. (2.6); afterwards
!       continue the integration from x = B to some large x (set here
!       at x = 10^6).
! 
	iflag = 1
	C2 = y(2)**2
	V1 = gm1*(1.d0 + y(1))/gp1 + 2.d0*C2/(gp1*(1.d0 + y(1))) - 1.d0
	y(2) = sign(sqrt(C2 + 0.5d0*gm1*((1.d0 + y(1))**2 -
     & (1.d0 + V1)**2)), y(2))
	C1 = y(2)
	y(3) = y(3)*(1.d0 + y(1))/(1.d0 + V1)
	y(1) = V1
!
	energy0 = energy(x, y, gamma, lambda, nu, 0.d0)
	energymax(2) = 0.d0
	energymin(2) = 0.d0
	if (.not. final) go to 40
	dx = .05d0
	xlast = B
	j = 0
	do while (x .lt. 1.e6)
	   j = j + 1
	   jmod = mod(j-1, doublefreq) + 1
!
!.... jmod goes 1, 2, ..., doublefreq
!
	   xout = xlast + dx*jmod
	   call ode(f, neqn, y, x, xout, relerr, 
     1          abserr,iflag,work,iwork)
	   e = energy(x, y, gamma, lambda, nu, energy0)
!
!.... If the energy check goes bad, quit the integration.
!
!       if (abs(e) .ge. 1.e-8) then
	   if (.false.) then
!	      write (*, 35) e, energy0
 35	      format ('energy, energy0', 1p, 2e15.7)
	      go to 40
	   endif
	   energymin(2) = min(energymin(2), e)
	   energymax(2) = max(energymax(2), e)
	   if (iflag .ne. 2) then
	      write (*, 20) iflag, abserr, relerr
	      go to 99
	   endif
	   if (jmod .eq. doublefreq) then
!
!.... Double the spacing of the output.
!
	      xlast = xout
	      dx = 2.d0*dx
	   endif
	enddo
!
!.... Now switch variables to w = k*x^(-sigma) and integrate from
!       w near zero to some positive w. This (redundant) integration
!       is necessary to determine the parameter B to more than the
!       six digits given by Lazarus (through the procedure described
!       in the function description space.
!
 40	iflag = 1
	nuz = (lambda - 1.d0)/gamma
	V0 = - 2.d0*nuz/(nu + 1)
	sigma = (1.d0 + nuz/(1.d0 + V0))/lambda
	w = 1.d-10
!
!.... Initial w is chosen as follows:
!       The first neglected term is the asymptotic expansion for
!       V = y(1) is w*w*V2, and in C = y(2) is w*c1, which are
!       O(w*w) less than the first terms.  Thus the neglected terms
!       are O(10^(-18)), less than the O(10^(-16)) machine precision.
!
      y(1) = V0
      y(2) = -1.d0/w
      intno = 2
      neq = 2
!
!.... We integrate only 2 differential equations for V and C.
!       Trying to integrate the R equation requires knowing k, which
!       we are trying to determine.  Thus no energy check is possible
!       here.  But it is not needed since no singularities arise.
!    
 50	format ( '#      w          velocity      sound speed')
	dw = w
	wlast = w
 55	format (1p, e14.7, 2e15.7)
	j = 0
	do
	   j = j + 1
	   jmod = mod(j-1, doublefreq) + 1
!
!.... jmod goes 1, 2, ..., doublefreq
!
	   wout = wlast + dw*jmod
	   call deroot(f, neq, y, w, wout, relerr, abserr, iflag, Vdiff,
     &   reroot, aeroot, phi)
	   if (iflag .ne. 2 .and. iflag .ne. 7) then
!	      write (*, 20) iflag, abserr, relerr
	      go to 99
	   endif
	   if (iflag .eq. 7) then
!
!.... Root found.  Let D be the number of correct digits in
!       lambda.  Then min(D, -log_10(|y(1) - V1|) or -log_10(
!       |y(2) - C1|)) is roughly the number of correct digits in B.
!           
	      exit
	   endif
	   if (jmod .eq. doublefreq) then
	      wlast = wout
!
!.... double the spacing of the output
!
	      if (dw .lt. .0025d0) dw = 2.d0*dw
	   endif
	enddo
!
!.... The phase space is calculated here and returned as output of
!       the function Guderley.
!
	Guderley = y(2) - C1
!
 65	format ('# differences: V', 1p, e10.2, ' C', e10.2,
     &  ' energy', 1p, 2e10.2)
 98	return
 99	call exit(1)
	end function Guderley
!
! End of Function "Guderley"
!<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
!
	real*8 function energy(x, y, gamma, lambda, nu, energy0)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                           !
!       The function energy is used during numerical integrations as a      !
!       consistency check. It computes the difference between the           !
!       adiabatic energy integral given by Lazarus Eq. (2.7) and its        !
!       initial value energy0. Its constancy is a good check on the         !
!       accuracy of the integration.                                        !
!                                                                           !
!       Inputs (passed in from the driver program and Guderley):            !
!            x         independent similary variable; space-time position   !
!            y(i)      V, C, or R for i = 1, 2, or 3                        !
!	     gamma     polytropic index                                     !
!            lambda    similarity exponent                                  !
!            nu        n - 1; space index                                   !
!            energy0   initial value of the adiabatic energy integral       !
!                                                                           !
!       Output:                                                             !
!            energy    The difference between the adiabatic energy          !
!	               integral evaluated at a particular space-time        !
!		       combination of x, V,C, and R.                        !
!                                                                           !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Start of Function "energy"
!
	implicit none
!
	real*8 energy0, gamma, lambda, q, x, y(3)
	integer nu
!
!---------------------------------------------------------------------------!
!
	q = 2.d0*(lambda - 1.d0)/(nu + 1)
	if (abs(x) .ge. 1.d-8) then
	   energy = (y(2)/x)**2*(1.d0 + y(1))**q*y(3)**(q - gamma + 1.d0)
     &   - energy0
	else
!
!.... It is impossible to compute C/x = dC/dx = 0/0 at x = 0,
!       since the differential equations are singular there. We punt
!       to avoid a machine infinity or NaN.
!
	   energy = 0.d0
	endif
	return
	end function energy
!
! End of Function "energy"
!<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
!
	subroutine f(xorw, y, yp)
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                           !
!       This subroutine evaluates the differential equations given by       !
!       Lazarus Eqs. (2.8), (2.9) and the R-equation.                       !
!                                                                           !
!       This subroutine (as opposed to the subroutine g) is for use with    !
!       the "Guderley" function. The difference between this subroutine     !
!       and "g" is the inclusion of diagnostic write statements appearing   !
!       in "g." Since this subroutine is used to evaluate Eqs. (2.8), (2.9) !
!       and the R-equation for choices of B that are incorrect (or          !
!       inprecise), the corresponding diagnostic statements have been       !
!       commented out.                                                      !
!                                                                           !
!	Calls: none         Called by: ode (intno = 1, xorw = x)            !
!                                      deroot (intno = 2, xorw = w)         !
!                               (in "Guderley" only)                        !
!                                                                           !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Start of Subroutine "f"
!
	implicit none
!
	real*8 C, C2, denom, factor, gamma, lambda, num(3), sigma, V,
     & Vp1, xorw, y(3), yp(3)
	integer intno, nu
	character*1 label(2)
	common /param/ gamma, lambda, sigma, intno, nu
	data label /'x', 'w'/
!

!
      V = y(1)
      C = y(2)
      Vp1 = V + 1.d0
      C2 = C*C
      denom = (C2 - Vp1**2)*xorw*lambda
      factor = (lambda - 1.d0)/gamma
      num(1) = ((nu + 1)*V + 2.d0*factor)*C2 - V*Vp1*(V + lambda)
      num(2) = (1.d0 + factor/Vp1)*C2 - 0.5d0*nu*(gamma - 1.d0)*V*Vp1
     & - Vp1**2 - 0.5d0*(lambda - 1.d0)*((3.d0 - gamma)*V + 2.d0)
!
!.... The next equation is redundant, as the density can be found
!       from energy conservation (2.7).  But we compute it so that we
!       can use (2.7) as a consistency/accuracy check.
!
      num(3) = - 2.d0*factor*C2/Vp1 + V*(V + lambda) - (nu + 1)*V*Vp1
!
!.... The diagnostic statements have been commented out.
!
!      if (abs(denom) .le. 1.d-6) then
!        Near a singular point such as x = 0, dV/dx = dC/dx = 0/0.
!        If this message is triggered, the calculation may eventually
!        terminate prematurely.  The remedy is to very slightly loosen
!        the tolerances abserr or relerr.
!         write (*, 20) label(intno), xorw, denom
!   20    format ('*** Warning, ', a, ' =', 1p, e23.15, '  denom =',
!     &   e10.2)
!      endif
      if (intno .eq. 2) then
!
!.... Here df/dw = df/dx / dw/dx with dw/dx = -sigma*w/x.  The 1/x
!       cancels the 1/x in df/dx, so the x's wash out.
!
         denom = -denom*sigma
	endif
	yp(1) = num(1)/denom
	yp(2) = C*num(2)/denom
	yp(3) = y(3)*num(3)/denom
	return
	end subroutine f
!
! End of Subroutine "f"
!<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
!
	subroutine g(t, y, yp)
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                           !
!       This subroutine evaluates the differential equations given by       !
!       Lazarus Eqs. (2.8), (2.9) and the R-equation.                       !
!                                                                           !
!       This subroutine (as opposed to the subroutine f) is for use with    !
!       the "sim" subroutine. The diagnostic statements have been left      !
!       in here.                                                            !
!                                                                           !
!	Calls: none         Called by: ode (intno = 1, xorw = x)            !
!                                      deroot (intno = 2, xorw = w)         !
!                               (in "sim" only)                             !
!                                                                           !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Start of Subroutine "g"
!
	implicit none
!
	real*8 C, C2, denom, factor, gamma, lambda, num(3), sigma, V,
     & Vp1, xorw, y(3), yp(3), t
	integer intno, nu
	character*1 label(2)
	common /param/ gamma, lambda, sigma, intno, nu
	data label /'x', 'w'/
!
!---------------------------------------------------------------------------!
!
	intno = 1
	V = y(1)
	C = y(2)
	Vp1 = V + 1.d0
	C2 = C*C
	denom = (C2 - Vp1**2)*t*lambda
	factor = (lambda - 1.d0)/gamma
	num(1) = ((nu + 1)*V + 2.d0*factor)*C2 - V*Vp1*(V + lambda)
	num(2) = (1.d0 + factor/Vp1)*C2 - 0.5d0*nu*(gamma - 1.d0)*V*Vp1
     &       - Vp1**2 - 0.5d0*(lambda - 1.d0)*((3.d0 - gamma)*V + 2.d0)
!
!.... The next equation is redundant, as the density can be gotten
!       from energy conservation (2.7).  But we compute it so that we
!       can use (2.7) as a consistency/accuracy check.
!
	num(3) = - 2.d0*factor*C2/Vp1 + V*(V + lambda) - (nu + 1)*V*Vp1
	if (abs(denom) .le. 1.d-8) then
!
!.... Near a singular point such as x = 0, dV/dx = dC/dx = 0/0.
!       If this message is triggered, the calculation may eventually
!       terminate prematurely.  The remedy is to very slightly loosen
!       the tolerances abserr or relerr.
!
	   write (*, 20) label(intno), t, denom
 20	   format ('*** Warning, ', a, ' =', 1p, e23.15, '  denom =',
     &   e10.2)
	endif
!	if (intno .eq. 2) then
!
!.... Here df/dw = df/dx / dw/dx with dw/dx = -sigma*w/x.  The 1/x
!       cancels the 1/x in df/dx, so the x's wash out.
!
!	   denom = -denom*sigma
!	endif
	yp(1) = num(1)/denom
	yp(2) = C*num(2)/denom
	yp(3) = y(3)*num(3)/denom
	return
	end subroutine g
!
!End of Subroutine "g"
!<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
!
	real*8 function Vdiff(w, y, yp)
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                           !
!       This function cComputes the difference between V1 (value of V       !
!	behind the reflected shock obtained by integrating in increasing    !
!	x) and y(1) (that is obtained from integrating in increasing w      !
!	(decreasing x)). The smaller the absolute value of this difference, !
!	and the corresponding C difference are, the better the choice of    !
!	B.                                                                  !
!                                                                           !
!       Calls: none        Called by: deroot                                !
!                                                                           !
!       Inputs (passed from "Guderley"):                                    !
!            w    transformed independent similarity variable               !
!            y    y(i) = V, C, or R for i = 1, 2, or 3                      !
!            yp   RHS of Eqs. (2.8), (2.9), and the R-equation              !
!                                                                           !
!       Output:                                                             !
!            Vdiff  difference between V1 (from x) and corresponding        !
!                   value from integrating in w.                            !
!                                                                           !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Start of Function "Vdiff"
!
	implicit none
!
	real*8 V1, w, y(2), yp(2)
	common /V1/ V1
!
!---------------------------------------------------------------------------!
!
	Vdiff = y(1) - V1
	return
	end
!
! End of Function "Vdiff"
!<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


!
!End of Subroutine "state"
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

      subroutine state(r, rho0, n, gammad, lambdad, B, targetxd,
     2      den, vel, pres, snd, sie)
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                           !
!       This subroutine, given the various parameters computed in other     !
!       parts of the driver program guderley_1D, integrates the governing   !
!       ODEs up to a pre-specified point (targetx, which is computed in     !
!       the guderley_1D driver program. It then transforms the similarity   !
!       variable data at the targetx point to physical data at a particular !
!       space-time point.                                                   !
!                                                                           !
!       Lazarus Eqs. (2.5) are used to transform the similarity variables   !
!       back to physical variable space in this subroutine. The results     !
!       will be in terms of the "Lazarus Time," as opposed to "Caramana     !
!       and Whalen" Time.                                                   !
!                                                                           !
!       Calls: ode               Called by: guderley_1D                     !
!                                                                           !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Start of Subroutine "state"
!
	implicit none
!
	integer iflag, iwork(5), n, neqn, nu, j, jmod, doublefreq
	parameter (neqn = 3)
	real*8 abserr, relerr, work(100+21*neqn), y(neqn), B, targetx
	real*8 lambda, lambdad, gamma, gammad, R, C, pressure, targetxd
	real*8 gp1, gm1, t, C2, V1, C1, tout, dx, xlast, xout, rho0, p
	common /param/ gamma, lambda, nu
	external g
	data abserr /6.d-14/, relerr /5.d-13/
        real*8 den, vel, snd, pres,sie
	pressure(C, R) = R*C*C/gamma
	nu = n - 1
	gamma = gammad
	lambda = lambdad
	targetx = targetxd
!
!.... Factors gamma + 1 and gamma - 1.
!
	gp1 = gamma + 1.d0
	gm1 = gamma - 1.d0
!
!.... Initializing the similarity variables at the position of the 
!       converging shock wave. This is the starting point of all
!       integrations of the governing equations.
!
	y(1) = -2.d0/gp1
	y(2) = sqrt(2.d0*gamma*gm1)/gp1
	y(3) = gp1/gm1
	t = -1.d0
!
	iflag = 1
!
!.... x < -1 represents the unshocked state (interior to hte converging
!       shock wave), where the physical variables have constant values
!       given by:
!
!          density = constant (specified by user)
!          velocity = 0 (required)
!          pressure = sound speed = 0 (required)
!
!       When a combination of space and time variables are specified
!       such that x < -1, the constant state data is returned as output.
!
	if (targetx .lt. -1.d0) then
	   den  = rho0
           vel  = 0.0
           pres = 0.0
	   snd  = 0.0
	   sie  = 0.0
!          write(*,*) 'case 1:'
!	   write(*,*) 'Density = ', den
!	   write(*,*) 'Velocity = ', vel
!	   write(*,*) 'Pressure = ', pres
!	   write(*,*) 'Sound Speed =', snd
!	   write(*,*) 'SIE =', sie
!
!.... If -1 < x < 0, then we are behind the converging shock wave, and
!       reflection has yet to occur. The integration of the governing
!       ODEs is initiated at the position of the converging shock
!       (x = -1) and carried through to whatever negative value of x
!       that results from the specification of the space and time 
!       variables.
!
	else if (targetx .lt. 0.d0 .and. targetx .ge. -1.d0) then
	   do while (t .lt. targetx)
	      call ode(g,neqn,y,t,targetx,relerr,abserr,
     &        iflag,work,iwork)
	   enddo
!
!.... Definition of the PHYSICAL pressure variable, as a function of the
!       dimensionless similarity variables.
!
	   p = (((y(2)*r**(1.d0-lambda))/
     &       (targetx*(-1.d0)*lambda))**2)/
     &       (gamma*(1.d0/rho0)*(1.d0/y(3)))
!
!.... Writing of solution data.
!
	   den  = y(3)*rho0
           vel  = (y(1)*r**(1.d0-lambda))/(targetx*(-1.d0)*lambda)
           pres = p
	   snd  = (y(2)*r**(1.d0-lambda))/(targetx*(-1.d0)*lambda)
	   sie  = p/(gm1*rho0*y(3))

!          write(*,*) 'case 2:'
!	   write(*,*) 'Density = ', den
!	   write(*,*) 'Velocity = ', vel
!	   write(*,*) 'Pressure = ', pres
!	   write(*,*) 'Sound Speed = ', snd
!	   write(*,*) 'SIE = ', sie
!
!.... If 0 < x < B (the space-time position of the reflected shock wave),
!       then we are upstream of the reflected shock wave. The integration
!       of the governing ODEs is again started at the position of the 
!       convergent shock wave and integrated through x = 0 into a portion
!       of the phase space representing the flow ahead of the reflected
!       shock wave. The integration terminates before the position x = B
!       is reached.
!
	else if (targetx .gt. 0.d0 .and. targetx .lt. B) then
	   do while (t .lt. targetx)
	      call ode(g,neqn,y,t,targetx,relerr,abserr,
     &        iflag,work,iwork)
	   enddo
!
!.... Physical pressure variable.
!
	   p = (((y(2)*r**(1.d0-lambda))/
     &       (targetx*(-1.d0)*lambda))**2)/
     &       (gamma*(1.d0/rho0)*(1.d0/y(3)))
!
!.... Writing of solution data.
!
	   den  = y(3)*rho0
           vel  = (y(1)*r**(1.d0-lambda))/(targetx*(-1.d0)*lambda)
           pres = p
	   snd  = (y(2)*r**(1.d0-lambda))/(targetx*(-1.d0)*lambda)
	   sie  = p/(gm1*rho0*y(3))

!          write(*,*) 'case 3:'
!	   write(*,*) 'Density = ', den
!	   write(*,*) 'Velocity = ', vel
!	   write(*,*) 'Pressure = ', p
!	   write(*,*) 'Sound Speed = ', snd
!	   write(*,*) 'SIE = ', sie
!
!.... If B < x < infinity, then we are behind the reflected shock wave.
!       The numerical integration starts at the position of the convergent
!       shock wave as before and is carred through x = 0 until x = B.
!
	else if (targetx .gt. B) then
	   do while (t. lt. B)
	      call ode(g,neqn,y,t,B,relerr,abserr,iflag,work,iwork)
	   enddo
!
	   iflag = 1
!
!.... At x = B, the general-strength Rankine-Hugoniot conditions are
!       applied, and we move to the other side of the reflected shock
!       wave (just downstream).
!
	   C2 = y(2)**2
	   V1 = gm1*(1.d0 + y(1))/gp1 + 2.d0*C2/
     &       (gp1*(1.d0 + y(1))) - 1.d0
	   y(2) = sign(sqrt(C2 + 0.5d0*gm1*(
     &       (1.d0 + y(1))**2-(1.d0+V1)**2)), y(2))
	   C1 = y(2)
	   y(3) = y(3)*(1.d0 + y(1))/(1.d0 + V1)
	   y(1) = V1
!
!.... Numerical integration of the governing ODEs continues from x = B
!       (with the similarity variables taking their shocked values) until
!       the targetx point is reached.
!
	   do while (t .lt. targetx)
	      call ode(g,neqn,y,t,targetx,relerr,abserr,
     &        iflag,work,iwork)
	   enddo
!
!.... Physical pressure variable.
!
	   p = (((y(2)*r**(1.d0-lambda))/
     &       (targetx*(-1.d0)*lambda))**2)/
     &       (gamma*(1.d0/rho0)*(1.d0/y(3)))
!
!....Writing of solution data.
!
	   den  = y(3)*rho0
           vel  = (y(1)*r**(1.d0-lambda))/(targetx*(-1.d0)*lambda)
	   snd  = (y(2)*r**(1.d0-lambda))/(targetx*(-1.d0)*lambda)
           pres = p
	   sie  = p/(gm1*rho0*y(3))

!          write(*,*) 'case 4:'
!	   write(*,*) 'Density = ', den
!	   write(*,*) 'Velocity = ', vel
!	   write(*,*) 'Pressure = ', pres
!	   write(*,*) 'Sound Speed = ', snd
!	   write(*,*) 'SIE = ', sie
	endif
!
	return
	end subroutine state
