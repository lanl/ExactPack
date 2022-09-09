      Subroutine riemann_kamm(time, npts, x, xd0,       &
           gammal, gammar, rhol, pl, ul, rhor, pr, ur,  &
           rho, p, u, sound, sie, entropy)
      implicit none
!f2py intent(out)  :: rho, p, u, sie, sound, entropy
!f2py intent(hide) :: npts
!f2py integer      :: npts
!f2py real         :: time, x(npts), xd0
!f2py real         :: gammal, gammar, rhol, pl, ul, rhor, pr, ur
!f2py real         :: rho(npts), p(npts), u(npts), sie(npts), sound(npts)
!f2py real         :: entropy(npts)

      integer  npts
      real*8   x(npts), rho(npts), p(npts), u(npts), entropy(npts)
      real*8   sie(npts), sound(npts)
      real*8   time, xd0, gammal, gammar, rhol, pl, ul, rhor, pr, ur

      integer it, ierr, nx
      real*8 rhoi, presi, ui, soundi, siei, entropyi
      real*8 xmin, xmax, xi

      xmin = x(1)
      xmax = x(npts)
      nx   = 3
      Do it=1, npts
         xi=x(it)
         Call SHKTUB ( xi, rhol, rhor, pl, pr, ul, ur, gammal, gammar,  &
              nx, xmin, xmax, xd0, time, ierr, rhoi, presi, ui, soundi, &
              siei, entropyi)

         rho(it) = rhoi
         p(it) = presi
         u(it) = ui
         sound(it) = soundi
         sie(it) = siei
         entropy(it) = entropyi
      EndDo

      End Subroutine riemann_kamm
!
! nx=3 for two intervals
! x_point <== return solution here
!
      Subroutine SHKTUB ( xi, rhol, rhor, pl, pr, ul, ur, gammal, &
           gammar, nx, xmin, xmax, xd0, time, ierr,               &
           rhoi, presi, ui, ci, ei, si )
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
! ----- this subroutine computes the solution to the shock tube problem,
! ----- using the technique outlined in Gottlieb & Groth, "Assessment
! ----- of riemann solvers for unsteady one-dimensional inviscid 
! ----- flows of perfect gases", J.Comp.Phys. 78 (2), 437-458 (1988)
!
! ----- this includes the calculation of the exact solution 
! ----- in the rarefaction wave region, and the calculation 
! ----- of the expanded (vacuum) state
!
! ----- Input data:
! -----   xi              <-> position of desired solution
! -----   rhol,rhor       <-> left and right densities
! -----   pl,pr           <-> left and right pressures
! -----   ul,ur           <-> left and right velocities
! -----   gammal,gammar   <-> left and right gas constants
! -----   xd0             <-> initial x-location of diaphragm
! -----   time            <-> simulation time
! -----   nx              <-> number of zones
! -----   xmin,xmax       <-> min/max of edges of entire domain
! -----
! ----- Output values are written to the arrays:
! -----   xval(i), i=1,nx <-> array of x-positions of cell centers
! -----   uout(i), i=1,nx              velocity
! -----   pout(i), i=1,nx              pressure
! -----   rout(i), i=1,nx              density
! -----   eout(i), i=1,nx              specifi! internal energy
! -----   cout(i), i=1,nx              sound speed
! -----   sout(i), i=1,nx              entropy
! -----   rhoi                         density at xi
! -----   presi                        pressure at xi
! -----   ui                           velocity at xi
! -----   ci                           sound speed at xi
! -----   ei                           specifi! internal energy at xi
! -----   si                           entropy at xi
!   
! -----   Calls:  none              Called by: DOIT
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Start of Subroutine SHKTUB
!
      Implicit None
!
!.... Include files
!
      Include "param.h"
!
!.... Input variables
!
      Integer ierr                  ! error flag
      Integer nx                    ! index
!
      Real*8  ur                    ! initial right velocity   (cm/s)
      Real*8  ul                    ! initial left  velocity   (cm/s)
      Real*8  pr                    ! initial right pressure   (dyne/cm2)
      Real*8  pl                    ! initial left  pressure   (dyne/cm2)
      Real*8  rhor                  ! initial right density    (g/cm3)
      Real*8  rhol                  ! initial left  density    (g/cm3)
      Real*8  gammar                ! right gamma-law gas constant
      Real*8  gammal                ! left  gamma-law gas constant
      Real*8  xmin                  ! minimum x-edge location  (cm)
      Real*8  xmax                  ! maximum x-edge location  (cm)
      Real*8  xd0                   ! x-location of diaphragm  (cm)
      Real*8  time                  ! simulation time          (s)
      Real*8  xi                    ! position of solution     (cm)
!
!.... Output variables
!
      Real*8 rhoi                   ! density at position xi   (g/cm3)
      Real*8 presi                  ! pressure at position xi  (dyne/cm2)
      Real*8 ui                     ! velocity at position xi  (cm/s)
      Real*8 ci                     ! sound speed at xi        (cm/s)
      Real*8 ei                     ! specifi! energy at  xi   (erg/g)
      Real*8 si                     ! entropy at position xi   (units k_boltzmann?)

! c
! c.... Local variables
! c
      Real*8  x                     ! dummy value of location  (cm)
      Real*8  dx                    ! zone size                (cm)
      Real*8  rl                    ! initial left  density
      Real*8  el                    ! initial left  sie
      Real*8  al                    ! initial left  sound speed
      Real*8  rr                    ! initial right density
      Real*8  er                    ! initial right sie
      Real*8  ar                    ! initial right sound speed
      Real*8  tol                   ! iterative solution tolerance
      Real*8  xcd                   ! x-location of contact discontinuity
      Real*8  xd                    ! x-location of diaphragm
      Real*8  urar                  ! x-velocity   of point in the rarefaction
      Real*8  arar                  ! sound speed  of point in the rarefaction
      Real*8  rrar                  ! density      of point in the rarefaction
      Real*8  prar                  ! pressure     of point in the rarefaction
      Real*8  erar                  ! sie          of point in the rarefaction
      Real*8  cl0                   ! sound speed  of left  state
      Real*8  cr0                   ! sound speed  of right state
      Real*8  xprime                ! non-dim dist from raref'n location to xd
      Real*8  gampr                 ! gampr = ( gamma4 - 1.0 ) / ( 2.0 * a4 )
      Real*8  uscn                  ! speed of shock-contact-no_wave bndry
      Real*8  uncs                  ! speed of no_wave-contact-shock bndry
      Real*8  uncr                  ! speed of no_wave-contact-raref'n bndry
      Real*8  urcn                  ! speed of raref'n-contact-no_wave bndry
      Real*8  urcvr                 ! speed of raref'n-contact-vacuum-raref'n boundary
      Real*8  ulsnk                 ! see G&G eq. 3.48
      Real*8  ursnk                 ! see G&G eq. 3.48
      Real*8  zed                   ! see G&G eq. 3.49
      Real*8  sigma                 ! see G&G eq. 3.49
      Real*8  ustar                 ! velocity in contact region
      Real*8  ustarl                ! velocity in contact region
      Real*8  ustarr                ! velocity in contact region
      Real*8  vl                    ! velocity 
      Real*8  vr                    ! velocity 
      Real*8  wl                    ! velocity 
      Real*8  wr                    ! velocity 
      Real*8  xlshk                 ! location of leftward  shock
      Real*8  xrshk                 ! location of rightward shock
      Real*8  rlst                  ! density     to left  in contact region
      Real*8  elst                  ! sie         to left  in contact region
      Real*8  plst                  ! pressure    to left  in contact region
      Real*8  plstpr                ! pressure'   to left  in contact region
      Real*8  alst                  ! sound speed to left  in contact region
      Real*8  rrst                  ! density     to right in contact region
      Real*8  erst                  ! sie         to right in contact region
      Real*8  prst                  ! pressure    to right in contact region
      Real*8  prstpr                ! pressure'   to right in contact region
      Real*8  arst                  ! sound speed to right in contact region
      Real*8  vlrhead               ! velocity of leftward rarefaction head
      Real*8  vlrtail               ! velocity of leftward rarefaction tail
      Real*8  xlrhead               ! location of leftward rarefaction head
      Real*8  xlrtail               ! location of leftward rarefaction tail
      Real*8  vrrhead               ! velocity of rightward rarefaction head
      Real*8  vrrtail               ! velocity of rightward rarefaction tail
      Real*8  xrrhead               ! location of rightward rarefaction head
      Real*8  xrrtail               ! location of rightward rarefaction tail
      Real*8  xlcnt                 ! location of leftward  contact
      Real*8  xrcnt                 ! location of rightward contact
      Real*8  err1                  ! error term in iteration
      Real*8  err2                  ! error term in iteration
!
!.... Output variables
!
      Real*8  xval(1:nx)            ! x-positions for solution (cm)
      Real*8  uout(1:nx)            ! velocity                 (cm/s)
      Real*8  pout(1:nx)            ! pressure                 (dyne/cm2)
      Real*8  rout(1:nx)            ! density                  (g/cm3)
      Real*8  eout(1:nx)            ! specifi! internal energy (erg/g)
      Real*8  cout(1:nx)            ! sound speed              (cm/s)
      Real*8  sout(1:nx)            ! specifi! entropy         (?)
!
      Integer i                      ! index
      Integer istate                 ! state flag
      Integer itmax                  ! maximum number of iterations
      Integer itnum                  ! iteration number
      Integer iun                    ! unit number of shocktube diagnosti! file
!
      Character*5 cvac               ! 'vacum' state - vacuum state   
      Character*5 cshock             ! 'shock' bndry - shock bndry
      Character*5 ccd                ! 'cd   ' bndry - contact discontinuity bndry
      Character*5 crhead             ! 'rhead' bndry - rarefaction-head bndry
      Character*5 crtail             ! 'rtail' bndry - rarefaction-tail bndry
!
      Logical ldebug                 ! debug flag
      Logical ldump                  ! Write diagnostics flag
!
      Data cvac   / 'vacum' /
      Data cshock / 'shock' /
      Data ccd    / 'cd   ' /
      Data crhead / 'rhead' /
      Data crtail / 'rtail' /
!
      Data itmax  / 200     /        ! maximum number of iterations
      Data tol    / 1.e-8   /        ! iterative solution tolerance
!
!----------------------------------------------------------------------
!
      ierr   = 0
      ldebug = .true.
      ldump  = .false.
      itnum  = 0
      iun    = 20

      If ( ldump ) Then
         Open( unit=iun, file = 'shktub.log', status='unknown')
      End If
!
!.... Assign the cell centers of each zone
!
! Code modified to provide values of xval(i) at
! edges rather than centers.
!
      dx = ( xmax - xmin ) / Dble( nx -1 )
      Do i = 1, nx
        xval(i) = xmin + (i-1) * dx
      EndDo

!
!.... Assign initial data
!
      rr = rhor
      rl = rhol
      xd = xd0
!
!.... Exit with unacceptable initial data
!
      if ( ( pr .eq. pl ) .and. ( ur .eq. ul ) )     ierr = 1
      if ( ( pr .eq. pl ) .and. ( pr .le. ZERO ) )   ierr = 2
      if ( ( pr .eq. ZERO ) .and. ( ur .ne. ZERO ) ) ierr = 3
      if ( ( pl .eq. ZERO ) .and. ( ul .ne. ZERO ) ) ierr = 4
      if ( gammar .le. ONE ) ierr = 5
      if ( gammal .le. ONE ) ierr = 6
      if ( rr .le. ZERO )    ierr = 7
      if ( rl .le. ZERO )    ierr = 8
      if ( ierr .ne. 0 ) goto 799 
! c
! c.... Assign related values
! c
      er = ( pr / rr ) / ( gammar - ONE )
      el = ( pl / rl ) / ( gammal - ONE )
      ar = Sqrt( gammar * pr / rr )
      al = Sqrt( gammal * pl / rl )
      
      If ( ldebug .and. ldump ) Then
        Write(iun,*) '** rl =',rl,'   rr =',rr
        Write(iun,*) '** pl =',pl,'   pr =',pr
        Write(iun,*) '** el =',el,'   er =',er
        Write(iun,*) '** ul =',ul,'   ur =',ur
      End If ! ldebug
! c
! c=======================================================================
! c.... Special case of zero right pressure:  rarefaction-contact-zero_pres
! c=======================================================================
! c
      If ( pr .eq. ZERO ) Then
!
        If ( ldump ) Then
           Write(iun,920) 
           Write(iun,996) 
           Write(iun,997)
        End If
!
        ustarl = ul + TWO * al / ( gammal - ONE ) &
                    + TWO * ar / ( gammar - ONE ) 
!
  10    Continue
!
        itnum = itnum + 1
        If ( itnum .gt. itmax ) ierr = 10
        If ( ierr .ne. 0 ) Go To 799
! c
! c.... Leftward moving rarefaction conditions  G&G eqs. 3.41-3.43
! c
        alst = al - half * ( gammal - ONE ) * ( ustarl - ul )
        plst = pl &       
             * ( ( alst / al )**( TWO * gammal / ( gammal - ONE ) ) )
        plstpr = -gammal * plst / alst
! c
! c.... Rightward vacuum condition
! c
        arst   = ZERO
        prst   = ZERO
        prstpr = ZERO
! c
! c.... Check for convergence
! c
        err1 = ONE
        err2 = ( plst - prst ) / ( plstpr - prstpr )
        If ( ldump ) Then
           Write(iun,998) itnum, Abs( err1 ),  Abs( err2 ), Abs( err2 )  
        End If
        If ( Abs( err2 ) .lt. tol ) Go To 19
!
        ustarl = ustarl - err2
        Go To 10
!
  19    Continue                             ! converged left rarefaction solution
!
        ustarl = ustarl - err2
! c
! c.... Compute boundary of left rarefaction head & tail
! c
        vlrhead = ul     - al                ! velocity of head of rarefaction
        vlrtail = ustarl - alst              ! velocity of tail of rarefaction
        xlrhead = xd + vlrhead * time        ! location of head of rarefaction
        xlrtail = xd + vlrtail * time        ! location of tail of rarefaction
! c
! c.... Compute contact locations
! c
        xlcnt = xd + ustarl * time           ! location of left  contact
! c
! c.... Write data
! c
        If ( ldump ) Then
           Write(iun,989) ustarl
           Write(iun,990) 
           Write(iun,991) 
           Write(iun,992) crhead,vlrhead
           Write(iun,992) crtail,vlrtail
           Write(iun,992) ccd,ustarl
           Write(iun,992) cvac,zero
           Write(iun,981) xlrhead, xlrtail, xlcnt
        End If
!
!.... Assign the values to the given positions
!

!
! don't loop over i, set x = x_point
!
!*c        Do i = 1, nx
!*c          x = xval(i)
          x = xi
! c
! c.... To the left of the head of the leftward rarefaction
! c
          If ( x .le. xlrhead ) Then
!
            ui    = ul  ! return these values uouti, pouti, ...
            presi = pl
            rhoi  = rl
            ei    = el
            ci    = Sqrt( gammal * presi / rhoi )
            si    = Log( eout(i)  &                   
                 * ( ONE / rhoi )**( gammal - ONE ) )
!c            uout(i) = ul  ! return these values uouti, pouti, ...
!c            pout(i) = pl
!c            rout(i) = rl
!c            eout(i) = el
!c            cout(i) = Sqrt( gammal * pout(i) / rout(i) )
!c            sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammal - ONE ) )

!
!.... Between the head and tail of the leftward rarefaction
!
          Else If ( x .gt. xlrhead .and. x .lt. xlrtail ) Then
!
            gampr = HALF * ( gammal - ONE ) / al
            xprime = ( x - xd ) / ( al * time )
            urar   = ( TWO * al / ( gammal + ONE ) )  &             
                 * ( xprime + ONE + gampr * ul )
            arar = al + HALF * ( gammal - ONE ) * ( ul - urar )
            rrar = rl * ( ONE - gampr     &           
                 * ( urar - ul ) )**( TWO / ( gammal - ONE ) )
            prar = pl * ( ONE - gampr     &          
                 * ( urar - ul ) )**( ( TWO * gammal ) /      &                               
                   ( gammal - ONE ) )
            erar = ( prar / rrar) / ( gammal - ONE )
            ui    = urar
            presi = prar
            rhoi  = rrar
            ei    = erar
            ci    = Sqrt( gammal * presi / rhoi )
            si    = Log( ei      &                   
                 * ( ONE / rhoi )**( gammal - ONE ) )
! cc            uout(i) = urar
! cc            pout(i) = prar
! cc            rout(i) = rrar
! cc            eout(i) = erar
! cc            cout(i) = Sqrt( gammal * pout(i) / rout(i) )
! cc            sout(i) = Log( eout(i)
! cc     &                   * ( ONE / rout(i) )**( gammal - ONE ) )

! c
! c.... Between the tail of the leftward rarefaction and the contact
! c
          Else If ( x .ge. xlrtail .and. x .lt. xlcnt ) Then
!
            ui    = ustarl
            presi = prst
            rhoi  = ZERO
            ei    = ZERO
            ci    = ZERO
            si    = ZERO
! cc            uout(i) = ustarl
! cc            pout(i) = prst
! cc            rout(i) = ZERO
! cc            eout(i) = ZERO
! cc            cout(i) = ZERO
! cc            sout(i) = ZERO
! c
! c.... To the right of the contact
! c
          Else
!
            ui    = ur  
            presi = pr
            rhoi  = rr
            ei    = er
            ci    = ZERO
            si    = ZERO
! cc            uout(i) = ur  
! cc            pout(i) = pr
! cc            rout(i) = rr
! cc            eout(i) = er
! cc            cout(i) = ZERO
! cc            sout(i) = ZERO
! c
          End If ! x .le. xlrhead
!
! c*!        End Do ! i
! c
! c======================================================================
! c.... Special case of zero left  pressure:  zero_pres-contact-rarefaction
! c======================================================================
! c
      Else If ( pl .eq. zero ) Then
!
        If ( ldump ) Then
           Write(iun,930) 
           Write(iun,996) 
           Write(iun,997) 
        End If
! c
! c.... Solve for rightward contact
! c
        ustarr = ul + TWO * al / ( gammal - ONE )      &              
             + TWO * ar / ( gammar - ONE ) 
!
  50    Continue
!
        itnum = itnum + 1
        If ( itnum .gt. itmax ) ierr = 10
        If ( ierr .ne. 0 ) Go To 799
! c
! c.... Rightward moving rarefaction conditions  G&G eqs. 3.44-3.46
! c
        arst = ar + HALF * ( gammar - ONE ) * ( ustarr - ur )
        prst = pr      &       
             * ( ( arst / ar )**( TWO * gammar / ( gammar - ONE ) ) )
        prstpr = gammar * prst / arst
! c
! c.... Leftward vacuum condition
! c
        alst   = ZERO
        plst   = ZERO
        plstpr = ZERO
! c
! c.... Check for convergence
! c
        err1 = ONE
        err2 = ( plst - prst ) / ( plstpr - prstpr )
        If ( ldump ) Then
           Write(iun,998) itnum, Abs( err1 ),  Abs( err2 ), Abs( err2 )  
        End If
        if ( Abs( err2 ) .lt. tol ) goto 59
!
        ustarr = ustarr - err2
        goto 50
!
  59    continue                             ! converged right rarefaction solution
!
        ustarr = ustarr - err2
! c
! c.... Compute boundary of right rarefaction tail & head 
! c
        vrrhead = ur     + ar                ! velocity of head of rarefaction
        vrrtail = ustarr + arst              ! velocity of tail of rarefaction
        xrrhead = xd + vrrhead * time        ! location of head of rarefaction
        xrrtail = xd + vrrtail * time        ! location of tail of rarefaction
! c
! c.... Compute contact locations
! c
        xrcnt = xd + ustarr * time           ! location of right contact
! c
! c.... Write data
! c
        If ( ldump ) Then
           Write(iun,989) ustarr
           Write(iun,990) 
           Write(iun,991) 
           Write(iun,992) cvac,zero
           Write(iun,992) ccd,ustarr
           Write(iun,992) crtail,vlrtail
           Write(iun,992) crhead,vlrhead
           Write(iun,982) xrcnt, xrrtail, xrrhead
        End If
!
! c.... Assign the values to the given positions
! c
! c*!        Do i = 1, nx
! c*!          x = xval(i)
           x = xi
! c
! c.... To the left of the contact
! c
          If ( x .le. xrcnt ) Then
!
            ui    = ul
            presi = pl
            rhoi  = rl
            ei    = el
            ci    = ZERO
            si    = ZERO
! !c            uout(i) = ul
! !c            pout(i) = pl
! !c            rout(i) = rl
! !c            eout(i) = el
! !c            cout(i) = ZERO
! !c            sout(i) = ZERO
! c
! c.... Between the contact and tail of the rightward rarefaction
! c
          Else If ( x .gt. xrcnt .and. x .le. xrrtail ) Then
!
            ui    = ustarr
            presi = prst
            rhoi  = ZERO
            ei    = ZERO
            ci    = ZERO
            si    = ZERO
! !c            uout(i) = ustarr
! !c            pout(i) = prst
! !c            rout(i) = ZERO
! !c            eout(i) = ZERO
! !c            cout(i) = ZERO
! !c            sout(i) = ZERO
! c
! c.... Between the tail and head of the rightward rarefaction
! c
          Else If ( x .gt. xrrtail .and. x .le. xrrhead ) Then
!
            gampr = half * ( gammar - ONE ) / ar
!
            xprime = ( x - xd ) / ( al * time )
            urar   = ( TWO * ar / ( gammar + ONE ) )      &             
                 * ( xprime - ONE + gampr * ur )
            arar = ar + HALF * ( gammar - ONE ) * ( urar - ur )
            rrar = rr * ( ONE + gampr     &           
                 * ( urar - ur ) )**( TWO / ( gammar - ONE ) )
            prar = pr * ( ONE + gampr     &        
                 * ( urar - ur ) )**( ( TWO * gammar ) / ( gammar - ONE ) )
            erar = ( prar / rrar) / ( gammar - ONE )
            ui     = urar
            presi  = prar
            rhoi   = rrar
            ei     = erar
            ci     = Sqrt( gammal * pout(i) / rout(i) )
            si     = Log( eout(i)     &                   
                 * ( ONE / rout(i) )**( gammar - ONE ) )

! !c            uout(i) = urar
! !c            pout(i) = prar
! !c            rout(i) = rrar
! !c            eout(i) = erar
! !c            cout(i) = Sqrt( gammal * pout(i) / rout(i) )
! !c            sout(i) = Log( eout(i)
! !c     &                   * ( ONE / rout(i) )**( gammar - ONE ) )
! c
! c.... To the right of the rightward rarefaction
! c
          Else
!
            ui    = ur
            presi = pr
            rhoi  = rr
            ei    = er
            ci    = Sqrt( gammal * pout(i) / rout(i) )
            si    = Log( eout(i)     &                   
                 * ( ONE / rout(i) )**( gammar - ONE ) )

! !c            uout(i) = ur
! !c            pout(i) = pr
! !c            rout(i) = rr
! !c            eout(i) = er
! !c            cout(i) = Sqrt( gammal * pout(i) / rout(i) )
! !c            sout(i) = Log( eout(i)
! !c     &                   * ( ONE / rout(i) )**( gammar - ONE ) )
! c
          End If ! x .le. xrcnt
!*c        End Do ! i
!
      Else
! c
! c======================================================================
! !.... Multiple wave cases
! c======================================================================
! c
!.... Determine the nature of the solution, which must be 
!.... one of the following, from left-to-right:
!
!....  (1) shock-contact-shock
!....  (2) shock-contact-rarefation
!....  (3) rarefaction-contact-rarefaction
!....  (4) rarefaction-contact-shock
!....  (5) rarefaction-contact-vacuum-contact-rarefaction
!
        urcvr = ul + TWO * al / ( gammal - ONE )      &             
             + TWO * ar / ( gammar - ONE ) 
!
!.... Equal pressure case
!
        If ( pr .eq. pl ) Then
          If ( ur .lt. ul ) Then
            istate = 1      ! shock-contact-shock
          Else If ( ur .gt. ul .and. ur .lt. urcvr ) Then
            istate = 3      ! rarefaction-contact-rarefaction
          Else
            istate = 5      ! rarefaction-contact-vacuum-contact-rarefaction
          End If ! ur .lt. ul
!
!.... Right pressure > left pressure
!
        Else If ( pr .gt. pl ) Then
          uscn = ul - ( al / gammal ) * ( ( pr / pl ) - ONE )      &   
               / Sqrt( HALF * ( ( gammal + ONE ) / gammal ) * ( pr / pl )      &         
               + HALF * ( ( gammal - ONE ) / gammal ) )
          uncr = ul + ( TWO * ar / ( gammar - ONE ) )      & 
               * ( ONE - ( pl / pr )**( HALF * ( ( gammar - ONE ) / gammar ) ) )
          If ( ur .le. uscn ) Then
            istate = 1      ! shock-contact-shock
          Else If ( ur .gt. uscn .and. ur .lt. uncr ) Then
            istate = 2      ! shock-contact-rarefaction
          Else If ( ur .ge. uncr .and. ur .lt. urcvr ) Then
            istate = 3      ! rarefaction-contact-rarefaction
          Else
            istate = 5      ! rarefaction-contact-vacuum-contact-rarefaction
          End If ! ur .le. uscn
!
!.... Right pressure < left pressure
!
        Else 
          uncs = ul - ( ar / gammar ) * ( ( pl / pr ) - ONE )     &   
               / Sqrt( HALF * ( ( gammar + ONE ) / gammar ) * ( pl / pr )      &         
               + HALF * ( ( gammar - ONE ) / gammar ) )
          urcn = ul + ( TWO * al / ( gammal - ONE ) )      & 
               * ( ONE - ( pr / pl )**( HALF * ( ( gammal - ONE ) / gammal ) ) )
          If ( ur .le. uncs ) Then
            istate = 1      ! shock-contact-shock
          Else If ( ur .gt. uncs .and. ur .lt. urcn ) Then
            istate = 4      ! rarefaction-contact-shock
          Else If ( ur .ge. urcn .and. ur .lt. urcvr ) Then
            istate = 3      ! rarefaction-contact-rarefaction
          Else
            istate = 5      ! rarefaction-contact-vacuum-contact-rarefaction
          End If ! ur .le. uncs
        End If ! pr .eq. pl
!
!.... Assign initial guess to the intermediate velocity
!
        If ( istate .ne. 5 ) Then
          ulsnk = ul + TWO * al / ( gammal - ONE )
          ursnk = ur - TWO * ar / ( gammar - ONE )
          If ( pl .ge. pr ) Then
            sigma = gammal
          Else
            sigma = gammar
          End If ! pl .ge. pr
          zed = ( ( gammal - ONE ) / ( gammar - ONE ) ) * ( ar / al )     &        
               * ( ( pl / pr )**( HALF * ( sigma - ONE ) / sigma ) )
          ustar = ( ulsnk * zed + ursnk ) / ( ONE + zed )
        End If ! istate .ne. 5
!
!======================================================================
!.... (1) shock-contact-shock
!======================================================================
!
        If ( istate .eq. 1 ) Then
!
          If ( ldump ) Then
             Write(iun,940) 
             Write(iun,996) 
             Write(iun,997) 
          End If
!
          cl0 = gammal * pl / al
          cr0 = gammar * pr / ar
!
 110      Continue
!
          itnum = itnum + 1
          If ( itnum .gt. itmax ) ierr = 10
          If ( ierr .ne. 0 ) goto 799
!
!.... Leftward moving shock conditions  G&G eqs. 3.33-3.35
!
          wl = HALF * HALF * ( gammal + ONE ) * ( ustar - ul ) / al     &       
               - Sqrt( ONE      &     
               + ( HALF * HALF * ( gammal + ONE ) * ( ustar - ul ) / al )**2)
          plst   = pl + cl0 * ( ustar - ul ) * wl
          plstpr = TWO * cl0 * ( wl**3 ) / ( ONE + ( wl**2 ) )
!
!.... Rightward moving shock conditions  G&G eqs. 3.37-3.39
!
          wr = HALF * HALF * ( gammar + ONE ) * ( ustar - ur ) / ar     &      
               + Sqrt( ONE      &     
               + ( HALF * HALF * ( gammar + ONE ) * ( ustar - ur ) / ar )**2 )
          prst = pr + cr0 * ( ustar - ur ) * wr
          prstpr = TWO * cr0 * ( wr**3 ) / ( ONE + ( wr**2 ) )
!
!.... Check for convergence
!
          err1 = ONE - ( plst / prst )
          err2 = ( plst - prst ) / ( plstpr - prstpr )
          If ( ldump ) Then
             Write(iun,998) itnum, Abs( err1 ),  Abs( err2 ) ,    &                        
                  Abs( err1 ) + Abs( err2 )  
          End If
          If ( Abs( err1 ) .lt. tol .and. Abs( err2 ) .lt. tol )  Go To 199
!
          ustar = ustar - err2
          Go To 110
!
 199      Continue                           ! converged solution
!
          ustar = ustar - err2
          xcd   = xd + ustar * time
!
!.... Compute state between left shock & contact  G&G eq. 3.36
!
          alst = al * Sqrt(      &      
               ( ( gammal + ONE ) + ( gammal - ONE ) * plst / pl   )     &    
               / ( ( gammal + ONE ) + ( gammal - ONE ) * pl   / plst ) )
          rlst = gammal * plst / alst / alst
          elst = plst / ( gammal - ONE ) / rlst 
          vl   = ul + al * wl                ! velocity of leftward shock
          xlshk = xd + vl * time             ! location of leftward shock
!
!.... Compute contact location
!
          xlcnt = xd + ustar * time          ! location of contact
!
!.... Compute state between contact & right shock  G&G eq. 3.40
!
          arst = ar * Sqrt(      &      
               ( ( gammar + ONE ) + ( gammar - ONE ) * prst / pr   )     &    
               / ( ( gammar + ONE ) + ( gammar - ONE ) * pr   / prst ) )
          rrst = gammar * prst / arst / arst
          erst = prst / ( gammar - ONE ) / rrst 
          vr   = ur + ar * wr                ! velocity of rightward shock
          xrshk = xd + vr * time             ! location of rightward shock
!
!.... Write data
!
          If ( ldump ) Then
             Write(iun,989) ustar
             Write(iun,990) 
             Write(iun,991) 
             Write(iun,992) cshock,vl
             Write(iun,992) ccd,ustar
             Write(iun,992) cshock,vr
             Write(iun,983) xlshk, xlcnt, xrshk
          End If
!
!.... Assign the values to the given positions
!
! get rid of the i-loop and select x_point
!
!
!*!          Do i = 1, nx
!*!            x = xval(i)
            x = xi
!
!.... To the left of the leftward shock
!
            If ( x .lt. xlshk ) Then
!
              ui    = ul
              presi = pl
              rhoi  = rl
              ei    = el
              ci    = Sqrt( gammal * presi / rhoi )
              si    = Log( ei      &                   
                   * ( ONE / rhoi )**( gammal - ONE ) )
!c              uout(i) = ul
!c              pout(i) = pl
!c              rout(i) = rl
!c              eout(i) = el
!c              cout(i) = Sqrt( gammal * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammal - ONE ) )
!
!.... Region between leftward shock and contact
!
            Else If ( x .ge. xlshk .and. x .lt. xlcnt ) Then
!
              ui    = ustar
              presi = plst
              rhoi  = rlst
              ei    = elst
              ci    = Sqrt( gammal * presi / rhoi )
              si    = Log( ei     &                   
                   * ( ONE / rhoi )**( gammal - ONE ) )
!c              uout(i) = ustar
!c              pout(i) = plst
!c              rout(i) = rlst
!c              eout(i) = elst
!c              cout(i) = Sqrt( gammal * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammal - ONE ) )
!
!.... Region between contact and rightward shock
!
            Else If ( x .ge. xlcnt .and. x .lt. xrshk ) Then
!
              ui    = ustar
              presi = prst
              rhoi  = rrst
              ei    = erst
              ci    = Sqrt( gammal * presi / rhoi )
              si    = Log( eout(i)     &                   
                   * ( ONE / rhoi )**( gammar - ONE ) )
!c              uout(i) = ustar
!c              pout(i) = prst
!c              rout(i) = rrst
!c              eout(i) = erst
!c              cout(i) = Sqrt( gammal * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammar - ONE ) )

!
!.... To the right of the rightward shock
!
            Else
!
              ui    = ur
              presi = pr
              rhoi  = rr
              ei    = er
              ci    = Sqrt( gammal * presi / rhoi )
              si    = Log( eout(i)     &                   
                   * ( ONE / rhoi )**( gammar - ONE ) )
!c              uout(i) = ur
!c              pout(i) = pr
!c              rout(i) = rr
!c              eout(i) = er
!c              cout(i) = Sqrt( gammal * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammar - ONE ) )
!
            End If ! x .lt. xlshk
!*!          End Do ! i
!
!======================================================================
!.... (2) shock-contact-rarefation
!======================================================================
!
        Else If ( istate .eq. 2 ) Then
!
          If ( ldump ) Then
             Write(iun,950) 
             Write(iun,996) 
             Write(iun,997) 
          End If
!
          cl0 = gammal * pl / al
!
 210      Continue
!
          itnum = itnum + 1
          If ( itnum .gt. itmax ) ierr = 10
          If ( ierr .ne. 0 ) Go To 799
!
!.... Leftward moving shock conditions  G&G eqs. 3.33-3.35
!
          wl = HALF * HALF * ( gammal + ONE ) * ( ustar - ul ) / al     &         
               - Sqrt( ONE      &     
               + ( HALF * HALF * ( gammal + ONE ) * ( ustar - ul ) / al )**2 )
          plst = pl + cl0 * ( ustar - ul ) * wl
          plstpr = TWO * cl0 * ( wl**3 ) / ( ONE + ( wl**2 ) )
!
!.... Rightward moving rarefaction conditions  G&G eqs. 3.44-3.46
!
          arst = ar + HALF * ( gammar - ONE ) * ( ustar - ur )
          prst = pr      &         
               * ( ( arst / ar )**( TWO * gammar / ( gammar - ONE ) ) )
          prstpr = gammar * prst / arst
!
!.... Check for convergence
!
          err1 = ONE - ( plst / prst )
          err2 = ( plst - prst ) / ( plstpr - prstpr )
          If ( ldump ) Then
             Write(iun,998) itnum, Abs( err1 ),  Abs( err2 ) ,     &                       
                  Abs( err1 ) + Abs( err2 )  
          End If
          If ( Abs( err1 ) .lt. tol .and. Abs( err2 ) .lt. tol )  Go To 299
!
          ustar = ustar - err2
          Go To 210
!
 299      Continue                           !  converged solution
!
          ustar = ustar - err2
!
!.... Compute state between left shock & contact  G&G eq. 3.36
!
          alst = al * Sqrt(      &      
               ( ( gammal + ONE ) + ( gammal - ONE ) * plst / pl   )     &    
               / ( ( gammal + ONE ) + ( gammal - ONE ) * pl   / plst ) )
          rlst = gammal * plst / alst / alst
          elst = plst / ( gammal - ONE ) / rlst
          vl = ul + al * wl 
          xlshk = xd + vl * time             ! location of shock
!
!.... Compute contact location
!
          xlcnt = xd + ustar * time          ! location of contact
!
!.... Compute state between contact & right rarefaction tail
!
          rrst = gammar * prst / arst / arst
          erst = prst / ( gammar - ONE ) / rrst 
!
!.... Compute boundary of right rarefaction tail & head 
!
          vrrhead = ur    + ar               ! velocity of head of rarefaction
          vrrtail = ustar + arst             ! velocity of tail of rarefaction
          xrrhead = xd + vrrhead * time      ! location of head of rarefaction
          xrrtail = xd + vrrtail * time      ! location of tail of rarefaction
!
!.... Write data
!
          If ( ldump ) Then
             Write(iun,989) ustar
             Write(iun,990) 
             Write(iun,991) 
             Write(iun,992) cshock,vl
             Write(iun,992) ccd,ustar
             Write(iun,992) crtail,vrrtail
             Write(iun,992) crhead,vrrhead
             Write(iun,984) xlshk, xlcnt, xrrtail, xrrhead
          End If
!
!.... Assign the values to the given positions
!
!*!          Do i = 1, nx
!*!            x = xval(i)
            x = xi
!
!.... To the left of the leftward shock
!
            If ( x .lt. xlshk ) Then
!
              ui    = ul
              presi = pl
              rhoi  = rl
              ei    = el
              ci    = Sqrt( gammal * presi / rhoi )
              si    = Log( ei     &                   
                   * ( ONE / rhoi )**( gammal - ONE ) )
!c              uout(i) = ul
!c              pout(i) = pl
!c              rout(i) = rl
!c              eout(i) = el
!c              cout(i) = Sqrt( gammal * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammal - ONE ) )
!
!.... Compressed region between leftward shock and contact
!
            Else If ( x .ge. xlshk .and. x .lt. xlcnt ) Then
!
              ui    = ustar
              presi = plst
              rhoi  = rlst
              ei    = elst
              ci    = Sqrt( gammal * presi / rhoi )
              si    = Log( ei     &                   
                   * ( ONE / rhoi )**( gammal - ONE ) )
!c              uout(i) = ustar
!c              pout(i) = plst
!c              rout(i) = rlst
!c              eout(i) = elst
!c              cout(i) = Sqrt( gammal * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammal - ONE ) )
!
!.... Expanded region between contact and rightward rarefaction
!
            Else If ( x .ge. xlcnt .and. x .lt. xrrtail ) Then
!
              ui    = ustar
              presi = prst
              rhoi  = rrst
              ei    = erst
              ci    = Sqrt( gammar * presi / rhoi )
              si    = Log( ei     &                   
                   * ( ONE / rhoi )**( gammar - ONE ) )
!c              uout(i) = ustar
!c              pout(i) = prst
!c              rout(i) = rrst
!c              eout(i) = erst
!c              cout(i) = Sqrt( gammar * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammar - ONE ) )
!
!.... Within the rightward rarefaction
!
            Else If ( x .ge. xrrtail .and. x .lt. xrrhead ) Then
!
              gampr = HALF * ( gammar - ONE ) / ar
              xprime = ( x - xd ) / ( ar * time )
              urar   = ( TWO * ar / ( gammar + ONE ) )      &               
                   * ( xprime - ONE + gampr * ur )
              arar = ar + HALF * ( gammar - ONE ) * ( urar - ur )
              rrar = rr * ( ONE + gampr     &             
                   * ( urar - ur ) )**( TWO / ( gammar - ONE ) )
              prar = pr * ( ONE + gampr     &          
                   * ( urar - ur ) )**( ( TWO * gammar ) /      &                               
                   ( gammar - ONE ) )
              erar = ( prar / rrar) / ( gammar - ONE )
              ui    = urar
              presi = prar
              rhoi  = rrar
              ei    = erar
              ci    = Sqrt( gammar * presi / rhoi )
              si    = Log( ei     &                   
                   * ( ONE / rhoi )**( gammar - ONE ) )
!c              uout(i) = urar
!c              pout(i) = prar
!c              rout(i) = rrar
!c              eout(i) = erar
!c              cout(i) = Sqrt( gammar * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammar - ONE ) )
!
!.... To the right of the rightward rarefaction
!
            Else
!
              ui    = ur
              presi = pr
              rhoi  = rr
              ei    = er
              ci    = Sqrt( gammar * presi / rhoi )
              si    = Log( ei     &                   
                   * ( ONE / rhoi )**( gammar - ONE ) )
!c              uout(i) = ur
!c              pout(i) = pr
!c              rout(i) = rr
!c              eout(i) = er
!c              cout(i) = Sqrt( gammar * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammar - ONE ) )
!
            End If ! x .lt. xlshk
!*!          End Do ! i
!
!======================================================================
!.... (3) rarefaction-contact-rarefaction
!======================================================================
!
        Else If ( istate .eq. 3 ) Then
!
          If ( ldump ) Then
             Write(iun,960) 
             Write(iun,996) 
             Write(iun,997) 
          End If
!
 310      Continue
!
          itnum = itnum + 1
          If ( itnum .gt. itmax ) ierr = 10
          If ( ierr .ne. 0 ) Go To 799
!
!.... Leftward moving rarefaction conditions  G&G eqs. 3.41-3.43
!
          alst = al - HALF * ( gammal - ONE ) * ( ustar - ul )
          plst = pl      &         
               * ( ( alst / al )**( TWO * gammal / ( gammal - ONE ) ) )
          plstpr = -gammal * plst / alst
!
!.... Rightward moving rarefaction conditions  G&G eqs. 3.44-3.46
!
          arst = ar + HALF * ( gammar - ONE ) * ( ustar - ur )
          prst = pr      &         
               * ( ( arst / ar )**( TWO * gammar / ( gammar - ONE ) ) )
          prstpr = gammar * prst / arst
!
!.... Check for convergence
!
          err1 = ONE - ( plst / prst )
          err2 = ( plst - prst ) / ( plstpr - prstpr )
          If ( ldump ) Then
             Write(iun,998) itnum, Abs( err1 ),  Abs( err2 ) ,     &
                  Abs( err1 ) + Abs( err2 )  
          End If
          If ( Abs( err1 ) .lt. tol .and. Abs( err2 ) .lt. tol ) Go To 399
!
          ustar = ustar - err2
          Go To 310
!
 399      Continue                           ! converged solution
!
          ustar = ustar - err2
!
!.... Compute boundary of left rarefaction head & tail
!
          vlrhead = ul    - al               ! velocity of head of rarefaction
          vlrtail = ustar - alst             ! velocity of tail of rarefaction
          xlrhead = xd + vlrhead * time      ! location of head of rarefaction
          xlrtail = xd + vlrtail * time      ! location of tail of rarefaction
!
!.... Compute state between left rarefaction tail & contact
!
          rlst = gammal * plst / alst / alst
          elst = plst / ( gammal - ONE ) / rlst 
!
!.... Compute contact location
!
          xlcnt = xd + ustar * time          ! location of contact
!
!.... Compute state between contact & right rarefaction tail

          rrst = gammar * prst / arst / arst
          erst = prst / ( gammar - ONE ) / rrst 
!
!.... Compute boundary of right rarefaction tail & head 
!
          vrrhead = ur    + ar               ! velocity of head of rarefaction
          vrrtail = ustar + arst             ! velocity of tail of rarefaction
          xrrhead = xd + vrrhead * time      ! location of head of rarefaction
          xrrtail = xd + vrrtail * time      ! location of tail of rarefaction
!
!.... Write data
!
          If ( ldump ) Then
             Write(iun,989) ustar
             Write(iun,990) 
             Write(iun,991) 
             Write(iun,992) crhead,vlrhead
             Write(iun,992) crtail,vlrtail
             Write(iun,992) ccd,ustar
             Write(iun,992) crtail,vrrtail
             Write(iun,992) crhead,vrrhead
             Write(iun,985) xlrhead, xlrtail, xlcnt, xrrtail, xrrhead
          End If
!
!.... Assign the values to the given positions
!
!*!          Do i = 1, nx
!*!            x = xval(i)
            x = xi
!
!.... To the left of the leftward rarefction
!
            If ( x .lt. xlrhead ) Then
!
              ui    = ul
              presi = pl
              rhoi  = rl
              ei    = el
              ci    = Sqrt( gammal * presi / rhoi )
              si    = Log( ei * ( ONE / rhoi )**( gammal - ONE ) )
!c              uout(i) = ul
!c              pout(i) = pl
!c              rout(i) = rl
!c              eout(i) = el
!c              cout(i) = Sqrt( gammal * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammal - ONE ) )
!
!.... Within the leftward rarefaction
!
            Else If ( x .ge. xlrhead .and. x .lt. xlrtail ) Then
!
              gampr = HALF * ( gammal - ONE ) / al
              xprime = ( x - xd ) / ( al * time )
              urar   = ( TWO * al / ( gammal + ONE ) )      &               
                   * ( xprime + ONE + gampr * ul )
              arar = al + HALF * ( gammal - ONE ) * ( ul - urar )
              rrar = rl * ( ONE - gampr     &             
                   * ( urar - ul ) )**( TWO / ( gammal - ONE ) )
              prar = pl * ( ONE - gampr     &          
                   * ( urar - ul ) )**( ( TWO * gammal ) /      &                               
                   ( gammal - ONE ) )
              erar = ( prar / rrar) / ( gammal - ONE )
              ui    = urar
              presi = prar
              rhoi  = rrar
              ei    = erar
              ci    = Sqrt( gammal * presi / rhoi )
              si    = Log( eout(i)     &                   
                   * ( ONE / rhoi )**( gammal - ONE ) )
!c              uout(i) = urar
!c              pout(i) = prar
!c              rout(i) = rrar
!c              eout(i) = erar
!c              cout(i) = Sqrt( gammal * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammal - ONE ) )
!
!.... Between the leftward rarefaction and contact
!
            Else If ( x .ge. xlrhead .and. x .lt. xlcnt ) Then
!
              ui    = ustar
              presi = plst
              rhoi  = rlst
              ei    = elst
              ci    = Sqrt( gammal * presi / rhoi )
              si    = Log( ei     &                   
                   * ( ONE / rhoi )**( gammal - ONE ) )
!c              uout(i) = ustar
!c              pout(i) = plst
!c              rout(i) = rlst
!c              eout(i) = elst
!c              cout(i) = Sqrt( gammal * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammal - ONE ) )
!
!.... Between the contact and rightward rarefaction
!
            Else If ( x .ge. xlcnt .and. x .lt. xrrtail ) Then
!
              ui    = ustar
              presi = prst
              rhoi  = rrst
              ei    = erst
              ci    = Sqrt( gammar * presi / rhoi )
              si    = Log( ei     &                   
                   * ( ONE / rhoi )**( gammar - ONE ) )
!c              uout(i) = ustar
!c              pout(i) = prst
!c              rout(i) = rrst
!c              eout(i) = erst
!c              cout(i) = Sqrt( gammar * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammar - ONE ) )
!
!.... Within the rightward rarefaction
!
            Else If ( x .ge. xrrtail .and. x .lt. xrrhead ) Then
!
              gampr = HALF * ( gammar - ONE ) / ar
              xprime = ( x - xd ) / ( ar * time )
              urar   = ( TWO * ar / ( gammar + ONE ) )      &               
                   * ( xprime - ONE + gampr * ur )
              arar = ar + HALF * ( gammar - ONE ) * ( urar - ur )
              rrar = rr * ( ONE + gampr     &             
                   * ( urar - ur ) )**( TWO / ( gammar - ONE ) )
              prar = pr * ( ONE + gampr     &          
                   * ( urar - ur ) )**( ( TWO * gammar ) /      &                               
                   ( gammar - ONE ) )
              erar = ( prar / rrar) / ( gammar - ONE )
              ui    = urar
              presi = prar
              rhoi  = rrar
              ei    = erar
              ci    = Sqrt( gammar * presi / rhoi )
              si    = Log( ei     &                   
                   * ( ONE / rhoi )**( gammar - ONE ) )
!c              uout(i) = urar
!c              pout(i) = prar
!c              rout(i) = rrar
!c              eout(i) = erar
!c              cout(i) = Sqrt( gammar * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammar - ONE ) )
!
!.... To the right of the rightward rarefaction
!
            Else
!
              ui    = ur
              presi = pr
              rhoi  = rr
              ei    = er
              ci    = Sqrt( gammar * presi / rhoi )
              si    = Log( ei     &                   
                   * ( ONE / rhoi )**( gammar - ONE ) )
!c              uout(i) = ur
!c              pout(i) = pr
!c              rout(i) = rr
!c              eout(i) = er
!c              cout(i) = Sqrt( gammar * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammar - ONE ) )
!
            End If ! x .lt. xlrhead 
!*!          End Do ! i
!
!======================================================================
!.... (4) rarefaction-contact-shock
!======================================================================
!
        Else If ( istate .eq. 4 ) Then
!
          If ( ldump ) Then
             Write(iun,970) 
             Write(iun,996) 
             Write(iun,997) 
          End If
!
          cr0 = gammar * pr / ar
!
 410      Continue
!
          itnum = itnum + 1
          If ( itnum .gt. itmax ) ierr = 10
          If ( ierr .ne. 0 ) Go To 799
!
!.... Leftward moving rarefaction conditions  G&G eqs. 3.41-3.43
!
          alst = al - HALF * ( gammal - ONE ) * ( ustar - ul )
          plst = pl      &         
               * ( ( alst / al )**( TWO * gammal / ( gammal - ONE ) ) )
          plstpr = -gammal * plst / alst
!
!.... Rightward moving shock conditions  G&G eqs. 3.37-3.39
!
          wr = HALF * HALF * ( gammar + ONE ) * ( ustar - ur ) / ar     &       
               + Sqrt( ONE      &     
               + ( HALF * HALF * ( gammar + ONE ) * ( ustar - ur ) / ar )**2 )
          prst = pr + cr0 * ( ustar - ur ) * wr
          prstpr = TWO * cr0 * ( wr**3 ) / ( ONE + ( wr**2 )   )
!
!.... Check for convergence
!
          err1 = ONE - ( plst / prst )
          err2 = ( plst - prst ) / ( plstpr - prstpr )
          If ( ldump ) Then
             Write(iun,998) itnum, Abs( err1 ),  Abs( err2 ) ,     &                        
                  Abs( err1 ) + Abs( err2 )  
          End If
          If ( Abs( err1 ) .lt. tol .and. Abs( err2 ) .lt. tol )  Go To 499
!
          ustar = ustar - err2
          Go To 410
!
 499      Continue                           ! converged solution
!
          ustar = ustar - err2
!
!.... Compute state between left rarefaction head & tail
!
          vlrhead = ul    - al               ! velocity of head of rarefaction
          vlrtail = ustar - alst             ! velocity of tail of rarefaction
          xlrhead = xd + vlrhead * time      ! location of head of rarefaction
          xlrtail = xd + vlrtail * time      ! location of tail of rarefaction
!
!.... Compute state between left rarefaction tail & contact
!
          rlst = gammal * plst / alst / alst
          elst = plst / ( gammal - one ) / rlst 
!
!.... Compute contact location
!
          xlcnt = xd + ustar * time          ! location of contact
!
!.... Compute state between contact & shock  G&G eq. 3.40
!
          arst = ar * Sqrt(      &      
               ( ( gammar + ONE ) + ( gammar - ONE ) * prst / pr   )     &    
               / ( ( gammar + ONE ) + ( gammar - ONE ) * pr   / prst ) )
          rrst = gammar * prst / arst / arst
          erst = prst / ( gammar - ONE ) / rrst 
          vr   = ur + ar * wr                ! velocity of shock
          xrshk = xd + vr * time             ! location of shock
!
!.... Write data
!
          If ( ldump ) Then
             Write(iun,989) ustar
             Write(iun,990) 
             Write(iun,991) 
             Write(iun,992) crhead,vlrhead
             Write(iun,992) crtail,vlrtail
             Write(iun,992) ccd,ustar
             Write(iun,992) cshock,vr
             Write(iun,986) xlrhead, xlrtail, xlcnt, xrshk
          End If
!
!.... Assign the values to the given positions
!
!*c          Do i = 1, nx
!*c            x = xval(i)
            x = xi
!
!.... To the left of the leftward rarefaction
!
            If ( x .lt. xlrhead ) Then
!
              ui    = ul
              presi = pl
              rhoi  = rl
              ei    = el
              ci    = Sqrt( gammal * presi / rhoi )
              si    = Log( ei * ( ONE / rhoi )**( gammal - ONE ) )
!c              uout(i) = ul
!c              pout(i) = pl
!c              rout(i) = rl
!c              eout(i) = el
!c              cout(i) = Sqrt( gammal * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammal - ONE ) )
!
!.... Within the leftward rarefaction
!
            Else If ( x .ge. xlrhead .and. x .lt. xlrtail ) Then
!
              gampr = HALF * ( gammal - ONE ) / al
              xprime = ( x - xd ) / ( al * time )
              urar   = ( TWO * al / ( gammal + ONE ) )      &               
                   * ( xprime + ONE + gampr * ul )
              arar = al + HALF * ( gammal - ONE ) * ( ul - urar )
              rrar = rl * ( ONE - gampr     &             
                   * ( urar - ul ) )**( TWO / ( gammal - ONE ) )
              prar = pl * ( ONE - gampr     &          
                   * ( urar - ul ) )**( ( TWO * gammal ) /      &                               
                   ( gammal - ONE ) )
              erar = ( prar / rrar) / ( gammal - ONE )
              ui    = urar
              presi = prar
              rhoi  = rrar
              ei    = erar
              ci    = Sqrt( gammal * presi / rhoi )
              si    = Log( ei * ( ONE / rhoi )**( gammal - ONE ) )
!c              uout(i) = urar
!c              pout(i) = prar
!c              rout(i) = rrar
!c              eout(i) = erar
!c              cout(i) = Sqrt( gammal * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammal - ONE ) )
!
!.... Expanded region between the leftward rarefaction and contact
!
            Else If ( x .ge. xlrtail .and. x .lt. xlcnt ) Then
!
              ui    = ustar
              presi = plst
              rhoi  = rlst
              ei    = elst
              ci    = Sqrt( gammal * presi / rhoi )
              si    = Log( ei * ( ONE / rhoi )**( gammal - ONE ) )
!c              uout(i) = ustar
!c              pout(i) = plst
!c              rout(i) = rlst
!c              eout(i) = elst
!c              cout(i) = Sqrt( gammal * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammal - ONE ) )
!
!.... Compressed region between the contact and rightward shock
!
            Else If ( x .ge. xlcnt .and. x .lt. xrshk ) Then
!
              ui    = ustar
              presi = prst
              rhoi  = rrst
              ei    = erst
              ci    = Sqrt( gammar * presi / rhoi )
              si    = Log( ei * ( ONE / rhoi )**( gammar - ONE ) )
!c              uout(i) = ustar
!c              pout(i) = prst
!c              rout(i) = rrst
!c              eout(i) = erst
!c              cout(i) = Sqrt( gammar * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammar - ONE ) )
!
!.... To the right of the rightward shock
!
            Else
!
              ui    = ur
              presi = pr
              rhoi  = rr
              ei    = er
              ci    = Sqrt( gammar * presi / rhoi )
              si    = Log( ei * ( ONE / rhoi )**( gammar - ONE ) )
!c              uout(i) = ur
!c              pout(i) = pr
!c              rout(i) = rr
!c              eout(i) = er
!c              cout(i) = Sqrt( gammar * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammar - ONE ) )
!
            End If ! x .lt. xlrhead
!*!          End Do ! i
!
!======================================================================
!.... (5) rarefaction-contact-vacuum-contact-rarefaction
!======================================================================
!
        Else
!
          If ( ldump ) Then
             Write(iun,980) 
             Write(iun,996) 
             Write(iun,997) 
          End If
!
!.... Solve for leftward contact
!
          ustarl = ZERO
!
 510      Continue
!
          itnum = itnum + 1
          If ( itnum .gt. itmax ) ierr = 10
          If ( ierr .ne. 0 ) Go To 799
!
!.... Leftward moving rarefaction conditions  G&G eqs. 3.41-3.43
!
          alst = al - HALF * ( gammal - ONE ) * ( ustarl - ul )
          plst = pl      &         
               * ( ( alst / al )**( TWO * gammal / ( gammal - ONE ) ) )
          plstpr = -gammal * plst / alst
!
          If ( ldebug .and. ldump ) Then      
            Write(iun,*) ' ul        =',ul
            Write(iun,*) ' ustarl    =',ustarl
            Write(iun,*) ' ustarl-ul =',ustarl-ul
            Write(iun,*) ' pl        =',pl
            Write(iun,*) ' al        =',al
            Write(iun,*) ' alst      =',alst
            Write(iun,*) ' plst      =',plst
            Write(iun,*) ' plstpr    =',plstpr
          End If ! ldebug
!
!.... Rightward vacuum condition
!
          arst = ZERO
          prst = ZERO
          prstpr = ZERO
!
!.... Check for convergence
!
          err1 = ONE
          err2 = ( plst - prst ) / ( plstpr - prstpr )
          If ( ldump ) Then
             Write(iun,998) itnum, Abs( err1 ),  Abs( err2 ), Abs( err2 )  
          End If
          If ( Abs( err2 ) .lt. tol ) Go To 549
!
          ustarl = ustarl - err2
          Go To 510
!
 549      Continue                           ! converged left rarefaction solution
!
          ustarl = ustarl - err2
!
!.... Compute boundary of left rarefaction head & tail
!
          vlrhead = ul     - al              ! velocity of head of rarefaction
          vlrtail = ustarl - alst            ! velocity of tail of rarefaction
          xlrhead = xd + vlrhead * time      ! location of head of rarefaction
          xlrtail = xd + vlrtail * time      ! location of tail of rarefaction
!
!.... Solve for rightward contact
!
          If ( ldump ) Then
             Write(iun,996) 
             Write(iun,997) 
          End If
!
          ustarr = urcvr
          itnum = 0
!
 550      Continue
!
          itnum = itnum + 1
          If ( itnum .gt. itmax ) ierr = 10
          If ( ierr .ne. 0 ) Go To 799
!
!.... Rightward moving rarefaction conditions  G&G eqs. 3.44-3.46
!
          arst = ar + HALF * ( gammar - ONE ) * ( ustarr - ur )
          prst = pr      &         
               * ( ( arst / ar )**( TWO * gammar / ( gammar - ONE ) ) )
          prstpr = gammar * prst / arst
!
!.... Leftward vacuum condition
!
          alst = ZERO
          plst = ZERO
          plstpr = ZERO
!
!.... Check for convergence
!
          err1 = ONE
          err2 = ( plst - prst ) / ( plstpr - prstpr )
          If ( ldump ) Then
             Write(iun,998) itnum, Abs( err1 ), Abs( err2 ), Abs( err2 )  
          End If
          If ( Abs( err2 ) .lt. tol ) Go To 599
!
          ustarr = ustarr - err2
          Go To 550
!
 599      Continue                           ! converged right rarefaction solution
!
          ustarr = ustarr - err2
!
!.... Compute boundary of right rarefaction tail & head 
!
          vrrhead = ur     + ar              ! velocity of head of rarefaction
          vrrtail = ustarr + arst            ! velocity of tail of rarefaction
          xrrhead = xd + vrrhead * time      ! location of head of rarefaction
          xrrtail = xd + vrrtail * time      ! location of tail of rarefaction
!
!.... Compute contact locations
!
          xlcnt = xd + ustarl * time         ! location of left  contact
          xrcnt = xd + ustarr * time         ! location of right contact
!
!.... Write data
!
          If ( ldump ) Then
             Write(iun,989) ustarl
             Write(iun,989) ustarr
             Write(iun,990) 
             Write(iun,991) 
             Write(iun,992) crhead,vlrhead
             Write(iun,992) crtail,vlrtail
             Write(iun,992) ccd,ustarl
             Write(iun,992) cvac,zero
             Write(iun,992) ccd,ustarr
             Write(iun,992) crtail,vrrtail
             Write(iun,992) crhead,vrrhead
             Write(iun,987) xlrhead, xlrtail, xlcnt, xrcnt, xrrtail, xrrhead
          End If
!
!.... Assign the values to the given positions
!
!*!          Do i = 1, nx
!*!            x = xval(i)
            x = xi
!
!.... To the left of the leftward rarefaction
!
            If ( x .lt. xlrhead ) Then
!
              ui    = ul
              presi = pl
              rhoi  = rl
              ei    = el
              ci    = Sqrt( gammal * presi / rhoi )
              si    = Log( ei * ( ONE / rhoi )**( gammal - ONE ) )
!c              uout(i) = ul
!c              pout(i) = pl
!c              rout(i) = rl
!c              eout(i) = el
!c              cout(i) = Sqrt( gammal * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammal - ONE ) )
!
!.... Within the leftward rarefaction
!
            Else If ( x .ge. xlrhead .and. x .lt. xlrtail ) Then
!
              gampr = HALF * ( gammal - ONE ) / al
              xprime = ( x - xd ) / ( al * time )
              urar   = ( TWO * al / ( gammal + ONE ) )      &               
                   * ( xprime + ONE + gampr * ul )
              arar = al + HALF * ( gammal - ONE ) * ( ul - urar )
              rrar = rl * ( ONE - gampr     &             
                   * ( urar - ul ) )**( TWO / ( gammal - ONE ) )
              prar = pl * ( ONE - gampr     &          
                   * ( urar - ul ) )**( ( TWO * gammal ) /      &                               
                   ( gammal - ONE ) )
              erar = ( prar / rrar) / ( gammal - ONE )
              ui    = urar
              presi = prar
              rhoi  = rrar
              ei    = erar
              ci    = Sqrt( gammal * presi / rhoi )
              si    = Log( ei * ( ONE / rhoi )**( gammal - ONE ) )
!c              uout(i) = urar
!c              pout(i) = prar
!c              rout(i) = rrar
!c              eout(i) = erar
!c              cout(i) = Sqrt( gammal * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammal - ONE ) )
!
!.... Expanded region between the leftward rarefaction and leftward contact
!
            Else If ( x .ge. xlrtail .and. x .lt. xlcnt ) Then
!
              ui    = ustarl
              presi = plst
              rhoi  = rlst
              ei    = elst
              ci    = Sqrt( gammal * presi / rhoi )
              si    = Log( ei * ( ONE / rhoi )**( gammal - ONE ) )
!c              uout(i) = ustarl
!c              pout(i) = plst
!c              rout(i) = rlst
!c              eout(i) = elst
!c              cout(i) = Sqrt( gammal * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammal - ONE ) )
!
!.... Vacuum region between the leftward contact and rightward contact
!
            Else If ( x .ge. xlcnt .and. x .lt. xrcnt ) Then
!
              ui    = ustarl + ( ustarr - ustarl )     &               
                   * ( x - xlcnt ) / ( xrcnt - xlcnt )
              presi = ZERO
              rhoi  = ZERO
              ei    = ZERO
              ci    = ZERO
              si    = ZERO
!c              uout(i) = ustarl + ( ustarr - ustarl )
!c     &                      * ( x - xlcnt ) / ( xrcnt - xlcnt )
!c              pout(i) = ZERO
!c              rout(i) = ZERO
!c              eout(i) = ZERO
!c              cout(i) = ZERO
!c              sout(i) = ZERO
!
!.... Expanded region between the rightward contact and rightward rarefaction
!
            Else If ( x .ge. xrcnt .and. x .lt. xrrtail ) Then
!
              ui    = ustarr
              presi = prst
              rhoi  = rrst
              ei    = erst
              ci    = Sqrt( gammar * presi / rhoi )
              si    = Log( ei * ( ONE / rhoi )**( gammar - ONE ) )
!c              uout(i) = ustarr
!c              pout(i) = prst
!c              rout(i) = rrst
!c              eout(i) = erst
!c              cout(i) = Sqrt( gammar * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammar - ONE ) )
!
!.... Within the rightward rarefaction
!
            Else If ( x .ge. xrrtail .and. x .lt. xrrhead ) Then
!
              gampr = HALF * ( gammar - ONE ) / ar
              xprime = ( x - xd ) / ( ar * time )
              urar   = ( TWO * ar / ( gammar + ONE ) )      &               
                   * ( xprime - ONE + gampr * ur )
              arar = ar + HALF * ( gammar - ONE ) * ( urar - ur )
              rrar = rr * ( ONE + gampr * ( urar - ur ) )**( TWO / ( gammar - ONE ) )
              prar = pr * ( ONE + gampr     &          
                   * ( urar - ur ) )**( ( TWO * gammar ) /      &                               
                   ( gammar - ONE ) )
              erar = ( prar / rrar) / ( gammar - ONE )
              ui    = urar
              presi = prar
              rhoi  = rrar
              ei    = erar
              ci    = Sqrt( gammar * presi / rhoi )
              si    = Log( ei * ( ONE / rhoi )**( gammar - ONE ) )
!c              uout(i) = urar
!c              pout(i) = prar
!c              rout(i) = rrar
!c              eout(i) = erar
!c              cout(i) = Sqrt( gammar * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammar - ONE ) )
!
!.... To the right of the rightward rarefaction
!
            Else
!
              ui    = ur
              presi = pr
              rhoi  = rr
              ei    = er
              ci    = Sqrt( gammar * presi / rhoi )
              si    = Log( ei * ( ONE / rhoi )**( gammar - ONE ) )
!c              uout(i) = ur
!c              pout(i) = pr
!c              rout(i) = rr
!c              eout(i) = er
!c              cout(i) = Sqrt( gammar * pout(i) / rout(i) )
!c              sout(i) = Log( eout(i)
!c     &                   * ( ONE / rout(i) )**( gammar - ONE ) )
!
            End If ! x .lt. xlrhead
!*c          End Do ! i
        End If ! istate
      End If ! pr .eq. ZERO
      If ( ierr .ne. 0 ) Go To 799
!
!.... Write data to ASCII file
!
!c      Open( unit = iun+1, file = 'shktub.dat', status='new')
!c      Write(iun+1, 101)
!c        Write(iun+1, 102)
!c      Do i = 1, nx
!c        Write(iun+1, 103) xval(i), uout(i), pout(i), rout(i), eout(i),
!c     &                    cout(i), sout(i), time
!c      End Do ! i
!c      Close( iun+1 )
!
!.... Error conditions
!
  799 Continue
      If ( ierr .eq. 0 ) Go To 899 
      If ( ldump) Write(iun,900) ierr
      Go To ( 801, 802, 803, 804, 805, 806, 807, 808, 809, 810 ) ierr
  801 If ( ldump) Write(iun,901)  
      Go To 899 
  802 If ( ldump) Write(iun,902)  
      Go To 899 
  803 If ( ldump) Write(iun,903)  
      Go To 899 
  804 If ( ldump) Write(iun,904)  
      Go To 899 
  805 If ( ldump) Write(iun,905)  
      Go To 899 
  806 If ( ldump) Write(iun,906)  
      Go To 899 
  807 If ( ldump) Write(iun,907)  
      Go To 899 
  808 If ( ldump) Write(iun,908)  
      Go To 899 
  809 If ( ldump) Write(iun,909) nx
      Go To 899 
  810 If ( ldump) Write(iun,910)  
      Go To 899 
!
!.... Exit
!
  899 Continue
      If ( ldump ) Then
         Close( iun )
      End If
!
!-----------------------------------------------------------------------
!
!.... Format statements
!
  900 Format('** SHKTUB: FATAL ERROR  ierr = ',i2,' **')
  901 Format('** Equal pressures & equal velocities **')
  902 Format('** Equal zero pressures **')
  903 Format('** Zero right pressure & nonzero right velocity **')
  904 Format('** Zero left pressure & nonzero left velocity **')
  905 Format('** Right gamma .le. one **')
  906 Format('** Left gamma .le. one **')
  907 Format('** Right density .le. zero **')
  908 Format('** Left density .le. zero **')
  909 Format('** Increase nxmax to > ',i4,' and recompile **')
  910 Format('** Max iterations exceeded **')
!
  920 Format('Expansion into right zero-pressure region')
  930 Format('Expansion into left zero-pressure region')
  940 Format('State 1:  shock-contact-shock')
  950 Format('State 2:  shock-contact-rarefation')
  960 Format('State 3:  rarefaction-contact-rarefaction')
  970 Format('State 4:  rarefaction-contact-shock')
  980 Format('State 5:  rarefaction-contact-vacuum-contact-rarefaction')
!
  981 Format(/' Rarefaction Head: ',1pe12.5,     &
      /' Rarefaction Tail: ',1pe12.5,            &       
      /' Left Contact:     ',1pe12.5)
  982 Format(/' Right Contact:    ',1pe12.5     &       
           /' Rarefaction Tail: ',1pe12.5,      &       
           /' Rarefaction Head: ',1pe12.5)
  983 Format(/' Left Shock:   ',1pe12.5,     &       
           /' Contact:      ',1pe12.5,       &       
           /' Right Shock:  ',1pe12.5)
  984 Format(/' Left Shock:       ',1pe12.5,     &       
           /' Contact:          ',1pe12.5,       &       
           /' Rarefaction Tail: ',1pe12.5,       &       
           /' Rarefaction Head: ',1pe12.5)
  985 Format(/' Left Rarefaction Head:  ',1pe12.5,     &       
           /' Left Rarefaction Tail:  ',1pe12.5,       &       
           /' Contact:                ',1pe12.5,       &       
           /' Right Rarefaction Tail: ',1pe12.5,       &       
           /' Right Rarefaction Head: ',1pe12.5)
  986 Format(/' Rarefaction Head: ',1pe12.5,     &       
           /' Rarefaction Tail: ',1pe12.5,       &       
           /' Contact:          ',1pe12.5,       &       
           /' Right Shock:      ',1pe12.5)
  987 Format(/' Left Rarefaction Head:  ',1pe12.5,     &       
           /' Left Rarefaction Tail:  ',1pe12.5,       &       
           /' Left Contact:           ',1pe12.5,       &       
           /' Right Contact:          ',1pe12.5,       &       
           /' Right Rarefaction Tail: ',1pe12.5,       &       
           /' Right Rarefaction Head: ',1pe12.5)
!
  989 Format(/' ustar = ',1pe12.5)
  990 Format(/' Wave      Speed')
  991 Format(' ----      -----')
  992 Format(1x,a5,2x,1pe12.5)
  993 Format(/' State    Location     Velocity     Pressure     Density'  ,   &
      '       SIE         Sndspd')
  994 Format(' -----    --------     --------     --------     -------'   ,  &
      '       ---         ------')
  995 Format(1x,a5,1x,6(1x,1pe12.5))
  996 Format(/' itnum     err 1         err 2      tot err')
  997 Format(' -----     -----         -----      -------')
  998 Format(2x,i3,2x,3(1x,1pe12.5))
!
  101 Format('  Location      Velocity     Pressure     Density'     , &
      '       SIE         Sndspd      Entropy      Time')
  102 Format(/'  --------      --------     --------     -------'    , &
      '       ---         ------      -------      ----')
  103 Format(8(1x,1pe12.5))
!
!-----------------------------------------------------------------------
!
      Return
      End
!
! End of Subroutine SHKTUB
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

