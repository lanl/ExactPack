      Subroutine riemann_kamm_jwl(time, nx, x, xd0,      &
           rhol, pl, ul, rhor, pr, ur,                   &
           rho0l, sie0l, gammal, bigal, bigbl, r1l, r2l, &
           rho0r, sie0r, gammar, bigar, bigbr, r1r, r2r, &
           rho, p, u, sound, sie, entropy)
      Implicit None
!f2py intent(out)      :: rho, p, u, sie, sound, entropy
!f2py intent(hide)     :: nx
!f2py integer          :: nx
!f2py double           :: time, x(nx), xd0
!f2py double           :: rhol, pl, ul, rhor, pr, ur
!f2py double           :: rho(nx), p(nx), u(nx), sie(nx), sound(nx)
!f2py double           :: rho0l, sie0l, gammal, bigal, bigbl, r1l, r2l
!f2py double           :: rho0r, sie0r, gammar, bigar, bigbr, r1r, r2r
!f2py double           :: entropy(nx)
      Integer          :: nx, nrar
      Double Precision :: rho(nx), p(nx), u(nx), sound(nx), sie(nx)
      Double Precision :: entropy(nx), x(nx)
      Double Precision :: time, xd0
      Double Precision :: rhol, pl, ul, rhor, pr, ur
      Double Precision :: rho0l, sie0l, gammal, bigal, bigbl, r1l, r2l
      Double Precision :: rho0r, sie0r, gammar, bigar, bigbr, r1r, r2r

! local variables
      Double Precision :: xmin, xmax
      Integer          :: ierr, it
      Logical          :: ldebug, lwrite

      nrar = 1000 ! number of points in the rarefaction wave
      xmin = x(1)
      xmax = x(nx)
      ldebug = .false.
      lwrite = .false.
      CALL  RIEMANN( rhol, rhor, pl, pr, ul, ur, nx, xmin,         &
                     xmax, xd0, nrar, time, ldebug,                &
                     rho0l, sie0l, gammal, bigal, bigbl, r1l, r2l, &
                     rho0r, sie0r, gammar, bigar, bigbr, r1r, r2r, &
                     x, rho, u, p, sie, sound, lwrite,             &
                     ierr )

      End Subroutine riemann_kamm_jwl

!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
! Start of Subroutine RIEMANN
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!                                                                      c
! Solve the Riemann problem for a general EOS using a hybrid of the    c
! method of Colella & Glaz (1985) and Banks (2010), using also         c
! information from Fryxell et al. (2000), Toro (1999) and              c
! Shyue (2001).  This includes using an estimate of the wave structure c
! (rarefaction or contact) in the intermediate solution, and solving   c
! for the exact (nonlinear) structure in the rarefaction.              c
!                                                                      c
!   Called by: DOIT      Calls: EOS_LEFT_RHOP   - left  EOS evaluation c
!                               EOS_RIGHT_RHOP  - right EOS evaluation c
!                               GET_USTAR_LEFT  - left  u*  evaluation c
!                               GET_USTAR_RIGHT - right u*  evaluation c
!                               RARELEFTSOLN    - left  rarefaction    c
!                               RARERIGHTSOLN   - right rarefaction    c
!                               SPLINE          - spline coefficients  c
!                               SPLEVAL         - spline evaluation    c
!                               RIEMANN_ERROR   - write error info     c
!                                                                      c
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!  The Riemann problem for 1D compressible flow equations leads to
!  four states, separated by the three characteristics (u - cs, u, u + cs):
!
!
!        l_1      t    l_2       l_3
!         \       ^   .       /
!          \  *L  |   . *R   /
!           \     |  .     /
!            \    |  .    /
!        L    \   | .   /    R
!              \  | .  /
!               \ |. / 
!                \|./
!       ----------+----------------> x
!
!       l_1 = u - cs   eigenvalue
!       l_2 = u        eigenvalue (contact)
!       l_3 = u + cs   eigenvalue
!
!       density and SIE only jump across l_2
!
!  References:
!
!   CG:   Colella & Glaz 1985, JCP, 59, 264.
!
!   Fry:  Fryxell et al. 2000, ApJS, 131, 273.
!
!   Toro: Toro 1999, ``Riemann Solvers and Numerical Methods for Fluid
!         Dynamcs: A Practical Introduction, 2nd Ed.'', Springer-Verlag
!
!   Banks: Banks 2010, "On Exact Conservation for the Euler Equations 
!         with Complex Equations of State," Commun. Comput. Phys., 8(5), 
!         pp. 995-1015 (2010), doi: 10.4208/cicp/090909/100310a
!
!   Shyue: Shyue 2001, "A Fluid-Mixture Type Algorithm for Compressible 
!         Multicomponent Flow with Mie-Grueneisen Equation of State," 
!         J. Comput. Phys., 171, pp. 678-707 (2001), doi: 10.1006/jcph.2001.6801
!
!  This implementation is loosely based on the Riemann solver distributed with the
!  FLASH Code (see Fry), but is greatly extended to work with a general convex EOS.
!
      Subroutine RIEMANN( rhol, rhor, prsl, prsr, ul, ur, nx, xmin,    &
                 xmax, xd0, nrar, time, ldebugin,                      &
                 xrho0l, xsie0l, xgamma0l, xbigal, xbigbl, xr1l, xr2l, &
                 xrho0r, xsie0r, xgamma0r, xbigar, xbigbr, xr1r, xr2r, &
                 x_m, rho_m, u_m, p_m, sie_m, sound_m, lwrite,         &
                 ierr )
!
      Implicit none


!
!.... Include files
!
      Include "param.h"
!
!.... Call list variables
!
      Integer :: ierr                     ! error flag
      Integer :: nx                       ! nx+1 = # of x-positions
      Integer :: nrar                     ! number of pts in rarefaction
!
      Double Precision :: rhol            ! initial left  density    (g/cm3)
      Double Precision :: rhor            ! initial right density    (g/cm3)
      Double Precision :: prsl            ! initial left  pressure   (dyn/cm2)
      Double Precision :: prsr            ! initial right pressure   (dyn/cm2)
      Double Precision :: ul              ! initial left  velocity   (cm/s)
      Double Precision :: ur              ! initial right velocity   (cm/s)
      Double Precision :: xd0             ! x-location of diaphragm  (cm)
      Double Precision :: xmin            ! minimum x-edge location  (cm)
      Double Precision :: xmax            ! maximum x-edge location  (cm)
      Double Precision :: time            ! simulation time          (s)
      Double Precision :: xrho0l, xsie0l, xgamma0l, xbigal, xbigbl ! JWL EOS params
      Double Precision :: xrho0r, xsie0r, xgamma0r, xbigar, xbigbr ! JWL EOS params
      Double Precision :: xr1l, xr2l                               ! JWL EOS params
      Double Precision :: xr1r, xr2r                               ! JWL EOS params
      Double Precision :: x_m(1:nx), rho_m(1:nx), u_m(1:nx), p_m(1:nx)
      Double Precision :: sie_m(1:nx), sound_m(1:nx)

!
      Logical :: ldebugin                 ! debug flag
      Logical :: lwrite                   ! write output riemjwl.dat
!
!.... Local variables
!
      Integer :: i                        ! index
      Integer :: it                       ! secant iteration number
      Integer :: itmax                    ! max number of secant iterations
      Integer :: iun                      ! log    file unit number 'riemjwl.log'
      Integer :: iunm1                    ! debug  file unit number 'riemjwl.dbg'
      Integer :: iunp1                    ! result file unit number 'riemjwl.dat'
!
      Double Precision :: xval(1:nx)      ! x-positions for solution (cm)
      Double Precision :: x               ! local x-position (cm)
      Double Precision :: xprev           ! local x-position, previous index (cm)
      Double Precision :: deltax          ! ( xmax - xmin ) / nx
!
!.... Output variables
!
      Double Precision :: uout(1:nx)      ! velocity                 (cm/s)
      Double Precision :: pout(1:nx)      ! pressure                 (dyne/cm2)
      Double Precision :: rout(1:nx)      ! density                  (g/cm3)
      Double Precision :: eout(1:nx)      ! specific internal energy (erg/g)
!
!.... Fixed left and right states
!
      Double Precision :: dens_l          ! left  density
      Double Precision :: ener_l          ! left  total energy
      Double Precision :: prs_l           ! left  pressure
      Double Precision :: sie_l           ! left  SIE
      Double Precision :: sndspd_l        ! left  sound speed
      Double Precision :: dens_r          ! right density
      Double Precision :: ener_r          ! right total energy
      Double Precision :: prs_r           ! right pressure
      Double Precision :: sie_r           ! right SIE
      Double Precision :: sndspd_r        ! right sound speed
!
      Double Precision :: ulft            ! left  velocity
      Double Precision :: plft            ! left  pressure
      Double Precision :: clft            ! left  sound speed
      Double Precision :: urght           ! right velocity
      Double Precision :: prght           ! right pressure
      Double Precision :: crght           ! right sound speed
!
!.... Solver variables for the star-state iteration
!
      Double Precision :: tol             ! solver convergence tolerance
      Double Precision :: vel_max         ! Max( | ustrl2 |, | ustrr2 | )
      Double Precision :: vel_err         ! absolute error in ustar estimate
      Double Precision :: pres_err        ! absolute error in pstar estimate
      Double Precision :: pstar           ! star-state pressure
      Double Precision :: pstar1          ! k-1th iterate for pstar
      Double Precision :: pstar2          ! kth   iterate for pstar
      Double Precision :: wlft            ! left  Lagrangian wave speed
      Double Precision :: wlft1           ! k-1th left Lagrangian wave speed
      Double Precision :: wrght           ! right Lagrangian wave speed
      Double Precision :: wrght1          ! k-1th right Lagrantian wave speed
      Double Precision :: ustrl1          ! u*_left^{it-2}
      Double Precision :: ustrr1          ! u*_right^{it-2}
      Double Precision :: ustrl2          ! u*_left^{it-1}
      Double Precision :: ustrr2          ! u*_right^{it-1}
      Double Precision :: delu1           ! u_left^{it-2}  - u_right^{it-2}
      Double Precision :: delu2           ! u_left^{it-1}  - u_right^{it-1}
      Double Precision :: delul           ! u_left^{it-1}  - u_left^{it-2}
      Double Precision :: delur           ! u_right^{it-1} - u_right^{it-2}
!
      Double Precision :: scratch         ! delu2  - delu1
      Double Precision :: scratch2        ! scratch 
      Double Precision :: rhoshklo        ! JUMPLEFT or JUMPRIGHT value at rholo
      Double Precision :: rhoshkhi        ! JUMPLEFT or JUMPRIGHT value at rhohi
      Double Precision :: prsndlo         ! RARELEFT or RARERIGHT value at drholo
      Double Precision :: prsndhi         ! RARELEFT or RARERIGHT value at drhohi
!
      Double Precision :: vstar           ! star-state specific volume
      Double Precision :: rhostrl         ! left-star  density
      Double Precision :: rhostarl        ! left-star  density
      Double Precision :: prsstarl        ! left-star  density
      Double Precision :: siestarl        ! left-star  pressure
      Double Precision :: sndstarl        ! left-star  sound speed
      Double Precision :: velstarl        ! left-star  velocity
!
      Double Precision :: rhostrr         ! right-star density
      Double Precision :: rhostarr        ! right-star density
      Double Precision :: prsstarr        ! right-star density
      Double Precision :: siestarr        ! right-star pressure
      Double Precision :: sndstarr        ! right-star sound speed
      Double Precision :: velstarr        ! right-star velocity
!
!.... Variables for the rarefaction integration and solution
!
      Double Precision :: drhol           ! delta-rho for left  rarefaction integration
      Double Precision :: drhor           ! delta-rho for right rarefaction integration
      Double Precision :: drholo          ! bracketing value (lower) of drho
      Double Precision :: drhohi          ! bracketing value (upper) of drho
      Double Precision :: drholo2         ! bracketing value (lower) of drho
      Double Precision :: drhohi2         ! bracketing value (upper) of drho
      Double Precision :: prstol          ! integration tolerance on the pressure
!
      Double Precision :: xrarl(1:nrar)   ! left  rarefctn position   (cm)
      Double Precision :: velrarl(1:nrar) ! left  rarefctn velocity   (cm/s)
      Double Precision :: rhorarl(1:nrar) ! left  rarefctn density    (g/cm3)
      Double Precision :: prsrarl(1:nrar) ! left  rarefctn pressure   (dyne/cm2)
      Double Precision :: sierarl(1:nrar) ! left  rarefctn SIE        (erg/g)
      Double Precision :: sndrarl(1:nrar) ! left  rarefctn snd speed  (cm/s)
!
      Double Precision :: brrarl(1:nrar)  ! left  rarefctn r-spline linear  coeff.
      Double Precision :: crrarl(1:nrar)  ! left  rarefctn r-spline quadrtc coeff.
      Double Precision :: drrarl(1:nrar)  ! left  rarefctn r-spline cubic   coeff.
      Double Precision :: bprarl(1:nrar)  ! left  rarefctn p-spline linear  coeff.
      Double Precision :: cprarl(1:nrar)  ! left  rarefctn p-spline quadrtc coeff.
      Double Precision :: dprarl(1:nrar)  ! left  rarefctn p-spline cubic   coeff.
      Double Precision :: bvrarl(1:nrar)  ! left  rarefctn v-spline linear  coeff.
      Double Precision :: cvrarl(1:nrar)  ! left  rarefctn v-spline quadrtc coeff.
      Double Precision :: dvrarl(1:nrar)  ! left  rarefctn v-spline cubic   coeff.
!
      Double Precision :: xrarr(1:nrar)   ! right rarefctn position   (cm)
      Double Precision :: velrarr(1:nrar) ! right rarefctn velocity   (cm/s)
      Double Precision :: rhorarr(1:nrar) ! right rarefctn density    (g/cm3)
      Double Precision :: prsrarr(1:nrar) ! right rarefctn pressure   (dyne/cm2)
      Double Precision :: sierarr(1:nrar) ! right rarefctn SIE        (erg/g)
      Double Precision :: sndrarr(1:nrar) ! right rarefctn snd speed  (cm/s)
!
      Double Precision :: brrarr(1:nrar)  ! right rarefctn r-spline linear  coeff.
      Double Precision :: crrarr(1:nrar)  ! right rarefctn r-spline quadrtc coeff.
      Double Precision :: drrarr(1:nrar)  ! right rarefctn r-spline cubic   coeff.
      Double Precision :: bprarr(1:nrar)  ! right rarefctn p-spline linear  coeff.
      Double Precision :: cprarr(1:nrar)  ! right rarefctn p-spline quadrtc coeff.
      Double Precision :: dprarr(1:nrar)  ! right rarefctn p-spline cubic   coeff.
      Double Precision :: bvrarr(1:nrar)  ! right rarefctn v-spline linear  coeff.
      Double Precision :: cvrarr(1:nrar)  ! right rarefctn v-spline quadrtc coeff.
      Double Precision :: dvrarr(1:nrar)  ! right rarefctn v-spline cubic   coeff.
!
      Double Precision :: scrtch(1:nrar)  ! scratch rarefation array
!
!.... Values for shock solution
!
      Double Precision :: rholo           ! bracketing value (lower) of rho
      Double Precision :: rhohi           ! bracketing value (upper) of rho
      Double Precision :: rholo2          ! bracketing value (lower) of rho
      Double Precision :: rhohi2          ! bracketing value (upper) of rho
      Double Precision :: rhotol          ! iteration tolerance on the density
!
      Double Precision :: rhos
      Double Precision :: rhostr
      Double Precision :: siestr
      Double Precision :: sndstr
      Double Precision :: ustar
!
      Double Precision :: smlrho          ! Density  cut-off
      Double Precision :: smallp          ! Pressure cut-off
      Double Precision :: smallu          ! Velocity cut-off
!
!.... Values for converged-solution waves
!
      Double Precision :: qlft            ! mass flux of left-going  shock
      Double Precision :: qrght           ! mass flux of right-going shock
      Double Precision :: ushklft         ! speed     of left-going  shock
      Double Precision :: ushkrght        ! speed     of right-going shock
      Double Precision :: urhdlft         ! speed     of left-going  rarefctn head
      Double Precision :: urhdrght        ! speed     of right-going rarefctn head
      Double Precision :: urtllft         ! speed     of left-going  rarefctn tail
      Double Precision :: urtlrght        ! speed     of right-going rarefctn tail
!
      Double Precision :: xcont           ! position  of contact
      Double Precision :: xshklft         ! position  of left-going  shock
      Double Precision :: xshkrght        ! position  of right-going shock
      Double Precision :: xrhdlft         ! position  of left-going  rarefctn head
      Double Precision :: xrhdrght        ! position  of right-going rarefctn head
      Double Precision :: xrtllft         ! position  of left-going  rarefctn tail
      Double Precision :: xrtlrght        ! position  of right-going rarefctn tail

      Double Precision :: vs
      Double Precision :: ces
      Double Precision :: cestar
      Double Precision :: ws
!
!.... Final solution values
!
      Double Precision :: xout            ! position
      Double Precision :: rhoout          ! density 
      Double Precision :: prsout          ! pressure 
      Double Precision :: velout          ! velocity 
      Double Precision :: sieout          ! SIE      
      Double Precision :: sndout          ! sound speed 
!
!.... Logical variables
!
      Logical :: lconverged               ! .true. => converged
      Logical :: lrarel                   ! .true. => LEFT  rarefaction
      Logical :: lshockl                  ! .true. => LEFT  shock
      Logical :: lrarer                   ! .true. => RIGHT rarefaction
      Logical :: lshockr                  ! .true. => RIGHT shock
      Logical :: ldebug                   ! print-to-file debug flag 
      Logical :: ldoublevalue             ! .true. => write two values @ discontinuities 
      Logical :: lwritten                 ! .true. => data at point xval(i) has been written
!
!.... Functions called
!
      Double Precision :: JUMPLEFT        ! Evaluate left  shock jump conditions
      Double Precision :: RARELEFT        ! nonlinear function for left  rarfctn used by ZEROIN
      Double Precision :: JUMPRIGHT       ! Evaluate right shock jump conditions
      Double Precision :: RARERIGHT       ! nonlinear function for right rarfctn used by ZEROIN
      Double Precision :: SPLEVAL         ! spline evaluation function
      Double Precision :: ZEROIN          ! nonlinear 1D root-finder function
      External RARELEFT
      External RARERIGHT
      External JUMPLEFT
      External JUMPRIGHT
      External SPLEVAL
      External ZEROIN
!
      Double Precision :: rholft          ! left  density       in common block
      Double Precision :: sielft          ! left  SIE           in common block
      Double Precision :: prslft          ! left  pressure      in common block
      Double Precision :: sndlft          ! left  sound speed   in common block
      Double Precision :: vellft          ! left  velocity      in common block
      Double Precision :: rhorgt          ! right density       in common block
      Double Precision :: siergt          ! right SIE           in common block
      Double Precision :: prsrgt          ! right pressure      in common block
      Double Precision :: sndrgt          ! right sound speed   in common block
      Double Precision :: velrgt          ! right velocity      in common block
      Integer :: nrare                    ! # of pts in rarfctn in common block
!
      Double Precision :: rho0l, sie0l, gamma0l, bigal, bigbl, r1l, r2l ! JWL EOS values
      Double Precision :: rho0r, sie0r, gamma0r, bigar, bigbr, r1r, r2r ! JWL EOS values
!
      Common / left_jwl  / rho0l, sie0l, gamma0l, bigal, bigbl, r1l, r2l
      Common / right_jwl / rho0r, sie0r, gamma0r, bigar, bigbr, r1r, r2r
      Common / left_state  / rholft, sielft, prslft, sndlft, vellft
      Common / right_state / rhorgt, siergt, prsrgt, sndrgt, velrgt
      Common / star_state  / pstar
      Common / num_rare    / nrare
!
      Data itmax     / 10 /               ! maximum number of secant iterations
      Data tol       / 1.d-10 /           ! secant iteration tolerance
      Data smlrho    / 1.d-8 /            ! density  cutoff
      Data smallp    / 1.d-8 /            ! pressure cutoff
      Data smallu    / 1.d-8 /            ! velocity cutoff
!
!-----------------------------------------------------------------------
!
! convert input to common block for JWL EOS
      rho0l   = xrho0l
      sie0l   = xsie0l
      gamma0l = xgamma0l
      bigal   = xbigal
      bigbl   = xbigbl
      r1l     = xr1l
      r2l     = xr2l
      rho0r   = xrho0r
      sie0r   = xsie0r
      gamma0r = xgamma0r
      bigar   = xbigar
      bigbr   = xbigbr
      r1r     = xr1r
      r2r     = xr2r
!*!
!      PRINT *, "rho0, sie0, gamma0, biga, bigb, r1, r2"
!      PRINT *, "l:", rho0l, sie0l, gamma0l, bigal, bigbl, r1l, r2l
!      PRINT *, "r:", rho0r, sie0r, gamma0r, bigar, bigbr, r1r, r2r
!*!
!
!-----------------------------------------------------------------------
!
      ierr = 0
      lrarel  = .false.
      lshockl = .false.
      lrarer  = .false.
      lshockr = .false.
      ldebug  = ldebugin
      iun = 15
      iunm1 = iun - 1
      iunp1 = iun + 1
!*!      Open(unit=iun,file='riemjwl.log',status='unknown')
!*!      Open(unit=iunm1,file='riemjwl.dbg',status='unknown')
!
!.... Assign common block value for possible rarefaction
!
      nrare = nrar
!  
!=======================================================================
!.... Evaluate the complete left state
!=======================================================================
!
      dens_l = rhol
      ulft   = ul
      prs_l  = prsl
      plft   = prs_l
      Call EOS_LEFT_RHOP( dens_l, prs_l, sie_l, sndspd_l, ierr )
      If ( ierr .ne. 0 ) ierr = 1
      If ( ierr .ne. 0 ) Go To 799
!
!.... This is the general Lagrangian soundspeed
!
      clft = dens_l * sndspd_l
!
!.... Assign common block values for the left state
!
      rholft = dens_l
      sielft = sie_l
      prslft = prs_l
      sndlft = sndspd_l
      vellft = ulft
!*!      Write(iun,501) rholft, prslft, sielft, sndlft, vellft
!*!      If ( ldebug ) Write(iunm1,501) &
!*!                    rholft, prslft, sielft, sndlft, vellft
!
!=======================================================================
!.... Evaluate the complete right state
!=======================================================================
!
      dens_r = rhor
      urght  = ur
      prs_r  = prsr
      prght  = prs_r
      Call EOS_RIGHT_RHOP( dens_r, prs_r, sie_r, sndspd_r, ierr )
      If ( ierr .ne. 0 ) ierr = 2
      If ( ierr .ne. 0 ) Go To 799
!
!.... This is the general Lagrangian soundspeed
!
      crght = dens_r * sndspd_r
!
!.... Assign common block values for the right state
!
      rhorgt = dens_r
      siergt = sie_r
      prsrgt = prght
      sndrgt = sndspd_r
      velrgt = urght
!*!      Write(iun,502) rhorgt, prsrgt, siergt, sndrgt, velrgt
!*!      If ( ldebug ) Write(iunm1,502) &
!*!                    rhorgt, prsrgt, siergt, sndrgt, velrgt
!
!=======================================================================
!.... Right state vacuum: rarefaction-contact-zero_pres
!=======================================================================
!
! *** NOT IMPLEMENTED ***
!
!=======================================================================
!.... Left state vacuum: zero_pres-contact-rarefaction
!=======================================================================
!
! *** NOT IMPLEMENTED ***
!
!======================================================================
!.... Multiple wave cases
!======================================================================
!
!.... Determine the nature of the solution, which must be 
!.... one of the following, from left-to-right:
!
!.... (1) shock-contact-shock
!.... (2) shock-contact-rarefation
!.... (3) rarefaction-contact-rarefaction
!.... (4) rarefaction-contact-shock
!.... (5) rarefaction-contact-vacuum-contact-rarefaction ** NOT IMPLEMENTED **
!
!======================================================================
!.... The secant iteration requires TWO previous guesses to get started
!======================================================================
!.... 1st guess for secant iteration: in CG, superscripted ^{\nu-1}
!-----------------------------------------------------------------------
!
!     Construct 1st value for secant iteration by assuming that the
!     nonlinear wave speed is equal to the sound speed -- the result
!     is the same as Toro, Eq. 9.28: this is just the linearized 
!     (acoustic) Riemann solver for a general EOS.
!
      pstar1 = ( ( crght / ( clft + crght ) ) * plft )  &
             + ( ( clft  / ( clft + crght ) ) * prght ) &
             + ( ( clft * crght ) / ( clft + crght ) )  &
               * ( urght - ulft )
!
      If ( ( pstar1 .gt. plft ) .and. ( pstar1 .gt. prght ) ) &
        pstar1 = HALF * Max( plft, prght ) 
      If ( pstar1 .le. EPS8 ) pstar1 = EPS8
!
!.... Assign common block value of pstar1
!
      pstar = pstar1
!*!      If ( ldebug ) Write(iunm1,513) pstar1
!
!.... Using this common value of pstar, calculate the corresponding
!     particle velocity values for the left and right star-states, 
!     depending upon whether the left and right waves are putatively
!     rarefactions (if pstar <= p_{L,R}) or shocks (if pstar > p_{L,R}).
!
!.... Calculate left star-velocity
!
      Call GET_USTAR_LEFT( xd0, time, ustrl1, ldebug, ierr )
      If ( ierr .ne. 0 ) ierr = 3
      If ( ierr .ne. 0 ) Go To 799
!*!      If ( ldebug ) Write(iunm1,551) ustrl1
!
!.... Calculate right star-velocity
!
      Call GET_USTAR_RIGHT( xd0, time, ustrr1, ldebug, ierr )
      If ( ierr .ne. 0 ) ierr = 4
      If ( ierr .ne. 0 ) Go To 799
!*!      If ( ldebug ) Write(iunm1,552) ustrr1
!
!-----------------------------------------------------------------------
!.... Second guess for secant iteration: in CG, superscripted ^{\nu}
!-----------------------------------------------------------------------
!     Construct second guess for the pressure using the nonlinear wave
!     speeds from the first guess.  This is the same approxiation used
!     to obtain pstar1, except the modified wave speeds (instead of the
!     updated sound speeds) are used, i.e., this is a linearized Riemann 
!     solver using the _updated_ Lagrangian wavespeed
!
      If ( Abs( ustrl1 - ulft ) .lt. EPS12 ) wlft1 = clft
      If ( Abs( ustrl1 - ulft ) .ge. EPS12 ) &
        wlft1 = Abs( pstar1 - prslft ) / Abs( ustrl1 - ulft )
!*!      If ( ldebug ) Write(iunm1,503) rholft, sielft, prslft, sndlft
!*!      If ( ldebug ) Write(iunm1,505) wlft1
!
      If ( Abs( ustrr1 - urght ) .lt. EPS12 ) wrght1 = crght
      If ( Abs( ustrr1 - urght ) .ge. EPS12 ) &
        wrght1 = Abs( pstar1 - prsrgt ) / Abs( ustrr1 - urght )
!*!      If ( ldebug ) Write(iunm1,506) rhorgt, siergt, prsrgt, sndrgt
!*!      If ( ldebug ) Write(iunm1,507) wrght1
!
!.... The following expressions reduce to the relation for pstar1 above with
!     "wrght1" substituted for "crght" and "wlft1" substituted for "clft"
!
      pstar2 = prght - plft - wrght1 * ( urght - ulft )
      pstar2 = plft + pstar2 * ( wlft1 / ( wlft1 + wrght1 ) )
      If ( pstar2 .le. EPS8 ) pstar2 = HALF * pstar1
!      pstar2 = Max( smallp, pstar2 )
!
!.... Update the common block value to pstar2
!
      pstar = pstar2
!*!      If ( ldebug ) Write(iunm1,514) pstar
!  
!-----------------------------------------------------------------------
!.... Begin the secant iteration -- see CG Eqs. 17 and 18e
!-----------------------------------------------------------------------
!.... Iterate for convergence until either: (1) the sum of the errors 
!     in the velocity and in the pressure together falls below the 
!     tolerance, or (2) itmax iterations are done w/o convergence.
!
      lconverged = .false.
      it = 0
!*!      Write (iun,600) 
!
!.... Write out iteration information
!
      ustrl2 = ZERO
      ustrr2 = ZERO
      vel_err  = ONE / EPS8
      pres_err = ONE / EPS8
!*!      If ( ldebug ) Write(iunm1,600) 
!*!      If ( ldebug ) Write(iunm1,602) it, &
!*!        ustrl1, ustrr1, pstar1, ustrl2, ustrr2, pstar2,vel_err, pres_err
!
      Do While ( .not. lconverged )
!          
        it = it + 1
        If ( it .gt. itmax ) Go To 100
!
!.... As before, compute the velocities in the "star" state -- using CG
!.... Eq. 18 -- ustrl2 and ustrr2 are the velocities defined there.
!.... ustrl1 and ustrr1 are the velocities at the previous iterate,
!.... as pstar1 is the previous star-state pressure:  
!.... -> variables ending in "1" represent those in CG superscipted ^{\nu-1}
!.... -> variables ending in "2" represent those in CG superscipted ^{\nu}
!
!.... Calculate left star-velocity
!
        Call GET_USTAR_LEFT( xd0, time, ustrl2, ldebug, ierr )
        If ( ierr .ne. 0 ) ierr = 5
        If ( ierr .ne. 0 ) Go To 799
!*!        If ( ldebug ) Write(iunm1,553) it, ustrl2
!
!.... Calculate right star-velocity
!
        Call GET_USTAR_RIGHT( xd0, time, ustrr2, ldebug, ierr )
        If ( ierr .ne. 0 ) ierr = 6
        If ( ierr .ne. 0 ) Go To 799
!*!        If ( ldebug ) Write(iunm1,554) it, ustrr2
!
!.... Evaluate error terms in CG Eq. 18, allowing for possibily zero error,
!     and update the iterated value of pstar
!
        delul = Abs( ustrl2 - ustrl1 )
        delur = Abs( ustrr2 - ustrr1 )
        scratch = delul + delur       
        delu2 = ustrr2 - ustrl2
        If ( Abs( pstar2 - pstar1 ) .le. smallp ) scratch = ZERO
        If ( Abs( scratch ) .lt. smallu) delu2   = ZERO
        If ( Abs( scratch ) .lt. smallu) scratch = ONE
        pstar = pstar2 - delu2 * Abs( pstar2 - pstar1 ) / scratch
        pstar = Max( smallp, pstar )
!
!..... Evaluate the absolute errors in pstar and ustar estimates
!
        pres_err = Abs( pstar - pstar2 )
        vel_err  = Abs( ustrl2 - ustrr2 )
!
!..... Write out iteration information
!
!*!        Write(iun,602) it, ustrl1, ustrr1, pstar1, &
!*!         ustrl2, ustrr2, pstar2, vel_err, pres_err
!*!        If ( ldebug ) Write(iunm1,600) 
!*!        If ( ldebug ) Write(iunm1,602) it, ustrl1, ustrr1, pstar1, &
!*!          ustrl2, ustrr2, pstar2, vel_err, pres_err
!
!..... Check for iteration convergence using both pressure and velocity
!
        If ( Abs( vel_err ) + Abs( pres_err ) .lt. tol ) &
          lconverged = .true.
        If ( lconverged ) Go To 100              ! Exit if converged
!
!.... Update variables for the next iteration 
!
        pstar1 = pstar2
        pstar2 = pstar
        ustrl1 = ustrl2
        ustrr1 = ustrr2
!          
      End Do     ! Do While ( .not. lconverged )
 100  Continue
!
      If ( .not. lconverged ) ierr = 7
      If ( ierr .ne. 0 ) Go To 799
!*!      If ( ldebug ) Write(iunm1,520) it
!  
!=======================================================================
!.... End of secant iteration
!=======================================================================
!.... Assign ustar as the mean of the converged left and right values
!
      ustar = HALF * ( ustrl2 + ustrr2 )
!
!=======================================================================
!.... Determine nature of the LEFT wave
!=======================================================================
!*!        If ( ldebug ) Then
!*!          Write(iunm1,703)
!*!          Write(iunm1,705) rholft, prslft, vellft, sielft, sndlft
!*!        End If
        If ( pstar .gt. plft ) Then
!-----------------------------------------------------------------------
!.... Left-going shock: CG Eq. 15 and evaluate sndspd
!-----------------------------------------------------------------------
        lshockl  = .true.
!*!        If ( ldebug ) Write(iunm1,530)
!
!        rholo  = HALF * Min( rholft, rhorgt )
!        rhohi  = THREE * Max( rholft, rhorgt )  ! THREE might not be big enough
!        If ( ldebug ) Then
!          rhoshklo = JUMPLEFT( rholo )
!          rhoshkhi = JUMPLEFT( rhohi )
!          Write(iunm1,330) rholo, rhoshklo, rhohi, rhoshkhi
!          scratch2 = rhoshklo * rhoshkhi
!          If ( scratch2 .ge. ZERO ) ierr = 8
!        End If 
!        If ( ierr .ne. 0 ) Go To 799
!        rhostrl = ZEROIN( rholo, rhohi, JUMPLEFT, rhotol )
!
!.... Evaluate the shock jump conditions by fist determing values of 
!     the post-shock density that bracket the desired value of the 
!     post-shock pressure, i.e., pstar.  This is done by obtaining
!     values such that JUMPLEFT(rholo) * JUMPLEFT(rhohi) < 0
!
!.... Bracket a zero of JUMPLEFT:
!
        rholo = Min( rholft, rhorgt )
        rhohi = Max( rholft, rhorgt )
        Call BRACKET_LEFT_SHOCK( rholo, rhohi, rholo2, rhohi2, &
                                 ldebug, ierr )
        If ( ierr .ne. 0 ) ierr = 8
        If ( ierr .ne. 0 ) Go To 799
        rholo   = rholo2
        rhohi   = rhohi2
        rhotol  = EPS12
        rhostrl = ZEROIN( rholo, rhohi, JUMPLEFT, rhotol )
!
!.... Evaluate left-going shock speed 
!
        ushklft = vellft - Sqrt( ( rhostrl / rholft ) &
                * ( ( pstar - prslft ) / ( rhostrl - rholft ) ) )
!
!.... Evaluate u^{*}_{L} from jump conditions
!
        velstarl = ushklft + Sqrt( ( rholft / rhostrl ) &
                 * ( ( pstar - prslft ) / ( rhostrl - rholft ) ) ) 
!
        Call EOS_LEFT_RHOP( rhostrl, pstar, siestr, sndstr, ierr)
        If ( ierr .ne. 0 ) ierr = 9
        If ( ierr .ne. 0 ) Go To 799
!
!.... Assign left post-shock star values
!
        rhostarl = rhostrl
        prsstarl = pstar
        siestarl = siestr
        sndstarl = sndstr
        velstarl = ustar
!*!        If ( ldebug ) Write(iunm1,713)
!*!        If ( ldebug ) Write(iunm1,705) rholft, prslft, vellft, &
!*!                                       sielft, sndlft
!*!        If ( ldebug ) Write(iunm1,714)
!*!        If ( ldebug ) Write(iunm1,705) rhostarl,prsstarl,velstarl, &
!*!                                       siestarl,sndstarl
!
      Else If ( pstar .lt. plft ) Then
!-----------------------------------------------------------------------
!.... Left-going rarefaction: integrate the characteristic equation
!-----------------------------------------------------------------------
        lrarel = .true.
!*!        If ( ldebug ) Write(iunm1,531)
!
!.... Integrate CG Eq. 13 according to the procedure of Jeff Banks, 
!     "On Exact Conservation for the Euler Equations with Complex 
!     Equations of State," Commun. Comput. Phys., 8(5), pp. 995-1015 
!     (2010), doi: 10.4208/cicp/090909/100310a.   One must first
!     compute the density-increment, drhol, according to which a 
!     Runge-Kutta-evaluated estimate of the characteristic ODE for 
!     dp/drho is integrated from the left state to the star state.
!     This is done by first bracketing a value of drhol so that the 
!     computed pressures are above and below the target pstar value, 
!     i.e., so that RARELEFT(drholo) * RARELEFT(drhohi) < 0
!
!.... Bracket a zero of RARELEFT
!
        drholo = Max( EPS4, Abs( rhol - rhor ) / Dble( nrar ) )
        drholo = 0.01d0 * drholo
        drhohi = ( Max( rhol, rhor ) - Sqrt( EPS4 ) )  / Dble( nrar )
        drhohi = TWO * drholo
        Call BRACKET_LEFT_RARE( drholo, drhohi, drholo2, drhohi2, &
                                ldebug, ierr )
        If ( ierr .ne. 0 ) ierr = 10
        If ( ierr .ne. 0 ) Go To 799
        drholo = drholo2
        drhohi = drhohi2
        prstol = EPS12
        drhol  = ZEROIN( drholo, drhohi, RARELEFT, prstol )
        If ( drhol .eq. ZERO ) ierr = 11
        If ( ierr .ne. 0 ) Go To 799
!*!        If ( ldebug ) Write(iunm1,540) drhol 
!
!.... Evaluate the solution thru the rarefaction, obtaining a set of values
!     that will be interpolated to obtain values at the requested points.  
!
        Call RARELEFTSOLN( nrar, drhol, xd0, time, xrarl, velrarl, &
                           rhorarl, prsrarl, sierarl, sndrarl, ierr )
        If ( ierr .ne. 0 ) ierr = 12
        If ( ierr .ne. 0 ) Go To 799
        If ( ldebug ) Then
!*!          Write(iunm1,723)
!*!          Do i = 1, nrar
!*!            Write(iunm1,725) i, xrarl(i),rhorarl(i),prsrarl(i), &
!*!                              velrarl(i),sierarl(i),sndrarl(i)
!*!          End Do ! i
        End If ! ldebug
!
!.... Set up the spline interpolation arrays for left rarefaction values,
!     first for density, then for pressure, then for velocity
!
        Call SPLINE( nrar, xrarl, rhorarl, brrarl, crrarl, drrarl, &
                     ierr )
        If ( ierr .ne. 0 ) ierr = 13
        If ( ierr .ne. 0 ) Go To 799
        Call SPLINE( nrar, xrarl, prsrarl, bprarl, cprarl, dprarl, &
                     ierr )
        If ( ierr .ne. 0 ) ierr = 14
        If ( ierr .ne. 0 ) Go To 799
        Call SPLINE( nrar, xrarl, velrarl, bvrarl, cvrarl, dvrarl, &
                     ierr )
        If ( ierr .ne. 0 ) ierr = 15
        If ( ierr .ne. 0 ) Go To 799
!
!.... Assign left-star values
!
        rhostrl = rhorarl(nrar)
        Call EOS_LEFT_RHOP( rhostrl, pstar, siestr, sndstr, ierr )
        If ( ierr .ne. 0 ) ierr = 16
        rhostarl = rhostrl
        prsstarl = pstar
        siestarl = siestr
        sndstarl = sndstr
        velstarl = ustar
!*!        If ( ldebug ) Write(iunm1,733)
!*!        If ( ldebug ) Write(iunm1,705) rhostarl,prsstarl,velstarl, &
!*!                                       siestarl,sndstarl
      Else
!-----------------------------------------------------------------------
!.... pstar = plft:  Not sure what this means, so flag an error for now
!-----------------------------------------------------------------------
        ierr = 17
      End If ! pstar .gt. plft     
      If ( ierr .ne. 0 ) Go To 799
!=======================================================================
!.... Determine nature of the RIGHT wave
!=======================================================================
      If ( pstar .gt. prght ) Then
!-----------------------------------------------------------------------
!.... If pstar > prsr, then the right wave is a shock
!.......................................................................
        lshockr  = .true.
!*!        If ( ldebug ) Write(iunm1,532)
!
!        rholo  = HALF * Min( rholft, rhorgt )
!c        rhohi  = Max( rholft, rhorgt )
!        rhohi  = THREE * Max( rholft, rhorgt )  ! THREE might not be big enough
!        rhotol  = EPS12
!        If ( ldebug ) Then
!          rhoshklo = JUMPRIGHT( rholo )
!          rhoshkhi = JUMPRIGHT( rhohi )
!          Write(iunm1,350) rholo, rhoshklo, rhohi, rhoshkhi
!          scratch2 = rhoshklo * rhoshkhi
!          If ( scratch2 .ge. ZERO ) ierr = 18
!        End If 
!        If ( ierr .ne. 0 ) Go To 799
!
!.... Bracket a zero of JUMPRIGHT
!
        rholo = Min( rhol, rhor )
        rhohi = Max( rhol, rhor )
        Call BRACKET_RIGHT_SHOCK( rholo, rholo, rholo2, rhohi2, &
                                  ldebug, ierr )
        If ( ierr .ne. 0 ) ierr = 18
        If ( ierr .ne. 0 ) Go To 799
        rholo   = rholo2
        rhohi   = rhohi2
        rhotol  = EPS12
        rhostrr = ZEROIN( rholo, rhohi, JUMPRIGHT, rhotol )
!
!.... Evaluate right-going shock speed 
!
        ushkrght = velrgt + Sqrt( ( rhostrr / rhorgt ) &
              * ( ( pstar - prsrgt ) / ( rhostrr - rhorgt ) ) )
!
!.... Evaluate u^{*}_{R} from jump conditions
!
        velstarr = ushkrght - Sqrt( ( rhorgt / rhostrr ) &
                * ( ( pstar - prsrgt ) / ( rhostrr - rhorgt ) ) ) 
!
        Call EOS_RIGHT_RHOP( rhostrr, pstar, siestr, sndstr, ierr)
        If ( ierr .ne. 0 ) ierr = 19
        If ( ierr .ne. 0 ) Go To 799
!
!.... Assign right-star values
!
        rhostarr = rhostrr
        prsstarr = pstar
        siestarr = siestr
        sndstarr = sndstr
        velstarr = ustar
!*!        If ( ldebug ) Write(iunm1,743)
!*!        If ( ldebug ) Write(iunm1,705) rhostarr,prsstarr,velstarr, &
!*!                                       siestarr,sndstarr
!*!        If ( ldebug ) Write(iunm1,744)
!*!        If ( ldebug ) Write(iunm1,705) rhorgt, prsrgt, velrgt, &
!*!                                       siergt, sndrgt
!
      Else If ( pstar .lt. prght ) Then
!-----------------------------------------------------------------------
!.... Right-going rarefaction: integrate CG Eq. 13 and evaluate sndspd
!-----------------------------------------------------------------------
!.... This is like the left-going rarefaction calculation above.
!
        lrarer  = .true.
!*!        If ( ldebug ) Write(iunm1,533)
!
!.... Assign limiting argument values for ZEROIN such that:
!       RARERIGHT(drholo) * RARERIGHT(drhohi) < 0
!
!        drholo = HALF * Abs( rhorgt - rholft ) / Dble( nrar )
!        If ( drholo .lt. EPS8 ) drholo = EPS8
!        drhohi = TWO * Abs( rhorgt - rholft ) / Dble( nrar )
!        If ( drhohi .lt. EPS8 ) &
!          drhohi = Max( rholft, rhorgt ) / Dble( nrar )
!        prstol = EPS12
!        If ( ldebug ) Then
!          prsndlo = RARERIGHT( drholo )
!          prsndhi = RARERIGHT( drhohi )
!          scratch2 = prsndlo * prsndhi
!          Write(iunm1,360) drholo, prsndlo, drhohi, prsndhi
!          If ( scratch2 .ge. ZERO ) ierr = 20
!        End If 
!        If ( ierr .ne. 0 ) Go To 799
!
!.... Bracket a zero of RARERIGHT
!
        drholo = Max( EPS4, Abs( rholft - rhorgt ) / Dble( nrar ) )
        drholo = 0.01d0 * drholo
        drhohi = ( Max( rholft, rhorgt ) - Sqrt( EPS4 ) )  &
               / Dble( nrar )
        drhohi = TWO * drholo
        Call BRACKET_RIGHT_RARE( drholo, drhohi, drholo2, drhohi2, &
                                 ldebug, ierr )
        If ( ierr .ne. 0 ) ierr = 20
        If ( ierr .ne. 0 ) Go To 799
        drholo = drholo2
        drhohi = drhohi2
        prstol = EPS12
        drhor  = ZEROIN( drholo, drhohi, RARERIGHT, prstol )
        If ( drhor .eq. ZERO ) ierr = 21
!
!.... Evaluate the solution thru the rarefaction, obtaining a set of 
!     that will be interpolated to obtain values at the requested points.  
!
        Call RARERIGHTSOLN( nrar, drhor, xd0, time, xrarr, velrarr, &
                            rhorarr, prsrarr, sierarr, sndrarr, ierr )
        If ( ierr .ne. 0 ) ierr = 22
!
!.... Because the right rarefaction is integrated from right-to-left,
!     in order to use the following SPLINE interpolation routine, 
!     we have to switch the ordering of the spline-arrays, so that 
!     the x-array has increasing values for increasing index
!
        Do i = 1, nrar
          scrtch(i) = xrarr(nrar+1-i)
        End Do
        Do i = 1, nrar
          xrarr(i) = scrtch(i)
        End Do
        Do i = 1, nrar
          scrtch(i) = rhorarr(nrar+1-i)
        End Do
        Do i = 1, nrar
          rhorarr(i) = scrtch(i)
        End Do
        Do i = 1, nrar
          scrtch(i) = prsrarr(nrar+1-i)
        End Do
        Do i = 1, nrar
          prsrarr(i) = scrtch(i)
        End Do
        Do i = 1, nrar
          scrtch(i) = velrarr(nrar+1-i)
        End Do
        Do i = 1, nrar
          velrarr(i) = scrtch(i)
        End Do
!
!.... Set up the spline interpolation arrays for left rarefaction values,
!     first for density, then for pressure, then for velocity
!
        Call SPLINE( nrar, xrarr, rhorarr, brrarr, crrarr, drrarr, ierr )
        If ( ierr .ne. 0 ) ierr = 23
        If ( ierr .ne. 0 ) Go To 799
        Call SPLINE( nrar, xrarr, prsrarr, bprarr, cprarr, dprarr, ierr )
        If ( ierr .ne. 0 ) ierr = 24
        If ( ierr .ne. 0 ) Go To 799
        Call SPLINE( nrar, xrarr, velrarr, bvrarr, cvrarr, dvrarr, ierr )
        If ( ierr .ne. 0 ) ierr = 25
        If ( ierr .ne. 0 ) Go To 799
!
!.... Assign right-star values: we've reversed arrays, so use index-1 value
!
        rhostr = rhorarr(1)
        Call EOS_RIGHT_RHOP( rhostr, pstar, siestr, sndstr, ierr )
        If ( ierr .ne. 0 ) ierr = 26
        rhostarr = rhostr
        prsstarr = pstar
        siestarr = siestr
        sndstarr = sndstr
        velstarr = ustar
!*!        If ( ldebug ) Then
!*!          Write(iunm1,753)
!*!          Write(iunm1,705) rhostarr,prsstarr,velstarr, &
!*!                           siestarr,sndstarr
!*!          Write(iunm1,763)
!*!          Do i = 1, nrar
!*!            Write(iunm1,765) i, xrarr(i),rhorarr(i),prsrarr(i), &
!*!                             velrarr(i),sierarr(i),sndrarr(i)
!*!          End Do ! i
!*!        End If ! ldebug
!
      Else
!-----------------------------------------------------------------------
!.... pstar = prght:  Not sure what this means, so flag an error for now
!-----------------------------------------------------------------------
        ierr = 27
!
      End If ! pstar .gt. prght     
      If ( ierr .ne. 0 ) Go To 799
!*!      If ( ldebug ) Write(iunm1,773)
!*!      If ( ldebug ) Write(iunm1,705) &
!*!        rhorgt, prsrgt, velrgt, siergt, sndrgt
!
!=======================================================================
!.... Assign possible wave speeds and locations
!=======================================================================
!*!      Write(iun,400)
!
!.... Left-going shock speed:  Toro 4.51 + 4.16
!
      If ( lshockl ) Then
        qlft = -( pstar - plft ) / ( ustar - ulft ) 
        ushklft = ulft - ( qlft / rholft ) 
        xshklft = xd0 + ushklft * time
!*!        Write(iun,410) xshklft, ushklft
      End If ! lshockl
!
!.... Left-going rarefaction head:  Toro 4.55
!
      If ( lrarel ) Then
        urhdlft = ulft - sndspd_l
        xrhdlft = xd0 + urhdlft * time
!*!        Write(iun,420) xrhdlft, urhdlft
!
!.... Left-going rarefaction tail:  Toro 4.55
!
        urtllft = ustar - sndstarl
        xrtllft = xd0 + urtllft * time
!*!        Write(iun,430) xrtllft, urtllft
      End If ! lrarel
!
!.... Contact location
!
      xcont = xd0 + ustar * time
!*!      Write(iun,440) xcont, ustar
!
!.... Right-going shock speed:  Toro 4.58 + derivation from 4.28
!
      If ( lshockr ) Then
        qrght = ( pstar - prght ) / ( ustar - urght )
        ushkrght = urght + ( qrght / rhorgt )
        xshkrght = xd0 + ushkrght * time
!*!        Write(iun,450) xshkrght, ushkrght
      End If ! lshockr
!
!.... Right-going rarefaction tail:  Toro 4.62
!
      If ( lrarer ) Then
        urtlrght = ustar + sndstarr
        xrtlrght = xd0 + urtlrght * time
!*!        Write(iun,460) xrtlrght, urtlrght
!
!.... Right-going rarefaction head:  Toro 4.62
!
        urhdrght = urght + sndspd_r
        xrhdrght = xd0 + urhdrght * time
!*!        Write(iun,470) xrhdrght, urhdrght
      End If ! lrarer
!
!=======================================================================
!.... Assign the values to the cell centers and write to file
!=======================================================================
!
!*!      deltax = ( xmax - xmin ) / Dble( nx )
!*!      xval(1) = xmin + HALF * deltax
!*!      Do i = 2, nx
!*!        xval(i) = xval(i-1) + deltax
!*!      End Do ! i
!
!*!
      If (lwrite) THEN
         Open(unit=iunp1,file='riemjwl.dat',status='unknown')
         Write(iunp1,101)
      End If
!*!
!
!.... NOTES:  
!     (1) this currently writes out the additional abscissae and
!         values corresponding to the head and tail of rarefactions.
!     (2) there are two available options for writing data at 
!         discontinuities (i.e., at shocks and contacts):
!         ldoublevalue = .true. => write the abscissa corresponding
!           to a discontinuty TWICE, once with the ordinate of the 
!           left state, and once with the ordinate of the right state.
!         ldoublevalue = .false. => write the abscissa corresponding
!           to a discontinuty ONCE, with the ordinate being the 
!           the arithmetic average of the left and right states.
!
      ldoublevalue = .true.
      Do i = 1, nx
        If (i .eq. 1) Then
           deltax = x_m(2) - x_m(1)
        Else
           deltax = x_m(i) - x_m(i-1)
        End If
        lwritten = .false.
        x = x_m(i)
        xprev = x - deltax

!-----------------------------------------------------------------------
!.... Left rarefaction?
!-----------------------------------------------------------------------
        If ( lrarel ) Then
!
!.... Point is to the left of the leading left-rarefaction head
!
          If ( xprev .lt. xrhdlft .and. x .lt. xrhdlft ) Then
            lwritten = .true.
            xout = x
            rhoout = rholft
            prsout = prslft
            velout = vellft
            sieout = sielft
            sndout = sndlft
!
!.... Point and previous point straddle leading left-rarefaction head
!
          Else If ( xprev .lt. xrhdlft .and. x .gt. xrhdlft ) Then
            lwritten = .true.
            xout = xrhdlft        ! Left-rarefaction head
            rhoout = rholft
            prsout = prslft
            velout = vellft
            sieout = sielft
            sndout = sndlft
!*!            Write(iunp1,103) xout, velout, prsout, rhoout, &
!*!                                   sieout, sndout, time
            xout = x
            rhoout = SPLEVAL( nrar, x, xrarl, rhorarl, brrarl, crrarl, &
                              drrarl )
           prsout = SPLEVAL( nrar, x, xrarl, prsrarl, bprarl, cprarl, &
                             dprarl )
            Call EOS_LEFT_RHOP( rhoout, prsout, sieout, sndout, ierr )
            If ( ierr .ne. 0 ) ierr = 28
            velout = SPLEVAL( nrar, x, xrarl, velrarl, bvrarl, cvrarl, &
                              dvrarl )
!
!.... Point is inside the left-rarefaction fan: interpolate
!
          Else If ( xprev .gt. xrhdlft .and. x .gt. xrhdlft &
                                       .and. x .lt. xrtllft ) Then
            lwritten = .true.
            xout = x
            rhoout = SPLEVAL( nrar, x, xrarl, rhorarl, brrarl, crrarl, &
                              drrarl )
            prsout = SPLEVAL( nrar, x, xrarl, prsrarl, bprarl, cprarl, &
                              dprarl )
            Call EOS_LEFT_RHOP( rhoout, prsout, sieout, sndout, ierr )
            If ( ierr .ne. 0 ) ierr = 29
            velout = SPLEVAL( nrar, x, xrarl, velrarl, bvrarl, cvrarl, &
                              dvrarl )
!
!.... Point and previous point straddle left-rarefaction tail
!
          Else If ( xprev .lt. xrtllft .and. x .gt. xrtllft ) Then
            lwritten = .true.
            xout = xrtllft           ! Left-rarefaction tail 
            rhoout = rhostarl
            prsout = pstar
            velout = velstarl
            sieout = siestarl
            sndout = sndstarl
!*!            Write(iunp1,103) xout, velout, prsout, rhoout, &
!*!                                   sieout, sndout, time
            xout = x
            rhoout = rhostarl
            prsout = pstar
            velout = velstarl
            sieout = siestarl
            sndout = sndstarl
!
!.... Point is right of the rarefact'n tail and left of the contact
!
          Else If ( xprev .gt. xrtllft .and. x .gt. xrtllft &
                                       .and. x .lt. xcont ) Then
            lwritten = .true.
            xout = x
            rhoout = rhostarl
            prsout = pstar
            velout = velstarl
            sieout = siestarl
            sndout = sndstarl
          End If ! 
!-----------------------------------------------------------------------
!.... Left shock?
!-----------------------------------------------------------------------
        Else If ( lshockl ) Then
!
!.... Point is to the left of the left-shock: undisturbed left state
!
          If ( xprev .lt. xshklft .and. x .lt. xshklft ) Then
            lwritten = .true.
            xout = x
            rhoout = rholft
            prsout = prslft
            velout = vellft
            sieout = sielft
            sndout = sndlft
!
!.... Point and previous point straddle the left shock
!
          Else If ( xprev .lt. xshklft .and. x .gt. xshklft ) Then
            lwritten = .true.
            xout = xshklft           ! Left-shock location
            If ( ldoublevalue ) Then ! Write left- and right-values
              rhoout = rholft
              prsout = prslft
              velout = vellft
              sieout = sielft
              sndout = sndlft
!*!              Write(iunp1,103) xout, velout, prsout, rhoout, &
!*!                                     sieout, sndout, time
              rhoout = rhostarl
              prsout = pstar
              velout = velstarl
              sieout = siestarl
              sndout = sndstarl
!*!              Write(iunp1,103) xout, velout, prsout, rhoout, &
!*!                                     sieout, sndout, time
            Else                     ! Write average value
              rhoout = HALF * ( rholft + rhostarl )
              prsout = HALF * ( prslft + pstar    )
              velout = HALF * ( vellft + velstarl )
              sieout = HALF * ( sielft + siestarl )
              sndout = HALF * ( sndlft + sndstarl )
!*!              Write(iunp1,103) xout, velout, prsout, rhoout, &
!*!                                     sieout, sndout, time
            End If
            xout = x
            rhoout = rhostarl
            prsout = pstar
            velout = velstarl
            sieout = siestarl
            sndout = sndstarl
!
!.... Point is in the left, post-shock star-state region
!
          Else If ( xprev .gt. xshklft .and. x .gt. xshklft &
                                       .and. x .lt. xcont ) Then
            lwritten = .true.
            xout = x
            rhoout = rhostarl
            prsout = pstar
            velout = velstarl
            sieout = siestarl
            sndout = sndstarl
!
          End If ! xprev .lt. xshklft .and. x .lt. xshklft
        End If ! lrarel
!-----------------------------------------------------------------------
!.... Point and previous point straddle the contact
!-----------------------------------------------------------------------
        If ( xprev .lt. xcont .and. x .gt. xcont &
                              .and. ( .not. lwritten ) ) Then
          lwritten = .true.
          xout = xcont               ! Contact location
          If ( ldoublevalue ) Then   ! Write left- and right-values
            rhoout = rhostarl
            prsout = pstar
            velout = ustar
            sieout = siestarl
            sndout = sndstarl
!*!            Write(iunp1,103) xout, velout, prsout, rhoout, &
!*!                                   sieout, sndout, time
            rhoout = rhostarr
            prsout = pstar
            velout = ustar
            sieout = siestarr
            sndout = sndstarr
!*!            Write(iunp1,103) xout, velout, prsout, rhoout, &
!*!                                   sieout, sndout, time
          Else                       ! Write average value
            rhoout = HALF * ( rhostarl + rhostarr )
            prsout = pstar
            velout = ustar
            sieout = HALF * ( siestarl + siestarr )
            sndout = HALF * ( sndstarl + sndstarr )
!*!            Write(iunp1,103) xout, velout, prsout, rhoout, &
!*!                                   sieout, sndout, time
          End If
          xout = x
          rhoout = rhostarr
          prsout = pstar
          velout = velstarr
          sieout = siestarr
          sndout = sndstarr
        End If ! xprev .lt. xcont .and. x .gt. xcont
!-----------------------------------------------------------------------
!.... Is there a Right rarefaction in the solution?
!-----------------------------------------------------------------------
        If ( lrarer .and. ( .not. lwritten ) ) Then
!
!.... Point is right of the contact and left of the right-rarefact'n tail
!
          If ( x .gt. xcont .and. x .lt. xrtlrght &
                            .and. ( .not. lwritten ) ) Then
            lwritten = .true.
            xout = x
            rhoout = rhostarr
            prsout = pstar
            velout = velstarr
            sieout = siestarr
            sndout = sndstarr
!
!.... Point and previous point straddle right-rarefaction tail
!
          Else If ( xprev .lt. xrtlrght .and. x .gt. xrtlrght &
                                        .and. ( .not. lwritten ) ) Then
            lwritten = .true.
            xout = xrtlrght          ! Right-rarefaction tail
            rhoout = rhostarr
            prsout = pstar
            velout = velstarr
            sieout = siestarr
            sndout = sndstarr
!*!            Write(iunp1,103) xout, velout, prsout, rhoout, &
!*!                                   sieout, sndout, time
            xout = x
            rhoout = SPLEVAL( nrar, x, xrarr, rhorarr, brrarr, crrarr, &
                              drrarr )
            prsout = SPLEVAL( nrar, x, xrarr, prsrarr, bprarr, cprarr, &
                              dprarr )
            Call EOS_RIGHT_RHOP( rhoout, prsout, sieout, sndout, ierr )
            If ( ierr .ne. 0 ) ierr = 30
            velout = SPLEVAL( nrar, x, xrarr, velrarr, bvrarr, cvrarr, &
                              dvrarr )
!
!.... Point is inside the right-rarefaction fan: interpolate
!
          Else If ( xprev .gt. xrtlrght .and. x .gt. xrtlrght &
                                        .and. x .lt. xrhdrght &
                                        .and. ( .not. lwritten ) ) Then
            lwritten = .true.
            xout = x
            rhoout = SPLEVAL( nrar, x, xrarr, rhorarr, brrarr, crrarr, &
                              drrarr )
            prsout = SPLEVAL( nrar, x, xrarr, prsrarr, bprarr, cprarr, &
                              dprarr )
            Call EOS_RIGHT_RHOP( rhoout, prsout, sieout, sndout, ierr )
            If ( ierr .ne. 0 ) ierr = 31
            velout = SPLEVAL( nrar, x, xrarr, velrarr, bvrarr, cvrarr, &
                              dvrarr )
!
!.... Two points straddle leading right-rarefaction head
!
          Else If ( xprev .lt. xrhdrght .and. x .gt. xrhdrght  &
                                        .and. ( .not. lwritten ) ) Then
            lwritten = .true.
            xout = xrhdrght          ! Right-rarefaction head
            rhoout = rhorgt
            prsout = prsrgt
            velout = velrgt
            sieout = siergt
            sndout = sndrgt
!*!            Write(iunp1,103) xout, velout, prsout, rhoout, &
!*!                                   sieout, sndout, time
            xout = x
            rhoout = rhorgt
            prsout = prsrgt
            velout = velrgt
            sieout = siergt
            sndout = sndrgt
!
!.... Point is right of the leading right-rarefaction head
!
          Else If ( xprev .gt. xrhdrght .and. x .gt. xrhdrght  &
                                        .and. ( .not. lwritten ) ) Then
            lwritten = .true.
            xout = x
            rhoout = rhorgt
            prsout = prsrgt
            velout = velrgt
            sieout = siergt
            sndout = sndrgt
          End If ! lrarer
!-----------------------------------------------------------------------
!.... Right shock?
!-----------------------------------------------------------------------
        Else If ( lshockr .and. ( .not. lwritten ) ) Then
!
!.... Point is in the right, post-shock star-state region
!
          If ( x .gt. xcont .and. x .lt. xshkrght  &
                            .and. ( .not. lwritten ) ) Then
            lwritten = .true.
            xout = x
            rhoout = rhostarr
            prsout = pstar
            velout = velstarr
            sieout = siestarr
            sndout = sndstarr
!
!.... Two points straddle the right shock
!
          Else If ( xprev .lt. xshkrght .and. x .gt. xshkrght  &
                                        .and. ( .not. lwritten ) ) Then
            lwritten = .true.
            xout = xshkrght          ! Right-shock location
            If ( ldoublevalue ) Then ! Write left- and right-values
              rhoout = rhostarr
              prsout = pstar
              velout = velstarr
              sieout = siestarr
              sndout = sndstarr
!*!              Write(iunp1,103) xout, velout, prsout, rhoout, &
!*!                                     sieout, sndout, time
              rhoout = rhorgt
              prsout = prsrgt
              velout = velrgt
              sieout = siergt
              sndout = sndrgt
!*!              Write(iunp1,103) xout, velout, prsout, rhoout, &
!*!                                     sieout, sndout, time
            Else                     ! Write average value
              rhoout = HALF * ( rhorgt + rhostarr )
              prsout = HALF * ( prsrgt + pstar    )
              velout = HALF * ( velrgt + velstarr )
              sieout = HALF * ( siergt + siestarr )
              sndout = HALF * ( sndrgt + sndstarr )
!*!              Write(iunp1,103) xout, velout, prsout, rhoout, &
!*!                                     sieout, sndout, time
            End If
            xout = x
            rhoout = rhorgt
            prsout = prsrgt
            velout = velrgt
            sieout = siergt
            sndout = sndrgt
!
!.... Point is to the right of the right-shock: undisturbed right state
!
          Else If ( x .gt. xshkrght .and. ( .not. lwritten ) ) Then
            lwritten = .true.
            xout = x
            rhoout = rhorgt
            prsout = prsrgt
            velout = velrgt
            sieout = siergt
            sndout = sndrgt
          End If ! x .gt. xcont .and. x .lt. xshkrght  
!
        End If ! lrarer
!
!.... Write data to ASCII file
!**** start returning values here ****
!*!
        If (lwrite) Then
           Write(iunp1,103) x_m(i), velout, prsout, rhoout, &
                sieout, sndout, time
        End If
!*!        Write(iun,103)   xout, velout, prsout, rhoout, &
!*!                               sieout, sndout, time
! write output to common block arrays
        u_m(i)       = velout
        p_m(i)       = prsout
        rho_m(i)     = rhoout
        sie_m(i)     = sieout
        sound_m(i)   = sndout
!
      End Do ! i = 1, nx
!*!      Close(unit=iunp1)
!  
!-----------------------------------------------------------------------
!.... Error conditions
!-----------------------------------------------------------------------
  799 Continue
      If ( ierr .eq. 0 ) Go To 899 
      Call RIEMANN_ERROR( ierr, it, itmax, drhol, drhor, & 
                          plft, prght, pstar, scratch2)
  899 Continue
!*!      Close(unit=iun)
!*!      Close(unit=iunm1)
!
!-----------------------------------------------------------------------
!.... Format statements
!-----------------------------------------------------------------------
!
  101 Format('  Location      Velocity     Pressure     Density' &
      ,'       SIE         Sndspd      Time')
  103 Format(7(1x,1pe12.5))
!
  330 Format('   RIEMANN: Converged Left SHOCK, pre-ZEROIN:' &
     /' rholo = ',1pe12.5,' non-dim rhoshklo  = ',1pe12.5    &
     /' rhohi = ',1pe12.5,' non-dim rhoshkhi  = ',1pe12.5)
  340 Format('   RIEMANN: Converged Left Rarefaction, pre-ZEROIN:' &
     /' drholo = ',1pe12.5,' non-dim pres_lo = ',1pe12.5           &
     /' drhohi = ',1pe12.5,' non-dim pres_hi = ',1pe12.5)
  350 Format('   RIEMANN: Converged Right SHOCK, pre-ZEROIN:' &
      /' rholo = ',1pe12.5,' non-dim rhoshklo  = ',1pe12.5    &
      /' rhohi = ',1pe12.5,' non-dim rhoshkhi  = ',1pe12.5)
  360 Format('   RIEMANN: Converged Right Rarefaction, pre-ZEROIN:' &
      /' drholo = ',1pe12.5,' non-dim pres_lo = ',1pe12.5           &
      /' drhohi = ',1pe12.5,' non-dim pres_hi = ',1pe12.5)
!
  400 Format(/' Wave                      Position      Speed' &
             /' ----                      --------      -----')
  410 Format(' Left Shock:             ',1pe12.5,1x,1pe12.5)
  420 Format(' Left Rarefaction Head:  ',1pe12.5,1x,1pe12.5)
  430 Format(' Left Rarefaction Tail:  ',1pe12.5,1x,1pe12.5)
  440 Format(' Contact:                ',1pe12.5,1x,1pe12.5)
  450 Format(' Right Shock:            ',1pe12.5,1x,1pe12.5)
  460 Format(' Right Rarefaction Tail: ',1pe12.5,1x,1pe12.5)
  470 Format(' Right Rarefaction Head: ',1pe12.5,1x,1pe12.5)
!
  501 Format('** LEFT  STATE: rho = ',1pe9.2,' prs = ',1pe9.2, &
             ' sie = ',1pe9.2,' snd = ',1pe9.2, ' vel = ',1pe9.2)
  502 Format('** RIGHT STATE: rho = ',1pe9.2,' prs = ',1pe9.2, &
             ' sie = ',1pe9.2,' snd = ',1pe9.2, ' vel = ',1pe9.2)
  503 Format('** rholft = ',1pe9.2,' sielft = ',1pe9.2, &
               ' prslft = ',1pe9.2,' sndlft = ',1pe9.2)
  505 Format('** wlft1 = ',1pe9.2)
  506 Format('** rhorgt = ',1pe9.2,' siergt = ',1pe9.2, &
               ' prsrgt = ',1pe9.2,' sndrgt = ',1pe9.2)
  507 Format('** wrght1 = ',1pe9.2)
  513 Format('** pstar1 = ',1pe9.2)
  514 Format('** pstar2 = ',1pe9.2)
  520 Format('** RIEMANN: Secant iteration converged in ',i2, &
             ' iterations **')
  530 Format('** RIEMANN: Left  shock **')
  531 Format('** RIEMANN: Left  rarefaction **')
  532 Format('** RIEMANN: Right shock **')
  533 Format('** RIEMANN: Right rarefaction **')
  540 Format('** RIEMANN: RARELEFT return: drhol = ',1pe9.2,' **')
  551 Format('** RIEMANN: 1st call to GET_USTAR_LEFT:  ustrl1 = ' &
            ,1pe9.2,' **')
  552 Format('** RIEMANN: 1st call to GET_USTAR_RIGHT: ustrr1 = ' &
            ,1pe9.2,' **')
  553 Format('** RIEMANN: call ',i2,' to GET_USTAR_LEFT:  ustrl2 = ' &
            ,1pe9.2,' **')
  554 Format('** RIEMANN: call ',i2,' to GET_USTAR_RIGHT: ustrr2 = ' &
            ,1pe9.2,' **')
!
  600 Format(/'itn    uLk-1     uRk-1     p*k-1     uLk       uRk   '  &
      ,'    p*k       u_err     p_err'                                 &
             /'---  --------- --------- --------- --------- ---------' &
      ,' --------- --------- ---------')
  602 Format(1x,i2,1x,8(1x,1pe9.2))
!
  703 Format(/'Left state:         rhol      prsl      vell      siel  ' &
      ,' sndl'                                                           &
             /'---------------- --------- --------- --------- ---------' &
      ,' ---------')
  705 Format(15x,5(1x,1pe9.2))
  713 Format(/'Left pre-shock:     rhol      prsl      vell      siel  ' &
      ,' sndl'                                                           &
             /'---------------- --------- --------- --------- ---------' &
      ,' ---------')
  714 Format(/'Left post-shock: rhostarl  prsstarl  velstarl  siestarl ' &
      ,' sndstarl'                                                       &
             /'---------------- --------- --------- --------- ---------' &
      ,' ---------')
  723 Format(/' i     xrarl    rhorarl   prsrarl   velrarl   sierarl '   &
      ,' sndrarl'                                                        &
             /'---- --------- --------- --------- --------- ---------'   &
      ,' ---------')
  725 Format(i4,1x,6(1x,1pe9.2))
  733 Format(/'Left post-rarfn: rhostarl  prsstarl  velstarl  siestarl ' &
      ,' sndstarl'                                                       &
             /'---------------- --------- --------- --------- ---------' &
      ,' ---------')
  743 Format(/'Rght post-shock: rhostarr  prsstarr  velstarr  siestarr ' &
      ,' sndstarr'                                                       &
             /'---------------- --------- --------- --------- ---------' &
      ,' ---------')
  744 Format(/'Rght pre-shock:  rhostarr  prsstarr  velstarr  siestarr ' &
      ,' sndstarr'                                                       &
             /'---------------- --------- --------- --------- ---------' &
      ,' ---------')
  753 Format(/'Rght post-rarfn: rhostarr  prsstarr  velstarr  siestarr ' &
      ,' sndstarr'                                                       &
             /'---------------- --------- --------- --------- ---------' &
      ,' ---------')
  755 Format(15x,5(1x,1pe9.2))
  763 Format(/' i     xrarr    rhorarr   prsrarr   velrarr   sierarr '   &
      ,' sndrarr'                                                        &
             /'---- --------- --------- --------- --------- ---------'   &
      ,' ---------')
  765 Format(i4,1x,6(1x,1pe9.2))
  773 Format(/'Right state:        rhor      prsr      velr      sier  ' &
      ,' sndr'                                                           &
             /'---------------- --------- --------- --------- ---------' &
      ,' ---------')
  775 Format(15x,5(1x,1pe9.2))
!
!-----------------------------------------------------------------------
!
      Return
      End 
!
! End of Subroutine RIEMANN
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
! Start of Subroutine GET_USTAR_LEFT
!
      Subroutine GET_USTAR_LEFT( xd0, time, velstar, ldebugin, ierr )
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!                                                                      c
! This subroutine takes the given LEFT state and an estimate of the    c
! star-state pressure, pstar, and returns a value of the corresponding c
! star-state particle velocity.  This value depends on the implicit    c
! wave structure: If pstar > prslft, then the left wave is a shock,    c
! while if pstar <= prslft, then the left wave is a rarefaction.       c
!                                                                      c
!   Called by: RIEMANN   Calls: BRACKET_LEFT_SHOCK                     c
!                               BRACKET_LEFT_RARE                      c
!                               JUMPLEFT                               c
!                               RARELEFT                               c
!                               ZEROIN                                 c
!                               RARELEFTSOLN                           c
!                                                                      c
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
      Implicit none
!
!.... Include files
!
      Include "param.h"
!
!.... Call list variables
!
      Double Precision :: xd0             ! x-location of diaphragm  (cm)
      Double Precision :: time            ! simulation time          (s)
      Double Precision :: velstar         ! left star-state velocity (cm/s)
!
      Logical :: ldebugin                 ! debug flag
!
!.... Local variables
!
      Integer :: i                        ! index
      Integer :: ierr                     ! Error flag
      Integer :: iun1                     ! file unit number
!
      Double Precision :: rhomin          ! min density in root-finder
      Double Precision :: rhomax          ! max density in root-finder
      Double Precision :: rhomin2         ! min density in root-finder
      Double Precision :: rhomax2         ! max density in root-finder
      Double Precision :: rhotol          ! tolerance for sol'n in root-finder
      Double Precision :: rhostar         ! sol'n density in star-state
      Double Precision :: siestar         ! sol'n SIE     in star-state
      Double Precision :: sndstar         ! sol'n snd spd in star-state
!
      Double Precision :: drhol           ! delta-rho for left  rarefaction integration
      Double Precision :: drholo          ! bracketing value (lower) of drho
      Double Precision :: drhohi          ! bracketing value (upper) of drho
      Double Precision :: drholo2         ! bracketing value (lower) of drho
      Double Precision :: drhohi2         ! bracketing value (upper) of drho
      Double Precision :: prstol          ! integration tolerance on the pressure
      Double Precision :: ushkl           ! speed of left-going  shock
!
      Double Precision :: rhoshklo        ! JUMPLEFT value at rhomin
      Double Precision :: rhoshkhi        ! JUMPLEFT value at rhomax
      Double Precision :: prsndlo         ! RARELEFT value at drholo
      Double Precision :: prsndhi         ! RARELEFT value at drhohi
      Double Precision :: scratch         ! scratch scalar
!
      Double Precision :: xrarl(1:nrar)   ! left rarefctn position   (cm)
      Double Precision :: velrarl(1:nrar) ! left rarefctn velocity   (cm/s)
      Double Precision :: rhorarl(1:nrar) ! left rarefctn density    (g/cm3)
      Double Precision :: prsrarl(1:nrar) ! left rarefctn pressure   (dyne/cm2)
      Double Precision :: sierarl(1:nrar) ! left rarefctn SIE        (erg/g)
      Double Precision :: sndrarl(1:nrar) ! left rarefctn snd speed  (cm/s)
!
      Logical :: ldebug                   ! print-to-file   debug flag 
!
!.... Functions called
!
      Double Precision :: JUMPLEFT        ! Evaluate left shock jump conditions
      Double Precision :: RARELEFT        ! Evaluate left rarefaction delta-density value
      Double Precision :: ZEROIN          ! Nonlinear 1D root-finder function
      External JUMPLEFT
      External RARELEFT
      External ZEROIN
!
      Double Precision :: rhol            ! left  density       in common block
      Double Precision :: siel            ! left  SIE           in common block
      Double Precision :: prsl            ! left  pressure      in common block
      Double Precision :: sndl            ! left  sound speed   in common block
      Double Precision :: vell            ! left  velocity      in common block
      Double Precision :: rhor            ! right density       in common block
      Double Precision :: sier            ! right SIE           in common block
      Double Precision :: prsr            ! right pressure      in common block
      Double Precision :: sndr            ! right sound speed   in common block
      Double Precision :: velr            ! right velocity      in common block
      Double Precision :: pstar           ! star-state pressure
      Integer :: nrar                     ! # of pts in rarfctn in common block
!
      Common / left_state  / rhol, siel, prsl, sndl, vell
      Common / right_state / rhor, sier, prsr, sndr, velr
      Common / star_state  / pstar
      Common / num_rare    / nrar
!
!-----------------------------------------------------------------------
!
      ierr = 0
      ldebug = ldebugin
      iun1 = 30
!*!      If ( ldebug ) Open(unit=iun1,file='get_ustar_left.dbg', &
!*!                         status='unknown')
      If ( pstar .le. ZERO ) ierr = 1
      If ( ierr .ne. 0 ) Go To 799
      If ( rhol .le. ZERO ) ierr = 2
      If ( ierr .ne. 0 ) Go To 799
!.......................................................................
!.... If pstar > prsl, then the left wave is a shock
!.......................................................................
      If ( pstar .gt. prsl ) Then
!*!        If ( ldebug ) Write(iun1,310)
!
!.... For Shyue problem:
!        rhomax  = THREE * Max( rhol, rhor ) ! THREE might not be big enough
!
!        rhomin  = HALF * Min( rhol, rhor )
!        rhomax  = HALF * Max( rhol, rhor ) ! For Banks problem
!
!        rhotol  = EPS12
!        If ( ldebug ) Then
!          rhoshklo = JUMPLEFT( rhomin )
!          rhoshkhi = JUMPLEFT( rhomax )
!          Write(iun1,330) rhomin, rhoshklo, rhomax, rhoshkhi
!          scratch = rhoshklo * rhoshkhi
!          If ( scratch .ge. ZERO ) ierr = 3
!        End If 
!        If ( ierr .ne. 0 ) Go To 799
!
!.... Bracket a zero of JUMPLEFT:
!
        rhomin = Min( rhol, rhor )
        rhomax = Max( rhol, rhor )
        Call BRACKET_LEFT_SHOCK( rhomin, rhomax, rhomin2, rhomax2, &
                                 ldebug, ierr )
        If ( ierr .ne. 0 ) ierr = 3
        If ( ierr .ne. 0 ) Go To 799
        rhomin = rhomin2
        rhomax = rhomax2
        rhotol = EPS12
        rhostar = ZEROIN( rhomin, rhomax, JUMPLEFT, rhotol )
!
!.... Evaluate left-going shock speed 
! -->  ** THE SIGN OF SQRT MIGHT BE WRONG ** <--
!
! original
        ushkl = vell - Sqrt( ( rhostar / rhol ) &
              * ( ( pstar - prsl ) / ( rhostar - rhol ) ) )
! new
!        ushkl = vell + Sqrt( ( rhostar / rhol ) &
!              * ( ( pstar - prsl ) / ( rhostar - rhol ) ) )
!
!.... Evaluate u^{*}_{L} from jump conditions
! -->  ** THE SIGN OF SQRT MIGHT BE WRONG ** <--
!
! original
        velstar = ushkl + Sqrt( ( rhol / rhostar ) &
                * ( ( pstar - prsl ) / ( rhostar - rhol ) ) ) 
! new
!        velstar = ushkl - Sqrt( ( rhol / rhostar ) &
!                * ( ( pstar - prsl ) / ( rhostar - rhol ) ) ) 
!.......................................................................
!.... If pstar <= prslft, then the left wave is a rarefaction
!.......................................................................
      Else
!*!        If ( ldebug ) Write(iun1,320)
!
!        drholo = HALF * Abs( rhol - rhor ) / Dble( nrar )
!        If ( drholo .lt. EPS8 ) drholo = EPS6
!c        drhohi = Abs( rhol - rhor ) / Dble( nrar )
!        drhohi = TWO * Abs( rhol - rhor ) / Dble( nrar )
!        If ( drhohi .lt. EPS8 ) &
!          drhohi = Max( rhol, rhor ) / Dble( nrar )
!        prstol = EPS12
!        If ( ldebug ) Then
!          prsndlo = RARELEFT( drholo )
!          prsndhi = RARELEFT( drhohi )
!          scratch = prsndlo * prsndhi
!          Write(iun1,340) drholo, prsndlo, drhohi, prsndhi
!          If ( scratch .ge. ZERO ) ierr = 4
!        End If 
!        If ( ierr .ne. 0 ) Go To 799
!
        drholo = Max( EPS4, Abs( rhol - rhor ) / Dble( nrar ) )
        drholo = 0.01d0 * drholo
        drhohi = ( Max( rhol, rhor ) - Sqrt( EPS4 ) )  / Dble( nrar )
        drhohi = TWO * drholo
        Call BRACKET_LEFT_RARE( drholo, drhohi, drholo2, drhohi2, &
                                ldebug, ierr )
        If ( ierr .ne. 0 ) ierr = 4
        If ( ierr .ne. 0 ) Go To 799
        drholo = drholo2
        drhohi = drhohi2
        prstol = EPS12
        drhol = ZEROIN( drholo, drhohi, RARELEFT, prstol )
        If ( drhol .eq. ZERO ) ierr = 5
        If ( ierr .ne. 0 ) Go To 799
!*!        If ( ldebug ) Write(iun1,302) drhol 
!
!.... Evaluate the solution thru the rarefaction to get the left star-state
!
        Call RARELEFTSOLN( nrar, drhol, xd0, time, xrarl, velrarl, &
                           rhorarl, prsrarl, sierarl, sndrarl, ierr )
        If ( ierr .ne. 0 ) ierr = 6
        If ( ierr .ne. 0 ) Go To 799
!*!        If ( ldebug ) Then
!*!          Write(iun1,701)
!*!          Do i = 1, nrar
!*!            Write(iun1,703) i, xrarl(i),rhorarl(i),prsrarl(i), &
!*!                               velrarl(i),sierarl(i),sndrarl(i)
!*!          End Do ! i
!*!        End If ! ldebug
        velstar = velrarl(nrar)
!
      End If ! pstar .gt. prslft
!*!      If ( ldebug ) Write(iun1,303) velstar
!  
!-----------------------------------------------------------------------
!.... Error conditions
!-----------------------------------------------------------------------
  799 Continue
      If ( ierr .eq. 0 ) Go To 899 
      Write(*,900) ierr
      Go To ( 801, 802, 803, 804, 805, 806 ) ierr
  801 Write(*,901) pstar
      Go To 899 
  802 Write(*,902) rhol
      Go To 899 
  803 Write(*,903) 
!  803 Write(*,903) scratch
      Go To 899 
  804 Write(*,904)
!  804 Write(*,904) scratch
      Go To 899 
  805 Write(*,905) drhol
      Go To 899 
  806 Write(*,906)
      Go To 899 
!
!.... Exit
!
  899 Continue
!*!      If ( ldebug ) Close(iun1)
!
!-----------------------------------------------------------------------
!.... Format statements
!-----------------------------------------------------------------------
!
  302 Format('--> Return from ZEROIN: drhol = ',1pe12.5,' <--')
  303 Format('--> Return from GET_USTAR_LEFT: velstar = ' &
            ,1pe12.5,' <--')
  310 Format('   GET_USTAR_LEFT: Left Shock')
  320 Format('   GET_USTAR_LEFT: Left Rarefaction')
!
  330 Format('   GET_USTAR_LEFT: Left SHOCK, pre-ZEROIN:' &
      /' rhomin = ',1pe12.5,' non-dim rho_lo  = ',1pe12.5 &
      /' rhomax = ',1pe12.5,' non-dim rho_hi  = ',1pe12.5)
  340 Format('   GET_USTAR_LEFT: Left Rarefaction, pre-ZEROIN:' &
      /' drholo = ',1pe12.5,' non-dim pres_lo = ',1pe12.5       &
      /' drhohi = ',1pe12.5,' non-dim pres_hi = ',1pe12.5)
!
  701 Format(/' i     xrarl    rhorarl   prsrarl   velrarl   sierarl ' &
      ,' sndrarl'                                                      &
             /'---- --------- --------- --------- --------- ---------' &
      ,' ---------')
  703 Format(i4,1x,6(1x,1pe9.2))
!
  900 Format('** GET_USTAR_LEFT: FATAL ERROR   ierr = ',i2,' **')
  901 Format('** pstar = ',1pe12.5,' <= 0 **')
  902 Format('** rhol = ',1pe12.5,' < 0 **')
  903 Format('** Fatal error in BRACKET_LEFT_SHOCK **')
!  903 Format('** LEFT Shock ZEROIN bounds: rhoshklo * rhoshkhi = ' &
!      ,1pe12.5,' >= 0 **')
  904 Format('** Fatal error in BRACKET_LEFT_RARE **')
!  904 Format('** LEFT Rarefaction ZEROIN bounds: prsndlo * prsndhi = ' &
!      ,1pe12.5,' >= 0 **')
  905 Format('** Error return from LEFT Rarefaction: ZEROIN: drhol = ' &
      ,1pe12.5,' **')
  906 Format('** Error return from LEFT Rarefaction: RARELEFTSOLN **')
!
!-----------------------------------------------------------------------
!
      Return
      End
!
! End of Subroutine GET_USTAR_LEFT
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
! Start of Subroutine GET_USTAR_RIGHT
!
      Subroutine GET_USTAR_RIGHT( xd0, time, velstar, ldebugin, ierr )
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!                                                                      c
! This subroutine takes the given RIGHT state and an estimate of the   c
! star-state pressure, pstar, and returns a value of the corresponding c
! star-state particle velocity.  This value depends on the implicit    c
! wave structure: If pstar > prsrght, then the right wave is a shock,  c
! while if pstar <= prsrght, then the right wave is a rarefaction.     c
!                                                                      c
!   Called by: RIEMANN   Calls: BRACKET_RIGHT_SHOCK                    c
!                               BRACKET_RIGHT_RARE                     c
!                               JUMPRIGHT                              c
!                               RARERIGHT                              c
!                               ZEROIN                                 c
!                               RARERIGHTSOLN                          c
!                                                                      c
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
      Implicit none
!
!.... Include files
!
      Include "param.h"
!
!.... Call list variables
!
      Double Precision :: xd0             ! x-location of diaphragm  (cm)
      Double Precision :: time            ! simulation time          (s)
      Double Precision :: velstar         ! right star-state velocity (cm/s)
!
      Logical :: ldebugin                 ! debug flag
!
!.... Local variables
!
      Integer :: i                        ! index
      Integer :: ierr                     ! Error flag
      Integer :: iun1                     ! file unit number
!
      Double Precision :: rhomin          ! min density in root-finder
      Double Precision :: rhomax          ! max density in root-finder
      Double Precision :: rhomin2         ! min density in root-finder
      Double Precision :: rhomax2         ! max density in root-finder
      Double Precision :: rhotol          ! tolerance for sol'n in root-finder
      Double Precision :: rhostar         ! sol'n density in star-state
      Double Precision :: siestar         ! sol'n SIE     in star-state
      Double Precision :: sndstar         ! sol'n snd spd in star-state
      Double Precision :: ushkr           ! speed of right-going  shock
!
      Double Precision :: drhor           ! delta-rho for right rarefaction integration
      Double Precision :: drholo          ! bracketing value (lower) of drho
      Double Precision :: drhohi          ! bracketing value (upper) of drho
      Double Precision :: drholo2         ! bracketing value (lower) of drho
      Double Precision :: drhohi2         ! bracketing value (upper) of drho
      Double Precision :: prstol          ! integration tolerance on the pressure
!
      Double Precision :: rhoshklo        ! JUMPRIGHT value at rhomin
      Double Precision :: rhoshkhi        ! JUMPRIGHT value at rhomax
      Double Precision :: prsndlo         ! RARERIGHT value at drholo
      Double Precision :: prsndhi         ! RARERIGHT value at drhohi
      Double Precision :: scratch         ! scratch scalar
!
      Double Precision :: xrarr(1:nrar)   ! right rarefctn position   (cm)
      Double Precision :: velrarr(1:nrar) ! right rarefctn velocity   (cm/s)
      Double Precision :: rhorarr(1:nrar) ! right rarefctn density    (g/cm3)
      Double Precision :: prsrarr(1:nrar) ! right rarefctn pressure   (dyne/cm2)
      Double Precision :: sierarr(1:nrar) ! right rarefctn SIE        (erg/g)
      Double Precision :: sndrarr(1:nrar) ! right rarefctn snd speed  (cm/s)
!
!.... Functions called
!
      Double Precision :: JUMPRIGHT       ! Evaluate right shock jump conditions
      Double Precision :: RARERIGHT       ! Evaluate right rarefaction delta-density value
      Double Precision :: ZEROIN          ! Nonlinear 1D root-finder function
      External JUMPRIGHT
      External RARERIGHT
      External ZEROIN
!
      Logical :: ldebug                   ! debug flag 
!
      Double Precision :: rhol            ! left  density       in common block
      Double Precision :: siel            ! left  SIE           in common block
      Double Precision :: prsl            ! left  pressure      in common block
      Double Precision :: sndl            ! left  sound speed   in common block
      Double Precision :: vell            ! left  velocity      in common block
      Double Precision :: rhor            ! right density       in common block
      Double Precision :: sier            ! right SIE           in common block
      Double Precision :: prsr            ! right pressure      in common block
      Double Precision :: sndr            ! right sound speed   in common block
      Double Precision :: velr            ! right velocity      in common block
      Double Precision :: pstar           ! star-state pressure
      Integer :: nrar                     ! # of pts in rarfctn in common block
!
      Common / left_state  / rhol, siel, prsl, sndl, vell
      Common / right_state / rhor, sier, prsr, sndr, velr
      Common / star_state  / pstar
      Common / num_rare    / nrar
!
!-----------------------------------------------------------------------
!
      ierr = 0
      ldebug = ldebugin
      iun1 = 31
!*!      If ( ldebug ) Open(unit=iun1,file='get_ustar_right.dbg', &
!*!                         status='unknown')
      If ( pstar .le. ZERO ) ierr = 1
      If ( ierr .ne. 0 ) Go To 799
      If ( rhor .le. ZERO ) ierr = 2
      If ( ierr .ne. 0 ) Go To 799
!.......................................................................
!.... If pstar > prsr, then the right wave is a shock
!.......................................................................
      If ( pstar .gt. prsr ) Then
!*!        If ( ldebug ) Write(iun1,310)
!
!.... Attempt to bracket a zero of JUMPRIGHT:
!
!        rhomin  = HALF * Min( rhol, rhor )
!        rhomax  = THREE * Max( rhol, rhor ) ! THREE might not be big enough
        rhomin = Min( rhol, rhor )
        rhomax = Max( rhol, rhor )
        Call BRACKET_RIGHT_SHOCK( rhomin, rhomax, rhomin2, rhomax2, &
                                  ldebug, ierr )
        If ( ierr .ne. 0 ) ierr = 3
        If ( ierr .ne. 0 ) Go To 799
        rhomin = rhomin2
        rhomax = rhomax2
        rhotol = EPS12
        rhostar = ZEROIN( rhomin, rhomax, JUMPRIGHT, rhotol )
!
!.... Evaluate right-going shock speed 
!
        ushkr = velr + Sqrt( ( rhostar / rhor ) &
              * ( ( pstar - prsr ) / ( rhostar - rhor ) ) )
!
!.... Evaluate u^{*}_{R} from jump conditions
!
        velstar = ushkr - Sqrt( ( rhor / rhostar ) &
                * ( ( pstar - prsr ) / ( rhostar - rhor ) ) ) 
!.......................................................................
!.... If pstar <= prsrght, then the right wave is a rarefaction
!.......................................................................
      Else
!*!        If ( ldebug ) Write(iun1,320)
!
!.... Attempt to bracket a zero of RARERIGHT:
!
!        drholo = HALF * Abs( rhol - rhor ) / Dble( nrar )
!        If ( drholo .lt. EPS8 ) drholo = EPS6
!        drhohi = TWO * Abs( rhol - rhor ) / Dble( nrar )
!        If ( drhohi .lt. EPS8 ) &
!          drhohi = Max( rhol, rhor ) / Dble( nrar )
!        prstol = EPS12
!        If ( ldebug ) Then
!          prsndlo = RARERIGHT( drholo )
!          prsndhi = RARERIGHT( drhohi )
!          scratch = prsndlo * prsndhi
!          Write(iun1,340) drholo, prsndlo, drhohi, prsndhi
!          If ( scratch .ge. ZERO ) ierr = 4
!        End If 
!        If ( ierr .ne. 0 ) Go To 799
!
        drholo = Max( EPS4, Abs( rhol - rhor ) / Dble( nrar ) )
        drholo = 0.01d0 * drholo
        drhohi = ( Max( rhol, rhor ) - Sqrt( EPS4 ) )  / Dble( nrar )
        drhohi = TWO * drholo
        Call BRACKET_RIGHT_RARE( drholo, drhohi, drholo2, drhohi2, &
                                 ldebug, ierr )
        If ( ierr .ne. 0 ) ierr = 4
        If ( ierr .ne. 0 ) Go To 799
        drholo = drholo2
        drhohi = drhohi2
        prstol = EPS12
        drhor = ZEROIN( drholo, drhohi, RARERIGHT, prstol )
        If ( drhor .eq. ZERO ) ierr = 5
        If ( ierr .ne. 0 ) Go To 799
!*!        If ( ldebug ) Write(iun1,302) drhor
!
!.... Evaluate the solution thru the rarefaction to get the right star-state
!
        Call RARERIGHTSOLN( nrar, drhor, xd0, time, xrarr, velrarr, &
                           rhorarr, prsrarr, sierarr, sndrarr, ierr )
        If ( ierr .ne. 0 ) ierr = 6
        If ( ierr .ne. 0 ) Go To 799
!*!        If ( ldebug ) Then
!*!          Write(iun1,701)
!*!          Do i = 1, nrar
!*!            Write(iun1,703) i, xrarr(i),rhorarr(i),prsrarr(i), &
!*!                               velrarr(i),sierarr(i),sndrarr(i)
!*!          End Do ! i
!*!        End If ! ldebug
        velstar = velrarr(nrar)
!
      End If ! pstar .gt. prsrght
!*!      If ( ldebug ) Write(iun1,303) velstar
!  
!-----------------------------------------------------------------------
!.... Error conditions
!-----------------------------------------------------------------------
  799 Continue
      If ( ierr .eq. 0 ) Go To 899 
      Write(*,900) ierr
      Go To ( 801, 802, 803, 804, 805, 806 ) ierr
  801 Write(*,901) pstar
      Go To 899 
  802 Write(*,902) rhor
      Go To 899 
  803 Write(*,903)
!  803 Write(*,903) scratch
      Go To 899 
  804 Write(*,904)
!  804 Write(*,904) scratch
      Go To 899 
  805 Write(*,905) drhor
      Go To 899 
  806 Write(*,906)
      Go To 899 
!
!.... Exit
!
  899 Continue
!*!      If ( ldebug ) Close(iun1)
!
!-----------------------------------------------------------------------
!.... Format statements
!-----------------------------------------------------------------------
!
  302 Format('--> Return from ZEROIN: drhor = ',1pe12.5,' <--')
  303 Format('--> Return from GET_USTAR_RIGHT: velstar = ' &
            ,1pe12.5,' <--')
  310 Format('   GET_USTAR_RIGHT: Right Shock')
  320 Format('   GET_USTAR_RIGHT: Right Rarefaction')
!
  330 Format('   GET_USTAR_RIGHT: Right SHOCK, pre-ZEROIN:' &
      /' rhomin = ',1pe12.5,' non-dim rho_lo  = ',1pe12.5   &
      /' rhomax = ',1pe12.5,' non-dim rho_hi  = ',1pe12.5)
  340 Format('   GET_USTAR_RIGHT: Right Rarefaction, pre-ZEROIN:' &
      /' drholo = ',1pe12.5,' non-dim pres_lo = ',1pe12.5         &
      /' drhohi = ',1pe12.5,' non-dim pres_hi = ',1pe12.5)
!
  701 Format(/' i     xrarr    rhorarr   prsrarr   velrarr   sierarr ' &
      ,' sndrarr'                                                      &
             /'---- --------- --------- --------- --------- ---------' &
      ,' ---------')
  703 Format(i4,1x,6(1x,1pe9.2))
!
  900 Format('** GET_USTAR_RIGHT: FATAL ERROR   ierr = ',i2,' **')
  901 Format('** pstar = ',1pe12.5,' <= 0 **')
  902 Format('** rhor = ',1pe12.5,' < 0 **')
  903 Format('** Fatal error in BRACKET_RIGHT_SHOCK **')
!  903 Format('** RIGHT Shock ZEROIN bounds: rhoshklo * rhoshkhi = ' &
!      ,1pe12.5,' >= 0 **')
  904 Format('** Fatal error in BRACKET_RIGHT_RARE **')
!  904 Format('** RIGHT Rarefaction ZEROIN bounds: prsndlo * prsndhi = ' &
!      ,1pe12.5,' >= 0 **')
  905 Format('** Error return from RIGHT Rarefaction: ZEROIN: drhor = ' &
      ,1pe12.5,' **')
  906 Format('** Error return from RIGHT Rarefaction: RARERIGHTSOLN **')
!
!-----------------------------------------------------------------------
!
      Return
      End
!
! End of Subroutine GET_USTAR_RIGHT
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
! Start of Subroutine BRACKET_LEFT_SHOCK
!
      Subroutine BRACKET_LEFT_SHOCK( rholin,  rhorin,                 &
                                     rholout, rhorout, ldebugin, ierr )
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!                                                                      c
! This subroutine takes the input values of left and right density,    c
! "rholin" and "rhorin", and determines whether those arguments        c
! bracket a zero of the JUMPLEFT function, which is used in a ZEROIN   c
! call with JUMPLEFT in the calling routine.  If so, then those values c
! are returned in the output variables "rholout" and "rhorout".        c
! If not, then the "rhol" or "rhor" are modified, in a simplistic way, c
! in a search for a pair of values that will bracket a zero;  this     c
! modification is attempted for at most "itmax" iterations.            c
!                                                                      c
!   Called by: GET_USTAR_LEFT   Calls: JUMPLEFT                        c
!                                                                      c
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
      Implicit none
!
!.... Include files
!
      Include "param.h"
!
!.... Call list variables
!
      Integer :: ierr                     ! error flag
!
      Double Precision :: rholin          ! input  left  density
      Double Precision :: rhorin          ! input  right density
      Double Precision :: rholout         ! output left  density
      Double Precision :: rhorout         ! output right density
!
      Logical :: ldebugin                 ! debug flag
!
!.... Local variables
!
      Integer :: it                       ! iteration counter
      Integer :: itmax                    ! max number of iterations
      Integer :: iun1                     ! debug file unit number
!
      Double Precision :: rhol            ! left  density
      Double Precision :: rhor            ! right density
      Double Precision :: rhomin          ! min density for root-finder
      Double Precision :: rhomax          ! max density max root-finder
      Double Precision :: rhotmp          ! temporary density value
      Double Precision :: rhotol          ! tolerance for density difference
      Double Precision :: rhoshklo        ! JUMPLEFT value at rhomin
      Double Precision :: rhoshkhi        ! JUMPLEFT value at rhomax
      Double Precision :: scratch         ! = rhoshklo * rhoshkhi
!
      Logical :: ldebug                   ! .true. => print debug statements
      Logical :: lbracketed               ! .true. => root is bracketed
!
!.... Functions called
!
      Double Precision :: JUMPLEFT        ! Evaluate left shock jump conditions
!
!-----------------------------------------------------------------------
!
      ierr = 0
      iun1 = 32
      ldebug = ldebugin
!*!      If ( ldebug ) Open(unit=iun1,file='bracket_left_shock.dbg', &
!*!                         status='unknown')
!-----------------------------------------------------------------------
      itmax  = 10
      rhotol = 1.e-2
!-----------------------------------------------------------------------
      rhol = rholin
      rhor = rhorin
      If ( rhol .le. ZERO ) ierr = 1
      If ( ierr .ne. 0 ) Go To 799
      If ( rhor .le. ZERO ) ierr = 2
      If ( ierr .ne. 0 ) Go To 799
!
!.... Attempts to ensure bracketing values of the density sent to 
!     the ZEROIN call with JUMPLEFT that determines the LEFT SHOCK SPEED
!
      rhomin = 0.9d0 * Min( rhol, rhor )
      rhomax = 1.1d0 * Max( rhol, rhor )
      lbracketed = .false.
      it = 0
!*!      If ( ldebug ) Write(iun1,101)
!*!      If ( ldebug ) Write(iun1,102) it, rhomin, ZERO, rhomax, ZERO
      Do While ( ( .not. lbracketed ) .and. ( it .lt. itmax ) )
        it = it + 1
        rhoshklo = JUMPLEFT( rhomin )
        rhoshkhi = JUMPLEFT( rhomax )
!*!        If ( ldebug ) Write(iun1,102) &
!*!          it, rhomin, rhoshklo, rhomax, rhoshkhi
        scratch = rhoshklo * rhoshkhi
!
!.... rhoshklo * rhoshkhi = 0
!
        If ( scratch .eq. ZERO ) Then
          If ( rhoshkhi .ne. ZERO ) Then       ! rhoshklo = 0 => increase rhomin
            rhomin = 1.1d0 * rhomin
          Else If ( rhoshklo .ne. ZERO ) Then  ! rhoshkhi = 0 => decrease rhomax
            rhomax = 0.9d0 * rhomax
          Else                                 ! rhoshklo = rhoshkhi = 0 => ERROR
            ierr = 3
          End If
!
!.... rhoshklo * rhoshkhi > 0
!
!     Here, the two function values have the same sign: (1) reset the abscissa with 
!     the _larger_ absolute ordinate to be the abscissa with the _smaller_ absolute; 
!     (2) re-assign that smaller-absolute-abscissa ordinate to the linearly extrapolated
!     zero-ordinate abscissa, which is "nudged" by one-tenth of the delta-abcissa-value
!
        Else If ( scratch .gt. ZERO ) Then
          If ( rhoshklo .gt. ZERO .and. rhoshkhi .gt. ZERO ) Then
            If ( rhoshklo .lt. rhoshkhi ) Then ! 0 < rhoshklo < rhoshkhi
              rhotmp = rhomin
              rhomin = ( rhotmp * rhoshkhi - rhomax * rhoshklo ) &
                     / ( rhoshkhi - rhoshklo )
              rhomax = rhotmp
              rhotmp = rhomax - rhomin
              rhomin = rhomin - TENTH * rhotmp
            Else                               ! 0 < rhoshkhi < rhoshklo
              rhotmp = rhomax
              rhomax = ( rhomin * rhoshkhi - rhotmp * rhoshklo ) &
                     / ( rhoshkhi - rhoshklo )
              rhomin = rhotmp
              rhotmp = rhomax - rhomin
              rhomax = rhomax + TENTH * rhotmp
            End If
          Else
            If ( rhoshklo .lt. rhoshkhi ) Then ! rhoshklo < rhoshkhi < 0
              rhotmp = rhomax
              rhomax = ( rhomin * rhoshkhi - rhotmp * rhoshklo ) &
                     / ( rhoshkhi - rhoshklo )
              rhomin = rhotmp
              rhotmp = rhomax - rhomin
              rhomax = rhomax + TENTH * rhotmp
            Else                               ! rhoshkhi < rhoshklo < 0
              rhotmp = rhomin
              rhomin = ( rhotmp * rhoshkhi - rhomax * rhoshklo ) &
                     / ( rhoshkhi - rhoshklo )
              rhomax = rhotmp
              rhotmp = rhomax - rhomin
              rhomin = rhomin - TENTH * rhotmp
            End If
          End If
!
!.... rhoshklo * rhoshkhi < 0
!
        Else
          lbracketed = .true.
        End If
        If ( ierr .ne. 0 ) Go To 799           ! ierr = 3
      End Do 
!
!..... Solution that brackets the root has been found
!
      If ( lbracketed ) Then
        If ( rholin .lt. rhorin ) Then
          rholout = rhomin
          rhorout = rhomax
        Else
          rholout = rhomax
          rhorout = rhomin
        End If
!*!        If ( ldebug ) Write(iun1,200) 
      End If
!
      If ( it .ge. itmax ) ierr = 4
      If ( ierr .ne. 0 ) Go To 799
!  
!-----------------------------------------------------------------------
!.... Error conditions
!-----------------------------------------------------------------------
  799 Continue
      If ( ierr .eq. 0 ) Go To 899 
      Write(*,900) ierr
      Go To ( 801, 802, 803, 804 ) ierr
  801 Write(*,901) rhol
      Go To 899 
  802 Write(*,902) rhor
      Go To 899 
  803 Write(*,903) rhomin, rhoshklo, rhomax, rhoshkhi
      Go To 899 
  804 Write(*,904) it, itmax
      Go To 899 
!
!.... Exit
!
  899 Continue
!*!      If ( ldebug ) Close(iun1)
!
!-----------------------------------------------------------------------
!.... Format statements
!-----------------------------------------------------------------------
!
  101 Format(/' it   rhomin   rhoshklo   rhomax   rhoshkhi'  &
             /'---- --------- --------- --------- ---------')
  102 Format(i3,1x,4(1x,1pe9.2))
  200 Format('** Left shock bracketing values FOUND **')
!
  900 Format('** BRACKET_LEFT_SHOCK: FATAL ERROR   ierr = ',i2,' **')
  901 Format('** rhol = ',1pe12.5,' <= 0 **')
  902 Format('** rhor = ',1pe12.5,' <= 0 **')
  903 Format('** rhomin = ',1pe12.5,' rhoshklo = ',1pe12.5, &
               ' rhomax = ',1pe12.5,' rhoshkhi = ',1pe12.5)
  904 Format('** it = ',i3,' > ',i3,' = itmax **')
!
!-----------------------------------------------------------------------
!
      Return
      End
!
! End of Subroutine BRACKET_LEFT_SHOCK
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
! Start of Function JUMPLEFT
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!                                                                      c
! LEFT material: Evaluate the function for ZEROIN based on the         c
!     shock jump conditions given in Eq. (3.2) of Banks                c
!                                                                      c
!   Called by: GET_USTAR_LEFT  Calls: EOS_LEFT_RHOP                    c
!              ZEROIN                                                  c
!                                                                      c
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
      Double Precision Function JUMPLEFT( rho )
!
      Implicit none
!
!.... Include files
!
      Include "param.h"
!
      Double Precision :: rho             ! density
!
!.... Local variables
!
      Integer :: ierr                     ! Error flag
!
      Double Precision :: rhoval          ! density (sent) 
      Double Precision :: sieval          ! SIE(rhoval,pstar)
      Double Precision :: sndval          ! sound_speed(rhoval,pstar)
      Double Precision :: scratch         ! (1/2)(p*-p_l)/(rho_l-rho)
      Double Precision :: term1           ! 1st term in function
      Double Precision :: term2           ! 2ne term in function
!
      Double Precision :: rhol            ! left density
      Double Precision :: siel            ! left SIE
      Double Precision :: prsl            ! left pressure
      Double Precision :: sndl            ! left sound speed
      Double Precision :: vell            ! left velocity
      Double Precision :: pstar           ! star-state pressure
!
      Common / left_state / rhol, siel, prsl, sndl, vell
      Common / star_state / pstar
!
!-----------------------------------------------------------------------
!
      ierr = 0
      If ( pstar .le. ZERO ) ierr = 1
      If ( ierr .ne. 0 ) Go To 799
      If ( rhol .le. ZERO ) ierr = 2
      If ( ierr .ne. 0 ) Go To 799
!      
      rhoval = rho
      Call EOS_LEFT_RHOP( rhoval, pstar, sieval, sndval, ierr)
      If ( ierr .ne. 0 ) ierr = 3
      If ( ierr .ne. 0 ) Go To 799
!
      scratch = HALF * ( pstar - prsl ) / ( rho - rhol )
      term1 = siel   + ( prsl  / rhol ) + ( rho  / rhol ) * scratch
      term2 = sieval + ( pstar / rho  ) + ( rhol / rho  ) * scratch
!
      JUMPLEFT = term1 - term2
!
!-----------------------------------------------------------------------
!.... Error conditions
!-----------------------------------------------------------------------
  799 Continue
      If ( ierr .eq. 0 ) Go To 899 
!
!.... Here, there's an error, so set the function to zero before returning
!
      JUMPLEFT = ZERO
      Write(*,900) ierr
      Go To ( 801, 802, 803 ) ierr
  801 Write(*,901) pstar
      Go To 899 
  802 Write(*,902) rhol
      Go To 899 
  803 Write(*,903)  
      Go To 899 
!
!.... Exit
!
  899 Continue
!
!-----------------------------------------------------------------------
!.... Format statements
!-----------------------------------------------------------------------
!
  900 Format('** JUMPLEFT: FATAL ERROR   ierr = ',i2,' **')
  901 Format('**   pstar = ',1pe12.5,' <= 0 **')
  902 Format('**   rhol  = ',1pe12.5,' <= 0 **')
  903 Format('**   Error return from EOS_LEFT_RHOP **')
!
!-----------------------------------------------------------------------
!
      End
!
! End of Function JUMPLEFT
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
! Start of Subroutine BRACKET_RIGHT_SHOCK
!
      Subroutine BRACKET_RIGHT_SHOCK( rholin,  rhorin,                 &
                                      rholout, rhorout, ldebugin, ierr )
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!                                                                      c
! This subroutine takes the input values of left and right density,    c
! "rholin" and "rhorin", and determines whether those arguments        c
! bracket a zero of the JUMPRIGHT function, which is used in a ZEROIN  c
! call with JUMPRIGHT in the calling routine.  If so, then those valuesc
! are returned in the output variables "rholout" and "rhorout".        c
! If not, then the "rhol" or "rhor" are modified, in a simplistic way, c
! in a search for a pair of values that will bracket a zero;  this     c
! modification is attempted for at most "itmax" iterations.            c
!                                                                      c
!   Called by: GET_USTAR_RIGHT   Calls: JUMPRIGHT                      c
!                                                                      c
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
      Implicit none
!
!.... Include files
!
      Include "param.h"
!
!.... Call list variables
!
      Integer :: ierr                     ! error flag
!
      Double Precision :: rholin          ! input  left  density
      Double Precision :: rhorin          ! input  right density
      Double Precision :: rholout         ! output left  density
      Double Precision :: rhorout         ! output right density
!
      Logical :: ldebugin                 ! debug flag
!
!.... Local variables
!
      Integer :: it                       ! iteration counter
      Integer :: itmax                    ! max number of iterations
      Integer :: iun1                     ! debug file unit number
!
      Double Precision :: rhol            ! left  density
      Double Precision :: rhor            ! right density
      Double Precision :: rhomin          ! min density for root-finder
      Double Precision :: rhomax          ! max density max root-finder
      Double Precision :: rhotmp          ! temporary density value
      Double Precision :: rhotol          ! tolerance for density difference
      Double Precision :: rhoshklo        ! JUMPLEFT value at rhomin
      Double Precision :: rhoshkhi        ! JUMPLEFT value at rhomax
      Double Precision :: scratch         ! = rhoshklo * rhoshkhi
!
      Logical :: ldebug                   ! .true. => print debug statements
      Logical :: lbracketed               ! .true. => root is bracketed
!
!.... Functions called
!
      Double Precision :: JUMPRIGHT       ! Evaluate right shock jump conditions
!
!-----------------------------------------------------------------------
!
      ierr = 0
      iun1 = 33
      ldebug = ldebugin
!*!      If ( ldebug ) Open(unit=iun1,file='bracket_right_shock.dbg', &
!*!                         status='unknown')
!-----------------------------------------------------------------------
      itmax  = 10
      rhotol = 1.e-2
!-----------------------------------------------------------------------
      rhol = rholin
      rhor = rhorin
      If ( rhol .le. ZERO ) ierr = 1
      If ( ierr .ne. 0 ) Go To 799
      If ( rhor .le. ZERO ) ierr = 2
      If ( ierr .ne. 0 ) Go To 799
!
!.... Attempts to ensure bracketing values of the density sent to 
!     the ZEROIN call with JUMPRIGHT that determines the RIGHT SHOCK SPEED
!
      rhomin = 0.9d0 * Min( rhol, rhor )
      rhomax = 1.1d0 * Max( rhol, rhor )
      lbracketed = .false.
      it = 0
!*!      If ( ldebug ) Write(iun1,101)
!*!      If ( ldebug ) Write(iun1,102) it, rhomin, ZERO, rhomax, ZERO
      Do While ( ( .not. lbracketed ) .and. ( it .lt. itmax ) )
        it = it + 1
        rhoshklo = JUMPRIGHT( rhomin )
        rhoshkhi = JUMPRIGHT( rhomax )
!*!        Write(iun1,102) it, rhomin, rhoshklo, rhomax, rhoshkhi
        scratch = rhoshklo * rhoshkhi
!
!.... rhoshklo * rhoshkhi = 0
!
        If ( scratch .eq. ZERO ) Then
          If ( rhoshkhi .ne. ZERO ) Then       ! rhoshklo = 0 => increase rhomin
            rhomin = 1.1d0 * rhomin
          Else If ( rhoshklo .ne. ZERO ) Then  ! rhoshkhi = 0 => decrease rhomax
            rhomax = 0.9d0 * rhomax
          Else                                 ! rhoshklo = rhoshkhi = 0 => ERROR
            ierr = 3
          End If
!
!.... rhoshklo * rhoshkhi > 0
!
!     Here, the two function values have the same sign: (1) reset the abscissa with 
!     the _larger_ absolute ordinate to be the abscissa with the _smaller_ absolute; 
!     (2) re-assign that smaller-absolute-abscissa ordinate to the linearly extrapolated
!     zero-ordinate abscissa, which is "nudged" by one-tenth of the delta-abcissa-value
!
        Else If ( scratch .gt. ZERO ) Then
          If ( rhoshklo .gt. ZERO .and. rhoshkhi .gt. ZERO ) Then
            If ( rhoshklo .lt. rhoshkhi ) Then ! 0 < rhoshklo < rhoshkhi
              rhotmp = rhomin
              rhomin = ( rhotmp * rhoshkhi - rhomax * rhoshklo ) &
                     / ( rhoshkhi - rhoshklo )
              rhomax = rhotmp
              rhotmp = rhomax - rhomin
              rhomin = rhomin - TENTH * rhotmp
            Else                               ! 0 < rhoshkhi < rhoshklo
              rhotmp = rhomax
              rhomax = ( rhomin * rhoshkhi - rhotmp * rhoshklo ) &
                     / ( rhoshkhi - rhoshklo )
              rhomin = rhotmp
              rhotmp = rhomax - rhomin
              rhomax = rhomax + TENTH * rhotmp
            End If
          Else
            If ( rhoshklo .lt. rhoshkhi ) Then ! rhoshklo < rhoshkhi < 0
              rhotmp = rhomax
              rhomax = ( rhomin * rhoshkhi - rhotmp * rhoshklo ) &
                     / ( rhoshkhi - rhoshklo )
              rhomin = rhotmp
              rhotmp = rhomax - rhomin
              rhomax = rhomax + TENTH * rhotmp
            Else                               ! rhoshkhi < rhoshklo < 0
              rhotmp = rhomin
              rhomin = ( rhotmp * rhoshkhi - rhomax * rhoshklo ) &
                     / ( rhoshkhi - rhoshklo )
              rhomax = rhotmp
              rhotmp = rhomax - rhomin
              rhomin = rhomin - TENTH * rhotmp
            End If
          End If
!
!.... rhoshklo * rhoshkhi < 0
!
        Else
          lbracketed = .true.
        End If
        If ( ierr .ne. 0 ) Go To 799           ! ierr = 3
      End Do 
!
!..... Solution that brackets the root has been found
!
      If ( lbracketed ) Then
        If ( rholin .lt. rhorin ) Then
          rholout = rhomin
          rhorout = rhomax
        Else
          rholout = rhomax
          rhorout = rhomin
        End If
!*!        If ( ldebug ) Write(iun1,200)
      End If
!
      If ( it .ge. itmax ) ierr = 4
      If ( ierr .ne. 0 ) Go To 799
!  
!-----------------------------------------------------------------------
!.... Error conditions
!-----------------------------------------------------------------------
  799 Continue
      If ( ierr .eq. 0 ) Go To 899 
      Write(*,900) ierr
      Go To ( 801, 802, 803, 804 ) ierr
  801 Write(*,901) rhol
      Go To 899 
  802 Write(*,902) rhor
      Go To 899 
  803 Write(*,903) rhomin, rhoshklo, rhomax, rhoshkhi
      Go To 899 
  804 Write(*,904) it, itmax
      Go To 899 
!
!.... Exit
!
  899 Continue
!*!      If ( ldebug ) Close(iun1)
!
!-----------------------------------------------------------------------
!.... Format statements
!-----------------------------------------------------------------------
!
  101 Format(/' it   rhomin   rhoshklo   rhomax   rhoshkhi' &
             /'---- --------- --------- --------- ---------')
  102 Format(i3,1x,4(1x,1pe9.2))
  200 Format('** Right shock bracketing values FOUND **')
!
  900 Format('** BRACKET_RIGHT_SHOCK: FATAL ERROR   ierr = ',i2,' **')
  901 Format('** rhol = ',1pe12.5,' <= 0 **')
  902 Format('** rhor = ',1pe12.5,' <= 0 **')
  903 Format('** rhomin = ',1pe12.5,' rhoshklo = ',1pe12.5, &
               ' rhomax = ',1pe12.5,' rhoshkhi = ',1pe12.5)
  904 Format('** it = ',i3,' > ',i3,' = itmax **')
!
!-----------------------------------------------------------------------
!
      Return
      End
!
! End of Subroutine BRACKET_RIGHT_SHOCK
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
! Start of Function JUMPRIGHT
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!                                                                      c
! RIGHT material: Evaluate the function for ZEROIN based on the        c
!     shock jump conditions given in Eq. (3.2) of Banks                c
!                                                                      c
!   Called by: GET_USTAR_RIGHT  Calls: EOS_RIGHT_RHOP                  c
!              ZEROIN                                                  c
!                                                                      c
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
      Double Precision Function JUMPRIGHT( rho )
!
      Implicit none
!
!.... Include files
!
      Include "param.h"
!
      Double Precision :: rho             ! density
!
!.... Local variables
!
      Integer :: ierr                     ! Error flag
!
      Double Precision :: rhoval          ! density (sent) 
      Double Precision :: sieval          ! SIE(rhoval,pstar)
      Double Precision :: sndval          ! sound_speed(rhoval,pstar)
      Double Precision :: scratch         ! (1/2)(p*-p_r)/(rho_r-rho)
      Double Precision :: term1           ! 1st term in function
      Double Precision :: term2           ! 2ne term in function
!
      Double Precision :: rhor            ! right density
      Double Precision :: sier            ! right SIE
      Double Precision :: prsr            ! right pressure
      Double Precision :: sndr            ! right sound speed
      Double Precision :: velr            ! right velocity
      Double Precision :: pstar           ! star-state pressure
!
      Common / right_state / rhor, sier, prsr, sndr, velr
      Common / star_state  / pstar
!
!-----------------------------------------------------------------------
!
      ierr = 0
      If ( pstar .le. ZERO ) ierr = 1
      If ( ierr .ne. 0 ) Go To 799
      If ( rhor .le. ZERO ) ierr = 2
      If ( ierr .ne. 0 ) Go To 799
!      
      rhoval = rho
      Call EOS_RIGHT_RHOP( rhoval, pstar, sieval, sndval, ierr)
      If ( ierr .ne. 0 ) ierr = 3
      If ( ierr .ne. 0 ) Go To 799
!
      scratch = HALF * ( pstar - prsr ) / ( rho - rhor )
      term1 = sier   + ( prsr  / rhor ) + ( rho  / rhor ) * scratch
      term2 = sieval + ( pstar / rho  ) + ( rhor / rho  ) * scratch
!
      JUMPRIGHT = term1 - term2
!
!-----------------------------------------------------------------------
!.... Error conditions
!-----------------------------------------------------------------------
  799 Continue
      If ( ierr .eq. 0 ) Go To 899 
!
!.... Here, there's an error, so set the function to zero before returning
!
      JUMPRIGHT = ZERO
      Write(*,900) ierr
      Go To ( 801, 802, 803 ) ierr
  801 Write(*,901) pstar
      Go To 899 
  802 Write(*,902) rhor
      Go To 899 
  803 Write(*,903)  
      Go To 899 
!
!.... Exit
!
  899 Continue
!
!-----------------------------------------------------------------------
!.... Format statements
!-----------------------------------------------------------------------
!
  900 Format('** JUMPRIGHT: FATAL ERROR   ierr = ',i2,' **')
  901 Format('**   pstar = ',1pe12.5,' <= 0 **')
  902 Format('**   rhor  = ',1pe12.5,' <= 0 **')
  903 Format('**   Error return from EOS_RIGHT_RHOP **')
!
!-----------------------------------------------------------------------
!
      End
!
! End of Function JUMPRIGHT
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
! Start of Subroutine EOS_LEFT_RHOP_JWL
!
      Subroutine EOS_LEFT_RHOP(rho, prs, sie, snd, ierr)
!
!.... EOS for LEFT JWL EOS: given density and pressure, 
!     return SIE.  Return a non-zero error flag with any problems.
!
!.... JWL GAS EOS -- See:  K.-M. Shyue, J. Comput. Phys. 171:678-707 (2001) 
!
      Implicit none
!
!.... Include files
!
      Include "param.h"
!
      Integer :: ierr                   ! Error flag
      Double Precision :: rho           ! Density
      Double Precision :: prs           ! Pressure
      Double Precision :: sie           ! SIE
      Double Precision :: snd           ! Sound speed

!
!.... Local variables
!
      Integer :: inml                   ! unit number of namelist input file
      Logical :: lend                   ! logical "end" flag
      Double Precision :: rho0          ! reference density
      Double Precision :: sie0          ! reference SIE
      Double Precision :: gamma0        ! \gamma0
      Double Precision :: biga          ! "A"
      Double Precision :: bigb          ! "B"
      Double Precision :: r1            ! "R_1"
      Double Precision :: r2            ! "R_2"
!
      Double Precision :: gamma         ! \gamma(\rho) = gamma0 for JWL
      Double Precision :: gamrho        ! \gamma * \rho
      Double Precision :: rhoratio      ! rho0 / rho
      Double Precision :: expval1       ! exp( -r1 * rhoratio )
      Double Precision :: expval2       ! exp( -r2 * rhoratio )
      Double Precision :: prsref        ! biga * expval1 + bigb * expval2
      Double Precision :: sieref        ! ( biga / r1 ) * expval1 / rho0
      Double Precision :: dgammadrho    ! d(gamma)/d(rho)
      Double Precision :: bigaorho      ! biga / rho
      Double Precision :: bigborho      ! bigb / rho
      Double Precision :: dprsrefdrho   ! d(prsref)/d(rho)
      Double Precision :: dsierefdrho   ! d(sieref)/d(rho)
!
      Double Precision :: rho0in        ! namelist reference density
      Double Precision :: sie0in        ! namelist reference SIE
      Double Precision :: gamma0in      ! namelist \gamma0
      Double Precision :: bigain        ! namelist "A"
      Double Precision :: bigbin        ! namelist "B"
      Double Precision :: r1in          ! namelist "R_1"
      Double Precision :: r2in          ! namelist "R_2"
!
      Common / left_jwl / rho0in, sie0in, gamma0in, bigain, bigbin, &
                          r1in, r2in
!
!      Namelist / eos_data / rho0in, sie0in, gamma0in, &
!                            bigain, bigbin, r1in, r2in
!
!-----------------------------------------------------------------------
!
      lend = .false.
      ierr = 0
      If ( rho .lt. ZERO ) ierr = 1
      If ( ierr .ne. 0 ) Go To 799
      If ( prs .lt. ZERO ) ierr = 2
      If ( ierr .ne. 0 ) Go To 799
!
!.... Special case of zero density or zero pressure
!
      If ( ( rho .eq. ZERO ) .or. ( prs .eq. ZERO ) ) Then
        sie = ZERO
        snd = ZERO
        lend = .true.
      End If 
      If ( lend ) Go To 799
!
!.... Assign default namelist input parameters
!
!      rho0in   =  2.d0
!      sie0in   =  0.d0
!      omegain =  0.8938d0
!      gamma0in =  1.1188185d0
!      bigain   =  6.925067d2
!      bigbin   = -4.4776d-2
!      r1in     =  1.13d1
!      r2in     =  1.13d0
!
!.... Read namelist input file
!
!      inml = 10
!      Open ( inml, file = 'eos_left.nml', status = 'old' )
!      Rewind ( inml ) 
!      Read ( inml, nml = eos_data )
!      Close( inml )
!  rho0in   = 1.905d0,
!  sie0in   = 0.d0,
!  gamma0in = 0.8938d0,
!  bigain   = 6.321d2,
!  bigbin   = -4.472d-2,
!  r1in     = 1.13d1,
!  r2in     = 1.13d0,
!
      rho0   = rho0in  
      sie0   = sie0in  
      gamma0 = gamma0in 
      biga   = bigain  
      bigb   = bigbin  
      r1     = r1in    
      r2     = r2in    

      If ( rho0 .le. ZERO ) ierr = 3
      If ( ierr .ne. 0 ) Go To 799
      If ( gamma0 .le. ZERO ) ierr = 4
      If ( ierr .ne. 0 ) Go To 799
!
!.... JWL EOS: SIE - direct
!
!      spcvol = ONE / rho
!      fv0 = biga * ( ( spcvol / omega ) - ( ONE / r1 ) ) &
!                 * exp( -r1 * spcvol )                   &
!          + bigb * ( ( spcvol / omega ) - ( ONE / r2 ) ) &
!                 * exp( -r2 * spcvol )
!      spcvol = ONE / rho
!      fv  = biga * ( ( spcvol / omega ) - ( ONE / r1 ) ) &
!                 * exp( -r1 * spcvol )                   &
!          + bigb * ( ( spcvol / omega ) - ( ONE / r2 ) ) &
!                 * exp( -r2 * spcvol )
!      sie = ( prs * spcvol / omega ) - fv + fv0
!      If ( sie .le. ZERO ) ierr = 5
!      If ( ierr .ne. 0 ) Go To 799
!
!.... JWL EOS: SIE - from generalized Mie-Grueneisen form
!     See: Shyue (2001) Eqs. 1 and 5
!
      gamma    = gamma0          ! Recall, gamma(rho) = gamma0
      gamrho   = gamma * rho
      rhoratio = rho0 / rho
      expval1  = exp( -r1 * rhoratio )
      expval2  = exp( -r2 * rhoratio )
      prsref = biga * expval1 + bigb * expval2
      sieref = ( ( ( biga / r1 / rho0 ) * expval1 ) &
               + ( ( bigb / r2 / rho0 ) * expval2 ) &
               ) - sie0
      sie = sieref + ( ( prs - prsref ) / gamrho )
      sie = sieref + ( ( prs - prsref ) / gamrho )
!      If ( sie .le. ZERO ) ierr = 5
      If ( ierr .ne. 0 ) Go To 799
!
!.... JWL EOS: Sound speed - from generalized Mie-Grueneisen form
!     See: Shyue (2001) Eq. 9
!
      dgammadrho = ZERO          ! Recall, gamma(rho) = gamma0
      bigaorho = biga / rho
      bigborho = bigb / rho
      dprsrefdrho = bigaorho * r1 * rhoratio * expval1 &
                  + bigborho * r2 * rhoratio * expval2
      dsierefdrho = bigaorho * expval1 / rho &
                  + bigborho * expval2 / rho
      snd = ( gamma + ONE + ( rho * ( dgammadrho / gamma ) ) ) &
          * ( ( prs - prsref ) / rho )                         &
          + ( gamma * prsref / rho )                           &
          + dprsrefdrho - ( gamrho * dsierefdrho )
      If ( snd .le. ZERO ) ierr = 6
      If ( ierr .ne. 0 ) Go To 799
      snd = Sqrt( snd )
!
!-----------------------------------------------------------------------
!.... Error conditions
!-----------------------------------------------------------------------
  799 Continue
      If ( ierr .eq. 0 ) Go To 899 
      Write(*,900) ierr
      Go To ( 801, 802, 803, 804, 805, 806 ) ierr
  801 Write(*,901) rho
      Go To 899 
  802 Write(*,902) prs
      Go To 899 
  803 Write(*,903) rho0
      Go To 899 
  804 Write(*,904) gamma0
      Go To 899 
  805 Write(*,905) sie
      Go To 899 
  806 Write(*,906) snd
      Go To 899 
!
!.... Exit
!
  899 Continue
!
!-----------------------------------------------------------------------
!.... Format statements
!-----------------------------------------------------------------------
!
  900 Format('** EOS_LEFT_RHOP: FATAL ERROR   ierr = ',i2,' **')
  901 Format('**   Density = ',1pe12.5,' < 0 **')
  902 Format('**   Pressure = ',1pe12.5,' < 0 **')
  903 Format('**   rho0 = ',1pe12.5,' <= 0 **')
  904 Format('**   gamma0 = ',1pe12.5,' <= 0 **')
  905 Format('**   SIE = ',1pe12.5,' <= 0 **')
  906 Format('**   Snd spd squared = ',1pe12.5,' <= 0 **')
!
!-----------------------------------------------------------------------
!
      Return
      End
!
! End of Subroutine EOS_LEFT_RHOP_JWL
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
! Start of Subroutine EOS_RIGHT_RHOP_JWL
!
      Subroutine EOS_RIGHT_RHOP( rho, prs, sie, snd, ierr)
!
!.... EOS for RIGHT JWL EOS: given density and pressure, 
!     return SIE.  Return a non-zero error flag with any problems.
!
!.... JWL GAS EOS -- See:  K.-M. Shyue, J. Comput. Phys. 171:678-707 (2001) 
!
      Implicit none
!
!.... Include files
!
      Include "param.h"
!
      Integer :: ierr                   ! Error flag
      Double Precision :: rho           ! Density
      Double Precision :: prs           ! Pressure
      Double Precision :: sie           ! SIE
      Double Precision :: snd           ! Sound speed
!
!.... Local variables
!
      Integer :: inml                   ! unit number of namelist input file
      Logical :: lend                   ! logical "end" flag
      Double Precision :: rho0          ! reference density
      Double Precision :: sie0          ! reference SIE
      Double Precision :: gamma0        ! \gamma0
      Double Precision :: biga          ! "A"
      Double Precision :: bigb          ! "B"
      Double Precision :: r1            ! "R_1"
      Double Precision :: r2            ! "R_2"
!
      Double Precision :: gamma         ! \gamma(\rho) = gamma0 for JWL
      Double Precision :: omega         ! = gamma0 for JWL
      Double Precision :: gamrho        ! \gamma * \rho
      Double Precision :: rhoratio      ! rho0 / rho
      Double Precision :: spcvol        ! 1 / rho   or    1 / rho0
      Double Precision :: expval1       ! exp( -r1 * rhoratio )
      Double Precision :: expval2       ! exp( -r2 * rhoratio )
      Double Precision :: fv            ! F-function of rho  -- see Banks 2.3
      Double Precision :: fv0           ! F-fucntion of rho0 -- see Banks 2.3
      Double Precision :: dfdrho        ! dF/d(rho)
      Double Precision :: prsref        ! biga * expval1 + bigb * expval2
      Double Precision :: sieref        ! ( biga / r1 ) * expval1 / rho0
      Double Precision :: dgammadrho    ! d(gamma)/d(rho)
      Double Precision :: bigaorho      ! biga / rho
      Double Precision :: bigborho      ! bigb / rho
      Double Precision :: dprsrefdrho   ! d(prsref)/d(rho)
      Double Precision :: dsierefdrho   ! d(sieref)/d(rho)
      Double Precision :: dedrho        ! d(sie)/d(rho) @ constant pressure
      Double Precision :: dedp          ! d(sie)/d(prs) @ constant density
!
      Double Precision :: rho0in        ! namelist reference density
      Double Precision :: sie0in        ! namelist reference SIE
      Double Precision :: gamma0in      ! namelist \gamma0
      Double Precision :: bigain        ! namelist "A"
      Double Precision :: bigbin        ! namelist "B"
      Double Precision :: r1in          ! namelist "R_1"
      Double Precision :: r2in          ! namelist "R_2"
!
      Common / right_jwl / rho0in, sie0in, gamma0in, bigain, bigbin, &
                           r1in, r2in
!
!      Namelist / eos_data / rho0in, sie0in, gamma0in, &
!                            bigain, bigbin, r1in, r2in
!
!-----------------------------------------------------------------------
!
      lend = .false.
      ierr = 0
      If ( rho .lt. ZERO ) ierr = 1
      If ( ierr .ne. 0 ) Go To 799
      If ( prs .lt. ZERO ) ierr = 2
      If ( ierr .ne. 0 ) Go To 799
!
!.... Special case of zero density or zero pressure
!
      If ( ( rho .eq. ZERO ) .or. ( prs .eq. ZERO ) ) Then
        sie = ZERO
        snd = ZERO
        lend = .true.
      End If 
      If ( lend ) Go To 799
!
!.... Assign default namelist input parameters
!
!      rho0in   =  2.d0
!      sie0in   =  0.d0
!      omegain =  0.8938d0
!      gamma0in =  1.1188185d0
!      bigain   =  6.925067d2
!      bigbin   = -4.4776d-2
!      r1in     =  1.13d1
!      r2in     =  1.13d0
!
!.... Read namelist input file
!
!      inml = 10
!      Open ( inml, file = 'eos_right.nml', status = 'old' )
!      Rewind ( inml ) 
!      Read ( inml, nml = eos_data )
!      Close( inml )
! Lee
!  rho0in   = 1.905d0,
!  sie0in   = 0.d0,
!  gamma0in = 0.8938d0,
!  bigain   = 6.321d2,
!  bigbin   = -4.472d-2,
!  r1in     = 1.13d1,
!  r2in     = 1.13d0,
!
      rho0   = rho0in  
      sie0   = sie0in  
      gamma0 = gamma0in 
      biga   = bigain  
      bigb   = bigbin  
      r1     = r1in    
      r2     = r2in    
      If ( rho0 .le. ZERO ) ierr = 3
      If ( ierr .ne. 0 ) Go To 799
      If ( gamma0 .le. ZERO ) ierr = 4
      If ( ierr .ne. 0 ) Go To 799
!
!.... JWL EOS: SIE - direct from Banks (2010) Eqs. 2.2, 2.3
!
!      omega = gamma0
!      spcvol = ONE / rho0
!      fv0 = biga * ( ( spcvol / omega ) - ( ONE / r1 ) ) 
!     &           * Exp( -r1 * spcvol )
!     &    + bigb * ( ( spcvol / omega ) - ( ONE / r2 ) ) 
!     &           * Exp( -r2 * spcvol )
!      spcvol = ONE / rho
!      fv  = biga * ( ( spcvol / omega ) - ( ONE / r1 ) ) 
!     &           * Exp( -r1 * spcvol )
!     &    + bigb * ( ( spcvol / omega ) - ( ONE / r2 ) ) 
!     &           * Exp( -r2 * spcvol )
!      sie = ( prs * spcvol / omega ) - fv + fv0
!      If ( sie .le. ZERO ) ierr = 5
!      If ( ierr .ne. 0 ) Go To 799
!
!.... JWL EOS: Sound speed - from thermodynamic relation
!     See: Banks (2010) Eq. 3.4
!
!      gamma  = gamma0          ! Recall, gamma(rho) = gamma0
!      gamrho = gamma * rho
!      dedp   = ONE / gamrho
!      dfdrho = biga * spcvol * Exp( -r1 * spcvol ) * 
!     &         ( r1 * spcvol * ( dedp - ( ONE / r1 ) ) - dedp )
!     &       + bigb * spcvol * Exp( -r2 * spcvol ) * 
!     &         ( r2 * spcvol * ( dedp - ( ONE / r2 ) ) - dedp )
!      dedrho = -( prs / gamrho / rho ) - dfdrho
!      snd = ( ( prs / rho / rho ) - dedrho ) / dedp
!      If ( snd .le. ZERO ) ierr = 6
!      If ( ierr .ne. 0 ) Go To 799
!      snd = Sqrt( snd )
!      Go To 799
!
!.... JWL EOS: SIE - from generalized Mie-Grueneisen form
!     See: Shyue (2001) Eqs. 1 and 5
!
      gamma    = gamma0          ! Recall, gamma(rho) = gamma0
      gamrho   = gamma * rho
      rhoratio = rho0 / rho
      expval1  = Exp( -r1 * rhoratio )
      expval2  = Exp( -r2 * rhoratio )
      prsref = biga * expval1 + bigb * expval2
      sieref = ( ( ( biga / r1 / rho0 ) * expval1 ) &
               + ( ( bigb / r2 / rho0 ) * expval2 ) &
               ) - sie0
      sie = sieref + ( ( prs - prsref ) / gamrho )
!      If ( sie .le. ZERO ) ierr = 5
      If ( ierr .ne. 0 ) Go To 799
!
!.... JWL EOS: Sound speed - from generalized Mie-Grueneisen form
!     See: Shyue (2001) Eq. 9
!
      dgammadrho = ZERO          ! Recall, gamma(rho) = gamma0
      bigaorho = biga / rho
      bigborho = bigb / rho
      dprsrefdrho = bigaorho * r1 * rhoratio * expval1 &
                  + bigborho * r2 * rhoratio * expval2
      dsierefdrho = bigaorho * expval1 / rho &
                  + bigborho * expval2 / rho
      snd = ( gamma + ONE + ( rho * ( dgammadrho / gamma ) ) ) &
          * ( ( prs - prsref ) / rho )                         &
          + ( gamma * prsref / rho )                           &
          + dprsrefdrho - ( gamrho * dsierefdrho )
      If ( snd .le. ZERO ) ierr = 6
      If ( ierr .ne. 0 ) Go To 799
      snd = Sqrt( snd )
!
!-----------------------------------------------------------------------
!.... Error conditions
!-----------------------------------------------------------------------
  799 Continue
      If ( ierr .eq. 0 ) Go To 899 
      Write(*,900) ierr
      Go To ( 801, 802, 803, 804, 805, 806 ) ierr
  801 Write(*,901) rho
      Go To 899 
  802 Write(*,902) prs
      Go To 899 
  803 Write(*,903) rho0
      Go To 899 
  804 Write(*,904) gamma0
      Go To 899 
  805 Write(*,905) sie
      Go To 899 
  806 Write(*,906) snd
      Go To 899 
!
!.... Exit
!
  899 Continue
!
!-----------------------------------------------------------------------
!.... Format statements
!-----------------------------------------------------------------------
!
  900 Format('** EOS_RIGHT_RHOP: FATAL ERROR   ierr = ',i2,' **')
  901 Format('**   Density = ',1pe12.5,' < 0 **')
  902 Format('**   Pressure = ',1pe12.5,' < 0 **')
  903 Format('**   rho0 = ',1pe12.5,' <= 0 **')
  904 Format('**   gamma0 = ',1pe12.5,' <= 0 **')
  905 Format('**   SIE = ',1pe12.5,' <= 0 **')
  906 Format('**   Snd spd squared = ',1pe12.5,' <= 0 **')
!
!-----------------------------------------------------------------------
!
      Return
      End
!
! End of Subroutine EOS_RIGHT_RHOP_JWL
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
! Start of Subroutine RARELEFTSOLN
!
!.... LEFT material: Solve for values at "nrar" points along the
!     rarefaction.  These points are equally spaced in _density_
!     using the value "drho" computed by routine RARELEFT.
!
!.... See:  J.W.Banks, "On Exact Conservation for the Euler Equations
!           with Complex Equations of State," Commun. Comput. Phys.
!           8(5), pp. 995-1015 (2010), doi: 10.4208/cicp/090909/100310a
!
      Subroutine RARELEFTSOLN( nrar, drho, xd0, time, xarr, velarr, &
                               rhoarr, prsarr, siearr, sndarr, ierr )
!
      Implicit none
!
!.... Include files
!
      Include "param.h"
!
!.... Call list variables
!
      Integer :: nrar                     ! # of steps thru rarefaction
!
      Double Precision :: drho            ! density-increment thru rarefaction
      Double Precision :: xd0             ! x-location of diaphragm  (cm)
      Double Precision :: time            ! simulation time          (s)
      Double Precision :: xarr(1:nrar)    ! position values along rarefaction
      Double Precision :: velarr(1:nrar)  ! velocity values along rarefaction
      Double Precision :: rhoarr(1:nrar)  ! density  values along rarefaction
      Double Precision :: prsarr(1:nrar)  ! pressure values along rarefaction
      Double Precision :: siearr(1:nrar)  ! SIE      values along rarefaction
      Double Precision :: sndarr(1:nrar)  ! snd_spd  values along rarefaction
!
!.... Local variables
!
      Integer :: ierr                     ! Error flag
      Integer :: k                        ! index
!
      Double Precision :: sie             ! EOS-returned SIE
      Double Precision :: snd             ! EOS-returned sound speed
      Double Precision :: csqrd           ! sie**2
      Double Precision :: pk              ! pressure at kth-pt   along rarefaction
      Double Precision :: pkp1            ! pressure at k+1st-pt along rarefaction
      Double Precision :: rhok            ! density  at kth-pt   along rarefaction
      Double Precision :: rhokp1          ! density  at k+1st-pt along rarefaction
      Double Precision :: rhokph          ! ( rhok + rhokp1 ) / 2
      Double Precision :: uk              ! velocity at kth-pt   along rarefaction
      Double Precision :: ukp1            ! velocity at k+1st-pt along rarefaction
      Double Precision :: phat1           ! \hat{p}_1 in Banks
      Double Precision :: phat2           ! \hat{p}_2 in Banks
      Double Precision :: phat3           ! \hat{p}_3 in Banks
      Double Precision :: phat4           ! \hat{p}_4 in Banks
      Double Precision :: snd1            ! snd_spd(rho,phat1)
      Double Precision :: snd2            ! snd_spd(rhokph,phat2)
      Double Precision :: snd3            ! snd_spd(rhokph,phat3)
      Double Precision :: snd4            ! snd_spd(rhokp1,phat4)
!
      Double Precision :: rhol            ! left  density
      Double Precision :: siel            ! left  SIE
      Double Precision :: prsl            ! left  pressure
      Double Precision :: sndl            ! left  sound speed
      Double Precision :: vell            ! left  velocity
      Double Precision :: pstar           ! star-state pressure
!
      Common / left_state / rhol, siel, prsl, sndl, vell
      Common / star_state / pstar
!
!-----------------------------------------------------------------------
!
      ierr = 0
      If ( pstar .le. ZERO ) ierr = 1
      If ( ierr .ne. 0 ) Go To 799
      If ( rhol .le. ZERO ) ierr = 2
      If ( ierr .ne. 0 ) Go To 799
      If ( prsl .le. ZERO ) ierr = 3
      If ( ierr .ne. 0 ) Go To 799
!
      Do k = 1, nrar
        xarr(k)   = ZERO
        velarr(k) = ZERO
        rhoarr(k) = ZERO
        prsarr(k) = ZERO
        siearr(k) = ZERO
        sndarr(k) = ZERO
      End Do
!
!.... Initialize quantities with the left_state values
!     See equations following Eq. (3.3) in Banks
!
      uk     = vell
      pk     = prsl
      rhok   = rhol
      rhokp1 = rhok - drho
      rhokph = HALF * ( rhok + rhokp1 )
!
      xarr(1)   = xd0 + ( vell - sndl ) * time
      velarr(1) = vell
      rhoarr(1) = rhol
      prsarr(1) = prsl
      siearr(1) = siel
      sndarr(1) = sndl
!      
!.... Loop over the number of points in the rarefaction
!
      Do k = 2, nrar
!
!.... Evaluate the intermediate pressure values
!     See equations following Eq. (3.3) in Banks
!
        phat1 = pk
        Call EOS_LEFT_RHOP( rhok, phat1, sie, snd1, ierr )
        If ( ierr .ne. 0 ) ierr = 4
        If ( ierr .ne. 0 ) Go To 799
!        
        Call EOS_LEFT_RHOP( rhokph, phat1, sie, snd, ierr )
        If ( ierr .ne. 0 ) ierr = 5
        If ( ierr .ne. 0 ) Go To 799
        phat2 = pk - HALF * drho * ( snd * snd )
        Call EOS_LEFT_RHOP( rhokph, phat2, sie, snd2, ierr )
        If ( ierr .ne. 0 ) ierr = 6
        If ( ierr .ne. 0 ) Go To 799
!
        phat3 = pk - HALF * drho * ( snd2 * snd2 )
        Call EOS_LEFT_RHOP( rhokph, phat3, sie, snd3, ierr )
        If ( ierr .ne. 0 ) ierr = 7
        If ( ierr .ne. 0 ) Go To 799
!
        Call EOS_LEFT_RHOP( rhokp1, phat3, sie, snd, ierr )
        If ( ierr .ne. 0 ) ierr = 8
        If ( ierr .ne. 0 ) Go To 799
        phat4 = pk - HALF * drho * ( snd * snd )
        Call EOS_LEFT_RHOP( rhokp1, phat4, sie, snd4, ierr )
        If ( ierr .ne. 0 ) ierr = 9
        If ( ierr .ne. 0 ) Go To 799
!
!.... Increment the pressure, velocity, and similarity variable 
!     per the RK-like expression following Eq. (3.3) in Banks
!
        pkp1  = pk - SIXTH * drho *                         &
         (       ( snd1 * snd1 )   + TWO * ( snd2 * snd2 )  &
         + TWO * ( snd3 * snd3 )   +       ( snd4 * snd4 ) )
!
!... MAYBE THIS SIGN (in Banks) IS WRONG
!        ukp1  = uk - SIXTH * drho * 
!... Try this instead:
        ukp1  = uk + SIXTH * drho *                          &
         (       ( snd1 / rhok )   + TWO * ( snd2 / rhokph ) &
         + TWO * ( snd3 / rhokph ) +       ( snd4 / rhokp1 ) )
        Call EOS_LEFT_RHOP( rhokp1, pkp1, sie, snd, ierr )
        If ( ierr .ne. 0 ) ierr = 10
        If ( ierr .ne. 0 ) Go To 799
!
!.... Assign values at the k-th point along the rarefaction
!
        velarr(k) = ukp1
        rhoarr(k) = rhokp1
        prsarr(k) = pkp1
        siearr(k) = sie
        sndarr(k) = snd
        xarr(k)   = xd0 + ( velarr(k) - sndarr(k) ) * time
!
!.... Update quantities for the next point on the rarefaction
!
        pk     = pkp1
        rhok   = rhokp1
        rhokp1 = rhok - drho
        rhokph = HALF * ( rhok + rhokp1 )
        uk = ukp1
!
      End Do ! k
!
!-----------------------------------------------------------------------
!.... Error conditions
!-----------------------------------------------------------------------
  799 Continue
      If ( ierr .eq. 0 ) Go To 899 
      Write(*,900) ierr
      Go To ( 801, 802, 803, 804, 805, 806, 807, 808, 809, 810 ) ierr
  801 Write(*,901) pstar
      Go To 899 
  802 Write(*,902) rhol
      Go To 899 
  803 Write(*,903) prsl
      Go To 899 
  804 Write(*,904) rhok, phat1
      Go To 899 
  805 Write(*,905) rhokph, phat1
      Go To 899 
  806 Write(*,906) rhokph, phat2
      Go To 899 
  807 Write(*,907) rhokph, phat3
      Go To 899 
  808 Write(*,908) rhokp1, phat3
      Go To 899 
  809 Write(*,909) rhokph, phat4
      Go To 899 
  810 Write(*,910) rhokp1, pkp1
      Go To 899 
!
!.... Exit
!
  899 Continue
!
!-----------------------------------------------------------------------
!.... Format statements
!-----------------------------------------------------------------------
!
  900 Format('** RARELEFTSOLN: FATAL ERROR   ierr = ',i2,' **')
  901 Format('**   pstar = ',1pe12.5,' <= 0 **')
  902 Format('**   rhol  = ',1pe12.5,' <= 0 **')
  903 Format('**   prsl  = ',1pe12.5,' <= 0 **')
  904 Format('**   Error from EOS_LEFT_RHOP: rhok = ',1pe12.5, &
             '  phat1 = ',1pe12.5,' **')
  905 Format('**   Error from EOS_LEFT_RHOP: rhokph = ',1pe12.5, &
             '  phat1 = ',1pe12.5,' **')
  906 Format('**   Error from EOS_LEFT_RHOP: rhokph = ',1pe12.5, &
             '  phat2 = ',1pe12.5,' **')
  907 Format('**   Error from EOS_LEFT_RHOP: rhokph = ',1pe12.5, &
             '  phat3 = ',1pe12.5,' **')
  908 Format('**   Error from EOS_LEFT_RHOP: rhokp1 = ',1pe12.5, &
             '  phat3 = ',1pe12.5,' **')
  909 Format('**   Error from EOS_LEFT_RHOP: rhokph = ',1pe12.5, &
             '  phat4 = ',1pe12.5,' **')
  910 Format('**   Error from EOS_LEFT_RHOP: rhokp1 = ',1pe12.5, &
             '  pkp1 = ',1pe12.5,' **')
!
!-----------------------------------------------------------------------
!
      End
!
! End of Subroutine RARELEFTSOLN
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
! Start of Subroutine RARERIGHTSOLN
!
!.... RIGHT material: Solve for values at "nrar" points along the
!     rarefaction.  These points are equally spaced in _density_
!     using the value "drho" computed by routine RARERIGHT.
!
!.... See:  J.W.Banks, "On Exact Conservation for the Euler Equations
!           with Complex Equations of State," Commun. Comput. Phys.
!           8(5), pp. 995-1015 (2010), doi: 10.4208/cicp/090909/100310a
!
      Subroutine RARERIGHTSOLN( nrar, drho, xd0, time, xarr, velarr, &
                                rhoarr, prsarr, siearr, sndarr, ierr )
!
      Implicit none
!
!.... Include files
!
      Include "param.h"
!
!.... Call list variables
!
      Integer :: nrar                     ! # of steps thru rarefaction
!
      Double Precision :: drho            ! density-increment thru rarefaction
      Double Precision :: xd0             ! x-location of diaphragm  (cm)
      Double Precision :: time            ! simulation time          (s)
      Double Precision :: xarr(1:nrar)    ! position values along rarefaction
      Double Precision :: velarr(1:nrar)  ! velocity values along rarefaction
      Double Precision :: rhoarr(1:nrar)  ! density  values along rarefaction
      Double Precision :: prsarr(1:nrar)  ! pressure values along rarefaction
      Double Precision :: siearr(1:nrar)  ! SIE      values along rarefaction
      Double Precision :: sndarr(1:nrar)  ! snd_spd  values along rarefaction
!
!.... Local variables
!
      Integer :: ierr                     ! Error flag
      Integer :: k                        ! index
!
      Double Precision :: sie             ! EOS-returned SIE
      Double Precision :: snd             ! EOS-returned sound speed
      Double Precision :: csqrd           ! sie**2
      Double Precision :: pk              ! pressure at kth-pt   along rarefaction
      Double Precision :: pkp1            ! pressure at k+1st-pt along rarefaction
      Double Precision :: rhok            ! density  at kth-pt   along rarefaction
      Double Precision :: rhokp1          ! density  at k+1st-pt along rarefaction
      Double Precision :: rhokph          ! ( rhok + rhokp1 ) / 2
      Double Precision :: uk              ! velocity at kth-pt   along rarefaction
      Double Precision :: ukp1            ! velocity at k+1st-pt along rarefaction
      Double Precision :: phat1           ! \hat{p}_1 in Banks
      Double Precision :: phat2           ! \hat{p}_2 in Banks
      Double Precision :: phat3           ! \hat{p}_3 in Banks
      Double Precision :: phat4           ! \hat{p}_4 in Banks
      Double Precision :: snd1            ! snd_spd(rho,phat1)
      Double Precision :: snd2            ! snd_spd(rhokph,phat2)
      Double Precision :: snd3            ! snd_spd(rhokph,phat3)
      Double Precision :: snd4            ! snd_spd(rhokp1,phat4)
!
      Double Precision :: rhor            ! right density
      Double Precision :: sier            ! right SIE
      Double Precision :: prsr            ! right pressure
      Double Precision :: sndr            ! right sound speed
      Double Precision :: velr            ! right velocity
      Double Precision :: pstar           ! star-state pressure
!
      Common / right_state / rhor, sier, prsr, sndr, velr
      Common / star_state  / pstar
!
!-----------------------------------------------------------------------
!
      ierr = 0
      If ( pstar .le. ZERO ) ierr = 1
      If ( ierr .ne. 0 ) Go To 799
      If ( rhor .le. ZERO ) ierr = 2
      If ( ierr .ne. 0 ) Go To 799
      If ( prsr .le. ZERO ) ierr = 3
      If ( ierr .ne. 0 ) Go To 799
!
      Do k = 1, nrar
        xarr(k)   = ZERO
        velarr(k) = ZERO
        rhoarr(k) = ZERO
        prsarr(k) = ZERO
        siearr(k) = ZERO
        sndarr(k) = ZERO
      End Do
!
!.... Initialize quantities with the right_state values
!     See equations following Eq. (3.3) in Banks
!
      uk     = velr
      pk     = prsr
      rhok   = rhor
      rhokp1 = rhok - drho
      rhokph = HALF * ( rhok + rhokp1 )
!
!.... Note: xarr has a sign difference from the left rarefaction case
!
      xarr(1)   = xd0 + ( velr + sndr ) * time
      velarr(1) = velr
      rhoarr(1) = rhor
      prsarr(1) = prsr
      siearr(1) = sier
      sndarr(1) = sndr
!      
!.... Loop over the number of points in the rarefaction
!
      Do k = 2, nrar
!
!.... Evaluate the intermediate pressure values
!     See equations following Eq. (3.3) in Banks
!
        phat1 = pk
        Call EOS_RIGHT_RHOP( rhok, phat1, sie, snd1, ierr )
        If ( ierr .ne. 0 ) ierr = 4
        If ( ierr .ne. 0 ) Go To 799
!        
        Call EOS_RIGHT_RHOP( rhokph, phat1, sie, snd, ierr )
        If ( ierr .ne. 0 ) ierr = 5
        If ( ierr .ne. 0 ) Go To 799
        phat2 = pk - HALF * drho * ( snd * snd )
        Call EOS_RIGHT_RHOP( rhokph, phat2, sie, snd2, ierr )
        If ( ierr .ne. 0 ) ierr = 6
        If ( ierr .ne. 0 ) Go To 799
!
        phat3 = pk - HALF * drho * ( snd2 * snd2 )
        Call EOS_RIGHT_RHOP( rhokph, phat3, sie, snd3, ierr )
        If ( ierr .ne. 0 ) ierr = 7
        If ( ierr .ne. 0 ) Go To 799
!
        Call EOS_RIGHT_RHOP( rhokp1, phat3, sie, snd, ierr )
        If ( ierr .ne. 0 ) ierr = 8
        If ( ierr .ne. 0 ) Go To 799
        phat4 = pk - HALF * drho * ( snd * snd )
        Call EOS_RIGHT_RHOP( rhokp1, phat4, sie, snd4, ierr )
        If ( ierr .ne. 0 ) ierr = 9
        If ( ierr .ne. 0 ) Go To 799
!
!.... Increment the pressure, velocity, and similarity variable 
!     per the RK-like expression following Eq. (3.3) in Banks
!
        pkp1  = pk - SIXTH * drho *                         &
         (       ( snd1 * snd1 )   + TWO * ( snd2 * snd2 )  &
         + TWO * ( snd3 * snd3 )   +       ( snd4 * snd4 ) )
!
        ukp1  = uk - SIXTH * drho *                          &
         (       ( snd1 / rhok )   + TWO * ( snd2 / rhokph ) &
         + TWO * ( snd3 / rhokph ) +       ( snd4 / rhokp1 ) )
        Call EOS_RIGHT_RHOP( rhokp1, pkp1, sie, snd, ierr )
        If ( ierr .ne. 0 ) ierr = 10
        If ( ierr .ne. 0 ) Go To 799
!
!.... Assign values at the k-th point along the rarefaction
!
        velarr(k) = ukp1
        rhoarr(k) = rhokp1
        prsarr(k) = pkp1
        siearr(k) = sie
        sndarr(k) = snd
!
!.... Note: xarr has a sign difference from the left rarefaction case
!
        xarr(k)   = xd0 + ( velarr(k) + sndarr(k) ) * time
!
!.... Update quantities for the next point on the rarefaction
!
        pk     = pkp1
        rhok   = rhokp1
        rhokp1 = rhok - drho
        rhokph = HALF * ( rhok + rhokp1 )
        uk = ukp1
!
      End Do ! k
!
!      Write(*,701)
!      Do k = 1, nrar
!        Write(*,703) k, xarr(k),rhoarr(k),prsarr(k), &
!                        velarr(k),siearr(k),sndarr(k)
!      End Do !
!
!-----------------------------------------------------------------------
!.... Error conditions
!-----------------------------------------------------------------------
  799 Continue
      If ( ierr .eq. 0 ) Go To 899 
      Write(*,900) ierr
      Go To ( 801, 802, 803, 804, 805, 806, 807, 808, 809, 810 ) ierr
  801 Write(*,901) pstar
      Go To 899 
  802 Write(*,902) rhor
      Go To 899 
  803 Write(*,903) prsr
      Go To 899 
  804 Write(*,904) rhok, phat1
      Go To 899 
  805 Write(*,905) rhokph, phat1
      Go To 899 
  806 Write(*,906) rhokph, phat2
      Go To 899 
  807 Write(*,907) rhokph, phat3
      Go To 899 
  808 Write(*,908) rhokp1, phat3
      Go To 899 
  809 Write(*,909) rhokph, phat4
      Go To 899 
  810 Write(*,910) rhokp1, pkp1
      Go To 899 
!
!.... Exit
!
  899 Continue
!
!-----------------------------------------------------------------------
!.... Format statements
!-----------------------------------------------------------------------
!
  701 Format(/' i     xarr      rhoarr    prsarr    velarr    siearr ' &
      ,'   sndarr'                                                     &
             /'---- --------- --------- --------- --------- ---------' &
      ,' ---------')
  703 Format(i4,1x,6(1x,1pe9.2))
!
  900 Format('** RARERIGHTSOLN: FATAL ERROR   ierr = ',i2,' **')
  901 Format('**   pstar = ',1pe12.5,' <= 0 **')
  902 Format('**   rhor  = ',1pe12.5,' <= 0 **')
  903 Format('**   prsr  = ',1pe12.5,' <= 0 **')
  904 Format('**   Error from EOS_RIGHT_RHOP: rhok = ',1pe12.5,   &
             '  phat1 = ',1pe12.5,' **')
  905 Format('**   Error from EOS_RIGHT_RHOP: rhokph = ',1pe12.5, &
             '  phat1 = ',1pe12.5,' **')
  906 Format('**   Error from EOS_RIGHT_RHOP: rhokph = ',1pe12.5, &
             '  phat2 = ',1pe12.5,' **')
  907 Format('**   Error from EOS_RIGHT_RHOP: rhokph = ',1pe12.5, &
             '  phat3 = ',1pe12.5,' **')
  908 Format('**   Error from EOS_RIGHT_RHOP: rhokp1 = ',1pe12.5, &
             '  phat3 = ',1pe12.5,' **')
  909 Format('**   Error from EOS_RIGHT_RHOP: rhokph = ',1pe12.5, &
             '  phat4 = ',1pe12.5,' **')
  910 Format('**   Error from EOS_RIGHT_RHOP: rhokp1 = ',1pe12.5, &
             '  pkp1 = ',1pe12.5,' **')
!
!-----------------------------------------------------------------------
!
      End
!
! End of Subroutine RARERIGHTSOLN
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
! Start of Subroutine BRACKET_LEFT_RARE
!
      Subroutine BRACKET_LEFT_RARE( drholin,  drhorin, drholout, &
                                    drhorout, ldebugin, ierr )
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!                                                                      c
! This subroutine takes the input values of left and right             c
! delta-density, "drholin" and "drhorin", and determines whether those c
! arguments bracket a zero of the RARELEFT function, which is used in  c
! a ZEROIN call with RARELEFT in the calling routine.  If so, then     c
! those values are returned in the output variables "drholout" and     c
! "drhorout".  If not, then the "drhol" or "drhor" are modified, in a  c
! simplistic way, in a search for a pair of values that will bracket a c
! zero;  this modification is attempted for at most "itmax" iterations.c
!                                                                      c
!   Called by: GET_USTAR_LEFT   Calls: RARELEFT                        c
!                                                                      c
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
      Implicit none
!
!.... Include files
!
      Include "param.h"
!
!.... Call list variables
!
      Integer :: ierr                     ! error flag
!
      Double Precision :: drholin         ! input  left  delta-density
      Double Precision :: drhorin         ! input  right delta-density
      Double Precision :: drholout        ! output left  delta-density
      Double Precision :: drhorout        ! output right delta-density
!
      Logical :: ldebugin                 ! debug flag
!
!.... Local variables
!
      Integer :: it                       ! iteration counter
      Integer :: itmax                    ! max number of iterations
      Integer :: iun1                     ! debug file unit number
!
      Double Precision :: drhol           ! left  delta-density
      Double Precision :: drhor           ! right delta-density
      Double Precision :: drhomin         ! min delta-density for root-finder
      Double Precision :: drhomax         ! max delta-density max root-finder
      Double Precision :: drhotmp         ! temp delta-density
      Double Precision :: prsndlo         ! RARELEFT value at drhomin
      Double Precision :: prsndhi         ! RARELEFT value at drhomax
      Double Precision :: scratch         ! = prsndlo * prsndhi
!
      Logical :: ldebug                   ! .true. => print debug statements
      Logical :: lbracketed               ! .true. => root is bracketed
!
!.... Functions called
!
      Double Precision :: RARELEFT       ! Evaluate left rarefaction
!
!.... Common block variables
!
      Integer :: nrar                     ! # of steps thru rarefaction
      Double Precision :: rhol            ! left density
      Double Precision :: siel            ! left SIE
      Double Precision :: prsl            ! left pressure
      Double Precision :: sndl            ! left sound speed
      Double Precision :: vell            ! left velocity
      Double Precision :: pstar           ! star-state pressure
!
      Common / left_state / rhol, siel, prsl, sndl, vell
      Common / star_state  / pstar
      Common / num_rare    / nrar
!
!-----------------------------------------------------------------------
!
      ierr = 0
      iun1 = 34                    ! unit number
      ldebug = ldebugin
!*!      If ( ldebug ) Open(unit=iun1,file='bracket_left_rare.dbg', &
!*!                         status='unknown')
!-----------------------------------------------------------------------
      itmax  = 20
!-----------------------------------------------------------------------
      drhol = drholin
      drhor = drhorin
      If ( drhol .le. ZERO ) ierr = 1
      If ( ierr .ne. 0 ) Go To 799
      If ( drhor .le. ZERO ) ierr = 2
      If ( ierr .ne. 0 ) Go To 799
!
!.... Attempts to ensure bracketing values of the delta-density sent to 
!     the ZEROIN call with RARELEFT that determines the corresponding
!     post-rarefaction pressure; the returned quantity is the difference
!     between that value and the common-block star-state pressure, all
!     nondimensionalized by that star-state pressure
!
      drhomin = 0.9d0 * Min( drhol, drhor )
      drhomax = 1.1d0 * Max( drhol, drhor )
      lbracketed = .false.
      it = 0
!*!      If ( ldebug ) Write(iun1,101)
!*!      If ( ldebug ) Write(iun1,102) it, drhomin, ZERO, drhomax, ZERO
      Do While ( ( .not. lbracketed ) .and. ( it .lt. itmax ) )
        it = it + 1
        prsndlo = RARELEFT( drhomin )
        prsndhi = RARELEFT( drhomax )
!*!        If ( ldebug ) Write(iun1,102) it, drhomin, prsndlo, &
!*!                                          drhomax, prsndhi
        scratch = prsndlo * prsndhi
!
!.... prsndlo * prsndhi = 0
!
        If ( scratch .eq. ZERO ) Then
          If ( prsndhi .ne. ZERO ) Then        ! prsndlo = 0 => increase drhomin?
            drhomin = 1.2d0 * drhomin
          Else If ( prsndlo .ne. ZERO ) Then   ! prsndhi = 0 => decrease drhomax?
            drhomax = 0.8d0 * drhomax
          Else                                 ! prsndlo = prsndhi = 0 => ERROR
            ierr = 3
          End If
!
!.... prsndlo * prsndhi > 0
!
!     Here, the two function values have the same sign: (1) reset the abscissa with 
!     the _larger_ absolute ordinate to be the abscissa with the _smaller_ absolute; 
!     (2) re-assign that smaller-absolute-abscissa ordinate to the linearly extrapolated
!     zero-ordinate abscissa, which is "nudged" by one-tenth of the delta-abcissa-value
!
        Else If ( scratch .gt. ZERO ) Then
          If ( prsndlo .gt. ZERO .and. prsndhi .gt. ZERO ) Then
            If ( prsndlo .lt. prsndhi ) Then   ! 0 < prsndlo < prsndhi
              drhotmp = drhomin                ! 
              drhomin = ( drhotmp * prsndhi - drhomax * prsndlo ) &
                      / ( prsndhi - prsndlo )
              drhomax = drhotmp
              drhotmp = drhomax - drhomin
              drhomin = drhomin - TENTH * drhotmp
            Else                               ! 0 < prsndhi < prsndlo
              drhotmp = drhomax
              drhomax = ( drhomin * prsndhi - drhotmp * prsndlo ) &
                      / ( prsndhi - prsndlo )
              drhomin = drhotmp
              drhotmp = drhomax - drhomin
              drhomax = drhomax + TENTH * drhotmp
            End If
          Else
            If ( prsndlo .lt. prsndhi ) Then   ! prsndlo < prsndhi < 0
              drhotmp = drhomax
              drhomax = ( drhomin * prsndhi - drhotmp * prsndlo ) &
                      / ( prsndhi - prsndlo )
              drhomin = drhotmp
              drhotmp = drhomax - drhomin
              drhomax = drhomax + TENTH * drhotmp
            Else                               ! prsndhi < prsndlo < 0
              drhotmp = drhomin
              drhomin = ( drhotmp * prsndhi - drhomax * prsndlo ) &
                      / ( prsndhi - prsndlo )
              drhomax = drhotmp
              drhotmp = drhomax - drhomin
              drhomin = drhomin - TENTH * drhotmp
            End If
          End If
!
!.... prsndlo * prsndhi < 0
!
        Else
          lbracketed = .true.
        End If
        If ( lbracketed ) Go To 798
        If ( ierr .ne. 0 ) Go To 799           ! ierr = 3
      End Do 
!
!..... Solution that brackets the root has been found
!
 798  Continue
      If ( lbracketed ) Then
        If ( drholin .lt. drhorin ) Then
          drholout = drhomin
          drhorout = drhomax
        Else
          drholout = drhomax
          drhorout = drhomin
        End If
!*!        If ( ldebug ) Write(iun1,200) 
      End If
!
      If ( it .ge. itmax ) ierr = 4
      If ( ierr .ne. 0 ) Go To 799
!  
!-----------------------------------------------------------------------
!.... Error conditions
!-----------------------------------------------------------------------
  799 Continue
      If ( ierr .eq. 0 ) Go To 899 
      Write(*,900) ierr
      Go To ( 801, 802, 803, 804 ) ierr
  801 Write(*,901) drhol
      Go To 899 
  802 Write(*,902) drhor
      Go To 899 
  803 Write(*,903) drhomin, prsndlo, drhomax, prsndhi
      Go To 899 
  804 Write(*,904) it, itmax
      Go To 899 
!
!.... Exit
!
  899 Continue
!*!      If ( ldebug ) Close(iun1)
!
!-----------------------------------------------------------------------
!.... Format statements
!-----------------------------------------------------------------------
!
  101 Format(/' it   drhomin   prsndlo   drhomax   prsndhi' &
             /'---- --------- --------- --------- ---------')
  102 Format(i3,1x,4(1x,1pe9.2))
  200 Format('** Left rarefaction bracketing values FOUND **')
!
  900 Format('** BRACKET_LEFT_RARE: FATAL ERROR   ierr = ',i2,' **')
  901 Format('** drhol = ',1pe12.5,' <= 0 **')
  902 Format('** drhor = ',1pe12.5,' <= 0 **')
  903 Format('** drhomin = ',1pe12.5,' prsndlo = ',1pe12.5, &
               ' drhomax = ',1pe12.5,' prsndhi = ',1pe12.5)
  904 Format('** it = ',i3,' > ',i3,' = itmax **')
!
!-----------------------------------------------------------------------
!
      Return
      End
!
! End of Subroutine BRACKET_LEFT_RARE
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
! Start of Function RARELEFT
!
!.... LEFT material: Function for ZEROIN of rarefaction delta-rho
!     This function returns the nondimensionalized pressure, 
!                RARELEFT = ( pcomputed - pstar ) / pstar
!     where "pcomputed" is the computed star-state pressure obtained
!     by integrating down the rarefaction from the left-state to
!     the (left) star-state using the sent-value of the density-increment
!     for the integration, "drho".  Here, "pstar" is the given p-star 
!     value that is sent in through the common block "star_state".  
!     Thus, this function gives a non-dimensional measure of the mismatch
!     between the given p-star value and that computed by following 
!     the isentrope down the rarefaction.
!
!.... See:  J.W.Banks, "On Exact Conservation for the Euler Equations
!           with Complex Equations of State," Commun. Comput. Phys.
!           8(5), pp. 995-1015 (2010), doi: 10.4208/cicp/090909/100310a
!
! 141208 Modification from Banks' RK4-like formula to true RK4 formulation,
!        which uses two fewer EOS calls
!
      Double Precision Function RARELEFT( drho )
!
      Implicit none
!
!.... Include files
!
      Include "param.h"
!
!.... Call list variables
!
      Double Precision :: drho            ! density-increment thru rarefaction
!
!.... Local variables
!
      Integer :: ierr                     ! error flag
      Integer :: k                        ! index
      Integer :: kerr                     ! index at error
!
      Double Precision :: sie             ! EOS-returned SIE
      Double Precision :: snd             ! EOS-returned sound speed
      Double Precision :: csqrd           ! sie**2
      Double Precision :: pk              ! pressure at kth-pt   along rarefaction
      Double Precision :: pkp1            ! pressure at k+1st-pt along rarefaction
      Double Precision :: rhok            ! density  at kth-pt   along rarefaction
      Double Precision :: rhokp1          ! density  at k+1st-pt along rarefaction
      Double Precision :: rhokph          ! ( rhok + rhokp1 ) / 2
      Double Precision :: phat1           ! \hat{p}_1 in Banks
      Double Precision :: phat2           ! \hat{p}_2 in Banks
      Double Precision :: phat3           ! \hat{p}_3 in Banks
      Double Precision :: phat4           ! \hat{p}_4 in Banks
      Double Precision :: snd1            ! snd_spd(rhok,phat1)
      Double Precision :: snd2            ! snd_spd(rhokph,phat2)
      Double Precision :: snd3            ! snd_spd(rhokph,phat3)
      Double Precision :: snd4            ! snd_spd(rhokp1,phat4)
      Logical :: ldebug                   ! debug flag
!
!.... Common block variables
!
      Integer :: nrar                     ! # of steps thru rarefaction
      Double Precision :: rhol            ! left  density
      Double Precision :: siel            ! left  SIE
      Double Precision :: prsl            ! left  pressure
      Double Precision :: sndl            ! left  sound speed
      Double Precision :: vell            ! left  velocity
      Double Precision :: pstar           ! star-state pressure
!
      Common / left_state / rhol, siel, prsl, sndl, vell
      Common / star_state / pstar
      Common / num_rare   / nrar
!
!-----------------------------------------------------------------------
!
!     NOTE:  this debug flag is _solely_ for this routine and is NOT
!            passed from any calling routine
!
      ldebug = .false.
      If ( ldebug ) Write(*,*) '** Enter RARELEFT...**'
      ierr = 0
      If ( pstar .le. ZERO ) ierr = 1
      If ( ierr .ne. 0 ) Go To 799
      If ( rhol .le. ZERO ) ierr = 2
      If ( ierr .ne. 0 ) Go To 799
      If ( prsl .le. ZERO ) ierr = 3
      If ( ierr .ne. 0 ) Go To 799
!
!.... Initialize quantities with the left_state values and proceed
!     along the rarefaction wave from the left state toward the
!     star-state.  Recall that the both the density and the pressure
!     decrease thru the rarefaction. See eq'ns following Eq. (3.3) in Banks.
!
      pk   = prsl
      rhok = rhol
      rhokp1 = rhok - drho
      rhokph = HALF * ( rhok + rhokp1 )
!
      k = 1
!      
!.... Loop through the number of points in the rarefaction
!
      Do k = 2, nrar
        If ( ldebug ) Write(*,100) k
!
!.... Evaluate the intermediate pressures and sound speeds:
!     see equations following Eq. (3.3) in Banks
!
        phat1 = pk
        Call EOS_LEFT_RHOP( rhok, phat1, sie, snd1, ierr )
        If ( ierr .ne. 0 ) ierr = 4
        kerr = k
        If ( ierr .ne. 0 ) Go To 799
        If ( ldebug ) Write(*,101) k, rhok, phat1, sie, snd1
!
!.... Banks formula:
!        
!        Call EOS_LEFT_RHOP( rhokph, phat1, sie, snd, ierr )
!        If ( ierr .ne. 0 ) ierr = 5
!        kerr = k
!        If ( ierr .ne. 0 ) Go To 799
!        phat2 = pk - HALF * drho * ( snd * snd )
!        Call EOS_LEFT_RHOP( rhokph, phat2, sie, snd2, ierr )
!        If ( ierr .ne. 0 ) ierr = 6
!        kerr = k
!        If ( ierr .ne. 0 ) Go To 799
!
!.... RK4 formula:
!
        phat2 = pk - HALF * drho * ( snd1 * snd1 )
        Call EOS_LEFT_RHOP( rhokph, phat2, sie, snd2, ierr )
        If ( ierr .ne. 0 ) ierr = 6
        kerr = k
        If ( ierr .ne. 0 ) Go To 799
        If ( ldebug ) Write(*,102) k, rhokph, phat2, sie, snd2
!
        phat3 = pk - HALF * drho * ( snd2 * snd2 )
        Call EOS_LEFT_RHOP( rhokph, phat3, sie, snd3, ierr )
        If ( ierr .ne. 0 ) ierr = 7
        kerr = k
        If ( ierr .ne. 0 ) Go To 799
        If ( ldebug ) Write(*,103) k, rhokph, phat3, sie, snd3
!
!.... Banks formula:
!
!        Call EOS_LEFT_RHOP( rhokp1, phat3, sie, snd, ierr )
!        If ( ierr .ne. 0 ) ierr = 8
!        kerr = k
!        If ( ierr .ne. 0 ) Go To 799
!        phat4 = pk - HALF * drho * ( snd * snd )
!        If ( ldebug ) Write(*,104) k, rhokp1, phat4, snd
!
!.... RK4 formula:
!
        phat4 = pk - drho * ( snd3 * snd3 )
        Call EOS_LEFT_RHOP( rhokp1, phat4, sie, snd4, ierr )
        If ( ierr .ne. 0 ) ierr = 9
        kerr = k
        If ( ierr .ne. 0 ) Go To 799
        If ( ldebug ) Write(*,105) k, rhokp1, phat4, sie, snd4
!
!.... Increment the pressure per the RK-like expression
!     following Eq. (3.3) in Banks
!
        pkp1  = pk - SIXTH * drho *                      &
         (       ( snd1 * snd1 ) + TWO * ( snd2 * snd2 ) &
         + TWO * ( snd3 * snd3 ) +       ( snd4 * snd4 ) )
!
!.... Update quantities for the next point on the rarefaction
!
        pk     = pkp1
        rhok   = rhokp1
        rhokp1 = rhok - drho
        rhokph = HALF * ( rhok + rhokp1 )
!
      End Do ! k
!
      RARELEFT = ( pk - pstar ) / pstar
      If ( ldebug ) Write(*,*) '** ...Exit RARELEFT **'
!
!-----------------------------------------------------------------------
!
!.... Error conditions
!
  799 Continue
      If ( ierr .eq. 0 ) Go To 899 
!
!.... Here, there's an error, so set the function to zero before returning
!
      RARELEFT = ZERO
      Write(*,900) ierr
      Go To ( 801, 802, 803, 804, 805, 806, 807, 808, 809 ) ierr
  801 Write(*,901) pstar
      Go To 899 
  802 Write(*,902) rhol
      Go To 899 
  803 Write(*,903) prsl
      Go To 899 
  804 Write(*,904) kerr, rhok, phat1
      Go To 899 
  805 Write(*,905) kerr, rhokph, phat1
      Go To 899 
  806 Write(*,906) kerr, rhokph, phat2
      Go To 899 
  807 Write(*,907) kerr, rhokph, phat3
      Go To 899 
  808 Write(*,908) kerr, rhokp1, phat3
      Go To 899 
  809 Write(*,909) kerr, rhokp1, phat4
      Go To 899 
!
!.... Exit
!
  899 Continue
!
!-----------------------------------------------------------------------
!.... Format statements
!-----------------------------------------------------------------------
!
  100   Format('** Start left rarefaction at point k = ',i3,' **')
  101   Format('** After 1st call: k = ',i3,' rhok   = ',1pe9.2,       &
               ' phat1 = ',1pe9.2,' sie1 = ',1pe9.2,' snd1 = ',1pe9.2, &
               ' **')
  102   Format('** After 2nd call: k = ',i3,' rhokph = ',1pe9.2,       &
               ' phat2 = ',1pe9.2,' sie2 = ',1pe9.2,' snd2 = ',1pe9.2, &
               ' **')
  103   Format('** After 3rd call: k = ',i3,' rhokph = ',1pe9.2,       &
               ' phat3 = ',1pe9.2,' sie3 = ',1pe9.2,' snd3 = ',1pe9.2, &
               ' **')
  104   Format('** After 4th call: k = ',i3,' rhokp1 = ',1pe9.2,       &
               ' phat4 = ',1pe9.2,' snd = ',1pe9.2,' **')
  105   Format('** After 4th call: k = ',i3,' rhokp1 = ',1pe9.2,       &
               ' phat4 = ',1pe9.2,' sie4 = ',1pe9.2,' snd4 = ',1pe9.2, &
               ' **')
  106   Format('** End   left rarefaction at point k = ',i3,' **')
!
  900 Format('** RARELEFT: FATAL ERROR   ierr = ',i2,' **')
  901 Format('**   pstar = ',1pe12.5,' <= 0 **')
  902 Format('**   rhol  = ',1pe12.5,' <= 0 **')
  903 Format('**   prsl  = ',1pe12.5,' <= 0 **')
  904 Format('**   Error from EOS_LEFT_RHOP: k = ',i3,' rhok = '   &
             ,1pe12.5,'  phat1 = ',1pe12.5,' **')
  905 Format('**   Error from EOS_LEFT_RHOP: k = ',i3,' rhokph = ' &
             ,1pe12.5,'  phat1 = ',1pe12.5,' **')
  906 Format('**   Error from EOS_LEFT_RHOP: k = ',i3,' rhokph = ' &
             ,1pe12.5,'  phat2 = ',1pe12.5,' **')
  907 Format('**   Error from EOS_LEFT_RHOP: k = ',i3,' rhokph = ' &
             ,1pe12.5,'  phat3 = ',1pe12.5,' **')
  908 Format('**   Error from EOS_LEFT_RHOP: k = ',i3,' rhokp1 = ' &
             ,1pe12.5,'  phat3 = ',1pe12.5,' **')
  909 Format('**   Error from EOS_LEFT_RHOP: k = ',i3,' rhokp1 = ' &
             ,1pe12.5,'  phat4 = ',1pe12.5,' **')
!
!-----------------------------------------------------------------------
!
      End
!
! End of Function RARELEFT
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
! Start of Subroutine BRACKET_RIGHT_RARE
!
      Subroutine BRACKET_RIGHT_RARE( drholin,  drhorin, drholout, &
                                     drhorout, ldebugin, ierr )
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!                                                                      c
! This subroutine takes the input values of left and right             c
! delta-density, "drholin" and "drhorin", and determines whether those c
! arguments bracket a zero of the RARERIGHT function, which is used in c
! a ZEROIN call with RARERIGHT in the calling routine.  If so, then    c
! those values are returned in the output variables "drholout" and     c
! "drhorout".  If not, then the "drhol" or "drhor" are modified, in a  c
! simplistic way, in a search for a pair of values that will bracket a c
! zero;  this modification is attempted for at most "itmax" iterations.c
!                                                                      c
!   Called by: GET_USTAR_RIGHT   Calls: RARERIGHT                      c
!                                                                      c
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
      Implicit none
!
!.... Include files
!
      Include "param.h"
!
!.... Call list variables
!
      Integer :: ierr                     ! error flag
!
      Double Precision :: drholin         ! input  left  delta-density
      Double Precision :: drhorin         ! input  right delta-density
      Double Precision :: drholout        ! output left  delta-density
      Double Precision :: drhorout        ! output right delta-density
!
      Logical :: ldebugin                 ! debug flag
!
!.... Local variables
!
      Integer :: it                       ! iteration counter
      Integer :: itmax                    ! max number of iterations
      Integer :: iun1                     ! debug file unit number
!
      Double Precision :: drhol           ! left  delta-density
      Double Precision :: drhor           ! right delta-density
      Double Precision :: drhomin         ! min delta-density for root-finder
      Double Precision :: drhomax         ! max delta-density max root-finder
      Double Precision :: drhotmp         ! temp delta-density
      Double Precision :: prsndlo         ! RARERIGHT value at drhomin
      Double Precision :: prsndhi         ! RARERIGHT value at drhomax
      Double Precision :: scratch         ! = prsndlo * prsndhi
!
      Logical :: ldebug                   ! .true. => print debug statements
      Logical :: lbracketed               ! .true. => root is bracketed
!
!.... Functions called
!
      Double Precision :: RARERIGHT       ! Evaluate right rarefaction
!
!.... Common block variables
!
      Integer :: nrar                     ! # of steps thru rarefaction
      Double Precision :: rhor            ! right density
      Double Precision :: sier            ! right SIE
      Double Precision :: prsr            ! right pressure
      Double Precision :: sndr            ! right sound speed
      Double Precision :: velr            ! right velocity
      Double Precision :: pstar           ! star-state pressure
!
      Common / right_state / rhor, sier, prsr, sndr, velr
      Common / star_state  / pstar
      Common / num_rare    / nrar
!
!-----------------------------------------------------------------------
!
      ierr = 0
      iun1 = 35                    ! unit number
      ldebug = ldebugin
!*!      If ( ldebug ) Open(unit=iun1,file='bracket_right_rare.dbg', &
!*!                         status='unknown')
!-----------------------------------------------------------------------
      itmax  = 20
!-----------------------------------------------------------------------
      drhol = drholin
      drhor = drhorin
      If ( drhol .le. ZERO ) ierr = 1
      If ( ierr .ne. 0 ) Go To 799
      If ( drhor .le. ZERO ) ierr = 2
      If ( ierr .ne. 0 ) Go To 799
!
!.... Attempts to ensure bracketing values of the delta-density sent to 
!     the ZEROIN call with RARERIGHT that determines the corresponding
!     post-rarefaction pressure; the returned quantity is the difference
!     between that value and the common-block star-state pressure, all
!     nondimensionalized by that star-state pressure
!
      drhomin = 0.9d0 * Min( drhol, drhor )
      drhomax = 1.1d0 * Max( drhol, drhor )
      lbracketed = .false.
      it = 0
!*!      If ( ldebug ) Write(iun1,101)
!*!      If ( ldebug ) Write(iun1,102) it, drhomin, ZERO, drhomax, ZERO
      Do While ( ( .not. lbracketed ) .and. ( it .lt. itmax ) )
        it = it + 1
        prsndlo = RARERIGHT( drhomin )
        prsndhi = RARERIGHT( drhomax )
!*!        If ( ldebug ) Write(iun1,102) it, drhomin, prsndlo,  &
!*!                                          drhomax, prsndhi
        scratch = prsndlo * prsndhi
!
!.... prsndlo * prsndhi = 0
!
        If ( scratch .eq. ZERO ) Then
          If ( prsndhi .ne. ZERO ) Then        ! prsndlo = 0 => increase drhomin?
            drhomin = 1.2d0 * drhomin
          Else If ( prsndlo .ne. ZERO ) Then   ! prsndhi = 0 => decrease drhomax?
            drhomax = 0.8d0 * drhomax
          Else                                 ! prsndlo = prsndhi = 0 => ERROR
            ierr = 3
          End If
!
!.... prsndlo * prsndhi > 0
!
!     Here, the two function values have the same sign: (1) reset the abscissa with 
!     the _larger_ absolute ordinate to be the abscissa with the _smaller_ absolute; 
!     (2) re-assign that smaller-absolute-abscissa ordinate to the linearly extrapolated
!     zero-ordinate abscissa, which is "nudged" by one-tenth of the delta-abcissa-value
!
        Else If ( scratch .gt. ZERO ) Then
          If ( prsndlo .gt. ZERO .and. prsndhi .gt. ZERO ) Then
            If ( prsndlo .lt. prsndhi ) Then   ! 0 < prsndlo < prsndhi
              drhotmp = drhomin                ! 
              drhomin = ( drhotmp * prsndhi - drhomax * prsndlo ) &
                      / ( prsndhi - prsndlo )
              drhomax = drhotmp
              drhotmp = drhomax - drhomin
              drhomin = drhomin - TENTH * drhotmp
            Else                               ! 0 < prsndhi < prsndlo
              drhotmp = drhomax
              drhomax = ( drhomin * prsndhi - drhotmp * prsndlo ) &
                      / ( prsndhi - prsndlo )
              drhomin = drhotmp
              drhotmp = drhomax - drhomin
              drhomax = drhomax + TENTH * drhotmp
            End If
          Else
            If ( prsndlo .lt. prsndhi ) Then   ! prsndlo < prsndhi < 0
              drhotmp = drhomax
              drhomax = ( drhomin * prsndhi - drhotmp * prsndlo ) &
                      / ( prsndhi - prsndlo )
              drhomin = drhotmp
              drhotmp = drhomax - drhomin
              drhomax = drhomax + TENTH * drhotmp
            Else                               ! prsndhi < prsndlo < 0
              drhotmp = drhomin
              drhomin = ( drhotmp * prsndhi - drhomax * prsndlo ) &
                      / ( prsndhi - prsndlo )
              drhomax = drhotmp
              drhotmp = drhomax - drhomin
              drhomin = drhomin - TENTH * drhotmp
            End If
          End If
!
!.... prsndlo * prsndhi < 0
!
        Else
          lbracketed = .true.
        End If
        If ( lbracketed ) Go To 798
        If ( ierr .ne. 0 ) Go To 799           ! ierr = 3
      End Do 
!
!..... Solution that brackets the root has been found
!
 798  Continue
      If ( lbracketed ) Then
        If ( drholin .lt. drhorin ) Then
          drholout = drhomin
          drhorout = drhomax
        Else
          drholout = drhomax
          drhorout = drhomin
        End If
!*!        If ( ldebug ) Write(iun1,200)
      End If
!
      If ( it .ge. itmax ) ierr = 4
      If ( ierr .ne. 0 ) Go To 799
!  
!-----------------------------------------------------------------------
!.... Error conditions
!-----------------------------------------------------------------------
  799 Continue
      If ( ierr .eq. 0 ) Go To 899 
      Write(*,900) ierr
      Go To ( 801, 802, 803, 804 ) ierr
  801 Write(*,901) drhol
      Go To 899 
  802 Write(*,902) drhor
      Go To 899 
  803 Write(*,903) drhomin, prsndlo, drhomax, prsndhi
      Go To 899 
  804 Write(*,904) it, itmax
      Go To 899 
!
!.... Exit
!
  899 Continue
!*!        If ( ldebug ) Close(iun1)
!
!-----------------------------------------------------------------------
!.... Format statements
!-----------------------------------------------------------------------
!
  101 Format(/' it   drhomin   prsndlo   drhomax   prsndhi'   &
             /'---- --------- --------- --------- ---------')
  102 Format(i3,1x,4(1x,1pe9.2))
  200 Format('** Right rarefaction bracketing values FOUND **')
!
  900 Format('** BRACKET_RIGHT_RARE: FATAL ERROR   ierr = ',i2,' **')
  901 Format('** drhol = ',1pe12.5,' <= 0 **')
  902 Format('** drhor = ',1pe12.5,' <= 0 **')
  903 Format('** drhomin = ',1pe12.5,' prsndlo = ',1pe12.5, &
               ' drhomax = ',1pe12.5,' prsndhi = ',1pe12.5)
  904 Format('** it = ',i3,' > ',i3,' = itmax **')
!
!-----------------------------------------------------------------------
!
      Return
      End
!
! End of Subroutine BRACKET_RIGHT_RARE
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
! Start of Function RARERIGHT
!
!.... RIGHT material: Function for ZEROIN of rarefaction delta-rho
!
!.... See:  J.W.Banks, "On Exact Conservation for the Euler Equations
!           with Complex Equations of State," Commun. Comput. Phys.
!           8(5), pp. 995-1015 (2010), doi: 10.4208/cicp/090909/100310a
!
! 141204 Modification from Banks' RK4-like formula to true RK4 formulation,
!        which uses two fewer EOS calls
!
      Double Precision Function RARERIGHT( drho )
!
      Implicit none
!
!.... Include files
!
      Include "param.h"
!
!.... Call list variables
!
      Double Precision :: drho            ! density-increment thru rarefaction
!
!.... Local variables
!
      Integer :: ierr                     ! error flag
      Integer :: k                        ! index
      Integer :: kerr                     ! index at error
!
      Double Precision :: sie             ! EOS-returned SIE
      Double Precision :: snd             ! EOS-returned sound speed
      Double Precision :: csqrd           ! sie**2
      Double Precision :: pk              ! pressure at kth-pt   along rarefaction
      Double Precision :: pkp1            ! pressure at k+1st-pt along rarefaction
      Double Precision :: rhok            ! density  at kth-pt   along rarefaction
      Double Precision :: rhokp1          ! density  at k+1st-pt along rarefaction
      Double Precision :: rhokph          ! ( rhok + rhokp1 ) / 2
      Double Precision :: phat1           ! \hat{p}_1 in Banks
      Double Precision :: phat2           ! \hat{p}_2 in Banks
      Double Precision :: phat3           ! \hat{p}_3 in Banks
      Double Precision :: phat4           ! \hat{p}_4 in Banks
      Double Precision :: snd1            ! snd_spd(rhok,phat1)
      Double Precision :: snd2            ! snd_spd(rhokph,phat2)
      Double Precision :: snd3            ! snd_spd(rhokph,phat3)
      Double Precision :: snd4            ! snd_spd(rhokp1,phat4)
      Logical :: ldebug                   ! debug flag
!
!.... Common block variables
!
      Integer :: nrar                     ! # of steps thru rarefaction
      Double Precision :: rhor            ! right density
      Double Precision :: sier            ! right SIE
      Double Precision :: prsr            ! right pressure
      Double Precision :: sndr            ! right sound speed
      Double Precision :: velr            ! right velocity
      Double Precision :: pstar           ! star-state pressure
!
      Common / right_state / rhor, sier, prsr, sndr, velr
      Common / star_state  / pstar
      Common / num_rare    / nrar
!
!-----------------------------------------------------------------------
!
!     NOTE:  this debug flag is _solely_ for this routine and is NOT
!            passed from any calling routine
!
      ldebug = .false.
      If ( ldebug ) Write(*,*) '** Enter RARERIGHT...**'
      ierr = 0
      If ( pstar .le. ZERO ) ierr = 1
      If ( ierr .ne. 0 ) Go To 799
      If ( rhor .le. ZERO ) ierr = 2
      If ( ierr .ne. 0 ) Go To 799
      If ( prsr .le. ZERO ) ierr = 3
      If ( ierr .ne. 0 ) Go To 799
!
!.... Initialize quantities with the right_state values and proceed
!     along the rarefaction wave from the right state toward the
!     star-state.  See equations following Eq. (3.3) in Banks.
!
      pk   = prsr
      rhok = rhor
      rhokp1 = rhok - drho
      rhokph = HALF * ( rhok + rhokp1 )
!      
!.... Loop over the number of points in the rarefaction
!
      Do k = 2, nrar
        If ( ldebug ) Write(*,100) k
!
!.... Evaluate the intermediate pressures and sound speeds:
!     see equations following Eq. (3.3) in Banks
!
        phat1 = pk
        Call EOS_RIGHT_RHOP( rhok, phat1, sie, snd1, ierr )
        If ( ierr .ne. 0 ) ierr = 4
        kerr = k
        If ( ierr .ne. 0 ) Go To 799
        If ( ldebug ) Write(*,101) k, rhok, phat1, sie, snd1
!        
!.... Banks formula:
!
!        Call EOS_RIGHT_RHOP( rhokph, phat1, sie, snd, ierr )
!        If ( ierr .ne. 0 ) ierr = 5
!        kerr = k
!        If ( ierr .ne. 0 ) Go To 799
!        phat2 = pk - HALF * drho * ( snd * snd )
!        Call EOS_RIGHT_RHOP( rhokph, phat2, sie, snd2, ierr )
!        If ( ierr .ne. 0 ) ierr = 6
!        kerr = k
!        If ( ierr .ne. 0 ) Go To 799
!
!.... RK4 formula:
!
        phat2 = pk - HALF * drho * ( snd1 * snd1 )
        Call EOS_RIGHT_RHOP( rhokph, phat2, sie, snd2, ierr )
        If ( ierr .ne. 0 ) ierr = 6
        kerr = k
        If ( ierr .ne. 0 ) Go To 799
        If ( ldebug ) Write(*,102) k, rhokph, phat2, sie, snd2
!
        phat3 = pk - HALF * drho * ( snd2 * snd2 )
        Call EOS_RIGHT_RHOP( rhokph, phat3, sie, snd3, ierr )
        If ( ierr .ne. 0 ) ierr = 7
        kerr = k
        If ( ierr .ne. 0 ) Go To 799
        If ( ldebug ) Write(*,103) k, rhokph, phat3, sie, snd3
!
!.... Banks formula:
!
!        Call EOS_RIGHT_RHOP( rhokp1, phat3, sie, snd, ierr )
!        If ( ierr .ne. 0 ) ierr = 8
!        kerr = k
!        If ( ierr .ne. 0 ) Go To 799
!        phat4 = pk - HALF * drho * ( snd * snd )
!        If ( ldebug ) Write(*,104) k, rhokp1, phat4, snd
!
!.... RK4 formula:
!
        phat4 = pk - drho * ( snd3 * snd3 )
        Call EOS_RIGHT_RHOP( rhokp1, phat4, sie, snd4, ierr )
        If ( ierr .ne. 0 ) ierr = 9
        kerr = k
        If ( ierr .ne. 0 ) Go To 799
        If ( ldebug ) Write(*,105) k, rhokp1, phat4, sie, snd4
!
!.... Increment the pressure per the RK-like expression
!     following Eq. (3.3) in Banks
!
        pkp1  = pk - SIXTH * drho *                       &
         (       ( snd1 * snd1 ) + TWO * ( snd2 * snd2 )  &
         + TWO * ( snd3 * snd3 ) +       ( snd4 * snd4 ) )
!
!.... Update quantities for the next point on the rarefaction
!
        pk     = pkp1
        rhok   = rhokp1
        rhokp1 = rhok - drho
        rhokph = HALF * ( rhok + rhokp1 )
        If ( ldebug ) Write(*,106) k
!
      End Do ! k
!
      RARERIGHT = ( pk - pstar ) / pstar
      If ( ldebug ) Write(*,*) '** ...Exit RARERIGHT **'
!
!-----------------------------------------------------------------------
!
!.... Error conditions
!
  799 Continue
      If ( ierr .eq. 0 ) Go To 899 
!
!.... Here, there's an error, so set the function to zero before returning
!
      RARERIGHT = ZERO
      Write(*,900) ierr
      Go To ( 801, 802, 803, 804, 805, 806, 807, 808, 809 ) ierr
  801 Write(*,901) pstar
      Go To 899 
  802 Write(*,902) rhor
      Go To 899 
  803 Write(*,903) prsr
      Go To 899 
  804 Write(*,904) kerr, rhok, phat1
      Go To 899 
  805 Write(*,905) kerr, rhokph, phat1
      Go To 899 
  806 Write(*,906) kerr, rhokph, phat2
      Go To 899 
  807 Write(*,907) kerr, rhokph, phat3
      Go To 899 
  808 Write(*,908) kerr, rhokp1, phat3
      Go To 899 
  809 Write(*,909) kerr, rhokp1, phat4
      Go To 899 
!
!.... Exit
!
  899 Continue
!
!-----------------------------------------------------------------------
!.... Format statements
!-----------------------------------------------------------------------
!
  100   Format('** Start right rarefaction at point k = ',i3,' **')
  101   Format('** After 1st call: k = ',i3,' rhok   = ',1pe9.2,       &
               ' phat1 = ',1pe9.2,' sie1 = ',1pe9.2,' snd1 = ',1pe9.2, &
               ' **')
  102   Format('** After 2nd call: k = ',i3,' rhokph = ',1pe9.2,       &
               ' phat2 = ',1pe9.2,' sie2 = ',1pe9.2,' snd2 = ',1pe9.2, &
               ' **')
  103   Format('** After 3rd call: k = ',i3,' rhokph = ',1pe9.2,       &
               ' phat3 = ',1pe9.2,' sie3 = ',1pe9.2,' snd3 = ',1pe9.2, &
               ' **')
  104   Format('** After 4th call: k = ',i3,' rhokp1 = ',1pe9.2,       &
               ' phat4 = ',1pe9.2,' snd = ',1pe9.2,' **')      
  105   Format('** After 4th call: k = ',i3,' rhokp1 = ',1pe9.2,       &
               ' phat4 = ',1pe9.2,' sie4 = ',1pe9.2,' snd4 = ',1pe9.2, &
               ' **')
  106   Format('** End   right rarefaction at point k = ',i3,' **')
!
  900 Format('** RARERIGHT: FATAL ERROR   ierr = ',i2,' **')
  901 Format('**   pstar = ',1pe12.5,' <= 0 **')
  902 Format('**   rhor  = ',1pe12.5,' <= 0 **')
  903 Format('**   prsr  = ',1pe12.5,' <= 0 **')
  904 Format('**   Error from EOS_RIGHT_RHOP: k = ',i3,' rhok = '   &
             ,1pe12.5,'  phat1 = ',1pe12.5,' **')
  905 Format('**   Error from EOS_RIGHT_RHOP: k = ',i3,' rhokph = ' &
             ,1pe12.5,'  phat1 = ',1pe12.5,' **')
  906 Format('**   Error from EOS_RIGHT_RHOP: k = ',i3,' rhokph = ' &
             ,1pe12.5,'  phat2 = ',1pe12.5,' **')
  907 Format('**   Error from EOS_RIGHT_RHOP: k = ',i3,' rhokph = ' &
             ,1pe12.5,'  phat3 = ',1pe12.5,' **')
  908 Format('**   Error from EOS_RIGHT_RHOP: k = ',i3,' rhokp1 = ' &
             ,1pe12.5,'  phat3 = ',1pe12.5,' **')
  909 Format('**   Error from EOS_RIGHT_RHOP: k = ',i3,' rhokp1 = ' &
             ,1pe12.5,'  phat4 = ',1pe12.5,' **')
!
!-----------------------------------------------------------------------
!
      End
!
! End of Function RARERIGHT
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
! Start of Function ZEROIN
!
      Double Precision Function ZEROIN( ax, bx, f, tol )
!
      Double Precision ax           ! left  endpoint of search interval
      Double Precision bx           ! right endpoint of search interval
      Double Precision f            ! EXTERNAL function whose zero is sought
      Double Precision tol          ! relative solution tolerance
      External f
!
!.... Forsythe, Malcolm & Moler 1D root-finding routine
!
!     see:  http://www.netlib.org/fmm/zeroin.f
!
!     a zero of the function  f(x)  is computed in the interval ax,bx .
!
!  input..
!
!  ax     left endpoint of initial interval
!  bx     right endpoint of initial interval
!  f      function subprogram which evaluates f(x) for any x in
!         the interval  ax,bx
!  tol    desired length of the interval of uncertainty of the
!         final result ( .ge. 0.0d0)
!
!  output..
!
!  zeroin abcissa approximating a zero of  f  in the interval ax,bx
!
!      it is assumed  that   f(ax)   and   f(bx)   have  opposite  signs
!  without  a  check.  zeroin  returns a zero  x  in the given interval
!  ax,bx  to within a tolerance  4*macheps*abs(x) + tol, where macheps
!  is the relative machine precision.
!      this function subprogram is a slightly  modified  translation  of
!  the algol 60 procedure  zero  given in  richard brent, algorithms for
!  minimization without derivatives, prentice - hall, inc. (1973).
!
      Double Precision  a,b,c,d,e,eps,fa,fb,fc,tol1,xm,p,q,r,s
      Double Precision  dabs,dsign
      Double Precision  scratch
      Logical           ldebug
!
! --> NOTE:  this debug flag is _ONLY_ for this routine and 
!                            is _NOT_  passed from any calling routine
!
      ldebug = .false.
!
!  compute eps, the relative machine precision
!
      eps = 1.0d0
   10 eps = eps/2.0d0
      tol1 = 1.0d0 + eps
      if (tol1 .gt. 1.0d0) go to 10
!
! initialization
!
      a = ax
      b = bx
      fa = f(a)
      fb = f(b)
!
      If ( ldebug ) Write(*,100) a,b,fa,fb
 100  Format('** ZEROIN:  a = ',1pe9.2,' b = ',1pe9.2,        &
                       ' fa = ',1pe9.2,' fb = ',1pe9.2,' **')
!
! begin step
!
   20 c = a
!   20 Write(*,*) '** ZEROIN: at 20 **'
!      c = a
      fc = fa
      d = b - a
      e = d
!
   30 if (dabs(fc) .ge. dabs(fb)) go to 40
!   30 Write(*,*) '** ZEROIN: at 30 **'
!      if (dabs(fc) .ge. dabs(fb)) go to 40
      a = b
      b = c
      c = a
      fa = fb
      fb = fc
      fc = fa
!
! convergence test
!
   40 tol1 = 2.0d0*eps*dabs(b) + 0.5d0*tol
!   40 Write(*,*) '** ZEROIN: at 40 **'
!      tol1 = 2.0d0*eps*dabs(b) + 0.5d0*tol
      xm = 0.5d0*(c - b)
      if (dabs(xm) .le. tol1) go to 90
      if (fb .eq. 0.0d0) go to 90
!
! is bisection necessary
!
      if (dabs(e) .lt. tol1) go to 70
      if (dabs(fa) .le. dabs(fb)) go to 70
!
! is quadratic interpolation possible
!
      if (a .ne. c) go to 50
!
! linear interpolation
!
      s = fb/fa
      p = 2.0d0*xm*s
      q = 1.0d0 - s
      go to 60
!
! inverse quadratic interpolation
!
   50 q = fa/fc
!   50 Write(*,*) '** ZEROIN: at 50 **'
!      q = fa/fc
      r = fb/fc
      s = fb/fa
      p = s*(2.0d0*xm*q*(q - r) - (b - a)*(r - 1.0d0))
      q = (q - 1.0d0)*(r - 1.0d0)*(s - 1.0d0)
!
! adjust signs
!
   60 if (p .gt. 0.0d0) q = -q
!   60 Write(*,*) '** ZEROIN: at 60 **'
!      if (p .gt. 0.0d0) q = -q
      p = dabs(p)
!
! is interpolation acceptable
!
      if ((2.0d0*p) .ge. (3.0d0*xm*q - dabs(tol1*q))) go to 70
      if (p .ge. dabs(0.5d0*e*q)) go to 70
      e = d
      d = p/q
      go to 80
!
! bisection
!
   70 d = xm
!   70 Write(*,*) '** ZEROIN: at 70 **'
!      d = xm
      e = d
!
! complete step
!
   80 a = b
!   80 Write(*,*) '** ZEROIN: at 80 **'
!      a = b
      fa = fb
      if (dabs(d) .gt. tol1) b = b + d
      if (dabs(d) .le. tol1) b = b + dsign(tol1, xm)
      fb = f(b)
      if ((fb*(fc/dabs(fc))) .gt. 0.0d0) go to 20
      go to 30
!
! done
!
   90 zeroin = b
!   90 Write(*,*) '** ZEROIN: at 90 **'
!      zeroin = b
!
      fb = f(b)
      If ( ldebug ) Write(*,200) b, fb
 200  Format('** ZEROIN:  root = ',1pe9.2,' value = ',1pe9.2,' **')
!      
      return
      end
!
! End of Function ZEROIN
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
! Start of Subroutine SPLINE
!
!.... Forsythe, Malcolm & Moler spline routine
!
!     see:  http://www.netlib.org/fmm/spline.f
!
!.... Compute the coefficients b,c,d for a cubic interpolating spline
!     so that the interpolated value is given by
!       s(x) = y(k) + b(k)*(x-x(k)) + c(k)*(x-x(k))**2 + d(k)*(x-x(k))**3
!         when x(k) <= x <= x(k+1)
!     The end conditions match the third derivatives of the interpolated curve to
!     the third derivatives of the unique polynomials thru the first four and
!     last four points.
!
!     Use SPLEVAL to evaluate the spline.
!
      Subroutine SPLINE( n, x, y, b, c, d, ierr )
!
      Implicit none
!
!.... Include files
!
      Include "param.h"
!
!.... Call list variables
!
      Integer :: n                        ! number of pts in array
      Integer :: ierr                     ! error flag
!
      Double Precision :: x(1:n)          ! abscissas of knots
      Double Precision :: y(1:n)          ! ordinates of knots
      Double Precision :: b(1:n)          ! linear coeff.
      Double Precision :: c(1:n)          ! quadratic coeff.
      Double Precision :: d(1:n)          ! cubic coeff.
!
!.... Local variables
!
      Integer k                           ! counter
      Integer kstar                       ! counter value
      Double Precision :: t               ! t = d(k-1) / b(k-1)
!
!-----------------------------------------------------------------------
!
      ierr = 0
      If ( n .lt. 2 ) ierr = 1
      If ( ierr .ne. 0 ) Go To 799
      Do k = 1, n-1
        If ( x(k+1) - x(k) .le. ZERO ) ierr = 2
        kstar = k
        If ( ierr .ne. 0 ) Go To 799
      End Do ! k
!
!.... Straight line - special case for n < 3
!
      If ( n .lt. 3 ) Then
        b(1) = ZERO
        If ( n .eq. 2 ) b(1) = ( y(2) - y(1) ) / ( x(2) - x (1) )
        c(1) = ZERO
        d(1) = ZERO
        b(2) = b(1)
        c(2) = ZERO
        d(2) = ZERO
        Go To 799
      End If ! n .lt. 3
!
!.... Set up tridiagonal system:
!     b = diagonal, d = offdiagonal, c = right-hand side
!
      d(1) = x(2) - x(1)
      c(2) = ( y(2) - y(1) ) / d(1)
      Do k = 2, n-1
        d(k) = x(k+1) - x(k)
        b(k) = TWO * ( d(k-1) + d(k) )
        c(k+1) = ( y(k+1) - y(k) ) / d(k)
        c(k) = c(k+1) - c(k)
      End Do ! k
!
!.... End conditions:  Third derivatives at x(1) and x(n) obtained
!                      from divided differences
!
      b(1) = -d(1)
      b(n) = -d(n-1)
      c(1) = ZERO
      c(n) = ZERO
      If ( n .gt. 3 ) Then
        c(1) = c(3)   / ( x(4) - x(2)   ) - c(2)   / ( x(3)   - x(1)   )
        c(n) = c(n-1) / ( x(n) - x(n-2) ) - c(n-2) / ( x(n-1) - x(n-3) )
        c(1) =  c(1) * d(1)   * d(1)   / ( x(4) - x(1)   )
        c(n) = -c(n) * d(n-1) * d(n-1) / ( x(n) - x(n-3) )
      End If ! n
!
!.... Forward elimination
!
      Do k = 2, n
        t = d(k-1) / b(k-1)
        b(k) = b(k) - t * d(k-1)
        c(k) = c(k) - t * c(k-1)
      End Do ! k
!
!.... Back substitution ( makes c the sigma of text)
!
      Do k = n, 1, -1
        If ( b(k) .eq. ZERO ) ierr = 3
        kstar = k
        If ( ierr .ne. 0 ) Go To 799
      End Do ! k
      c(n) = c(n) / b(n)
      Do k = n-1 , 1, -1
        c(k) = ( c(k) - d(k) * c(k+1) ) / b(k)
      End Do ! k
!
!.... Compute polynomial coefficients
!
      b(n) = ( y(n) - y(n-1) ) / d(n-1)        &
           + d(n-1) * ( c(n-1) + c(n) + c(n) )
      Do k = 1, n-1
        b(k) = ( y(k+1) - y(k) ) / d(k)        &
             - d(k) * ( c(k+1) + c(k) + c(k) )
        d(k) = ( c(k+1) - c(k) ) / d(k)
        c(k) = THREE * c(k)
      End Do ! k
      c(n) = THREE * c(n)
      d(n) = d(n-1)
!
!-----------------------------------------------------------------------
!
!.... Error conditions
!
  799 Continue
      If ( ierr .eq. 0 ) Go To 899 
      Write(*,900) ierr
      Go To ( 801, 802, 803 ) ierr
  801 Write(*,901) n
      Go To 899 
  802 Write(*,902) kstar+1, kstar, x(kstar+1) - x(kstar)
      Go To 899 
  803 Write(*,903) kstar, b(kstar)
      Go To 899 
!
!.... Exit
!
  899 Continue
!
!-----------------------------------------------------------------------
!.... Format statements
!-----------------------------------------------------------------------
!
  900 Format('** SPLINE: FATAL ERROR   ierr = ',i2,' **')
  901 Format('**   n = ',i3,' < 2 **')
  902 Format('**   x(',i3,') - x(',i3,') = ',1pe12.5,' <= 0 **')
  903 Format('**   b(',i3,') = ',1pe12.5,' = 0 **')
!
!-----------------------------------------------------------------------
!
      End
!
! End of Subroutine SPLINE
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
! Start of Function SPLEVAL
!
!.... Forsythe, Malcolm & Moler spline evaluation routine
!
!     see:  http://www.netlib.org/fmm/seval.f
!
!.... Evaluate the cubic spline function
!     SPLEVAL = y(i)+b(i)!(u-x(i))+c(i)*(u-x(i))**2+d(i)*(u-x(i))**3
!     where  x(i) < =  u < x(i+1)
!
!     Note:  if u < x(1), i = 1 is used;  if u > x(n), i = n is used
!
      Double Precision Function SPLEVAL( n, u, x, y, b, c, d )
!
      Implicit none
!
!.... Include files
!
      Include "param.h"
!
!.... Local variables
!
      Integer :: n                        ! number of pts in array
!
      Double Precision :: u               ! abscissa  at which to evaluate
      Double Precision :: x(1:n)          ! abscissas of knots
      Double Precision :: y(1:n)          ! ordinates of knots
      Double Precision :: b(1:n)          ! linear coefficient
      Double Precision :: c(1:n)          ! quadratic coefficient
      Double Precision :: d(1:n)          ! cubic coefficient
!
!.... Local variables
!
      Integer i                           ! counter
      Integer j                           ! counter
      Integer k                           ! counter
      Save    i
      Data    i / 1 /
      Double Precision :: dx              ! dx = u - x(i) 
!
!-----------------------------------------------------------------------
!
!.... First check if u is in the same interval found on the last call
!
      If ( ( i .lt. 1 ) .or. ( i .ge. n ) ) i = 1
      If ( ( u .lt. x(i) )  .or.  ( u .ge. x(i+1) ) ) Then
        i = 1   ! binary search
        j = n + 1
        Do
          k = ( i + j ) / 2
          If ( u .lt. x(k) ) Then
            j = k
          Else
            i = k
          End If 
          If ( j .le. i+1 ) Exit
        End Do
      End If
!
!.... Evaluate the spline with Horner's Rule
!
      dx = u - x(i) 
      SPLEVAL = y(i) + dx * ( b(i) + dx * ( c(i) + dx * d(i) ) )
!
!-----------------------------------------------------------------------
!
      End
!
! End of Function SPLEVAL
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
! Start of Subroutine RIEMANN_ERROR
!
      Subroutine RIEMANN_ERROR( ierr, it, itmax, drhol, drhor, &
                                plft, prght, pstar, scratch)
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!                                                                      c
! This subroutine writes out the specific error flagged in RIEMANN     c
!                                                                      c
!   Called by: RIEMANN   Calls: none                                   c
!                                                                      c
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
      Implicit none
!
!.... Include files
!
      Include "param.h"
!
      Integer :: ierr                     ! error flag
      Integer :: it                       ! secant iteration number
      Integer :: itmax                    ! max number of secant iterations
      Double Precision :: drhol           ! delta-rho for left  rarefaction integration
      Double Precision :: drhor           ! delta-rho for right rarefaction integration
      Double Precision :: plft            ! left  pressure
      Double Precision :: prght           ! right pressure
      Double Precision :: pstar           ! star-state pressure
      Double Precision :: scratch         ! scratch value
!
!-----------------------------------------------------------------------
!
      Write(*,900) ierr
      Go To ( 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, &
              811, 812, 813, 814, 815, 816, 817, 818, 819, 820, &
              821, 822, 823, 824, 825, 826, 827, 828, 829, 830, &
              831) ierr
  801 Write(*,901)  
      Go To 899 
  802 Write(*,902)  
      Go To 899 
  803 Write(*,903)
      Go To 899 
  804 Write(*,904)
      Go To 899 
  805 Write(*,905) it
      Go To 899 
  806 Write(*,906) it
      Go To 899 
  807 Write(*,907) itmax
      Go To 899 
  808 Write(*,908) 
      Go To 899 
  809 Write(*,909)
      Go To 899 
  810 Write(*,910)
      Go To 899 
  811 Write(*,911) drhol
      Go To 899 
  812 Write(*,912)
      Go To 899 
  813 Write(*,913)
      Go To 899 
  814 Write(*,914)  
      Go To 899 
  815 Write(*,915)
      Go To 899 
  816 Write(*,916)  
      Go To 899 
  817 Write(*,917) plft, pstar
      Go To 899 
  818 Write(*,918)
      Go To 899 
  819 Write(*,919)  
      Go To 899 
  820 Write(*,920)
      Go To 899 
  821 Write(*,921) drhor
      Go To 899 
  822 Write(*,922)  
      Go To 899 
  823 Write(*,923)
      Go To 899 
  824 Write(*,924)  
      Go To 899 
  825 Write(*,925)
      Go To 899 
  826 Write(*,926)
      Go To 899 
  827 Write(*,927) prght, pstar
      Go To 899 
  828 Write(*,928)
      Go To 899 
  829 Write(*,929)
      Go To 899 
  830 Write(*,930)
      Go To 899 
  831 Write(*,931)
      Go To 899 
!
!.... Exit
!
  899 Continue
!
!-----------------------------------------------------------------------
!.... Format statements
!-----------------------------------------------------------------------
!
  900 Format('** RIEMANN: FATAL ERROR   ierr = ',i2,' **')
  901 Format('** Error return from EOS_LEFT_RHOP **')
  902 Format('** Error return from EOS_RIGHT_RHOP **')
  903 Format('** Error return from GET_USTAR_LEFT, k = 0 **')
  904 Format('** Error return from GET_USTAR_RIGHT, k =  **')
  905 Format('** Error return from GET_USTAR_LEFT, k = ',i2,' **')
  906 Format('** Error return from GET_USTAR_RIGHT, k = ',i2,' **')
  907 Format('** NO convergence after ',i2,' iterations **')
  908 Format('** LEFT Shock: Fatal error from BRACKET_LEFT_SHOCK **')
!  908 Format('** LEFT Shock ZEROIN bounds: rhoshklo * rhoshkhi = ' &
!      ,1pe12.5,' >= 0 **')
  909 Format('** Error return from LEFT Shock: EOS_LEFT_RHOP **')
  910 Format('** LEFT Rarefaction: Fatal error from' &
      ' BRACKET_LEFT_RARE **')
!  910 Format('** LEFT Rarefaction ZEROIN bounds: prsndlo * prsndhi = ' &
!      ,1pe12.5,' >= 0 **')
  911 Format('** Error return from LEFT Rarefaction: ZEROIN: drhol = ' &
      ,1pe12.5,' **')
  912 Format('** Error return from LEFT Rarefaction: RARELEFTSOLN **')
  913 Format('** Error return from LEFT Rarefaction: SPLINE -- ' &
      ,'interpolation of density **')
  914 Format('** Error return from LEFT Rarefaction: SPLINE -- ' &
      ,'interpolation of pressure **')
  915 Format('** Error return from LEFT Rarefaction: SPLINE -- ' &
      ,'interpolation of velocity **')
  916 Format('** Error return from LEFT star-state: EOS_LEFT_RHOP **')
  917 Format('** Error return from LEFT state: plft = ',1pe12.5 &
      ,' = ',1pe12.5,' = pstar **')
  918 Format('** RIGHT Shock: Fatal error from BRACKET_RIGHT_SHOCK **')
!c  918 Format('** RIGHT Shock ZEROIN bounds: rhoshklo * rhoshkhi = ' &
!      ,1pe12.5,' >= 0 **')
  919 Format('** Error return from RIGHT Shock: EOS_RIGHT_RHOP **')
  920 Format('** Right Rarefaction: Fatal error from' &
      ' BRACKET_RIGHT_RARE **')
!  920 Format('** RIGHT Rarefaction ZEROIN bounds: prsndlo * prsndhi = ' &
!      ,1pe12.5,' >= 0 **')
  921 Format('** Error return from RIGHT Rarefaction ZEROIN: drhor = ' &
      ,1pe12.5,' **')
  922 Format('** Error return from RIGHT Rarefaction: RARERIGHTSOLN **')
  923 Format('** Error return from RIGHT Rarefaction: SPLINE -- ' &
      ,'interpolation of density **')
  924 Format('** Error return from RIGHT Rarefaction: SPLINE -- ' &
      ,'interpolation of pressure **')
  925 Format('** Error return from RIGHT Rarefaction: SPLINE -- ' &
      ,'interpolation of velocity **')
  926 Format('** Error return from RIGHT Rarefaction:' &
      ,' EOS_RIGHT_RHOP **')
  927 Format('** Error return from RIGHT state: prght = ',1pe12.5 &
      ,' = ',1pe12.5,' = pstar **')
  928 Format('** Error return from LEFT Rarefaction: EOS_LEFT_RHOP **')
  929 Format('** Error return from LEFT Rarefaction: EOS_LEFT_RHOP **')
  930 Format('** Error return from RIGHT Rarefaction: ', &
      'EOS_RIGHT_RHOP **')
  931 Format('** Error return from RIGHT Rarefaction: ', &
      'EOS_RIGHT_RHOP **')
!
!-----------------------------------------------------------------------
!
      Return
      End
!
! End of Subroutine RIEMANN_ERROR
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
