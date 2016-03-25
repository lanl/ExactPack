!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!   This subroutine computes the solution to the Sedov blast-wave problem
!   at a specified time for planar, cylindrical, or spherical symmetry,
!   for a gamma-law gas, with data output to a 1D, 2D, or 3D file in
!   Cartesian, cylindrical, or spherical coordinates.  In the 1D case,
!   the output coordinates must match the ascribed symmetry of the 
!   problem;  in the 2D and 3D case, the output coordinates need not
!   match the symmetry of the problem.
!
!   130426 -- updated code for Mac gfortran
!   130426 -- modified for 1D only: no 2D/3D calls, no 2D/3D output
!   150515 -- outputs non-dimensional function values
!   150720 -- scaled down version for ExactPack.
!   150730 -- port to ExactPack: self_sim=T uses self-sim vars f, g, h
!                                                                          
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      Subroutine sedov_kamm_1d( nx, time, xval, eblast, rho1, 
     &     omega, u1, e1, p1, c1, gamval, icoord, 
     $     rhoval, eval, pval, uval, self_sim )
!f2py intent(out)  :: rhoval, eval, pval, uval
!f2py intent(hide) :: nx
!f2py integer      :: nx
!f2py real*8       :: time, xval(nx)
!f2py real*8       :: eblast, rho1, omega, u1, e1, p1, c1, gamval
!f2py integer      :: icoord
!f2py real*8       :: rhoval(nx), eval(nx), pval(nx), uval(nx)
!f2py logical      :: self_sim
!
! Buffer subroutine between python and Kamm's original source code.
!
      Implicit None
      Include "param.h"
      Integer :: nx
      Real*8  time                  ! time for similarity solution     (s)
      Real*8  xval(1:nx)            ! positions 
      Real*8  eblast                ! initial blast energy             (erg)
      Real*8  rho1                  ! initial pre-blast density const  (gm/cm^3)
      Real*8  omega                 ! initial pre-blast density exponent
      Real*8  u1                    ! initial pre-blast x-velocity     (cm/s)     
      Real*8  e1                    ! initial pre-blast SIE            (erg/gm)  
      Real*8  p1                    ! initial pre-blast pressure       (dyne/cm^2)
      Real*8  c1                    ! initial pre-blast snd spd        (cm/s)
      Real*8  gamval                ! value of gamma for gamma-law gas
      Real*8  rhoval(1:nx)          ! density
      Real*8  uval(1:nx)            ! velocity
      Real*8  eval(1:nx)            ! SIE
      Real*8  pval(1:nx)            ! pressure
      Logical self_sim               ! .true. => use Sedov lambda values for check
      Integer icoord                ! coordinate system for output values

      Real*8  xmin                  ! minimum x-coordinate for solution
      Real*8  xmax                  ! maximum x-coordinate for solution
      Real*8  abserr                ! absolute error in the energy integrals
      Real*8  relerr                ! relative error in the energy integrals
      Integer nshk                  ! # of sol'n pts between origin & shock
      Integer nfar                  ! # of sol'n pts between shock  & rmax
      Logical lpla                  ! .true. => compute planar      sedov sol'n
      Logical lcyl                  ! .true. => compute cylindrical sedov sol'n
      Logical lsph                  ! .true. => compute spherical   sedov sol'n
      
      Real*8 v_n, v_c, v_l, v_r     ! self-similar variable V. See (7)-(11) LA-UR-00-6055
      Real*8 l_fun                  ! self-similar variable lambda = r/r2
      Real*8 f_fun                  ! self-similar variable f = v/v2
      Real*8 g_fun                  ! self-similar variable g = rho/rho2
      Real*8 h_fun                  ! self-similar variable h = p/p2
      Real*8 atemp, btemp           ! temporary variables
      Real*8 SEDFUN0, ZEROIN        ! functions
      Integer it                    ! iteration index

      abserr  = 1.d-8               ! absolute error tolerance
      relerr  = 1.d-8               ! relative error tolerance
      nshk    = 400                 ! number of point in/before/or after? the shock
      nfar    = 100                 ! number of point in the rarefaction.
      If (icoord .eq. 3) Then       ! spherical
         lpla    = .false.
         lcyl    = .false.
         lsph    = .true.
      Else IF (icoord .eq. 2) Then ! cylindrical
         lpla    = .false.
         lcyl    = .true.
         lsph    = .false.
      Else IF (icoord .eq. 1) Then ! planar
         lpla    = .true.
         lcyl    = .false.
         lsph    = .false.
      Else 
         print *, "sedov.f: ERROR"
         print *, "Only icoord values of 1,2, or 3 are permitted."
      End If

      ! if self_sim=.true.: use dimensionless self-similar variables
      ! input : xval=lambda
      ! output: rhoval=g, eval=v_ss, pval=h, uval=f, and eval=V
      If (self_sim) Then 
         ! Call self-similar routine
         Call sedov_self_similar(nx, xval, 
     &        omega, gamval, lpla, lcyl, lsph,                 
     &        self_sim, uval, rhoval, pval, eval)

      ! if self_sim=.false: use dimenional physical variables
      ! input : xval
      ! output: rhoval, eval, pval, uval
      Else              
         xmin = xval(1)
         xmax = xval(nx)
         Call sedov_1d(nx, xmin, xmax, xval, eblast, rho1, 
     &        omega, u1, e1, p1, c1, gamval, 
     &        abserr, relerr,
     &        time, nshk, nfar, lpla, lcyl, lsph,
     &        self_sim, icoord, 
     &        rhoval, eval, pval, uval)
      End If

      Return
      End Subroutine sedov_kamm_1d
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

      Subroutine sedov_self_similar ( nx, lambda, omega, gamval, 
     &                      lpla, lcyl, lsph, self_sim, f, g, h, v )

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!   This routine computes the solution to the Sedov problem
!   in dimensionless self-similar variables.
!   input : lambda 
!   output: f, g, h, V
!                                                                          
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Start of Subroutine sedov_self_similar
 
      Implicit None
 
!.... Include files
 
      Include "param.h"

!.... Call list variables
      
      Integer nx                    ! x-grid size of output file
 
      Real*8  omega                 ! initial pre-blast density exponent
      Real*8  gamval                ! value of gamma for gamma-law gas
      Real*8  lambda(1:nx)          ! position
      Real*8  f(1:nx)               ! velocity
      Real*8  g(1:nx)               ! density
      Real*8  h(1:nx)               ! pressure
      Real*8  v(1:nx)               ! position
      Logical lpla                  ! .true. => compute planar      sedov sol'n
      Logical lcyl                  ! .true. => compute cylindrical sedov sol'n
      Logical lsph                  ! .true. => compute spherical   sedov sol'n
      Logical self_sim               ! .true. => use Sedov lambda values for check

!.... Local variables

      Real*8  l_fun                 ! distance similarity function
      Real*8  f_fun                 ! velocity similarity function
      Real*8  g_fun                 ! density  similarity function
      Real*8  h_fun                 ! pressure similarity function
      Real*8  gamma                 ! common-block value of gamma
      Real*8  gamm1                 ! gamma minus one
      Real*8  gamp1                 ! gamma plus  one
      Real*8  gpogm                 ! ( gamma + 1 ) / ( gamma - 1 )
      Real*8  a0                    ! constant in function evaluation
      Real*8  a1                    ! constant in function evaluation
      Real*8  a2                    ! constant in function evaluation
      Real*8  a3                    ! constant in function evaluation
      Real*8  a4                    ! constant in function evaluation
      Real*8  a5                    ! constant in function evaluation
      Real*8  a_val                 ! scratch scalar
      Real*8  b_val                 ! scratch scalar
      Real*8  c_val                 ! scratch scalar
      Real*8  d_val                 ! scratch scalar
      Real*8  e_val                 ! scratch scalar
      Real*8  sedval                ! Sedov value
      Real*8  SEDFUN0               ! function:  zero of Sedov function in lambda, omega .eq. 0
      Real*8  ZEROIN                ! function:  root finder

      Real*8  v_n                   ! converged value from ZEROIN call
      Real*8  v_l                   ! "left"   value  in   ZEROIN call
      Real*8  v_r                   ! "right"  value  in   ZEROIN call
      Real*8  v_c                   ! "center" value  for  ZEROIN call
      Real*8  atemp                 ! temporary scalar
      Real*8  btemp                 ! temporary scalar

      Integer i                     ! index
      Integer ierr                  ! error flag
      Integer ii                    ! index
      Integer iflag                 ! integral evaluation reliability flag
      Integer it                    ! iteration counter
      Integer iun1                  ! unit number of output ascii file
      Integer iun2                  ! unit number of output ascii file w/exact sol'n points
      Integer iun3                  ! unit number of output ascii file w/non-dim fun values
      Integer j                     ! geometry flag: 2 => cyl, 3 => sph
      Integer jp2                   ! j + 2
      Integer numvac                ! # of positions between x=0 & inner bndry

      Integer key                   ! key for local integration rule
      Integer last                  ! # of subintervals produced
      Integer limit                 ! # of subintervals for integration
      Parameter ( limit = 8192 )
      Integer lenw                  ! dimension parameter for work  array
      Parameter ( lenw  = 46 * limit )
      Integer leniw                 ! dimension parameter for iwork array
      Parameter ( leniw =  3 * limit )
      Integer iwork(1:leniw)        ! scratch integer work array
      Real*8  work(1:lenw)          ! scratch real*8  work array

      Logical lsng                  ! singular-state solution flag
      Logical lvac                  ! vaccum-region-near-origin flag
      Logical debug                 ! print .dat files if true

      Integer iexact(1:nx)          ! position flag
  
      Character*16 fileout          ! output filename

!.... Common blocks, etc.

      Common / gascon / gamma, gamm1, gamp1, gpogm
      Common / simfuni / j, jp2
      Common / simfunr / a0, a1, a2, a3, a4, a5,
     &                   a_val, b_val, c_val, d_val, e_val
      Common / sedcom / sedval

      External EFUN01, EFUN02, EFUN11, EFUN12
      External SEDFUN0, SEDFUNR0, SEDFUNR1
      External ZEROIN

!-----------------------------------------------------------------------
 
      Do i = 1, nx
        f(i) = ZERO
        g(i) = ZERO
        h(i) = ZERO
        v(i) = ZERO
      End Do ! i

!-----------------------------------------------------------------------
! populate values in Common/gascon
      ierr  = 0
      gamma = gamval
      gamm1 = gamma - ONE
      gamp1 = gamma + ONE
      gpogm = gamp1 / gamm1

!.... Note the symmetry of the problem

      If ( lpla ) Then
        j = 1
      Else If ( lcyl ) Then
        j = 2
      Else
        j = 3
      End If ! lcyl

!.... Determine solution type:  regular, singular, or vacuum

      lsng = .false.
      lvac = .false.
      If ( omega .gt. (j*(THREE-gamma)+TWO*gamm1)/gamp1 ) Then
        lvac = .true.
        If ( lvac .and. lpla ) ierr = 2
      Else If
     &   ( Abs( omega * gamp1 - (j*(THREE-gamma)+TWO*gamm1) )
     &   .lt. EPS8 ) Then
        lsng = .true.
        If ( lsng .and. lpla ) ierr = 3
      End If ! omega
      If ( ierr .ne. 0 ) Go To 799

!=======================================================================
!.... Assign values used in function definition
!=======================================================================

      jp2 = j + 2
      If ( Abs( omega ) .le. EPS8 ) Then       ! constant density case
        a0  = TWO / Dble( jp2 )
        a2  = -gamm1 / ( TWO * gamm1 + j )
        a1  =  jp2 * gamma / ( TWO + j * gamm1 )
     &      * ( ( ( TWO * j * ( TWO - gamma ) ) 
     &          / ( gamma * jp2**2 ) ) - a2 )
        a3  = j / ( TWO * gamm1 + j )
        a4  = jp2 * a1 / ( TWO - gamma )
        a5  = -TWO / ( TWO - gamma )
      Else                                     ! variable density case
        a0  = TWO / Dble( jp2 - omega)
        a2  = -gamm1 / ( TWO * gamm1 + Dble( j ) - gamma * omega )
        a1  = ( jp2 - omega ) * gamma / ( TWO + Dble( j ) * gamm1 )
     &      * ( ( ( TWO * ( Dble( j ) * ( TWO - gamma ) - omega ) ) 
     &          / ( gamma * ( jp2 - omega )**2 ) ) - a2 )
        a3  = ( Dble( j ) - omega ) /
     &        ( TWO * gamm1 + Dble( j ) - gamma * omega )
        a4  = ( Dble( jp2 ) - omega ) * ( Dble( j ) - omega ) * a1 / 
     &        ( Dble( j ) * ( TWO - gamma ) - omega )
        a5  = ( ( omega * gamp1 ) - TWO * Dble( j ) ) / 
     &        ( Dble( j ) * ( TWO - gamma ) - omega )
      End If ! Abs

      If ( Abs( omega ) .le. EPS8 ) Then       ! constant density case
        a_val = FOURTH * jp2 * gamp1
        b_val = gpogm
        c_val = HALF * jp2 * gamma
        d_val = ( jp2 * gamp1 ) 
     &        / ( jp2 * gamp1 - TWO * ( TWO + j * gamm1 ) )
        e_val = HALF * ( TWO + j * gamm1 )
      Else                                     ! variable density case
        a_val = FOURTH * ( jp2 - OMEGA ) * gamp1
        b_val = gpogm
        c_val = HALF * ( jp2 - OMEGA ) * gamma
        d_val = ( ( jp2 - omega ) * gamp1 ) 
     &        / ( ( jp2 - omega ) * gamp1 - TWO * ( TWO + j * gamm1 ) )
        e_val = HALF * ( TWO + Dble( j ) * gamm1 )
      End If ! Abs

! given V and lambda, find f, g, h
      If ( lpla ) v_c = 0.55d0 ! planar value
      If ( lcyl ) v_c = 0.42d0 ! cylindrical value
      If ( lsph ) v_c = 0.33d0 ! spherical value
      Do i = 1, nx
         sedval = lambda(i)
         it  = 0
         v_l = v_c - 0.02d0
         v_r = v_c + 0.02d0
!        Write(6,*) ' v_l = ',v_l,'  v_r = ',v_r
 100     Continue               ! bracket the zero
         atemp = SEDFUN0( v_l )
!        Write(6,*) ' atemp = ',atemp
         btemp = SEDFUN0( v_r ) 
!        Write(6,*) ' btemp = ',btemp
         If ( atemp * btemp .lt. ZERO ) Go To 200
         it = it + 1
         If ( it .gt. 100 ) Stop '** Cannot bracket SEDOV 100 tries **'
         v_l = v_l + TENTH * ( v_c - v_l )
         v_r = v_r - TENTH * ( v_c - v_r )
         Go To 100
 200     Continue               ! compute the zero
         v_n = ZEROIN( v_l, v_r, SEDFUN0, EPS16 )

         Call GET_FUN0( v_n, l_fun, f_fun, g_fun, h_fun )
         f(i) = f_fun
         g(i) = g_fun
         h(i) = h_fun
         v(i) = v_n
!        Write(6,*)
!        Write(6,*) v_n, l_fun, f_fun, g_fun, h_fun
         v_c = v_n ! reset initial guess
      End Do ! i

  799 Continue
      If ( ierr .eq. 0 ) Go To 899
      Write(*,900)
      Go To ( 801, 802, 803 ) ierr
  801 Write(*,901)
      Go To 899
  802 Write(*,902)
      Go To 899
  803 Write(*,903)
      Go To 899
  899 Continue
  900 Format('** sedov_self_similar:  Fatal Error **')
  901 Format('** Error in namelist input:  omega .ge. j+2 **')
  902 Format('** Error:  planar vacuum case not allowed **')
  903 Format('** Error:  planar singular case not allowed **')

      End Subroutine sedov_self_similar
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
 
      Subroutine sedov_1d ( nx, xmin, xmax, xval, eblast, rho1, 
     &                      omega, u1, e1, p1, c1, gamval, 
     &                      abserr, relerr,
     &                      time, nshk, nfar, lpla, lcyl, lsph,
     &                      llambda, icoord, 
     &                      rhoval, eval, pval, uval )
 
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!   This routine computes the solution to the Sedov problem
!   at a specified time for planar, cylindrical, or spherical 
!   geometry, for a gamma-law gas, and outputs the result to a 1-D file
!
!   If lpla then the planar-sym'c Sedov problem with x = planar "x"
!   If lcyl then the cyl'ly-sym'c Sedov problem with x = cylindrical "r"
!   If lsph then the sph'ly-sym'c Sedov problem with x = spherical "r"
!
!   Called by: SEDOV          Calls: DQXG, SEDOV_CHECK, GET_FUN0, 
!                                    GET_FUN1, ZEROIN
!2013  OUTPUT_HDF_1D is not supported
!                                                                          
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Start of Subroutine sedov_1d
 
      Implicit None
 
!.... Include files
 
      Include "param.h"

!.... Call list variables
      
      Integer icoord                ! coordinate system for output values
      Integer nx                    ! x-grid size of output file
      Integer nshk                  ! # of sol'n pts between origin & shock
      Integer nfar                  ! # of sol'n pts between shock  & rmax
 
      Real*8  eblast                ! initial blast energy             (erg)
      Real*8  rho1                  ! initial pre-blast density const  (gm/cm^3)
      Real*8  omega                 ! initial pre-blast density exponent
      Real*8  u1                    ! initial pre-blast x-velocity     (cm/s)     
      Real*8  e1                    ! initial pre-blast SIE            (erg/gm)  
      Real*8  p1                    ! initial pre-blast pressure       (dyne/cm^2)
      Real*8  c1                    ! initial pre-blast snd spd        (cm/s)
      Real*8  gamval                ! value of gamma for gamma-law gas
      Real*8  abserr                ! absolute error in the energy integrals
      Real*8  relerr                ! relative error in the energy integrals
      Real*8  time                  ! time for similarity solution     (s)
      Real*8  xmin                  ! minimum x-coordinate for solution
      Real*8  xmax                  ! maximum x-coordinate for solution
      Real*8  xval(1:nx)            ! positions
      Real*8  rhoval(1:nx)          ! density
      Real*8  uval(1:nx)            ! velocity
      Real*8  eval(1:nx)            ! SIE
      Real*8  pval(1:nx)            ! pressure
      Logical lpla                  ! .true. => compute planar      sedov sol'n
      Logical lcyl                  ! .true. => compute cylindrical sedov sol'n
      Logical lsph                  ! .true. => compute spherical   sedov sol'n
      Logical llambda               ! .true. => use Sedov lambda values for check

!.... Local variables
  
      Integer ival                  ! variable flag for HDF output

      Real*8  t                     ! time                                 (s)
      Real*8  r2                    ! shock position                       (cm)
      Real*8  r2val                 ! shock position (common block)        (cm)
      Real*8  rmax                  ! r-max for sim'ty solution            (cm)
      Real*8  rexact                ! desired exact position               (cm)
      Real*8  us                    ! shock velocity                       (cm/s)
      Real*8  u2                    ! immediate post-shock radial velocity (cm/s)
      Real*8  rho2                  ! immediate post-shock density         (gm/cm^3)
      Real*8  rhop                  ! immediate pre-shock  density         (gm/cm^3)
      Real*8  p2                    ! immediate post-shock pressure        (dyne/cm^2)
      Real*8  e2                    ! immediate post-shock internal energy (erg/gm)
      Real*8  c2                    ! immediate post-shock sound speed     (cm/s)
      Real*8  rex                   ! exact position                       (cm)
      Real*8  rhoex                 ! exact density                        (gm/cm^3)
      Real*8  uex                   ! exact velocity                       (cm/s)
      Real*8  pex                   ! exact pressure                       (dyne/cm^2)
      Real*8  eex                   ! exact internal energy                (erg/gm)
      Real*8  cex                   ! exact sound speed                    (cm/s)
      Real*8  v0                    ! similarity parameter @ origin
      Real*8  v0_int                ! similarity parameter near the origin
      Real*8  vs                    ! similarity parameter @ shock
      Real*8  dv                    ! delta-similarity parameter
      Real*8  v                     ! similarity parameter
      Real*8  vprev                 ! previous similarity parameter
      Real*8  dr                    ! delta-position
      Real*8  dx                    ! delta-position
      Real*8  drv                   ! delta-position for vacuum case
      Real*8  r                     ! position
      Real*8  rprev                 ! previous similarity solution position
      Real*8  rsim                  ! similarity solution position
      Real*8  rsimv                 ! similarity solution position of vacuum region extent
      Real*8  usim                  ! similarity solution velocity
      Real*8  rhosim                ! similarity solution density
      Real*8  psim                  ! similarity solution pressure
      Real*8  esim                  ! similarity solution internal energy
      Real*8  csim                  ! similarity solution snd spd
      Real*8  alpha                 ! ratio of blast energy to non-dim. blast energy
      Real*8  xiw                   ! (1/alpha)**(1/(j+2-kappa))
      Real*8  endim                 ! nondimensional blast energy
      Real*8  EFUN01                ! function:  first  energy integral integrand, omega .eq. 0
      Real*8  EFUN02                ! function:  second energy integral integrand, omega .eq. 0
      Real*8  EFUN11                ! function:  first  energy integral integrand, omega .ne. 0
      Real*8  EFUN12                ! function:  second energy integral integrand, omega .ne. 0
      Real*8  SEDFUN0               ! function:  zero of Sedov function in lambda, omega .eq. 0
      Real*8  SEDFUNR0              ! function:  zero of Sedov function in r, omega .eq. 0
      Real*8  SEDFUNR1              ! function:  zero of Sedov function in r, omega .ne. 0
      Real*8  ZEROIN                ! function:  root finder
      Real*8  eval1                 ! first  energy integral
      Real*8  eval2                 ! second energy integral
      Real*8  errest                ! estimate of the absolute error in the energy integrals
      Real*8  l_fun                 ! distance similarity function
      Real*8  f_fun                 ! velocity similarity function
      Real*8  g_fun                 ! density  similarity function
      Real*8  h_fun                 ! pressure similarity function
      Real*8  omval                 ! common-block value of omega
      Real*8  gamma                 ! common-block value of gamma
      Real*8  gamm1                 ! gamma minus one
      Real*8  gamp1                 ! gamma plus  one
      Real*8  gpogm                 ! ( gamma + 1 ) / ( gamma - 1 )
      Real*8  a0                    ! constant in function evaluation
      Real*8  a1                    ! constant in function evaluation
      Real*8  a2                    ! constant in function evaluation
      Real*8  a3                    ! constant in function evaluation
      Real*8  a4                    ! constant in function evaluation
      Real*8  a5                    ! constant in function evaluation
      Real*8  a_val                 ! scratch scalar
      Real*8  b_val                 ! scratch scalar
      Real*8  c_val                 ! scratch scalar
      Real*8  d_val                 ! scratch scalar
      Real*8  e_val                 ! scratch scalar
      Real*8  sedval                ! Sedov value
      Real*8  vstar                 ! big-V function in singular case
      Real*8  zstar                 ! big-Z function in singular case

      Real*8  v_n                   ! converged value from ZEROIN call
      Real*8  v_l                   ! "left"   value  in   ZEROIN call
      Real*8  v_r                   ! "right"  value  in   ZEROIN call

      Integer i                     ! index
      Integer ierr                  ! error flag
      Integer ii                    ! index
      Integer iflag                 ! integral evaluation reliability flag
      Integer it                    ! iteration counter
      Integer iun1                  ! unit number of output ascii file
      Integer iun2                  ! unit number of output ascii file w/exact sol'n points
      Integer iun3                  ! unit number of output ascii file w/non-dim fun values
      Integer j                     ! geometry flag: 2 => cyl, 3 => sph
      Integer jp2                   ! j + 2
      Integer numvac                ! # of positions between x=0 & inner bndry

      Integer key                   ! key for local integration rule
      Integer last                  ! # of subintervals produced
      Integer limit                 ! # of subintervals for integration
      Parameter ( limit = 8192 )
      Integer lenw                  ! dimension parameter for work  array
      Parameter ( lenw  = 46 * limit )
      Integer leniw                 ! dimension parameter for iwork array
      Parameter ( leniw =  3 * limit )
      Integer iwork(1:leniw)        ! scratch integer work array
      Real*8  work(1:lenw)          ! scratch real*8  work array

      Logical lsng                  ! singular-state solution flag
      Logical lvac                  ! vaccum-region-near-origin flag
      Logical debug                 ! print .dat files if true

      Integer iexact(1:nx)          ! position flag
  
      Character*16 fileout          ! output filename

!.... Common blocks, etc.

      Common / gascon  / gamma, gamm1, gamp1, gpogm
      Common / simfuni / j, jp2
      Common / simfunr / a0, a1, a2, a3, a4, a5, 
     &                   a_val, b_val, c_val, d_val, e_val
      Common / sedcom  / sedval
      Common / rexcom  / r2val, rexact
      Common / omcom   / omval

      External EFUN01, EFUN02, EFUN11, EFUN12
      External SEDFUN0, SEDFUNR0, SEDFUNR1
      External ZEROIN

!-----------------------------------------------------------------------
 
!.... Initialize the output arrays

!         Do i = 1, nx
!           xval(i) = ZERO
!         End Do ! i
!      xmin = xval(1)
!      xmax = xval(nx)
!      print *, xmin, xmax
!      print *, xval(1), xval(nx)
!      stop
      Do i = 1, nx
        rhoval(i) = ZERO
        uval(i)   = ZERO
        eval(i)   = ZERO
        pval(i)   = ZERO
      End Do ! i

!-----------------------------------------------------------------------

      ierr  = 0

      gamma = gamval
      gamm1 = gamma - ONE
      gamp1 = gamma + ONE
      gpogm = gamp1 / gamm1
      omval = omega

!.... Open output ASCII files

      debug = .false.
      If (debug) Then
      iun1 = 11
      Open( iun1, file='sedov.1d.dat', status='new')
      Write(iun1,*) '  '
      iun2 = iun1 + 1
      Open( iun2, file='sedov.1d.ex.dat', status='new')
      Write(iun2,*) '  '
      iun3 = iun2 + 1
      Open( iun3, file='sedov.1d.nd.dat', status='new')
      Write(iun3,*) '  '
      End If

!.... Assign the cell-centered positions at which the solution is sought

!         dx = ( xmax - xmin ) / nx
!         xval(1) = HALF * dx
!         Do i = 2, nx
!           xval(i) = xval(1) + ( i - 1 ) * dx
!         End Do ! i

!.... Note the symmetry of the problem

      If ( lpla ) Then
        j = 1
        !Write(iun1,100)
      Else If ( lcyl ) Then
        j = 2
        !Write(iun1,101)
      Else
        j = 3
        !Write(iun1,102)
      End If ! lcyl
      If ( omega .ge. j + 2 ) ierr = 1
      If ( ierr .ne. 0 ) Go To 799

!.... Determine solution type:  regular, singular, or vacuum

      lsng = .false.
      lvac = .false.
      If ( omega .gt. (j*(THREE-gamma)+TWO*gamm1)/gamp1 ) Then
        lvac = .true.
        !Write(*,105)
        !Write(iun1,105)
        If ( lvac .and. lpla ) ierr = 2
      Else If ( Abs( omega * gamp1 - (j*(THREE-gamma)+TWO*gamm1) )
     &  .lt. EPS8 ) Then
        lsng = .true.
        !Write(*,106)
        !Write(iun1,106)
        If ( lsng .and. lpla ) ierr = 3
      Else
        !Write(*,107)
        !Write(iun1,107)
      End If ! omega
      If ( ierr .ne. 0 ) Go To 799

      !Write(iun1,110) eblast
      !Write(iun1,111) rho1
      !Write(iun1,116) omega
      !Write(iun1,112) u1
      !Write(iun1,113) e1
      !Write(iun1,114) p1
      !Write(iun1,115) c1
      !Write(iun1,*) ' '

!=======================================================================
!.... Assign values used in function definition
!=======================================================================

      jp2 = j + 2
      If ( Abs( omega ) .le. EPS8 ) Then       ! constant density case
        a0  = TWO / Dble( jp2 )
        a2  = -gamm1 / ( TWO * gamm1 + j )
        a1  =  jp2 * gamma / ( TWO + j * gamm1 )
     &      * ( ( ( TWO * j * ( TWO - gamma ) ) 
     &          / ( gamma * jp2**2 ) ) - a2 )
        a3  = j / ( TWO * gamm1 + j )
        a4  = jp2 * a1 / ( TWO - gamma )
        a5  = -TWO / ( TWO - gamma )
      Else                                     ! variable density case
        a0  = TWO / Dble( jp2 - omega)
        a2  = -gamm1 / ( TWO * gamm1 + Dble( j ) - gamma * omega )
        a1  = ( jp2 - omega ) * gamma / ( TWO + Dble( j ) * gamm1 )
     &      * ( ( ( TWO * ( Dble( j ) * ( TWO - gamma ) - omega ) ) 
     &          / ( gamma * ( jp2 - omega )**2 ) ) - a2 )
        a3  = ( Dble( j ) - omega ) /
     &        ( TWO * gamm1 + Dble( j ) - gamma * omega )
        a4  = ( Dble( jp2 ) - omega ) * ( Dble( j ) - omega ) * a1 / 
     &        ( Dble( j ) * ( TWO - gamma ) - omega )
        a5  = ( ( omega * gamp1 ) - TWO * Dble( j ) ) / 
     &        ( Dble( j ) * ( TWO - gamma ) - omega )
      End If ! Abs
      !Write(*,*) '  '
      !Write(*,135) a0, a1, a2, a3, a4, a5

      If ( Abs( omega ) .le. EPS8 ) Then       ! constant density case
        a_val = FOURTH * jp2 * gamp1
        b_val = gpogm
        c_val = HALF * jp2 * gamma
        d_val = ( jp2 * gamp1 ) 
     &        / ( jp2 * gamp1 - TWO * ( TWO + j * gamm1 ) )
        e_val = HALF * ( TWO + j * gamm1 )
      Else                                     ! variable density case
        a_val = FOURTH * ( jp2 - OMEGA ) * gamp1
        b_val = gpogm
        c_val = HALF * ( jp2 - OMEGA ) * gamma
        d_val = ( ( jp2 - omega ) * gamp1 ) 
     &        / ( ( jp2 - omega ) * gamp1 - TWO * ( TWO + j * gamm1 ) )
        e_val = HALF * ( TWO + Dble( j ) * gamm1 )
      End If ! Abs
      !Write(*,*) '  '
      !Write(*,136) a_val, b_val, c_val, d_val, e_val

!.... Assign the nondimensional locations of the origin & shock

      If ( .not. lvac .and. .not. lsng ) Then  ! no vacuum region
        If ( Abs( omega ) .le. EPS8 ) Then     ! constant density case
          v0 = TWO  / ( jp2 * gamma )
          vs = FOUR / ( jp2 * gamp1 )
        Else                                   ! variable density case
          v0 = TWO  / ( ( jp2 - omega ) * gamma )
          vs = FOUR / ( ( jp2 - omega ) * gamp1 )
        End If ! Abs
      Else If ( lvac ) Then                    ! vacuum region
        If ( Abs( omega ) .le. EPS8 ) Then     ! constant density case
          v0 = TWO  / jp2
          vs = FOUR / ( jp2 * gamp1 )
        Else                                   ! variable density case
          v0 = TWO  / ( jp2 - omega )
          vs = FOUR / ( ( jp2 - omega ) * gamp1 )
        End If ! Abs
      Else If ( lsng ) Then                    ! singular-state solution
        v0 = TWO / ( j * gamm1 + TWO )
        vs = TWO / ( j * gamm1 + TWO )
      End If ! .not. lvac
      !Write(*,*) '  '
      !Write(*,137) v0, vs

!=======================================================================
!.... Evaluate the energy integrals in the non-singular solution cases
!=======================================================================

      If ( .not. lsng ) Then

!.... The integrands are singular near the lower limit of integration,
!.... so modify that value (i.e., v0) by a small amount
!.... Recall, in the vacuum case, v0 > vs, so decrement in that case

        v0_int = v0 + TWO * EPS15 * ( vs - v0 )
        !Write(*,138) v0_int
        !Write(*,*) '  '
        key = 4

        If ( Abs( omega ) .le. EPS8 ) Then       ! constant density case
          Call DQXG( EFUN01, v0_int, vs, abserr, relerr, key, 
     &    eval1, errest, iflag, limit, leniw, lenw, last, iwork, work )
        Else                                     ! variable density case
          Call DQXG( EFUN11, v0_int, vs, abserr, relerr, key, 
     &    eval1, errest, iflag, limit, leniw, lenw, last, iwork, work )
        End If ! Abs
        !Write(iun1,120) eval1, errest, iflag, last
        !If ( iflag .ne. 0 ) Then
          !Write(iun1,122)
          !If ( iflag .eq. 1 ) Write(iun1,161)
          !If ( iflag .eq. 2 ) Write(iun1,162)
          !If ( iflag .eq. 3 ) Write(iun1,163)
          !If ( iflag .eq. 4 ) Write(iun1,164)
          !If ( iflag .eq. 5 ) Write(iun1,165)
          !If ( iflag .eq. 6 ) Write(iun1,166)
        !End If ! iflag

        If ( Abs( omega ) .le. EPS8 ) Then       ! constant density case
          Call DQXG( EFUN02, v0_int, vs, abserr, relerr, key, 
     &    eval2, errest, iflag, limit, leniw, lenw, last, iwork, work )
        Else                                     ! variable density case
          Call DQXG( EFUN12, v0_int, vs, abserr, relerr, key, 
     &    eval2, errest, iflag, limit, leniw, lenw, last, iwork, work )
        End If ! Abs
        !Write(iun1,121) eval2, errest, iflag, last
        !If ( iflag .ne. ZERO ) Then
          !Write(iun1,123)
          !If ( iflag .eq. 1 ) Write(iun1,161)
          !If ( iflag .eq. 2 ) Write(iun1,162)
          !If ( iflag .eq. 3 ) Write(iun1,163)
          !If ( iflag .eq. 4 ) Write(iun1,164)
          !If ( iflag .eq. 5 ) Write(iun1,165)
          !If ( iflag .eq. 6 ) Write(iun1,166)
        !End If ! iflag
        !Write(iun1,*) '  '

!.... Compute the nondimensional energy

        If ( lpla ) Then
          alpha = HALF * eval1 + eval2 / gamm1
        Else
          alpha = ( j - 1 ) * PI    * eval1
     &          + ( j - 1 ) * TWOPI * eval2 / gamm1
        End If ! lpla
        If ( Abs( omega ) .le. EPS8 ) Then       ! constant density case
          xiw = ( ONE / alpha ) **( ONE / jp2 )
        Else                                     ! variable density case
          xiw = ( ONE / alpha ) **( ONE / ( jp2 - omega ) )
        End If ! Abs
        !Write(iun1,125) j, gamma, alpha, xiw
        !Write(iun1,*) '  '
        endim = eblast / alpha
        !Write(iun1,126) endim

!.... Check results against Sedov values in the rho_0 = const, 
!.... gamma = 1.4 case;  see Sedov, p. 222-223

        If ( Abs( omega )         .le. EPS8 .and. 
     &       Abs( gamma - 1.4d0 ) .le. EPS8 ) 
     &    Call SEDOV_CHECK ( lpla, lcyl, lsph, llambda )

      Else 

!.... Evaluate "alpha" in the singular case

        If ( j .ne. 1 ) Then
          alpha = PI * TWO**j * ( gamp1 / gamm1 ) / j /
     &            ( gamm1 * j + TWO )**2
        Else
          alpha = TWO / gamp1 / gamm1
        End If ! j .ne. 1
        xiw = ( ONE / alpha ) **( ONE / ( jp2 - omega ) )
        !Write(iun1,125) j, gamma, alpha, xiw
        endim = eblast / alpha
        !Write(iun1,126) endim

      End If ! .not. lsng

!=======================================================================
!.... Compute the similarity solution over the chosen time interval
!=======================================================================

      t = time

!.... Set exact-position flags all to false

      Do i = 1, nx
        iexact(i) = 0
      End Do ! i

!.... Compute immediate post-shock values using strong shock relations

      If ( Abs( omega ) .le. EPS8 ) Then           ! constant density case
        r2   = ( endim / rho1 )**( ONE / jp2 )     ! shock position    (cm)
     &       * t**( TWO / jp2 )
        us   = ( TWO / jp2 ) * r2 / t              ! shock velocity    (cm/s)
        rhop = rho1                                ! pre-shock density (gm/cm^3)
        u2   = TWO * us / gamp1                    ! flow  velocity    (cm/s)
        rho2 = gpogm * rhop                        ! density           (gm/cm^3)
        p2   = TWO * rhop * us**2 / gamp1          ! pressure          (dyne/cm^2)
        e2   = p2 / gamm1 / rho2                   ! internal energy   (erg/gm)
        c2   = Sqrt ( gamma * p2 / rho2 )          ! sound speed       (cm/s)
      Else                                         ! variable density case
        r2   = ( endim / rho1 )**( ONE / ( jp2 - omega ) )
     &       * t**( TWO / ( jp2 - omega ) )        ! shock position    (cm)
        us   = ( TWO / ( jp2 - omega ) ) * r2 / t  ! shock velocity    (cm/s)
        rhop = rho1 * r2**(-omega)                 ! pre-shock density (gm/cm^3)
        u2   = TWO * us / gamp1                    ! flow  velocity    (cm/s)
        rho2 = gpogm * rhop                        ! density           (gm/cm^3)
        p2   = TWO * rhop * us**2 / gamp1          ! pressure          (dyne/cm^2)
        e2   = p2 / gamm1 / rho2                   ! internal energy   (erg/gm)
        c2   = Sqrt ( gamma * p2 / rho2 )          ! sound speed       (cm/s)
      End If ! Abs

      !Write(iun1,146) r2
      !Write(iun1,147) rhop
      !Write(iun1,141) rho2
      !Write(iun1,142) u2
      !Write(iun1,143) e2
      !Write(iun1,144) p2
      !Write(iun1,145) c2
      If ( lvac ) Then                             ! position of vacuum boundary
        rsimv = ( a_val * v0 )**( -a0 ) *
     &          ( b_val * Max( EPS16, c_val * v0 - ONE ) )**( -a2 ) *
     &          ( d_val * ( ONE - e_val * v0 ) )**( -a1 )
        !Write(iun1,148) rsimv
      End If ! lvac

      !Write(iun1,*) '  '
      If ( icoord .eq. 1) Then
        !Write(iun1,129)
        !Write(iun2,129)
      Else
        !Write(iun1,130)
        !Write(iun2,130)
      End If ! icoord
      !Write(iun1,131)
      !Write(iun2,131)

!-----------------------------------------------------------------------
!.... Singular state case:  assign values between the origin & shock
!-----------------------------------------------------------------------

      If ( lsng ) Then

        vstar = TWO / ( gamm1 * j + TWO )
        zstar = TWO * gamma * gamm1 / ( gamm1 * j + TWO )**2

!.... Assign solution values only at desired exact solution points

        Do i = 1, nx
          If ( ( xval(i)   .lt. r2   ) .and. 
     &         ( xval(i)   .gt. ZERO ) .and. 
     &         ( iexact(i) .eq. 0    ) ) Then
            iexact(i) = 1
            rexact    = xval(i)                   ! position        (cm)

!.... Sedov's solution in Eq. 14.11 (p.265)

            l_fun = rexact / r2
            If ( j .ne. 2 ) Then                  ! density         (gm/cm^3)
              rhoex = rho2 * l_fun**(j-2)
            Else
              rhoex = rho2
            End If ! j .ne. 2
            uex   = u2   * l_fun                  ! flow velocity   (cm/s)
            pex   = p2   * l_fun**j               ! pressure        (dyne/cm^2)
            eex   = pex / gamm1 / rhoex           ! internal energy (erg/gm)
            cex   = sqrt ( gamma * pex / rhoex )  ! sound speed     (cm/s)
            !Write(iun1,140) t, rexact, rhoex, uex, eex, pex, cex
            !Write(iun2,140) t, rexact, rhoex, uex, eex, pex, cex
            rhoval(i) = rhoex
            uval(i)   = uex
            eval(i)   = eex
            pval(i)   = pex

          End If ! xval
        End Do ! i

!-----------------------------------------------------------------------
!.... Non-singular cases
!-----------------------------------------------------------------------

      Else

!-----------------------------------------------------------------------
!.... Vacuum case:  assign the values between the origin & inner boundary
!-----------------------------------------------------------------------

        If ( lvac ) Then
          numvac = 100
          drv    = rsimv / numvac
          rprev  = ZERO
          !Write(iun1,140) t, rprev, ZERO,ZERO,ZERO,ZERO,ZERO
          Do i = 1, numvac
            rsim = i * drv
            !Write(iun1,140) t, rsim, ZERO,ZERO,ZERO,ZERO,ZERO
            Do ii = 1, nx
              If ( ( rprev      .lt. xval(ii)  ) .and. 
     &             ( xval(ii)   .le. rsim      ) .and. 
     &             ( iexact(ii) .eq. 0         ) ) Then
                iexact(ii) = 1
                rexact     = xval(ii) 
                !Write(iun2,140) t, rexact, ZERO,ZERO,ZERO,ZERO,ZERO
                rhoval(ii) = ZERO
                uval(ii)   = ZERO
                eval(ii)   = ZERO
                pval(ii)   = ZERO
              End If ! rprev
            End Do ! ii
            rprev = rsim

          End Do ! i
        End If ! lvac

!.... Compute the values immediately near the lower limit, where
!.... v0_int  has been modified to be slightly away from v0

        !Write(*,150)
        v = v0_int                                        ! similarity parameter
        If ( Abs( omega ) .le. EPS8 ) Then                ! constant density case
          Call GET_FUN0( v, l_fun, f_fun, g_fun, h_fun )
        Else                                              ! variable density case
          Call GET_FUN1( v, l_fun, f_fun, g_fun, h_fun, omega )
        End If ! Abs
        !If ( it .eq. 0 ) Write(*,151) v, l_fun, f_fun, g_fun, h_fun
        rsim   = r2   * l_fun                             ! position        (cm)
        rhosim = rho2 * g_fun                             ! density         (gm/cm^3)
        usim   = u2   * f_fun                             ! flow velocity   (cm/s)
        psim   = p2   * h_fun                             ! pressure        (dyne/cm^2)
        esim   = psim / gamm1 / rhosim                    ! internal energy (erg/gm)
        csim   = Sqrt ( gamma * psim / rhosim )           ! sound speed     (cm/s)
        !Write(iun1,140) t, rsim, rhosim, usim, esim, psim, csim

!-----------------------------------------------------------------------
!.... Non-singular cases: compute values between the lower limit & shock
!-----------------------------------------------------------------------

        r2val = r2
        v  = v0_int                                 ! similarity parameter
        dv = ( vs - v0 ) / nshk
        Do i = 1, nshk-1
          vprev = v
          rprev = rsim
          v = v0 + i * dv                           ! similarity parameter
          If ( Abs( omega ) .le. EPS8 ) Then        ! constant density case
            Call GET_FUN0( v, l_fun, f_fun, g_fun, h_fun )
          Else                                      ! variable density case
            Call GET_FUN1( v, l_fun, f_fun, g_fun, h_fun, omega )
          End If ! Abs
          !If ( it .eq. 0 ) Write(*,151) v, l_fun, f_fun, g_fun, h_fun
          rsim   = r2   * l_fun                     ! position        (cm)
          rhosim = rho2 * g_fun                     ! density         (gm/cm^3)
          usim   = u2   * f_fun                     ! flow velocity   (cm/s)
          psim   = p2   * h_fun                     ! pressure        (dyne/cm^2)
          esim   = psim / gamm1 / rhosim            ! internal energy (erg/gm)
          csim   = sqrt ( gamma * psim / rhosim )   ! sound speed     (cm/s)
          !Write(iun1,140) t, rsim, rhosim, usim, esim, psim, csim

!.... Search for v-values that bracket desired exact solution points

          Do ii = 1, nx
            If ( ( rprev      .lt. xval(ii)  ) .and. 
     &           ( xval(ii)   .le. rsim      ) .and. 
     &           ( iexact(ii) .eq. 0         ) ) Then
              iexact(ii) = 1
              rexact     = xval(ii) 
              v_l = vprev
              v_r = v
              If ( Abs( omega ) .le. EPS8 ) Then    ! constant density case
                v_n = ZEROIN( v_l, v_r, SEDFUNR0, EPS16 )
                Call GET_FUN0( v_n, l_fun, f_fun, g_fun, h_fun)
              Else                                  ! variable density case
                v_n = ZEROIN( v_l, v_r, SEDFUNR1, EPS16 )
                Call GET_FUN1( v_n, l_fun, f_fun, g_fun, h_fun, omega )
              End If ! Abs

              rex   = r2   * l_fun                  ! position        (cm)
              rhoex = rho2 * g_fun                  ! density         (gm/cm^3)
              uex   = u2   * f_fun                  ! flow velocity   (cm/s)
              pex   = p2   * h_fun                  ! pressure        (dyne/cm^2)
              eex   = pex / gamm1 / rhoex           ! internal energy (erg/gm)
              cex   = sqrt ( gamma * pex / rhoex )  ! sound speed     (cm/s)
              !Write(iun2,140) t, rex, rhoex, uex, eex, pex, cex
              rhoval(ii) = rhoex
              uval(ii)   = uex
              eval(ii)   = eex
              pval(ii)   = pex
   
            End If ! rprev
          End Do ! ii

        End Do ! i

      End If ! lsng

!-----------------------------------------------------------------------
!.... Write the values at the shock
!-----------------------------------------------------------------------

      !Write(iun1,140) t,   r2, rho2, u2, e2, p2, c2
      If ( Abs( omega ) .le. EPS8 ) Then            ! constant density case
        !Write(iun1,140) t, r2, rho1, u1, e1, p1, c1
      Else                                          ! variable density case
        !Write(iun1,140) t, r2, rhop, u1, e1, p1, c1
      End If ! Abs

!-----------------------------------------------------------------------
!.... Write the values beyond the shock
!-----------------------------------------------------------------------

      rmax = xmax - xmin
      dr   = ( rmax - r2 ) / nfar
      Do i = 1, nfar
        rprev = rsim
        r = r2 + i * dr                             ! radius value
        If ( Abs( omega ) .le. EPS8 ) Then          ! constant density case
          !Write(iun1,140) t, r, rho1, u1, e1, p1, c1
        Else                                        ! variable density case
          !Write(iun1,140) t, r, rho1*r**(-omega), u1, e1, p1, c1
        End If ! Abs
      End Do ! i

      Do i = 1, nx
        If ( iexact(i) .eq. 0 ) Then
          If ( xval(i) .ge. r2 ) Then
            iexact(i) = 1
            If ( Abs( omega ) .le. EPS8 ) Then
              rhoex = rho1
            Else
              rhoex = rho1*xval(i)**(-omega)
            End If ! Abs( omega )
            !Write(iun2,140) t, xval(i), rhoex, u1, e1, p1, c1
            rhoval(i) = rhoex
            uval(i)   = u1
            eval(i)   = e1
            pval(i)   = p1
            pval(i)   = p1
          Else
            !Write(*,160)    i, xval(i)
            !Write(iun2,160) i, xval(i)
          End If ! xval(i)
        End If ! iexact
      End Do ! i

      Close(iun1)
      Close(iun2)
      Close(iun3)

!2013 OUTPUT_HDF_1D is not supported
!-----------------------------------------------------------------------
!.... Write the HDF files
!-----------------------------------------------------------------------
!
!2013      ival = 1
!2013      fileout = 'sedov_1d.rho.hdf'
!2013      Call OUTPUT_HDF_1D (fileout, ival, xval, rhoval, nx )
!2013      ival = 2
!2013      fileout = 'sedov_1d.vel.hdf'
!2013      Call OUTPUT_HDF_1D (fileout, ival, xval, uval, nx )
!2013      ival = 3
!2013      fileout = 'sedov_1d.sie.hdf'
!2013      Call OUTPUT_HDF_1D (fileout, ival, xval, eval, nx )
!2013      ival = 4
!2013      fileout = 'sedov_1d.prs.hdf'
!2013      Call OUTPUT_HDF_1D (fileout, ival, xval, pval, nx )
!
!-----------------------------------------------------------------------

  799 Continue
      If ( ierr .eq. 0 ) Go To 899
      Write(*,900)
      Go To ( 801, 802, 803 ) ierr
  801 Write(*,901)
      Go To 899
  802 Write(*,902)
      Go To 899
  803 Write(*,903)
      Go To 899
  899 Continue

!-----------------------------------------------------------------------
 
!.... Format statements

  100 Format(' Planar blast wave problem: 1-D output')
  101 Format(' Cylindrical blast wave problem: 1-D output')
  102 Format(' Spherical blast wave problem: 1-D output')
  105 Format(' * Vacuum solution *')
  106 Format(' * Singular solution *')
  107 Format(' * Standard solution *')
  110 Format('   Total energy               = ',1pe12.5,' erg')
  111 Format('   Pre-shock density constant = ',1pe12.5,' gm/cm3')
  112 Format('   Pre-shock velocity         = ',1pe12.5,' cm/s')
  113 Format('   Pre-shock internal energy  = ',1pe12.5,' erg/gm')
  114 Format('   Pre-shock pressure         = ',1pe12.5,' dyne/cm2')
  115 Format('   Pre-shock sound speed      = ',1pe12.5,' cm/s')
  116 Format('   Pre-shock density exponent = ',1pe12.5)
  120 Format(' 1st energy integral = ',1pe12.5,'   error = ',1pe12.5,
     &       '  error flag = ',i1,'  last = ',i8)
  121 Format(' 2nd energy integral = ',1pe12.5,'   error = ',1pe12.5
     &       '  error flag = ',i1,'  last = ',i8)
  122 Format(' * WARNING:  1st energy integral may be unreliable *')
  123 Format(' * WARNING:  2nd energy integral may be unreliable *')
  124 Format(' *   nofun = ',i8,'  flag = ',f6.3,' *')
  125 Format(' j = ',i2,'  gamma = ',f14.7,'  alpha = ',f14.7,
     &       '  xiw = ',f14.7)
  126 Format(' Non-dimensional energy = endim = ',1pe14.7)
  129 Format('      t         x         dens       vel        ener      
     & pres       c_s')
  130 Format('      t         r         dens       vel        ener      
     & pres       c_s')
  131 Format('     (s)       (cm)     (gm/cm3)    (cm/s)    (erg/gm)  (d
     &yne/cm2)   (cm/s)')
  135 Format(' a0 = ',1pe12.5/' a1 = ',1pe12.5/' a2 = ',1pe12.5/
     &       ' a3 = ',1pe12.5/' a4 = ',1pe12.5/' a5 = ',1pe12.5)
  136 Format(' aval = ',1pe12.5/' bval = ',1pe12.5/' cval = ',
     & 1pe12.5/' dval = ',1pe12.5/' eval = ',1pe12.5)
  137 Format(' v0     = ',1pe19.12/' vs     = ',1pe19.12)
  138 Format(' v0_int = ',1pe19.12)

  140 Format(7(1x,1pe10.4))
  141 Format('   Post-shock density         = ',1pe12.5,' gm/cm3')
  142 Format('   Post-shock velocity        = ',1pe12.5,' cm/s')
  143 Format('   Post-shock internal energy = ',1pe12.5,' erg/gm')
  144 Format('   Post-shock pressure        = ',1pe12.5,' dyne/cm2')
  145 Format('   Post-shock sound speed     = ',1pe12.5,' cm/s')
  146 Format('   Shock position             = ',1pe12.5,' cm')
  147 Format('   Pre-shock density          = ',1pe12.5,' gm/cm3')
  148 Format('   Vacuum boundary position   = ',1pe12.5,' cm')
  150 Format('     v        lambda       f          g          h')           
  151 Format(5(1x,f10.8))
  152 Format(5(1x,f10.8))
  160 Format('** exact value ',i6,' = ',1pe12.5,'  was not located **')
  161 Format('** Max # subdivs exceeded - increase limit **')
  162 Format('** Roundoff detected - error may be underestimated **')
  163 Format('** Extremely bad integrand behavior **')
  164 Format('** Algorithm does not converge **')
  165 Format('** Integral at best slowly convergent **')
  166 Format('** Invalid input **')
  170 Format(' Test planar Sedov values - see p. 222 of Sedov')
  171 Format(' Test cylindrical Sedov values - see p. 222 of Sedov')
  172 Format(' Test spherical Sedov values - see p. 223 of Sedov')
  180 Format(/'  sedov(',i2,') = ',1pe12.5)
  181 Format(' End of Sedov test values'/)

  900 Format('** sedov_kamm_1d:  Fatal Error **')
  901 Format('** Error in namelist input:  omega .ge. j+2 **')
  902 Format('** Error:  planar vacuum case not allowed **')
  903 Format('** Error:  planar singular case not allowed **')

!-----------------------------------------------------------------------
 
      Return
      End

!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
 
      Subroutine SEDOV_CHECK ( lpla, lcyl, lsph, llambda )
 
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!   This routine writes out values at the points specified by
!   Sedov in his book 
!
!   Called by: sedov_kamm_1d        Calls: SEDOV_ASSIGN, GET_FUN0, ZEROIN
!                                                                          
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Start of Subroutine SEDOV_CHECK
 
      Implicit None
 
!.... Include files
 
      Include "param.h"

!.... Call list variables

      Logical lpla                  ! .true. => compute planar      sedov sol'n
      Logical lcyl                  ! .true. => compute cylindrical sedov sol'n
      Logical lsph                  ! .true. => compute spherical   sedov sol'n
      Logical llambda               ! .true. => use Sedov lambda values

!.... Local Variables

      Integer i                     ! index
      Integer it                    ! iteration counter
      Integer iun                   ! unit number of output ascii file

      Integer nsedovp               ! # of planar points (Sedov, p. 232-233)
      Parameter ( nsedovp = 18 )    ! planar ( lpla = .true. )
      Integer nsedovc               ! # of cyl.   points (Sedov, p. 232-233)
      Parameter ( nsedovc = 24 )    ! cylindrical ( lcyl = .true. )
      Integer nsedovs               ! # of sph.   points (Sedov, p. 232-233)
      Parameter ( nsedovs = 16 )    ! spherical ( lsph = .true. )
      Integer nsedovmx              ! max # of Sedov points
      Parameter ( nsedovmx = 24 )   ! max # of Sedov points
      Integer nsedov                ! actual # of Sedov  points

      Real*8  sedov(1:nsedovmx)     ! Sedov values

      Real*8  atemp                 ! temporary scalar
      Real*8  btemp                 ! temporary scalar
      Real*8  sedval                ! Sedov value
      Real*8  l_fun                 ! distance similarity function
      Real*8  f_fun                 ! velocity similarity function
      Real*8  g_fun                 ! density  similarity function
      Real*8  h_fun                 ! pressure similarity function
      Real*8  v_c                   ! "center" value  for  ZEROIN call
      Real*8  v_l                   ! "left"   value  in   ZEROIN call
      Real*8  v_r                   ! "right"  value  in   ZEROIN call
      Real*8  v_n                   ! converged value from ZEROIN call

      Real*8   SEDFUN0              ! function: zero of Sedov function
                                    !   for lambda = omega = 0
      Real*8   ZEROIN               ! function: root finder

      External SEDFUN0
      External ZEROIN

      Common / sedcom / sedval

!-----------------------------------------------------------------------

      iun = 23
      If ( lpla ) Then
        nsedov = nsedovp
        !Open( iun, file='sedov.check.pla.dat', status='new')
        !Write(iun,970)
      Else If ( lcyl ) Then
        nsedov = nsedovc
        !Open( iun, file='sedov.check.cyl.dat', status='new')
        !Write(iun,971)
      Else If ( lsph ) Then
        nsedov = nsedovs
        !Open( iun, file='sedov.check.sph.dat', status='new')
        !Write(iun,972)
      End If ! lpla
      !Write(iun,*) '  '

!...  Assigns the values specified in Sedov's book

      Call SEDOV_ASSIGN ( lpla, lcyl, lsph, llambda, nsedovmx, sedov )

      If ( llambda ) Then                      ! lambda-values

        If ( lpla ) v_c = 0.55d0 ! planar value
        If ( lcyl ) v_c = 0.42d0 ! cylindrical value
        If ( lsph ) v_c = 0.33d0 ! spherical value

        Do i = 1, nsedov                       ! loop over Sedov values
          sedval = sedov(i)
          !Write(iun,980) i, sedval
          it  = 0
          v_l = v_c - 0.02d0
          v_r = v_c + 0.02d0
          !Write(iun,*) ' v_l = ',v_l,'  v_r = ',v_r
  100     Continue                             ! bracket the zero
          atemp = SEDFUN0( v_l )
          !Write(iun,*) ' atemp = ',atemp
          btemp = SEDFUN0( v_r ) 
          !Write(iun,*) ' btemp = ',btemp
          If ( atemp * btemp .lt. ZERO ) Go To 200
          it = it + 1
          If ( it .gt. 100 ) Stop '** Cannot bracket SEDOV 100 tries **'
          v_l = v_l + TENTH * ( v_c - v_l )
          v_r = v_r - TENTH * ( v_c - v_r )
          Go To 100
  200     Continue                             ! compute the zero
          v_n = ZEROIN( v_l, v_r, SEDFUN0, EPS16 )
          Call GET_FUN0( v_n, l_fun, f_fun, g_fun, h_fun )
          !Write(iun,950)
          !Write(iun,952) v_n, l_fun, f_fun, g_fun, h_fun
          v_c = v_n                            ! assign next starting value
        End Do ! i

      Else                                     ! V-values

        Do i = 1, nsedov                       ! loop over Sedov values
          sedval = sedov(i)
          !Write(iun,980) i, sedval
          Call GET_FUN0( sedval, l_fun, f_fun, g_fun, h_fun )
          !Write(iun,950)
          !Write(iun,952) sedval, l_fun, f_fun, g_fun, h_fun
        End Do ! i

      End If ! llambda
      !Write(iun,981)
      Close( iun )

!-----------------------------------------------------------------------
 
!.... Format statements

  950 Format('     v        lambda       f          g          h')           
  951 Format(5(1x,f10.8))
  952 Format(5(1x,f10.8))
  970 Format(' Test planar Sedov values - see p. 222 of Sedov')
  971 Format(' Test cylindrical Sedov values - see p. 222 of Sedov')
  972 Format(' Test spherical Sedov values - see p. 223 of Sedov')
  980 Format(/'  sedov(',i2,') = ',1pe12.5)
  981 Format(' End of Sedov test values'/)

!----------------------------------------------------------------------
 
      Return
      End

! End of Subroutine SEDOV_CHECK
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
 
      Subroutine SEDOV_ASSIGN ( lpla, lcyl, lsph, llambda, nsedovmx, 
     &                          sedov )
 
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!   This routine assigns the values specified in Sedov's book
!
!   Called by: SEDOV_CHECK    Calls: none
!                                                                          
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Start of Subroutine SEDOV_ASSIGN
 
      Implicit None
 
!.... Include files
 
      Include "param.h"

!.... Call list variables

      Logical lpla                  ! namelist .true. => planar  sedov sol'n
      Logical lcyl                  ! namelist .true. => cylin'l sedov sol'n
      Logical lsph                  ! namelist .true. => spher'l sedov sol'n
      Logical llambda               ! .true.  => use Sedov lambda values
                                    ! .false. => use inferred "V" values

      Integer nsedovmx              ! max # of Sedov points

      Real*8  sedov(1:nsedovmx)     ! Sedov values

!-----------------------------------------------------------------------

      If ( lpla ) Then
        If ( llambda ) Then
          sedov( 1) = 0.9797d0 
          sedov( 2) = 0.9420d0 
          sedov( 3) = 0.9013d0 
          sedov( 4) = 0.8565d0
          sedov( 5) = 0.8050d0 
          sedov( 6) = 0.7419d0 
          sedov( 7) = 0.7029d0 
          sedov( 8) = 0.6553d0
          sedov( 9) = 0.5925d0 
          sedov(10) = 0.5396d0 
          sedov(11) = 0.4912d0 
          sedov(12) = 0.4589d0
          sedov(13) = 0.4161d0 
          sedov(14) = 0.3480d0 
          sedov(15) = 0.2810d0 
          sedov(16) = 0.2320d0
          sedov(17) = 0.1680d0 
          sedov(18) = 0.1040d0
        Else
          sedov( 1) = 0.5500d0 
          sedov( 2) = 0.5400d0 
          sedov( 3) = 0.5300d0 
          sedov( 4) = 0.5200d0
          sedov( 5) = 0.5100d0 
          sedov( 6) = 0.5000d0 
          sedov( 7) = 0.4950d0 
          sedov( 8) = 0.4900d0
          sedov( 9) = 0.4850d0 
          sedov(10) = 0.4820d0 
          sedov(11) = 0.4800d0 
          sedov(12) = 0.4790d0
          sedov(13) = 0.4780d0 
          sedov(14) = 0.4770d0 
          sedov(15) = 0.47650d0 
          sedov(16) = 0.47632d0
          sedov(17) = 0.47622d0 
          sedov(18) = 0.47619d0
        End If ! llambda
      Else If ( lcyl ) Then
        If ( llambda ) Then
          sedov( 1) = 0.9998d0
          sedov( 2) = 0.9802d0
          sedov( 3) = 0.9644d0
          sedov( 4) = 0.9476d0
          sedov( 5) = 0.9295d0
          sedov( 6) = 0.9096d0
          sedov( 7) = 0.8725d0
          sedov( 8) = 0.8442d0
          sedov( 9) = 0.8094d0
          sedov(10) = 0.7629d0
          sedov(11) = 0.7242d0
          sedov(12) = 0.6894d0
          sedov(13) = 0.6390d0
          sedov(14) = 0.5745d0
          sedov(15) = 0.5180d0
          sedov(16) = 0.4748d0
          sedov(17) = 0.4222d0
          sedov(18) = 0.3654d0
          sedov(19) = 0.3000d0
          sedov(20) = 0.2500d0
          sedov(21) = 0.2000d0
          sedov(22) = 0.1500d0
          sedov(23) = 0.1000d0
          sedov(24) = 0.0500d0
        Else
          sedov( 1) = 0.4166d0
          sedov( 2) = 0.4100d0
          sedov( 3) = 0.4050d0
          sedov( 4) = 0.4000d0
          sedov( 5) = 0.3950d0
          sedov( 6) = 0.3900d0
          sedov( 7) = 0.3820d0
          sedov( 8) = 0.3770d0
          sedov( 9) = 0.3720d0
          sedov(10) = 0.3670d0
          sedov(11) = 0.3640d0
          sedov(12) = 0.3620d0
          sedov(13) = 0.3600d0
          sedov(14) = 0.3585d0
          sedov(15) = 0.3578d0
          sedov(16) = 0.3575d0
          sedov(17) = 0.3573d0
          sedov(18) = 0.3572d0
          sedov(19) = 0.35716d0
          sedov(20) = 0.35715d0
          sedov(21) = 0.357144d0
          sedov(22) = 0.357143d0
          sedov(23) = 0.3571429d0
          sedov(24) = 0.3571428d0
        End If ! llambda
      Else 
        If ( llambda ) Then
          sedov( 1) = 0.9913d0 
          sedov( 2) = 0.9773d0 
          sedov( 3) = 0.9622d0 
          sedov( 4) = 0.9342d0
          sedov( 5) = 0.9080d0 
          sedov( 6) = 0.8747d0 
          sedov( 7) = 0.8359d0 
          sedov( 8) = 0.7950d0
          sedov( 9) = 0.7493d0 
          sedov(10) = 0.6788d0 
          sedov(11) = 0.5794d0 
          sedov(12) = 0.4560d0
          sedov(13) = 0.3600d0 
          sedov(14) = 0.2960d0 
          sedov(15) = 0.2000d0 
          sedov(16) = 0.1040d0
        Else
          sedov( 1) = 0.3300d0
          sedov( 2) = 0.3250d0
          sedov( 3) = 0.3200d0
          sedov( 4) = 0.3120d0
          sedov( 5) = 0.3060d0
          sedov( 6) = 0.3000d0
          sedov( 7) = 0.2950d0
          sedov( 8) = 0.2915d0
          sedov( 9) = 0.2890d0
          sedov(10) = 0.2870d0
          sedov(11) = 0.2860d0
          sedov(12) = 0.2857d0
          sedov(13) = 0.28568d0
          sedov(14) = 0.28566d0
          sedov(15) = 0.28564d0
          sedov(16) = 0.28562d0
        End If ! llambda
      End If ! lpla

!-----------------------------------------------------------------------
 
      Return
      End

! End of Subroutine SEDOV_ASSIGN
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

      Subroutine GET_RVAL_2D ( xval, yval, xsq, ysq, icoord, 
     &                         lpla, lcyl, lsph, lxdir, lydir, lzdir, 
     &                         lcyl_rz, lcyl_rt, lsph_rt, lsph_rp, 
     &                         lsph_tp, zval, rval, thetaval, phival, 
     &                         rexact, ierr )
 
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!   This routine computes the value of the coordinate required for 
!   the Sedov solution (either "x" or "r") and returns this value as
!   the variable "rexact".  The assumption is made that this routine
!   is called only for the case of two-dimensional output values.
!                                                                          
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Start of Subroutine GET_RVAL_2D
 
      Implicit None
 
!.... Include files
 
      Include "param.h"

!.... Call list variables
      
      Integer icoord                ! coordinate system for output values
      Integer ierr                  ! error flag
 
      Real*8  xval                  ! input x-value
      Real*8  yval                  ! input y-value
      Real*8  xsq                   ! x-value**2
      Real*8  ysq                   ! y-value**2
      Real*8  zval                  ! fixed z-value
      Real*8  rval                  ! fixed r-value
      Real*8  thetaval              ! fixed theta-value
      Real*8  phival                ! fixed phi-value
      Real*8  rexact                ! output "x" or "r" value

      Logical lpla                  ! .true. => compute planar      sedov sol'n
      Logical lcyl                  ! .true. => compute cylindrical sedov sol'n
      Logical lsph                  ! .true. => compute spherical   sedov sol'n
      Logical lxdir                 ! .true. & lpla & idim.ge.2 => x-direction wave
      Logical lydir                 ! .true. & lpla & idim.ge.2 => y-direction wave
      Logical lzdir                 ! .true. & lpla & idim.ge.2 => z-direction wave
      Logical lcyl_rz               ! .true. & idim=2 & lcyl => r-z       coords
      Logical lcyl_rt               ! .true. & idim=2 & lcyl => r-theta   coords
      Logical lsph_rt               ! .true. & idim=2 & lsph => r-theta   coords
      Logical lsph_rp               ! .true. & idim=2 & lsph => r-phi     coords
      Logical lsph_tp               ! .true. & idim=2 & lsph => theta-phi coords

!-----------------------------------------------------------------------

      ierr = 0

!=======================================================================
!.... Planar Sedov problem - assign position value
!=======================================================================

      If ( lpla ) Then
        If ( icoord .eq. 1 ) Then       ! Cartesian output coord's
          If ( lxdir ) Then
            rexact = xval
          Else If ( lydir ) Then
            rexact = yval
          Else
            ierr = 1
          End If ! lxdir
        Else If ( icoord .eq. 2 ) Then  ! Cylindrical output coord's
          If ( lcyl_rt ) Then           ! r-theta coordinates
            If ( lxdir ) Then
              rexact = xval * Cos( yval )
            Else If ( lydir ) Then
              rexact = xval * Sin( yval )
            Else 
              ierr = 2
            End If ! lxdir
          Else If ( lcyl_rz ) Then      ! r-z coordinates
            If ( lxdir ) Then
              rexact = xval * Cos( thetaval )
            Else If ( lydir ) Then
              rexact = xval * Sin( thetaval )
            Else 
              ierr = 3
            End If ! lxdir
          End If ! lcyl_rt
          If ( ierr .ne. 0 ) Go To 799
        Else If ( icoord .eq. 3 ) Then  ! Spherical output coord's
          If ( lsph_rt ) Then           ! r-theta coordinates
            If ( lxdir ) Then
              rexact = xval * Cos( yval ) * Sin( phival )
            Else If ( lydir ) Then
              rexact = xval * Sin( yval ) * Sin( phival )
            Else If ( lzdir ) Then
              rexact = xval * Cos( phival )
            Else 
              ierr = 4
            End If ! lxdir
          Else If ( lsph_rp ) Then      ! r-phi coordinates
            If ( lxdir ) Then
              rexact = xval * Cos( thetaval ) * Sin( yval )
            Else If ( lydir ) Then
              rexact = xval * Sin( thetaval ) * Sin( yval )
            Else If ( lzdir ) Then
              rexact = xval * Cos( yval )
            Else 
              ierr = 5
            End If ! lxdir
          Else If ( lsph_tp ) Then      ! theta-phi coordinates
            If ( lxdir ) Then
              rexact = rval * Cos( xval ) * Sin( yval )
            Else If ( lydir ) Then
              rexact = rval * Sin( xval ) * Sin( yval )
            Else If ( lzdir ) Then
              rexact = rval * Cos( yval )
            Else 
              ierr = 6
            End If ! lxdir
          End If ! lsph_rt
          If ( ierr .ne. 0 ) Go To 799
        End If ! icoord

!=======================================================================
!.... Cylindrical Sedov problem - assign radial value
!=======================================================================

      Else If ( lcyl ) Then                    
        If ( icoord .eq. 1 ) Then       ! Cartesian output coord's
          rexact = Sqrt( xsq + ysq )
        Else If ( icoord .eq. 2 ) Then  ! Cylindrical output coord's
          rexact = xval
        Else If ( icoord .eq. 3 ) Then  ! Spherical output coord's
          If ( lsph_rt ) Then           ! r-theta coordinates
            rexact = xval * Sin( phival )
          Else If ( lsph_rp ) Then      ! r-phi coordinates
            rexact = xval * Sin( yval )
          Else If ( lsph_tp ) Then      ! theta-phi coordinates
            rexact = rval * Sin( yval )
          Else 
            ierr = 7
          End If ! lsph_rt
        End If ! icoord
        If ( ierr .ne. 0 ) Go To 799
      
!=======================================================================
!.... Spherical Sedov problem - assign radial value
!=======================================================================

      Else If ( lsph ) Then
        If ( icoord .eq. 1 ) Then       ! Cartesian output coord's
          rexact = Sqrt( xsq + ysq + zval**2 )
        Else If ( icoord .eq. 2 ) Then  ! Cylindrical output coord's
          If ( lcyl_rt ) Then           ! r-theta coordinates
            rexact = Sqrt( xsq + zval**2 )
          Else If ( lcyl_rz ) Then      ! r-z coordinates
            rexact = Sqrt( xsq + ysq )
          Else 
            ierr = 8
          End If ! lcyl_rt
        Else If ( icoord .eq. 3 ) Then  ! Spherical output coord's
          If ( .not. lsph_tp ) Then     ! r-theta or r-phi coordinates
            rexact = xval
          Else                          ! theta-phi coordinates
            rexact = rval
          End If ! lsph_rt
        End If ! icoord
        If ( ierr .ne. 0 ) Go To 799
      End If ! lpla

!-----------------------------------------------------------------------

  799 Continue
      If ( ierr .eq. 0 ) Go To 899
      Write(*,900)
      Go To ( 801, 802, 803, 804, 805, 806, 807, 808 ) ierr
  801 Write(*,901)
      Go To 899
  802 Write(*,902)
      Go To 899
  803 Write(*,903)
      Go To 899
  804 Write(*,904)
      Go To 899
  805 Write(*,905)
      Go To 899
  806 Write(*,906)
      Go To 899
  807 Write(*,907)
      Go To 899
  808 Write(*,908)
      Go To 899
  899 Continue

!-----------------------------------------------------------------------
 
!.... Format statements

  900 Format('** GET_RVAL_2D:  Fatal Error **')
  901 Format('** lpla & icoord = 1 & .not. lxdir & .not. lydir **')
  902 Format('** lpla & icoord = 2 & lcyl_rt & .not. lxdir & .not. '
     &'lydir **')
  903 Format('** lpla & icoord = 2 & lcyl_rz & .not. lxdir & .not. '
     &'lydir **')
  904 Format('** lpla & icoord = 3 & lsph_rt & .not. lxdir & .not. '
     &'lydir & .not. lzdir **')
  905 Format('** lpla & icoord = 3 & lsph_rp & .not. lxdir & .not. '
     &'lydir & .not. lzdir **')
  906 Format('** lpla & icoord = 3 & lsph_tp & .not. lxdir & .not. '
     &'lydir & .not. lzdir **')
  907 Format('** lcyl & icoord = 3 & .not. lsph_rt & .not. lsph_rp '
     &'& .not. lsph_tp **')
  908 Format('** lsph & icoord = 2 & .not. lcyl_rt & .not. lcyl_rz '
     &'**')
      
!-----------------------------------------------------------------------
 
      Return
      End

! End of Subroutine GET_RVAL_2D
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

      Subroutine GET_RVAL_3D ( xval, yval, zval, xsq, ysq, zsq, icoord, 
     &                         lpla, lcyl, lsph, lxdir, lydir, lzdir, 
     &                         rexact, ierr )
 
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!   This routine computes the value of the coordinate required for 
!   the Sedov solution (either "x" or "r") and returns this value as
!   the variable "rexact".  The assumption is made that this routine
!   is called only for the case of three-dimensional output values.
!                                                                          
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Start of Subroutine GET_RVAL_3D
 
      Implicit None
 
!.... Include files
 
      Include "param.h"

!.... Call list variables
      
      Integer icoord                ! coordinate system for output values
      Integer ierr                  ! error flag
 
      Real*8  xval                  ! input x-value
      Real*8  yval                  ! input y-value
      Real*8  zval                  ! input z-value
      Real*8  xsq                   ! x-value**2
      Real*8  ysq                   ! y-value**2
      Real*8  zsq                   ! z-value**2
      Real*8  rexact                ! output "x" or "r" value

      Logical lpla                  ! .true. => compute planar      sedov sol'n
      Logical lcyl                  ! .true. => compute cylindrical sedov sol'n
      Logical lsph                  ! .true. => compute spherical   sedov sol'n
      Logical lxdir                 ! .true. & lpla & idim.ge.2 => x-direction wave
      Logical lydir                 ! .true. & lpla & idim.ge.2 => y-direction wave
      Logical lzdir                 ! .true. & lpla & idim.ge.2 => z-direction wave

!-----------------------------------------------------------------------

      ierr = 0

!=======================================================================
!.... Planar Sedov problem - assign position value
!=======================================================================

      If ( lpla ) Then
        If ( icoord .eq. 1 ) Then       ! Cartesian output coord's
          If ( lxdir ) Then
            rexact = xval
          Else If ( lydir ) Then
            rexact = yval
          Else If ( lzdir ) Then
            rexact = zval
          Else 
            ierr = 1
          End If ! lxdir
        Else If ( icoord .eq. 2 ) Then  ! Cylindrical output coord's
          If ( lxdir ) Then
            rexact = xval * Cos( yval )
          Else If ( lydir ) Then
            rexact = xval * Sin( yval )
          Else If ( lzdir ) Then
            rexact = zval
          Else 
            ierr = 2
          End If ! lxdir
        Else If ( icoord .eq. 3 ) Then  ! Spherical output coord's
          If ( lxdir ) Then
            rexact = xval * Cos( yval ) * Sin( zval )
          Else If ( lydir ) Then
            rexact = xval * Sin( yval ) * Sin( zval )
          Else If ( lzdir ) Then
            rexact = xval * Cos( zval )
          Else 
            ierr = 3
          End If ! lxdir
        End If ! icoord
        If ( ierr .ne. 0 ) Go To 799

!=======================================================================
!.... Cylindrical Sedov problem - assign cylindrical-r value
!=======================================================================

      Else If ( lcyl ) Then                    
        If ( icoord .eq. 1 ) Then       ! Cartesian output coord's
          rexact = Sqrt( xsq + ysq )
        Else If ( icoord .eq. 2 ) Then  ! Cylindrical output coord's
          rexact = xval
        Else If ( icoord .eq. 3 ) Then  ! Spherical output coord's
          rexact = xval * Sin( zval )
        End If ! icoord
      
!=======================================================================
!.... Spherical Sedov problem - assign spherical-r value
!=======================================================================

      Else If ( lsph ) Then
        If ( icoord .eq. 1 ) Then       ! Cartesian output coord's
          rexact = Sqrt( xsq + ysq + zsq )
        Else If ( icoord .eq. 2 ) Then  ! Cylindrical output coord's
          rexact = Sqrt( xsq + zsq )
        Else If ( icoord .eq. 3 ) Then  ! Spherical output coord's
          rexact = xval
        End If ! icoord
      End If ! lpla

!-----------------------------------------------------------------------

  799 Continue
      If ( ierr .eq. 0 ) Go To 899
      Write(*,900)
      Go To ( 801, 802, 803 ) ierr
  801 Write(*,901)
      Go To 899
  802 Write(*,902)
      Go To 899
  803 Write(*,903)
      Go To 899
  899 Continue

!-----------------------------------------------------------------------
 
!.... Format statements

  900 Format('** GET_RVAL_3D:  Fatal Error **')
  901 Format('** lpla & icoord = 1 & .not. lxdir & .not. lydir & .not. '
     &'lzdir **')
  902 Format('** lpla & icoord = 2 & .not. lxdir & .not. lydir & .not. '
     &'lzdir **')
  903 Format('** lpla & icoord = 3 & .not. lxdir & .not. lydir & .not. '
     &'lzdir **')
      
!-----------------------------------------------------------------------
 
      Return
      End

! End of Subroutine GET_RVAL_3D
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

      Double Precision Function SEDFUN0( v )

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!   This function evaluates the function whose zero is sought
!   for an exact lambda-value in the constant pre-shock density case
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Start of Function SEDFUN0
 
      Implicit None
 
!.... Include files
 
      Include "param.h"
 
!.... Call list variables
 
      Real*8   v                     ! value of the similarity parameter
 
!.... Local variables
 
      Real*8   l_fun                 ! value of the similarity function lambda
      Real*8   f_fun                 ! value of the similarity function f
      Real*8   g_fun                 ! value of the similarity function g
      Real*8   h_fun                 ! value of the similarity function h

      Real*8   gamma                 ! common-block value of gamma
      Real*8   gamm1                 ! gamma minus one
      Real*8   gamp1                 ! gamma plus  one
      Real*8   gpogm                 ! ( gamma + 1 ) / ( gamma - 1 )
      Real*8   a0                    ! constant in function evaluation
      Real*8   a1                    ! constant in function evaluation
      Real*8   a2                    ! constant in function evaluation
      Real*8   a3                    ! constant in function evaluation
      Real*8   a4                    ! constant in function evaluation
      Real*8   a5                    ! constant in function evaluation
      Real*8   a_val                 ! scratch scalar
      Real*8   b_val                 ! scratch scalar
      Real*8   c_val                 ! scratch scalar
      Real*8   d_val                 ! scratch scalar
      Real*8   e_val                 ! scratch scalar
      Real*8   sedval                ! Sedov value

      Integer  j                     ! geometry flag: 2 => cyl, 3 => sph
      Integer  jp2                   ! j + 2

      Common / gascon / gamma, gamm1, gamp1, gpogm
      Common / simfuni / j, jp2
      Common / simfunr / a0, a1, a2, a3, a4, a5,
     &                   a_val, b_val, c_val, d_val, e_val
      Common / sedcom / sedval

!-----------------------------------------------------------------------
 
      Call GET_FUN0( v, l_fun, f_fun, g_fun, h_fun )
      SEDFUN0 = l_fun - sedval
 
!-----------------------------------------------------------------------
 
      Return
      End

! End of Function SEDFUN0
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

      Double Precision Function SEDFUNR0( v )

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!   This function evaluates the function whose zero is sought
!   for an exact r-value in the constant pre-shock density case
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Start of Function SEDFUNR0
 
      Implicit None
 
!.... Include files
 
      Include "param.h"
 
!.... Call list variables
 
      Real*8   v                     ! value of the similarity parameter
 
!.... Local variables
 
      Real*8   l_fun                 ! value of the similarity function lambda
      Real*8   f_fun                 ! value of the similarity function f
      Real*8   g_fun                 ! value of the similarity function g
      Real*8   h_fun                 ! value of the similarity function h

      Real*8   gamma                 ! common-block value of gamma
      Real*8   gamm1                 ! gamma minus one
      Real*8   gamp1                 ! gamma plus  one
      Real*8   gpogm                 ! ( gamma + 1 ) / ( gamma - 1 )
      Real*8   a0                    ! constant in function evaluation
      Real*8   a1                    ! constant in function evaluation
      Real*8   a2                    ! constant in function evaluation
      Real*8   a3                    ! constant in function evaluation
      Real*8   a4                    ! constant in function evaluation
      Real*8   a5                    ! constant in function evaluation
      Real*8   a_val                 ! scratch scalar
      Real*8   b_val                 ! scratch scalar
      Real*8   c_val                 ! scratch scalar
      Real*8   d_val                 ! scratch scalar
      Real*8   e_val                 ! scratch scalar
      Real*8   r2                    ! shock position value
      Real*8   rexval                ! r-exact value

      Integer  j                     ! geometry flag: 2 => cyl, 3 => sph
      Integer  jp2                   ! j + 2

      Common / gascon / gamma, gamm1, gamp1, gpogm
      Common / simfuni / j, jp2
      Common / simfunr / a0, a1, a2, a3, a4, a5,
     &                   a_val, b_val, c_val, d_val, e_val
      Common / rexcom / r2, rexval

!-----------------------------------------------------------------------
 
      Call GET_FUN0( v, l_fun, f_fun, g_fun, h_fun )
      SEDFUNR0 = r2 * l_fun - rexval
 
!-----------------------------------------------------------------------
 
      Return
      End

! End of Function SEDFUNR0
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

      Double Precision Function SEDFUNR1( v )

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!   This function evaluates the function whose zero is sought
!   for an exact r-value in the variable pre-shock density case
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Start of Function SEDFUNR1
 
      Implicit None
 
!.... Include files
 
      Include "param.h"
 
!.... Call list variables
 
      Real*8   v                     ! value of the similarity parameter
 
!.... Local variables
 
      Real*8   l_fun                 ! value of the similarity function lambda
      Real*8   f_fun                 ! value of the similarity function f
      Real*8   g_fun                 ! value of the similarity function g
      Real*8   h_fun                 ! value of the similarity function h

      Real*8   gamma                 ! common-block value of gamma
      Real*8   gamm1                 ! gamma minus one
      Real*8   gamp1                 ! gamma plus  one
      Real*8   gpogm                 ! ( gamma + 1 ) / ( gamma - 1 )
      Real*8   a0                    ! constant in function evaluation
      Real*8   a1                    ! constant in function evaluation
      Real*8   a2                    ! constant in function evaluation
      Real*8   a3                    ! constant in function evaluation
      Real*8   a4                    ! constant in function evaluation
      Real*8   a5                    ! constant in function evaluation
      Real*8   a_val                 ! scratch scalar
      Real*8   b_val                 ! scratch scalar
      Real*8   c_val                 ! scratch scalar
      Real*8   d_val                 ! scratch scalar
      Real*8   e_val                 ! scratch scalar
      Real*8   r2                    ! shock position value
      Real*8   rexval                ! r-exact value
      Real*8   omega                 ! common-block value of omega

      Integer  j                     ! geometry flag: 2 => cyl, 3 => sph
      Integer  jp2                   ! j + 2

      Common / gascon / gamma, gamm1, gamp1, gpogm
      Common / simfuni / j, jp2
      Common / simfunr / a0, a1, a2, a3, a4, a5,
     &                   a_val, b_val, c_val, d_val, e_val
      Common / rexcom / r2, rexval
      Common / omcom  / omega

!-----------------------------------------------------------------------
 
      Call GET_FUN1( v, l_fun, f_fun, g_fun, h_fun, omega )
      SEDFUNR1 = r2 * l_fun - rexval
 
!-----------------------------------------------------------------------
 
      Return
      End

! End of Function SEDFUNR1
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

      Subroutine GET_FUN0( v, l_fun, f_fun, g_fun, h_fun )

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!   This subroutine evaluates the functions lambda, f, g & h
!   in the constant pre-shock density case (see Sedov, p. 219)
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Start of Subroutine GET_FUN0
 
      Implicit None
 
!.... Include files
 
      Include "param.h"
  
!.... Call list variables
 
      Real*8   v                     ! value of the similarity parameter
      Real*8   l_fun                 ! value of the similarity function lambda
      Real*8   f_fun                 ! value of the similarity function f
      Real*8   g_fun                 ! value of the similarity function g
      Real*8   h_fun                 ! value of the similarity function h
 
!.... Local variables

      Real*8   gamma                 ! common-block value of gamma
      Real*8   gamm1                 ! gamma minus one
      Real*8   gamp1                 ! gamma plus  one
      Real*8   gpogm                 ! ( gamma + 1 ) / ( gamma - 1 )
      Real*8   a0                    ! constant in function evaluation
      Real*8   a1                    ! constant in function evaluation
      Real*8   a2                    ! constant in function evaluation
      Real*8   a3                    ! constant in function evaluation
      Real*8   a4                    ! constant in function evaluation
      Real*8   a5                    ! constant in function evaluation
      Real*8   a_val                 ! scratch scalar
      Real*8   b_val                 ! scratch scalar
      Real*8   c_val                 ! scratch scalar
      Real*8   d_val                 ! scratch scalar
      Real*8   e_val                 ! scratch scalar

      Integer  j                     ! geometry flag: 2 => cyl, 3 => sph
      Integer  jp2                   ! j + 2

      Common / gascon / gamma, gamm1, gamp1, gpogm
      Common / simfuni / j, jp2
      Common / simfunr / a0, a1, a2, a3, a4, a5,
     &                   a_val, b_val, c_val, d_val, e_val

!-----------------------------------------------------------------------
 
      l_fun = ( a_val * v )**( -a0 )
      l_fun = l_fun * ( b_val * Max( EPS16, c_val * v - ONE ) )**( -a2 )
      l_fun = l_fun * ( d_val * ( ONE - e_val * v ) )**( -a1 )

      f_fun = a_val * v * l_fun

      g_fun = ( b_val * Max( EPS16, c_val * v - ONE ) )**( a3 )
      g_fun = g_fun * ( b_val * ( ONE - HALF * jp2 * v ) )**( a5 )
      g_fun = g_fun * ( d_val * ( ONE - e_val * v ) )**( a4 )

      h_fun = ( a_val * v )**( a0 * j )
      h_fun = h_fun * ( b_val * ( ONE - HALF * jp2 * v ) )**( ONE + a5 )
      h_fun = h_fun * ( d_val * ( ONE - e_val * v ) )**( a4 - TWO * a1 )

!-----------------------------------------------------------------------
 
      Return
      End

! End of Subroutine GET_FUN0
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

      Subroutine GET_FUN1( v, l_fun, f_fun, g_fun, h_fun, omega )

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!   This subroutine evaluates the functions lambda, f, g & h
!   in the variable pre-shock density case (see Sedov, p. 265)
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Start of Subroutine GET_FUN1
 
      Implicit None
 
!.... Include files
 
      Include "param.h"
  
!.... Call list variables
 
      Real*8   v                     ! value of the similarity parameter
      Real*8   l_fun                 ! value of the similarity function lambda
      Real*8   f_fun                 ! value of the similarity function f
      Real*8   g_fun                 ! value of the similarity function g
      Real*8   h_fun                 ! value of the similarity function h
      Real*8   omega                 ! initial pre-blast density exponent
 
!.... Local variables

      Real*8   gamma                 ! common-block value of gamma
      Real*8   gamm1                 ! gamma minus one
      Real*8   gamp1                 ! gamma plus  one
      Real*8   gpogm                 ! ( gamma + 1 ) / ( gamma - 1 )
      Real*8   a0                    ! constant in function evaluation
      Real*8   a1                    ! constant in function evaluation
      Real*8   a2                    ! constant in function evaluation
      Real*8   a3                    ! constant in function evaluation
      Real*8   a4                    ! constant in function evaluation
      Real*8   a5                    ! constant in function evaluation
      Real*8   a_val                 ! scratch scalar
      Real*8   b_val                 ! scratch scalar
      Real*8   c_val                 ! scratch scalar
      Real*8   d_val                 ! scratch scalar
      Real*8   e_val                 ! scratch scalar

      Integer  j                     ! geometry flag: 2 => cyl, 3 => sph
      Integer  jp2                   ! j + 2

      Common / gascon / gamma, gamm1, gamp1, gpogm
      Common / simfuni / j, jp2
      Common / simfunr / a0, a1, a2, a3, a4, a5,
     &                   a_val, b_val, c_val, d_val, e_val

!-----------------------------------------------------------------------
 
      l_fun = ( a_val * v )**( -a0 )
      l_fun = l_fun * ( b_val * Max( EPS16, c_val * v - ONE ) )**( -a2 )
      l_fun = l_fun * ( d_val * ( ONE - e_val * v ) )**( -a1 )

      f_fun = a_val * v * l_fun

      g_fun = ( b_val * Max( EPS16, c_val * v - ONE ) 
     &        )**( a3 + omega * a2 )
      g_fun = g_fun * ( b_val * ( ONE - HALF * ( jp2 - omega ) * v )
     &                )**( a5 )
      g_fun = g_fun * ( d_val * ( ONE - e_val * v )
     &                )**( a4 + omega * a1 )
      g_fun = g_fun * ( a_val * v )**( omega * a0 )  

      h_fun = ( a_val * v )**( j * a0 )
      h_fun = h_fun * ( b_val * ( ONE - HALF * ( jp2 - omega ) * v ) 
     &                )**( ONE + a5 )
      h_fun = h_fun * ( d_val * ( ONE - e_val * v ) 
     &                )**( a4 + ( omega - TWO ) * a1 )

!-----------------------------------------------------------------------
 
      Return
      End

! End of Subroutine GET_FUN1
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

      Double Precision Function EFUN01( v )

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!   This function evaluates the integrand of the first energy integral,
!   for  omega .eq. 0
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Start of Function EFUN01

      Implicit None
 
!.... Include files
 
      Include "param.h"
 
!.... Call list variables
 
      Real*8   v                     ! value of the similarity parameter
 
!.... Local variables
 
      Real*8   gamma                 ! common-block value of gamma
      Real*8   gamm1                 ! gamma minus one
      Real*8   gamp1                 ! gamma plus  one
      Real*8   gpogm                 ! ( gamma + 1 ) / ( gamma - 1 )
      Real*8   a0                    ! constant in function evaluation
      Real*8   a1                    ! constant in function evaluation
      Real*8   a2                    ! constant in function evaluation
      Real*8   a3                    ! constant in function evaluation
      Real*8   a4                    ! constant in function evaluation
      Real*8   a5                    ! constant in function evaluation
      Real*8   a_val                 ! scratch scalar
      Real*8   b_val                 ! scratch scalar
      Real*8   c_val                 ! scratch scalar
      Real*8   d_val                 ! scratch scalar
      Real*8   e_val                 ! scratch scalar

      Integer  j                     ! geometry flag: 2 => cyl, 3 => sph
      Integer  jp2                   ! j + 2

      Common / gascon / gamma, gamm1, gamp1, gpogm
      Common / simfuni / j, jp2
      Common / simfunr / a0, a1, a2, a3, a4, a5,
     &                   a_val, b_val, c_val, d_val, e_val

!-----------------------------------------------------------------------
 
!      EFUN01 = -gpogm * v**2 *
!     & ( ( a0 / v ) + ( a2 * c_val / ( c_val * v - ONE ) )
!     &              - ( a1 * e_val / ( ONE - e_val * v ) ) ) *
!     & ( b_val * ( ONE - c_val * v / gamma ) )**( a5 ) *
!     & ( a_val * v )**( -jp2 * a0 ) *
!     & ( b_val * ( c_val * v - ONE ) )**( -jp2 * a2 + a3 ) *
!     & ( d_val * ( ONE - e_val * v ) )**( -jp2 * a1 + a4 )

      If ( a_val * v .le. ZERO ) Then
        Write(*,*) '** EFUN01:  a_val * v  = ',
     &             a_val * v,' .le. 0 **'
        Read(*,'()')
      Else If ( b_val * ( c_val * v - ONE ) * v .le. ZERO ) Then
        Write(*,*) '** EFUN01:  b_val * ( c_val * v - ONE ) = ',
     &             b_val * ( c_val * v - ONE ),' .le. 0 **'
        Read(*,'()')
      Else If ( d_val * ( ONE - e_val * v ) .le. ZERO ) Then
        Write(*,*) '** EFUN01:  d_val * ( ONE - e_val * v ) = ',
     &             d_val * ( ONE - e_val * v ),' .le. 0 **'
        Read(*,'()')
      Else If ( b_val * ( ONE - c_val * v / gamma ) .le. ZERO ) Then
        Write(*,*) '** EFUN01:  b_val * ( ONE - c_val * v / gamma ) = ',
     &             b_val * ( ONE - c_val * v / gamma ),' .le. 0 **'
        Read(*,'()')
      End If ! a_val * v

      EFUN01 = -gpogm * v**2 *
     & ( ( a0 / v ) + ( a2 * c_val / ( c_val * v - ONE ) )
     &              - ( a1 * e_val / ( ONE - e_val * v ) ) 
     & ) *
     & ( ( a_val * v )**a0 * 
     &   ( b_val * ( c_val * v - ONE ) )**a2 *
     &   ( d_val * ( ONE - e_val * v ) )**a1 
     & )**(-jp2) *
     & ( b_val * ( c_val * v - ONE ) )**a3 *
     & ( d_val * ( ONE - e_val * v ) )**a4 *
     & ( b_val * ( ONE - c_val * v / gamma ) )**a5

!-----------------------------------------------------------------------
 
      Return
      End

! End of Function EFUN01
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

      Double Precision Function EFUN02( v )

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!   This function evaluates the integrand of the second energy integral
!   for  omega .eq. 0
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Start of Function EFUN02
 
      Implicit None

!.... Include files
 
      Include "param.h"
 
      Real*8   v                     ! value of the similarity parameter
 
!.... Local variables
 
      Real*8   gamma                 ! common-block value of gamma
      Real*8   gamm1                 ! gamma minus one
      Real*8   gamp1                 ! gamma plus  one
      Real*8   gpogm                 ! ( gamma + 1 ) / ( gamma - 1 )
      Real*8   a0                    ! constant in function evaluation
      Real*8   a1                    ! constant in function evaluation
      Real*8   a2                    ! constant in function evaluation
      Real*8   a3                    ! constant in function evaluation
      Real*8   a4                    ! constant in function evaluation
      Real*8   a5                    ! constant in function evaluation
      Real*8   a_val                 ! scratch scalar
      Real*8   b_val                 ! scratch scalar
      Real*8   c_val                 ! scratch scalar
      Real*8   d_val                 ! scratch scalar
      Real*8   e_val                 ! scratch scalar

      Integer  j                     ! geometry flag: 2 => cyl, 3 => sph
      Integer  jp2                   ! j + 2

      Common / gascon / gamma, gamm1, gamp1, gpogm
      Common / simfuni / j, jp2
      Common / simfunr / a0, a1, a2, a3, a4, a5,
     &                   a_val, b_val, c_val, d_val, e_val

!-----------------------------------------------------------------------
 
!      EFUN02 = -HALF * gamp1 * v**2 / gamma *
!     & ( ( a0 / v ) + ( a2 * c_val / ( c_val * v - ONE ) )
!     &              - ( a1 * e_val / ( ONE - e_val * v ) ) ) *
!     & ( ( c_val * v - gamma ) / ( ONE - c_val * v ) ) *
!     & ( b_val * ( ONE - c_val * v / gamma ) )**( a5 ) *
!     & ( a_val * v )**( -jp2 * a0 ) *
!     & ( b_val * ( c_val * v - ONE ) )**( -jp2 * a2 + a3 ) *
!     & ( d_val * ( ONE - e_val * v ) )**( -jp2 * a1 + a4 )

      If ( a_val * v .le. ZERO ) Then
        Write(*,*) '** EFUN02:  a_val * v  = ',
     &             a_val * v,' .le. 0 **'
        Read(*,'()')
      Else If ( b_val * ( c_val * v - ONE ) * v .le. ZERO ) Then
        Write(*,*) '** EFUN02:  b_val * ( c_val * v - ONE ) = ',
     &             b_val * ( c_val * v - ONE ),' .le. 0 **'
        Read(*,'()')
      Else If ( d_val * ( ONE - e_val * v ) .le. ZERO ) Then
        Write(*,*) '** EFUN02:  d_val * ( ONE - e_val * v ) = ',
     &             d_val * ( ONE - e_val * v ),' .le. 0 **'
        Read(*,'()')
      Else If ( b_val * ( ONE - c_val * v / gamma ) .le. ZERO ) Then
        Write(*,*) '** EFUN02:  b_val * ( ONE - c_val * v / gamma ) = ',
     &             b_val * ( ONE - c_val * v / gamma ),' .le. 0 **'
        Read(*,'()')
      End If ! a_val * v

      EFUN02 = -HALF * gamp1 * v**2 / gamma *
     & ( ( a0 / v ) + ( a2 * c_val / ( c_val * v - ONE ) )
     &              - ( a1 * e_val / ( ONE - e_val * v ) )
     & ) *
     & ( ( c_val * v - gamma ) / ( ONE - c_val * v ) ) *
     & ( ( a_val * v )**a0 *
     &   ( b_val * ( c_val * v - ONE ) )**a2 *
     &   ( d_val * ( ONE - e_val * v ) )**a1
     & )**(-jp2) *
     & ( b_val * ( c_val * v - ONE ) )**a3 *
     & ( d_val * ( ONE - e_val * v ) )**a4 *
     & ( b_val * ( ONE - c_val * v / gamma ) )**a5

!      EFUN02 = -EIGHT / gamp1 / jp2**2 *
!     & ( ( a0 / v ) + ( a2 * c_val / ( c_val * v - ONE ) )
!     &              - ( a1 * e_val / ( ONE - e_val * v ) )
!     & ) *
!     & ( b_val * ( c_val * v - ONE ) )**(-j*a2) *
!     & ( d_val * ( ONE - e_val * v ) )**(a4-jp2*a1) *
!     & ( b_val * ( ONE - c_val * v / gamma ) )**(1+a5)

!-----------------------------------------------------------------------
 
      Return
      End

! End of Function EFUN02
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

      Double Precision Function EFUN11( v )

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!   This function evaluates the integrand of the first energy integral,
!   for  omega .ne. 0
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Start of Function EFUN11
 
      Implicit None
 
!.... Include files
 
      Include "param.h"
 
!.... Call list variables
 
      Real*8   v                     ! value of the similarity parameter
 
!.... Local variables
 
      Real*8   gamma                 ! common-block value of gamma
      Real*8   gamm1                 ! gamma minus one
      Real*8   gamp1                 ! gamma plus  one
      Real*8   gpogm                 ! ( gamma + 1 ) / ( gamma - 1 )
      Real*8   omega                 ! pre-blast density exponent
      Real*8   a0                    ! constant in function evaluation
      Real*8   a1                    ! constant in function evaluation
      Real*8   a2                    ! constant in function evaluation
      Real*8   a3                    ! constant in function evaluation
      Real*8   a4                    ! constant in function evaluation
      Real*8   a5                    ! constant in function evaluation
      Real*8   a_val                 ! scratch scalar
      Real*8   b_val                 ! scratch scalar
      Real*8   c_val                 ! scratch scalar
      Real*8   d_val                 ! scratch scalar
      Real*8   e_val                 ! scratch scalar

      Integer  j                     ! geometry flag: 2 => cyl, 3 => sph
      Integer  jp2                   ! j + 2

      Common / gascon / gamma, gamm1, gamp1, gpogm
      Common / simfuni / j, jp2
      Common / simfunr / a0, a1, a2, a3, a4, a5,
     &                   a_val, b_val, c_val, d_val, e_val
      Common / omcom  / omega

!-----------------------------------------------------------------------
 
!      EFUN11 = -gpogm * v**2 *
!     & ( ( a0 / v ) + ( a2 * c_val / ( c_val * v - ONE ) )
!     &              - ( a1 * e_val / ( ONE - e_val * v ) ) ) *
!     & ( b_val * ( ONE - c_val * v / gamma ) )**( a5 ) *
!     & ( a_val * v )**( -( jp2 - omega ) * a0 ) *
!     & ( b_val * ( c_val * v - ONE ) )**( -( jp2 - omega ) * a2 + a3 ) *
!     & ( d_val * ( ONE - e_val * v ) )**( -( jp2 - omega ) * a1 + a4 )

      If ( a_val * v .le. ZERO ) Then
        Write(*,*) '** EFUN11:  a_val * v  = ',
     &             a_val * v,' .le. 0 **'
        Read(*,'()')
      Else If ( b_val * ( c_val * v - ONE ) * v .le. ZERO ) Then
        Write(*,*) '** EFUN11:  b_val * ( c_val * v - ONE ) = ',
     &             b_val * ( c_val * v - ONE ),' .le. 0 **'
        Read(*,'()')
      Else If ( d_val * ( ONE - e_val * v ) .le. ZERO ) Then
        Write(*,*) '** EFUN11:  d_val * ( ONE - e_val * v ) = ',
     &             d_val * ( ONE - e_val * v ),' .le. 0 **'
        Read(*,'()')
      Else If ( b_val * ( ONE - c_val * v / gamma ) .le. ZERO ) Then
        Write(*,*) '** EFUN11:  b_val * ( ONE - c_val * v / gamma ) = ',
     &             b_val * ( ONE - c_val * v / gamma ),' .le. 0 **'
        Read(*,'()')
      End If ! a_val * v

      EFUN11 = -gpogm * v**2 *
     & ( ( a0 / v ) + ( a2 * c_val / ( c_val * v - ONE ) )
     &              - ( a1 * e_val / ( ONE - e_val * v ) )
     & ) *
     & ( ( a_val * v )**a0 *
     &   ( b_val * ( c_val * v - ONE ) )**a2 *
     &   ( d_val * ( ONE - e_val * v ) )**a1
     & )**(omega-jp2) *
     & ( b_val * ( c_val * v - ONE ) )**a3 *
     & ( d_val * ( ONE - e_val * v ) )**a4 *
     & ( b_val * ( ONE - c_val * v / gamma ) )**a5

!-----------------------------------------------------------------------
 
      Return
      End

! End of Function EFUN11
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

      Double Precision Function EFUN12( v )

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!   This function evaluates the integrand of the second energy integral
!   for  omega .ne. 0
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Start of Function EFUN12
 
      Implicit None
 
!.... Include files
 
      Include "param.h"
 
      Real*8   v                     ! value of the similarity parameter
 
!.... Local variables
 
      Real*8   gamma                 ! common-block value of gamma
      Real*8   gamm1                 ! gamma minus one
      Real*8   gamp1                 ! gamma plus  one
      Real*8   gpogm                 ! ( gamma + 1 ) / ( gamma - 1 )
      Real*8   omega                 ! pre-blast density exponent
      Real*8   a0                    ! constant in function evaluation
      Real*8   a1                    ! constant in function evaluation
      Real*8   a2                    ! constant in function evaluation
      Real*8   a3                    ! constant in function evaluation
      Real*8   a4                    ! constant in function evaluation
      Real*8   a5                    ! constant in function evaluation
      Real*8   a_val                 ! scratch scalar
      Real*8   b_val                 ! scratch scalar
      Real*8   c_val                 ! scratch scalar
      Real*8   d_val                 ! scratch scalar
      Real*8   e_val                 ! scratch scalar

      Integer  j                     ! geometry flag: 2 => cyl, 3 => sph
      Integer  jp2                   ! j + 2

      Common / gascon / gamma, gamm1, gamp1, gpogm
      Common / simfuni / j, jp2
      Common / simfunr / a0, a1, a2, a3, a4, a5,
     &                   a_val, b_val, c_val, d_val, e_val
      Common / omcom  / omega

!-----------------------------------------------------------------------
 
!      EFUN12 = -HALF * gamp1 * v**2 / gamma *
!     & ( ( a0 / v ) + ( a2 * c_val / ( c_val * v - ONE ) )
!     &              - ( a1 * e_val / ( ONE - e_val * v ) ) ) *
!     & ( ( c_val * v - gamma ) / ( ONE - c_val * v ) ) *
!     & ( b_val * ( ONE - c_val * v / gamma ) )**( a5 ) *
!     & ( a_val * v )**( -( jp2 - omega ) * a0 ) *
!     & ( b_val * ( c_val * v - ONE ) )**( -( jp2 - omega ) * a2 + a3 ) *
!     & ( d_val * ( ONE - e_val * v ) )**( -( jp2 - omega ) * a1 + a4 )

      If ( a_val * v .le. ZERO ) Then
        Write(*,*) '** EFUN12:  a_val * v  = ',
     &             a_val * v,' .le. 0 **'
        Read(*,'()')
      Else If ( b_val * ( c_val * v - ONE ) * v .le. ZERO ) Then
        Write(*,*) '** EFUN12:  b_val * ( c_val * v - ONE ) = ',
     &             b_val * ( c_val * v - ONE ),' .le. 0 **'
        Read(*,'()')
      Else If ( d_val * ( ONE - e_val * v ) .le. ZERO ) Then
        Write(*,*) '** EFUN12:  d_val * ( ONE - e_val * v ) = ',
     &             d_val * ( ONE - e_val * v ),' .le. 0 **'
        Read(*,'()')
      Else If ( b_val * ( ONE - c_val * v / gamma ) .le. ZERO ) Then
        Write(*,*) '** EFUN12:  b_val * ( ONE - c_val * v / gamma ) = ',
     &             b_val * ( ONE - c_val * v / gamma ),' .le. 0 **'
        Read(*,'()')
      End If ! a_val * v

      EFUN12 = -HALF * gamp1 * v**2 / gamma *
     & ( ( a0 / v ) + ( a2 * c_val / ( c_val * v - ONE ) )
     &              - ( a1 * e_val / ( ONE - e_val * v ) )
     & ) *
     & ( ( c_val * v - gamma ) / ( ONE - c_val * v ) ) *
     & ( ( a_val * v )**a0 *
     &   ( b_val * ( c_val * v - ONE ) )**a2 *
     &   ( d_val * ( ONE - e_val * v ) )**a1
     & )**(omega-jp2) *
     & ( b_val * ( c_val * v - ONE ) )**a3 *
     & ( d_val * ( ONE - e_val * v ) )**a4 *
     & ( b_val * ( ONE - c_val * v / gamma ) )**a5

!      EFUN12 = -EIGHT / gamp1 / ( jp2 - omega )**2 *
!     & ( ( a0 / v ) + ( a2 * c_val / ( c_val * v - ONE ) )
!     &              - ( a1 * e_val / ( ONE - e_val * v ) )
!     & ) *
!     & ( b_val * ( c_val * v - ONE ) )**(-j*a2) *
!     & ( d_val * ( ONE - e_val * v ) )**(a4+(omega-jp2)*a1) *
!     & ( b_val * ( ONE - c_val * v / gamma ) )**(1+a5)

!-----------------------------------------------------------------------
 
      Return
      End

! End of Function EFUN12
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
      double precision function ZEROIN(ax,bx,f,tol)
      double precision ax,bx,f,tol
c
c      a zero of the function  f(x)  is computed in the interval ax,bx .
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
c
      double precision  a,b,c,d,e,eps,fa,fb,fc,tol1,xm,p,q,r,s
      double precision  dabs,dsign
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
   30 if (dabs(fc) .ge. dabs(fb)) go to 40
      a = b
      b = c
      c = a
      fa = fb
      fb = fc
      fc = fa
c
c convergence test
c
   40 tol1 = 2.0d0*eps*dabs(b) + 0.5d0*tol
      xm = .5*(c - b)
      if (dabs(xm) .le. tol1) go to 90
      if (fb .eq. 0.0d0) go to 90
c
c is bisection necessary
c
      if (dabs(e) .lt. tol1) go to 70
      if (dabs(fa) .le. dabs(fb)) go to 70
c
c is quadratic interpolation possible
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
      p = dabs(p)
c
c is interpolation acceptable
c
      if ((2.0d0*p) .ge. (3.0d0*xm*q - dabs(tol1*q))) go to 70
      if (p .ge. dabs(0.5d0*e*q)) go to 70
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
      if (dabs(d) .gt. tol1) b = b + d
      if (dabs(d) .le. tol1) b = b + dsign(tol1, xm)
      fb = f(b)
      if ((fb*(fc/dabs(fc))) .gt. 0.0d0) go to 20
      go to 30
c
c done
c
   90 zeroin = b
      return
      end
!><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
