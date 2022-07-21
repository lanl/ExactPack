c23456789012345678901234567890123456789012345678901234567890123456789012
c
c /Applications/Absoft/bin/f90 interp_laz.f
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c                                                                      c
c   This function calls the FMM routine "spline" to interpolate values c
c   of "lambda" and "bhat" ( = ((gamma-1)/(gamma+1))*B ) for the       c
c   converging shock problem in cylindrical or spherical geometry      c
c   from the tabular values in Tables 6.4 (cyl) and 6.5 (sph) of:      c
c                                                                      c
c   Lazarus, R.B, "Self-Similar Solutions for converging Shocks and    c
c   Collapsing Cavities," SIAM J. Numer Anal. 18, pp. 316-371 (1981)   c
c                                                                      c
c   Note:  per the subsequent erratum,                                 c
c                                                                      c
c   Lazarus, R.B, "Erratum: Self-Similar Solutions for Converging      c
c   Shocks and Collapsing Cavities," SIAM J. Numer Anal. 19, p. 1090   c
c   (1982)                                                             c
c                                                                      c
c   The values in the third column of these tables, listed as "B" in   c
c   the first reference above, are actually:                           c
c                                                                      c
c            "bhat" = ( (gamma-1) / (gamma+1) ) * B.                   c
c                                                                      c
c   Calls: DO_IT        Called by: none                                c
c                                                                      c
c   2007.07.03  Kamm    initial development -- seems to work!          c
c   2007.07.05  Kamm    clean up the code, which appears to work fine  c
c   2007.07.20  Ramsey  code converted to a function			     c
c                                                                      c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
      real*8 function INTERP_LAZ(n, gamma, lambda)
c
      Implicit None
	real*8 bhat, gamma, lambda
	integer n
	logical lcyl, lsph
c
c-----------------------------------------------------------------------
c
c.... Call the driver routine
c
      if (n .eq. 2) then
		lcyl = .true.
		lsph = .false.
	else if (n .eq. 3) then
		lcyl = .false.
		lsph = .true.
	else
		go to 3
	endif
c
c
	Call DO_IT(lcyl, lsph, gamma, bhat)
c
c.... Strictly using "bhat" as the upper bound for the more precise 
c	value of bhat will in all likeihood produce a "endpoints do not
c	have opposite signs" error in the "zeroin" routine used in the
c	"guderley_1D driving program. Therefore, the interpolated value
c	of bhat is adjusted slightly to counter this error. The setting 
c	at which it is currently placed may need to be adjusted for 
c 	various cases of gamma and n.
c
	INTERP_LAZ = bhat
c
c-----------------------------------------------------------------------
c
3     Return
      End
c
c End of Function INTERP_LAZ
c><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
c
      Subroutine DO_IT(lcyl, lsph, gamma, bhat) 
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c                                                                      c
c   Driver routine for the interpolation of values from Lazarus's      c
c   table of Guderley parameters                                       c
c                                                                      c
c   Called by: INTERP      Calls: GET_PARAMS, DO_INTERP                c
c                                                                      c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Start of Subroutine DO_IT
c
      Implicit None
c
c.... Include files
c
      Include "param.h"
 
c
c.... Local variables
c
      Integer ierr        
! error flag
c
      Real*8  gamma       
! polytropic gas constant
      Real*8  lambda      
! interpolated Guderley "lambda"-value
      Real*8  bhat        
! interpolated Guderley "bhat"-value
c
      Logical lcyl        
! .true. => cylindrical geometry
      Logical lsph        
! .true. => spherical   geometry
c
c-----------------------------------------------------------------------
c
      ierr = 0
c
c.... Do the interpolation to the value "gamma" read from the input file
c
	Call DO_INTERP( lcyl, lsph, gamma, lambda, bhat, ierr )
      If ( ierr .ne. 0 ) ierr = 2
      If ( ierr .ne. 0 ) Go To 799
c
c.... Write the results to standard output
c
c      If ( lcyl ) Then
c        Write(*,101) gamma, bhat
c      Else If ( lsph ) Then
c        Write(*,102) gamma, bhat
c      Else
c        ierr = 3
c      End If ! lcyl
c      If ( ierr .ne. 0 ) Go To 799
c
c-----------------------------------------------------------------------
c
  799 Continue
      If ( ierr .eq. 0 ) Go To 889
c
c.... Error conditions
c
      Write(*,900)
      Go To ( 801, 802, 803 ) ierr
  801 Continue
      Write(*,901)
      Go To 889
  802 Continue
      Write(*,902)
      Go To 889
  803 Continue
      Write(*,903)
      Go To 889
  889 Continue
c
c-----------------------------------------------------------------------
c
c.... Format statements
c
  101 Format(' Cylindrical: gamma = ',1pe19.12,'  bhat = ',1pe19.12)
  102 Format(' Spherical: gamma = ',1pe19.12,'  bhat = ',1pe19.12)
  900 Format('** DO_IT: FATAL ERROR **')
  901 Format('** Error return from GET_PARAM **')
  902 Format('** Error return from DO_INTERP **')
  903 Format('** Error with problem specification: lcyl=lsph=.false **')
c
c-----------------------------------------------------------------------
c
      Return
      End
c
c End of Subroutine DO_IT
c><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
c
      Subroutine DO_INTERP( lcyl, lsph, gamma, lambda, bhat, ierr )
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c                                                                      c
c This routine:                                                        c
c (i)  assigns the geometry-dependent Lazarus/Guderley variables       c
c (ii) interpolates values of lambda and bval from those values,       c
c      based on the input value of gamma                               c
c                                                                      c
c   Called by: DO_IT      Calls: GET_CYL, GET_SPH, SPLINE, SEVAL       c
c                                                                      c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Start of Subroutine DO_INTERP
c
      Implicit None
c
c.... Include files
c
      Include "param.h"
c
c.... Call list variables
c
      Integer ierr           
! error flag
c
      Real*8  gamma          
! polytropic gas constant
      Real*8  lambda         
! interpolated Guderley "lambda"-value
      Real*8  bhat           
! interpolated Guderley "b"-value
c
      Logical lcyl           
! .true. => cylindrical geometry
      Logical lsph           
! .true. => spherical   geometry
c
c.... Local variables
c
      Integer i              
! index
      Integer nmax           
! maximum number of Lazarus-tabular values
      Parameter ( nmax = 52 )
      Integer nval           
! actual  number of Lazarus-tabular values
c
      Real*8  gamval(1:nmax) 
! gamma values
      Real*8  lamval(1:nmax) 
! lambda values
      Real*8  bval(1:nmax)   
! bhat values, with bhat = ((g-1)/(g+1))*B
      Real*8  barray(1:nmax) 
! spline-fit coefficients
      Real*8  carray(1:nmax) 
! spline-fit coefficients
      Real*8  darray(1:nmax) 
! spline-fit coefficients
      Real*8  SEVAL          
! spline evaluation function
      External SEVAL
c
c----------------------------------------------------------------------
c
      ierr = 0
c
c.... Zero-out the parameter arrays
c
      Do i = 1, nmax
        gamval(i) = ZERO
        lamval(i) = ZERO
        bval(i)   = ZERO
      End Do ! i
c
c.... Assign values from the Lazarus tables to arrays according to 
c.... the user-specified geometry
c
      If ( lcyl ) Then
        Call GET_CYL( nmax, nval, gamval, lamval, bval, ierr )
        If ( ierr .ne. 0 ) ierr = 1
      Else If ( lsph ) Then
        Call GET_SPH( nmax, nval, gamval, lamval, bval, ierr )
        If ( ierr .ne. 0 ) ierr = 2
      End If ! lcyl
      If ( ierr .ne. 0 ) Go To 799
c----------------------------------------------------------------------
c.... Do the interpolation for bhat-values
c----------------------------------------------------------------------
c.... First compute the spline coefficients
c
      Do i = 1, nmax
        barray(i) = ZERO
        carray(i) = ZERO
        darray(i) = ZERO
      End Do ! i
      Call SPLINE( nval, gamval, bval, barray, carray, darray, ierr )
      If ( ierr .ne. 0 ) ierr = 4
      If ( ierr .ne. 0 ) Go To 799
c
c.... Now interpolate a bhat-value for the specified gamma
c
      bhat = SEVAL( nval, gamma, gamval, bval, barray, carray, 
     &              darray )
c
c-----------------------------------------------------------------------
c
  799 Continue
      If ( ierr .eq. 0 ) Go To 899
c
c.... Error conditions
c
      Write(*,900)
      Go To ( 801, 802, 803, 804 ) ierr
  801 Continue
      Write(*,901) 
      Go To 899
  802 Continue
      Write(*,902) 
      Go To 899
  803 Continue
      Write(*,903) 
      Go To 899
  804 Continue
      Write(*,904) 
      Go To 899
  899 Continue
c
c-----------------------------------------------------------------------
c
c.... Format statements
c
  900 Format('** DO_INTERP: FATAL ERROR **')
  901 Format('** Error return from GET_CYL **')
  902 Format('** Error return from GET_SPH **')
  903 Format('** Error return from SPLINE for lambda-values **')
  904 Format('** Error return from SPLINE for bhat-values **')
c
c-----------------------------------------------------------------------
c
      Return
      End
c
c End of Subroutine DO_INTERP
c><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
c
      Subroutine GET_CYL( nmax, nval, gamval, lamval, bval, ierr )
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c                                                                      c
c   This routine reads in the values from Lazarus Table 6.4 for the    c
c   Guderley lambda and bhat = ((gamma-1)/(gamma+1))*B in the          c
c   cylindrical geometry case.                                         c
c                                                                      c
c    Called by: DO_INTERP        Calls: none                           c
c                                                                      c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Start of Subroutine GET_CYL
c
      Implicit None
c
c.... Include files
c
      Include "param.h"
c
c.... Call list variables
c
      Integer ierr           
! error flag
      Integer nmax           
! maximum number of Lazarus values
      Integer nval           
! actual  number of Lazarus values
c
      Real*8  gamval(1:nmax) 
! gamma values
      Real*8  lamval(1:nmax) 
! lambda values
      Real*8  bval(1:nmax)   
! bhat values, with bhat = ((g-1)/(g+1))*B
c
c.... Local variables
c
      Integer i              
! index
c
c----------------------------------------------------------------------
c
      ierr = 0
c
c.... Zero out the arrays
c
      Do i = 1, nmax
        gamval(i) = ZERO
        lamval(i) = ZERO
        bval(i)   = ZERO
      End Do ! i
c
c.... Check the array size
c
      nval = 46
      If ( nval .gt. nmax ) ierr = 1
      If ( ierr .ne. 0 ) Go To 799
c
c.... Assign values from Table 6.4 to arrays according to geometry
c
      gamval( 1) = 1.00001d0
      gamval( 2) = 1.0001d0 
      gamval( 3) = 1.001d0  
      gamval( 4) = 1.005d0  
      gamval( 5) = 1.01d0   
      gamval( 6) = 1.03d0   
      gamval( 7) = 1.05d0   
      gamval( 8) = 1.07d0   
      gamval( 9) = 1.10d0   
      gamval(10) = 1.15d0   
      gamval(11) = 1.2d0    
      gamval(12) = 1.3d0    
      gamval(13) = 1.4d0    
      gamval(14) = 1.5d0    
      gamval(15) = 1.66667d0
      gamval(16) = 1.7d0    
      gamval(17) = 1.8d0    
      gamval(18) = 1.9d0    
      gamval(19) = 1.92d0   
      gamval(20) = 2.0d0    
      gamval(21) = 2.0863d0 
      gamval(22) = 2.0883d0 
      gamval(23) = 2.125d0  
      gamval(24) = 2.2d0    
      gamval(25) = 2.3676d0 
      gamval(26) = 2.3678d0 
      gamval(27) = 2.4d0    
      gamval(28) = 2.6d0    
      gamval(29) = 2.8d0    
      gamval(30) = 2.83920d0
      gamval(31) = 2.83929d0
      gamval(32) = 3.0d0    
      gamval(33) = 3.4d0    
      gamval(34) = 4.0d0    
      gamval(35) = 5.0d0    
      gamval(36) = 6.0d0    
      gamval(37) = 7.0d0    
      gamval(38) = 8.0d0    
      gamval(39) = 10.0d0   
      gamval(40) = 15.0d0   
      gamval(41) = 20.0d0   
      gamval(42) = 30.0d0   
      gamval(43) = 50.0d0   
      gamval(44) = 100.0d0  
      gamval(45) = 1000.0d0 
      gamval(46) = 9999.0d0 
c
      lamval( 1) = 1.0022073240d0
      lamval( 2) = 1.0068195769d0
      lamval( 3) = 1.0202846866d0
      lamval( 4) = 1.0414733956d0
      lamval( 5) = 1.0553973808d0
      lamval( 6) = 1.0850737604d0
      lamval( 7) = 1.1023892512d0
      lamval( 8) = 1.1150692073d0
      lamval( 9) = 1.1296268597d0
      lamval(10) = 1.1475773258d0
      lamval(11) = 1.1612203175d0
      lamval(12) = 1.1817213587d0
      lamval(13) = 1.1971414294d0
      lamval(14) = 1.2095591324d0
      lamval(15) = 1.2260537880d0
      lamval(16) = 1.2288931032d0
      lamval(17) = 1.2367055181d0
      lamval(18) = 1.2436278359d0
      lamval(19) = 1.2449208188d0
      lamval(20) = 1.2498244759d0
      lamval(21) = 1.2546830116d0
      lamval(22) = 1.2547907910d0
      lamval(23) = 1.2567323668d0
      lamval(24) = 1.2604989804d0
      lamval(25) = 1.2680643171d0
      lamval(26) = 1.2680727188d0
      lamval(27) = 1.2694076380d0
      lamval(28) = 1.2769816100d0
      lamval(29) = 1.2835139723d0
      lamval(30) = 1.2846912316d0
      lamval(31) = 1.2846938989d0
      lamval(32) = 1.2892136582d0
      lamval(33) = 1.2986950941d0
      lamval(34) = 1.3095267323d0
      lamval(35) = 1.3220499813d0
      lamval(36) = 1.3305627751d0
      lamval(37) = 1.3367301837d0
      lamval(38) = 1.3414054776d0
      lamval(39) = 1.3480251307d0
      lamval(40) = 1.3569909807d0
      lamval(41) = 1.3615356210d0
      lamval(42) = 1.3661223915d0
      lamval(43) = 1.3698225859d0
      lamval(44) = 1.3726158889d0
      lamval(45) = 1.3751432790d0
      lamval(46) = 1.3753967176d0
c
      bval  ( 1) = 0.521740d0
      bval  ( 2) = 0.554609d0
      bval  ( 3) = 0.625514d0
      bval  ( 4) = 0.697737d0
      bval  ( 5) = 0.724429d0
      bval  ( 6) = 0.731819d0
      bval  ( 7) = 0.708880d0
      bval  ( 8) = 0.682234d0
      bval  ( 9) = 0.644590d0
      bval  (10) = 0.593262d0
      bval  (11) = 0.554542d0
      bval  (12) = 0.502117d0
      bval  (13) = 0.469268d0
      bval  (14) = 0.447230d0
      bval  (15) = 0.423698d0
      bval  (16) = 0.420261d0
      bval  (17) = 0.411663d0
      bval  (18) = 0.405047d0
      bval  (19) = 0.403911d0
      bval  (20) = 0.399877d0
      bval  (21) = 0.396295d0
      bval  (22) = 0.396220d0
      bval  (23) = 0.394904d0
      bval  (24) = 0.392529d0
      bval  (25) = 0.388444d0
      bval  (26) = 0.388440d0
      bval  (27) = 0.387812d0
      bval  (28) = 0.384755d0
      bval  (29) = 0.382794d0
      bval  (30) = 0.382506d0
      bval  (31) = 0.382505d0
      bval  (32) = 0.381580d0
      bval  (33) = 0.380564d0
      bval  (34) = 0.380920d0
      bval  (35) = 0.383355d0
      bval  (36) = 0.386279d0
      bval  (37) = 0.389064d0
      bval  (38) = 0.391561d0
      bval  (39) = 0.395687d0
      bval  (40) = 0.402440d0
      bval  (41) = 0.406405d0
      bval  (42) = 0.410797d0
      bval  (43) = 0.414640d0
      bval  (44) = 0.417726d0
      bval  (45) = 0.420658d0
      bval  (46) = 0.420960d0
c
c-----------------------------------------------------------------------
c
  799 Continue
      If ( ierr .eq. 0 ) Go To 899
c
c.... Error conditions
c
      Write(*,900)
      Go To ( 801 ) ierr
  801 Continue
      Write(*,901) nval, nmax
      Go To 899
  899 Continue
c
c-----------------------------------------------------------------------
c
c.... Format statements
c
  900 Format('** GET_CYL: FATAL ERROR **')
  901 Format('** nval = ',i2,' .gt. ',i2,' = nmax **')
c
c-----------------------------------------------------------------------
c
      Return
      End
c
c End of Subroutine GET_CYL
c><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
c
      Subroutine GET_SPH( nmax, nval, gamval, lamval, bval, ierr )
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c                                                                      c
c   This routine reads in the values from Lazarus Table 6.5 for the    c
c   Guderley lambda and bhat = ((gamma-1)/(gamma+1))*B in the          c
c   spherical geometry case.                                           c
c                                                                      c
c    Called by: DO_INTERP        Calls: none                           c
c                                                                      c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Start of Subroutine GET_SPH
c
      Implicit None
c
c.... Include files
c
      Include "param.h"
c
c.... Call list variables
c
      Integer ierr           
! error flag
      Integer nmax           
! maximum number of Lazarus values
      Integer nval           
! actual  number of Lazarus values
c
      Real*8  gamval(1:nmax) 
! gamma values
      Real*8  lamval(1:nmax) 
! lambda values
      Real*8  bval(1:nmax)   
! bhat values, with bhat = ((g-1)/(g+1))*B
c
c.... Local variables
c
      Integer i              
! index
c
c----------------------------------------------------------------------
c
      ierr = 0
c
c.... Zero out the arrays
c
      Do i = 1, nmax
        gamval(i) = ZERO
        lamval(i) = ZERO
        bval(i)   = ZERO
      End Do ! i
c
c.... Check the array size
c
      nval = 52
      If ( nval .gt. nmax ) ierr = 1
      If ( ierr .ne. 0 ) Go To 799
c
c.... Assign values from Table 6.5 to arrays according to geometry
c
      gamval( 1) = 1.00001d0
      gamval( 2) = 1.0001d0 
      gamval( 3) = 1.001d0  
      gamval( 4) = 1.01d0   
      gamval( 5) = 1.03d0   
      gamval( 6) = 1.05d0   
      gamval( 7) = 1.07d0   
      gamval( 8) = 1.10d0   
      gamval( 9) = 1.15d0   
      gamval(10) = 1.2d0    
      gamval(11) = 1.3d0    
      gamval(12) = 1.4d0    
      gamval(13) = 1.5d0    
      gamval(14) = 1.6d0    
      gamval(15) = 1.66667d0
      gamval(16) = 1.7d0    
      gamval(17) = 1.8d0    
      gamval(18) = 1.86d0   
      gamval(19) = 1.88d0   
      gamval(20) = 1.9d0    
      gamval(21) = 2.0d0    
      gamval(22) = 2.010d0  
      gamval(23) = 2.012d0  
      gamval(24) = 2.2d0    
      gamval(25) = 2.2215d0 
      gamval(26) = 2.2217d0 
      gamval(27) = 2.4d0    
      gamval(28) = 2.5518d0 
      gamval(29) = 2.55194d0
      gamval(30) = 2.6d0    
      gamval(31) = 2.8d0    
      gamval(32) = 3.0d0    
      gamval(33) = 3.2d0    
      gamval(34) = 3.4d0    
      gamval(35) = 3.6d0    
      gamval(36) = 3.8d0    
      gamval(37) = 4.0d0    
      gamval(38) = 4.5d0    
      gamval(39) = 5.0d0    
      gamval(40) = 5.5d0    
      gamval(41) = 6.0d0    
      gamval(42) = 6.5d0    
      gamval(43) = 7.0d0    
      gamval(44) = 8.0d0    
      gamval(45) = 10.0d0   
      gamval(46) = 15.0d0   
      gamval(47) = 20.0d0   
      gamval(48) = 30.0d0   
      gamval(49) = 50.0d0   
      gamval(50) = 100.0d0  
      gamval(51) = 1000.0d0 
      gamval(52) = 9999.0d0 
c
      lamval( 1) = 1.0044047883d0
      lamval( 2) = 1.0135647885d0
      lamval( 3) = 1.0401005736d0
      lamval( 4) = 1.1088100742d0
      lamval( 5) = 1.1671691602d0
      lamval( 6) = 1.2015664277d0
      lamval( 7) = 1.2269581432d0
      lamval( 8) = 1.2563291060d0
      lamval( 9) = 1.2928404943d0
      lamval(10) = 1.3207565353d0
      lamval(11) = 1.3628123548d0
      lamval(12) = 1.3943607838d0
      lamval(13) = 1.4195913539d0
      lamval(14) = 1.4405288149d0
      lamval(15) = 1.4526927211d0
      lamval(16) = 1.4583285785d0
      lamval(17) = 1.4737227445d0
      lamval(18) = 1.4820184714d0
      lamval(19) = 1.4846461951d0
      lamval(20) = 1.4872097129d0
      lamval(21) = 1.4991468274d0
      lamval(22) = 1.5002661592d0
      lamval(23) = 1.5004885113d0
      lamval(24) = 1.5193750470d0
      lamval(25) = 1.5213088378d0
      lamval(26) = 1.5213266323d0
      lamval(27) = 1.5358986669d0
      lamval(28) = 1.5465622206d0
      lamval(29) = 1.5465714207d0
      lamval(30) = 1.5496663736d0
      lamval(31) = 1.5613198923d0
      lamval(32) = 1.5713126233d0
      lamval(33) = 1.5799755842d0
      lamval(34) = 1.5875567751d0
      lamval(35) = 1.5942459679d0
      lamval(36) = 1.6001909794d0
      lamval(37) = 1.6055087137d0
      lamval(38) = 1.6166309698d0
      lamval(39) = 1.6254243269d0
      lamval(40) = 1.6325476141d0
      lamval(41) = 1.6384333257d0
      lamval(42) = 1.6433769444d0
      lamval(43) = 1.6475870992d0
      lamval(44) = 1.6543738548d0
      lamval(45) = 1.6637583967d0
      lamval(46) = 1.6760512867d0
      lamval(47) = 1.6821004429d0
      lamval(48) = 1.6880830534d0
      lamval(49) = 1.6928204564d0
      lamval(50) = 1.6963447551d0
      lamval(51) = 1.6994953607d0
      lamval(52) = 1.6998093041d0
c
      bval  ( 1) = 0.541777d0
      bval  ( 2) = 0.607335d0
      bval  ( 3) = 0.758422d0
      bval  ( 4) = 0.988008d0
      bval  ( 5) = 0.996617d0
      bval  ( 6) = 0.931071d0
      bval  ( 7) = 0.860781d0
      bval  ( 8) = 0.769242d0
      bval  ( 9) = 0.658324d0
      bval  (10) = 0.584657d0
      bval  (11) = 0.496984d0
      bval  (12) = 0.448082d0
      bval  (13) = 0.417547d0
      bval  (14) = 0.397073d0
      bval  (15) = 0.386974d0
      bval  (16) = 0.382711d0
      bval  (17) = 0.372341d0
      bval  (18) = 0.367499d0
      bval  (19) = 0.366070d0
      bval  (20) = 0.364725d0
      bval  (21) = 0.359085d0
      bval  (22) = 0.358608d0
      bval  (23) = 0.358514d0
      bval  (24) = 0.351834d0
      bval  (25) = 0.351293d0
      bval  (26) = 0.351288d0
      bval  (27) = 0.348072d0
      bval  (28) = 0.346707d0
      bval  (29) = 0.346707d0
      bval  (30) = 0.346472d0
      bval  (31) = 0.346267d0
      bval  (32) = 0.346985d0
      bval  (33) = 0.348323d0
      bval  (34) = 0.350078d0
      bval  (35) = 0.352112d0
      bval  (36) = 0.354327d0
      bval  (37) = 0.356656d0
      bval  (38) = 0.362682d0
      bval  (39) = 0.368678d0
      bval  (40) = 0.374437d0
      bval  (41) = 0.379873d0
      bval  (42) = 0.384959d0
      bval  (43) = 0.389698d0
      bval  (44) = 0.398201d0
      bval  (45) = 0.411949d0
      bval  (46) = 0.434177d0
      bval  (47) = 0.447247d0
      bval  (48) = 0.461834d0
      bval  (49) = 0.474726d0
      bval  (50) = 0.485184d0
      bval  (51) = 0.495226d0
      bval  (52) = 0.496265d0
c
c-----------------------------------------------------------------------
c
  799 Continue
      If ( ierr .eq. 0 ) Go To 899
c
c.... Error conditions
c
      Write(*,900)
      Go To ( 801 ) ierr
  801 Continue
      Write(*,901) nval, nmax
      Go To 899
  899 Continue
c
c-----------------------------------------------------------------------
c
c.... Format statements
c
  900 Format('** GET_SPH: FATAL ERROR **')
  901 Format('** nval = ',i2,' .gt. ',i2,' = nmax **')
c
c-----------------------------------------------------------------------
c
      Return
      End
c
c End of Subroutine GET_SPH
c><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
c
      Subroutine SPLINE ( n, x, y, b, c, d, ierr )
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c                                                                      c
c   This routine implements the Forsythe, Malcolm, and Moler version   c
c   of spline interpolation.                                           c
c                                                                      c
c   Ref:  G.E. Forsythe, M.A. Malcolm, and C.B. Moler,                 c
c         "Computer Methods for Mathematical Computations"             c
c                                                                      c
c  The coefficients b(i), c(i), and d(i), i=1,2,...,n are computed     c
c  for a cubic interpolating spline                                    c
c                                                                      c
c    s(x) = y(i) + b(i)*(x-x(i)) + c(i)*(x-x(i))**2 + d(i)*(x-x(i))**3 c
c                                                                      c
c  for  x(i) .le. x .le. x(i+1)                                        c
c                                                                      c
c  Input:                                                              c
c                                                                      c
c    n = the number of data points or knots ( n .ge. 2 )               c
c    x = the abscissas of the knots in strictly increasing order       c
c    y = the ordinates of the knots                                    c
c                                                                      c
c  Output:                                                             c
c                                                                      c
c    b, c, d  = arrays of spline coefficients as defined above.        c
c                                                                      c
c  Using  p  to denote differentiation, the arrays are as follows:     c
c                                                                      c
c    y(i) = s(x(i))                                                    c
c    b(i) = sp(x(i))                                                   c
c    c(i) = spp(x(i)) / 2                                              c
c    d(i) = sppp(x(i)) / 6  (derivative from the right)                c
c                                                                      c
c  The accompanying function subprogram  SEVAL  can be used            c
c  to evaluate the spline.                                             c
c                                                                      c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Start of Subroutine SPLINE
c
      Implicit None
c
c.... Include files
c
      Include "param.h"
c
c.... Call list variables
c
      Integer ierr      
! error flag
      Integer n         
! size of arrays
c
      Real*8  x(1:n)    
! abscissae
      Real*8  y(1:n)    
! ordinates
      Real*8  b(1:n)    
! 1st spline coeff't
      Real*8  c(1:n)    
! 2nd spline coeff't
      Real*8  d(1:n)    
! 3rd spline coeff't
c
c.... Local variables
c
      Integer i         
! index
      Integer ib        
! index
      Integer nm1       
! = n - 1
c
      Real*8  t         
! ratio of derivatives
c
c----------------------------------------------------------------------
c
      ierr = 0
      nm1  = n - 1
      If ( n .lt. 2 ) ierr = 1
      If ( ierr .ne. 0 ) Go To 799
      If ( n .lt. 3 ) Go To 50
c
c  Set up tridiagonal system
c
c  b = diagonal, d = offdiagonal, c = right hand side.
c
      d(1) = x(2) - x(1)
      c(2) = (y(2) - y(1)) / d(1)
      Do 10 i = 2, nm1
         d(i)   = x(i+1) - x(i)
         b(i)   = TWO * ( d(i-1) + d(i) )
         c(i+1) = ( y(i+1) - y(i) ) / d(i)
         c(i)   = c(i+1) - c(i)
   10 Continue
c
c  End conditions.  Third derivatives at  x(1)  and  x(n)
c  obtained from divided differences.
c
      b(1) = -d(1)
      b(n) = -d(n-1)
      c(1) = ZERO
      c(n) = ZERO
      If ( n .eq. 3 ) Go To 15
      c(1) = c(3) / ( x(4) - x(2) ) - c(2) / ( x(3) - x(1) )
      c(n) = c(n-1) / ( x(n) - x(n-2) ) - c(n-2) / ( x(n-1) - x(n-3) )
      c(1) = c(1) * d(1)**2 / ( x(4) - x(1) )
      c(n) = -c(n) * d(n-1)**2  / ( x(n) - x(n-3) )
c
c  Forward elimination
c
   15 Do 20 i = 2, n
         t = d(i-1) / b(i-1)
         b(i) = b(i) - t * d(i-1)
         c(i) = c(i) - t * c(i-1)
   20 Continue
c
c  Back substitution
c
      c(n) = c(n) / b(n)
      Do 30 ib = 1, nm1
         i = n - ib
         c(i) = ( c(i) - d(i) * c(i+1) ) / b(i)
   30 Continue
c
c  c(i) is now the sigma(i) of the textbook
c
c  Compute polynomial coefficients
c
      b(n) = ( y(n) - y(nm1) ) / d(nm1) +
     &       d(nm1) * ( c(nm1) + TWO * c(n) )
      Do 40 i = 1, nm1
         b(i) = ( y(i+1) - y(i) ) / d(i) -
     &          d(i) * ( c(i+1) + TWO * c(i) )
         d(i) = ( c(i+1) - c(i) ) / d(i)
         c(i) = THREE * c(i)
   40 Continue
      c(n) = THREE * c(n)
      d(n) = d(n-1)
      Go To 899
c
   50 b(1) = ( y(2) - y(1) ) / ( x(2) - x(1) )
      c(1) = ZERO
      d(1) = ZERO
      b(2) = b(1)
      c(2) = ZERO
      d(2) = ZERO
c
c-----------------------------------------------------------------------
c
  799 Continue
      If ( ierr .eq. 0 ) Go To 899
c
c.... Error conditions
c
      Write(*,900)
      Go To ( 801 ) ierr
  801 Continue
      Write(*,901) n
      Go To 899
  899 Continue
c
c-----------------------------------------------------------------------
c
c.... Format statements
c
  900 Format('** SPLINE: FATAL ERROR **')
  901 Format('** Number of points = ',i4,' .lt. 2 **')
c
c-----------------------------------------------------------------------
c
      Return
      End
c
c End of Subroutine SPLINE
c><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
c
      Real*8 Function SEVAL( n, u, x, y, b, c, d )
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c                                                                      c
c   This routine implements the Forsythe, Malcolm, and Moler version   c
c   of spline interpolation.                                           c
c                                                                      c
c   Ref:  G.E. Forsythe, M.A. Malcolm, and C.B. Moler,                 c
c         "Computer Methods for Mathematical Computations"             c
c                                                                      c
c  This subroutine evaluates the cubic spline function                 c
c                                                                      c
c    seval = y(i) + b(i)*(u-x(i)) + c(i)*(u-x(i))**2 + d(i)*(u-x(i))**3c
c                                                                      c
c  where  x(i) .lt. u .lt. x(i+1), using Horner's rule                 c
c                                                                      c
c  if  u .lt. x(1) then  i = 1  is used.                               c
c  if  u .ge. x(n) then  i = n  is used.                               c
c                                                                      c
c  Input:                                                              c
c                                                                      c
c    n = the number of data points                                     c
c    u = the abscissa at which the spline is to be evaluated           c
c    x,y = the arrays of data abscissas and ordinates                  c
c    b,c,d = arrays of spline coefficients computed by spline          c
c                                                                      c
c  If  u  is not in the same interval as the previous call, then a     c
c  binary search is performed to determine the proper interval.        c
c                                                                      c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Start of Subroutine SPLINE
c
      Implicit None
c
c.... Call list variables
c
      Integer n         
! size of arrays
c
      Real*8  u         
! abscissa at which spline is to be evaluated
      Real*8  x(1:n)    
! abscissae
      Real*8  y(1:n)    
! ordinates
      Real*8  b(1:n)    
! 1st spline coeff't
      Real*8  c(1:n)    
! 2nd spline coeff't
      Real*8  d(1:n)    
! 3rd spline coeff't
c
c.... Local variables
c
      Integer i         
! index
      Integer j         
! index
      Integer k         
! index
c
      Real*8  dx        
! abscissa increment
c
      Data i / 1 /
c
c----------------------------------------------------------------------
c
      If ( i .ge. n ) i = 1
      If ( u .lt. x(i) )   Go To 10
      If ( u .le. x(i+1) ) Go To 30
c
c  Binary search
c
   10 i = 1
      j = n + 1
   20 k = ( i + j ) / 2
      If ( u .lt. x(k) ) j = k
      If ( u .ge. x(k) ) i = k
      If ( j .gt. i+1 ) Go To 20
c
c  Evaluate spline
c
   30 dx = u - x(i)
      SEVAL = y(i) + dx * ( b(i) + dx * ( c(i) + dx * d(i) ) )
c
c-----------------------------------------------------------------------
c
      Return
      End
c
c End of Function SEVAL
c><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>







