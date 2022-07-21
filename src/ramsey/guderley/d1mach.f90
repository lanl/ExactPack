!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                   !
!     This function contains double-precision machine               !
!     constants.                                                    !
!                                                                   !
!     d1mach( 1) = b**(emin-1), the smallest positive magnitude.    !
!     d1mach( 2) = b**emax*(1 - b**(-t)), the largest magnitude.    !
!     d1mach( 3) = b**(-t), the smallest relative spacing.          !
!     d1mach( 4) = b**(1-t), the largest relative spacing.          !
!     d1mach( 5) = log10(b)                                         !
!                                                                   !
!     Inputs:                                                       !
!                                                                   !
!          i     Case variable corresponding to machine             !
!                parameter desired.                                 !
!                                                                   !
!     Output:                                                       !
!                                                                   !
!          d1mach(i)   Machine parameter                            !
!                                                                   !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Start of Function "d1mach"
!
      real*8 function d1mach(i)
!      
      implicit none
      real*8 dummy, epsilon, huge, log10, tiny
      integer i, radix
!
      select case(i)
      case (1)
         d1mach = tiny(dummy)
      case (2)
         d1mach = huge(dummy)
      case (3)
         d1mach = epsilon(dummy)/radix(dummy)
      case (4)
         d1mach = epsilon(dummy)
      case (5)
         d1mach = log10(dble(radix(dummy)))
      end select
!
      return
      end
!
! End of Function "d1mach"
!<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
