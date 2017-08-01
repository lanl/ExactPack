!========================================================================
! Developed by A Dawes from AWE (UK) on secondment at LANL (USA)
! Source code is unrestricted.
!========================================================================
   module kinds
      integer, parameter :: ik=4
      integer, parameter :: rk=8
    end module kinds
!========================================================================

    subroutine planar_sand(time, ny, nsteps, l, T1, T2, y, temp)
      implicit none
      integer :: ny, nsteps
      double precision :: time, l, T1, T2 ! T1=bottom, T2=top
      double precision :: y(ny), temp(ny)
!f2py intent(out)      :: temp
!f2py intent(hide)     :: ny
!f2py integer          :: ny
!f2py double           :: time, l, T1, T2
!f2py double           :: y(ny), temp(ny)

      call planar_sandwich_analytic_solution(time, ny, nsteps, l, T1, T2, y, temp)

    end subroutine planar_sand

 subroutine planar_sandwich_analytic_solution(tim, nc, nsteps, l, T0, Tl, yvec, temp)

 use kinds
 implicit none
 integer   (kind=ik)              :: i
 integer   (kind=ik)              :: r
 integer   (kind=ik)              :: nc ! number of cells along y
 integer   (kind=ik)              :: nv
 integer   (kind=ik)              :: nsteps
 real      (kind=rk)              :: T0
 real      (kind=rk)              :: Tl
 real      (kind=rk)              :: pi
 real      (kind=rk)              :: tim
 real      (kind=rk)              :: y
 real      (kind=rk)              :: l
 real      (kind=rk)              :: ar
 real      (kind=rk)              :: br
 real      (kind=rk)              :: temp(nc) ! temperature array
 real      (kind=rk)              :: yvec(nc) ! y-array
 
 pi = 4.0_rk*atan(1.0_rk)
 nv = nc + 1_ik

 do i = 1, nc
  y = yvec(i)
  temp(i) = 0.0_rk
  do r = 1, nsteps
   ar      = (2.0_rk/(real(r,rk)*pi))*((-1.0_rk)**(real(r,rk)) * Tl - T0) * sin (pi*y*real(r,rk)/l)
   br      = exp(-tim*pi*pi*real(r,rk)**2 / (l*l))
   temp(i) = temp(i) + ar*br
  enddo
  temp(i) = temp(i) + T0 + y*(Tl - T0)/l
 enddo

 end subroutine planar_sandwich_analytic_solution
