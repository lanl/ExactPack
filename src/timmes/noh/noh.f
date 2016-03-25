      subroutine noh_1d(time,xpos,nstep,
     1                 rho1,u1,gamma,xgeom,
     2                 den,ener,pres,vel,jumps)
      implicit none
      save

cf2py intent(out) :: den, ener, pres, vel, jumps
cf2py intent(hide) :: nstep
cf2py double precision :: time, rho1, u1, gamma, xgeom
cf2py double precision :: xpos(nstep)
cf2py double precision :: den(nstep), ener(nstep), pres(nstep), vel(nstep) 
cf2py double precision :: jumps(9)

c..solves the standard case, (as opposed to the singular or vacuum case), 
c..constant density (omega = 0) sedov problem in one-dimension.


c..input: 
c..time     = temporal point where solution is desired seconds
c..xpos     = spatial point where solution is desired cm


c..output:
c..den  = density  g/cm**3
c..ener = specific internal energy erg/g
c..pres = presssure erg/cm**3
c..vel  = velocity cm/sh


c..declare the pass
      integer nstep
      double precision time,xpos(*),
     1                 rho1,u1,gamma,xgeom,
     3                 den(*),ener(*),pres(*),vel(*),jumps(9)

c..local variables
      double precision gamm1,gamp1,gpogm,xgm1,us,r2,rhop,rho2,u2,e2,p2
      integer i

c..some parameters
      gamm1 = gamma - 1.0d0
      gamp1 = gamma + 1.0d0
      gpogm = gamp1 / gamm1
      xgm1  = xgeom - 1.0d0


c..immediate post-chock values using strong shock relations
c..shock velocity, position, pre- and post-shock density,
c..flow velocity, internal energy, and pressure

      us   = 0.5d0 * gamm1 * abs(u1)                  
      r2   = us * time                                
      rhop = rho1 * (1.0d0 - (u1*time/r2))**xgm1
      rho2 = rho1 * gpogm**xgeom      
      u2   = 0.0d0
      e2   = 0.5d0 * u1**2
      p2   = gamm1 * rho2 * e2

      jumps(1) = r2
      jumps(2) = rho2
      jumps(3) = rhop
      jumps(4) = e2
      jumps(5) = 0.0d0
      jumps(6) = p2
      jumps(7) = 0.0d0
      jumps(8) = u2
      jumps(9) = u1

      do i=1, nstep

c..if we are farther out than the shock front
        if (xpos(i) .gt. r2) then
          den(i)  = rho1 * (1.0d0 - (u1*time/xpos(i)))**xgm1
          vel(i)  = u1
          ener(i) = 0.0d0
          pres(i) = 0.0d0

c..if we are between the origin and the shock front
        else  
          den(i)  = rho2
          vel(i)  = u2
          ener(i) = e2
          pres(i) = p2
        end if

      enddo

      return
      end
