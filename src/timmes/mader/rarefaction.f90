      subroutine mader(t,x,u,p,c,rho,xdet,nstep,
     1                 p_cj,d_cj,gamma,u_piston)
      implicit none
      double precision t, p_cj, d_cj, gamma, u_piston
      double precision x(nstep), u(nstep), p(nstep), c(nstep)
      double precision rho(nstep), xdet(nstep)
      integer          nstep
      double precision xi, ui, pi, ci, rhoi, xdeti, dx
      integer          i
cf2py intent(hide)     :: nstep
cf2py intent(out)      :: u, p, c, rho, xdet
cf2py integer          :: nstep
cf2py double precision :: u(nstep), p(nstep), c(nstep)
cf2py double precision :: rho(nstep), xdet(nstep)
cf2py double precision :: x(nstep), t, p_cj, d_cj, gamma, u_piston 


      dx = (x(nstep) - x(1))/float(nstep)
      do i=1,nstep
         xi = x(i)
         call rare(t,xi,dx,p_cj,d_cj,gamma,u_piston,
     1                ui,pi,ci,rhoi,xdeti)
         u(i) = ui
         p(i) = pi
         c(i) = ci
         rho(i) = rhoi
         xdet(i) = xdeti
      enddo

      end subroutine mader


      subroutine rare(time,xlab,dx,p_cj,d_cj,gam,u_piston,
     1                u,p,c,rho,xdet)
      implicit none
      save

c..returns the rarefaction wave solution given on page 24
c..of fickett and davis

c..input:
c..time     =  time for desired solutioon (s)
c..xlab     = position in fixed lab frame, eularian frame (cm)
c..dx       = width of grid cell (cm)
c..p_cj     = chapman-jouget pressure (erg/cm**3)
c..d_cj     = chapman-jouget density (g/cm**3)
c..gam      = ratio of specific heats (dimensionless)
c..u_piston = speed of piston  (cm/s)

c..output:
c..u    = material speed (cm/s)
c..p    = pressure (erg/cm**3)
c..c    = sound speed (cm/s)
c..rho  = mass density (g/cm**3)
c..xdet = position relative to detonation front, lagrangian frame (cm)


c..declare the pass
      double precision time,xlab,dx,p_cj,d_cj,gam,u_piston,
     1                 u,p,c,rho,xdet

c..local variables
      double precision gamp1,rho_0,rho_cj,c_cj,u_cj,
     1                 gamm1,aa,bb,b,d,dd,ee,bp1,dp1,
     2                 um,xp,half,x,x1,h,x2,dxp,ur,pr,cr,rhor,dist,tol


c..some constants and factors
      gamp1  = gam + 1.0d0
      rho_0  = gamp1 * p_cj /d_cj**2
      rho_cj = rho_0 * gamp1/gam
      c_cj   = gam*d_cj/gamp1
      u_cj   = d_cj/gamp1

      gamm1  = gam - 1.0d0 
      aa     = 1.0d0/(2.0d0 * c_cj * time)
      bb     = (2.0d0 - gamm1 * u_cj /c_cj)/gamp1
      b      = 2.0d0 * gam/gamm1
      d      = 2.0d0/gamm1
      dd     = 2.0d0/(time*gamp1)
      ee     = gamm1 * (u_cj - 2.0d0*c_cj/gamm1)/gamp1
      bp1    = b + 1.0d0
      dp1    = d + 1.0d0

      um     = gamm1 * (u_cj - 2.0d0*c_cj/gamm1)/gamp1
      xp     = 0.5d0 * gamp1 * time * (u_piston - um)
      xdet   = d_cj*time - xlab
      dist   = abs(xdet - xp)
      tol    = 0.1d0 * dx


c..solution in the frame relative to detonation front, lagrangian frame
      x      = xdet
      half   = 0.5d0*dx
      x1     = x - half

c..solution in the rarefaction fan
      if (dist.gt.tol  .and. xdet.gt.xp) then
       u   = dd*(x1+half) + ee
       p   = p_cj*((aa*(x1+dx)+bb)**bp1 - (aa*x1 + bb)**bp1)/(dx*aa*bp1)
       c   = c_cj*(aa*(x1+half) + bb)
       rho = rho_cj*((aa*(x1+dx)+bb)**dp1 - (aa*x1+bb)**dp1)/(dx*aa*dp1)


c..solution if right at the transition point
      else if (dist .le. tol)  then

c.. partial q's
       x2  = x1+dx
       dxp = (x2-xp)
       h   = dxp/2
       u   = dd*(x1+h) + ee
       p   = p_cj*((aa*(x1+dxp)+bb)**bp1 - (aa*x1+bb)**bp1)/(dxp*aa*bp1)
       c   = c_cj*(aa*(x1+h) + bb)
       rho = rho_cj*((aa*(x1+dxp)+bb)**dp1-(aa*x1+bb)**dp1)/(dxp*aa*dp1)

c..residual q's
       ur   = u_piston
       pr   = p_cj*(1+gamm1*(u-u_cj)/(2.0d0*c_cj))**(2.0d0*gam/gamm1)
       cr   = c_cj*(1+gamm1*(u-u_cj)/(2.0d0*c_cj))
       rhor = rho_cj*(p/p_cj)**(1.0d0/gam)

c..avg q's
       u   = ur + (u-ur)*2.0d0*h/dx
       p   = pr +(p-pr)*2.0d0*h/dx
       c   = cr + (c-cr)*2.0d0*h/dx
       rho = rho + (rho - rhor)*2.0d0*h/dx


c..solution in the constant state
      else
       u   = u_piston
       p   = p_cj*(1+gamm1*(u-u_cj)/(2.0d0*c_cj))**(2.0d0*gam/gamm1)
       c   = c_cj*(1+gamm1*(u-u_cj)/(2.0d0*c_cj))
       rho = rho_cj*(p/p_cj)**(1.0d0/gam)
      endif
      return
      end
