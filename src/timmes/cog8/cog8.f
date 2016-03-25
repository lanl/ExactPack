      subroutine cog8_timmes(t,r,den,tev,ener,pres,vel,nstep,
     1     rho0,temp0,alpha,beta,gamma,cv)
      implicit none
      integer          nstep
      double precision r(nstep),den(nstep),tev(nstep)
      double precision ener(nstep),pres(nstep),vel(nstep)
      double precision t
cf2py intent(hide)     :: nstep
cf2py intent(out)      :: den, tev, ener, pres, vel
cf2py integer          :: nstep
cf2py double precision :: den(nstep),tev(nstep), ener(nstep)
cf2py double precision :: pres(nstep),vel(nstep)
cf2py double precision :: r(nstep), t
      integer          :: i
      double precision :: deni,tevi,eneri,presi,veli,zstep
      double precision :: xl,xr,rho0,temp0,alpha,beta,gamma,cv

      zstep = (r(nstep) - r(1))/float(nstep)
      do i=1,nstep
         xl = r(i) - 0.5d0 * zstep
         xr = r(i) + 0.5d0 * zstep
         call cog8_1d_sph(t,xl,xr,
     1        rho0,temp0,alpha,beta,gamma,cv,
     2        deni,tevi,eneri,presi,veli)
         den(i) = deni
         tev(i) = tevi
         ener(i) = eneri
         pres(i) = presi
         vel(i) = veli
      enddo
      end subroutine cog8_timmes


      subroutine cog8_1d_sph(time,xl,xr,
     1                  rho0,temp0,alpha,beta,gamma,cv,
     2                  den,tev,ener,pres,vel)
      implicit none
      save


c..solves the coggeshall problem #8 in one-dimension, spherical coordinates
c..s.v. coggeshall, phys fluids a 3, 757, 191

c..input: 
c..time   = time in shakes units of 1e-8 s,
c..xl     = left boundary of cell in cm
c..xr     = right boundary of cell in cm
c..rho0   = density constant g/cm**3
c..temp0  = temperature constant in ev
c..alpha  = dimensionless constant
c..beta   = dimensionless consnat
c..gamma  = perfect gas 
c..cv     = specific heat @ constant volume, erg/g/eV

c..output:
c..den  = density  g/cm**3
c..tev  = temperature ev
c..ener = specific internal energy erg/g
c..pres = presssure erg/cm**3
c..vel  = velocity cm/sh


c..declare the pass
      double precision  t, time,xl,xr,
     1                  rho0,temp0,alpha,beta,gamma,cv,
     2                  den,tev,ener,pres,vel

c..local variables
      integer           ik,ikp1
      parameter         (ik = 2,  ikp1 = ik + 1)
      double precision  seventh,rexp1,texp1,
     1                  dv,dm,dmv,de,aied,pi,volfac,velfac
      parameter         (pi     = 3.1415926535897932384d0,
     1                   volfac = 4.0d0 * pi,
     2                   velfac = 1.0d8)

c..the various exponents
      seventh = ( ik - 1 ) / ( beta - alpha + 4.0d0)
      rexp1   =  seventh
      texp1   = ( ik + 1 ) + seventh


c..this is the cell averaged solution, slightly different than
c..the exact point solution

      dv  = volfac * ( xr**ikp1 - xl**ikp1 ) / ikp1 
      dm  = volfac * rho0 * ( xr**(ik+1+rexp1) - xl**(ik+1+rexp1) ) 
     &                      / (texp1 * time**texp1)
      dmv = volfac * rho0 * ( xr**(ik+2+rexp1) - xl**(ik+2+rexp1) ) 
     &                      / ((texp1 + 1) * time**(texp1 + 1))
      de  = volfac * rho0 * ( xr**(ik+3+rexp1) - xl**(ik+3+rexp1) )
     &                      / ((texp1 + 2) * time**(texp1 + 2)) * 0.50d0


c..and the quantities of interest 

      den   = dm  / dv                          
      vel   = dmv / dm  * velfac 
      vel   = vel * 1.e-8 ! convert from cm/sh to cm/s
      aied  = rho0 * temp0 * cv / time**5      
      ener  = aied / den
      tev   = ener / cv             
      pres  = (gamma - 1.0d0) * aied

      return
      end






      double precision function value(string)
      implicit none
      save


c..this routine takes a character string and converts it to a real number. 
c..on error during the conversion, a fortran stop is issued

c..declare
      logical          pflag
      character*(*)    string
      character*1      plus,minus,decmal,blank,se,sd,se1,sd1
      integer          noblnk,long,ipoint,power,psign,iten,j,z,i
      double precision x,sign,factor,rten,temp
      parameter        (plus = '+'  , minus = '-' , decmal = '.'   ,
     1                  blank = ' ' , se = 'e'    , sd = 'd'       ,
     2                  se1 = 'E'   , sd1 = 'D'   , rten =  10.0,
     3                  iten = 10                                   )

c..initialize
      x      =  0.0d0
      sign   =  1.0d0
      factor =  rten
      pflag  =  .false.
      noblnk =  0
      power  =  0
      psign  =  1
      long   =  len(string)


c..remove any leading blanks and get the sign of the number
      do z = 1,7
       noblnk = noblnk + 1
       if ( string(noblnk:noblnk) .eq. blank) then
        if (noblnk .gt. 6 ) goto  30
       else
        if (string(noblnk:noblnk) .eq. plus) then
         noblnk = noblnk + 1
        else if (string(noblnk:noblnk) .eq. minus) then
         noblnk = noblnk + 1
         sign =  -1.0d0
        end if
        goto 10
       end if
      enddo


c..main number conversion loop
 10   continue
      do i = noblnk,long
       ipoint = i + 1


c..if a blank character then we are done
       if ( string(i:i) .eq. blank ) then
        x     = x * sign
        value = x 
        return


c..if an exponent character, process the whole exponent, and return
       else if (string(i:i).eq.se  .or. string(i:i).eq.sd .or.
     1          string(i:i).eq.se1 .or. string(i:i).eq.sd1   ) then
        if (x .eq. 0.0 .and. ipoint.eq.2)     x = 1.0d0
        if (sign .eq. -1.0 .and. ipoint.eq.3) x = 1.0d0
        if (string(ipoint:ipoint) .eq. plus) ipoint = ipoint + 1
        if (string(ipoint:ipoint) .eq. minus) then
         ipoint = ipoint + 1
         psign = -1
        end if
        do z = ipoint,long
         if (string(z:z) .eq. blank)  then
          x = sign * x * rten**(power*psign)
          value = x
          return
         else
          j = ichar(string(z:z)) - 48
          if ( (j.lt.0) .or. (j.gt.9) ) goto 30
          power= (power * iten)  + j
         end if
        enddo


c..if an ascii number character, process ie
       else if (string(i:i) .ne. decmal) then
        j = ichar(string(i:i)) - 48
        if ( (j.lt.0) .or. (j.gt.9) ) goto 30
        if (.not.(pflag) ) then
         x = (x*rten) + j
        else
         temp   = j
         x      = x + (temp/factor)
         factor = factor * rten
         goto 20
        end if

c..must be a decimal point if none of the above
c..check that there are not two decimal points!
       else
        if (pflag) goto 30
        pflag = .true.
       end if
 20   continue
      end do

c..if we got through the do loop ok, then we must be done
      x     = x * sign
      value = x 
      return
      

c..error processing the number
 30   write(6,40) long,string(1:long)
 40   format(' error converting the ',i4,' characters ',/,
     1       ' >',a,'< ',/,
     2       ' into a real number in function value')
      stop ' error in routine value'
      end



