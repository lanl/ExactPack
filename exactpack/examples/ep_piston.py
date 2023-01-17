#
#  Creates plots for the elastic-plastic piston problem
#  in ExactPack
#

from exactpack.solvers.ep_piston import EPpiston
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', size=14)

import numpy as np

#Call the analytic solution solver for each kind of elastic model using the same parameters
solution_Ifin = EPpiston(gamma=2.,c0=0.533,s0=1.34,model='hyperIfin',G=0.286,Y=0.0026,rho0=2.79,up=0.01)

print('Infinitesimal Strain Hyperelastic Analytic Solution')
print('Elastic wave values')
print('Density, Pressure, Wave Speed, Particle Speed')
print(solution_Ifin.rho_y,solution_Ifin.p_y,solution_Ifin.wv_el,solution_Ifin.vel_y)
print()
    
print('Plastic wave values')
print('Density, Pressure, Wave Speed')
print(solution_Ifin.rho2,solution_Ifin.p2,solution_Ifin.wv_pl)
print()

solution_Fin = EPpiston(gamma=2.,c0=0.533,s0=1.34,model='hyperFin',G=0.286,Y=0.0026,rho0=2.79,up=0.01)

print('Finite Strain Hyperelastic Analytic Solution')
print('Elastic wave values')
print('Density, Pressure, Wave Speed, Particle Speed')
print(solution_Fin.rho_y,solution_Fin.p_y,solution_Fin.wv_el,solution_Fin.vel_y)
print()
    
print('Plastic wave values')
print('Density, Pressure, Wave Speed')
print(solution_Fin.rho2,solution_Fin.p2,solution_Fin.wv_pl)
print()

solution_Hypo = EPpiston(gamma=2.,c0=0.533,s0=1.34,model='hypo',G=0.286,Y=0.0026,rho0=2.79,up=0.01)

print('Hypoelastic Analytic Solution')
print('Elastic wave values')
print('Density, Pressure, Wave Speed, Particle Speed')
print(solution_Hypo.rho_y,solution_Hypo.p_y,solution_Hypo.wv_el,solution_Hypo.vel_y)
print()
    
print('Plastic wave values')
print('Density, Pressure, Wave Speed')
print(solution_Hypo.rho2,solution_Hypo.p2,solution_Hypo.wv_pl)
print()

#  Set vector of locations to evaluate the solution at the set final time
xmax = 2.0
t = 2.0
NP = 201
xvec = np.linspace(0, xmax, NP)

#  Evaluate the solution
print('Values for Infinitesimal Strain Hyperelastic-plastic')
result_Ifin = solution_Ifin._run(xvec, t, xmax)

print('Values for Finite Strain Hyperelastic-plastic')
result_Fin = solution_Fin._run(xvec, t, xmax)

print('Values for Hypoelastic-plastic')
result_Hypo = solution_Hypo._run(xvec, t, xmax)

wv_elx=solution_Ifin.wv_el*t
wv_plx=solution_Ifin.wv_pl*t
#Plot the pressure from each solution for the whole problem size
fig = plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1)
plt.plot(xvec, result_Ifin['pressure'], 'r')
plt.plot(xvec, result_Fin['pressure'], 'b')
plt.plot(xvec, result_Hypo['pressure'], 'g')
plt.title('Piston Solution')
plt.ylabel(r'Pressure [$\rm{Mbar}$]')
plt.xlabel(r'x [$\rm{cm}$]')
plt.legend(["Infinitesimal Hyperelastic","Finite Hyperelastic","Hypoelastic"])
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.ylim(0.0,1.5e-2)
plt.xlim(solution_Ifin.up*t,2.)
plt.axvline(x=wv_plx-0.1,ymin=(solution_Ifin.p_y-7.5e-4)/1.5e-2,ymax=(solution_Ifin.p_y+7.5e-4)/1.5e-2)
plt.axvline(x=wv_elx+0.1,ymin=(solution_Ifin.p_y-7.5e-4)/1.5e-2,ymax=(solution_Ifin.p_y+7.5e-4)/1.5e-2)
plt.axhline(y=solution_Ifin.p_y-7.5e-4,xmin=(wv_plx-0.1)/2.,xmax=(wv_elx+0.1)/2.)
plt.axhline(y=solution_Ifin.p_y+7.5e-4,xmin=(wv_plx-0.1)/2.,xmax=(wv_elx+0.1)/2.)

#Plot the pressure from each solution zoomed in at the elastic wave plateau
plt.subplot(1, 2, 2)
plt.plot(xvec, result_Ifin['pressure'], 'r')
plt.plot(xvec, result_Fin['pressure'], 'b')
plt.plot(xvec, result_Hypo['pressure'], 'g')
plt.title('Elastic Wave Differences')
plt.ylabel(r'Pressure [$\rm{Mbar}$]')
plt.xlabel(r'x [$\rm{cm}$]')
plt.legend(["Infinitesimal Hyperelastic","Finite Hyperelastic","Hypoelastic"])
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.xlim(wv_plx,wv_elx)
#plt.xlim(1.29,1.31)
plt.ylim(solution_Ifin.p_y-1e-4,solution_Ifin.p_y+1e-4)
#plt.ylim(0.0036,0.0037)
plt.tight_layout()
plt.show()

plt.close()
