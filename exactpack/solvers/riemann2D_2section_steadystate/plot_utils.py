from numpy import interp, mgrid, array, tan, sqrt, inf, where, linspace
from scipy.optimize import bisect
from scipy.integrate import quad
import matplotlib.pyplot as plt


def streakplot(solver, soln, N=101, xlim=(0., 0.02, 0.62), ylim=(0., 0.4, 0.6), var_str='pressure'):
    Y, X = mgrid[ylim[0]:ylim[2]:complex(0,N), xlim[0]:xlim[2]:complex(0,N)]
    Z = [interp((Y[:,0]-ylim[1]),
                soln['y_position'] * x / (X[0][-1]-2.*xlim[1]),
                soln[var_str])
         for x in (X[0] - xlim[1])]
    Z = array(Z).transpose()
    fig, ax = plt.subplots(1,1)
    ax.set_aspect('equal', adjustable='box')
    colormap = plt.cm.get_cmap('cool')
    colors = colormap(Z)
    c = ax.pcolor(X - xlim[1], Y - ylim[1], Z, shading='auto', vmin=Z.min(), vmax=Z.max(), cmap=colormap)
    angles = solver.angles
    morphology = solver.morphology
    if (morphology[0] == 'R'):
        plt.plot([xlim[0], xlim[2]], [ylim[0], xlim[2] * tan(angles['BR'][0])], '--k')
        plt.plot([xlim[0], xlim[2]], [ylim[0], xlim[2] * tan(angles['BR'][1])], '--k')
    elif (morphology[0] == 'S'):
        plt.plot([xlim[0], xlim[2]], [ylim[0], xlim[2] * tan(angles['BS'])], 'k')
    plt.plot([xlim[0], xlim[2]], [ylim[0], xlim[2] * tan(angles['CD'])], ':k')   
    if (morphology[4] == 'R'):
        plt.plot([xlim[0], xlim[2]], [ylim[0], xlim[2] * tan(angles['TR'][0])], '--k')
        plt.plot([xlim[0], xlim[2]], [ylim[0], xlim[2] * tan(angles['TR'][1])], '--k')
    elif (morphology[4] == 'S'):
        plt.plot([xlim[0], xlim[2]], [ylim[0], xlim[2] * tan(angles['TS'])], 'k')
    plt.xlim((xlim[0], xlim[2] - xlim[1]))
    plt.ylim((ylim[0] - ylim[1], ylim[2] - ylim[1]))
    plt.title(var_str)
    fig.colorbar(c, ax=ax)
    plt.show()


def plot_pressure_deflection(solver):
    bottom_state, top_state = solver.bottom_state, solver.top_state
    pB, thetaB_rad = solver.pB, solver.thetaB_rad
    pBc, dBc = solver.pBc, solver.dBc
    pBe, dBe = solver.pBe, solver.dBe
    pT, thetaT_rad = solver.pT, solver.thetaT_rad
    pTc, dTc = solver.pTc, solver.dTc
    pTe, dTe = solver.pTe, solver.dTe
    plt.plot(thetaB_rad - dBc, pBc, 'g', label=r'bottom compression')
    plt.plot(thetaB_rad - dBe, pBe, '--g', label=r'bottom expansion')
    plt.axhline(pB, color='g', linestyle=':', label=r'bottom-state pressure')    
    plt.axvline(solver.deflection_angle_solution, color='k', linestyle='--', label=r'analytic deflection angle')
    plt.plot(thetaT_rad + dTc, pTc, 'b', label=r'top compression')
    plt.plot(thetaT_rad + dTe, pTe, '--b', label=r'top expansion')
    plt.axhline(pT, color='b', linestyle=':', label=r'top-state pressure')
    plt.axhline(solver.pressure_solution, color='k', linestyle='--', label=r'analytic pressure')
    plt.legend(ncol=2)
    plt.xlabel(r'deflection angle')
    plt.ylabel(r'pressure')
    plt.show()


def plot_lineouts(solver, option='', save_filename=''):
    vals = solver.lineout_vals
    position = vals[1]
    pressure = vals[3]
    density = vals[4]
    sie = vals[5]
    Mach = vals[6]
    velocity_x = vals[7]
    velocity_y = vals[8]
    speed = vals[9]
    if (option == 'divide_maxs'):
        plt.plot(position, pressure / max(pressure), label=r'1 x pressure / max pressure')
        plt.plot(position, 2. * density / max(density), label=r'2 x density / max density')
        plt.plot(position, 3. * sie / max(sie), label=r'3 x sie / max sie')
        plt.plot(position, 4. * Mach / max(Mach), label=r'4 x Mach / max Mach')
        plt.plot(position, 5. * vals[9] / max(vals[9]), label=r'5 x speed / max speed')
    else:
        plt.plot(position, pressure, label=r'pressure')
        plt.plot(position, density, label=r'density')
        plt.plot(position, sie, label=r'sie')
        plt.plot(position, Mach, label=r'Mach')
        plt.plot(position, velocity_x, label=r'x-velocity')
        plt.plot(position, velocity_y, label=r'y-velocity')
        plt.plot(position, speed, label=r'speed')
    plt.legend()
    if (save_filename != ''):
        plt.savefig(save_filename, transparent=True)
    plt.show()


def plot_data(solver, soln, N=21, xlim=(0., 0.5, 1.), ylim=(0., 0.5, 1.), var_str='pressure'):
    plot_pressure_deflection(solver)
    plot_lineouts(solver)
    for var in var_str:
        streakplot(solver, soln, N=N, xlim=xlim, ylim=ylim, var_str=var)
