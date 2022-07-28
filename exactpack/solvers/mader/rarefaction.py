"""Python implementation of the mader solver."""
import numpy as np

# t=t,
# x=r,
# p_cj=self.p_cj,
# d_cj=self.d_cj,
# gamma=self.gamma,
# u_piston=self.u_piston

def mader(t, x, p_cj, d_cj, gamma, u_piston):
    """Blah

    Args:
        t (float):
        x (ndarray): Size nstep
        p_cj (float):
        d_cj (float):
        gamma (float):
        u_piston (float)

    Returns:
        tuple: A 5-tuple containing:
            - *ndarray*: u, Size nstep
            - *ndarray*: p, Size nstep
            - *ndarray*: c, Size nstep
            - *ndarray*: rho, Size nstep
            - *ndarray*: xdet, Size nstep
    """
    nstep = len(x)
    dx = (x[-1] - x[0]) / nstep
    u = np.zeros(nstep)
    p = np.zeros(nstep)
    c = np.zeros(nstep)
    rho = np.zeros(nstep)
    xdet = np.zeros(nstep)
    
    for i in range(nstep):
        xi = x[i]
        ui, pi, ci, rhoi, xdeti = rare(t, xi, dx, p_cj, d_cj, gamma, u_piston)
        u[i] = ui
        p[i] = pi
        c[i] = ci
        rho[i] = rhoi
        xdet[i] = xdeti

    return u, p, c, rho, xdet


def rare(time, xlab, dx, p_cj, d_cj, gam, u_piston):
    """Returns the rarefaction wave solution given on page 24 of fickett and
    davis.

    Args:
        time (float): Time for desired solutioon (s)
        xlab (float): Position in fixed lab frame, eularian frame (cm)
        dx (float): Width of grid cell (cm)
        p_cj (float): Chapman-jouget pressure (erg/cm**3)
        d_cj (float): Chapman-jouget density (g/cm**3)
        gam (float): Ratio of specific heats (dimensionless)
        u_piston (float): Speed of piston  (cm/s)

    Returns:
        tuple: A 5-tuple containing:
            - *float*: u, material speed (cm/s)
            - *float*: p, pressure (erg/cm**3)
            - *float*: c, sound speed (cm/s)
            - *float*: rho, mass density (g/cm**3)
            - *float*: xdet, position relative to detonation front, lagrangian
                frame (cm)
    """
    # some constants and factors
    gamp1 = gam + 1.0
    rho_0 = gamp1 * p_cj / d_cj**2
    rho_cj = rho_0 * gamp1 / gam
    c_cj = gam * d_cj / gamp1
    u_cj = d_cj / gamp1

    gamm1 = gam - 1.0
    aa = 1.0 / (2.0 * c_cj * time)
    bb = (2.0 - gamm1 * u_cj / c_cj) / gamp1
    b = 2.0 * gam / gamm1
    d = 2.0 / gamm1
    dd = 2.0 / (time*gamp1)
    ee = gamm1 * (u_cj - 2.0 * c_cj / gamm1) / gamp1
    bp1 = b + 1.0
    dp1 = d + 1.0

    um = gamm1 * (u_cj - 2.0 * c_cj / gamm1) / gamp1
    xp = 0.5 * gamp1 * time * (u_piston - um)
    xdet = d_cj * time - xlab
    dist = abs(xdet - xp)
    tol = 0.1 * dx

    # solution in the frame relative to detonation front, lagrangian frame
    x = xdet
    half = 0.5 * dx
    x1 = x - half

    # solution in the rarefaction fan
    if dist > tol and xdet > xp:
        u = dd * (x1 + half) + ee
        p = p_cj * ((aa * (x1 + dx) + bb)**bp1 - (aa * x1 + bb)**bp1) / (dx * aa * bp1)
        c = c_cj * (aa * (x1 + half) + bb)
        rho = rho_cj * ((aa * (x1 + dx) + bb)**dp1 - (aa * x1 + bb)**dp1) / (dx * aa * dp1)
    # solution if right at the transition point
    elif (dist <= tol):
        #  partial q's
        x2 = x1 + dx
        dxp = (x2 - xp)
        h = dxp / 2
        u = dd * (x1 + h) + ee
        p = p_cj * ((aa * (x1 + dxp) + bb)**bp1 - (aa * x1 + bb)**bp1) / (dxp * aa * bp1)
        c = c_cj * (aa * (x1 + h) + bb)
        rho = rho_cj * ((aa * (x1 + dxp) + bb)**dp1 - (aa * x1 + bb)**dp1) / (dxp * aa * dp1)

        # residual q's
        ur = u_piston
        pr = p_cj * (1 + gamm1 * (u - u_cj) / (2.0 * c_cj))**(2.0 * gam / gamm1)
        cr = c_cj * (1 + gamm1 * (u - u_cj) / (2.0 * c_cj))
        rhor = rho_cj * (p / p_cj)**(1.0 / gam)

        # avg q's
        u = ur + (u - ur) * 2.0 * h / dx
        p = pr + (p - pr) * 2.0 * h / dx
        c = cr + (c - cr) * 2.0 * h / dx
        rho = rho + (rho - rhor) * 2.0 * h / dx
    # solution in the constant state
    else:
        u = u_piston
        p = p_cj * (1 + gamm1 * (u - u_cj) / (2.0 * c_cj))**(2.0 * gam / gamm1)
        c = c_cj * (1 + gamm1 * (u - u_cj) / (2.0 * c_cj))
        rho = rho_cj * (p / p_cj)**(1.0 / gam)

    return u, p, c, rho, xdet
