from numpy import linspace, arctan, sqrt, pi, append, argmin, argmax, interp
from numpy import sin, cos, array, tan, arcsin, vstack
from scipy.optimize import bisect, fsolve
import matplotlib.pyplot as plt

class SetupRiemannProblem(object):
    r"""Sets upa  2D Riemann problem with separate left- and right-states. This
        is for an ideal-gas EOS, but the adiabatic constant can be different for
        the left and right states.
        An incoming state contains values for:
        state = [pressure, density, Mach, flow angle in degrees, gamma]
     """
    def __init__(self, bottom_state, top_state):
        self.bottom_state = bottom_state
        self.top_state = top_state
        self.set_initial_state_values()
        self.bottom_compression_arrays, self.bottom_expansion_arrays = \
            self.setup_initial_arrays(bottom_state)
        self.top_compression_arrays, self.top_expansion_arrays = \
            self.setup_initial_arrays(top_state)
        self.find_overlap()
        self.set_starstate_values()


    def set_initial_state_values(self):
        pB, rB, MB, thetaB_deg, gB = self.bottom_state
        pT, rT, MT, thetaT_deg, gT = self.top_state
        thetaB_rad, thetaT_rad = array([thetaB_deg, thetaT_deg]) / 180. * pi
        muB_rad = thetaB_rad - arcsin(1. / MB)
        muT_rad = thetaT_rad + arcsin(1. / MT)
        cB = sqrt(gB * pB / rB)
        uB, vB = cB * MB * array([cos(thetaB_rad), sin(thetaB_rad)])
        cT = sqrt(gT * pT / rT)
        uT, vT = cT * MT * array([cos(thetaT_rad), sin(thetaT_rad)])
        self.pB, self.rB, self.MB, self.gB = pB, rB, MB, gB
        self.thetaB_deg, self.thetaB_rad = thetaB_deg, thetaB_rad
        self.muB_rad, self.muT_rad = muB_rad, muT_rad
        self.uB, self.vB = uB, vB
        self.pT, self.rT, self.MT, self.gT = pT, rT, MT, gT
        self.thetaT_deg, self.thetaT_rad = thetaT_deg, thetaT_rad
        self.uT, self.vT = uT, vT


    def compression_states(self, ps, state):
        p0, r0, M0, theta0_deg, g = state
        c0 = sqrt(g * p0 / r0)
        u0 = M0 * c0 * cos(theta0_deg / 180. * pi)
        v0 = M0 * c0 * sin(theta0_deg / 180. * pi)
        alphas = ps / p0
        M0 = sqrt((u0**2. + v0**2.) / (g * p0 / r0))
        rho_num = (g + 1.) * alphas + g - 1.
        rho_den = (g - 1.) * alphas + g + 1.
        tmps  = sqrt(2. * g * M0**2. / rho_num - 1.)
        tmps *= (alphas - 1.) / (g * M0**2. - alphas + 1.)
    
        deflection_angles = arctan(tmps)
        rs = r0 * rho_num / rho_den
        Ms = sqrt((M0**2. * rho_num - 2. * (alphas**2. - 1.)) / alphas/rho_den)
        return deflection_angles, rs, Ms
    
    
    def PrandtlMeyer_function(self, Ms, g):
        mu2 = sqrt((g + 1.) / (g - 1.))
        M2m1 = Ms**2. - 1.
        return mu2 * arctan(sqrt(M2m1) / mu2) - arctan(M2m1)
    
    
    def expansion_states(self, ps, state):
        p0, r0, M0, theta0_deg, g = state
        c0 = sqrt(g * p0 / r0)
        u0 = M0 * c0 * cos(theta0_deg / 180. * pi)
        v0 = M0 * c0 * sin(theta0_deg / 180. * pi)
        alphas = ps / p0
        M0 = sqrt((u0**2. + v0**2.) / (g * p0 / r0))
        gm1 = g - 1.
    
        Ms = sqrt(((gm1 * M0**2. / 2. + 1.) / alphas**(gm1 / g) - 1.) * 2 / gm1)
        rs = r0 * alphas**(1./g)
        nu_M0 = self.PrandtlMeyer_function(M0,g)
        nu_Ms = self.PrandtlMeyer_function(Ms,g)
        deflection_angles = nu_M0 - nu_Ms
        return deflection_angles, rs, Ms
    
    
    def test_for_nans(self, ps, ds, rs, Ms):
        tmp_d = len(ds)
        tmp_M = len(Ms)
        if (argmin(ds) == argmax(ds)):
           tmp_d = argmin(ds)
        if (argmin(Ms) == argmax(Ms)):
           tmp_M = argmin(Ms)
        tmp = min(tmp_d, tmp_M)
        return ps[:tmp], ds[:tmp], rs[:tmp], Ms[:tmp]
    
    
    def remove_subsonic_compression(self, compression_array):
        ps, ds, rs, Ms = compression_array
        d_argval = argmax(ds) + 1
        M_argval = sum(Ms > 1.) - 1
        argval = min(d_argval, M_argval)
        return ps[:argval], ds[:argval], rs[:argval], Ms[:argval]
    
    
    def setup_initial_arrays(self, state, num=10000):
        ps_comp = linspace(state[0], 10. * state[0], num)
        ps_exp = linspace(1.e-10, state[0], num)[:-1]
        ds, rs, Ms = self.compression_states(ps_comp, state)
        compression_array = self.test_for_nans(ps_comp, ds, rs, Ms)
        ps_comp, ds_comp, rs_comp, Ms_comp = \
            self.remove_subsonic_compression(compression_array)
        ds_exp,  rs_exp,  Ms_exp  = self.expansion_states(ps_exp, state)
        return [ps_comp,ds_comp,rs_comp,Ms_comp],[ps_exp,ds_exp,rs_exp,Ms_exp]
    
    
    def determine_state_functions(self, pressure_guess):
        pTc = self.top_compression_arrays[0]
        pTe = self.top_expansion_arrays[0]
        pBc = self.bottom_compression_arrays[0]
        pBe = self.bottom_expansion_arrays[0]
        bottom_state, top_state = self.bottom_state, self.top_state
        bottom_flow_angle, top_flow_angle = self.thetaB_rad, self.thetaT_rad
        if (pBc[::-1][-1] < pressure_guess < pBc[::-1][0]):
            bottom_function = lambda x: bottom_flow_angle - self.compression_states(x, bottom_state)[0]
            morphology = 'S-C-'
        elif (pBe[::-1][-1] < pressure_guess < pBe[::-1][0]):
            bottom_function = lambda x: bottom_flow_angle - self.expansion_states(x, bottom_state)[0]
            morphology = 'R-C-'
        if (pTe[0] < pressure_guess < pTe[-1]):
            top_function = lambda x: top_flow_angle + self.expansion_states(x, top_state)[0]
            morphology += 'R'
        elif (pTc[0] < pressure_guess < pTc[-1]):
            top_function = lambda x: top_flow_angle + self.compression_states(x, top_state)[0]
            morphology += 'S'
        self.morphology = morphology
        return top_function, bottom_function
    
    
    def find_overlap(self):
        bottom_flow_angle = self.thetaB_rad
        pTc, dTc = self.top_compression_arrays[:2]
        pTe, dTe = self.top_expansion_arrays[:2]
        pBc, dBc = self.bottom_compression_arrays[:2]
        pBe, dBe = self.bottom_expansion_arrays[:2]
        dB = append(bottom_flow_angle-dBc[::-1], bottom_flow_angle-dBe[::-1])
        dT = append(bottom_flow_angle+dTe, bottom_flow_angle + dTc)
        left_ds_bound = max(min(dB), min(dT)) 
        right_ds_bound = min(max(dT), max(dB)) 
        ds = linspace(left_ds_bound, right_ds_bound, int(1e4))
        pTs = interp(ds, dT, append(pTe, pTc))
        pBs = interp(ds, dB, append(pBc[::-1], pBe[::-1]))
        guess_deflection_angle = bisect(lambda x: interp(x, ds, pTs - pBs),
                       left_ds_bound, right_ds_bound)
        guess_pressure = interp(guess_deflection_angle, ds, pTs)
        top_function, bottom_function = \
            self.determine_state_functions(guess_pressure)
        self.pressure_solution = fsolve(lambda x: top_function(x) - bottom_function(x), guess_pressure)[0]
        self.deflection_angle_solution = top_function(self.pressure_solution)
    
    
    def determine_shock_angle(self, state):
        angle = self.deflection_angle_solution
        _, _, M, _, g = state
        def get_shock_contact_angle(x):
            val  = 2. / tan(x) * (M**2 * sin(x)**2 - 1.)
            val /= (2. + M**2 * (g + cos(2. * x)))
            return val
        return fsolve(lambda x:
                      get_shock_contact_angle(x)-tan(angle), angle)[0]
    

    def set_starstate_values(self):
        top_state = self.top_state
        bottom_state = self.bottom_state
        thetaB_rad, gB = self.thetaB_rad, self.gB
        thetaT_rad, gT = self.thetaT_rad, self.gT
        p_star,cd_angle = self.pressure_solution,self.deflection_angle_solution
        angles = {}
        if (self.morphology[0] == 'R'):
            rB_star, MB_star = \
                self.expansion_states(p_star, bottom_state)[1:3]
            angles['BR'] = self.muB_rad + array([0., cd_angle - thetaB_rad])
        elif (self.morphology[0] == 'S'):
            rB_star, MB_star = \
                self.compression_states(p_star, bottom_state)[1:3]
            angles['BS'] = self.determine_shock_angle(bottom_state)
        angles['CD'] = self.deflection_angle_solution
        if (self.morphology[4] == 'R'):
            rT_star, MT_star = \
                self.expansion_states(p_star, top_state)[1:3]
            angles['TR'] = self.muT_rad + array([cd_angle - thetaT_rad, 0.])
        elif (self.morphology[4] == 'S'):
            rT_star, MT_star = \
                self.compression_states(p_star, top_state)[1:3]
            angles['TS'] = self.determine_shock_angle(top_state)
        self.angles = angles
        uB_star, vB_star = sqrt(gB * p_star / rB_star) * MB_star * array([cos(cd_angle), sin(cd_angle)])
        uT_star, vT_star = sqrt(gT * p_star / rT_star) * MT_star * array([cos(cd_angle), sin(cd_angle)])
        bottom_star_vals = [p_star,rB_star,MB_star,uB_star,vB_star]
        top_star_vals = [p_star,rT_star,MT_star,uT_star,vT_star]
        self.rB_star, self.MB_star = rB_star, MB_star
        self.rT_star, self.MT_star = rT_star, MT_star
        self.uB_star,self.vB_star = uB_star,vB_star
        self.uT_star,self.vT_star = uT_star,vT_star
        self.bottom_star_vals,self.top_star_vals=bottom_star_vals,top_star_vals


    def assign_lineout_vals(self, xs, ys):
        xy_thetas_rad = [arctan(ys[ii] / x) for ii, x in enumerate(xs)]
        xy_thetas_rad = array(xy_thetas_rad)
        bottom_state, top_state = self.bottom_state, self.top_state
        pB, rB, MB, thetaB_deg, gB = bottom_state
        uB, vB = self.uB, self.vB
        pT, rT, MT, thetaT_deg, gT = top_state
        uT, vT = self.uT, self.vT
        thetaB_rad, thetaT_rad = array([thetaB_deg, thetaT_deg]) / 180. * pi
        p_star = self.pressure_solution
        rB_star, MB_star, uB_star = self.rB_star, self.MB_star, self.uB_star
        vB_star = self.vB_star
        rT_star, MT_star, uT_star = self.rT_star, self.MT_star, self.uT_star
        vT_star = self.vT_star
        sieB = pB / rB / (gB - 1.)
        sieT = pT / rT / (gT - 1.)
        sieB_star = p_star / rB_star / (gB - 1.)
        sieT_star = p_star / rT_star / (gT - 1.)
        bottom_vals = [pB, rB, sieB, MB, uB, vB]
        top_vals = [pT, rT, sieT, MT, uT, vT]
        bottom_star_vals = [p_star,rB_star, sieB_star,MB_star,uB_star,vB_star]
        top_star_vals = [p_star,rT_star, sieT_star,MT_star,uT_star,vT_star]
        lineout_vals = array([xs, ys, xy_thetas_rad, 0.*xs + pB, 0.*xs + rB,
                              0.*xs + sieB, 0.*xs + MB, 0.*xs + uB,
                              0.*xs + vB]).transpose()
        angles = self.angles
        p_low = pB - 1.e-5
        p_high = p_star + 1.e-5
        for ii, vals in enumerate(lineout_vals):
            if (self.morphology[0] == 'R'):
                if (angles['BR'][0]<vals[2]<angles['BR'][1]):
                    this_angle = vals[2] - angles['BR'][0]
                    p_low = fsolve(lambda x: self.expansion_states(x, bottom_state)[0] + this_angle, p_low)[0]
                    d, r, M = self.expansion_states(p_low, bottom_state)
                    sie = p_low / r / (gB - 1.)
                    c = sqrt(gB * p_low / r)
                    u, v = c * M * array([cos(this_angle + thetaB_rad), sin(this_angle + thetaB_rad)])
                    vals[3:] = [p_low, r, sie, M, u, v]
                elif (angles['BR'][1]<vals[2]<angles['CD']):
                    vals[3:] = bottom_star_vals
            elif (self.morphology[0] == 'S'):
                if (angles['BS']<vals[2]<angles['CD']):
                    vals[3:] = bottom_star_vals
            if (self.morphology[4] == 'R'):
                if (angles['CD']<vals[2]<angles['TR'][0]):
                    vals[3:] = top_star_vals
                elif (angles['TR'][0]<vals[2]<angles['TR'][1]):
                    this_angle = vals[2] - angles['TR'][1]
                    p_high = fsolve(lambda x: self.expansion_states(x, top_state)[0] - this_angle, p_high)[0]
                    d, r, M = self.expansion_states(p_high, top_state)
                    sie = p_high / r / (gT - 1.)
                    c = sqrt(gT * p_high / r)
                    u, v = c * M * array([cos(this_angle + thetaT_rad), sin(this_angle + thetaT_rad)])
                    vals[3:] = [p_high, r, sie, M, u, v]
                elif (angles['TR'][1]<vals[2]):
                    vals[3:] = top_vals
            elif (self.morphology[4] == 'S'):
                if (angles['CD']<vals[2]<angles['TS']):
                    vals[3:] = top_star_vals
                elif (angles['TS']<vals[2]):
                    vals[3:] = top_vals
        lineout_vals = lineout_vals.transpose()
        speed = sqrt(lineout_vals[7]**2 + lineout_vals[8]**2)
        lineout_vals = vstack([lineout_vals, speed])
        self.lineout_vals = lineout_vals
