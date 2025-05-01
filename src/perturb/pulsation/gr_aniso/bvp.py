'''Locate the working directory'''
if __name__ == '__main__':    
    import sys
    from _path import workdir
    sys.path.append(workdir)
    print(sys.path[-1])

#=========================================================
'''Main content of the module'''
import numpy as np
from scipy.integrate import solve_ivp
from src.util.constants import G, c

from src.perturb.pulsation.gr_aniso.ode import ode_LD_aniso
from src.perturb.pulsation.Zerilli import ode_Zerilli

class bvp_LD_aniso:
    def __init__(self,ell,w,bsol):
        self.ell = ell
        self.w = w
        self.bsol = bsol

        self.r_inf = 25.*c/np.real(self.w)
        self.atol_factor = 1.e-6
        self.rtol = 1.e-8

        self.z_atol_factor = 1.e-14
        self.z_rtol = 1.e-6

        # self.ode_method = 'LSODA'
        # self.ode_method = 'DOP853'
        self.ode_method = 'RK45'

    def solve_all(self,save_data=False):
        ell = self.ell
        w = self.w
        bsol = self.bsol

        ode_int = ode_LD_aniso(ell,w,bsol)
        
        yc = np.zeros(shape=(4,5),dtype=np.complex128)

        # Integrate outward from r = 0
        yc0 = ode_int.centerbc()
        for i in range(2):
            _ , yout = self.integrate_fwd(yc0[i,:],ode_int.deriv)
            yc[:,i] = yout[:,-1]

        # Integrate inward from r = R
        ycf = ode_int.surfbc()
        for i in range(3):
            _ , yout = self.integrate_bwd(ycf[i,:],ode_int.deriv)
            yc[:,i+2] = yout[:,0]
        
        # Matching the solutions at r = R/2
        coef = self.solve_coef(yc)

        yvc0 = np.matmul(np.transpose(ycf),coef[2:5])
        Wm = np.dot(yc[2,2:5],coef[2:5])
        yvc0 /= Wm
        # (H0, K) are continuous at r = R
        H0 = ode_int.alg_H0(bsol.r[-1],yvc0,True)
        K = yvc0[1]

        # Integrate Zerilli's equations outward from r = R
        ode_ext = ode_Zerilli(ell,w,bsol.r[-1],bsol.m[-1])
        zc0 = ode_ext.surfbc(H0,K)
        rout_ext, zout = self.integrate_Zerilli(zc0,ode_ext.deriv)

        # Obtaining the wave amplitudes in the wave zone (r -> inf)
        Aout, Ain = ode_ext.infbc(self.r_inf,zout[:,-1])

        if save_data:
            yc0_correct = np.matmul(np.transpose(yc0),coef[0:2])
            rout, yout = self.integrate_fwd(yc0_correct,ode_int.deriv)
            self.yc0 = yc0_correct
            self.rsol, self.ysol = rout, yout

            ycf_correct = np.matmul(np.transpose(ycf),coef[2:5])
            rout, yout = self.integrate_bwd(ycf_correct,ode_int.deriv)
            self.ycf = ycf_correct
            self.rsol = np.append(self.rsol,rout)
            self.ysol = np.append(self.ysol,yout,axis=1)

            self.rsol_ext = rout_ext
            self.zsol = zout

        return Aout, Ain
    
    def integrate_fwd(self,yc0,deriv):

        bsol = self.bsol
        
        r0 = bsol.r[0]
        rf = bsol.r[-1]/2
        first_step = np.abs(rf-r0)/200

        y0 = np.zeros(8)
        y0[0:4] = yc0.real
        y0[4:8] = yc0.imag

        yscale = y0*0.
        yscale[0:2] = G/c**2*bsol.rho[0]
        yscale[2] = 1.
        yscale[3] = (G/c**2*bsol.rho[0])**2
        yscale[4:8]=yscale[0:4]
        atol = yscale*self.atol_factor

        isol = solve_ivp(deriv, t_span=[r0,rf], first_step = first_step, method = self.ode_method \
                            , y0 = y0, atol = atol, rtol = self.rtol)
        r = isol.t
        yc = isol.y[0:4,:] + 1j*isol.y[4:8,:]

        return r,yc
    
    def integrate_bwd(self,yc0,deriv):

        bsol = self.bsol
        
        r0 = bsol.r[-1]
        rf = bsol.r[-1]/2
        first_step = np.abs(rf-r0)/200

        y0 = np.zeros(8)
        y0[0:4] = yc0.real
        y0[4:8] = yc0.imag
        
        yscale = y0*0.
        yscale[0:2] = 1./r0**2
        yscale[2] = 1.
        yscale[3] = 1./r0**4
        yscale[4:8]=yscale[0:4]
        atol = yscale*self.atol_factor

        isol = solve_ivp(deriv, t_span=[r0,rf], first_step = first_step, method = self.ode_method \
                            , y0 = y0, atol = atol, rtol = self.rtol)
        r = isol.t[::-1]
        yc = isol.y[0:4,::-1] + 1j*isol.y[4:8,::-1]
        return r,yc
    
    def solve_coef(self,yc):
        a = np.copy(yc[:,1:5])
        a[:,0] *= -1.
        b = np.array(yc[:,0])
        x = np.linalg.solve(a,b)
        return np.append(np.array([1.]),x)
    
    def integrate_Zerilli(self,zc0,deriv):
        
        bsol = self.bsol

        r0 = bsol.r[-1]
        rf = self.r_inf
        first_step = np.abs(rf-r0)/500

        z0 = np.zeros(4)
        z0[0:2] = zc0.real
        z0[2:4] = zc0.imag

        zscale = z0*0.
        zscale[0] = 1./r0**2
        zscale[1] = 1./r0**3
        zscale[2:4] = zscale[0:2]
        atol = zscale*self.z_atol_factor

        isol = solve_ivp(deriv, t_span=[r0,rf], first_step = first_step \
                            , y0 = z0, atol = atol, rtol = self.z_rtol)
        r = isol.t
        zc = isol.y[0:2,:] + 1j*isol.y[2:4,:]
        return r,zc

#=========================================================
'''Example run of this module'''
def test_run():
    from src.eos.poly import eos_poly
    from src.static.tov import solve_tov
    
    return

if __name__ == '__main__':
    test_run()