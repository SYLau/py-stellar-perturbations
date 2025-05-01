'''Locate the working directory'''
if __name__ == '__main__':    
    import sys
    from _path import workdir
    sys.path.append(workdir)
    print(sys.path[-1])

#=========================================================
'''Main content of the module'''
import numpy as np
from src.util.constants import G, c
import matplotlib.pyplot as plt
# import scienceplots

from src.perturb.pulsation.gr_aniso.bvp import bvp_LD_aniso
from src.perturb.pulsation.gr_aniso.ode import ode_LD_aniso

class evp_LD_aniso:
    
    def __init__(self,ell,bsol):
        self.ell = ell
        self.bsol = bsol
        self.atol_factor = 1.e-6
        self.rtol = 1.e-8
        self.z_atol_factor = 1.e-14
        self.z_rtol = 1.e-6
        self.muller_imax = 30
        self.muller_eps = 1.e-8
        self.muller_print = False

        self.ode_method = 'RK45'

    def scan_Ain(self,flist,add=False):
        Ain = np.array(list(map(self._get_Ain, flist)))
        if add:
            self.flist = np.append(self.flist,flist)
            self.Ain = np.append(self.Ain,Ain)
            idx = np.argsort(self.flist)
            self.flist = self.flist[idx]
            self.Ain = self.Ain[idx]
        else:
            self.flist = flist
            self.Ain = Ain
    
    def solve_mode(self,f0,fw):
        fguess = np.array([f0, f0-fw, f0+fw])
        wmode = self._rtmuller(self._get_Ain,fguess)
        if wmode != None:
            bvp = bvp_LD_aniso(self.ell,wmode,self.bsol)
            bvp.ode_method = self.ode_method
            bvp.atol_factor, bvp.rtol = self.atol_factor, self.rtol
            bvp.z_atol_factor, bvp.z_rtol = self.z_atol_factor, self.z_rtol
            _, _ = bvp.solve_all(save_data=True)
            self._get_eigenfunctions(wmode, bvp.rsol, bvp.ysol, bvp.rsol_ext, bvp.zsol)
            return wmode
        else:
            return None
    
    def plot_Ain(self,xlim = None):
        plt.figure(figsize=(8.6,6.4), dpi= 100)
        # plt.style.use('science')
        plt.plot(self.flist, np.abs(self.Ain), 'k-')
        plt.title(r'Ingoing wave amplitude')
        plt.xlabel(r'$f$ (Hz)',fontsize=20)
        plt.ylabel(r'$|A_\text{in}|$',fontsize=20)
        plt.xticks(fontsize='20')
        plt.yticks(fontsize='20')
        plt.yscale('log')
        if xlim != None:
            plt.xlim(xlim)
        plt.show()
        plt.close()

    def _get_Ain(self,f):
        bvp = bvp_LD_aniso(self.ell,2*np.pi*f,self.bsol)
        bvp.ode_method = self.ode_method
        bvp.atol_factor, bvp.rtol = self.atol_factor, self.rtol
        bvp.z_atol_factor, bvp.z_rtol = self.z_atol_factor, self.z_rtol
        _ , Ain = bvp.solve_all()
        return Ain
    
    def _get_eigenfunctions(self,w,r,y,rext,zext):
        ode = ode_LD_aniso(self.ell,w,self.bsol)
        
        self.r = r
        self.H1 = y[0,:]
        self.K = y[1,:]
        self.W = y[2,:]
        self.X = y[3,:]

        self.H0 = np.zeros(len(r),dtype=np.complex128)
        self.Z = np.zeros(len(r),dtype=np.complex128)
        self.V = np.zeros(len(r),dtype=np.complex128)
        for i in range(len(r)):
            self.H0[i], self.Z[i], self.V[i] = ode.alg_all(r[i],y[:,i],find_yr=True)
        
        self.rext = rext
        self.Zer = zext[0,:]
        self.dZer = zext[1,:]

    def _rtmuller(self,func,x):
        imax = self.muller_imax
        eps = self.muller_eps
        x1,x2,x3 = x[:]
        x2b = (np.abs(x3)+np.abs(x2))/2

        i = 0
        while i <= imax:
            i += 1
            f1,f2,f3 = func(x1), func(x2), func(x3)
            q=(x3-x2)/(x2-x1)
            aa = q*f3-q*(1.-q)*f2+q**2*f1
            bb = (2.*q+1.)*f3-(1.+q)**2*f2+q**2*f1
            cc = (1.+q)*f3

            discrim = np.sqrt(bb*bb - 4*aa*cc)
            if (np.abs(bb+discrim) > np.abs(bb-discrim)):
                denom = bb + discrim
            else:
                denom = bb - discrim

            xtemp = x3 - (x3-x2)*(2*cc) / denom

            x2b = (abs(x3) + abs(x2))/2
            x1 = x2
            x2 = x3
            x3 = xtemp

            if self.muller_print:
                print('%10.4e + %10.4e i, |f1| = %10.4e, error = %10.4e'%(np.real(xtemp), np.imag(xtemp), np.abs(f1), abs((x3-x2)/x2b)))

            if (abs(x3-x2) <= eps*abs(x2b)):
                break
        if i > imax:
            return None
        else:
            return x3
