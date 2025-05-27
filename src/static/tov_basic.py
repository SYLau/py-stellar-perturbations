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

class solve_tov:
    def __init__(self,eos):
            self.eos = eos
            self.r = np.array([])
            self.ysol = np.array([])
            self.p = np.array([])
            self.m = np.array([])
            self.nu = np.array([])
            self.rho = np.array([])

            self.atol = 0.
            self.rtol = 1.e-12

    def deriv(self,r,y):
        p=y[0]
        m=y[1]
        nu=y[2]
        rho = self.eos.rho(p)
        dydr=0.0*y
        dydr[0]= -G*(rho+p/c**2)*(m+4*np.pi*r**3*p/c**2)/r**2/(1-2*G*m/r/c**2)
        dydr[1]= 4*np.pi*r**2*rho
        dydr[2]= 2*G/c**2*(m+4*np.pi*r**3*p/c**2)/r**2/(1-2*G*m/r/c**2)
        return dydr

    def ivp(self, p0, pcut):
        rho0 = self.eos.rho(p0)
        Rs = np.sqrt(3.*p0/(2.*np.pi*G*(rho0+p0/c**2)*(rho0+3.*p0/c**2)))
        dr = 1e-5*Rs
        r0 = dr
        m0 = 4.*np.pi/3*rho0*r0**3
        nu0 = 2.*np.pi/3*G/c**2*(rho0+3.*p0/c**2)*r0**2

        y = np.zeros(3)
        y[0] = p0
        y[1] = m0
        y[2] = nu0

        def pcut_event(r,y):
            return y[0]-pcut
        pcut_event.terminal = True
        pcut_event.direction = -1
        
        isol = solve_ivp(self.deriv, t_span=[r0,Rs*30], y0 = y \
                        , atol = self.atol, rtol = self.rtol, events = pcut_event)

        # self.r = np.append(isol.t,isol.t_events[0])
        # self.p = np.append(isol.y[0,:],isol.y_events[0][0,0])
        # self.m = np.append(isol.y[1,:],isol.y_events[0][0,1])
        # self.nu = np.append(isol.y[2,:],isol.y_events[0][0,2])

        self.r = isol.t
        self.ysol = isol.y

        self.get_sol()

    def find_yr(self,r):
        idx = (np.abs(self.r - r)).argmin()
        isol = solve_ivp(self.deriv, t_span=[self.r[idx],r], y0 = np.array([self.p[idx], self.m[idx], self.nu[idx]]) \
                        , atol = self.atol, rtol = self.rtol)

        self.pf, self.mf, self.nuf = isol.y[0,-1], isol.y[1,-1], isol.y[2,-1]
        self.rhof = self.eos.rho(self.pf)
        self.gaf = self.eos.ga(self.pf)

    def get_sol(self):
        self.p = self.ysol[0,:]
        self.m = self.ysol[1,:]
        self.nu = self.ysol[2,:]

        nu_R=np.log(1.-2.*G/c**2*self.m[-1]/self.r[-1])-self.nu[-1]
        self.nu=self.nu+nu_R

        # self.rho = np.array([self.eos.rho(p) for p in self.p])
        # self.ga = np.array([self.eos.ga(p) for p in self.p])
        self.rho = np.array(list(map(self.eos.rho, self.p)))
        self.ga = np.array(list(map(self.eos.ga, self.p)))


#=========================================================
'''Example run of this module'''
def test_run():
    import matplotlib.pyplot as plt
    from src.eos.poly import eos_poly

    '''Define EOS'''
    eos = eos_poly(k = 1.e35/(1.e15)**2, n = 1.)
    '''Initialize the TOV solver'''
    tov=solve_tov(eos)
    '''Solve TOV equation'''
    tov.ivp(p0=1.5e35,pcut=1e20)
    '''TOV solutions are now stored in the lists: tov.r, tov.p, tov.rho, tov.m, tov.nu'''

    '''Use a one-step integration to locate the point at r = R/2'''
    tov.find_yr(tov.r[-1]/2)

    '''Uncomment to save solutions to output.dat file'''
    # np.savetxt('output.dat', np.c_[tov.r,tov.p,tov.rho,tov.m,tov.nu], fmt='%1.4e'\
    #             ,header='%10s %10s %10s %10s %10s'%('r (cm)','p (cgs)','rho (cgs)','m (cgs)','nu')\
    #             ,comments='')

    '''Plot the solution'''
    plt.figure(figsize=(8.6,6.4), dpi= 100)
    plt.plot(tov.r, tov.p/tov.p[0], 'k-', label=r'pressure (p_0)')
    plt.plot([tov.r[-1]/2],[tov.pf/tov.p[0]], 'k^')
    plt.plot(tov.r, tov.rho/tov.rho[0], 'r-', label=r'density (rho_0)')
    plt.title(r'Static profiles')
    plt.xlabel(r'$r$ (cm)',fontsize=15)
    plt.ylabel(r'Profiles',fontsize=15)
    plt.yscale('log')
    plt.legend()
    plt.show()
    plt.close()

if __name__ == '__main__':
    test_run()