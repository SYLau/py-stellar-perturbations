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

class bvp_love:
    def __init__(self,bsol):
        self.bsol = bsol
        self.atol_factor = 1.e-8
        self.rtol = 1.e-8
        self.ode_method = 'RK45'

    '''Define ODE'''
    def deriv(self,r,y):
        self.bsol.find_yr(r)
        p, m, nu = G/c**4*self.bsol.pf, G/c**2*self.bsol.mf, self.bsol.nuf
        rho, ga = G/c**2*self.bsol.rhof, self.bsol.gaf

        ell = 2.
        elam = 1./(1-2*m/r)
        cs2 = ga*p/(rho+p)
        dnu = 2*(m+4*np.pi*r**3*p)/r**2*elam
        Q = np.pi*4*elam*(5.*rho+9.*p+(rho+p)/cs2)-ell*(ell+1)*elam/r**2-dnu**2
        '''Eq. (4.7) of Chan 2015'''
        dydr = np.empty_like(y)
        dydr[0]=-(y[0]**2+y[0]*elam*(1 + np.pi*4*r**2*(p-rho))+r**2*Q)/r

        return dydr
    
    '''Integrate ODE'''
    def integrate(self):

        bsol = self.bsol

        r0 = bsol.r[0]
        rf = bsol.r[-1]
        first_step = np.abs(rf-r0)/400

        y0 = np.array([2.])
        
        atol = np.ones(1)*self.atol_factor

        isol = solve_ivp(self.deriv, t_span=[r0,rf], first_step = first_step, \
                        method = self.ode_method, y0 = y0, atol = atol, rtol = self.rtol)
        self.r = isol.t
        self.y  = isol.y[0,:]

    '''Match with exterior solution to find k2'''
    def solve_Love(self):
        '''Solve the differential equations to get y'''
        self.integrate()

        r = self.bsol.r[-1]
        m = G/c**2*self.bsol.m[-1]
        rho = G/c**2*self.bsol.rho[-1]

        '''Junction condition for the possible surface density jump. Eq. (4.6) of Chan 2015'''
        y = self.y[-1] - np.pi*4*r**3/m*rho

        Co = m/r
        if (Co>3.e-3):
            '''Eq. (23) of Hinderer 2008. 
                Eq. (4.5) of Chan 2015 is the same equation but contains 2 typos
            '''
            t1 = 8./5*Co**5*(1-2*Co)**2*(2*Co*(y-1)+2-y)
            t2 = 2*Co*(4*(y+1)*Co**4+(6*y-4)*Co**3+(26-22*y)*Co**2 \
                +3*Co*(5*y-8)-3*y+6)
            t3 = 3*(1-2*Co)**2*(2*Co*(y-1)-y+2)*np.log(1-2*Co)
            k2 = t1/(t2+t3)
        else:
            '''Eq. (6.25) of Chan 2016'''
            t1 = (2 - y)/2/(y + 3)
            t2 = 5*Co*(y**2 + 2*y - 6)/2/(y + 3)**2
            t3 = 5*Co**2 * (11*y**3 + 66*y**2 + 52*y - 204)/14/(y + 3)**3
            k2 = t1 + t2 + t3
        
        return k2


    '''
    References
        Hinderer 2008:
        https://iopscience.iop.org/article/10.1086/533487
        Chan 2015:
        https://journals.aps.org/prd/abstract/10.1103/PhysRevD.91.044017
        Chan 2016:
        https://journals.aps.org/prd/abstract/10.1103/PhysRevD.93.024033
    '''

#=========================================================
'''Example run of this module'''
def test_run():
    from src.eos.poly import eos_poly
    from src.static.tov import solve_tov
    
    return

if __name__ == '__main__':
    test_run()