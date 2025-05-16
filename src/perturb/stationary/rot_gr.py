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

class bvp_rot:
    def __init__(self,bsol):
        self.bsol = bsol
        self.atol_factor = 1.e-8
        self.rtol = 1.e-8
        self.ode_method = 'RK45'

    '''Define ODE'''
    '''w is -bar{omega} in Hartle 1967'''
    def deriv(self,r,y):
        w=y[0]
        dw=y[1]
        h2_p=y[2]
        u2_p=y[3]
        h2_h=y[4]
        u2_h=y[5]

        self.bsol.find_yr(r)
        p, m, nu = G/c**4*self.bsol.pf, G/c**2*self.bsol.mf, self.bsol.nuf
        rho = G/c**2*self.bsol.rhof
        
        elam = 1./(1-2*m/r)
        enu = np.exp(nu)
        dnu = 2*(m+4*np.pi*r**3*p)/r**2*elam
        jj = 1./np.sqrt(elam*enu)
        jjp=-jj*np.pi*8*r*(rho+p)*elam

        dydr = np.empty_like(y)
        '''Hartle 1967 Eq. (43)'''
        dydr[0] = dw
        dydr[1] = np.pi*16*(rho+p)*elam*w - (4./r - np.pi*4*r*(rho+p)*elam )*dw
        '''Hartle 1967 Eqs. (125) & (126)'''
        '''The first two solves for the particular solutions, 
            the other two are for the homogeneous solutions'''
        dydr[2] = -4./r**2*elam/dnu*u2_p + (-dnu + elam/dnu*(np.pi*8*(rho+p)-4*m/r**3))*h2_p \
                + r**3/6*(r*dnu/2-elam/r/dnu)*jj*dw**2 - r**2/3*(r*dnu/2+ elam/r/dnu)*jjp*w**2
        dydr[3] = -dnu*h2_p +(1./r+dnu/2)*(-r**3/3*jjp*w**2+jj/3*r**4*dw**2)
        dydr[4] = -4./r**2*elam/dnu*u2_h + (-dnu + elam/dnu*(np.pi*8*(rho+p)-4*m/r**3))*h2_h
        dydr[5] = -dnu*h2_h
        return dydr

    '''Regular condition at stellar center'''
    def r0_reg(self,r):
        bsol = self.bsol

        bsol.find_yr(r)
        p, m, nu = G/c**4*bsol.pf, G/c**2*bsol.mf, bsol.nuf
        rho = G/c**2*bsol.rhof

        y0 = np.zeros(6)
        y0[0]= 1. + 8./5*np.pi*(rho+p)*r**2
        y0[1]= 16./5*np.pi*(rho+p)*r**2
        y0[2]= r**2
        y0[3]= (-np.pi*2*(rho/3+p) -np.pi*4/3*(rho+p)/np.exp(nu)*y0[0]**2)*r**4
        y0[4]= r**2
        y0[5]= -np.pi*2*(rho/3+p)*r**4
        return y0
    
    '''Integrate ODE'''
    def integrate(self):

        bsol = self.bsol

        r0 = bsol.r[0]
        rf = bsol.r[-1]
        first_step = np.abs(rf-r0)/400

        y0 = self.r0_reg(r0)
        
        yscale = y0*0.
        yscale[0] = 1.
        yscale[1] = 1./bsol.r[-1]
        yscale[2:6] = 1.
        atol = yscale*self.atol_factor

        isol = solve_ivp(self.deriv, t_span=[r0,rf], first_step = first_step, \
                        method = self.ode_method , y0 = y0, atol = atol, rtol = self.rtol)
        self.r = isol.t
        self.w  = isol.y[0,:]
        self.dw  = isol.y[1,:]
        self.h2_p  = isol.y[2,:]
        self.u2_p  = isol.y[3,:]
        self.h2_h  = isol.y[4,:]
        self.u2_h  = isol.y[5,:]
        return

    '''Match with exterior solution to find I and Q'''
    def solve_IQ(self):
        '''Solve the differential equations to get w, dw, h2_p, u2_p, h2_h, u2_h'''
        self.integrate()

        r = self.bsol.r[-1]
        m = G/c**2*self.bsol.m[-1]
        w = self.w[-1]
        dw = self.dw[-1]
        h2_p = self.h2_p[-1]
        u2_p = self.u2_p[-1]
        h2_h = self.h2_h[-1]
        u2_h = self.u2_h[-1]

        '''Moment of inertia: Eq. (47) of Hartle 1967'''
        moi= r**4/(6*w/dw+2*r)

        '''Quadrupole moment: Hartle 1967 Eqs. (139) & (140)'''
        '''JJ is the spin^2 defined in Eq. (47) of Hartle.
            We take a derivative in r so we don't have to solve for Omega_s.
        '''
        z=r/m-1
        JJ=(r**4/6*dw)**2
        Q21=np.sqrt(z**2-1)*((3*z**2-2)/(z**2-1) - 1.5*z*np.log((z+1)/(z-1)) )
        Q22=1.5*(z**2-1)*np.log((z+1)/(z-1)) - (3*z**3-5*z)/(z**2-1)
        '''Matrix problem structure: 
                [a00, a01][A]   [p1]
                [a10, a11][B] = [p2]
        '''
        a00, a01 = Q22, -h2_h
        a10, a11 = 2*m/r/np.sqrt(1-2*m/r)*Q21, -u2_h
        p1 = h2_p - JJ*(1./m/r**3 + 1./r**4)
        p2 = u2_p + JJ/r**4
        A = (a11*p1 - a01*p2)/(a00*a11-a01*a10)
        '''Solving with numpy (removed)'''
        # mat = np.array([[a00,a01],[a10,a11]])
        # vec = np.array([p1,p2])
        # A = np.linalg.solve(mat,vec)[0]
        
        
        '''Determine Q: Eq. (138) of Hartle 1967
            ,but fixed a typo of (16/5 -> 8/5) according to Hartle 1968
            ,and changed a minus sign following Yagi and Yunes 2013. (See references below)
        '''
        Q = -(1./m + 8*A/JJ*m**3/5)
        '''Note: Q = Q_raw/(I Omega_s)^2, where Q_raw is defined by 
            Q_{ij} = Q_raw (n_i n_j - 1/3 delta_{ij})
        '''
        return moi/(G/c**2), Q/(G/c**2)

    '''
    References
        Hartle 1967:
        https://ui.adsabs.harvard.edu/abs/1967ApJ...150.1005H/abstract
        Hartle 1968:
        https://ui.adsabs.harvard.edu/abs/1968ApJ...153..807H/abstract
        Yagi and Yunes 2013:
        https://journals.aps.org/prd/abstract/10.1103/PhysRevD.88.023009
    '''

#=========================================================
'''Example run of this module'''
def test_run():
    from src.eos.poly import eos_poly
    from src.static.tov import solve_tov
    
    return

if __name__ == '__main__':
    test_run()