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

class bvp_love_solid:
    def __init__(self,bsol,eos_shear):
        self.bsol = bsol
        self.atol_factor = 1.e-8
        self.rtol = 1.e-8
        self.ode_method = 'RK45'
        self.shear = eos_shear

    '''Define ODE'''
    ''' Eqs. 25-30 of Lau 2019.
        A reformulation of Penner 2011, and fixed multiple mistakes in their equations.
        The equations are consistent with that in Gittins 2020'''
    def deriv(self,r,y):
        self.bsol.find_yr(r)
        p, m, nu = G/c**4*self.bsol.pf, G/c**2*self.bsol.mf, self.bsol.nuf
        rho, ga = G/c**2*self.bsol.rhof, self.bsol.gaf
        mu = G/c**4*self.shear.mu(self.bsol.pf)

        ell = 2.
        elam = 1./(1-2*m/r)
        cs2 = ga*p/(rho+p)
        dnu = 2*(m+4*np.pi*r**3*p)/r**2*elam
        dlam = 2*(-m+4*np.pi*r**3*rho)/r**2*elam
        ddnu=dnu*(dlam-2/r) +np.pi*4*elam*(rho*2+p*6-r*dnu*(rho+p))
        dpr=-(rho+p)/2*dnu
        a1,a2,a3 = mu, cs2*(rho+p)-mu*2/3, cs2*(rho+p)+mu*4/3

        W, Zr, V, Zt, H0, J = y

        '''Algebraic equation for K'''
        ''' Note that in Eq. 34 of Lau 2019, coef_W is zero 
            (no mistakes, just did not notice all the terms cancel out).
            See also Eq. 36 of Dong 2024'''
        # coef_W = (dnu**2+dnu*dlam+np.pi*16*elam*dpr*r)/elam/(ell+2)/(ell-1)
        coef_Zr = -np.pi*16*r**2/(ell+2)/(ell-1)
        coef_Zt = -np.pi*16*(2+r*dnu)*r**2/(ell+2)/(ell-1)
        coef_H0 = (ell*(ell+1)*elam-2+(r*dnu)**2)/elam/(ell+2)/(ell-1)
        coef_J = r**2*dnu/elam/(ell+2)/(ell-1)
        K = coef_Zr*Zr + coef_Zt*Zt + coef_H0*H0 + coef_J*J
        '''Algebraic equation for H2'''
        H2 = H0+32*np.pi*a1*V

        dydr = np.empty_like(y)
        '''dW equation'''
        coef_W = (1-2*a2/a3-r*dlam/2)/r
        coef_Zr = -r/a3
        coef_V = a2/a3*ell*(ell+1)/r
        coef_K = -a2/a3*r
        coef_H2 = -r/2
        dydr[0]= coef_W*W + coef_Zr*Zr + coef_V*V + coef_K*K + coef_H2*H2

        '''dZr equation (except the dH0 term for now)'''
        coef_W = (dpr*(r*ddnu/dnu-r*dlam/2-2) - 4/r*a1/a3*(a3+2*a2))/r**2
        coef_Zr = -(r*dnu/2+4*a1/a3)/r
        coef_V = (dpr+2/r*a1/a3*(a3+2*a2))*ell*(ell+1)/r**2
        coef_Zt = ell*(ell+1)*elam/r
        coef_K = -(dpr+2/r*a1/a3*(a3+2*a2))
        coef_H2 = -dpr/2
        dydr[1]= coef_W*W + coef_Zr*Zr + coef_V*V + coef_Zt*Zt + coef_K*K + coef_H2*H2

        '''dV equation'''
        coef_W = -elam/r
        coef_V = 2./r
        coef_Zt = -r*elam/a1
        dydr[2]= coef_W*W + coef_V*V + coef_Zt*Zt

        '''dZt equation'''
        coef_W= (dpr+2/r*a1/a3*(a3+2*a2))/r**2
        coef_Zr= -a2/a3/r
        coef_V = -(-2*a1/r+2*a1*(1+a2/a3)*ell*(ell+1)/r)/r**2
        coef_Zt= -(r*dlam/2+r*dnu/2+3)/r
        coef_H0 = (rho+p)/2/r
        coef_K = a1/a3*(a3+2*a2)/r
        dydr[3]= coef_W*W + coef_Zr*Zr + coef_V*V + coef_Zt*Zt + coef_H0*H0 + coef_K*K

        '''dH0 equation'''
        coef_W=(dlam+dnu)/r**2
        coef_V=-np.pi*16*dnu*a1
        coef_J= 1.
        dydr[4]= coef_W*W + coef_V*V + coef_J*J

        '''dJ equation'''
        coef_W = np.pi*32*elam/r**2*a1/a3*(a3+2*a2) -1.5/r**2*dnu*(dlam+dnu)
        coef_Zr = -np.pi*8*elam/a3*(a3+2*a2)
        coef_V = -np.pi*8/r**2*((rho+p)*elam*ell*(ell+1)+2*a1/a3*(a3+2*a2)*elam*ell*(ell+1)\
                +4*a1*(1-elam)-2*a1*(r*dnu)**2 )
        coef_Zt = -np.pi*16*elam*(r*dnu)
        coef_H0 = (ell*(ell+1)*elam+2*(elam-1)-r*(dlam/2+5*dnu/2)+(r*dnu)**2)/r**2
        coef_J = (r/2*(dlam-dnu)-2)/r
        coef_K = (dlam+dnu)/r+np.pi*16*elam*a1/a3*(a3+2*a2)
        dydr[5]= coef_W*W + coef_Zr*Zr + coef_V*V + coef_Zt*Zt + coef_H0*H0 + coef_J*J + coef_K*K
        '''Final step: Add the dH0 term to dZr'''
        dydr[1] += (rho+p)/2*dydr[4]

        return dydr
    
    '''Integrate ODE'''
    def integrate(self):

        bsol = self.bsol

        ''' r0 too close to stellar center causes error in the integration with y0_true.
            The resulting Zr and Zt at r=R are not small enough.
            It is probably because it requires a much higher accuracy near the center.
            Decreasing rtol helps. But the better solution is to set r0 larger.'''
        # r0 = bsol.r[0]
        r0 = bsol.r[-1]*1.e-3
        rf = bsol.r[-1]
        first_step = np.abs(rf-r0)/400

        y0 = np.zeros(shape=(6,3))
        ysol = np.empty_like(y0)

        yscale = np.array([1., 1./rf**4, 1., 1./rf**4, 1./rf**2, 1./rf**3])
        atol = yscale*self.atol_factor
        '''Integrate the 3 independent regular solutions with unknown coefficeints'''
        for idx in range(3):
            y0[:,idx] = self._regBC(r0, i = idx)

            isol = solve_ivp(self.deriv, t_span=[r0,rf], first_step = first_step, \
                    method = self.ode_method, y0 = y0[:,idx], atol = atol, rtol = self.rtol)
            ysol[:,idx] = isol.y[:,-1]
        '''Solve surface BC to determine the coefficients of each independent regular solution'''
        coef = self._surBC(ysol)
        '''Use the correct coefficients to construct the true solution'''
        y0_true = coef[0]*y0[:,0]+coef[1]*y0[:,1]+coef[2]*y0[:,2]
        isol = solve_ivp(self.deriv, t_span=[r0,rf], first_step = first_step, \
                    method = self.ode_method, y0 = y0_true, atol = atol, rtol = self.rtol)
        
        self.r = isol.t
        self.W  = isol.y[0,:]
        self.Zr  = isol.y[1,:]
        self.V  = isol.y[2,:]
        self.Zt  = isol.y[3,:]
        self.H0  = isol.y[4,:]
        self.J  = isol.y[5,:]

        ''' Below shows that the values of Zr and Zt at r = R are not as close to zero as
            the one obtained directly from ysol. This is due to the numerical error.'''
        # print(coef[0]*ysol[1,0]+coef[1]*ysol[1,1]+coef[2]*ysol[1,2] \
        #       ,coef[0]*ysol[3,0]+coef[1]*ysol[3,1]+coef[2]*ysol[3,2])
        # print(self.Zr[-1],self.Zt[-1])

    '''Match with exterior solution to find k2'''
    def solve_Love(self):
        '''Solve the differential equations to get H0 and J'''
        self.integrate()

        r = self.bsol.r[-1]
        m = G/c**2*self.bsol.m[-1]
        '''Note that in vacuum, J = dH0.
            dH0 is not continuous across the stellar surface when rho or mu are discontinous.
            J is continous. That's why we replace dH0 with J as a dependent variable.
            See Lau 2019.'''
        y = r*self.J[-1]/self.H0[-1]

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
    
    '''Regular condition at stellar center'''
    def _regBC(self,r, i):
        bsol = self.bsol

        bsol.find_yr(r)
        p = G/c**4*bsol.pf
        rho, ga = G/c**2*self.bsol.rhof, self.bsol.gaf
        mu = G/c**4*self.shear.mu(bsol.pf)
        chic = self.shear.chi(bsol.pf)
        mu2nd = -chic*np.pi*2/3*(p+rho)*(p*3+rho)/2 #Eq. A4 of Perot 2022
        
        ell = 2.
        cs2 = ga*p/(rho+p)

        if i == 0:
            K0, V2, V0= p+rho, 0., 0.
        elif i == 1:
            K0, V2, V0= 0., 1./r**2, 0.
        elif i == 2:
            K0, V2, V0= 0., 0., 1.

        W0= ell*V0
        H00= K0-np.pi*32*mu*V0  # Eq. 48a of Finn 1990 missing the term (32 pi mu V0)

        term1= -(1+3*cs2)/2*(rho+p)
        term2= 4./9*(ell*3*p*(ell*p+(-ell*2+6.)*mu)-mu2nd*9*(ell-1.) \
        +rho*(np.pi*2*(ell*3*p*(2.-cs2)+mu*(ell*(ell*3-4.)+9)) +np.pi*3*ell*rho*(1.-cs2*2) ))
        term3= (ell+1.)/3*(mu*(ell-6.)+ell*3*(rho+p)*cs2)  # Eq. 48c of Finn 1990 missing (l+1)
        term0= -(ell+9.)/3*mu -(ell+3.)*(rho+p)*cs2
        W2=(term1*K0 + term2*V0 + term3*V2)/(-term0) # Negative Signs account for notation in Penner 2011

        Z20 = -2*mu*(ell-1)*V0
        Z22 = -mu*(ell*V2+W2-np.pi*8/3*(ell-2)*rho*V0)-2*mu2nd*(ell-1)*V0

        term1=np.pi*4*(ell+3)*p-np.pi*4/3*(ell*(2*ell+3)+3)*rho-36*np.pi*cs2*(p+rho)
        term2=32./3*np.pi**2*ell*(p*3*(p*(3.+1./cs2)-mu*8) \
        +rho*(p*2*(2./cs2+5.-cs2*3)+mu*8*(ell+1.)+rho*(1./cs2+1.-cs2*6) ) )
        term3=-np.pi*8*(ell+3.)*(rho+p)*(1.+cs2*3)
        term4=np.pi*8*ell*(ell+1.)*(rho+p)*(1.+cs2*3)
        term0=ell*4+6.
        H02=(term1*K0 + term2*V0 + term3*W2 + term4*V2)/(-term0)

        Z10=-2*mu*ell*(ell-1)*V0
        a2=cs2*(rho+p)-2*mu/3
        a3=cs2*(rho+p)+4*mu/3
        term1=-(8*np.pi/3*ell*rho*a3+16*np.pi*(a3+2*a2)*mu)-2*mu2nd*ell*(ell-1)
        term2=-(a3+2*a2)/2
        term3=-(ell*a3+a3+2*a2)
        term4=ell*(ell+1)*a2
        Z12=term1*V0+term2*H00+term3*W2+term4*V2

        J0=ell*(K0-np.pi*8*(rho+p+4*mu)*V0)
        p2=-np.pi*4/3*(rho+3*p)*(rho+p)
        rho2=p2/cs2
        J2=(ell+2)*H02 + np.pi*8/3*(np.pi*16*mu*(rho+3*p)*V0-3*ell*(rho2+p2)*V0-3*(rho+p)*W2)

        y0 = np.zeros(6)
        y0[0]= r**ell*(W0+ r**2*W2)
        y0[1]= r**(ell-2)*(Z10+ r**2*Z12)
        y0[2]= r**ell*(V0+ r**2*V2)
        y0[3]= r**(ell-2)*(Z20+ r**2*Z22)
        y0[4]= r**ell*(H00+r**2*H02)
        y0[5]= r**(ell-1)*(J0+r**2*J2)

        return y0
    
    '''Surface boundary conditions: Zr = Zt = 0'''
    def _surBC(self, ysol):
        '''Matrix problem structure: 
                [Zr1, Zr2][A]   [-Zr0]
                [Zt1, Zt2][B] = [-Zr1]
        '''
        Zr1, Zr2 = ysol[1,1], ysol[1,2]
        Zt1, Zt2 = ysol[3,1], ysol[3,2]
        p1 = -ysol[1,0]
        p2 = -ysol[3,0]
        A = (Zt2*p1 - Zr2*p2)/(Zr1*Zt2-Zr2*Zt1)
        B = (Zr1*p2 - Zt1*p1)/(Zr1*Zt2-Zr2*Zt1)
        return np.array([1., A, B])

    '''
    References
        Finn 1990:
        https://ui.adsabs.harvard.edu/abs/1990MNRAS.245...82F
        Hinderer 2008:
        https://iopscience.iop.org/article/10.1086/533487
        Penner 2011:
        https://journals.aps.org/prd/abstract/10.1103/PhysRevD.84.103006
        Chan 2016:
        https://journals.aps.org/prd/abstract/10.1103/PhysRevD.93.024033
        Lau 2019:
        https://journals.aps.org/prd/abstract/10.1103/PhysRevD.99.023018
        Gittins 2020:
        https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.103025
        Perot 2022:
        https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.023012
        Dong 2024:
        https://link.springer.com/article/10.1007/s10714-024-03302-z
    '''

#=========================================================
'''Example run of this module'''
def test_run():
    from src.eos.poly import eos_poly
    from src.static.tov import solve_tov
    
    return

if __name__ == '__main__':
    test_run()