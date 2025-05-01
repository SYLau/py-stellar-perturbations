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

class ode_LD_aniso:
    def __init__(self,ell,w,bsol):
        self.ell = ell
        self.w = w
        self.bsol = bsol

    def deriv(self,r,y):
        ell = self.ell
        wc2 = self.w**2/c**2

        self.bsol.find_yr(r)
        p, m, nu = G/c**4*self.bsol.pf, G/c**2*self.bsol.mf, self.bsol.nuf
        rho, ga, sch = G/c**2*self.bsol.rhof, self.bsol.gaf, self.bsol.schf

        s, dsdr = G/c**4*self.bsol.sf, G/c**4*self.bsol.dsdrf
        sb= s/(rho+p)

        elam = 1./(1-2*m/r)
        # dlam = 2.*elam/r*(4*np.pi*r**2*rho - m/r)
        enu = np.exp(nu)
        dnu = 2*(m+4*np.pi*r**3*p)/r**2/(1-2*m/r)
        
        yc = y[0:4]+1j*y[4:8]
        H1, K, W, X = yc
        H0, Z, V = self.alg_all(r,yc)

        deriv = np.zeros(4,dtype=np.complex128)
        '''dH1 equation'''
        coef_H1= np.pi*4*(rho-p)*elam*r - 2*m/r**2*elam - (ell+1)/r
        coef_K= elam/r
        coef_H0= elam/r
        coef_V= -16*np.pi*(rho+p)/r*elam*(1-sb)

        deriv[0] = coef_H1*H1 + coef_K*K +coef_H0*H0 + coef_V*V

        '''dK equation'''

        coef_H1=ell*(ell+1)/2/r
        coef_K=dnu/2-(ell+1)/r
        coef_W=-np.pi*8*(rho+p)*np.sqrt(elam)/r
        coef_H0=1/r

        deriv[1] = coef_H1*H1 + coef_K*K +coef_W*W +coef_H0*H0

        '''dW equation'''
        coef_K= r*np.sqrt(elam)*(1-sb)
        coef_W= -(ell+1)/r +2*sb/r
        coef_X= r*np.sqrt(elam/enu)/ga/p
        coef_H0= r*np.sqrt(elam)/2
        coef_V= -ell*(ell+1)/r*np.sqrt(elam)*(1-sb)

        deriv[2] = coef_K*K +coef_W*W  +coef_X*X +coef_H0*H0 +coef_V*V

        '''dX equation'''
        dp= -(rho+p)/2*dnu -2*s/r
        t1= (rho+p)/2*np.sqrt(enu)
        t2= elam/r**4*(7*m**2 -4*r*m -8*np.pi*r**3*m*rho -16*np.pi**2*r**6*p**2 \
                    -np.pi*4*(p-rho)*r**4 - 8*np.pi*s/elam*r**4) \
            - ( (6/r**2-2*dnu/r)*sb -2/r**2*r*dsdr/(rho+p) - 4/r**2*sb**2)/elam
        coef_H1= t1*(r*wc2/enu + ell*(ell+1)/2/r*(1-2*sb))
        coef_K= t1*( (1.5-2*sb)*dnu - (1-6*sb)/r -4*sb**2/r)
        coef_W= -t1*2/r*np.sqrt(elam)*(4*np.pi*(rho+p) +wc2/enu - t2)
        coef_X=-1/r*(ell-2*(rho+p)/ga/p*sb)
        coef_H0=t1*(1/r-dnu/2)
        coef_V=ell*(ell+1)*np.sqrt(enu)*dp/r**2*(1-sb)
        coef_Z=2*np.sqrt(enu)/r

        deriv[3] = coef_H1*H1 + coef_K*K +coef_W*W  +coef_X*X +coef_H0*H0 +coef_V*V +coef_Z*Z

        dydr = np.zeros(8)
        dydr[0:4] = deriv.real
        dydr[4:8] = deriv.imag

        return dydr
    
    def alg_all(self,r,yc,find_yr=False):
        if find_yr:
            self.bsol.find_yr(r)
        H0, Z, V = self.alg_H0(r,yc), self.alg_Z(r,yc), self.alg_V(r,yc)
        return H0, Z, V

    def alg_H0(self,r,yc,find_yr=False):
        ell=self.ell
        wc2 = self.w**2/c**2
        if find_yr:
            self.bsol.find_yr(r)
        p, m, nu = G/c**4*self.bsol.pf, G/c**2*self.bsol.mf, self.bsol.nuf
        rho = G/c**2*self.bsol.rhof

        s = G/c**4*self.bsol.sf
        sb= s/(rho+p)

        elam = 1./(1-2*m/r)
        enu = np.exp(nu)
        
        H1, K, W, X = yc

        t1=3*m+(ell+2)*(ell-1)/2*r+4*np.pi*r**3*p
        coef_H1= -(ell*(ell+1)/2*(m+4*np.pi*r**3*p) -wc2/enu *r**3/elam)/t1
        coef_K= ((ell+2)*(ell-1)/2*r -wc2/enu*r**3 -elam/r*(m+4*np.pi*r**3*p)*(3*m-r+4*np.pi*r**3*p) )/t1
        coef_W= -16*np.pi*r/np.sqrt(elam)*(rho+p)*sb/t1
        coef_X= 8*np.pi*r**3/np.sqrt(enu)/t1
        return coef_H1*H1 + coef_K*K + coef_W*W + coef_X*X

    def alg_Z(self,r,yc,find_yr=False):
        if find_yr:
            self.bsol.find_yr(r)
        p, m, nu = G/c**4*self.bsol.pf, G/c**2*self.bsol.mf, self.bsol.nuf
        rho, ga, sch = G/c**2*self.bsol.rhof, self.bsol.gaf, self.bsol.schf

        s = G/c**4*self.bsol.sf
        dsdp, dsdrho, dsdmu = self.bsol.dsdpf, 1./c**2*self.bsol.dsdrhof, G/c**4*self.bsol.dsdmuf

        elam = 1./(1-2*m/r)
        enu = np.exp(nu)
        dnu = 2*(m+4*np.pi*r**3*p)/r**2/(1-2*m/r)

        cs2 = p/(rho+p)*ga
        dp = -(rho+p)/2*dnu - 2*s/r
        
        H1, K, W, X = yc
        H0 = self.alg_H0(r,yc)

        coef_W= -dsdp*dp/r/np.sqrt(elam) -dsdrho*((rho+p)*sch/r+dp/r/cs2)/np.sqrt(elam)
        coef_X= -dsdp/np.sqrt(enu) -dsdrho/cs2/np.sqrt(enu)
        coef_H0= -dsdmu/elam

        return coef_W*W + coef_X*X + coef_H0*H0

    def alg_V(self,r,yc,find_yr=False):
        wc2 = self.w**2/c**2
        if find_yr:
            self.bsol.find_yr(r)
        p, m, nu = G/c**4*self.bsol.pf, G/c**2*self.bsol.mf, self.bsol.nuf
        rho = G/c**2*self.bsol.rhof

        s = G/c**4*self.bsol.sf
        sb= s/(rho+p)

        elam = 1./(1-2*m/r)
        enu = np.exp(nu)
        dnu = 2*(m+4*np.pi*r**3*p)/r**2/(1-2*m/r)

        dp = -(rho+p)/2*dnu - 2*s/r
        
        H1, K, W, X = yc
        H0 = self.alg_H0(r,yc)
        Z = self.alg_Z(r,yc)

        t1 = wc2/np.sqrt(enu)*(rho+p)*(1-sb)

        coef_W=dp/r*np.sqrt(enu/elam)/t1
        coef_X=1/t1
        coef_H0=-(rho+p)/2*np.sqrt(enu)/t1
        coef_Z=np.sqrt(enu)/t1

        return coef_W*W + coef_X*X + coef_H0*H0 + coef_Z*Z

    def centerbc(self):
        ell = self.ell
        wc2 = self.w**2/c**2
        self.bsol.find_yr(self.bsol.r[0])
        p, nu = G/c**4*self.bsol.pf, self.bsol.nuf
        rho = G/c**2*self.bsol.rhof

        enu = np.exp(nu)

        s2 = G/c**4*self.bsol.s2f
        
        def find_H10_W0(K0,W0):
            H10 = (2*ell*K0 + 16*np.pi*(p+rho)*W0)/(ell*(ell+1))
            term1 = np.pi*4/3*(3*p+rho)-wc2/enu/ell + 2*s2/(rho+p)
            X0 = (p+rho)*enu**0.5 *(K0/2 + term1 * W0)
            return H10, X0
        
        yc1, yc2 = np.zeros(4), np.zeros(4)

        K0, W0 = p+rho, 1.
        H10, X0 = find_H10_W0(K0, W0)
        yc1 = np.array([H10,K0,W0,X0])

        K0, W0 = -(p+rho), 1.
        H10, X0 = find_H10_W0(K0, W0)
        yc2 = np.array([H10,K0,W0,X0])

        return np.array([yc1, yc2])
    
    def surfbc(self):
        R=self.bsol.r[-1]

        yc1, yc2, yc3 = np.zeros(4), np.zeros(4), np.zeros(4)
        yc1[0] = 1./R**2
        yc2[1] = 1./R**2
        yc3[2] = 1.
        return np.array([yc1, yc2, yc3])


#=========================================================
'''Example run of this module'''
def test_run():
    from src.eos.poly import eos_poly
    from src.static.tov import solve_tov

    return

if __name__ == '__main__':
    test_run()