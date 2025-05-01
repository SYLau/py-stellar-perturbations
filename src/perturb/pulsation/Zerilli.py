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

class ode_Zerilli:
    def __init__(self,ell,w,r0,m0):
        self.ell = ell
        self.w = w
        self.r0 = r0
        self.m0 = m0

    def deriv(self,r,z):
        ell = self.ell
        wc2=(self.w/c)**2
        m0 = G/c**2*self.m0

        zc = z[0:2]+1j*z[2:4]

        nn=(ell-1)*(ell+2)/2

        drt=1./(1-2*m0/r)
        b1=(1.-2*m0/r)/(r**3 * (nn*r + 3 * m0)**2)
        b2=2*nn**2*(nn+1)*r**3
        b3=6*nn**2*m0*r**2
        b4=18*nn*m0**2*r
        b5=18*m0**3
        Vz=b1*(b2+b3+b4+b5)

        deriv = np.zeros(2,dtype=np.complex128)
        deriv[0] = zc[1]
        deriv[1] = (Vz-wc2)*drt**2 * zc[0] + (-2*m0/r**2*drt) * zc[1]

        dzdr = np.zeros(4)
        dzdr[0:2] = deriv.real
        dzdr[2:4] = deriv.imag
        return dzdr


    def surfbc(self,H0,K):
        ell = self.ell
        wc2=(self.w/c)**2
        r0 = self.r0
        m0 = G/c**2*self.m0

        tcf = (1. - 2*m0/r0)**(-1)

        n = (ell-1)*(ell+2)/2
        a = -(n*r0 + 3*m0)/(wc2*r0**2 - (n+1)*m0/r0)
        b = (n*r0*(r0-2*m0) - wc2*r0**4 + m0*(r0-3*m0))/(r0-2*m0)/(wc2*r0**2 - (n+1)*m0/r0) #Typo in LD 1983. This mistake caused ~15% error in f-mode
        gg = (n*(n+1)*r0**2 + 3*n*m0*r0 + 6*m0**2)/r0**2/(n*r0 + 3 * m0)
        ll = 1.
        hh = (-n*r0**2 + 3*n*m0*r0 + 3*m0**2)/(r0-2*m0)/(n*r0 + 3 * m0)
        kk = -r0**2/(r0 - 2*m0)
        det = gg * kk - hh * ll

        yc = np.zeros(2, dtype=np.complex128)
        yc[0] = (-a*ll * H0 + (kk - b * ll) * K )/det
        yc[1] = (gg *a * H0 + (-hh + b * gg) * K )/det*tcf
        return yc
    
    def infbc(self,r,zc):
        ell = self.ell
        w=(self.w/c)
        m0 = G/c**2*self.m0
        
        rt = r + 2*m0*np.log(r/(2*m0) - 1)

        nn = (ell-1)*(ell+2)/2
        drt = 1./(1-2*m0/r)

        a = np.zeros(3, dtype=np.complex128)
        a[0] = 1.
        a[1] = -1j*(nn + 1.)/w*a[0]
        # A typo not reported in a2 until Lu and Suen 2011 \
        # https://iopscience.iop.org/article/10.1088/1674-1056/20/4/040401
        a[2] = -1./w**2*(nn*(nn+1.)/2 - 1j*1.5*m0*w*(1 + 2./nn))*a[0]
        ca = np.conj(a)

        z11 = np.exp(-1j*w*rt)*(a[0] + a[1]/r + a[2]/r**2)
        z12 = np.exp(1j*w*rt)*(ca[0] + ca[1]/r + ca[2]/r**2)
        z21 = -1j*w*z11 + np.exp(-1j*w*rt)*(-a[1]/r**2 -2*a[2]/r**3)/drt
        z22 = 1j*w*z12 + np.exp(1j*w*rt)*(-ca[1]/r**2 -2*ca[2]/r**3)/drt

        det = z11*z22 - z12*z22

        beta = (zc[0]*z22 - z12*zc[1]/drt)/det
        gamma = (zc[1]*z11/drt - z21*zc[0])/det

        return beta, gamma