'''Locate the working directory'''
if __name__ == '__main__':    
    import sys
    from _path import workdir
    sys.path.append(workdir)
    print(sys.path[-1])

#=========================================================
'''Main content of the module'''
import numpy as np
from src.util.constants import c, MeV, hbar

'''Analytical EOS for quark matter: Modified MIT Bag model'''
'''
    Eq. (3.1) of Alford 2005. This paper uses a unit system with h_bar = c = 1.
    All the dimensions are expressed in terms of MeV: [M]=[1/L]=[1/T]=[MeV]
'''
class eos_mmit:
    def __init__(self,a4,a2,B):
        self.a4 = a4            # Dimensionless
        self.a2 = a2            # [MeV^2]
        self.B = B              # [MeV^4]
        self.unitconv = (hbar*c)**3/MeV**4 #erg cm^{-3} to MeV^4
        self.rho_min = (4*B + 3*a2/4/np.pi**2/a4*(a2+np.sqrt(a2**2 + 16*np.pi**2/3*a4*B)))\
                        /(self.unitconv*c**2) #(MeV)^4 to g cm^{-3}

    def rho(self, p):
        a4, a2, B= self.a4, self.a2, self.B
    
        p_MeV4 = p*self.unitconv
        rho_MeV4 = (3*p_MeV4+4*B) \
                    + 3.*a2/4/np.pi**2/a4*(a2+np.sqrt(a2**2+16*np.pi**2/3*a4*(p_MeV4+B)))

        return rho_MeV4/self.unitconv/c**2

    def p(self, rho):
        a4, a2, B= self.a4, self.a2, self.B
        if rho < self.rho_min:
            raise ValueError('eos_mmit eos_p p < 0. Use a larger rho.')

        rho_MeV4= rho*self.unitconv*c**2
        p_MeV4= (rho_MeV4-4*B)/3 \
                - a2/12/np.pi**2/a4*(a2+np.sqrt(a2**2+16*np.pi**2*a4*(rho_MeV4-B)))

        return p_MeV4/self.unitconv
    
    def ga(self, p):
        a4, a2, B= self.a4, self.a2, self.B
        fa=3./4/np.pi**2

        p_MeV4= p*self.unitconv
        muq2= a2/a4/2+np.sqrt(a2**2+4.*a4*(B+p_MeV4)/fa)/2/a4
        rho_MeV4= 3.*fa*a4*muq2**2-fa*a2*muq2+B

        return (rho_MeV4+p_MeV4)/p_MeV4*(a4*2*muq2-a2)/(a4*6*muq2-a2)
    
'''
References
    Alford 2005:
    https://iopscience.iop.org/article/10.1086/430902
    Pereira 2018:
    https://iopscience.iop.org/article/10.3847/1538-4357/aabfbf
'''

#=========================================================
'''Example run of this module'''
def test_run():
    
    '''Initialize class: provide modified MIT bag model parameters'''
    eos = eos_mmit(a4 = 1., a2 = 100.**2, B = 145.**4)

    '''Compute p(rho), rho(p), and gamma(p)'''
    print('rho_min = %1.6e'%eos.rho_min)
    print('p(rho) = %1.6e'%eos.p(4.5e14))
    print('rho(p) = %1.6e'%eos.rho(eos.p(4.5e14)))
    print('ga(p) = %1.6e'%eos.ga(eos.p(4.5e14)))

if __name__ == '__main__':
    test_run()