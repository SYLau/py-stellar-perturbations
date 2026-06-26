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
    Lau 2017:
    https://journals.aps.org/prd/abstract/10.1103/PhysRevD.95.101302
    Pereira 2018:
    https://iopscience.iop.org/article/10.3847/1538-4357/aabfbf
'''

'''Subclass for the CCSC phase under the Modified MIT Bag model'''
class eos_ccsc(eos_mmit):
    def __init__(self,a4,a2,B,gap):
        # Initialize the parent class eos_mmit
        super().__init__(a4,a2,B)
        self.gap = gap              # [MeV]

    '''Shear modulus of CCSC phase. Eq. 105 of Mannarelli 2007'''
    def mu(self, p):
        a4, a2, B= self.a4, self.a2, self.B
        fa=3./4/np.pi**2

        p_MeV4= p*self.unitconv
        muq2= a2/a4/2+np.sqrt(a2**2+4.*a4*(B+p_MeV4)/fa)/2/a4
        return 2.47*(self.gap/10)**2 * muq2/(400.)**2 *(1.e39*MeV) #MeV fm^{-3} to erg cm^{-3}
    
    '''dmu/dp of CCSC phase. Used in Eq. A4 of Perot 2022'''
    def chi(self, p):
        a4, a2, B= self.a4, self.a2, self.B
        fa=3./4/np.pi**2

        p_MeV4= p*self.unitconv
        muq2= a2/a4/2+np.sqrt(a2**2+4.*a4*(B+p_MeV4)/fa)/2/a4
        '''dp/d(mu_q^2) and dmu/d(mu_q^2)'''
        dp_MeV4= 2.*fa*a4*muq2-fa*a2
        dmu_MeV4= 2.47*(self.gap/10)**2/(400.)**2 *(1.e39*(hbar*c)**3/MeV**3) #MeV fm^{-3} to MeV^4
        return dmu_MeV4/dp_MeV4
    
    '''
    Mannarelli 2007
    https://journals.aps.org/prd/abstract/10.1103/PhysRevD.76.074026
    Perot 2022
    https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.023012
    '''

#=========================================================
'''Example run of this module'''
def test_run():
    
    '''Initialize class: provide modified MIT bag model parameters'''
    eos = eos_mmit(a4 = 1., a2 = 100.**2, B = 145.**4)

    '''MMIT: Compute p(rho), rho(p), and gamma(p)'''
    print('Modified Bag model ============================')
    print('rho_min = %1.6e'%eos.rho_min)
    print('p(rho) = %1.6e'%eos.p(4.5e14))
    print('rho(p) = %1.6e'%eos.rho(eos.p(4.5e14)))
    print('ga(p) = %1.6e'%eos.ga(eos.p(4.5e14)))

    '''Initialize subclass: the CCSC phasae'''
    eos_solid = eos_ccsc(a4 = 1., a2 = 100.**2, B = 145.**4, gap = 15.)
    print('CCSC phase ====================================')
    print('rho_min = %1.6e'%eos_solid.rho_min)
    print('p(rho) = %1.6e'%eos_solid.p(4.5e14))
    print('rho(p) = %1.6e'%eos_solid.rho(eos_solid.p(4.5e14)))
    print('ga(p) = %1.6e'%eos_solid.ga(eos_solid.p(4.5e14)))
    print('mu(p) = %1.6e'%eos_solid.mu(eos_solid.p(4.5e14)))
    print('chi(p) = %1.6e'%eos_solid.chi(eos_solid.p(4.5e14)))

if __name__ == '__main__':
    test_run()
