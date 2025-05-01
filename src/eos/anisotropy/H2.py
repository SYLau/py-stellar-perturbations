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

class sigma_H2:
    def __init__(self,beta):
        self.beta = beta

    def s(self, r,p,m,rho,ga0,sch):
        mu=2*G/c**2*m/r
        return self.beta*p*mu**2
    
    def dsdp(self, r,p,m,rho,ga0,sch):
        mu=2*G/c**2*m/r
        return self.beta*mu**2
    
    def dsdrho(self, r,p,m,rho,ga0,sch):
        return 0.0
    
    def dsdmu(self, r,p,m,rho,ga0,sch):
        mu=2*G/c**2*m/r
        return 2.*self.beta*p*mu
    
    def dsdr(self, r,p,m,rho,ga0,sch):
        mu=2*G/c**2*m/r
        s = self.s(r,p,m,rho,ga0,sch)
        dpdr =  -G*(rho+p/c**2)*(m+4*np.pi*r**3*p/c**2)/r**2/(1-2*G*m/r/c**2) - 2*s/r
        dmudr = G/c**2*(8*np.pi*rho*r-2*m/r**2)
        return self.beta*(mu**2*dpdr + 2*p*mu*dmudr)
    
    def s2(self, r,p,m,rho,ga0,sch):
        return 0.
    
#=========================================================
'''Example run of this module'''
def test_run():
    return
if __name__ == '__main__':
    test_run()
