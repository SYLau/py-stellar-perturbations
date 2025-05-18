'''Locate the working directory'''
if __name__ == '__main__':    
    import sys
    from _path import workdir
    sys.path.append(workdir)
    print(sys.path[-1])

#=========================================================
'''Main content of the module'''
import numpy as np

class eos_poly:
    def __init__(self,k,n):
        self.k = k
        self.n = n

    def rho(self, p):
        k = self.k
        n = self.n
        return np.sign(p)*np.abs(p/k)**(n/(n+1))
    
    def p(self, rho):
        k = self.k
        n = self.n
        return np.sign(rho)*k*np.abs(rho)**((n+1)/n)
    
    def ga(self, p):
        n = self.n
        return (n+1.)/n
    
#=========================================================
'''Example run of this module'''
def test_run():
    
    '''Initialize class: provide polytropic constant and index'''
    eos = eos_poly(k = 1.e35/(1.e15)**2, n = 1.)

    '''Compute p(rho), rho(p), and gamma(p)'''
    print('p(rho) = %1.6e'%eos.p(1.e15))
    print('rho(p) = %1.6e'%eos.rho(eos.p(1.e15)))
    print('ga(p) = %1.6e'%eos.ga(eos.p(1.e15)))

if __name__ == '__main__':
    test_run()
