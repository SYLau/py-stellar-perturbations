'''Locate the working directory'''
if __name__ == '__main__':    
    import sys
    from _path import workdir
    sys.path.append(workdir)
    print(sys.path[-1])

#=========================================================
'''Main content of the module'''
import sys
import numpy as np
from src.util.constants import c

class eos_table:
    def __init__(self,rho,p):
        self.rho_list = rho
        self.p_list = p

        self._check_input()

    def _check_input(self):
        rho_sorted = np.all(self.rho_list[:-1] <= self.rho_list[1:])
        p_sorted = np.all(self.p_list[:-1] <= self.p_list[1:])
        if not (rho_sorted):
            print('eos_table: rho_list not sorted in ascending order')
            sys.exit(1)
        if not (p_sorted):
            print('eos_table: p_list not sorted in ascending order')
            sys.exit(1)
        if len(self.rho_list) != len(self.p_list):
            print('eos_table: rho_list, p_list not of the same length')
            sys.exit(1)

    def rho(self, p):
        rho_list = self.rho_list
        p_list = self.p_list
        log_rho = np.interp(np.log(p),np.log(p_list),np.log(rho_list))
        return np.exp(log_rho)

    def p(self, rho):
        rho_list = self.rho_list
        p_list = self.p_list
        log_p = np.interp(np.log(rho),np.log(rho_list),np.log(p_list))
        return np.exp(log_p)
    
    def ga(self, p):
        rho_list = self.rho_list
        p_list = self.p_list
        i = np.searchsorted(p_list, p, side='left')
        dlnp_dlnrho = (np.log(p_list[i+1])-np.log(p_list[i]))/(np.log(rho_list[i+1])-np.log(rho_list[i]))
        return (1+p/c**2/self.rho(p))*dlnp_dlnrho

#=========================================================
'''Example run of this module'''
def test_run():

    '''Read EOS tables into lists'''
    rho_list, p_list = np.loadtxt('src/eos/sample_eos_sly4.dat'\
                                  ,skiprows=1, usecols=(2,3),unpack=True)
    
    '''Initialize class: provide discrete lists of rho and p of the same length'''
    eos = eos_table(rho_list, p_list)

    '''Compute p(rho), rho(p), and gamma(p) using linear log interpolation of the EOS table'''
    print('p(rho) = %1.6e'%eos.p(1.e15))
    print('rho(p) = %1.6e'%eos.rho(eos.p(1.e15)))
    print('ga(p) = %1.6e'%eos.ga(eos.p(1.e15)))

if __name__ == '__main__':
    test_run()