import numpy as np

'''Table 1 of Yagi & Yunes 2017'''
moi_love = lambda x: np.exp(1.496 + 0.05951*np.log(x) + 0.02238*np.log(x)**2 - 6.953e-4*np.log(x)**3 + 8.345e-6*np.log(x)**4)
moi_Q = lambda x: np.exp(1.393 + 0.5471*np.log(x) + 0.03028*np.log(x)**2 + 0.01926*np.log(x)**3 + 4.434e-4*np.log(x)**4)
Q_love = lambda x: np.exp(0.1940 + 0.09163*np.log(x) + 0.04812*np.log(x)**2 - 4.283e-3*np.log(x)**3 + 1.245e-4*np.log(x)**4)
# Q_love = lambda x: np.exp(0.194 + 0.0936*np.log(x) + 0.0474*np.log(x)**2 - 4.21e-3*np.log(x)**3 + 1.23e-4*np.log(x)**4)  ## Yagi & Yunes 2013


'''
References
    Yagi & Yunes 2017:
    https://linkinghub.elsevier.com/retrieve/pii/S0370157317300492
'''