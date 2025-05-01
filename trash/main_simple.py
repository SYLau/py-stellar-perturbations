import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Copied from py Phil ivp example and precession.py
class const:
  def __init__(self):                   # cgs units
    self.c=2.998e10
    self.G=6.6743e-8
    self.msun=1.989e33
    self.rsun=6.96e10
    self.pi=np.pi

class eos_poly:
  def __init__(self,k,n):
    self.k = k
    self.n = n

  def rho(self, p):
    k = self.k
    n = self.n
    rho = np.sign(p)*np.abs(p/k)**(n/(n+1))
    return rho
  
class eq_tov:
  def __init__(self,eos_rho):
    self.eos_rho = eos_rho

  def deriv(self,r,y):
    con=const()
    pi= con.pi
    p=y[0]
    m=y[1]
    nu=y[2]
    rho = self.eos_rho(p)
    dydr=0.0*y
    dydr[0]= -(rho+p)*(m+4*pi*r**3*p)/r**2/(1-2*m/r)
    dydr[1]= 4*pi*r**2*rho
    dydr[2]= 0.5*(m+4*pi*r**3*p)/r**2/(1-2*m/r)
    return dydr

class tov_event:
  def __init__(self,pmin=0.0):
    self.pmin = pmin

  def event(self,r,y):
    return y[0]-self.pmin
  
  event.terminal = True
  event.direction = -1

class tov_sol:
  r = []
  p = []
  m = []
  nu = []

# Example plot function
def plotP(r,p):
  plt.figure()
  plt.plot(r/1.e5,p,'k.',linewidth=3)
  plt.xlabel(r'radius(km)',fontsize=15)
  plt.ylabel(r'pressure(km$^{-2}$)',fontsize=15)
  plt.yscale('log')
  plt.show()
  plt.close()

def main():
  con=const()
  G = con.G
  c = con.c
  pi= con.pi

  eos = eos_poly(k = 1e12, n = 1)

  eq = eq_tov(eos.rho)

  pcut=tov_event(pmin = G/c**4*1e20)
  
  p0 = G/c**4*1.5e35
  rho0 = eos.rho(p0)
  Rs = np.sqrt(3.*p0/(2.*pi*(rho0+p0)*(rho0+3.*p0)))
  dr = 1e-5*Rs
  r0 = dr
  m0 = 4.*pi/3*rho0*r0**3
  nu0 = 2.*pi/3*(rho0+3.*p0)*r0**2
  
  y = np.zeros(3)
  y[0] = p0
  y[1] = m0
  y[2] = nu0

  rlist = np.arange(r0,Rs*30,dr)
  
  isol = solve_ivp(eq.deriv, t_span=[r0,rlist[-1]], y0 = y, t_eval=rlist, 
                     method = 'LSODA', atol = 0.0, rtol = 1e-12, events = pcut.event)

  print('R = ', isol.t[-1]/1e5, ' km')
  print('R events = ', isol.t_events[0][0]/1e5, ' km')
  print('P(R) = ', isol.y[0,-1]/(G/c**4), ' cgs')
  print('P(R) events = ', isol.y_events[0][0,0]/(G/c**4), ' cgs')

  #input("Press Enter to continue...")

  #sol_r = np.append(isol.t,isol.t_events[0])
  #sol_y = np.column_stack((isol.y,isol.y_events[0][0]))
  #nu_R=0.5*np.log(1.-2.*sol_y[1,-1]/sol_r[-1])-sol_y[2,-1]
  #sol_y[2,:]=sol_y[2,:]+nu_R

  tov = tov_sol
  tov.r = np.append(isol.t,isol.t_events[0])
  tov.p = np.append(isol.y[0,:],isol.y_events[0][0,0])
  tov.m = np.append(isol.y[1,:],isol.y_events[0][0,1])
  tov.nu = np.append(isol.y[2,:],isol.y_events[0][0,2])
  nu_R=0.5*np.log(1.-2.*tov.m[-1]/tov.r[-1])-tov.nu[-1]
  tov.nu=tov.nu+nu_R

  plotP(tov.r,tov.p)

if __name__ == '__main__':
  main()