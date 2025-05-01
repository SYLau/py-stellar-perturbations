import numpy as np

# m = np.array([[1.,0.],[7.,4j]])
# v = np.array([0.,1.])
# print(np.matmul(m,v))
# print(m[0,1])

# a = np.array([[1,2,3,4],[5,6,7,8]])
# print(np.append(a,[[0,0,0,0]],axis=0))
# print(np.append(a,[[0],[0]],axis=1))

# def multi_values():
#     return [1,2],[2,3],[3,4]


# a = multi_values()
# b = np.array(multi_values())
# print(a)
# print(a[1])
# print(b)
# print(b[1])

# i = 0
# while i<=10:
#     i += 1
# print(i)

# def deriv():
#     return np.array([1.,2.,3.])
# x,y,z = deriv()
# print(x)
# print(y)
# print(z)

# def f(x):
#     print(x)
#     return x**2
# print(list(map(f,np.linspace(1.,10.,5))))

# print('%10.6e + i %10.6e'%(np.pi, np.pi/2))

# a = np.array([1.,2.,3.])
# b = a
# a *= -1.
# print(b)

# c = np.array([1.,2.,3.])
# d = np.copy(c)
# c *= -1
# print(d)

a = np.array([1.,2.,3.,4.,5.])
a[3:5]= a[0:2]*-1.
print(a)