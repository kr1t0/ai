import numpy as np
from matplotlib import pyplot as plt


def z_score(X):
    x_mean = np.mean(X)
    s2 =sum([(i - x_mean)**2 for i in X ]) / len(X)
    return [(i - x_mean) / s2 for i in X]


l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1=[]
cs=[]
for i in l:
    c=l.count(i)
    cs.append(c)

print(cs)
z=z_score(l)

print(z)

plt.plot(l,cs)
plt.plot(z,cs)
plt.show()