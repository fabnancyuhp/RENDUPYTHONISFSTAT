# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
from scipy import stats
import itertools
import matplotlib.pyplot as plt
#%%
#ex1
A=np.array([[5,2.6,5,9],[4,3.6,9,10],[5.5,4,6,7]])
mc=np.mean(A,axis=0)
ml=np.mean(A,axis=1)

Uni = np.random.uniform(2,6,size=(3,2))
Uni

proba = np.array([1/4,1/5,1/5,1-1/4-2*1/5])
etat = np.array(['a','b','c','d'])
Simu = np.random.choice(etat, size=20, replace=True, p=proba)
Simu
#%%
#exo2
etat = np.array(['voit','velo','trot'])
proba = np.array([1/6,2/6,3/6])
Simu = np.random.choice(etat, size=20, replace=True, p=proba)
Norm = np.random.normal(4,10,size=10)
#%%
#exo3
d=2*np.ones(4)
d=np.insert(d,2,5)
d=np.array([d])
d=np.concatenate([d,d,d,d,d],axis=0)

from scipy.special import gamma
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,10,100)
y = (1/(4*gamma(2)))*x*np.exp(-x/2)
plt.plot(x,y)
x1 = np.linspace(2,6,20)
y1 = (1/(4*gamma(2)))*x1*np.exp(-x1/2)
plt.fill_between(x1,y1, color='blue',alpha=0.3)
plt.show()
#%%
#exo4
x = np.linspace(0,20,20)
y = (1/(gamma(3)))*(x**2)*np.exp(-x)
plt.plot(x,y)
x1 = np.linspace(3,10,20)
y1 = (1/(gamma(3)))*(x1**2)*np.exp(-x1)
plt.fill_between(x1,y1, color='blue',alpha=0.3)
plt.show()

def f(x):
    y= (1/(gamma(3)))*(x**2)*np.exp(-x)
    return(y)
from scipy.integrate import quad
quad(f,3,10)





#%%
#exo5
np.random.seed(1998)
proba = [0.1,0.2,0.3,0.4]
etat = np.array(['A','B','C','D'])
Simu = np.random.choice(etat, size=1000, replace=True, p=proba)
a=0
b=0
c=0
d=0
for i in Simu:
    if i=='A':a+=1
    elif i=='B':b+=1
    elif i=='C':c+=1
    elif i=='D':d+=1


objects =  np.array(['A','B','C','D'])
y_pos = np.arange(len(objects))
performance = np.array([a/10,b/10,c/10,c/10])

plt.bar(y_pos, performance, align='center', alpha=0.7)
plt.xticks(y_pos, objects)
plt.ylabel('percent')
plt.title('prob')

plt.show()

#%%
#exo6
np.random.seed(1998)
nor = np.random.normal(6,3, 10000)
plt.hist(nor,bins='sturges',density=True) #sturges
x = np.linspace(-5,15,100)
y = (1/(np.sqrt(9*2*np.pi)))*np.exp(-(1/2)*((x-6)/3)**2)
plt.plot(x,y,color='black')
plt.show

#%%
#ex7
h=32455859
f=34534967
n=h+f


def norme(h,f,nb):
    n=h+f
    p=0
    
    etat = np.array(['h','f'])
    proba = np.array([h/n,f/n])
    for i in range(nb):
        Simu = np.random.choice(etat, size=1, replace=True, p=proba)
        
        if Simu=='h': p += np.random.normal(77.4,12,size=1)
        else: p += np.random.normal(62.4,10.9,size=1)
        
    return(p)

import math as mp
from scipy.integrate import quad
import scipy.integrate as integrate
def binom(n,k):
         return(mp.factorial(n)/(mp.factorial(n-k)*mp.factorial(k)))
p=0    
for i in range(100):
    p+=(1-stats.norm.cdf(7200,77.4*i+62.4*(100-i),np.sqrt(12**2*i+(100-i)*10.9**2)))*binom(100,i)*((h/n)**i)*((f/n)**(100-i))
    

def densitepoid(x):
    return ((h/n)*np.exp(-(x-77.4)**2/2/12**2)/np.sqrt(2*np.pi)/12 + (f/n)*np.exp(-(x-62.4)**2/2/10.9**2)/np.sqrt(2*np.pi)/10.9)

mu = 100*(77.4*(h/n)+62.4*(f/n))
mubis = 77.4*(h/n)+62.4*(f/n)
sigma = np.sqrt(100)*np.sqrt((np.sqrt(h/n)*12)**2+(np.sqrt(f/n)*10.9)**2) 

sigmabis = np.sqrt(100)*np.sqrt((h/n)*(12**2+77.4**2)+(f/n)*(10.9**2+62.4**2)-mubis**2)

from scipy.stats import norm

1-norm.cdf(7200, loc=mu, scale=sigmabis)

def densitepoidbis(x):
    return(np.exp(-(x-mu)**2/(2*sigmabis**2))/(np.sqrt(2*np.pi)*sigmabis))

result = integrate.quad(lambda x:densitepoidbis(x),7200, np.inf)
    
c=0
for i in range(1000):
    if norme(h,f,100)>7200:c+=1
    
#h=32455859
#f=34534967
#n=h+f
#
#etat = np.array(['h','f'])
#proba = np.array([h/n,f/n])
#
#list_croisiere = []
#
#
#for i in range(10000):
#    Simu = np.random.choice(etat, size=100, replace=True, p=proba)
#
#    POID_H= np.random.normal(77.4,12,size=100)
#    POID_F = np.random.normal(62.4,10.9,size=100)
#    POID_H_BIS = POID_H[Simu=='h']
#    POID_F_BIS = POID_F[Simu=='f']
#    POID_T = np.concatenate([POID_H_BIS,POID_F_BIS ],axis=0)
#    list_croisiere.append(np.sum(POID_T))
#    #list_croisiere = list_croisiere+[np.sum(POID_T)]
#    
#list_numpy = np.array(list_croisiere)    
#
#list_numpy_gt7200 = list_numpy[list_numpy>7200]
#
#
#proba = list_numpy_gt7200.shape[0]/10000     
#    
    
    
#%%
#exo8
u1=np.array([[0],[0]])
sig1=np.array([[1,0],[0,1]])
u2=np.array([[1],[1]])
sig2=np.array([[0.25,0],[0,0.25]])

a1=np.linalg.cholesky(sig1)
a2=np.linalg.cholesky(sig2)
x=np.array([[],[]])

for i in range(100):
    if np.random.random(1)<1/3:
        x1=[np.random.normal(0,1,1),np.random.normal(0,1,1)]
        y1=np.dot(a1,x1)+u1
        x=np.concatenate([x,y1],axis=1)
    else:
        x2=[np.random.normal(0,1,1),np.random.normal(0,1,1)]
        y2=np.dot(a2,x2)+u2
        x=np.concatenate([x,y2],axis=1)

plt.scatter(x[0],x[1],alpha=0.6)