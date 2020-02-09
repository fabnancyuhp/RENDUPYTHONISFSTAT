import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad

#####################################
######### Exercice 1  ###############
#####################################


A = np.array([[5, 2.6, 5, 9], [4, 3.6, 9, 10], [5.5, 4, 6, 7]])

moy_col = np.mean(A, axis=0)
moy_row = np.mean(A, axis=1)

# Exercice 2

etat = np.array(['voit', 'velo', 'trot'])
proba = np.array([1 / 6, 2 / 6, 3 / 6])
simu = np.random.choice(etat, size=10, replace=True, p=proba)
simu_normal = np.random.normal(4, 10, size=10)

# Exercice 3

V1 = np.ones(4) * 2
V1 = np.insert(V1, 2, 5)
V1 = V1.reshape(1, 5)
A = np.concatenate([V1] * 6, axis=0)


# Exercice 4

# 1Courbe de f entre 0 et 20

def f(v):
    return (1 / gamma(3)) * (v ** 2) * np.exp(-v)


x = np.linspace(0, 20, 200)
y = f(x)
plt.plot(x, y)

# 2AUC entre 3 et 10

x_1 = np.linspace(3, 10, 40)
y_1 = f(x_1)
plt.fill_between(x_1, y_1, color='blue', alpha=0.3)
plt.show()

# 3 Integrale entre 3 et 10 par somme Riemmann

n = 10000
a = 3
b = 10
sub = np.arange(n + 1)
S = ((b - a) / n) * sum(f(a + (sub * (b - a)) / n))

res = quad(f, a, b)

# Exercice 5

proba = [0.1, 0.2, 0.3, 0.4]
etat = np.array(['A', 'B', 'C', 'D'])
Simu = np.random.choice(etat, size=1000, replace=True, p=proba)

y_pos = np.arange(len(etat))


def count(tab, key):
    c = 0
    for el in tab:
        if el == key:
            c = c + 1
    return c


height = np.array([])
for letter in etat:
    height = np.append(height, count(tab=Simu, key=letter) / 10)

plt.bar(y_pos, height, edgecolor='blue', alpha=0.7,
        color=['blue'] * 4)
plt.xticks(y_pos, etat, color='black', fontweight='bold')
plt.box(False)
plt.yticks([])

for i in np.arange(len(height)):
    plt.text(y_pos[i], height[i] + 0.5, str(height[i]) + '%',
             color='black', fontweight='bold')
plt.show()


# Exercice 6

##Bar plot

def normale(z, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((z - mu) / sigma) ** 2)


norm = np.random.normal(6, 3, 10000)
plt.hist(norm, bins='sturges', density=True)
x_abs = np.linspace(-6, 16, 100)
plt.plot(x_abs, normale(x_abs, 6, 3), color='black')
plt.show()

# Exercice 8

# On va utiliser une méthode de Monte Carlo classique

n_h = 32455859
n_f = 34534967
p_h = n_h / (n_f + n_h)
poids = np.array([p_h, 1 - p_h])


def risque(nb, seuil):
    mc = 0
    for _ in np.arange(nb):
        pop = np.random.choice(np.array(['h', 'f']), size=100, replace=True, p=poids)
        nb_h = count(tab=pop, key='h')
        nb_f = 100 - nb_h
        x_h = np.random.normal(77.4, 12, size=nb_h)
        x_f = np.random.normal(62.4, 10.9, size=nb_f)
        tot = sum(np.append(x_h, x_f))
        mc = (tot > seuil) * 1 + mc
    return mc / nb


test = risque(50000, seuil=7200)

# Lerisque semble converger vers 4.4%


# Exercice 8

np.random.seed(seed=1998)

# nombre de points

n = 100

X = np.random.multivariate_normal(mean=np.zeros(2), cov=np.array([[1, 0], [0, 1]]), size=n)
Y = np.random.multivariate_normal(mean=np.array([1, 1]), cov=np.array([[0.25, 0], [0, 0.25]]), size=n)

proba = np.array([1 / 3, 2 / 3])

# Selection des lois 1 refère à la 1ere 2 a la 2ème

tir = np.random.choice(np.array([1, 2]), size=n, replace=True, p=proba)


# Représenter dans le nuage les points ayant été sélectionnés

# Function to find the indexes of a given item in an array


def find_index(array, item):
    resl = np.array([])
    for i in range(0, len(array)):
        if array[i] == item:
            resl = np.append(resl, i)
    return resl.astype(int)


Xn = X[find_index(tir, 1)]
Yn = Y[find_index(tir, 2)]

plt.scatter(np.concatenate([Xn[:, 0], Yn[:, 0]]), np.concatenate([Xn[:, 1], Yn[:, 1]]),
            color="red", alpha=0.6)
