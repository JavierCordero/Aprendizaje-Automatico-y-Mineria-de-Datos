import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures as PF

def carga_csv(file_name):
    """carga el fichero csv especificado y lo
    devuelve en un array de numpy"""
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def cost(theta, X, Y, landa):
    H = sigmoid(np.matmul(X, theta))
    cost = (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))
    
    return cost

def gradient(theta, XX, Y, landa):
    H = sigmoid(np.matmul(XX, theta))
    m=len(Y)
    grad = (1 / m) * np.matmul(XX.T, H - Y)
    
    aux=np.r_[[0],theta[1:]]

    result = grad+(landa*aux/m)
    return result
     
def calcOptTheta():
    result = opt.fmin_tnc(func=cost, x0=np.zeros(Xpoly.shape[1]), fprime=gradient, args=(Xpoly, Y, landa))
    return result[0]

def pinta_frontera_curva(X, Y, theta, landa):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(theta))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=2, colors='green')  

def pinta_puntos(X, Y):
    
    pos = np.where(Y == 1)
    plt.scatter(X[pos, 0], X[pos, 1], marker = '+', c = 'red', label = 'y = 1')

    neg = np.where(Y == 0)
    plt.scatter(X[neg, 0], X[neg, 1], marker = '.', c = 'blue', label   = 'y = 0')

def calcAciertos(X, Y, t):
    prediccion = 0 
    cont = 0
    aciertos = 0
    totales = len(Y)
    
    for i in X:
        if sigmoid(np.dot(i, t)) >= 0.5:
            prediccion = 1
        else:
            prediccion = 0
        
        if Y[cont] == prediccion:
            aciertos += 1

        cont += 1
            
    porcentaje = aciertos / totales * 100

    plt.text(-0.9,1.1, str("{0:.2f}".format(porcentaje)) + "% de aciertos")

datos = carga_csv('ex2data2.csv')

X = datos[:, :-1]
np.shape(X)         

Y = datos[:, -1]
np.shape(Y)     

pinta_puntos(X, Y)

poly = PF(6)
Xpoly = poly.fit_transform(X)
landa = 1

T = calcOptTheta()

pinta_frontera_curva(np.delete(Xpoly,0,axis=1),Y,T, landa)

calcAciertos(Xpoly, Y, T)

plt.text(0,1.3, str("λ: ") + str(landa))

plt.legend()
plt.show()