from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

#Función sigmoide
def sigmoide(x):
    s = 1 / (1 + np.exp(-x))
    return s

#Función para el coste
def coste(theta, X, Y, landa):
    H = sigmoide(np.matmul(X, theta))
    cost = (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H))) + (landa / 2 * len(X) * np.sum((np.power(theta, 2))))
    
    return cost

#Función para calculo de gradiente
def gradiente(theta, XX, Y, landa):
    H = sigmoide(np.matmul(XX, theta))
    m=len(Y)
    grad = (1 / m) * np.matmul(XX.T, H - Y)
    
    aux=np.r_[[0],theta[1:]]

    firstPart = grad+(landa*aux/m)
    thetaAux = theta
    thetaAux[0] = 0
    result = firstPart + (landa / m * thetaAux)
    return result

def calcOptTheta(Y):
    result = opt.fmin_tnc(func=coste, x0=np.zeros(X.shape[1]), fprime=gradiente, args=(X, Y, landa))
    return result[0]

def oneVsAll(X, y, num_etiquetas, reg):
    """oneVSAll entrena varios clasificadores por regresión logística con término
       de regularización 'reg' y devuelve el resultado en una matriz, donde
       la fila i-ésima corresponde al clasificadore de la etiqueta i-ésima. 
    """
    ThetasMatriz = np.zeros((num_etiquetas, X.shape[1]))

    i = 0
    while i < num_etiquetas:
        if i == 0:
            aux = 10
        else: aux = i

        auxY = (y == aux).astype(int)
        ThetasMatriz[i, :] = calcOptTheta(auxY)

        i += 1

    return ThetasMatriz

def calcAciertos(X, Y, t):
    #X = todas las X
    #Y = la Y de cada fila de X
    #t = cada fila de la matriz de thetas
    cont = 0
    aciertos = 0
    totales = len(Y)

    for i in X:    
        maxvalor=-1   
        if Y[cont] == 10:
            aux = 0
        else: aux = Y[cont]
        for x in range(len(t)):
            valor = sigmoide(np.dot(i, t[aux]))
            if(valor>maxvalor):
                maxvalor=valor
                thetaIesimo=x+1

        if(thetaIesimo==Y[cont]):aciertos+=1     

        cont += 1

    print(aciertos)
    porcentaje = aciertos / totales * 100
    return porcentaje

# Selecciona aleatoriamente ejemplos y los pinta
def pinta_aleatorio(X):
    sample = np.random.choice(X.shape[0], 10)
    aux = X[sample, :].reshape(-1, 20)
    plt.imshow(aux.T)
    plt.axis("off")

#datos.keys() consulta las claves
datos = loadmat ("ex3data1.mat")

#almacenamos los datos leídos en X e y
y = datos["y"]
X = datos["X"]
yaux = np.ravel(y) 

landa = 2

#print(oneVsAll(X, yaux, 10, landa))
print(str(calcAciertos(X, yaux, oneVsAll(X, yaux, 10, landa))) + "% de acierto")

#Mostramos los datos finalmente
plt.show()   