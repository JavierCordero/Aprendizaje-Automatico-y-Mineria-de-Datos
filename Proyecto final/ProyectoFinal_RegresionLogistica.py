from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os

#Función sigmoide
def sigmoide(x):
    s = 1 / (1 + np.exp(-x))
    return s

#Función para el coste
def coste(theta, X, Y, landa):
    H = sigmoide(np.matmul(X, theta))
    m = len(X)
    
    cost = ((- 1 / m) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))) + ((landa / (2 * m)) * (np.sum(np.power(theta, 2))))
    
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

def coste_y_gradiente(x0, X, Y, landa):
    return coste(x0,X,Y,landa), gradiente(x0, X, Y, landa)

def calcOptTheta(Y, maxIt):
    #result = opt.fmin_tnc(func=coste, x0=np.zeros(X.shape[1]), fprime=gradiente, args=(X, Y, landa))
    #return result[0]
    result = opt.minimize(
        fun=coste_y_gradiente, 
        x0=np.zeros(X.shape[1]), 
        args=(X, Y, landa), 
        method='TNC', 
        jac=True, 
        options={'maxiter': maxIt})

    return result.x

def oneVsAll(X, y, num_etiquetas, reg, maxIt):
    
    ThetasMatriz = np.zeros((num_etiquetas, X.shape[1]))

    i = 0
    while i < num_etiquetas:

        os.system('cls')
        print("Numero de etiquetas procesadas: ", i + 1, " de un total de ", num_etiquetas, " con lamda = ", landa, " y ", maxIt, " iteraciones.")
        auxY = (y == i).astype(int)
        ThetasMatriz[i, :] = calcOptTheta(auxY, maxIt)
        i += 1

    return ThetasMatriz

def calcAciertos(X, Y, t):
    #X = todas las X
    #Y = la Y de cada fila de X
    #t = cada fila de la matriz de thetas
    cont = 0
    aciertos = 0
    totales = len(Y)
    dimThetas = len(t)
    valores = np.zeros(dimThetas)

    for i in X:      
        p = 0
        for x in range(dimThetas):
            valores[p] = sigmoide(np.dot(i, t[x]))
            p+=1

        r = np.argmax(valores)

        #print(str(r) + "------>" + str(Y[cont]))

        if(r==Y[cont]):
            aciertos+=1     

        cont += 1

    porcentaje = aciertos / totales * 100
    return porcentaje

# Selecciona aleatoriamente ejemplos y los pinta
def pinta_aleatorio(X):
    sample = np.random.randint(low=0, high=len(X) - 1, size=1)
    aux = X[sample, :].reshape(-1, 20)
    plt.imshow(aux.T)
    plt.axis("off")

def dibuja_puntos(X, Y, color):
    a = np.arange(100)
    b = X
    plt.plot(a, b, c=color)

    d = Y[0:len(X)]
    plt.plot(a, d, c="orange")

    plt.show()

datos = loadmat("proyecto_final_data_TRAIN.mat")

#almacenamos los datos leídos en X e y

X = datos["Xval"]
y = datos["yval"]

yaux = np.ravel(y) 

landas = [0.001, 0.01, 0.1, 1, 10, 50, 100, 500]
maxIterations = [70, 100, 150, 200, 300]
num_labels = len(np.unique(yaux))

#pinta_aleatorio(X)

testedLandasValues = dict()
testedLandas = []
p = 0

aciertos = []
myLandas = []
myIter = []

for i in landas:
    for r in maxIterations:
        landa = i
        one = (oneVsAll(X, yaux, num_labels, i, r))
        testedLandasValues[p] = np.mean(one)
        testedLandas.append(one)

        myLandas.append(i)
        myIter.append(r)
        aciertos.append(calcAciertos(X, yaux, one))
        p += 1

os.system('cls')
print("Porcentajes de aciertos con distintas lambdas: ")

p = 0
for x in aciertos:
    print(str(x) + "% de acierto con un valor de lambda = ", myLandas[p], " con ", myIter[p], " iteraciones.")
    p += 1

val =  aciertos.index(max(aciertos))

print("Mejor porcentaje de acierto: " ,str(aciertos[val]) + "% de acierto con un valor de lambda = ", myLandas[val], " con ", myIter[val], " iteraciones.")

dibuja_puntos(landas, aciertos, "blue")

dibuja_puntos(myIter, aciertos, "orange")