import numpy as np
from scipy.io import loadmat

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

#Goal 303.993
#Función para el coste
def coste(X, y, m, theta, landa):
    h = np.dot(X, theta[:, None])
    return ((1 / (2 * m)) * (np.sum(np.square(h - y)))) + ((landa / (2 * m)) * np.sum(np.square(theta)))
    
#Función para calculo de gradiente
def gradiente(theta, XX, Y, landa, m):
    h = np.dot(XX, theta[:, None])

    grad = (1 / m) * np.matmul(XX.T, h - Y)
    

    print(grad)

    firstPart = grad+((landa/m) * theta)
    thetaAux = theta
    thetaAux[0] = 0


    result = firstPart + (landa / m * thetaAux)

    return result

def main():
    data = loadmat("ex5data1.mat")

    X = data["X"]
    y = data["y"]
    Xval = data["Xval"]
    yval = data["yval"]
    Xtest = data["Xtest"]
    ytest = data["ytest"]

    landa = 1

    XwithOnes=np.hstack((np.ones(shape=(X.shape[0],1)),X))

    theta = np.ones(XwithOnes.shape[1])

    print("Coste: ", str(coste(XwithOnes, y, XwithOnes.shape[0], theta, landa)))
    print("Gradiante: ", str(gradiente(theta, XwithOnes, y, landa, XwithOnes.shape[0])))

main()