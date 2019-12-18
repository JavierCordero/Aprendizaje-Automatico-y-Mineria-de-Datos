import numpy as np
from scipy.io import loadmat
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

#Función para el coste
def coste(theta, X, y, landa, m):
    h = np.dot(X, theta[:, None])

    thetaAux = np.delete(theta, 0, 0)
    return ((1 / (2 * m)) * (np.sum(np.square(h - y)))) + ((landa / (2 * m)) * np.sum(np.square(thetaAux)))
    
#Función para calculo de gradiente
def gradiente(theta, XX, Y, landa, m):
    h = np.dot(XX, theta[:, None])

    #Eliminamos la primera columna de theta
    thetaAux = np.delete(theta, 0, 0)
    return (1 / m) * np.matmul(XX.T, h - Y)+((landa/m) * thetaAux)

def coste_y_gradiente(theta, X, Y, landa, m):
    # Ensure theta shape(number of features+1, 1).
    theta = theta.reshape(-1, Y.shape[1])
    
    #############################################################
    ###################### Cost Computation #####################
    #############################################################
    # Compute the cost.
    unreg_term = (1 / (2 * m)) * np.sum(np.square(np.dot(X, theta) - Y))
    
    # Note that we should not regularize the theta_0 term!
    reg_term = (landa / (2 * m)) * np.sum(np.square(theta[1:len(theta)]))
    
    cost = unreg_term + reg_term
    
    #############################################################
    #################### Gradient Computation ###################
    #############################################################
    # Initialize grad.
    grad = np.zeros(theta.shape)

    # Compute gradient for j >= 1.
    grad = (1 / m) * np.dot(X.T, np.dot(X, theta) - Y) + (landa / m ) * theta
    
    # Compute gradient for j = 0,
    # and replace the gradient of theta_0 in grad.
    unreg_grad = (1 / m) * np.dot(X.T, np.dot(X, theta) - Y)
    grad[0] = unreg_grad[0]

    return (cost, grad.flatten())

def calcOptTheta(X, Y, landa):
    theta = np.zeros((X.shape[1], 1))
    
    def costFunction(theta):
        return coste_y_gradiente(theta, X, Y, landa, len(X))

    result = minimize(fun=costFunction,
                       x0=theta,
                       method='CG',
                       jac=True,
                       options={'maxiter':200})
    
    print(result.x)

    return result.x

def pinta_puntos(X, Y):
    plt.scatter(X, Y, marker = 'x', c = 'red', label = 'Entrada')

def curva_aprendizaje(X, y, landa, theta, Xval, yval):

    err1 = np.zeros((len(X)))
    err2 = np.zeros((len(Xval)))

    i = 1
    while (i < len(X) + 1):
        thetas = calcOptTheta(X[0:i], y[0:i], landa)
        
        err1[i - 1] = calc_error(thetas, X[0:i], y[0:i], landa, len(X))
        err2[i - 1] = calc_error(thetas, Xval, yval, landa, len(Xval))
        i += 1   

    return err1, err2    

def calc_error(theta, X, y, landa, m):
    h = np.dot(X, theta[:, None])
    return (1 / (2 * m)) * ((np.sum(np.square(h - y))))

def pinta_Curva_Aprendizaje(err1, err2):
    
    a = np.arange(len(err1))
    b = err1
    plt.plot(a, b, c="blue", label="Train")

    d = err2[0:len(err1)]
    plt.plot(a, d, c="orange", label="Cross Validation")

def normaliza_Matriz(X):
    numColumnas = X[1].size  
    
    media = np.zeros(numColumnas)
    varianza = np.zeros(numColumnas)

    for x in range(numColumnas):
        xData = X[:,x]
        media[x] = np.mean(xData)
        varianza[x] = np.std(xData)

    X_normalizadas = (X - media) / varianza
     
    return X_normalizadas

def transforma_entrada(X, p):
    matrix = np.zeros((len(X), p))
    matrix[:,0] = X[:,0]
    i = 1
    while i < p:
        matrix[:,i] = np.power(X[:,0], i + 1)
        i+=1
    
    return normaliza_Matriz(matrix)   
    
def pinta_frontera_curva(X, Y, theta, landa):
    plt.contour(X, Y, linewidths=2, colors='green')  

def main():
    data = loadmat("ex5data1.mat")

    X = data["X"]
    y = data["y"]
    #Xval = data["Xval"]
    #yval = data["yval"]
    #Xtest = data["Xtest"]
    #ytest = data["ytest"]

    landa = 0

    #XwithOnes=np.hstack((np.ones(shape=(X.shape[0],1)),X))

    #XvalWithOnes = np.hstack((np.ones(shape=(Xval.shape[0],1)),Xval))

    p = 8

    

    nuevaentrada = transforma_entrada(X, p)

    nuevaentrada = np.insert(nuevaentrada, 0, 1, axis=1)
    
    thetaOpt = calcOptTheta(nuevaentrada, y, landa)

    #pinta_frontera_curva(X, y, thetaOpt, landa)

    #err1, err2 = curva_aprendizaje(XwithOnes, y, landa, theta, XvalWithOnes, yval)

    #pinta_Curva_Aprendizaje(err1, err2)

    #plt.legend()
    plt.show()

main()