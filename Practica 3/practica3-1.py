from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

#Función sigmoide
def sigmoide(x):
    s = 1 / (1 + np.exp(-x))
    return s

#Función para el coste
def coste(theta, X, Y, landa):
    H = sigmoide(np.matmul(X, theta))
    cost = (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))
    
    return cost

# Selecciona aleatoriamente ejemplos y los pinta
def pinta_aleatorio(X):
    sample = np.random.choice(X.shape[0], 10)
    aux = X[sample, :].reshape(-1, 20)
    plt.imshow(aux.T)
    plt.axis("off")

def main():

    #datos.keys() consulta las claves
    datos = loadmat ("ex3data1.mat")

    #almacenamos los datos leídos en X e y
    y = datos["y"]
    X = datos["X"]

    pinta_aleatorio(X)

    #Mostramos los datos finalmente
    plt.show()

main()