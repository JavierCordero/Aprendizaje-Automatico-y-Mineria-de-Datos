from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import time
import os

#Función sigmoide
def sigmoide(x):
    s = 1 / (1 + np.exp(-x))
    return s

#Función para el coste
def coste_vectorizado(h, X, Y, landa):
  m = len(X)
  J = 0

  for i in range(m):
    J += np.sum(-Y[i] * mp.log(h[i]) - (1 - Y[i])* np.log(1 - h[i]))

  return J/float(m)  

#backprop devuelve el coste y el gradiente de una red neuronal de dos capas.    
def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn [num_ocultas * (num_entradas + 1 ):], (num_etiquetas, (num_ocultas + 1)))
    
    m = X.shape[0]
  
    a1, z2, a2, z3, h = propagacion_hacia_delante(X, theta1, theta2)

    #Aqui ya tenemos el coste de la función
    coste = coste_vectorizado(h, X, y, landa)
    print(coste)
    #Calculo para el gradiante
    for t in range(m):
      a1t = a1[t, :] # (1, 401)
      a2t = a2[t, :] # (1, 26)
      ht = h[t, :] # (1, 10)
      yt = y[t] # (1, 10)
      d3t = ht - yt # (1, 10)
      d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t)) # (1, 26)
      delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
      delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    #En este momento, tenemos calculado el gradiante de la función
    #Tenemos que sacar D(i,j)1 y D(i,j)2, que son las MATRICES de los gradientes.
    #Una vez tengamos estas dos matrices, las tenemos que poner en un vector que sea D1 + D2
    #La formula para DN = 1/m * deltaN + lambda*ThetaN si j != 0
    # o  DN = 1/m * deltaN para n == 0

    #para hacer "unroll" de matrices es Delta1 = np.reshape(delta1[1:110], (10,11)); Cambiando los números por los tamaños que sean

    #Para concatenar dos vectores, np.concatenate(delta1, delta2);

    #GradianteFinal = np.concatenate(delta1, delta2)
    #Devolver el coste y el nuevo vector V (que es D1 + D2)

    return coste#, GradianteFinal

def propagacion_hacia_delante(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.hstack([np.ones([m, 1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoide(z2)])
    z3 = np.dot(a2, theta2.T)
    h = sigmoide(z3)
    return a1, z2, a2, z3, h

data = loadmat ("ex4data1.mat")

#almacenamos los datos leídos en X e y
y = data['y'].ravel() # (5000, 1) --> (5000,)
X = data['X']

m = len(y)
input_size = X.shape[1]
num_labels = 10
capa_oculta = 25
y = (y - 1)
y_onehot = np.zeros((m, num_labels))  # 5000 x 10

for i in range(m):
  y_onehot[i][y[i]] = 1

weights = loadmat ("ex4weights.mat")
theta1, theta2 = weights["Theta1"], weights["Theta2"]
# Theta1 es de dimensión 25 x 401
# Theta2 es de dimensión 10 x 26

theta1Reshaped = theta1.reshape((1, 25*401))
theta2Reshaped = theta2.reshape((1, 10*26))

params = np.concatenate(theta1Reshaped, theta2Reshaped)

backprop(parms, input_size, capa_oculta,num_labels, X, y, 0)

plt.show()