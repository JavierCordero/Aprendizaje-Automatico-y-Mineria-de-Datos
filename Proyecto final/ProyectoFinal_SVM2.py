from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm as supportVectorMachine
from sklearn.model_selection import train_test_split

def test(svm, Xval, Yval):
    pred = svm.predict(Xval)
    acc = np.mean((pred == Yval).astype(int))

    return acc

def parte1():
    data = loadmat("proyecto_final_data_TRAIN.mat")

    X = data["X"]
    y = data["y"]
    Xval = data["Xval"]
    yval = data["yval"]
    yravel = y.ravel()

    Coef = 1.0

    svm = supportVectorMachine.SVC(kernel="linear", C=Coef)
    svm = svm.fit(X, yravel)

    prec = test(svm, Xval, Yval)

    print(prec)

def parte2():
    data = loadmat("proyecto_final_data_TRAIN.mat")
    
    X = data["X"]
    y = data["y"]
    Xval = data["Xval"]
    yval = data["yval"]
    yravel = y.ravel()
    
    Coef = 1.0
    sigma = 0.1

    svm = supportVectorMachine.SVC(C=Coef, kernel='precomputed', tol= 1e-3, max_iter= 100)
    svm = svm.fit(gaussianKernel(X, X, sigma=sigma), yravel)

    prec = test(svm, Xval, Yval)

    print(prec)

def main():

    #PARTE 1
    parte1()

    #PARTE 2
    #parte2()

main()