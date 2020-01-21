import PIL
from PIL import Image
import os
import shutil
import time
import numpy as np
from numpy import genfromtxt
import scipy.io as sciOutput

def transform_images_inside_path(path, outputName):
    files = []
    #Cogemos todos los archivos jpg que encontremos dentro de la carpeta especificada
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file and not "_resized" in file:
                files.append(file)

    #Comprobamos que el path no exista y si es así, lo eliminamos por completo para no tener mal los datos
    if os.path.exists(path + "Resized/"):
       shutil.rmtree(path + "Resized/")    
    
    #Creamos de nuevo el directorio que hemos borrado
    os.mkdir(path + "Resized/")

    totalFiles = len(files)
    procesImages = 1

    matEntrenamientoX = []
    matEntreanmientoY = []

    matValidacionX = []
    matValidacionY = []

    matTestX = []
    matTestY = []

    y = np.array(genfromtxt('train.csv', delimiter=',', dtype=None, encoding=None))

    for f in files:
        #BUSCAR EN Y EL NOMBRE DE LA IMAGEN Y COGER EL NOMBRE DE LA BALLENA ASOCIADO PARA SACAR LA Y FINAL
        os.system("cls")
       
        img = Image.open(path + f)
        img = img.resize((20, 20), Image.ANTIALIAS) #Reescalamos la imagen
        img.save(path + "Resized/" + f[:-4] + "_resized.jpg")
        
        processedImage = True
        values = []
        pix  = img.load()
        for i in range(20):
            for j in range(20): 
                rgb = pix[i,j]
                try:
                    rgbInteger = (int)(("%02x%02x%02x"%rgb), 16)
                    values.append(rgbInteger)
                except:
                    processedImage = False

        if(processedImage):
            aux, aux2 = np.where(y == f)

            index, = np.where(np.unique(y[:, 1]) == (y[aux,1]))

            #Dividir en entrenamiento 60%, validación 20% y test 20%
            if procesImages - 1 < (int)(0.2 * totalFiles):
                matTestX.append(values)
                matTestY.append(index)

            elif procesImages - 1 < (int)(0.4 * totalFiles):
                matValidacionX.append(values)
                matValidacionY.append(index)

            else:
                matEntrenamientoX.append(values)
                matEntreanmientoY.append(index)

            print("Se han procesado ", procesImages, " imagenes de un total de ", totalFiles)
            procesImages += 1

    X = np.array(matEntrenamientoX)
    y = np.array(matEntreanmientoY)

    dict = {
        "X": X,
        "y": y,
        "Xval" : matValidacionX,
        "yval" : matValidacionY,
        "Xtest": matTestX,
        "ytest" : matTestY
    }

    sciOutput.savemat(outputName, dict)

    print("Matriz guardada en ", outputName)

    #Una vez hemos terminado de procesar los datos, podemos borrar la carpeta con imágenes adicionales que hemos creado
    if os.path.exists(path + "Resized/"):
       shutil.rmtree(path + "Resized/")    


def main():

    tic = time.time()
    transform_images_inside_path("ResizeTraining/", "proyecto_final_data_TRAIN.mat")
    #transform_images_inside_path("train/", "proyecto_final_data.mat")

    toc = time.time()
    
    print("Tiempo empleado para modificar la entrada al formato necesario: ", round((toc - tic) / 60.), " minutos, ", (toc - tic), " segundos.")

main()