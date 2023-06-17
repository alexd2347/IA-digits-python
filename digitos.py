import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from time import time
import matplotlib.pyplot as plt

def mostrar_num(in_data):
    matriz = np.array(in_data.values)
    plt.imshow(matriz.reshape(28,28))
    plt.show()

csv_digit = pd.read_csv('digitos.csv')
new_csv = csv_digit.head(40000);
#new_csv = csv_digit.iloc[:40000, 0:785]

x_fields = new_csv.iloc[:, 1:785]
y_fields = new_csv.iloc[:, 0:1]


xtrain, xtest, Ytrain, Ytest = train_test_split(x_fields, y_fields)


svm = SVC()
startTime = time()


print("Entrenamiento iniciado...")
svm.fit(xtrain.values, Ytrain.values.ravel())
print("Training time: {:.2f} seconds".format(time() - startTime))


startPredictionTime = time()
print("Iniciando prediccion...")
yPred = svm.predict(xtest)
print("Prediction time: {:.2f} seconds".format(time() - startPredictionTime))


accuracy = accuracy_score(Ytest, yPred)
print("Accuracy: {:.2f}".format(accuracy))


mostrar_num(csv_digit.iloc[40005, 1:785])
new_prediction = svm.predict(csv_digit.iloc[40005, 1:785].to_numpy().reshape(1, -1))
print("El numero es: ",format(new_prediction))

