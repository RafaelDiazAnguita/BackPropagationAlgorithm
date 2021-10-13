# 1. Filtramos los datos de iteraciones y error con: 

# ./la1 -t train_sin.dat -T test_sin.dat |grep "Training error"|awk '{ print $2 " " $5; }'> output_sin.csv

# 2. Preprocesamos el fichero de salida a mano con pares pares de columnas [iteración, error entrenamiento].

# (Hemos borrado los valores 501 para los caos en que converge antes de llegar al límite de iteraciones). 

import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("output_sin1.csv")

plt.plot(df['Iteration'],df['TrainingError'], label='Seed 1')

df = pd.read_csv("output_sin2.csv")

plt.plot(df['Iteration'],df['TrainingError'], label='Seed 2')


plt.legend()

plt.xlabel('Epoch')

plt.ylabel('Training Error')

# Para centrar la parte que queremos dibujar.

#plt.xlim([0,500])

#plt.ylim([0.02,0.04])

print(plt.ylim())

plt.show()
