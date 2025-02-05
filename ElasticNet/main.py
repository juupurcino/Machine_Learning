import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

global Y
global X
global w

df = pd.read_csv('day.csv')
df = df.drop(['dteday', 'casual', 'registered'], axis=1)
Y = df['cnt']
X = df.drop('cnt', axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
b = 0.1

w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0.1, 0.2, 0.3, 0.4, 0.5]

def error(w, b, X, Y):
    sum = 0
    sumW = 0
    modW = 0
    p = 0.2
    s = 0.3

    for linha in range(len(X)):
        multVet = 0

        for coluna in range(len(X.iloc[linha])):
            multVet += X.iloc[linha, coluna] * w[coluna]
        
        result = (multVet + b - Y.iloc[linha])**2
        sum += result
    
    for item in w:
        sumW += (item ** 2)
        modW += abs(sumW) 

    multP = p * sumW
    sig = s * modW

    return sum / X.shape[0] + multP + sig
    
def derivativeW(w, index, X, Y):
    err = error(w, b, X, Y) 
    
    w[index] += 0.1 # Um passo numa variável especifica em um intervalo de 0.1
    nerr = error(w, b, X, Y) # erro no ponto futuro
    w[index] -= 0.1 # Reverte o passo
    
    derivative = (nerr - err) / 0.1 # final menos inicial divido pelo intervalo
    
    return derivative # derivada do erro em relação a variável

def derivativeB(b, X, Y):
    err = error(w, b, X, Y) 
    
    b += 0.1 # Um passo numa variável especifica em um intervalo de 0.1
    nerr = error(w, b, X, Y) # erro no ponto futuro
    b -= 0.1 # Reverte o passo
    
    derivative = (nerr - err) / 0.1 # final menos inicial divido pelo intervalo
    
    return derivative # derivada do erro em relação a variável

alfa = 0.00000004

def sgdW(w, index, X, Y):
    return w[index] - alfa * derivativeW(w, index, X, Y)

def sgdB(b, X, Y):
    return b - alfa * derivativeB(b, X, Y)

batchSize = 100
train = []
tests = []
for e in range(100):
    for k in range(0, X_train.shape[0], batchSize):

        for i in range(len(w)):
            w[i] = sgdW(w, i, X_train.iloc[i:i+batchSize], Y_train)
        b = sgdB(b, X_train.iloc[i:i+batchSize], Y_train)
    
        train.append(error(w, b, X_train, Y_train))
        tests.append(error(w, b, X_test, Y_test))
    
    print(error(w, b, X_test, Y_test))

#gráfico
plt.plot(train)
plt.plot(tests)
plt.show()