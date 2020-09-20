#considerando R=0 e M=1
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('sonar.csv',header=None)
import os
def clear():os.system('cls||clear')

def sigmoid(x,v):
    b = np.array(np.dot(x,v),dtype=np.float32)
    a = 1.0 / (1.0 + np.exp(-1.0*b))
    return a
def sigmoidDerivada(sig):
    return sig * (1 - sig)

x = np.array(data.values[:,0:59])
t = np.asmatrix(np.array(data.values[:,60])).T

nSaida = 1
nEscondidos = 25
ciclosMax = 10000
alfa = 0.005

v = 2*np.random.random((len(x[0,:]),nEscondidos)) - 1
w = 2*np.random.random((nEscondidos,1)) - 1

ciclo = 1
porcentagem = 0

for j in range(ciclosMax):
    if(porcentagem != int((j*100)/ciclosMax)):
        clear()
        porcentagem = int((j*100)/ciclosMax)
        print('Treinando...',porcentagem,'%')
    camadaOculta = sigmoid(x,v)
    camadaSaida = sigmoid(camadaOculta, w)

    erroCamadaSaida = np.array((t-camadaSaida),dtype=np.float32)
    erroQuadratico = 0.5 * (erroCamadaSaida**2)
    mediaAbsoluta = np.mean(np.abs(erroQuadratico))
    plt.scatter(j,mediaAbsoluta, color='orange', marker='.')

    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida

    wTransposta = w.T
    deltaSaidaxPeso = deltaSaida.dot(wTransposta)
    deltaBv = deltaSaidaxPeso * sigmoidDerivada(camadaOculta)

    camadaOcultaTrasposta = camadaOculta.T
    deltinhaW = camadaOcultaTrasposta.dot(deltaSaida)
    w = (w * ciclo) + (deltinhaW * alfa)

    xTransposta = x.T
    deltinhaV = xTransposta.dot(deltaBv)
    v = (v * ciclo) + (deltinhaV * alfa)

print("Erro final: " + str(mediaAbsoluta))
print('w: ', w)
print('v: ', v)

for i in range(len(t)):
    print(str(camadaSaida[i])) 

plt.show()