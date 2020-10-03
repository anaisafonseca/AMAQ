import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
wcss = []
data = pd.read_csv('observacoes.csv',sep=',',header=None)

k = 4
#kmeans = KMeans(n_clusters=k, init='random')
kmeans = KMeans(n_clusters=k, init='k-means++')
kmeans.fit(data)
center = kmeans.cluster_centers_
distance = kmeans.fit_transform(data)
labels = kmeans.labels_
data = np.array(data, dtype=np.float32)

print('Número de iterações: ', kmeans.n_iter_)
print('Centros: ')
print(center)
for i in range(k):
    plt.scatter(center[i][0], center[i][1], color='black', marker='o')

for i in range(len(data)):
    if(labels[i]==0):
        plt.scatter(data[i][0], data[i][1], color='blue', marker='.')
    if(labels[i]==1):
        plt.scatter(data[i][0], data[i][1], color='green', marker='.')
    if(labels[i]==2):
        plt.scatter(data[i][0], data[i][1], color='orange', marker='.')
    if(labels[i]==3):
        plt.scatter(data[i][0], data[i][1], color='red', marker='.')
plt.show()

blue = []
green = []
orange = []
red = []
for i in range(len(distance)):
    erro = (distance[i][labels[i]])**2
    if(labels[i]==0):
        blue.append(erro)
    if(labels[i]==1):
        green.append(erro)
    if(labels[i]==2):
        orange.append(erro)
    if(labels[i]==3):
        red.append(erro)
blue.sort(reverse=True)
green.sort(reverse=True)
orange.sort(reverse=True)
red.sort(reverse=True)

plt.plot(blue, color='blue')
plt.plot(green, color='green')
plt.plot(orange, color='orange')
plt.plot(red, color='red')
plt.show()