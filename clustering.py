import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans

#creating a dataset
x1 = np.random.normal(25, 5, 1000) #mean = 25, sd = 5, size = 1k
y1 = np.random.normal(25, 5 , 1000)

x2= np.random.normal(55,5,1000)
y2= np.random.normal(60,5,1000)

x3=np.random.normal(55,5,1000)
y3= np.random.normal(15,5,1000)

x = np.concatenate((x1,x2,x3),axis = 0)
y = np.concatenate((y1,y2,y3), axis = 0)

dictionary = {"x" : x, "y": y}

data = pd.DataFrame(dictionary)
#print(data.head(3))

#actual data divided by colors
plt.figure()
plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.xlabel("x")
plt.ylabel("y")
plt.title("values in data")
plt.show()

#k means division input
plt.figure()
plt.scatter(x1,y1, color = "black")
plt.scatter(x2,y2, color = "black")
plt.scatter(x3,y3, color = "black")
plt.xlabel("x")
plt.ylabel("y")
plt.title("k means input")
plt.show()

from sklearn.cluster import KMeans
wcss = []
for k in range (1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1,15), wcss)
plt.xticks(range(1,15))
plt.title("cluster number by k")
plt.ylabel("wcss")
plt.show()

#k = 3 is determined as best as expected by creating 3 different clusters

kmean = KMeans(n_clusters=3)
clusters = kmean.fit_predict(data)

data["label"] = clusters

#plotting k divided clusters
plt.figure()
plt.scatter(data.x[data.label ==0], data.y[data.label ==0], color ="red", label ="1")
plt.scatter(data.x[data.label ==1], data.y[data.label ==1], color ="blue", label ="2")
plt.scatter(data.x[data.label ==2], data.y[data.label ==2], color ="orange", label ="3")
plt.scatter(kmean.cluster_centers_[:,0], kmean.cluster_centers_[:,1], color = "black")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("clusters divided by 3 means")
plt.show()