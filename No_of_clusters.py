import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = pd.read_csv("E:/file3.csv")
data.head()

data.describe()

mms = MinMaxScaler()
print(mms)
mms.fit(data)

data_transformed = mms.transform(data)
print(data_transformed)
Sum_of_squared_distances = []
K = range(1,15)

for k in K:
     km = KMeans(n_clusters=k)
     km = km.fit(data_transformed)
     Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')

plt.show()
