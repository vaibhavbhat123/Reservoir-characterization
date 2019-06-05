import pandas as pd
x = pd.read_csv("E:/file3.csv")
print(x.head())

from sklearn.cluster import KMeans
km = KMeans(n_clusters=5,init='k-means++',n_init=10)
km.fit(x)

df = km.fit_predict(x)

print(df)

x["cluster"] = df

print(x.head())
x1 = x.sort_values(by=['cluster'],ascending=False)

print(x1)
x1.to_csv('E:/file7.csv')

data = pd.read_csv('E:/file7.csv')

print(data)

c = km.cluster_centers_

print(c)
