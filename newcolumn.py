import pandas as pd
import numpy as np

data = pd.read_csv("E:/file7.csv")
print(data.head())

data['Probability'] = np.where(data['POR']>6,'yes','no')

print(data.head())

data.to_csv("E:/file97.csv")
