import pandas as pd
import numpy as np
x = pd.read_csv("E:/LOG.csv")

x = x[x["RHOB"]>0]

x["POR"] = np.sqrt((x["RHOB"]**2 + x["NPHI"]**2)/2)

x.to_csv("E:/file6.csv")

print(x)
