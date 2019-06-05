#import all the libraries
import numpy as np
import pandas as pd
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("E:/file3.csv")
X = dataset.drop('POR',axis=1) #Predictor Variable
y = dataset['POR'] #Target Variable



# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.08, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) #Normalising training data
X_test = sc.transform(X_test) #Normalising testing data

# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(32, activation = 'relu', input_dim = 4))

# Adding the second hidden layer
model.add(Dense(units = 32, activation = 'relu'))

# Adding the third hidden layer
model.add(Dense(units = 32, activation = 'relu'))

# Adding the output layer

model.add(Dense(units = 1))

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=50)


# Fitting the ANN to the Training set
history=model.fit(X_train, y_train,validation_split=0.2, batch_size = 10, epochs = 10,callbacks=[es])

y_pred = model.predict(X_test) #Predicting Target using Test set

print("test=",X_test.shape)
print("Predicted porosity is= ",y_pred)

print("original porosity is= ",y_test)

y_final = np.column_stack((y_pred,y_test))
print("final=",y_final)

training_loss = history.history['loss']
test_loss = history.history['val_loss']

epoch_count = range(1, len(training_loss) + 1)

fig, ax = plt.subplots()
c = np.random.randint(1,3,size=255)

ax.scatter(y_final[:,0], y_final[:,1], c=c, edgecolors=(0, 0, 0))
ax.set_yscale('log')
ax.set_xscale('log')

#Plot Actual and Predicted Values 
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=4)

ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()



#Plot Loss vs Epoch Curve
fig
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

