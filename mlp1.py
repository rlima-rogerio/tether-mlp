# importing modules
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/data_perm.txt', sep=" ", header=None,
                names=["category","az_d","el_d", "az_t", "el_t", "length","tension"])

y = df["category"].to_numpy()
x = df[["az_d","el_d", "az_t", "el_t", "length","tension"]].to_numpy()

# Cast the records into float values
x = x.astype('float32')
x = x.astype('int')

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

print("Feature matrix:", X_train.shape)
print("Target matrix:", X_test.shape)
print("Feature vector:", y_train.shape)
print("Target vector:", y_test.shape)

# fig, ax = plt.subplots(10, 10)
# k = 0
# for i in range(10):
# 	for j in range(10):
# 		ax[i][j].imshow(X_train[k].reshape(28, 28),
# 						aspect='auto')
# 		k += 1
# plt.show()

model = Sequential([
	
	# reshape 28 row * 28 column data to 28*28 rows
	# Flatten(input_shape=(28, 28)),
	
	# dense layer 1
	Dense(256, activation='sigmoid'),
	
	# dense layer 2
	Dense(128, activation='sigmoid'),

  	# dense layer 3
	Dense(128, activation='sigmoid'),
	
	# output layer
	Dense(10, activation='sigmoid'),
])

model.compile(optimizer='adam',
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20,
		batch_size=2000,
		validation_split=0.2)

results = model.evaluate(X_test, y_test, verbose = 0)
print('test loss, test acc:', results)
