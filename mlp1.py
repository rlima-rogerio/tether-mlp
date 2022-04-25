# importing modules
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
<<<<<<< HEAD
from tensorflow.keras.regularizers import l2
=======
from tensorflow.keras.callbacks import ReduceLROnPlateau
>>>>>>> 1c6126b9f598dadeff7218b18547d3969750b743
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/data_perm.txt', sep=" ", header=None,
                names=["category","az_d","el_d", "az_t", "el_t", "length","tension"])

y = df["category"].to_numpy()
<<<<<<< HEAD
x = df[["az_d","el_d", "az_t", "el_t", "length"]].to_numpy()
=======
# x = df[["az_d", "el_d", "az_t", "el_t", "tension"]].to_numpy()
x = df[["az_d", "el_d", "az_t", "el_t", "length", "tension"]].to_numpy()
>>>>>>> 1c6126b9f598dadeff7218b18547d3969750b743

# Cast the records into float values
x = x.astype('float32')
x = x.astype('int')

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.70, random_state=42)

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
<<<<<<< HEAD
	# Dense(10, activation='relu',kernel_regularizer=l2(0.01)),
	Dense(64, activation='relu'),

	Dropout(0.5),
	
	# dense layer 2
	# Dense(5, activation='relu',kernel_regularizer=l2(0.01)),
	Dense(32, activation='relu'),
=======
	Dense(60, activation='relu'),
	Dropout(0.1),

	# dense layer 2
	Dense(25, activation='relu'),
	Dropout(0.2),
>>>>>>> 1c6126b9f598dadeff7218b18547d3969750b743

	Dropout(0.3),
  	# dense layer 3
<<<<<<< HEAD
	Dense(16, activation='relu'),
=======
	Dense(128, activation='relu'),
	Dropout(0.2),

	# dense layer 4
	# Dense(64, activation='relu'),
>>>>>>> 1c6126b9f598dadeff7218b18547d3969750b743
	
	# output layer
	Dense(3, activation='softmax'),
])

model.compile(optimizer='adam',
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy'])

<<<<<<< HEAD
model.fit(X_train, y_train, epochs=200,
		batch_size=64,
=======
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, min_lr=0.1)

model.fit(X_train, y_train, epochs=500,
		batch_size=32,
		callbacks=[reduce_lr],
>>>>>>> 1c6126b9f598dadeff7218b18547d3969750b743
		validation_split=0.2)

results = model.evaluate(X_test, y_test, verbose = 0)
print('test loss, test acc:', results)
