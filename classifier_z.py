# importing modules
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

df1 = pd.read_csv('data/train_singleFileZ.csv', sep=",", header=None,
                names=["category","az_d","el_d", "az_t", "el_t", "length","tension"])


x = df1[["az_d", "el_d", "az_t", "el_t", "length", "tension"]].to_numpy()
y = df1[["category"]].to_numpy()


# Cast the records into float values
x = np.asarray(x).astype(np.float32)
y = np.asarray(y).astype(np.int)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.90, random_state=42)

print("Feature matrix:", X_train.shape)
print("Target matrix:", X_test.shape)
print("Feature vector:", y_train.shape)
print("Target vector:", y_test.shape)


model = Sequential([
	
	# reshape 28 row * 28 column data to 28*28 rows
	# Flatten(input_shape=(28, 28)),
	
	# dense layer 1
	Dense(8, activation='relu'),
	Dropout(0.1),

# 	# dense layer 2
	Dense(16, activation='relu'),
	Dropout(0.2),

  	# dense layer 3
	Dense(32, activation='relu'),
	Dropout(0.3),

# 	# dense layer 4
# 	# Dense(64, activation='relu'),
	
	# output layer
	Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, min_lr=0.1)

model.fit(X_train, y_train, epochs=500,
		batch_size=32,
		callbacks=[reduce_lr],
		validation_split=0.2)

results = model.evaluate(X_test, y_test, verbose = 0)
print('test loss, test acc:', results)

model.save('models' + '/' +'classifier_z.hdf5')
print(model.summary())