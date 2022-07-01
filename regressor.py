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

df1 = pd.read_csv('data/train_tetherVariables.csv', sep=",", header=None,
                names=["az_d","el_d", "az_t", "el_t", "length","tension"])

df2 = pd.read_csv('data/train_ground_truth.csv', sep=",", header=None,
                names=["t_true","x_true", "y_true", "z_true"])


x = df1[["az_d", "el_d", "az_t", "el_t", "length", "tension"]].to_numpy()
y = df2[["x_true", "y_true", "z_true"]].to_numpy()

# y = df1["category"].to_numpy()
# x = df[["az_d", "el_d", "az_t", "el_t", "tension"]].to_numpy()


# Cast the records into float values
# x = x.astype('float32')
# x = x.astype('int')
x = np.asarray(x).astype(np.float32)
y = np.asarray(y).astype(np.float32)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.90, random_state=42)

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
	Dense(8, activation='sigmoid'),
	# Dropout(0.1),

# 	# dense layer 2
	# Dense(25, activation='relu'),
	# Dropout(0.2),

# 	Dropout(0.3),
#   	# dense layer 3
# 	Dense(128, activation='relu'),
# 	Dropout(0.2),

# 	# dense layer 4
# 	# Dense(64, activation='relu'),
	
	# output layer
	Dense(3, activation='linear')
])

model.compile(optimizer='adam',
			loss='mean_absolute_error',
			metrics=['mean_squared_error'])

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, min_lr=0.1)

model.fit(X_train, y_train, epochs=750,
		batch_size=32,
		callbacks=[reduce_lr],
		validation_split=0.2)

results = model.evaluate(X_test, y_test, verbose = 0)
print('test loss, test acc:', results)

model.save('models' + '/' +'regressor.hdf5')
model.save_weights('models' + '/' + 'regressor_init.hdf5')
print(model.summary())