# importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

modelXY = keras.models.load_model('models' + '/'  + 'classifier_xy.hdf5')
modelZ = keras.models.load_model('models' + '/'  + 'classifier_z.hdf5')
modelYaw = keras.models.load_model('models' + '/'  + 'classifier_yaw.hdf5')


dfXY = pd.read_csv('data/valid_singleFileXY.csv', sep=",", header=None,
                names=["category","az_d","el_d", "az_t", "el_t", "length","tension"])

dfZ = pd.read_csv('data/valid_singleFileZ.csv', sep=",", header=None,
                names=["category","az_d","el_d", "az_t", "el_t", "length","tension"])

dfYaw = pd.read_csv('data/valid_singleFileYaw.csv', sep=",", header=None,
                names=["category","az_d","el_d", "az_t", "el_t", "length","tension"])

df2 = pd.read_csv('data/valid_ground_truth.csv', sep=",", header=None,
                names=["t_true","x_true", "y_true", "z_true"])

x = dfXY[["az_d", "el_d", "az_t", "el_t", "length", "tension"]].to_numpy()
yXY = dfXY[["category"]].to_numpy()
yZ = dfZ[["category"]].to_numpy()
yYaw = dfYaw[["category"]].to_numpy()
t = df2[["t_true"]].to_numpy()

yXY_pred = modelXY.predict(x)
yZ_pred = modelZ.predict(x)
yYaw_pred = modelYaw.predict(x)

print('======')
print(len(yXY_pred.argmax(axis=1)))
print('======')

# plt.figure()
fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Position Prediction')
ax1.scatter(t, yXY, alpha = 0.5, s=0.5)
ax1.scatter(t, yXY_pred.argmax(axis=1), alpha = 0.5, s=0.5)
ax1.set(ylabel='xy')
ax1.grid(color = 'gray', linestyle = '--', linewidth = 0.2)

ax2.scatter(t, yZ, alpha = 0.5, s=0.5)
ax2.scatter(t, yZ_pred.argmax(axis=1), alpha = 0.5, s=0.5)
ax2.set(ylabel='z')
ax2.grid(color = 'gray', linestyle = '--', linewidth = 0.2)

ax3.scatter(t, yYaw, alpha = 0.5, s=0.5)
ax3.scatter(t, yYaw_pred.argmax(axis=1), alpha = 0.5, s=0.5)
ax3.set(xlabel='time (s)', ylabel='yaw')
ax3.grid(color = 'gray', linestyle = '--', linewidth = 0.2)

plt.savefig('figs/classifiers_performance.jpg', bbox_inches='tight')
plt.show()
plt.close()
