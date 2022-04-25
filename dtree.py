import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('data/data_perm.txt', sep=" ", header=None,
                names=["category","az_d","el_d", "az_t", "el_t", "length","tension"])

y = df["category"].to_numpy()
x = df[["az_d","el_d", "az_t", "el_t", "length","tension"]].to_numpy()

x = x.astype('float32')
x = x.astype('int')

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Creates classifier
classifier = DecisionTreeClassifier()

# Trains classifier
classifier.fit(X_train, y_train)

# Prediction on test data
y_pred = classifier.predict(X_test)

# Classifier performance
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))