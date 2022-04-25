import pandas as pd
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('data/data_perm.txt', sep=" ", header=None,
                names=["category","az_d","el_d", "az_t", "el_t", "length","tension"])

y = df["category"].to_numpy()
x = df[["az_d","el_d", "az_t", "el_t", "length","tension"]].to_numpy()

x = x.astype('float32')
x = x.astype('int')

# Dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# define the model
model = BaggingClassifier()

# fit the model on the whole dataset
model.fit(X_train, y_train)

# make a single prediction
y_pred = model.predict(X_test)
print(y_pred)
# print('Predicted Class: %d' % ypred)

# Classifier performance
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))