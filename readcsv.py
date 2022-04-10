import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/data_perm.txt', sep=" ", header=None,
                names=["category","az_d","el_d", "az_t", "el_t", "length","tension"])

y = df["category"].to_numpy()
x = df[["az_d","el_d", "az_t", "el_t", "length","tension"]].to_numpy()

x = x.astype('float32')
x = x.astype('int')

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# print(x.type())

# print(x.type())

print("===")
print(df)
print(y_test.shape)
print(y_train.shape)
print(X_test.shape)
print(X_train.shape)
print("---")


# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')

# print("Feature matrix:", x_train.shape)
# print("Target matrix:", x_test.shape)
# print("Feature matrix:", y_train.shape)
# print("Target matrix:", y_test.shape)