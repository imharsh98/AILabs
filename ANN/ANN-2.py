# ==================================================================================

# Regression MLP using sequential API

# --- Loading the housing dataset from scikit-learn ---

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()


# --- Splitting data into training and test data ---

X_train_full, X_test, y_train_full, y_test = train_test_split(
housing.data, housing.target, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(
X_train_full, y_train_full, test_size=0.2)


# --- Standardizing the features ---

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# --- Build a regression model ---

from tensorflow import keras
model = keras.models.Sequential([
  keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
  keras.layers.Dense(1)
])
model.compile(loss="mean_squared_error", optimizer="sgd")


# --- Model training and testing ---

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)

X_new = X_test[:3] # pretend these are new instances
y_pred = model.predict(X_new)