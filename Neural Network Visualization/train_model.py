import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

print("Fetching MNIST dataset (this may take a minute the first time)...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
X, y = mnist.data.astype(np.float32), mnist.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print("Fitting scaler...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training MLPClassifier (hidden_layer_sizes=(32, 16))...")
model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation='relu',
    solver='adam',
    max_iter=40,
    random_state=42,
    verbose=True,
    early_stopping=True,
    n_iter_no_change=5,
)
model.fit(X_train_scaled, y_train)

acc = model.score(X_test_scaled, y_test)
print(f"\nTest accuracy: {acc * 100:.2f}%")

script_dir = os.path.dirname(os.path.abspath(__file__))
joblib.dump(model, os.path.join(script_dir, 'model.pkl'))
joblib.dump(scaler, os.path.join(script_dir, 'scaler.pkl'))
print("Saved model.pkl and scaler.pkl")
