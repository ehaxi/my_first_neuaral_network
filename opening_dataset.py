import numpy as np

def load_dataset():
    with np.load("mnist.npz") as data:
        # Convert data pictures from RGB to Unit RGB
        X_train = data["x_train"].astype("float32") / 255

        # Reshape data from (60000, 28, 28) into (60000, 784)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2]))

        y_train = data["y_train"]

        # Сonverting to a multidimensional array
        y_train = np.eye(10)[y_train]

        return X_train, y_train