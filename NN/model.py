import numpy as np


class Model:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, X):
        output = X

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, output, y):
        self.loss.backward(output, y)
        dvalues = self.loss.dinputs

        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)

    def train_batch(self, X, y):
        output = self.forward(X)

        loss = self.loss.calculate(output, y)

        self.backward(output, y)

        self.optimizer.pre_update_params()

        for layer in self.layers:
            if hasattr(layer, "weights"):
                self.optimizer.update_params(layer)

        self.optimizer.post_update_params()

        return loss

    def train(self, X, y, epochs=100, print_every=1):
        for epoch in range(1, epochs + 1):

            loss = self.train_batch(X, y)

            if epoch % print_every == 0:
                print(f"Epoch {epoch} | Loss {loss:.4f}")

    def predict(self, X):
        return self.forward(X)

    def accuracy(self, X, y):
        output = self.predict(X)
        predictions = np.argmax(output, axis=1)

        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        return np.mean(predictions == y)

    def mse(self, X, y):
        output = self.predict(X)
        return np.mean((output - y) ** 2)

    def rmse(self, X, y):
        return np.sqrt(self.mse(X, y))