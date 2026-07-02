import numpy as np
from .activation import ActivationSoftmax


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)


class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):

        samples = len(y_pred)

        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred[np.arange(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred * y_true, axis=1)

        return -np.log(correct_confidences)

    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        labels = dvalues.shape[1]

        dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs /= samples


class Loss_MeanSquaredError(Loss):

    def forward(self, y_pred, y_true):
        return np.mean((y_true - y_pred) ** 2, axis=-1)

    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        outputs = dvalues.shape[1]

        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs /= samples


class Loss_BinaryCrossentropy(Loss):

    def forward(self, y_pred, y_true):

        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_losses = -(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

        return np.mean(sample_losses, axis=-1)

    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        outputs = dvalues.shape[1]

        clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)

        self.dinputs = -(
            y_true / clipped -
            (1 - y_true) / (1 - clipped)
        ) / outputs

        self.dinputs /= samples


class Activation_Softmax_Loss_CategoricalCrossentropy:

    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):

        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[np.arange(samples), y_true] -= 1
        self.dinputs /= samples