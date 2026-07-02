import numpy as np


class ActivationReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs


class ActivationSigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1.0 / (1.0 + np.exp(-inputs))
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues * (self.output * (1.0 - self.output))
        return self.dinputs


class ActivationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(
            zip(self.output, dvalues)
        ):
            single_output = single_output.reshape(-1, 1)

            jacobian = (
                np.diagflat(single_output)
                - np.dot(single_output, single_output.T)
            )

            self.dinputs[index] = np.dot(jacobian, single_dvalues)

        return self.dinputs