import numpy as np


class LayerEmbedding:
    def __init__(self, num_embeddings, embedding_dim):
        """
        Parameters
        ----------
        num_embeddings : int
            Number of unique IDs (users/anime/etc.)

        embedding_dim : int
            Size of each embedding vector.
        """

        self.weights = 0.01 * np.random.randn(
            num_embeddings,
            embedding_dim
        )

    def forward(self, indices):
        """
        indices : shape (batch_size,)
        """

        self.inputs = np.asarray(indices, dtype=np.int32).reshape(-1)

        self.output = self.weights[self.inputs]

        return self.output

    def backward(self, dvalues):
        """
        dvalues :
            Gradient coming from the next layer.
            Shape -> (batch_size, embedding_dim)
        """

        self.dweights = np.zeros_like(self.weights)

        # Accumulate gradients for repeated IDs
        np.add.at(self.dweights, self.inputs, dvalues)

        return None