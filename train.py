import numpy as np
import pandas as pd

from LayerEmbedding import LayerEmbedding

from NN.model import Model
from NN.layers import LayerDense
from NN.activation import ActivationReLU, ActivationSigmoid
from NN.losses import Loss_MeanSquaredError
from NN.optimization import SGD


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("ratings1.csv")

batch_size = 64
epochs = 20
embedding_dim = 32

# Encode IDs
user_map = {u: i for i, u in enumerate(df["user_id"].unique())}
anime_map = {a: i for i, a in enumerate(df["anime_id"].unique())}

df["user_index"] = df["user_id"].map(user_map)
df["anime_index"] = df["anime_id"].map(anime_map)

num_users = len(user_map)
num_anime = len(anime_map)

# -----------------------------
# Embeddings
# -----------------------------
user_embedding = LayerEmbedding(num_users, embedding_dim)
anime_embedding = LayerEmbedding(num_anime, embedding_dim)

# -----------------------------
# Model
# -----------------------------
model = Model()

model.add(LayerDense(64, 128))
model.add(ActivationReLU())

model.add(LayerDense(128, 64))
model.add(ActivationReLU())

model.add(LayerDense(64, 1))
model.add(ActivationSigmoid())

loss_function = Loss_MeanSquaredError()

optimizer = SGD(learning_rate=0.01)

model.set(
    loss=loss_function,
    optimizer=optimizer
)

# -----------------------------
# Training
# -----------------------------
for epoch in range(epochs):

    shuffled = df.sample(frac=1).reset_index(drop=True)

    epoch_loss = 0.0

    for start in range(0, len(shuffled), batch_size):

        batch = shuffled.iloc[start:start + batch_size]

        user_ids = batch["user_index"].to_numpy(np.int32)
        anime_ids = batch["anime_index"].to_numpy(np.int32)

        ratings = batch["rating"].to_numpy(np.float32).reshape(-1, 1)
        ratings /= 10.0

        # Forward embeddings
        user_vectors = user_embedding.forward(user_ids)
        anime_vectors = anime_embedding.forward(anime_ids)

        X = np.concatenate((user_vectors, anime_vectors), axis=1)

        # Train batch
        loss = model.train_batch(X, ratings)

        # Backprop to embeddings
        dX = model.layers[0].dinputs

        d_user = dX[:, :embedding_dim]
        d_anime = dX[:, embedding_dim:]

        user_embedding.backward(d_user)
        anime_embedding.backward(d_anime)

        optimizer.update_params(user_embedding)
        optimizer.update_params(anime_embedding)

        epoch_loss += loss

    print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f}")

import os

os.makedirs("saved_model", exist_ok=True)

np.savez(
    "saved_model/model.npz",

    # Dense layer 1
    dense1_weights=model.layers[0].weights,
    dense1_biases=model.layers[0].biases,

    # Dense layer 2
    dense2_weights=model.layers[2].weights,
    dense2_biases=model.layers[2].biases,

    # Dense layer 3
    dense3_weights=model.layers[4].weights,
    dense3_biases=model.layers[4].biases,

    # Embeddings
    user_embeddings=user_embedding.weights,
    anime_embeddings=anime_embedding.weights
)
import pickle

with open("saved_model/mappings.pkl", "wb") as f:
    pickle.dump(
        {
            "user_map": user_map,
            "anime_map": anime_map
        },
        f
    )
