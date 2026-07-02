import numpy as np
import pandas as pd
import pickle

from LayerEmbedding import LayerEmbedding
from NN.model import Model
from NN.layers import LayerDense
from NN.activation import ActivationReLU, ActivationSigmoid


# -----------------------------
# Load datasets
# -----------------------------
ratings = pd.read_csv("ratings1.csv")
anime_df = pd.read_csv("anime.csv")

anime_lookup = anime_df.set_index("MAL_ID").to_dict("index")


# -----------------------------
# Load mappings
# -----------------------------
with open("saved_model/mappings.pkl", "rb") as f:
    mappings = pickle.load(f)

user_map = mappings["user_map"]
anime_map = mappings["anime_map"]


# -----------------------------
# Create embeddings
# -----------------------------
embedding_dim = 32

user_embedding = LayerEmbedding(len(user_map), embedding_dim)
anime_embedding = LayerEmbedding(len(anime_map), embedding_dim)


# -----------------------------
# Build model
# -----------------------------
model = Model()

model.add(LayerDense(64, 128))
model.add(ActivationReLU())

model.add(LayerDense(128, 64))
model.add(ActivationReLU())

model.add(LayerDense(64, 1))
model.add(ActivationSigmoid())


# -----------------------------
# Load trained weights
# -----------------------------
weights = np.load("saved_model/model.npz", allow_pickle=True)

model.layers[0].weights = weights["dense1_weights"]
model.layers[0].biases = weights["dense1_biases"]

model.layers[2].weights = weights["dense2_weights"]
model.layers[2].biases = weights["dense2_biases"]

model.layers[4].weights = weights["dense3_weights"]
model.layers[4].biases = weights["dense3_biases"]

user_embedding.weights = weights["user_embeddings"]
anime_embedding.weights = weights["anime_embeddings"]


# -----------------------------
# Ask user
# -----------------------------
user_id = int(input("Enter User ID: "))

if user_id not in user_map:
    print("User not found.")
    exit()

user_index = user_map[user_id]


# -----------------------------
# Already watched anime
# -----------------------------
watched = set(
    ratings.loc[
        ratings["user_id"] == user_id,
        "anime_id"
    ]
)


# -----------------------------
# Predict
# -----------------------------
recommendations = []

for anime_id, anime_index in anime_map.items():

    if anime_id in watched:
        continue

    user_vec = user_embedding.forward([user_index])
    anime_vec = anime_embedding.forward([anime_index])

    X = np.concatenate((user_vec, anime_vec), axis=1)

    score = model.predict(X)[0][0]

    recommendations.append((anime_id, score))


# -----------------------------
# Sort
# -----------------------------
recommendations.sort(key=lambda x: x[1], reverse=True)


# -----------------------------
# Print Top 10
# -----------------------------
print("\nTop Recommendations\n")

for anime_id, score in recommendations[:10]:

    if anime_id not in anime_lookup:
        continue

    anime = anime_lookup[anime_id]

    print("=" * 60)
    print("Name      :", anime["Name"])
    print("Genres    :", anime["Genres"])
    print("Episodes  :", anime["Episodes"])
    print("Type      :", anime["Type"])
    print("MAL Score :", anime["Score"])
    print("Predicted :", round(score * 10, 2))
