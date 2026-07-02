# 🎌 Anime Recommendation System From Scratch

A complete Anime Recommendation System built entirely from scratch using **NumPy**, without relying on deep learning frameworks such as PyTorch or TensorFlow.

The project includes a custom neural network framework, embedding layers, backpropagation, optimization, and a recommendation model trained on anime ratings.

---

# Features

- Neural Network implemented completely from scratch
- Forward Propagation
- Backpropagation
- Gradient Descent (SGD)
- Dense Layers
- ReLU Activation
- Sigmoid Activation
- Softmax Activation
- Embedding Layer
- Categorical Cross Entropy
- Mean Squared Error
- Binary Cross Entropy
- Mini-batch Training
- Model Saving & Loading
- Anime Recommendation Pipeline

---

# Project Structure

```
Anime-Recommendation-System/

│
├── NN/
│   ├── activation.py
│   ├── layers.py
│   ├── losses.py
│   ├── model.py
│   └── optimization.py
│
├── embeddings.py
├── train.py
├── recommend.py
│
├── saved_model/
│   ├── model.npz
│   └── mappings.pkl
│
├── anime.csv
├── ratings1.csv
│
└── README.md
```

---

# Neural Network Framework

Instead of using existing libraries, every component of the neural network was implemented manually.

Implemented components include

- Dense Layer
- ReLU
- Sigmoid
- Softmax
- Embedding Layer
- Forward Propagation
- Backpropagation
- Loss Functions
- Optimizer

---

# Recommendation Model

Unlike image classification, the recommender predicts

```
(User, Anime)
        │
        ▼
 Predicted Rating
```

The network architecture is

```
User ID
    │
Embedding
    │
    ├──────────────┐
                   │
Anime ID           │
    │              │
Embedding          │
    └──────┬───────┘
           │
     Concatenate
           │
     Dense(64→128)
           │
         ReLU
           │
     Dense(128→64)
           │
         ReLU
           │
      Dense(64→1)
           │
        Sigmoid
           │
Predicted Rating
```

---

# Dataset

Two datasets are used.

## Ratings Dataset

Contains user interactions.

Columns

- user_id
- anime_id
- rating
- user_index
- anime_index

Example

| User | Anime | Rating |
|------|-------|--------|
| 0 | Naruto | 9 |
| 0 | Death Note | 10 |

---

## Anime Dataset

Contains metadata.

Columns include

- MAL_ID
- Name
- Genres
- Episodes
- Type
- Score

Used for displaying recommendations.

---

# Data Preprocessing

The preprocessing pipeline performs

- Read ratings dataset
- Normalize ratings from 1–10 to 0–1
- Encode User IDs
- Encode Anime IDs
- Generate user indices
- Generate anime indices

---

# Training Pipeline

For every mini-batch

```
User IDs
Anime IDs

↓

Embedding Lookup

↓

Concatenate

↓

Forward Pass

↓

Loss Calculation

↓

Backpropagation

↓

Update

• Dense Weights
• Biases
• Embedding Weights
```

---

# Saving the Model

After training, the following are saved

- Dense layer weights
- Dense layer biases
- User embeddings
- Anime embeddings
- User mapping
- Anime mapping

This allows recommendations to be generated without retraining.

---

# Results

Training Loss

```
Epoch 1  | Loss: 65.7884
Epoch 2  | Loss: 17.7868
Epoch 3  | Loss: 8.0169
Epoch 4  | Loss: 4.5274
Epoch 5  | Loss: 2.8920
Epoch 6  | Loss: 1.9985
Epoch 7  | Loss: 1.4631
Epoch 8  | Loss: 1.1209
Epoch 9  | Loss: 0.8901
Epoch 10 | Loss: 0.7294
Epoch 11 | Loss: 0.6146
Epoch 12 | Loss: 0.5305
Epoch 13 | Loss: 0.4675
Epoch 14 | Loss: 0.4199
Epoch 15 | Loss: 0.3835
Epoch 16 | Loss: 0.3555
Epoch 17 | Loss: 0.3333
Epoch 18 | Loss: 0.3158
Epoch 19 | Loss: 0.3018
Epoch 20 | Loss: 0.2911
```

The steady decrease in training loss demonstrates that the custom neural network successfully learns user–anime interactions.

---

# Current Capabilities

- Train a recommendation model from scratch
- Learn user embeddings
- Learn anime embeddings
- Predict ratings for user–anime pairs
- Save trained models
- Load trained models
- Recommend unseen anime for existing users

---

# Future Improvements

- Streamlit Web Interface
- Cold-start recommendation for new users
- Anime search
- Similar Anime using embedding similarity
- Cosine Similarity Visualization
- t-SNE/PCA Embedding Visualization
- Recommendation Evaluation (RMSE, MAE, Precision@K)

---

# Technologies Used

- Python
- NumPy
- Pandas
- Streamlit (planned)
- MyAnimeList Dataset

---

# Learning Outcomes

This project demonstrates understanding of

- Neural Networks
- Backpropagation
- Gradient Descent
- Matrix Calculus
- Embeddings
- Recommendation Systems
- NumPy-based Deep Learning
- Model Serialization
- Machine Learning Pipeline Design

---

## Author

**Darshan**

Built as part of a journey to implement modern Machine Learning systems completely from scratch before using high-level frameworks.
