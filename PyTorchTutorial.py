# app.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import itertools

from sklearn.datasets import make_circles, make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torch import nn


# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="Neural Network Playground", layout="wide", page_icon="🧠")

st.title("🧠 PyTorch Tutorial: A step-by-step walkthrough of building a neural network from scratch")
st.caption("An interactive demo to visualize how neural networks learn to classify non-linear data patterns.")

st.sidebar.title("⚙️ Control Panel")


# ======================
# SIDEBAR: DATA SETTINGS
# ======================
st.sidebar.subheader("📊 Data Settings")

data_type = st.sidebar.selectbox(
    "Dataset Type",
    ["make_circles", "make_moons", "make_classification"],
    help="Select a toy dataset for binary classification."
)

n_samples = st.sidebar.slider("Number of Samples", 500, 10000, 3000, step=500)
noise = st.sidebar.slider("Noise Level", 0.0, 0.4, 0.1, step=0.01)
test_size = st.sidebar.slider("Test Split Ratio", 0.1, 0.5, 0.33, step=0.05)
random_state = 26


# Generate Dataset
if data_type == "make_circles":
    X, y = make_circles(n_samples=n_samples, noise=noise, random_state=random_state)
elif data_type == "make_moons":
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
else:
    X, y = make_classification(
        n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, random_state=random_state
    )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# ======================
# SIDEBAR: MODEL SETTINGS
# ======================
st.sidebar.subheader("🤖 Model Settings")

hidden_dim = st.sidebar.slider("Hidden Layer Size", 5, 100, 20, step=5)
learning_rate = st.sidebar.number_input("Learning Rate", 0.0001, 1.0, 0.05, step=0.01, format="%.4f")
num_epochs = st.sidebar.slider("Epochs", 10, 500, 150, step=10)
batch_size = st.sidebar.slider("Batch Size", 16, 256, 64, step=16)

show_decision_boundary = st.sidebar.checkbox("Show Decision Boundary", value=True)


# ======================
# DATA PREPARATION
# ======================
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


train_data = Data(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = Data(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# ======================
# MODEL DEFINITION
# ======================
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.sigmoid(self.layer_2(x))
        return x


input_dim, output_dim = 2, 1
model = NeuralNetwork(input_dim, hidden_dim, output_dim)
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# ======================
# MAIN LAYOUT WITH TABS
# ======================
tab1, tab2, tab3 = st.tabs(["📘 Data Overview", "⚙️ Training Process", "📊 Model Evaluation"])


# ---- Tab 1: Data Overview ----
with tab1:
    st.subheader("Dataset Visualization")

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4), sharex=True, sharey=True)
    ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Spectral)
    ax1.set_title("Training Data")
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Spectral)
    ax2.set_title("Testing Data")
    st.pyplot(fig)

    st.write(f"**Samples:** {n_samples} | **Train/Test Split:** {1 - test_size:.2f}/{test_size:.2f}")
    st.write(f"**Noise:** {noise} | **Dataset Type:** {data_type}")


# ---- Tab 2: Training Process ----
with tab2:
    st.subheader("Model Training")

    if st.button("🚀 Start Training"):
        loss_values = []
        progress = st.progress(0)
        status = st.empty()

        for epoch in range(num_epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = loss_fn(preds, y_batch.unsqueeze(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            loss_values.append(avg_loss)
            progress.progress((epoch + 1) / num_epochs)
            status.text(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

        st.success("✅ Training Completed Successfully!")

        # Plot loss curve
        fig, ax = plt.subplots(figsize=(8, 4))
        plt.plot(range(len(loss_values)), loss_values, color='royalblue')
        plt.title("Loss Curve During Training")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        st.pyplot(fig)

        st.session_state["trained_model"] = model
        st.session_state["trained"] = True
    else:
        st.info("Click **Start Training** to begin model training.")


# ---- Tab 3: Evaluation ----
with tab3:
    st.subheader("Model Evaluation & Results")

    if "trained" in st.session_state and st.session_state["trained"]:
        model = st.session_state["trained_model"]
        y_true, y_pred = [], []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                preds = np.where(outputs.numpy() < 0.5, 0, 1)
                y_pred.extend(preds.flatten().tolist())
                y_true.extend(y_batch.numpy().tolist())

        accuracy = np.mean(np.array(y_pred) == np.array(y_true)) * 100
        st.metric(label="🎯 Model Accuracy", value=f"{accuracy:.2f}%")

        # Classification report
        st.text("Classification Report:")
        st.text(classification_report(y_true, y_pred))

        # Confusion Matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cf_matrix, annot=True, cmap="Blues", fmt="g", cbar=False)
        plt.title("Confusion Matrix")
        st.pyplot(fig)

        # Optional Decision Boundary
        if show_decision_boundary:
            st.subheader("🌀 Decision Boundary Visualization")
            xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 300), np.linspace(-1.5, 1.5, 300))
            grid = np.c_[xx.ravel(), yy.ravel()]
            with torch.no_grad():
                zz = model(torch.from_numpy(grid.astype(np.float32)))
                zz = zz.reshape(xx.shape).numpy()

            fig, ax = plt.subplots(figsize=(6, 5))
            plt.contourf(xx, yy, zz, cmap=plt.cm.Spectral, alpha=0.7)
            plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, s=15, edgecolors='k')
            plt.title("Decision Boundary (Test Data)")
            st.pyplot(fig)

    else:
        st.warning("⚠️ Please train the model first in the **Training Process** tab.")
