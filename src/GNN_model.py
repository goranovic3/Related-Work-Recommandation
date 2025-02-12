import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import random
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np

from tqdm import tqdm


class GNNEncoder(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = HeteroConv({
            ('paper','cites','paper'): SAGEConv((-1,-1), hidden_dim),
            ('paper','rev_writes','author'): SAGEConv((-1,-1), hidden_dim)
        }, aggr='sum')
        self.conv2 = HeteroConv({
            ('paper','cites','paper'): SAGEConv((hidden_dim,hidden_dim), out_dim),
            ('paper','rev_writes','author'): SAGEConv((hidden_dim,hidden_dim), out_dim)
        }, aggr='sum')
    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k,v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(2*in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)
    def forward(self, z_u, z_v):
        z = torch.cat([z_u, z_v], dim=-1)
        z = F.relu(self.lin1(z))
        return torch.sigmoid(self.lin2(z)).view(-1)

class CitationLinkModel(nn.Module):
    def __init__(self, hidden_dim, out_dim, link_hidden_dim):
        super().__init__()
        logger.info("Initializing CitationLinkModel...")
        self.encoder = GNNEncoder(hidden_dim, out_dim)
        self.link_predictor = LinkPredictor(out_dim, link_hidden_dim)
    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        z_paper = z_dict['paper']
        src = edge_label_index[0]
        dst = edge_label_index[1]
        z_u = z_paper[src]
        z_v = z_paper[dst]
        return self.link_predictor(z_u, z_v)

def train_link_predictor(model, data, train_edge_index, train_edge_label, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict, train_edge_index)
    loss = F.binary_cross_entropy(out, train_edge_label)
    loss.backward()
    optimizer.step()
    return float(loss.item())

@torch.no_grad()
def test_link_predictor(model, data, edge_index, edge_label):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict, edge_index)
    loss = F.binary_cross_entropy(out, edge_label)
    pred = (out > 0.5).float()
    acc = (pred == edge_label).float().mean() if len(edge_label) > 0 else float('nan')
    return float(loss.item()), float(acc.item()), pred

def recommend_citations(model, data, paper_idx, k=5):
    model.eval()
    with torch.no_grad():
        z_dict = model.encoder(data.x_dict, data.edge_index_dict)
        z_paper = z_dict['paper']
    num_papers = data['paper'].num_nodes
    all_idx = torch.arange(num_papers)
    all_idx = all_idx[all_idx != paper_idx]
    src = all_idx
    dst = torch.full_like(src, paper_idx)
    z_u = z_paper[src]
    z_v = z_paper[dst]
    link_probs = model.link_predictor(z_u, z_v)
    sorted_indices = torch.argsort(link_probs, descending=True)
    top_indices = sorted_indices[:k]
    return all_idx[top_indices].tolist(), link_probs[top_indices].tolist()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true.cpu(), y_pred.cpu())
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def visualize_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        z_dict = model.encoder(data.x_dict, data.edge_index_dict)
        z_paper = z_dict['paper']
    if z_paper.size(1) < 2:
        logger.info("Embedding dimension < 2; skipping PCA visualization.")
        return
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_paper.cpu().numpy())
    plt.figure(figsize=(6,5))
    plt.scatter(z_2d[:,0], z_2d[:,1], s=40, alpha=0.8, c='blue', edgecolors='black')
    plt.title("2D PCA of Paper Embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()

def run_link_prediction_training(
    data,
    train_edge_index, train_edge_label,
    val_edge_index, val_edge_label,
    test_edge_index, test_edge_label,
    hidden_dim=16, out_dim=16, link_hidden_dim=16,
    epochs=10, lr=0.01
):
    logger.info("Starting training...")
    model = CitationLinkModel(hidden_dim, out_dim, link_hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses, val_accs = [], [], []

    for epoch in tqdm(range(1, epochs+1), desc="Training epochs"):
        train_loss = train_link_predictor(model, data, train_edge_index, train_edge_label, optimizer)
        val_loss, val_acc, _ = test_link_predictor(model, data, val_edge_index, val_edge_label)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        logger.info(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    test_loss, test_acc, test_pred = test_link_predictor(model, data, test_edge_index, test_edge_label)
    logger.info(f"Final Test | test_loss={test_loss:.4f} | test_acc={test_acc:.4f}")

    plt.figure(figsize=(7,5))
    plt.plot(range(1, epochs+1), train_losses, marker='o', label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, marker='s', label='Val Loss')
    plt.title("Training/Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(7,5))
    plt.plot(range(1, epochs+1), val_accs, marker='^', color='green', label='Val Accuracy')
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, test_pred
