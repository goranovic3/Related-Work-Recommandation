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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperGraph:
    """
    Builds a heterogeneous graph of papers (and authors) from a DataFrame.
    Normalizes paper features to help avoid numeric instability.
    """
    def __init__(self, df):
        self.df = df
        self.build_graph()

    def build_graph(self):
        logger.info("Building HeteroData graph...")
        self.graph = HeteroData()

        paper_ids = list(self.df['id'].unique())
        author_names = list(set(author for authors in self.df['authors'] for author in authors))

        paper_index = {pid: i for i, pid in enumerate(paper_ids)}
        author_index = {author: i for i, author in enumerate(author_names)}

        paper_paper_edges = []
        author_paper_edges = []

        for _, row in self.df.iterrows():
            paper_id = row['id']
            p_idx = paper_index[paper_id]
            for ref_id in row['filtered_references']:
                if ref_id in paper_index:
                    ref_idx = paper_index[ref_id]
                    paper_paper_edges.append((ref_idx, p_idx))

            for author in row['authors']:
                if author in author_index:
                    a_idx = author_index[author]
                    author_paper_edges.append((a_idx, p_idx))

        self.graph['paper'].num_nodes = len(paper_ids)
        self.graph['author'].num_nodes = len(author_names)

        if len(paper_paper_edges) > 0:
            self.graph['paper','cites','paper'].edge_index = torch.tensor(
                paper_paper_edges, dtype=torch.long
            ).t().contiguous()
        else:
            self.graph['paper','cites','paper'].edge_index = torch.empty((2, 0), dtype=torch.long)

        if len(author_paper_edges) > 0:
            self.graph['author','writes','paper'].edge_index = torch.tensor(
                author_paper_edges, dtype=torch.long
            ).t().contiguous()
        else:
            self.graph['author','writes','paper'].edge_index = torch.empty((2, 0), dtype=torch.long)

        logger.info("Building paper feature matrix...")
        n_citation_tensor = torch.tensor(self.df['n_citation'].values, dtype=torch.float).unsqueeze(-1)
        venue_tensor = torch.tensor(self.df['venue_numeric'].values, dtype=torch.float).unsqueeze(-1)
        emb_list = []
        for emb in self.df['abstract_embeddings'].values:
            if isinstance(emb, str):
                import ast
                emb = ast.literal_eval(emb)
            emb_list.append(torch.tensor(emb, dtype=torch.float))
        emb_stack = torch.stack(emb_list, dim=0)

        paper_x = torch.cat([n_citation_tensor, venue_tensor, emb_stack], dim=-1)
        with torch.no_grad():
            paper_x = self._normalize_features(paper_x)

        self.graph['paper'].x = paper_x
        self.graph['author'].x = torch.zeros((len(author_names), paper_x.shape[1]))

        logger.info("Graph building complete.")

    def _normalize_features(self, x):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + 1e-9
        return (x - mean) / std

    def get_graph(self):
        return self.graph

def split_edges(edge_index, val_ratio=0.1, test_ratio=0.1, seed=42):
    edges = []
    if edge_index.size(1) > 0:
        torch.manual_seed(seed)
        arr = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        random.seed(seed)
        random.shuffle(arr)
        E = len(arr)
        n_val = int(E * val_ratio)
        n_test = int(E * test_ratio)
        val_edges = arr[:n_val]
        test_edges = arr[n_val:n_val+n_test]
        train_edges = arr[n_val+n_test:]
        edges = [train_edges, val_edges, test_edges]
    def to_tensor(e):
        if not e:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(e, dtype=torch.long).t().contiguous()
    if not edges:
        return to_tensor([]), to_tensor([]), to_tensor([])
    return tuple(to_tensor(e) for e in edges)

def get_edge_label_index_and_label(data, edge_type, pos_edge_index, num_neg_samples=None):
    E_pos = pos_edge_index.size(1)
    if E_pos == 0:
        return pos_edge_index, torch.empty(0, dtype=torch.float)
    if num_neg_samples is None:
        num_neg_samples = E_pos
    neg_edge_index = negative_sampling(
        edge_index=data[edge_type].edge_index,
        num_nodes=data['paper'].num_nodes,
        num_neg_samples=num_neg_samples,
        method='sparse'
    )
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    edge_label = torch.cat([
        torch.ones(E_pos),
        torch.zeros(neg_edge_index.size(1))
    ], dim=0)
    return edge_label_index, edge_label

def visualize_subgraph(data, num_papers=5):
    paper_edges = data['paper','cites','paper'].edge_index
    if paper_edges.size(1) == 0:
        logger.info("No paper->paper edges found. Skipping subgraph visualization.")
        return
    G = nx.DiGraph()
    all_p = torch.arange(data['paper'].num_nodes)
    selected = random.sample(all_p.tolist(), min(num_papers, data['paper'].num_nodes))
    selected_set = set(selected)
    for p in selected:
        G.add_node(f"paper_{p}", node_type='paper')
        mask = (paper_edges[1] == p)
        for s in paper_edges[0][mask].tolist():
            if s in selected_set:
                G.add_node(f"paper_{s}", node_type='paper')
                G.add_edge(f"paper_{s}", f"paper_{p}")
    a_edges = data['author','writes','paper'].edge_index
    for a, pp in zip(a_edges[0].tolist(), a_edges[1].tolist()):
        if pp in selected_set:
            a_node = f"author_{a}"
            p_node = f"paper_{pp}"
            if not G.has_node(a_node):
                G.add_node(a_node, node_type='author')
            if G.has_node(p_node):
                G.add_edge(a_node, p_node)
    colors = []
    for node in G.nodes:
        if G.nodes[node]['node_type'] == 'paper':
            colors.append('steelblue')
        else:
            colors.append('tomato')
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx(G, pos, node_color=colors, with_labels=True, font_size=8, edge_color='gray')
    plt.title("Small Subgraph Visualization")
    plt.axis('off')
    plt.show()

def visualize_full_graph(data, max_nodes=300):
    G = nx.DiGraph()
    paper_count = data['paper'].num_nodes
    author_count = data['author'].num_nodes
    total_nodes = paper_count + author_count
    if total_nodes == 0:
        logger.info("No nodes to visualize.")
        return
    paper_ids = list(range(paper_count))
    author_ids = list(range(author_count))
    if total_nodes > max_nodes:
        keep_papers = set(random.sample(paper_ids, min(len(paper_ids), max_nodes // 2)))
        keep_authors = set(random.sample(author_ids, min(len(author_ids), max_nodes // 2)))
    else:
        keep_papers = set(paper_ids)
        keep_authors = set(author_ids)
    for p in keep_papers:
        G.add_node(f"paper_{p}", node_type='paper', idx=p)
    for a in keep_authors:
        G.add_node(f"author_{a}", node_type='author', idx=a)
    pp_edges = data['paper','cites','paper'].edge_index
    if pp_edges.size(1) > 0:
        for s, t in zip(pp_edges[0].tolist(), pp_edges[1].tolist()):
            if s in keep_papers and t in keep_papers:
                G.add_edge(f"paper_{s}", f"paper_{t}", edge_type='paper2paper')
    ap_edges = data['author','writes','paper'].edge_index
    if ap_edges.size(1) > 0:
        for a, p in zip(ap_edges[0].tolist(), ap_edges[1].tolist()):
            if a in keep_authors and p in keep_papers:
                G.add_edge(f"author_{a}", f"paper_{p}", edge_type='author2paper')
    node_colors = []
    node_sizes = []
    for n, d in G.nodes(data=True):
        if d['node_type'] == 'paper':
            node_colors.append('steelblue')
            idx = d['idx']
            if data['paper'].x is not None and idx < data['paper'].x.shape[0]:
                # After normalization, we can scale up one of the features, e.g. n_citation
                citation_val = data['paper'].x[idx, 0].item()
                size = max(100, (citation_val*100 + 100))
                node_sizes.append(size)
            else:
                node_sizes.append(200)
        else:
            node_colors.append('tomato')
            node_sizes.append(100)
    edge_colors = []
    for u, v, d in G.edges(data=True):
        if d['edge_type'] == 'paper2paper':
            edge_colors.append('gray')
        else:
            edge_colors.append('green')
    pos = nx.spring_layout(G, seed=42, k=0.2, iterations=50)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=7)
    plt.title("Full Graph Visualization (Up to max_nodes)")
    plt.axis('off')
    plt.show()

