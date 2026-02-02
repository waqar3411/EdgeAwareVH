#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 13:39:40 2025

@author: waqar
"""

# model_hypothesis_scorer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from torch_geometric.nn import GATConv

from torch_geometric.utils import degree


class EdgeAwareVH(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden=128):
        super().__init__(aggr='add')
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, node_dim),
        )
        self.att_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1),
        )

    def forward(self, x, edge_index, edge_attr, node_type):
        src, dst = edge_index
        mask = (node_type[src] == 0) & (node_type[dst] == 1)  # view->hyp only
        if not torch.any(mask):
            return x
        return self.propagate(edge_index[:, mask], x=x, edge_attr=edge_attr[mask])

    def message(self, x_j, edge_attr, index):
        z = torch.cat([x_j, edge_attr], dim=-1)
        m = self.msg_mlp(z)
        a = self.att_mlp(z).squeeze(-1)
        a = softmax(a, index)            # normalize per destination hyp
        return m * a.unsqueeze(-1)

    def update(self, aggr_out, x):
        out = x.clone()
        out += aggr_out
        return out

class HypothesisScorer(nn.Module):
    def __init__(self, in_node_dim=6, in_edge_dim=3, hidden=128, layers=3):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_node_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.blocks = nn.ModuleList([EdgeAwareVH(hidden, in_edge_dim, hidden) for _ in range(layers)])
        self.norms  = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head   = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.ReLU(),
#             nn.Linear(hidden//2, hidden//4), nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )

    def forward(self, data):
        h = self.enc(data.x)
        for blk, ln in zip(self.blocks, self.norms):
            h_res = h
            h = blk(h, data.edge_index, data.edge_attr, data.node_type)
            h = ln(h)
            h = F.relu(h)
            h = h + h_res
        hyp_h = h[data.hypoth_idx]           # [Nh, hidden]
        logits = self.head(hyp_h).squeeze(-1)# [Nh]
        return logits
    
    



class HypothesisGAT(torch.nn.Module):
    def __init__(self, in_node_dim=6, hidden=128, heads=4, layers=3):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_node_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        self.gat_layers = nn.ModuleList()
        for _ in range(layers):
            self.gat_layers.append(GATConv(hidden, hidden, heads=heads, concat=False))
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, data):
        x = self.enc(data.x)
        for gat, norm in zip(self.gat_layers, self.norms):
            x_res = x
            x = gat(x, data.edge_index)
            x = F.relu(norm(x))
            x = x + x_res  # residual connection

        # Keep only hypothesis nodes (just like before)
        hyp_x = x[data.hypoth_idx]
        logits = self.head(hyp_x).squeeze(-1)  # [Nh]
        return logits






class ECCConv(MessagePassing):
    """
    Edge-Conditioned Convolution (ECC)
    Simonovsky & Komodakis, CVPR 2017
    h_i' = sum_j W(e_ij) * h_j
    where W(e_ij) is predicted by an MLP conditioned on the edge attributes.
    """
    def __init__(self, node_dim, edge_dim, hidden=128):
        super().__init__(aggr='add')
        # MLP that maps edge_attr -> edge-specific transformation matrix
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, node_dim * node_dim)  # flatten weight matrix
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # edge_attr: [E, edge_dim]
        W_flat = self.edge_mlp(edge_attr)          # [E, node_dim*node_dim]
        node_dim = x_j.size(-1)
        W = W_flat.view(-1, node_dim, node_dim)    # [E, node_dim, node_dim]
        msg = torch.bmm(W, x_j.unsqueeze(-1)).squeeze(-1)  # [E, node_dim]
        return msg

    def update(self, aggr_out):
        return aggr_out


class HypothesisECC(nn.Module):
    """
    ECC-based model for hypothesis scoring.
    Mirrors the structure of your EdgeAwareVH and GAT models.
    """
    def __init__(self, in_node_dim=6, in_edge_dim=3, hidden=128, layers=3):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_node_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        self.ecc_layers = nn.ModuleList([
            ECCConv(hidden, in_edge_dim, hidden) for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, data):
        x = self.enc(data.x)

        for ecc, norm in zip(self.ecc_layers, self.norms):
            x_res = x
            x = ecc(x, data.edge_index, data.edge_attr)
            x = norm(F.relu(x))
            x = x + x_res  # residual connection

        hyp_x = x[data.hypoth_idx]  # only hypothesis nodes
        logits = self.head(hyp_x).squeeze(-1)
        return logits
