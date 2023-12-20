# GINet_Feat: torch geometric data -> feature embedding
# Feat_MTL: feature embedding -> list of predictions
# MolCLR Paper: https://www.nature.com/articles/s42256-022-00447-x

import torch
from torch import nn
import torch.nn.functional as functional

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

num_atom_type = 119  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5  # including aromatic and self-loop edge
num_bond_direction = 3


class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINet_Feat(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """

    def __init__(
        self,
        drop_ratio,
        pool,
    ):
        super(GINet_Feat, self).__init__()
        self.num_layer = 5  # pretrained
        self.emb_dim = 300  # pretrained
        self.feat_dim = 512  # pretrained
        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(
            num_atom_type, self.emb_dim
        )  # num_atom_type -> src/ginet_finetune.py
        self.x_embedding2 = nn.Embedding(num_chirality_tag, self.emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(self.num_layer):
            self.gnns.append(GINEConv(self.emb_dim))  # -> src/ginet_finetune.py

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(self.num_layer):
            self.batch_norms.append(nn.BatchNorm1d(self.emb_dim))

        if pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        elif pool == "add":
            self.pool = global_add_pool
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

    def forward(self, databatch):
        x = databatch.x
        edge_index = databatch.edge_index
        edge_attr = databatch.edge_attr

        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer != self.num_layer - 1:
                h = functional.dropout(
                    functional.relu(h), self.drop_ratio, training=self.training
                )
            else:
                h = functional.dropout(h, self.drop_ratio, training=self.training)

        h = self.pool(h, databatch.batch)
        h = self.feat_lin(h)  # just before prediction head

        return h

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)


class Feat_MTL(nn.Module):
    def __init__(
        self,
        feat_dim,
        num_layer,
        num_tasks,
        pred_act,
    ):
        super(Feat_MTL, self).__init__()
        self.num_layer = num_layer
        self.feat_dim = feat_dim
        self.out_dim = num_tasks

        self.pred_heads = nn.ModuleList()
        for _ in range(num_tasks):
            if pred_act == "relu":
                pred_head = [
                    nn.Linear(self.feat_dim, self.feat_dim // 2),
                    nn.ReLU(inplace=True),
                ]
                for _ in range(self.num_layer - 1):
                    pred_head.extend(
                        [
                            nn.Linear(self.feat_dim // 2, self.feat_dim // 2),
                            nn.ReLU(inplace=True),
                        ]
                    )
            elif pred_act == "softplus":
                pred_head = [
                    nn.Linear(self.feat_dim, self.feat_dim // 2),
                    nn.Softplus(),
                ]
                for _ in range(self.num_layer - 1):
                    pred_head.extend(
                        [
                            nn.Linear(self.feat_dim // 2, self.feat_dim // 2),
                            nn.Softplus(),
                        ]
                    )
            else:
                raise ValueError("Undefined activation function")

            pred_head.append(nn.Linear(self.feat_dim // 2, self.out_dim))
            self.pred_heads.append(nn.Sequential(*pred_head))

    def forward(self, data):
        return torch.cat([head(data) for head in self.pred_heads], dim=1)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)


class GINet_Feat_MTL(nn.Module):
    """
    Args:
        drop_ratio (float): dropout rate
    Output:
        node representations
    """

    def __init__(
        self,
        pool,  # 'mean', 'max', 'add'
        drop_ratio,  # 0 to 1
        pred_layer_depth,
        pred_act,  # 'relu', 'softplus'
        num_tasks,
    ):
        super(GINet_Feat_MTL, self).__init__()
        feat_dim = 512  # out dimension of gin. pretrained

        self.gin = GINet_Feat(pool=pool, drop_ratio=drop_ratio)
        self.mtl = Feat_MTL(
            feat_dim=feat_dim,
            num_layer=pred_layer_depth,
            num_tasks=num_tasks,
            pred_act=pred_act,
        )

    def forward(self, data):
        feat = self.gin(data)
        out = self.mtl(feat)
        return out

    def load_my_state_dict(self, state_dict):
        self.gin.load_my_state_dict(state_dict)
        self.mtl.load_my_state_dict(state_dict)


# Pretrained MolCLR Model
def load_pre_trained_weights(model, device, location=None):
    if location:
        checkpoint_file = location
    else:
        try:
            checkpoints_folder = os.path.join("./ckpt", "pretrained_gin", "checkpoints")
            checkpoint_file = os.path.join(checkpoints_folder, "model.pth")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    state_dict = torch.load(checkpoint_file, map_location=device)
    model.load_my_state_dict(state_dict)
    print("Loaded pre-trained model with success.")

    return model
