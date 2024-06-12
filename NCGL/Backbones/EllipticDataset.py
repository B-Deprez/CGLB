import os

os.environ["DGLBACKEND"] = "pytorch"  # tell DGL what backend to use
import dgl
import torch
from dgl.data import DGLDataset

import pandas as pd

class EllipticDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="elliptic")
        self.num_classes = 2#*49

    def process(self):
        node_data = pd.read_csv("NCGL/data/elliptic/elliptic_txs_features.csv", header=None)
        label_data = pd.read_csv("NCGL/data/elliptic/elliptic_txs_classes.csv")
        edge_data = pd.read_csv("NCGL/data/elliptic/elliptic_txs_edgelist.csv")

        map_id = {j:i for i,j in enumerate(node_data[0])}

        node_data[0] = node_data[0].map(map_id)
        label_data.txId = label_data.txId.map(map_id)   
        edge_data.txId1 = edge_data.txId1.map(map_id)
        edge_data.txId2 = edge_data.txId2.map(map_id)

        node_features = torch.from_numpy(node_data.drop(columns=[0, 1]).to_numpy())
        node_labels = torch.from_numpy(
            label_data["class"].astype("category").cat.codes.to_numpy()
            )

        edge_src = torch.from_numpy(edge_data["txId1"].to_numpy())
        edge_dst = torch.from_numpy(edge_data["txId2"].to_numpy())

        self.graph = dgl.graph(
            (edge_src, edge_dst), 
            num_nodes=node_data.shape[0]
            )
        self.graph.ndata["feat"] = node_features
        #self.graph.ndata["label"] = node_labels
        self.labels = node_labels

    def __getitem__(self, i):
        return self.graph, self.labels
    
    def __len__(self):
        return 1
    
dataset  = EllipticDataset()
graph = dataset[0]
print("Graph:")
print(graph)