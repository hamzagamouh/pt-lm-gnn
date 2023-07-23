import argparse
from re import M

parser = argparse.ArgumentParser()
parser.add_argument("--val_batch_size", default=1000, type=int, help="Evaluation batch size.")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate")
parser.add_argument("--no_emb", default=False, type=bool, help="Whether to include embedding information")
parser.add_argument("--emb_name", default="t5", type=str, help="Embedding name : onehot, bert, xlnet")
parser.add_argument("--fold", default=0, type=str, help="Cross-Validation fold number (ex : fold_0_")
parser.add_argument("--output_folder", default="", type=str, help="Output folder : Training sets | Testing sets")
parser.add_argument("--ligand", default="", type=str, help="Ligand")

print("HELLO !")
import os
import time
import numpy as np
import pandas as pd
import pickle
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import zipfile
import random
import scipy
from sklearn.model_selection import KFold

assert torch.cuda.is_available() , "CUDA is NOT available !!"

DEVICE="cuda"

args = parser.parse_args([] if "__file__" not in globals() else None)
# 
import warnings
warnings.filterwarnings("ignore")



##__________________ Load data and make batches _____________________________________

# Data folder
data_folder=f"/home/files/yu_gnn"
EMB_NAME=args.emb_name
FEAT_SIZES={"onehot":21,
            "bert":1024,
            "xlnet":1024,
            "t5":1024,
            "aaindex":566}




# _____________________________ Model Architecture ______________________________

from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgllife.model.gnn.gcn import GCN

#  ch-run --bind /home:/home ~/files/biopython bash

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv

__all__ = ["GAT"]

# pylint: disable=W0221
class GATLayer(nn.Module):
    r"""Single GAT layer from `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`__

    Parameters
    ----------
    in_feats : int
        Number of input node features
    out_feats : int
        Number of output node features
    num_heads : int
        Number of attention heads
    feat_drop : float
        Dropout applied to the input features
    attn_drop : float
        Dropout applied to attention values of edges
    alpha : float
        Hyperparameter in LeakyReLU, which is the slope for negative values.
        Default to 0.2.
    residual : bool
        Whether to perform skip connection, default to True.
    agg_mode : str
        The way to aggregate multi-head attention results, can be either
        'flatten' for concatenating all-head results or 'mean' for averaging
        all head results.
    activation : activation function or None
        Activation function applied to the aggregated multi-head results, default to None.
    bias : bool
        Whether to use bias in the GAT layer.
    allow_zero_in_degree: bool
        Whether to allow zero in degree nodes in graph. Defaults to False.
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop,
        attn_drop,
        alpha=0.2,
        residual=True,
        agg_mode="flatten",
        activation=None,
        bias=True,
        allow_zero_in_degree=False,
    ):
        super(GATLayer, self).__init__()

        self.gat_conv = GATConv(
            in_feats=in_feats,
            out_feats=out_feats,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=alpha,
            residual=residual,
            bias=bias,
            allow_zero_in_degree=allow_zero_in_degree,
        )
        assert agg_mode in ["flatten", "mean"]
        self.agg_mode = agg_mode
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.gat_conv.reset_parameters()

    def forward(self, bg, feats):
        """Update node representations

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which equals in_feats in initialization

        Returns
        -------
        feats : FloatTensor of shape (N, M2)
            * N is the total number of nodes in the batch of graphs
            * M2 is the output node representation size, which equals
              out_feats in initialization if self.agg_mode == 'mean' and
              out_feats * num_heads in initialization otherwise.
        """
        feats, att_weights = self.gat_conv(bg, feats, get_attention=True)
        if self.agg_mode == "flatten":
            feats = feats.flatten(1)
        else:
            feats = feats.mean(1)

        if self.activation is not None:
            feats = self.activation(feats)

        return feats,att_weights


class GAT(nn.Module):
    r"""GAT from `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`__

    Parameters
    ----------
    in_feats : int
        Number of input node features
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the output size of an attention head in the i-th GAT layer.
        ``len(hidden_feats)`` equals the number of GAT layers. By default, we use ``[32, 32]``.
    num_heads : list of int
        ``num_heads[i]`` gives the number of attention heads in the i-th GAT layer.
        ``len(num_heads)`` equals the number of GAT layers. By default, we use 4 attention heads
        for each GAT layer.
    feat_drops : list of float
        ``feat_drops[i]`` gives the dropout applied to the input features in the i-th GAT layer.
        ``len(feat_drops)`` equals the number of GAT layers. By default, this will be zero for
        all GAT layers.
    attn_drops : list of float
        ``attn_drops[i]`` gives the dropout applied to attention values of edges in the i-th GAT
        layer. ``len(attn_drops)`` equals the number of GAT layers. By default, this will be zero
        for all GAT layers.
    alphas : list of float
        Hyperparameters in LeakyReLU, which are the slopes for negative values. ``alphas[i]``
        gives the slope for negative value in the i-th GAT layer. ``len(alphas)`` equals the
        number of GAT layers. By default, this will be 0.2 for all GAT layers.
    residuals : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GAT layer.
        ``len(residual)`` equals the number of GAT layers. By default, residual connection
        is performed for each GAT layer.
    agg_modes : list of str
        The way to aggregate multi-head attention results for each GAT layer, which can be either
        'flatten' for concatenating all-head results or 'mean' for averaging all-head results.
        ``agg_modes[i]`` gives the way to aggregate multi-head attention results for the i-th
        GAT layer. ``len(agg_modes)`` equals the number of GAT layers. By default, we flatten
        all-head results for each GAT layer.
    activations : list of activation function or None
        ``activations[i]`` gives the activation function applied to the aggregated multi-head
        results for the i-th GAT layer. ``len(activations)`` equals the number of GAT layers.
        By default, no activation is applied for each GAT layer.
    biases : list of bool
        ``biases[i]`` gives whether to use bias for the i-th GAT layer. ``len(activations)``
        equals the number of GAT layers. By default, we use bias for all GAT layers.
    allow_zero_in_degree: bool
        Whether to allow zero in degree nodes in graph for all layers. By default, will not
        allow zero in degree nodes.
    """

    def __init__(
        self,
        in_feats,
        hidden_feats=None,
        num_heads=None,
        feat_drops=None,
        attn_drops=None,
        alphas=None,
        residuals=None,
        agg_modes=None,
        activations=None,
        biases=None,
        allow_zero_in_degree=False,
    ):
        super(GAT, self).__init__()

        if hidden_feats is None:
            hidden_feats = [32, 32]

        n_layers = len(hidden_feats)
        if num_heads is None:
            num_heads = [4 for _ in range(n_layers)]
        if feat_drops is None:
            feat_drops = [0.0 for _ in range(n_layers)]
        if attn_drops is None:
            attn_drops = [0.0 for _ in range(n_layers)]
        if alphas is None:
            alphas = [0.2 for _ in range(n_layers)]
        if residuals is None:
            residuals = [True for _ in range(n_layers)]
        if agg_modes is None:
            agg_modes = ["flatten" for _ in range(n_layers - 1)]
            agg_modes.append("mean")
        if activations is None:
            activations = [F.elu for _ in range(n_layers - 1)]
            activations.append(None)
        if biases is None:
            biases = [True for _ in range(n_layers)]
        lengths = [
            len(hidden_feats),
            len(num_heads),
            len(feat_drops),
            len(attn_drops),
            len(alphas),
            len(residuals),
            len(agg_modes),
            len(activations),
            len(biases),
        ]
        assert len(set(lengths)) == 1, (
            "Expect the lengths of hidden_feats, num_heads, "
            "feat_drops, attn_drops, alphas, residuals, "
            "agg_modes, activations, and biases to be the same, "
            "got {}".format(lengths)
        )
        self.hidden_feats = hidden_feats
        self.num_heads = num_heads
        self.agg_modes = agg_modes
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(
                GATLayer(
                    in_feats,
                    hidden_feats[i],
                    num_heads[i],
                    feat_drops[i],
                    attn_drops[i],
                    alphas[i],
                    residuals[i],
                    agg_modes[i],
                    activations[i],
                    biases[i],
                    allow_zero_in_degree=allow_zero_in_degree,
                )
            )
            if agg_modes[i] == "flatten":
                in_feats = hidden_feats[i] * num_heads[i]
            else:
                in_feats = hidden_feats[i]

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which equals in_feats in initialization

        Returns
        -------
        feats : FloatTensor of shape (N, M2)
            * N is the total number of nodes in the batch of graphs
            * M2 is the output node representation size, which equals
              hidden_sizes[-1] if agg_modes[-1] == 'mean' and
              hidden_sizes[-1] * num_heads[-1] otherwise.
        """
        for gnn in self.gnn_layers:
            feats, att_weights = gnn(g, feats)
        return feats,att_weights



def softmax_fct(h):
    return F.softmax(h,1)


n_feats=FEAT_SIZES[EMB_NAME]

class GNN(nn.Module):
    def __init__(self, model_class,layers):
        super(GNN, self).__init__()
        self.model_class=model_class
        n_layers=len(layers)
        ACTIVATIONS=[F.relu for _ in range(n_layers)]
        DROPOUTS=[args.dropout for _ in range(n_layers)]
        NUM_HEADS=None
        if model_class=="gat":
            self.gnn=GAT(n_feats, hidden_feats=layers, activations=ACTIVATIONS, 
                        num_heads=NUM_HEADS, feat_drops =DROPOUTS)
        if model_class=="gcn":
            self.gnn=GCN(n_feats,hidden_feats=layers,
                     residual=[True]*n_layers, batchnorm=[True]*n_layers,
                     dropout=[args.dropout]*n_layers)
            
        # self.gnn=GAT(n_feats, hidden_feats=layers, activations=ACTIVATIONS, 
        #             num_heads=NUM_HEADS, feat_drops =DROPOUTS)
    
        self.dense1=nn.Linear(layers[-1], 2)

    def forward(self, g, in_feat):
        if self.model_class=="gat": 
            h_feats,att_weights=self.gnn(g, in_feat)
        if self.model_class=="gcn": 
            h_feats=self.gnn(g, in_feat)
            att_weights=0
        h=self.dense1(h_feats)
        h = F.softmax(h,1)
        return h,att_weights

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.dense1=nn.Linear(n_feats, 64)
        self.dense2=nn.Linear(64, 64)
        self.dense3=nn.Linear(64, 64)
        self.dense4=nn.Linear(64, 2)

    def forward(self, g, in_feat):
        h=in_feat
        h=self.dense1(h)
        h = F.relu(h)
        h=F.dropout(h,args.dropout)
        h=self.dense2(h)
        h = F.relu(h)
        h=F.dropout(h,args.dropout)
        h=self.dense3(h)
        h = F.relu(h)
        h=F.dropout(h,args.dropout)
        h=self.dense4(h)
        h = F.softmax(h,1)
        return h
# ____________________________ Util functions _____________________________


def get_protein(LIGAND,CUTOFF,i):
    mode="Validation"
    folder="Testing_sets"
    with zipfile.ZipFile(data_folder+f'/{folder}/all_embs_{LIGAND}_{mode}_th_{CUTOFF}.zip','r') as thezip:
        file =thezip.namelist()[i]
        dest=f"{data_folder}/fold{args.fold}/{file}"
        thezip.extract(file,f"{data_folder}/fold{args.fold}")
        g=pickle.load(open(dest,'rb'))
        os.remove(dest)
        g=g.to(DEVICE)
        features = g.ndata[PREFIX+'feat'].to(DEVICE)
        labels = g.ndata['label'].to(DEVICE)
        return file,g,features,labels


PREFIX=""
if EMB_NAME=="bert":
    PREFIX='bert_'
if EMB_NAME=="aaindex":
    PREFIX='pc_'



def to_numpy_arr(t):
    return t.detach().cpu().numpy()


from scipy.sparse import lil_matrix
from sklearn.preprocessing import normalize

def preprocess_attention(edge_atten, g, to_normalize=True):
    """Organize attentions in the form of csr sparse adjacency
    matrices from attention on edges. 

    Parameters
    ----------
    edge_atten : numpy.array of shape (# edges, # heads, 1)
        Un-normalized attention on edges.
    g : dgl.DGLGraph.
    to_normalize : bool
        Whether to normalize attention values over incoming
        edges for each node.
    """
    edge_atten=to_numpy_arr(edge_atten)
    n_nodes = g.number_of_nodes()
    num_heads = edge_atten.shape[1]
    all_head_A = [lil_matrix((n_nodes, n_nodes)) for _ in range(num_heads)]
    for i in range(n_nodes):
        predecessors = to_numpy_arr(g.predecessors(i))
        edges_id = to_numpy_arr(g.edge_ids(predecessors, i))
        for j in range(num_heads):
            all_head_A[j][i, predecessors] = edge_atten[edges_id, j, 0]#.data.numpy()#.cpu().numpy()


    for j in range(num_heads):
        if j==0:
            final_A=all_head_A[j]
        else:
            final_A+=all_head_A[j]    
    final_A=final_A/num_heads
    if to_normalize:
        final_A = normalize(final_A, norm='l1').tocsr()
    return final_A
       


if args.no_emb:
    print("No Embedding !!")
else:
    print("Embedding : ",EMB_NAME)

print(f"CV Fold : {args.fold}")

print("Dropout rate : ",args.dropout)




if args.ligand=="":
    LIGANDS=["ADP", "AMP", "ATP", "CA", "DNA", "FE", "GDP", "GTP", "HEME", "MG", "MN", "ZN"]
else:
    LIGANDS=[args.ligand]

CUTOFFS=[4,6,8,10]

METRICS=["roc-auc","precision","recall","mcc"]
LAYERS={"gat":[512]*1,
        "gcn":[512]*1}


DF={"Ligand":[],"Embedding":[],"Model":[],"Cutoff":[]}

for metric in METRICS:
    DF[metric]=[]

TH=0.5

# MODEL_CLASS="gat"
# LIGAND=args.ligand
print(f"Ligand type : {LIGANDS}")
all_preds=[]
# for CUTOFF in CUTOFFS[-1]:
CUTOFF=8

print("Cutoff : ",CUTOFF)

# baseline_model = Baseline().cuda()
# baseline_model.load_state_dict(torch.load(f"/home/files/{EMB_NAME}_{LIGAND}_th_4_no_graph_deep_fold0_model.pt"))
# baseline_model.eval()


def process_protein(LIGAND,model,i):
    file,g_test,features,labels=get_protein(LIGAND,CUTOFF,i)
    # Forward propagation
    with torch.no_grad():
        logits,A = model(g_test, features)
        # baseline_logits=baseline_model(g_test,features)

    if model.model_class=="gat":
    # Get the attention weights     
        A = preprocess_attention(A, g_test).todense()
    else:
        A=0
    # Get predictions and labels for evaluation
    pred = logits[:,1] 
    # baseline_pred=baseline_logits[:,1] 

    pred=to_numpy_arr(pred>TH)
    # baseline_pred=to_numpy_arr(baseline_pred>TH)
    labels=to_numpy_arr(labels)


    return {"protein_name":file.replace("all_embs_","")[:6],
            # "protein_graph":g_test,
            "attention_weights":A,
            "predicted BS":pred,
            # "baseline BS":baseline_pred,
            "true BS":labels}



import random
from tqdm import tqdm

results={}
LIGAND="GTP"
# for LIGAND in tqdm(LIGANDS):
print(LIGAND)
MODEL_CLASS="gcn"
gcn_model = GNN(MODEL_CLASS,LAYERS[MODEL_CLASS]).cuda()
gcn_model.load_state_dict(torch.load(f"/home/files/{EMB_NAME}_{LIGAND}_th_{CUTOFF}_{MODEL_CLASS}_shallow_fold_model.pt"))
gcn_model.eval()

MODEL_CLASS="gat"
gat_model = GNN(MODEL_CLASS,LAYERS[MODEL_CLASS]).cuda()
gat_model.load_state_dict(torch.load(f"/home/files/{EMB_NAME}_{LIGAND}_th_{CUTOFF}_{MODEL_CLASS}_shallow_fold_model.pt"))
gat_model.eval()

results[LIGAND]=[]
if LIGAND=="GTP":
    indices= random.sample(range(7),3)
else:
    indices= random.sample(range(20),3)

from tqdm import tqdm
results[LIGAND+"_gcn"]=[]
results[LIGAND+"_gat"]=[]
for i in tqdm(range(7)):
    results[LIGAND+"_gcn"]+=[process_protein(LIGAND,gcn_model,i)]
    results[LIGAND+"_gat"]+=[process_protein(LIGAND,gat_model,i)]

pickle.dump(results,open(f"{data_folder}_results/{EMB_NAME}_{LIGAND}_attention_results.p","wb"))

        



