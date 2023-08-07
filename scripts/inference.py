import subprocess


import numpy as np
import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser,MMCIFParser
import pickle
import torch
import scipy
import zipfile
import dgl
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pdb_file", default="", type=str, help="PDB file")
parser.add_argument("--ligand", default="ADP", type=str, help="PDB file")


args = parser.parse_args()

from preprocessing_datasets import get_sequence,create_dgl_graph,create_dgl_data
from create_result_tables import GNN,majority_vote,to_numpy

LIGAND=args.ligand

print(f"Importing GAT models with T5 embeddings for {LIGAND} ligand...")

MODELS={}

for CUTOFF in [4,6,8,10]:
    FOLD=0
    if CUTOFF==8:
        FOLD=""

    model= GNN(model_class="gat",layers=[512])
    model.load_state_dict(torch.load(f"../models/t5_{LIGAND}_GAT/t5_{LIGAND}_th_{CUTOFF}_gat_shallow_fold{FOLD}_model.pt",map_location="cpu"))
    model.eval()
    MODELS[CUTOFF]=model


print("Processing PDB file...")

print("Computing embeddings...")

subprocess.run(f"""source ~/anaconda3/etc/profile.d/conda.sh
    conda activate protein_embs
    python compute_embeddings.py --pdb_file {args.pdb_file}
    conda activate plm-gnn""",
    shell=True, executable='/bin/bash', check=True)

try:
    all_feats=pickle.load(open(f"{args.pdb_file[:4]}_t5_embeddings.p","rb"))
except:
    raise FileNotFoundError("Please compute embeddings first !")
# PDB parser
pdb_parser = PDBParser()
structure = pdb_parser.get_structure("X",args.pdb_file)

results={}
for chain in structure.get_chains():
    chain_id=str(chain.id)
    seq=get_sequence(chain,"","")
    all_preds=[]
    for cutoff in [4,6,8,10]:
        g=create_dgl_graph(chain,binding_residues=None,pdb_id="",chain_id="",cutoff=cutoff)
        g=create_dgl_data(g,all_feats[chain_id])
        assert g.ndata["label"].shape[0]==g.ndata["t5"].shape[0]
        with torch.no_grad():
            logits = model(g, g.ndata["t5"])
            test_preds=to_numpy(logits>0.5)

        all_preds+=[test_preds]

    results[chain_id]=majority_vote(all_preds)


print(results)


    


