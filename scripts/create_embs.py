from aaindex import aaindex1
import numpy as np
from tqdm import tqdm
import numpy as np
import os,zipfile,pickle
import pandas as pd
import dgl
import torch

from bio_embeddings.embed.prottrans_bert_bfd_embedder import ProtTransBertBFDEmbedder
BERT_EMBEDDER=ProtTransBertBFDEmbedder()

CODES=aaindex1.record_codes()

AMINO_ACIDS=aaindex1.amino_acids()

ALL_VALUES={aa:[] for aa in AMINO_ACIDS}

for code in tqdm(CODES):
    values=aaindex1[code].values
    for aa in AMINO_ACIDS:
        ALL_VALUES[aa]+=[values[aa]]


for k,aa in enumerate(ALL_VALUES.keys()):
    vals=np.array(ALL_VALUES[aa]).reshape(1,-1)
    if k==0:
        pc_embs=vals
    else:
        pc_embs=np.concatenate([pc_embs,vals],axis=0)

PC_EMBS=(pc_embs-np.mean(pc_embs,axis=0).reshape(1,-1))/np.std(pc_embs,axis=0).reshape(1,-1)


def create_aaindex_emb(seq):
    for k,x in enumerate(seq):
        idx=AMINO_ACIDS.index(x)
        aa_emb=PC_EMBS[idx].reshape(1,-1)
        if k==0:
            emb=aa_emb
        else:
            emb=np.concatenate([emb,aa_emb],axis=0)
    return emb

def create_bert_emb(seq):
    return BERT_EMBEDDER.embed(seq)
        



data_folder=f"/home/gamouhh/files/yu_gnn"
yu_data_folder="/home/gamouhh/files/datasets/yu_merged/"




# LIGAND="AMP"
EMB_NAME="t5"
CUTOFFS=[6]#[4,6,10]
LIGANDS=["ADP"]#["ADP", "AMP", "ATP", "CA", "FE", "GDP", "GTP", "HEME", "MG", "MN", "ZN"]

for CUTOFF in CUTOFFS:
    for LIGAND in LIGANDS:
        print(LIGAND)
        yu_path=os.path.join(yu_data_folder,"Training_sets",f"{LIGAND}_Training.txt")
        train_df=pd.read_csv(yu_path,sep=";")
        yu_path=os.path.join(yu_data_folder,"Testing_sets",f"{LIGAND}_Validation.txt")
        test_df=pd.read_csv(yu_path,sep=";")

        train_seqs={}
        test_seqs={}

        for k in range(train_df.shape[0]):
            chain=train_df["pdb_id"][k]+"_"+train_df["chain_id"][k]
            train_seqs[chain]=train_df["sequence"][k]
        

        for k in range(test_df.shape[0]):
            chain=test_df["pdb_id"][k]+"_"+test_df["chain_id"][k]
            test_seqs[chain]=test_df["sequence"][k]


        train_zip=zipfile.ZipFile(f'{data_folder}/Training_sets/{EMB_NAME}_{LIGAND}_Training_th_{CUTOFF}.zip','r')
        test_zip=zipfile.ZipFile(f'{data_folder}/Testing_sets/{EMB_NAME}_{LIGAND}_Validation_th_{CUTOFF}.zip','r')
        train_graphs=train_zip.namelist()
        test_graphs=test_zip.namelist()


        with zipfile.ZipFile(f'{data_folder}/Training_sets/all_embs_{LIGAND}_Training_th_{CUTOFF}.zip','w') as new_train_zip:
            for file in tqdm(train_graphs):
                chain=file.replace('t5_','').replace(f'_th_{CUTOFF}.p','')
                seq=train_seqs[chain]
                dest=f"{data_folder}/{file}"
                train_zip.extract(file,data_folder)
                graph=pickle.load(open(dest,'rb'))

                graph.ndata["pc_feat"]=torch.tensor(create_aaindex_emb(seq)).float()
                graph.ndata["bert_feat"]=torch.tensor(create_bert_emb(seq)).float()
                assert graph.ndata["feat"].shape[0]==graph.ndata["pc_feat"].shape[0]
                assert graph.ndata["feat"].shape[0]==graph.ndata["bert_feat"].shape[0]
                
                file_name=file.replace("t5","all_embs")
                pickle.dump(graph,open(dest,"wb"))
                new_train_zip.write(dest,file_name,compress_type=zipfile.ZIP_BZIP2)
                os.remove(dest)



        with zipfile.ZipFile(f'{data_folder}/Testing_sets/all_embs_{LIGAND}_Validation_th_{CUTOFF}.zip','w') as new_test_zip:
            for file in tqdm(test_graphs):
                chain=file.replace('t5_','').replace(f'_th_{CUTOFF}.p','')
                seq=test_seqs[chain]
                dest=f"{data_folder}/{file}"
                test_zip.extract(file,data_folder)
                graph=pickle.load(open(dest,'rb'))

                graph.ndata["pc_feat"]=torch.tensor(create_aaindex_emb(seq)).float()
                graph.ndata["bert_feat"]=torch.tensor(create_bert_emb(seq)).float()
                assert graph.ndata["feat"].shape[0]==graph.ndata["pc_feat"].shape[0]
                assert graph.ndata["feat"].shape[0]==graph.ndata["bert_feat"].shape[0]
                
                file_name=file.replace("t5","all_embs")
                pickle.dump(graph,open(dest,"wb"))
                new_test_zip.write(dest,file_name,compress_type=zipfile.ZIP_BZIP2)
                os.remove(dest)
