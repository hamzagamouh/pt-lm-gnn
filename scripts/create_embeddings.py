from aaindex import aaindex1
import numpy as np
from tqdm import tqdm
import numpy as np
import os,zipfile,pickle
import pandas as pd

from bio_embeddings.embed.prottrans_bert_bfd_embedder import ProtTransBertBFDEmbedder
BERT_EMBEDDER=ProtTransBertBFDEmbedder()

from bio_embeddings.embed.prottrans_t5_embedder import ProtTransT5XLU50Embedder
T5_EMBEDDER=ProtTransT5XLU50Embedder(half_model=True)

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

def create_t5_emb(seq):
    return T5_EMBEDDER.embed(seq)
        



data_folder=f"embeddings"
yu_data_folder="../datasets/yu_merged/"





LIGANDS=["ADP", "AMP", "ATP", "CA", "FE", "GDP", "GTP", "HEME", "MG", "MN", "ZN"]

for LIGAND in LIGANDS:
    print(f"Computing embeddings for {LIGAND}")
    yu_path=os.path.join(yu_data_folder,"Training_sets",f"{LIGAND}_Training.txt")
    train_df=pd.read_csv(yu_path,sep=";")
    yu_path=os.path.join(yu_data_folder,"Testing_sets",f"{LIGAND}_Validation.txt")
    test_df=pd.read_csv(yu_path,sep=";")


    with zipfile.ZipFile(f'{data_folder}/Training_sets/all_embs_{LIGAND}_Training.zip','w') as new_zip:
        for k in range(train_df.shape[0]):  
            all_embs={}
            chain=train_df["pdb_id"][k]+"_"+train_df["chain_id"][k]
            seq=train_df["sequence"][k]
            all_embs["pc_feat"]=create_aaindex_emb(seq)
            all_embs["bert"]=create_bert_emb(seq)
            all_embs["t5"]=create_t5_emb(seq)
            file=f"{chain}.p"
            dest=f"{data_folder}/{file}"
            pickle.dump(all_embs,open(dest,"wb"))
            new_zip.write(dest,file,compress_type=zipfile.ZIP_BZIP2)
            os.remove(dest)
    
    with zipfile.ZipFile(f'{data_folder}/Testing_sets/all_embs_{LIGAND}_Validation.zip','w') as new_zip:
        for k in range(test_df.shape[0]):  
            all_embs={}
            chain=test_df["pdb_id"][k]+"_"+test_df["chain_id"][k]
            seq=test_df["sequence"][k]
            all_embs["pc_feat"]=create_aaindex_emb(seq)
            all_embs["bert"]=create_bert_emb(seq)
            all_embs["t5"]=create_t5_emb(seq)
            file=f"{chain}.p"
            dest=f"{data_folder}/{file}"
            pickle.dump(all_embs,open(dest,"wb"))
            new_zip.write(dest,file,compress_type=zipfile.ZIP_BZIP2)
            os.remove(dest)
