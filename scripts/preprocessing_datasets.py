from fileinput import filename
import os
import numpy as np
import pandas as pd
from Bio.PDB import *
import pickle
# import dgl
import torch
import scipy
import zipfile
import dgl
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--emb_name", default="all_embs", type=str, help="Embedding name : onehot, bert, xlnet, t5")
parser.add_argument("--cutoff", default=8, type=int, help="Define the graph threshold in Angstroms : 4,6,8,10,12")

args = parser.parse_args()

# Amino acid codons
AMINO_ACIDS={'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E',
             'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L',
                 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P', 'SER':'S'
             , 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V',
             'PYL':'X','SEC':'X','UNK':'X', # Unknown AAs
             'TOX':'W','MSE':'M','LLP':'K','TPO':'T','CME':'C','CSD':'C','MLY':'K', 'SEP':'S','CSO':'C'}  # Modified AA
             # MSE is SELENOMETHIONINE treated as Methylene in Yu dataset !!!


ACs=list(AMINO_ACIDS.keys())
AC_letters=list(AMINO_ACIDS.values())
MODIFIED_AA=["TOX","MSE",'LLP','TPO','CME','CSD','MLY','SEP','CSO']

# PDB parser
pdb_parser = PDBParser()
cif_parser = MMCIFParser()

# SeqIO parser (getting sequences)
from Bio import SeqIO

##### Useful functions

# Check if a residue is an amino acid
def is_AC(res):
    if res.get_resname() in MODIFIED_AA:
        return True
    return res.get_full_id()[3][0]==" "

# Check if a residue is a ligand
def is_ligand(res):
    return res.get_full_id()[3][0] not in ["W"," "]


from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.Polypeptide import PPBuilder

pp_parser=PPBuilder()


from manual_corrections import skip_letters,correct_sequence

# Get residues from a chain
def get_residues(chain,binding_residues,pdb_id,chain_id):
    residues=[]
    for k,res in enumerate(chain):
        # Manual corrections !!
        if skip_letters(k,pdb_id,chain_id):
            continue

        if is_AC(res):
            residues.append(res)

    if binding_residues is None:
        return residues
    
    else:
        chain_binding_residues_idx=[int(x[1:])-1 for x in binding_residues]
        res_with_labels=[(res,label_res(res_id,chain_binding_residues_idx)) for res_id,res in enumerate(residues)]
        return res_with_labels


# Get AA sequence
def get_sequence(chain,pdb_id,chain_id):
    s=""
    for res in get_residues(chain,None,pdb_id,chain_id):
        s+=AMINO_ACIDS[res.get_resname()]
    return s
    # return ''.join(three_to_one(r.get_resname()) for r in get_residues(chain,None))

# Label a residue based on the Yu information
def label_res(res_id,chain_binding_residues_idx):
    if res_id in chain_binding_residues_idx:
        return 1
    return 0


# Check if 2 residues are connected  (2 residues are connected if their alpha carbon atoms are close enough)
def are_connected(res1,res2,th):
    # Check all atoms in the first residue and get their coordinates
    for atom1 in res1.get_unpacked_list():
        coord1=atom1.get_coord()
        # if some atom is a central carbon atom take its coordinates
        if atom1.get_name()=="CA":
            break
            
    for atom2 in res2.get_unpacked_list():
        coord2=atom2.get_coord()
        # if some atom is a central carbon atom take its coordinates
        if atom2.get_name()=="CA":
            break
    
    # Check distance
    distance=np.linalg.norm(coord1-coord2)
    if distance<th:
        return 1
    
    return 0


# This function parses a protein and returns
# 1. an array of the protein's adjacency matrix (residue nodes)
# 2. an embedding for the sequence of residues in the protein (features)
# 3. an array of labels representing ligandability of a residue to a ligand

def get_graph(chain,binding_residues,pdb_id,chain_id,cutoff):
    residues=get_residues(chain,binding_residues,pdb_id,chain_id) 
    # Adjacency matrix at the residue level
    n_res=len(residues)
    A = np.zeros((n_res,n_res))  
    for i in range(n_res):
        for j in range(n_res):
            A[i,j]=are_connected(residues[i][0],residues[j][0],th=cutoff)   
    
    # Labels represent the binding site status of the residue
    labels = np.zeros((n_res,1))  
    for i in range(n_res):
        labels[i]=residues[i][1]  
    
    print((labels==1).sum()," binding sites of ",n_res)

    return scipy.sparse.csr_matrix(A),labels


## AAIndex embeddings
from aaindex import aaindex1
CODES=aaindex1.record_codes()

AA_INDEX_AMINO_ACIDS=aaindex1.amino_acids()

ALL_VALUES={aa:[] for aa in AA_INDEX_AMINO_ACIDS}

for code in CODES:
    values=aaindex1[code].values
    for aa in AA_INDEX_AMINO_ACIDS:
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
        idx=AA_INDEX_AMINO_ACIDS.index(x)
        aa_emb=PC_EMBS[idx].reshape(1,-1)
        if k==0:
            emb=aa_emb
        else:
            emb=np.concatenate([emb,aa_emb],axis=0)
    return emb

# ______________ Protrans language models ______________

print("Importing embedders...")
EMBEDDERS={}
# BERT 
from bio_embeddings.embed.prottrans_bert_bfd_embedder import ProtTransBertBFDEmbedder
EMBEDDERS["bert"]=ProtTransBertBFDEmbedder(device="cpu")

# T5
from bio_embeddings.embed.prottrans_t5_embedder import ProtTransT5XLU50Embedder
EMBEDDERS["t5"]=ProtTransT5XLU50Embedder(device="cpu",half_model=True)

def get_embeddings(seq,feats):
    all_embs={}
    if feats is None:
        all_embs["feat"]=EMBEDDERS["t5"].embed(seq)
    else:
        all_embs["feat"]=feats
    all_embs["pc_feat"]=create_aaindex_emb(seq)
    all_embs["bert_feat"]=EMBEDDERS["bert"].embed(seq)
    # Get the sequence embeddings
    return all_embs


def create_dgl_graph(chains,binding_residues,pdb_id,chain_id,cutoff):
    A,labels=get_graph(chains,binding_residues,pdb_id,chain_id,cutoff)
    g=dgl.from_scipy(A)
    g.ndata["label"]=torch.tensor(labels).long()
    return g


def create_dgl_data(g,all_feats):
    for feat_name,feats in all_feats.items():
        g.ndata[feat_name]=torch.tensor(np.round(feats,decimals=4)).float()
    return g


def process_file(folder,pdb_id,chain_id,dataset_seq,mode):
    # Get the PDB structure and get the chains from it
    print(f"Processing retrieving chain {chain_id} from PDB file {pdb_id}... ({mode})")
    if mode=="biolip":
        file=folder+f"receptor/{pdb_id.lower()}{chain_id.upper()}.pdb"
        structure = pdb_parser.get_structure("X",file)
    if mode=="pdb":
        file=folder+f"other_pdbs/pdb{pdb_id.lower()}.ent"
        structure = pdb_parser.get_structure("X",file)
    if mode=="cif":
        file=folder+f"obsolete/{pdb_id.lower()}.cif"
        structure = cif_parser.get_structure("X",file)
    

    for chain in structure.get_chains():
        if chain_id == str(chain.id):
            seq=get_sequence(chain,pdb_id,chain_id)


            # __________________ Manual modifications !! _____________________
            seq=correct_sequence(seq,pdb_id,chain_id)
            

            try:
                assert seq==dataset_seq

            except:
                print(len(seq),len(dataset_seq))
                print(seq)
                print(dataset_seq)
                assert seq==dataset_seq
            return chain



EMB_NAME=args.emb_name



##### _____________________PREPROCESSING FILES_______________________________ 
# Preprocess files and get graphs and chains
data_folder="/home/files/datasets/yu_merged/"
cutoff=args.cutoff
print("Cutoff",cutoff)


folder="/home/files/yu_gnn/biolip_structures/"
biolip_files=[x[:4].upper() for x in os.listdir(folder+"receptor")]
other_pdb_files=[x.replace("pdb","").replace(".ent","").upper() for x in os.listdir(folder+"other_pdbs")]
obsolete_files=[x[:4].upper() for x in os.listdir(folder+"obsolete")]

with zipfile.ZipFile(f'/home/files/yu_gnn/t5_embs/yu_embs.zip',"r") as thezip:
    for output_folder in ["Training_sets","Testing_sets"]:
        for yu_file in os.listdir(data_folder+output_folder):
            print(f"Processing {yu_file}...")
            yu_path=os.path.join(data_folder,output_folder,yu_file)
            # df=pd.read_csv(yu_path,sep=";",names=["pdb_id", "chain_id", "bs_num", "ligand", "binding_residues","sequence"])
            df=pd.read_csv(yu_path,sep=";")

            if ("DNA" not in yu_file):
                # print(thezip.namelist())
                if "HEME" in yu_file:
                    new_file=yu_file.replace("HEME","HEM")
                else:
                    new_file=yu_file
                t5_embeddings=pickle.load(thezip.open(f"yu/{new_file[:-4]}.p",'r'))


            with zipfile.ZipFile(f'/home/files/yu_gnn/{output_folder}/{EMB_NAME}_{yu_file[:-4]}_th_{cutoff}.zip',"w") as thezip:
                for k in tqdm(range(df.shape[0])):
                    pdb_id=df["pdb_id"][k]
                    chain_id=df["chain_id"][k]
                    binding_residues=df["binding_residues"][k].split(" ")
                    dataset_seq=df["sequence"][k]
                    if pdb_id in other_pdb_files:
                        mode="pdb"
                    elif pdb_id in biolip_files:
                        mode="biolip"
                    elif pdb_id in obsolete_files:
                        mode="cif"

                    # Problematic chains !!!! (We will not include those)
                    if pdb_id=="1WA5" and chain_id=='C':
                        continue

                    if pdb_id=="2FBW" and chain_id=="C":
                        continue

                    chain=process_file(folder,pdb_id,chain_id,dataset_seq,mode)

                    g=create_dgl_graph(chain,binding_residues,pdb_id,chain_id,cutoff)
                    if ("DNA" not in yu_file):
                        feats=t5_embeddings[f"{pdb_id}_{chain_id}"][0]
                    else:
                        feats=None
                    all_feats=get_embeddings(dataset_seq,feats)

                    g=create_dgl_data(g,all_feats)
                    for feat_name in all_feats.keys():
                        assert g.ndata["label"].shape[0]==g.ndata[feat_name].shape[0]
                    file_name=f"{EMB_NAME}_{pdb_id}_{chain_id}_th_{cutoff}.p"
                    filepath="/home/files/"+file_name
                    pickle.dump(g,open(filepath,"wb"))
                    thezip.write(filepath,file_name,compress_type=zipfile.ZIP_BZIP2)
                    os.remove(filepath)

