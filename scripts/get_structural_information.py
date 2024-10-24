import os
import numpy as np
import pandas as pd
from Bio.PDB import *
import pickle
import zipfile
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


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
def get_residues(chain,pdb_id,chain_id):
    residues=[]
    for k,res in enumerate(chain):
        # Manual corrections !!
        if skip_letters(k,pdb_id,chain_id):
            continue

        if is_AC(res):
            residues.append(res)

    return residues
    


# Get AA sequence
def get_sequence(chain,pdb_id,chain_id):
    s=""
    for res in get_residues(chain,pdb_id,chain_id):
        s+=AMINO_ACIDS[res.get_resname()]
    return s

# Label a residue based on the Yu information
def label_res(res_id,chain_binding_residues_idx):
    if res_id in chain_binding_residues_idx:
        return 1
    return 0


# Check if 2 residues are connected  (2 residues are connected if their alpha carbon atoms are close enough)
def get_coord(res):
    # Check all atoms in the first residue and get their coordinates
    for atom in res.get_unpacked_list():
        coord=atom.get_coord()
        # if some atom is a central carbon atom take its coordinates
        if atom.get_name()=="CA":
            break

    return coord



def get_3d_structure(chain,pdb_id,chain_id):
    residues=get_residues(chain,pdb_id,chain_id) 
    n_res=len(residues)
    M = np.zeros((n_res,3))  
    for i in range(n_res):
        M[i]=get_coord(residues[i])
    
    return M




def process_file(pdb_folder,pdb_id,chain_id):
    # Get the PDB structure and get the chains from it
    pdb_file=os.path.join(pdb_folder,f"{pdb_id.lower()}.pdb")
    structure = pdb_parser.get_structure("X",pdb_file)
    

    for chain in structure.get_chains():
        if chain_id == str(chain.id):
            seq=get_sequence(chain,pdb_id,chain_id)
            # __________________ Manual modifications !! _____________________
            seq=correct_sequence(seq,pdb_id,chain_id)

            return chain,seq






##### _____________________PREPROCESSING FILES_______________________________ 

# Put the path to the folder containing all PDB files here
PDB_FOLDER="../datasets/all_pdbs"


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    # Preprocess files and get graphs and chains
    data_folder="../datasets/yu_merged/"

    for output_folder in ["Training_sets","Testing_sets"]:
        for yu_file in os.listdir(data_folder+output_folder):
            if "ATP" not in yu_file:
                continue
            print("Processing",yu_file)
            yu_path=os.path.join(data_folder,output_folder,yu_file)
            df=pd.read_csv(yu_path,sep=";")
            with zipfile.ZipFile(f"../datasets/structural_information/{output_folder}/{yu_file.replace('.txt','')}_structural_info.zip","w") as thezip:
                for k in tqdm(range(df.shape[0])):
                    try:
                        pdb_id=df["pdb_id"][k]
                        chain_id=df["chain_id"][k]
                        binding_residues=df["binding_residues"][k].split(" ")
                        dataset_seq=df["sequence"][k]

                        # Problematic chains !!!! (We will not include those)
                        if pdb_id=="1WA5" and chain_id=='C':
                            continue

                        if pdb_id=="2FBW" and chain_id=="C":
                            continue

                        chain,modified_seq=process_file(PDB_FOLDER,pdb_id,chain_id)
                        coords=get_3d_structure(chain,pdb_id,chain_id)
                        assert len(coords)==len(modified_seq)

                        structural_info={"pdb_id":pdb_id,
                                        "chain_id":chain_id,
                                        "modified_sequence":modified_seq,
                                        "res_coordinates":coords}
                        file_name=f"{pdb_id}_{chain_id}.p"
                        filepath="../datasets/"+file_name
                        pickle.dump(structural_info,open(filepath,"wb"))
                        thezip.write(filepath,file_name,compress_type=zipfile.ZIP_BZIP2)
                        os.remove(filepath)
                    except Exception as e:
                        print(f"Problem in PDB : {pdb_id} - Chain {chain_id}")
                        print(e)
                        continue

