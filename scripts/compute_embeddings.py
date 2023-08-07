
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pdb_file", default="", type=str, help="PDB file")

args = parser.parse_args()

print("Importing T5 embedder...")
from bio_embeddings.embed.prottrans_t5_embedder import ProtTransT5XLU50Embedder
T5_EMBEDDER=ProtTransT5XLU50Embedder(half_model=True)

from Bio.PDB import PDBParser
import pickle
from preprocessing_datasets import get_sequence


pdb_parser = PDBParser()
structure = pdb_parser.get_structure("X",args.pdb_file)

all_feats={}

for chain in structure.get_chains():
    chain_id=str(chain.id)
    seq=get_sequence(chain,"","")
    print(seq)
    all_feats[chain_id]={"t5":T5_EMBEDDER.embed(seq)}


pickle.dump(open(f"{args.pdb_file[:4]}_t5_embeddings.p","wb"))