#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate protein_embs
python compute_embeddings.py --pdb_file "1a2b.pdb"