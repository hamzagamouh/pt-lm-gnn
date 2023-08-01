# Utilizing Protein Language Models combined with Graph Neural Networks for Protein-Ligand Binding Sites detection

This is the code repository for our paper "Utilizing Protein Language Models combined with Graph Neural Networks for Protein-Ligand Binding Sites detection".


## Environment setup
To run the scripts, please install the Conda environement by following these steps :
1. Create a Conda environment --> `conda create --name plm-gnn python=3.10`
2. Activate the environment --> `conda activate plm-gnn`
3. If you have GPUs install pytorch for CUDA --> `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 cuda -c pytorch -c "nvidia/label/cuda-11.7.1"`
4. Install other pip requirements --> `pip install requirements.txt`
   
## Datasets
The original sequence datasets designed by Yu et al. are in `datasets/yu_merged`. The datasets contain protein sequences organized by their protein PDB ID and chain ID, as well as their true binding residues. Originally, there were multiple entries of binding residues due to the existence of multiple instances of a ligand, we merged the binding residues for each unique protein sequence. We also extracted the corresponding structures from BioLip, you can download them [here](https://cunicz-my.sharepoint.com/:f:/g/personal/88889462_cuni_cz/Epl85n_aRMVGuwsELWLPvWsBSKG__2_e3x6F1Rv4itPoGg?e=9IVYdm).

## Embeddings computation
For embeddings computation, please note that the `bio-embeddings` package works best in Python 3.7. Therefore, please follow these steps :
1. Create another Conda environment --> `conda create --name protein_embs python=3.7`
2. Activate the environment --> `conda activate protein_embs`
3. Install pip requirements --> `pip install embs_requirements.txt`
4. To compute the embeddings for the whole dataset please run `scripts/create_embeddings.py`
5. Switch to the main environment by running `conda deactivate` and then `conda activate plm-gnn`

## Protein graph construction
To construct the input data to the models, please run `scripts/preprocessing_datasets.py`. This script :
1. Starts with a sequence from the Yu dataset
2. Parses the corresponding PDB files to extract the sequences and atom coordinates
3. Matches the parsed sequence with the original sequence by applying manual modifications defined in `scripts/manual_modifications.py`
4. Loads pre-computed embeddings
5. Constructs the residue contact map (using the CUTOFF distance parameter) and converts it to a DGL graph with embedding features and binary labels for residues.

## Training models
To train and test the models please run `scripts/train_model.py`. 

## Results
To generate our result tables, please run  `scripts/create_result_tables.py`

## Models
We trained two major architectures : Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) for all cutoff  distances 4, 6, 8 and 10 Angstroms. You can download our trained models [here](https://cunicz-my.sharepoint.com/:f:/g/personal/88889462_cuni_cz/EqFARaVLNctBn8kupuW26qkBgUew3qjhCo4HdDRXgvKyGQ?e=5v5lEn).

## Try a prediction by our GAT ensemble model
TBA
