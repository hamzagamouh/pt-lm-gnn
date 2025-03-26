This folder contains scripts for generating attention weights from pretrained models and visualizing them in PyMol.

1. Preprocess your PDB files of your input proteins. Please refer to `scripts/preprocessing_datasets`

2. Load one of our pretrained models 

3. Run `attention_visualization.py` script on your processed dataset

4. Use the `attention_viz.ipynb` Jupyter notebook to visualize the attention weights interactively. 

Please note that the attention visualization scripts expect a labeled dataset (for convenience). But you can easily ignore it by providing dummy labels (all zeros for example) 

The examples used in section 3.4 are provided as PyMol session files (`.pse`) that you can display in the PyMol interface.