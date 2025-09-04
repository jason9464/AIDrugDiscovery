All the code is from DIG repository. We have only made minor changes to check the GuacaMol benchmark.

### Requirements
#### Install PyTorch (>=1.10.0), PyG (>=2.0.0)

We recommend installing torch, torch_geometric, torch_scatter, and torch_sparse following the official webpage to avoid conflicts with CUDA. The following code is an example of how to install packages in our environment (CUDA 11.6):
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
pip install torch_geometric==2.2.0
```

#### Install DIG
```
pip install dive-into-graphs
```
#### Install rdkit
```
conda install rdkit -c rdkit
```
#### Install FCD
```
pip install fcd
```
#### Install GuacaMol
```
pip install guacamol
```

### Generate Molecules
```
python run_prop_opt.py --num_mols={$NUM_MOLS}
```
You can access the generated molecules and their QED score at the results/generated_molecules.txt