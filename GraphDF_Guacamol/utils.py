import rdkit
from rdkit.Chem import Draw

def get_mol_images(mols, save_dir):
    for i, mol in enumerate(mols):
        img = Draw.MolToImage(mol)
        img.save("{}/mol_{}.png".format(save_dir, i))