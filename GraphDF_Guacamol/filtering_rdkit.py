from tqdm import tqdm
from rdkit import Chem

def has_large_ring(mol, max_size=6):
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) > max_size:
            return True
    return False

def is_atom_in_multiple_rings(mol):
    ring_info = mol.GetRingInfo()
    atom_ring_count = [0] * mol.GetNumAtoms()
    
    atom_rings = ring_info.AtomRings()
    
    for ring in atom_rings:
        for atom_idx in ring:
            atom_ring_count[atom_idx] += 1
    
    for idx, count in enumerate(atom_ring_count):
        if count >= 3:
            return True
    
    return False

def has_sulfur_triple_bond(mol):
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        
        if (atom1.GetSymbol() == 'S' or atom2.GetSymbol() == 'S') and bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
            return True
    
    return False

def has_irregular_bond(mol):
    """
    1. Remove if S and P are bonded.
    2. Remove if S atoms are bonded to each other.
    3. Remove if P atoms are bonded to each other.
    4. Remove if S is bonded to 4 or more atoms.
    5. Remove if P is bonded to 3 or more atoms.
    6. Remove if the total number of S and P atoms is 3 or more (e.g., S, S, P).
    """
    sulfur_count = 0
    phosphorus_count = 0
    
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        
        if symbol == 'S':
            sulfur_count += 1
        elif symbol == 'P':
            phosphorus_count += 1
        
        for neighbor in atom.GetNeighbors():
            neighbor_symbol = neighbor.GetSymbol()
            bond_type = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType()
            
            if (symbol == 'S' and neighbor_symbol == 'P') or (symbol == 'P' and neighbor_symbol == 'S'):
                return True
            
            if symbol == 'S' and neighbor_symbol == 'S':
                return True
            
            if symbol == 'P' and neighbor_symbol == 'P':
                return True
        
        if symbol == 'S' and len(atom.GetNeighbors()) >= 4:
            return True
        
        if symbol == 'P' and len(atom.GetNeighbors()) >= 3:
            return True
    
    if sulfur_count + phosphorus_count >= 3:
        return True
    
    return False

def has_sharing_three_atoms_in_ring(mol):
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    num_rings = len(atom_rings)

    for i in range(num_rings):
        for j in range(i+1,num_rings):
            shared_atom = set(atom_rings[i]).intersection(set(atom_rings[j]))
            if len(shared_atom) >= 3:
                return True
    return False

def has_multiple_ring(mol, max_ring=3):
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    num_rings = len(atom_rings)

    if num_rings > max_ring:
        return True
    else:
        return False

def has_irregular_structure(smiles):
    mol = Chem.MolFromSmiles(smiles)

    irregular = has_large_ring(mol, max_size=6)
    irregular = irregular or has_sulfur_triple_bond(mol)
    irregular = irregular or has_sharing_three_atoms_in_ring(mol)
    irregular = irregular or has_irregular_bond(mol)
    irregular = irregular or has_multiple_ring(mol)

    return irregular

if __name__ == '__main__':
    with open('results/new_mols.txt', 'r') as f:
        lines = f.readlines()
    mols = lines[2:]

    """with open('results/HIT274.txt', 'r') as f:
        lines = f.readlines()
    mols = lines[1:]"""

    filtered_mol = []
    filtered_cnt = 0
    for mol in tqdm(mols):
        smiles, _, _, _ = mol.split(' ')
        #smiles = mol[:-1]
        if has_irregular_structure(smiles):
            filtered_cnt += 1
        else:
            filtered_mol.append(mol)

    with open('results/rdkit_filtered_mols.txt', 'w') as f:
        f.write(f'Unique mols: {len(filtered_mol)}\n')
        f.write('SMILES QED Validity Uniqueness\n')
        for line in filtered_mol:
            f.write(line)

