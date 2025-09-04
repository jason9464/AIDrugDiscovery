from dig.ggraph.dataset import PygDataset
from rdkit import Chem
from tqdm import tqdm
import pandas as pd
import os

class ChEMBL(PygDataset):
    def __init__(self,
                root='./datasets',
                prop_name=None,
                conf_dict=None,
                transform=None,
                pre_transform=None,
                pre_filter=None,
                processed_filename='data.pt',
                use_aug=False,
                one_shot=False):
        self.name='chembl24_train'
        self.root = root
        self.data_file_name = 'chembl24_canon_train.smiles'

        # 이미 있으면 안만들게 수정
        self.make_data_csv()
        num_max_node = 40
        #num_max_node = self.get_max_atoms()
        conf_dict['num_max_node'] = num_max_node
        super(ChEMBL, self).__init__(root, self.name, prop_name, conf_dict, transform, pre_transform, pre_filter, processed_filename, use_aug, one_shot)

    def make_data_csv(self):
        if not os.path.isfile(self.raw_dir+'/'+self.raw_file_names):
            data_path = self.raw_dir + '/' + self.data_file_name

            with open(data_path, 'r') as f:
                smiles_list = f.readlines()
            smiles_list = list(map(lambda s: s.strip(), smiles_list))

            data_df = pd.DataFrame()
            data_df['SMILES'] = smiles_list
            print(data_df)

            data_df.to_csv(self.raw_dir+'/'+self.raw_file_names)

    def get_max_atoms(self):
        data_df = pd.read_csv(self.raw_dir+'/'+self.raw_file_names)
        
        max_atoms = 0
        for smiles in tqdm(data_df['SMILES']):
            mol = Chem.MolFromSmiles(smiles)
            num_atoms = mol.GetNumAtoms()

            assert mol is not None
            assert num_atoms != 0

            if max_atoms < num_atoms:
                max_atoms = num_atoms

        return max_atoms

if __name__ == '__main__':
    conf_dict = dict()
    conf_dict['url'] = None
    conf_dict['prop_list'] = "[]"
    conf_dict['smile'] = 'SMILES'
    conf_dict['num_max_node'] = 40
    conf_dict['atom_list'] = "[1,5,6,7,8,9,14,15,16,17,34,35,53]"

    dataset = ChEMBL(conf_dict=conf_dict)
    print(dataset.get_max_atoms())
    print(dataset[0])
    print(dataset[0].x)
    print(dataset)