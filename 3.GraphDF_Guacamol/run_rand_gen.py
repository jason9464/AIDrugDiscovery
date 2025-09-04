import json
import argparse
import wandb
from model import GraphDFForGuacamol
import utils
from chembl import ChEMBL
from rdkit import RDLogger
from torch_geometric.data import DenseDataLoader
from dig.ggraph.dataset import QM9, ZINC250k, MOSES
from dig.ggraph.method import GraphDF
from dig.ggraph.evaluation import RandGenEvaluator

RDLogger.DisableLog('rdApp.*')

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='zinc250k', choices=['qm9', 'zinc250k', 'moses', 'chembl'], help='dataset name')
parser.add_argument('--model_path', type=str, default='./rand_gen_zinc250k/rand_gen_ckpt_1.pth', help='The path to the saved model file')
parser.add_argument('--num_mols', type=int, default=10, help='The number of molecules to be generated')
parser.add_argument('--train', action='store_true', default=False, help='specify it to be true if you are running training')

args = parser.parse_args()

wandb.init(project="GraphDF")
wandb.run.name = "rand_gen_{}".format(args.data)

if args.data == 'zinc250k':
    with open('config/rand_gen_zinc250k_config_dict.json') as f:
        conf = json.load(f)
    dataset = ZINC250k(conf_dict=conf['data'], one_shot=False, use_aug=True) 
elif args.data == 'chembl':
    with open('config/rand_gen_chembl_config_dict.json') as f:
        conf = json.load(f)
    dataset = ChEMBL(conf_dict=conf['data'], one_shot=False, use_aug=True)

runner = GraphDFForGuacamol(conf, 'rand_gen')

if args.train:
    loader = DenseDataLoader(dataset, batch_size=conf['batch_size'], shuffle=True)
    runner.train_rand_gen(loader, conf['lr'], conf['weight_decay'], conf['max_epochs'], conf['model'], conf['save_interval'], conf['save_dir'])
else:
    mols, pure_valids = runner.run_rand_gen(conf['model'], args.model_path, args.num_mols, conf['num_min_node'], conf['num_max_node'], conf['temperature'], conf['atom_list'])
    smiles = [data.smile for data in dataset]
    evaluator = RandGenEvaluator()
    input_dict = {'mols': mols, 'train_smiles': smiles}

    print('Evaluating...')
    results = evaluator.eval(input_dict)
    
    utils.get_mol_images(mols[:10], "imgs/rand_gen")
    print("Valid Ratio without valency check: {:.2f}%".format(sum(pure_valids) / args.num_mols * 100))