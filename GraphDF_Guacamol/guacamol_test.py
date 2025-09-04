import argparse
import os
import json
from model import GraphDFForGuacamol
from rdkit import Chem

from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.utils.helpers import setup_default_logger

if __name__ == '__main__':
    setup_default_logger()

    parser = argparse.ArgumentParser(description='Molecule distribution learning benchmark for random smiles sampler',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dist_file', default='data/guacamol_v1_all.smiles')
    parser.add_argument('--output_dir', default='results', help='Output directory')
    parser.add_argument('--suite', default='v2')
    args = parser.parse_args()

    pretrained_path = "rand_gen_zinc_ckpt.pth"

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))

    with open(args.dist_file, 'r') as smiles_file:
        smiles_list = [line.strip() for line in smiles_file.readlines()]

    with open('config/rand_gen_zinc250k_config_dict.json') as f:
        conf = json.load(f)
        
    generator = GraphDFForGuacamol(conf, 'rand_gen')
    generator.get_model('rand_gen', conf['model'], pretrained_path)

    json_file_path = os.path.join(args.output_dir, 'distribution_learning_results.json')

    assess_distribution_learning(generator,
                                 chembl_training_file=args.dist_file,
                                 json_output_file=json_file_path,
                                 benchmark_version=args.suite)