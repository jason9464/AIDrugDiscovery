import json
import argparse
import utils
import os
import numpy as np
import time
from rdkit import RDLogger, Chem
from dig.ggraph.method import GraphDF
from dig.ggraph.evaluation import PropOptEvaluator
from dig.ggraph.utils import check_chemical_validity, qed, calculate_min_plogp
from filtering_rdkit import has_irregular_structure

class PropOptEvaluator:
    r"""
    Evaluator for property optimization task. Metric is top-3 property scores among generated molecules.

    Args:
        prop_name (str): A string indicating the name of the molecular property, use 'plogp' for penalized logP or 'qed' for 
            Quantitative Estimate of Druglikeness (QED). (default: :obj:`plogp`)
    """

    def __init__(self, prop_name='plogp'):
        assert prop_name in ['plogp', 'qed']
        self.prop_name = prop_name
    
    def eval(self, input_dict):
        r""" Run evaluation in property optimization task. Find top-3 molucules which have highest property scores.
        
        Args:
            input_dict (dict): A python dict with the following items:
                "mols" --- a list of generated molecules reprsented by rdkit Chem.Mol or Chem.RWMol objects.
            
        :rtype: :class:`dict` a python dict with the following items:
                    1 --- information of molecule with the highest property score;
                    2 --- information of molecule with the second highest property score;
                    3 --- information of molecule with the third highest property score.
                    The molecule information is given in the form of a tuple (SMILES string, property score).
        """

        mols = input_dict['mols']
        prop_fn = qed if self.prop_name == 'qed' else calculate_min_plogp

        results = {}
        valid_mols = [mol for mol in mols if check_chemical_validity(mol)]
        valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
        props = [prop_fn(mol) for mol in valid_mols]
        sorted_index = np.argsort(props)[::-1]
        
        for i in range(len(valid_mols)):
            print("Top {} property score: {}".format(i+1, props[sorted_index[i]]))
            results[i+1] = (valid_smiles[sorted_index[i]], props[sorted_index[i]])
        
        return results
    

if __name__ == '__main__':
    RDLogger.DisableLog('rdApp.*')

    parser = argparse.ArgumentParser()
    parser.add_argument('--prop', type=str, default='qed', choices=['plogp', 'qed'], help='property name')
    parser.add_argument('--model_path', type=str, default='./saved_ckpts/prop_opt/prop_opt_qed.pth', help='The path to the saved model file')
    parser.add_argument('--num_mols', type=int, default=10, help='The number of molecules to be generated')
    parser.add_argument('--train', action='store_true', default=False, help='specify it to be true if you are running training')
    parser.add_argument('--result_path', type=str, default='generated_molecules')

    args = parser.parse_args()

    if args.prop == 'plogp':
        with open('config/prop_opt_plogp_config_dict.json') as f:
            conf = json.load(f)
    elif args.prop == 'qed':
        with open('config/prop_opt_qed_config_dict.json') as f:
            conf = json.load(f)
    else:
        print('Only plogp and qed properties are supported!')
        exit()

    runner = GraphDF()

    if args.train:
        runner.train_prop_opt(conf['lr'], conf['weight_decay'], conf['max_iters'], conf['warm_up'], conf['model'], conf['pretrain_model'], conf['save_interval'], conf['save_dir'])
    else:
        start_time = time.time()
        mols = runner.run_prop_opt(conf['model'], args.model_path, args.num_mols, conf['num_min_node'], conf['num_max_node'], conf['temperature'], conf['atom_list'])
        evaluator = PropOptEvaluator(prop_name=args.prop)
        input_dict = {'mols': mols}

        os.makedirs("imgs/prop_opt", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        utils.get_mol_images(mols[:10], "imgs/prop_opt")
        print('Evaluating...')
        results = evaluator.eval(input_dict)
        print(results)

        qed_list = []
        with open(f"results/{args.result_path}.txt", "w") as f:
            f.write(f'{args.num_mols} mols generated. Elapsed time: {time.time()-start_time:.2f}s\n')
            for smiles, prop_score in results.values():
                f.write(f'{smiles} {prop_score:.3f}\n')
                qed_list.append(prop_score)

        regular_cnt = 0
        with open(f'results/{args.result_path}_filtered.txt', "w") as f:
            for smiles, prop_score in results.values():
                if not has_irregular_structure(smiles):
                    f.write(f'{smiles} {prop_score:.3f}\n')
                    qed_list.append(prop_score)
                    regular_cnt += 1
        print(f'{regular_cnt} out of {args.num_mols} has regular structure.')

        print(f'Elapsed time: {time.time()-start_time}')
        print(f'mean_qed: {sum(qed_list)/len(qed_list)}')