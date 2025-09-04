import torch
import os
import torch.nn as nn
import numpy as np
import wandb
from dig.ggraph.method import GraphDF
from dig.ggraph.method.GraphDF.model import GraphFlowModel, GraphFlowModel_rl, GraphFlowModel_con_rl

from rdkit import Chem

from dig.ggraph.utils import check_chemical_validity, check_valency, calculate_min_plogp, qed
from dig.ggraph.utils import convert_radical_electrons_to_hydrogens, steric_strain_filter, zinc_molecule_filter
from dig.ggraph.method.GraphDF.train_utils import adjust_learning_rate, DataIterator

class GraphFlowModel_rl_ForGuacamol(GraphFlowModel_rl):
    def __init__(self, conf, model_path, type):
        super(GraphFlowModel_rl_ForGuacamol, self).__init__(conf)

    def reinforce_forward_optim_qed(self, in_baseline=None, cur_iter=None):
        return super().reinforce_forward_optim(in_baseline, cur_iter)

    def reinforce_forward_optim(self, in_baseline=None, cur_iter=None, prop_func=None):
        """
        Fintuning model using reinforce algorithm
        Args:
            existing_mol: molecule to be optimized. Practically, we provide 64 mols per call and the function may take less then 64 mols
            temperature: generation temperature
            batch_size: batch_size for collecting data
            max_size_rl: maximal num of atoms allowed for generation

        Returns:

        """
        assert cur_iter is not None
        atom_list, temperature, batch_size, max_size_rl = self.conf_rl['atom_list'], self.conf_rl['temperature'], self.conf_rl['batch_size'], self.conf_rl['max_size_rl']
        if cur_iter % self.conf_rl['update_iters'] == 0: # uodate the demenstration net every 4 iter.
            print('copying to old model at iter {}'.format(cur_iter))
            self.flow_core_old.load_state_dict(self.flow_core.state_dict())

            self.node_base_log_probs_old = nn.Parameter(self.node_base_log_probs.detach().clone(), requires_grad=False)
            self.edge_base_log_probs_old = nn.Parameter(self.edge_base_log_probs.detach().clone(), requires_grad=False)

        #assert cur_baseline is not None
        num2bond = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
        num2atom = {i:atom_list[i] for i in range(len(atom_list))}

        node_inputs = {}
        node_inputs['node_features'] = []
        node_inputs['adj_features'] = []
        node_inputs['node_features_cont'] = []
        node_inputs['rewards'] = []
        node_inputs['baseline_index'] = []

        adj_inputs = {}
        adj_inputs['node_features'] = []
        adj_inputs['adj_features'] = []
        adj_inputs['edge_features_cont'] = []
        adj_inputs['index'] = []
        adj_inputs['rewards'] = []
        adj_inputs['baseline_index'] = []
        adj_inputs['edge_cnt'] = []

        reward_baseline = torch.zeros([max_size_rl + 5, 2]).cuda()

        max_action_size = batch_size * (int(max_size_rl + (self.edge_unroll - 1) * self.edge_unroll / 2 + (max_size_rl - self.edge_unroll) * self.edge_unroll))

        batch_length = 0
        total_node_step = 0
        total_edge_step = 0

        per_mol_reward = []
        per_mol_property_score = []
        
        ### gather training data from generation
        self.eval() #! very important. Because we use batch normalization, training mode will result in unrealistic molecules
        
        with torch.no_grad():
            while total_node_step + total_edge_step < max_action_size and batch_length < batch_size:                
                traj_node_inputs = {}
                traj_node_inputs['node_features'] = []
                traj_node_inputs['adj_features'] = []
                traj_node_inputs['node_features_cont'] = []
                traj_node_inputs['rewards'] = []
                traj_node_inputs['baseline_index'] = []
                traj_adj_inputs = {}
                traj_adj_inputs['node_features'] = []
                traj_adj_inputs['adj_features'] = []
                traj_adj_inputs['edge_features_cont'] = []
                traj_adj_inputs['index'] = []
                traj_adj_inputs['rewards'] = []
                traj_adj_inputs['baseline_index'] = []
                traj_adj_inputs['edge_cnt'] = []

                step_cnt = 1.0
                min_action_node = 1
                rw_mol = Chem.RWMol()  # editable mol
                mol = None

                cur_node_features = torch.zeros([1, max_size_rl, self.node_dim])
                cur_adj_features = torch.zeros([1, self.bond_dim, max_size_rl, max_size_rl])
                if self.dp:
                    cur_node_features = cur_node_features.cuda()
                    cur_adj_features = cur_adj_features.cuda()

                is_continue = True
                edge_idx = 0

                step_num_data_edge = 0
                added_num = 0
                for i in range(max_size_rl):
                    if not is_continue:
                        break                    
                    
                    step_num_data_edge = 0 # generating new node and its edges. Not sure if this will add into the final mol.

                    if i < self.edge_unroll:
                        edge_total = i  # edge to sample for current node
                        start = 0
                    else:
                        edge_total = self.edge_unroll
                        start = i - self.edge_unroll
                    
                    # first generate node
                    ## reverse flow

                    prior_node_dist = torch.distributions.OneHotCategorical(logits=self.node_base_log_probs_old[i]*temperature[0])
                    latent_node = prior_node_dist.sample().view(1, -1)
                    
                    if self.dp:
                        latent_node = self.flow_core_old.module.reverse(cur_node_features, cur_adj_features, latent_node, mode=0).view(-1)  # (9, )
                    else:
                        latent_node = self.flow_core_old.reverse(cur_node_features, cur_adj_features, latent_node, mode=0).view(-1)  # (9, )
                    
                    feature_id = torch.argmax(latent_node).item()
                    total_node_step += 1
                    node_feature_cont = torch.zeros([1, self.node_dim]).cuda()
                    node_feature_cont[0, feature_id] = 1.0

                    # update traj inputs for node_id
                    traj_node_inputs['node_features'].append(cur_node_features.clone())  # (1, max_size_rl, self.node_dim)
                    traj_node_inputs['adj_features'].append(cur_adj_features.clone())  # (1, self.bond_dim, max_size_rl, max_size_rl)
                    traj_node_inputs['node_features_cont'].append(node_feature_cont)  # (1, self.node_dim)
                    traj_node_inputs['rewards'].append(torch.full(size=(1,1), fill_value=step_cnt).cuda())  # (1, 1)
                    traj_node_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1, 1)

                    cur_node_features[0, i, feature_id] = 1.0
                    cur_adj_features[0, :, i, i] = 1.0
                    rw_mol.AddAtom(Chem.Atom(num2atom[feature_id]))

                    # then generate edges
                    if i == 0:
                        is_connect = True
                    else:
                        is_connect = False
                    # cur_mol_size = mol.GetNumAtoms
                    for j in range(edge_total):
                        valid = False
                        resample_edge = 0
                        edge_dis = self.edge_base_log_probs_old[edge_idx].clone()
                        invalid_bond_type_set = set()
                        while not valid:
                            if len(invalid_bond_type_set) < 3 and resample_edge <= 50:  # haven't sampled all possible bond type or is not stuck in the loop
                                prior_edge_dist = torch.distributions.OneHotCategorical(logits=edge_dis/temperature[1])
                                latent_edge = prior_edge_dist.sample().view(1, -1)
                                latent_id = torch.argmax(latent_edge, dim=1)
                                
                                if self.dp:
                                    latent_edge = self.flow_core_old.module.reverse(cur_node_features, cur_adj_features, latent_edge,
                                                                                mode=1, edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1)  # (4, )
                                else:
                                    latent_edge = self.flow_core_old.reverse(cur_node_features, cur_adj_features, latent_edge, mode=1,
                                                                edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1)  # (4, )
                                edge_discrete_id = torch.argmax(latent_edge).item()
                            else:
                                assert resample_edge > 50 or len(invalid_bond_type_set) == 3
                                edge_discrete_id = 3  # we have no choice but to choose not to add edge between (i, j+start)
                            
                            total_edge_step += 1
                            edge_feature_cont = torch.zeros([1, self.bond_dim]).cuda()
                            edge_feature_cont[0, edge_discrete_id] = 1.0

                            # update traj inputs for edge_id
                            traj_adj_inputs['node_features'].append(cur_node_features.clone())  # 1, max_size_rl, self.node_dim
                            traj_adj_inputs['adj_features'].append(cur_adj_features.clone())  # 1, self.bond_dim, max_size_rl, max_size_rl
                            traj_adj_inputs['edge_features_cont'].append(edge_feature_cont)  # 1, self.bond_dim
                            traj_adj_inputs['index'].append(torch.Tensor([[j + start, i]]).long().cuda().view(1,-1)) # (1, 2)
                            traj_adj_inputs['edge_cnt'].append(torch.full(size=(1,), fill_value=float(edge_idx)).long().cuda())
                            step_num_data_edge += 1 # add one edge data, not sure if this should be added to the final train data

                            cur_adj_features[0, edge_discrete_id, i, j + start] = 1.0
                            cur_adj_features[0, edge_discrete_id, j + start, i] = 1.0
                            if edge_discrete_id == 3:  # virtual edge
                                valid = True # virtual edge is alway valid
                            else:  # single/double/triple bond
                                rw_mol.AddBond(i, j + start, num2bond[edge_discrete_id])
                                valid = check_valency(rw_mol)
                                if valid:
                                    is_connect = True
                                else:  # backtrack
                                    edge_dis[latent_id] = float('-inf')
                                    rw_mol.RemoveBond(i, j + start)
                                    cur_adj_features[0, edge_discrete_id, i, j + start] = 0.0
                                    cur_adj_features[0, edge_discrete_id, j + start, i] = 0.0
                                    resample_edge += 1

                                    invalid_bond_type_set.add(edge_discrete_id)

                            if valid:
                                traj_adj_inputs['rewards'].append(torch.full(size=(1, 1), fill_value=step_cnt).cuda())  # (1, 1)
                                traj_adj_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1)
                            else:
                                if self.conf_rl['penalty']:
                                    traj_adj_inputs['rewards'].append(torch.full(size=(1, 1), fill_value=-1.).cuda())  # (1, 1) invalid edge penalty
                                    traj_adj_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1,)
                                else:
                                    traj_adj_inputs['node_features'].pop(-1)
                                    traj_adj_inputs['adj_features'].pop(-1)
                                    traj_adj_inputs['edge_features_cont'].pop(-1)
                                    traj_adj_inputs['index'].pop(-1)
                                    traj_adj_inputs['edge_cnt'].pop(-1)
                                    step_num_data_edge -= 1 # if we do not penalize invalid edge, pop train data, decrease counter by 1                              

                        edge_idx += 1        

                    if is_connect:  # new generated node has at least one bond with previous node, do not stop generation, backup mol from rw_mol to mol
                        is_continue = True
                        mol = rw_mol.GetMol()
                        added_num += 1
                        #current_i_continue = False

                    else:
                        #! we may need to satisfy min action here
                        if added_num >= min_action_node:
                            # if we have already add 'min_action_node', ignore and let continue be false.
                            # the data added in last iter will be pop afterwards
                            is_continue = False
                        else:
                            pass
                    step_cnt += 1

                batch_length += 1

                #TODO: check the last iter of generation
                #(Thinking)
                # The last node was not added. So after we generate the second to last
                # node and all its edges, the rest adjacent matrix and node features should all be zero
                # But current implementation append
                num_atoms = mol.GetNumAtoms()
                assert num_atoms <= max_size_rl

                if num_atoms < max_size_rl:   
                    #! this implementation is buggy. we only mask the last node feature cont
                    #! But we ignore the non-zero node features in generating edges
                    #! this pattern will make model not to generated any edges between
                    #! the new-generated isolated node and exsiting subgraph.
                    #! this may be the biggest bug in Reinforce algorithm!!!!!
                    #! since the final iter/(step) has largest reward....!!!!!!!
                    #! work around1: add a counter and mask out all node feautres in generating edges of last iter.
                    #! work around2: do not append any data if the isolated node is not connected to subgraph.
                    # currently use work around2

                    # pop all the reinforce train-data add by at the generating the last isolated node and its edge
                    ## pop node
                    try:
                        traj_node_inputs['node_features'].pop(-1)
                        traj_node_inputs['adj_features'].pop(-1)
                        traj_node_inputs['node_features_cont'].pop(-1)
                        traj_node_inputs['rewards'].pop(-1)
                        traj_node_inputs['baseline_index'].pop(-1)
                   
                        ## pop adj
                        for _ in range(step_num_data_edge):
                            traj_adj_inputs['node_features'].pop(-1)
                            traj_adj_inputs['adj_features'].pop(-1)
                            traj_adj_inputs['edge_features_cont'].pop(-1)
                            traj_adj_inputs['index'].pop(-1)
                            traj_adj_inputs['rewards'].pop(-1)
                            traj_adj_inputs['baseline_index'].pop(-1)
                            traj_adj_inputs['edge_cnt'].pop(-1)
                    except:
                        print('pop from empty list, take min action fail.')

                reward_valid = 2
                reward_property = 0
                reward_length = 0 

                assert mol is not None, 'mol is None...'
                final_valid = check_chemical_validity(mol)
                s_tmp = Chem.MolToSmiles(mol, isomericSmiles=True)
                assert final_valid is True, 'warning: use valency check during generation but the final molecule is invalid!!!, \
                                 cur is %s' % (s_tmp)

                if not final_valid:
                    reward_valid -= 5 
                else:
                    final_mol = convert_radical_electrons_to_hydrogens(mol)
                    s = Chem.MolToSmiles(final_mol, isomericSmiles=True)

                    final_mol = Chem.MolFromSmiles(s)
                    # mol filters with negative rewards
                    if not steric_strain_filter(final_mol):  # passes 3D conversion, no excessive strain
                        reward_valid -= 1 #TODO: check the magnitude of this reward.
                    if not zinc_molecule_filter(final_mol):  # does not contain any problematic functional groups
                        reward_valid -= 1

                    #property_type = self.conf_rl['property_type']
                    """property_type = 'qed'
                    assert property_type in ['qed', 'plogp'], 'unsupported property optimization, choices are [qed, plogp]'
                    prop_fn = qed if property_type == 'qed' else calculate_min_plogp
                    try:
                        score = prop_fn(final_mol) # value in [0, 1]
                        print(score)
                        if self.conf_rl['reward_type'] == 'exp':
                            reward_property += (np.exp(score / self.conf_rl['exp_temperature']) - self.conf_rl['exp_bias'])
                        elif self.conf_rl['reward_type'] == 'linear':
                            reward_property += (score * self.conf_rl['linear_coeff'])
                    except:
                        print('generated mol does not pass qed/plogp')"""
                    # Guacamol goal-directed property ㅅㅏㅇㅛㅇㅎㅏㄹ ㅅㅜ ㅇㅣㅆㄷㅗㄹㅗㄱ ㅅㅜㅈㅓㅇ
                    #
                    #
                    prop_fn = prop_func
                    final_smiles = Chem.MolToSmiles(final_mol)
                    print(final_smiles)
                    score = prop_fn.wrapped_objective.score(final_smiles)
                    print(score)
                    try:
                        score = prop_fn.wrapped_objective.score(final_smiles) # value in [0, 1]
                        if self.conf_rl['reward_type'] == 'exp':
                            reward_property += (np.exp(score / self.conf_rl['exp_temperature']) - self.conf_rl['exp_bias'])
                        elif self.conf_rl['reward_type'] == 'linear':
                            reward_property += (score * self.conf_rl['linear_coeff'])
                    except:
                        print('generated mol does not pass prop_fn')
                    #
                    #
                    #

                reward_final_total = reward_valid + reward_property + reward_length
                per_mol_reward.append(reward_final_total)
                per_mol_property_score.append(reward_property)
                reward_decay = self.conf_rl['reward_decay']

                node_inputs['node_features'].append(torch.cat(traj_node_inputs['node_features'], dim=0)) #append tensor of shape (max_size_rl, max_size_rl, self.node_dim)
                node_inputs['adj_features'].append(torch.cat(traj_node_inputs['adj_features'], dim=0)) # append tensor of shape (max_size_rl, bond_dim, max_size_rl, max_size_rl)
                node_inputs['node_features_cont'].append(torch.cat(traj_node_inputs['node_features_cont'], dim=0)) # append tensor of shape (max_size_rl, 9)

                traj_node_inputs_baseline_index = torch.cat(traj_node_inputs['baseline_index'], dim=0) #(max_size_rl)
                traj_node_inputs_rewards = torch.cat(traj_node_inputs['rewards'], dim=0) # tensor of shape (max_size_rl, 1)
                traj_node_inputs_rewards[traj_node_inputs_rewards > 0] = \
                    reward_final_total * torch.pow(reward_decay, step_cnt - 1. - traj_node_inputs_rewards[traj_node_inputs_rewards > 0])
                node_inputs['rewards'].append(traj_node_inputs_rewards)  # append tensor of shape (max_size_rl, 1)                
                node_inputs['baseline_index'].append(traj_node_inputs_baseline_index)

                for ss in range(traj_node_inputs_rewards.size(0)):
                    reward_baseline[traj_node_inputs_baseline_index[ss]][0] += 1.0
                    reward_baseline[traj_node_inputs_baseline_index[ss]][1] += traj_node_inputs_rewards[ss][0]                
                
                if num_atoms > 1:
                    adj_inputs['node_features'].append(torch.cat(traj_adj_inputs['node_features'], dim=0)) # (step, max_size_rl, self.node_dim)
                    adj_inputs['adj_features'].append(torch.cat(traj_adj_inputs['adj_features'], dim=0)) # (step, bond_dim, max_size_rl, max_size_rl)
                    adj_inputs['edge_features_cont'].append(torch.cat(traj_adj_inputs['edge_features_cont'], dim=0)) # (step, 4)
                    adj_inputs['index'].append(torch.cat(traj_adj_inputs['index'], dim=0)) # (step, 2)
                    adj_inputs['edge_cnt'].append(torch.cat(traj_adj_inputs['edge_cnt'], dim=0))

                    traj_adj_inputs_baseline_index = torch.cat(traj_adj_inputs['baseline_index'], dim=0) #(step)                
                    traj_adj_inputs_rewards = torch.cat(traj_adj_inputs['rewards'], dim=0)
                    traj_adj_inputs_rewards[traj_adj_inputs_rewards > 0] = \
                        reward_final_total * torch.pow(reward_decay, step_cnt - 1. - traj_adj_inputs_rewards[traj_adj_inputs_rewards > 0])
                    adj_inputs['rewards'].append(traj_adj_inputs_rewards)
                    adj_inputs['baseline_index'].append(traj_adj_inputs_baseline_index)

                    for ss in range(traj_adj_inputs_rewards.size(0)):
                        reward_baseline[traj_adj_inputs_baseline_index[ss]][0] += 1.0
                        reward_baseline[traj_adj_inputs_baseline_index[ss]][1] += traj_adj_inputs_rewards[ss][0]

        self.flow_core.train()
        for module in self.modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d):
                module.eval()

        for i in range(reward_baseline.size(0)):
            if reward_baseline[i, 0] == 0:
                reward_baseline[i, 0] += 1.

        reward_baseline_per_step = reward_baseline[:, 1] / reward_baseline[:, 0] # (max_size_rl, )

        if in_baseline is not None:
            assert in_baseline.size() == reward_baseline_per_step.size()
            reward_baseline_per_step = reward_baseline_per_step * (1. - self.conf_rl['moving_coeff']) + in_baseline * self.conf_rl['moving_coeff']

        node_inputs_node_features = torch.cat(node_inputs['node_features'], dim=0) # (total_size, max_size_rl, 9)
        node_inputs_adj_features = torch.cat(node_inputs['adj_features'], dim=0) # (total_size, 4, max_size_rl, max_size_rl)
        node_inputs_node_features_cont = torch.cat(node_inputs['node_features_cont'], dim=0) # (total_size, 9)
        node_inputs_rewards = torch.cat(node_inputs['rewards'], dim=0).view(-1) # (total_size,)
        node_inputs_baseline_index = torch.cat(node_inputs['baseline_index'], dim=0).long() # (total_size,)
        node_inputs_baseline = torch.index_select(reward_baseline_per_step, dim=0, index=node_inputs_baseline_index) #(total_size, )

        adj_inputs_node_features = torch.cat(adj_inputs['node_features'], dim=0) # (total_size, max_size_rl, 9)
        adj_inputs_adj_features = torch.cat(adj_inputs['adj_features'], dim=0) # (total_size, 4, max_size_rl, max_size_rl)
        adj_inputs_edge_features_cont = torch.cat(adj_inputs['edge_features_cont'], dim=0) # (total_size, 4)
        adj_inputs_index = torch.cat(adj_inputs['index'], dim=0) # (total_size, 2)
        adj_inputs_rewards = torch.cat(adj_inputs['rewards'], dim=0).view(-1) # (total_size,)
        adj_inputs_baseline_index = torch.cat(adj_inputs['baseline_index'], dim=0).long() #(total_size,)
        adj_inputs_baseline = torch.index_select(reward_baseline_per_step, dim=0, index=adj_inputs_baseline_index) #(total_size, )
        adj_inputs_edge_cnts = torch.cat(adj_inputs['edge_cnt'], dim=0) # (total_size, )

        if self.dp:
            node_function = self.flow_core.module.forward_rl_node
            edge_function = self.flow_core.module.forward_rl_edge

            node_function_old = self.flow_core_old.module.forward_rl_node
            edge_function_old = self.flow_core_old.module.forward_rl_edge
        else:
            node_function = self.flow_core.forward_rl_node
            edge_function = self.flow_core.forward_rl_edge
            
            node_function_old = self.flow_core_old.forward_rl_node
            edge_function_old = self.flow_core_old.forward_rl_edge

        z_node, _ = node_function(node_inputs_node_features, node_inputs_adj_features,
                                            node_inputs_node_features_cont)  # (total_step, 9), (total_step, )

        z_edge, _ = edge_function(adj_inputs_node_features, adj_inputs_adj_features,
                                            adj_inputs_edge_features_cont, adj_inputs_index) # (total_step, 4), (total_step, )


        with torch.no_grad():
            z_node_old, _ = node_function_old(node_inputs_node_features, node_inputs_adj_features,
                                                node_inputs_node_features_cont)  # (total_step, 9), (total_step, )

            z_edge_old, _ = edge_function_old(adj_inputs_node_features, adj_inputs_adj_features,
                                                adj_inputs_edge_features_cont, adj_inputs_index) # (total_step, 4), (total_step, )

        node_total_length = z_node.size(0) * float(self.node_dim)
        edge_total_length = z_edge.size(0) * float(self.bond_dim)

        node_base_log_probs_sm = torch.nn.functional.log_softmax(self.node_base_log_probs, dim=-1)
        node_base_log_probs_sm_select = torch.index_select(node_base_log_probs_sm, dim=0, index=node_inputs_baseline_index-1) #(total_size, )
        # print(z_node.shape, node_base_log_probs_sm.shape, node_base_log_probs_sm_new.shape)
        ll_node = torch.sum(z_node * node_base_log_probs_sm_select, dim=(-1,-2))
        edge_base_log_probs_sm = torch.nn.functional.log_softmax(self.edge_base_log_probs, dim=-1)
        edge_base_log_probs_sm_select = torch.index_select(edge_base_log_probs_sm, dim=0, index=adj_inputs_edge_cnts)
        # print(z_edge.shape, edge_base_log_probs_sm.shape, edge_base_log_probs_sm_select.shape)
        ll_edge = torch.sum(z_edge * edge_base_log_probs_sm_select, dim=(-1,-2))
        
        node_base_log_probs_sm_old = torch.nn.functional.log_softmax(self.node_base_log_probs_old, dim=-1)
        node_base_log_probs_sm_old_select = torch.index_select(node_base_log_probs_sm_old, dim=0, index=node_inputs_baseline_index-1) #(total_size, )
        ll_node_old = torch.sum(z_node_old * node_base_log_probs_sm_old_select, dim=(-1,-2))
        edge_base_log_probs_sm_old = torch.nn.functional.log_softmax(self.edge_base_log_probs_old, dim=-1)
        edge_base_log_probs_sm_old_select = torch.index_select(edge_base_log_probs_sm_old, dim=0, index=adj_inputs_edge_cnts)
        ll_edge_old = torch.sum(z_edge_old * edge_base_log_probs_sm_old_select, dim=(-1,-2))

        ratio_node = torch.exp((ll_node - ll_node_old.detach()).clamp(max=10., min=-10.))
        ratio_edge = torch.exp((ll_edge - ll_edge_old.detach()).clamp(max=10., min=-10.))        

        if torch.isinf(ratio_node).any():
            raise RuntimeError('ratio node has inf entries')
       
        if torch.isinf(ratio_edge).any():
            raise RuntimeError('ratio edge has inf entries')
        if self.conf_rl['no_baseline']:
            advantage_node = node_inputs_rewards
            advantage_edge = adj_inputs_rewards
        else:
            advantage_node = (node_inputs_rewards - node_inputs_baseline)
            advantage_edge = (adj_inputs_rewards - adj_inputs_baseline)

        surr1_node = ratio_node * advantage_node
        surr2_node = torch.clamp(ratio_node, 1-0.2, 1+0.2) * advantage_node

        surr1_edge = ratio_edge * advantage_edge
        surr2_edge = torch.clamp(ratio_edge, 1-0.2, 1+0.2) * advantage_edge

        if torch.isnan(surr1_node).any():
            raise RuntimeError('surr1 node has NaN entries')
        if torch.isnan(surr2_node).any():
            raise RuntimeError('surr2 node has NaN entries')
        if torch.isnan(surr1_edge).any():
            raise RuntimeError('surr1 edge has NaN entries')
        if torch.isnan(surr2_edge).any():
            raise RuntimeError('surr2 edge has NaN entries')                       

        return -((torch.min(surr1_node, surr2_node).sum() + torch.min(surr1_edge, surr2_edge).sum()) / (node_total_length + edge_total_length) - 1.0), per_mol_reward, per_mol_property_score, reward_baseline_per_step


class GraphDFForGuacamol(GraphDF):
    def __init__(self, conf, type):
        self.type = type
        self.model_conf_dict = conf['model']
        self.num_min_node = conf['num_min_node']
        self.num_max_node = conf['num_max_node']
        self.temperature = conf['temperature']
        self.atomic_num_list = conf['atom_list']
        super(GraphDFForGuacamol, self).__init__()

    def get_model(self, task, model_conf_dict, checkpoint_path=None):
        if model_conf_dict['use_gpu'] and not torch.cuda.is_available():
            model_conf_dict['use_gpu'] = False
        if task == 'rand_gen':
            self.model = GraphFlowModel(model_conf_dict)
        elif task == 'prop_opt':
            self.model = GraphFlowModel_rl_ForGuacamol(model_conf_dict, checkpoint_path, 'prop_opt')
        elif task == 'const_prop_opt':
            self.model = GraphFlowModel_con_rl(model_conf_dict)
        else:
            raise ValueError('Task {} is not supported in GraphDF!'.format(task))
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))

        self.model.to('cuda')

    def generate(self, number_samples=10):
        self.model.eval()
        smiles_list = []
        cnt_mol = 0

        while cnt_mol < number_samples:
            mol, _, num_atoms = self.model.generate(atom_list=self.atomic_num_list, min_atoms=self.num_min_node, max_atoms=self.num_max_node, temperature=self.temperature)
            if (num_atoms >= self.num_min_node):
                cnt_mol += 1
                smiles = Chem.MolToSmiles(mol)
                smiles_list.append(smiles)
                if cnt_mol % 10 == 0:
                    print('Generated {} molecules'.format(cnt_mol))
        
        assert cnt_mol == number_samples, 'number of generated molecules does not equal num'

        return smiles_list

    def train_prop_opt_qed(self, lr, wd, max_iters, warm_up, model_conf_dict, pretrain_path, save_interval, save_dir):
        r"""
            Running fine-tuning for property optimization task.

            Args:
                lr (float): The learning rate for fine-tuning.
                wd (float): The weight decay factor for training.
                max_iters (int): The maximum number of training iters.
                warm_up (int): The number of linear warm-up iters.
                model_conf_dict (dict): The python dict for configuring the model hyperparameters.
                pretrain_path (str): The path to the saved pretrained model file.
                save_interval (int): Indicate the frequency to save the model parameters to .pth files,
                    *e.g.*, if save_interval=20, the model parameters will be saved for every 20 training iters.
                save_dir (str): The directory to save the model parameters.
        """

        self.get_model('prop_opt', model_conf_dict)
        self.load_pretrain_model(pretrain_path)
        self.model.train()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=wd)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        print('start finetuning model(reinforce)')
        moving_baseline = None
        for cur_iter in range(max_iters):
            optimizer.zero_grad()    
            loss, per_mol_reward, per_mol_property_score, moving_baseline = self.model.reinforce_forward_optim_qed(in_baseline=moving_baseline, cur_iter=cur_iter)

            num_mol = len(per_mol_reward)
            avg_reward = sum(per_mol_reward) / num_mol
            avg_score = sum(per_mol_property_score) / num_mol     
            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.flow_core.parameters()), 1.0)
            adjust_learning_rate(optimizer, cur_iter, lr, warm_up)
            optimizer.step()

            print('Iter {} | reward {}, score {}, loss {}'.format(cur_iter, avg_reward, avg_score, loss.item()))

            if cur_iter % save_interval == save_interval - 1:
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'prop_opt_net_{}.pth'.format(cur_iter)))

        print("Finetuning (Reinforce) Finished!")

    def train_prop_opt(self, lr, wd, max_iters, warm_up, model_conf_dict, pretrain_path, save_interval, save_dir, prop_fn):
        r"""
            Running fine-tuning for property optimization task.

            Args:
                lr (float): The learning rate for fine-tuning.
                wd (float): The weight decay factor for training.
                max_iters (int): The maximum number of training iters.
                warm_up (int): The number of linear warm-up iters.
                model_conf_dict (dict): The python dict for configuring the model hyperparameters.
                pretrain_path (str): The path to the saved pretrained model file.
                save_interval (int): Indicate the frequency to save the model parameters to .pth files,
                    *e.g.*, if save_interval=20, the model parameters will be saved for every 20 training iters.
                save_dir (str): The directory to save the model parameters.
        """
        self.get_model('prop_opt', model_conf_dict)
        self.load_pretrain_model(pretrain_path)
        self.model.train()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=wd)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        print('start finetuning model(reinforce)')
        moving_baseline = None
        for cur_iter in range(max_iters):
            optimizer.zero_grad()    
            loss, per_mol_reward, per_mol_property_score, moving_baseline = self.model.reinforce_forward_optim(in_baseline=moving_baseline, cur_iter=cur_iter, prop_func=prop_fn)

            num_mol = len(per_mol_reward)
            avg_reward = sum(per_mol_reward) / num_mol
            avg_score = sum(per_mol_property_score) / num_mol     
            loss.backward()
            wandb.log({'loss':loss})
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.flow_core.parameters()), 1.0)
            adjust_learning_rate(optimizer, cur_iter, lr, warm_up)
            optimizer.step()

            print('Iter {} | reward {}, score {}, loss {}'.format(cur_iter, avg_reward, avg_score, loss.item()))

            if cur_iter % save_interval == save_interval - 1:
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'prop_opt_net_{}.pth'.format(cur_iter)))

        print("Finetuning (Reinforce) Finished!")
