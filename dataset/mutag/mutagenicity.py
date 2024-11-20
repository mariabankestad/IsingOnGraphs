import torch
import itertools
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx, remove_isolated_nodes
from torch_geometric.utils.convert import to_networkx
from .explanation import Explanation
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from .utils import to_networkx_conv, aggregate_explanations
from .substruct_chem_match import match_substruct_mutagenicity, MUTAG_NO2, MUTAG_NH2
from .substruct_chem_match import match_aliphatic_halide, match_nitroso, match_azo_type


class GraphDataset:
    def __init__(self, name, split_sizes = (0.7, 0.2, 0.1), seed = None, device = None):

        self.name = name

        self.seed = seed
        self.device = device
        # explanation_list - list of explanations for each graph

        if split_sizes[1] > 0:
            self.train_index, self.test_index = train_test_split(torch.arange(start = 0, end = len(self.graphs)), 
                test_size = split_sizes[1] + split_sizes[2], random_state=self.seed, shuffle = False)
        else:
            self.test_index = None
            self.train_index = torch.arange(start = 0, end = len(self.graphs))

        if split_sizes[2] > 0:
            self.test_index, self.val_index = train_test_split(self.test_index, 
                test_size = split_sizes[2] / (split_sizes[1] + split_sizes[2]),
                random_state = self.seed, shuffle = False)

        else:
            self.val_index = None

        self.Y = torch.tensor([self.graphs[i].y for i in range(len(self.graphs))]).to(self.device)

    def get_data_list(
            self,
            index,
        ):
        data_list = [self.graphs[i].to(self.device) for i in index]
        exp_list = [self.explanations[i] for i in index]

        return data_list, exp_list

    def get_loader(
            self, 
            index,
            batch_size = 16,
            **kwargs
        ):

        data_list, exp_list = self.get_data_list(index)

        for i in range(len(data_list)):
            data_list[i].exp_key = [i]

        loader = DataLoader(data_list, batch_size = batch_size, shuffle = True)

        return loader, exp_list

    def get_train_loader(self, batch_size = 16):
        return self.get_loader(index=self.train_index, batch_size = batch_size)

    def get_train_list(self):
        return self.get_data_list(index = self.train_index)

    def get_test_loader(self):
        assert self.test_index is not None, 'test_index is None'
        return self.get_loader(index=self.test_index, batch_size = 1)

    def get_test_list(self):
        assert self.test_index is not None, 'test_index is None'
        return self.get_data_list(index = self.test_index)

    def get_val_loader(self):
        assert self.test_index is not None, 'val_index is None'
        return self.get_loader(index=self.val_index, batch_size = 1)

    def get_val_list(self):
        assert self.val_index is not None, 'val_index is None'
        return self.get_data_list(index = self.val_index)

    def get_train_w_label(self, label):
        inds_to_choose = (self.Y[self.train_index] == label).nonzero(as_tuple=True)[0]
        in_train_idx = inds_to_choose[torch.randint(low = 0, high = inds_to_choose.shape[0], size = (1,))]
        chosen = self.train_index[in_train_idx.item()]

        return self.graphs[chosen], self.explanations[chosen]

    def get_test_w_label(self, label):
        assert self.test_index is not None, 'test_index is None'
        inds_to_choose = (self.Y[self.test_index] == label).nonzero(as_tuple=True)[0]
        in_test_idx = inds_to_choose[torch.randint(low = 0, high = inds_to_choose.shape[0], size = (1,))]
        chosen = self.test_index[in_test_idx.item()]

        return self.graphs[chosen], self.explanations[chosen]

    def get_graph_as_networkx(self, graph_idx):
        '''
        Get a given graph as networkx graph
        '''

        g = self.graphs[graph_idx]
        return to_networkx_conv(g, node_attrs = ['x'], to_undirected=True)

    def download(self):
        pass

    def __getitem__(self, idx):
        return self.graphs[idx], self.explanations[idx]

    def __len__(self):
        return len(self.graphs)
    
def match_edge_presence(edge_index, node_idx):
    '''
    Returns edge mask with the spots containing node_idx highlighted
    '''

    emask = torch.zeros(edge_index.shape[1]).bool()

    if isinstance(node_idx, torch.Tensor):
        if node_idx.shape[0] > 1:
            for ni in node_idx:
                emask = emask | ((edge_index[0,:] == ni) | (edge_index[1,:] == ni))
        else:
            emask = ((edge_index[0,:] == node_idx) | (edge_index[1,:] == node_idx))
    else:
        emask = ((edge_index[0,:] == node_idx) | (edge_index[1,:] == node_idx))

    return emask





def make_iter_combinations(length):
    '''
    Builds increasing level of combinations, including all comb's at r = 1, ..., length - 1
    Used for building combinations of explanations
    '''

    if length == 1:
        return [[0]]

    inds = np.arange(length)

    exps = [[i] for i in inds]
    
    for l in range(1, length - 1):
        exps += list(itertools.combinations(inds, l + 1))

    exps.append(list(inds)) # All explanations

    return exps


class Mutagenicity(GraphDataset):
    '''
    GraphXAI implementation Mutagenicity dataset
        - Contains Mutagenicity with ground-truth 

    Args:
        root (str): Root directory in which to store the dataset
            locally.
        generate (bool, optional): (:default: :obj:`False`) 
    '''

    def __init__(self,
        root: str,
        use_fixed_split: bool = True, 
        generate: bool = True,
        split_sizes = (0.7, 0.2, 0.1),
        seed = None,
        test_debug = False,
        device = None,
        ):

        self.device = device

        self.graphs = list(TUDataset(root=root, name='Mutagenicity'))
        # self.graphs retains all qualitative and quantitative attributes from PyG

        # Remove isolated nodes:
        for i in range(len(self.graphs)):
            edge_idx, _, node_mask = remove_isolated_nodes(self.graphs[i].edge_index, num_nodes = self.graphs[i].x.shape[0])
            self.graphs[i].x = self.graphs[i].x[node_mask]
            #print('Shape x', self.graphs[i].x.shape)
            self.graphs[i].edge_index = edge_idx

        self.__make_explanations(test_debug)

        # Filter based on label-explanation validity:
        self.__filter_dataset()

        super().__init__(name = 'Mutagenicity', seed = seed, split_sizes = split_sizes, device = device)


    def __make_explanations(self, test: bool = False):
        '''
        Makes explanations for Mutagenicity dataset
        '''

        self.explanations = []

        # Testing
        if test:
            count_nh2 = 0
            count_no2 = 0
            count_halide = 0
            count_nitroso = 0
            count_azo_type = 0

        # Need to do substructure matching
        for i in range(len(self.graphs)):

            molG = self.get_graph_as_networkx(i)

            if test:
                if molG.number_of_nodes() != self.graphs[i].x.shape[0]:
                    print('idx', i)
                    print('from data', self.graphs[i].x.shape)
                    print('from molG', molG.number_of_nodes())
                    print('edge index unique:', torch.unique(self.graphs[i].edge_index).tolist())
                    tmpG = to_networkx(self.graphs[i], to_undirected=True)
                    print('From PyG nx graph', tmpG.number_of_nodes())

            # Screen for NH2:
            nh2_matches = match_substruct_mutagenicity(molG, MUTAG_NH2, nh2_no2 = 0)

            # Screen for NO2:
            no2_matches = match_substruct_mutagenicity(molG, MUTAG_NO2, nh2_no2 = 1)

            # Screen for aliphatic halide
            halide_matches = match_aliphatic_halide(molG)

            # Screen for nitroso
            nitroso_matches = match_nitroso(molG)

            # Screen for azo-type
            azo_matches = match_azo_type(molG)

            if test:
                count_nh2 += int(len(nh2_matches) > 0)
                count_no2 += int(len(no2_matches) > 0)
                count_halide += int(len(halide_matches) > 0)
                count_nitroso += int(len(nitroso_matches) > 0)
                count_azo_type += int(len(azo_matches) > 0)

            all_matches = nh2_matches + no2_matches + nitroso_matches + azo_matches + halide_matches 

            eidx = self.graphs[i].edge_index

            explanations_i = []

            for m in all_matches:
                node_imp = torch.zeros((molG.number_of_nodes(),))
                
                node_imp[m] = 1
                edge_imp = match_edge_presence(eidx, m)

                exp = Explanation(
                    node_imp = node_imp.float(),
                    edge_imp = edge_imp.float()
                )

                exp.set_whole_graph(self.graphs[i])

                exp.has_match = True

                explanations_i.append(exp)


            if len(explanations_i) == 0:
                # Set a null explanation:
                exp = Explanation(
                    node_imp = torch.zeros((molG.number_of_nodes(),), dtype = torch.float),
                    edge_imp = torch.zeros((eidx.shape[1],), dtype = torch.float)
                )

                exp.set_whole_graph(self.graphs[i])

                exp.has_match = False

                explanations_i = [exp]

                self.explanations.append(explanations_i)
            
            else:
                # Combinatorial combination of matches:
                exp_matches_inds = make_iter_combinations(len(all_matches))

                comb_explanations = []

                # Run combinatorial build of all explanations
                for eid in exp_matches_inds:
                    # Get list of explanations:
                    L = [explanations_i[j] for j in eid]
                    tmp_exp = aggregate_explanations(L, node_level = False)
                    tmp_exp.has_match = True
                    comb_explanations.append(tmp_exp) # No reference provided
                    

                self.explanations.append(comb_explanations)

        if test:
            print(f'NH2: {count_nh2}')
            print(f'NO2: {count_no2}')
            print(f'Halide: {count_halide}')
            print(f'Nitroso: {count_nitroso}')
            print(f'Azo-type: {count_azo_type}')

    def __filter_dataset(self):
        '''
        TODO: could merge this function into __make_explanations, easier to keep
            it here for now
        '''
        #self.label_exp_mask = torch.zeros(len(self.graphs), dtype = bool)

        new_graphs = []
        new_exps = []

        for i in range(len(self.graphs)):
            matches = int(self.explanations[i][0].has_match)
            yval = int(self.graphs[i].y.item())

            #self.label_exp_mask[i] = (matches == yval)
            if matches == yval:
                new_graphs.append(self.graphs[i])
                new_exps.append(self.explanations[i])

        # Perform filtering:
        #self.graphs = [self.graphs[] for i in self.label_exp_mask]
        self.graphs = new_graphs
        self.explanations = new_exps
        # mask_inds = self.label_exp_mask.nonzero(as_tuple = True)[0]
        # self.explanations = [self.explanations[i.item()] for i in mask_inds]