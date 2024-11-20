from models.ising_models import color_nodes
from torch_geometric.data import InMemoryDataset
from .mutagenicity import Mutagenicity
import random
from torch_geometric.seed import seed_everything
from torch_geometric.loader import DataLoader,ImbalancedSampler

def color_nodes_(dataset):
    dataset_ = []
    for de in dataset:
        d = de[0]
        if d.num_nodes == 0:
            continue
        d.color = color_nodes(edge_index=d.edge_index, num_nodes=d.num_nodes)
        dataset_.append(d)
    return dataset_


def color_nodes_and_mask(dataset):
    dataset_ = []
    for de in dataset:
        d = de[0]
        if d.num_nodes == 0:
            continue
        d.color = color_nodes(edge_index=d.edge_index, num_nodes=d.num_nodes)
        dataset_.append((d, de[1]))
    return dataset_



def get_mutag_dataset_model( root = r"mutag\mutagenicity", train_test_split=(0.9, 0.1), 
                      seed = 0, batch_size = 64):

    seed_everything(seed)
    dataset = Mutagenicity( root =root)
    dataset = color_nodes_(dataset)
    random.shuffle(dataset)
    n = int(len(dataset) * train_test_split[1])
    test_dataset = dataset[:n]
    train_dataset = dataset[n:]

    test_dataset_ = InMemoryDataset()
    data, slices = test_dataset_.collate(test_dataset)
    test_dataset_.data = data
    test_dataset_.slices = slices 

    train_dataset_ = InMemoryDataset()
    data, slices = train_dataset_.collate(train_dataset)
    train_dataset_.data = data
    train_dataset_.slices = slices  

    test_loader = DataLoader(test_dataset_, batch_size=1)
    sampler_train = ImbalancedSampler(train_dataset_)
    train_loader = DataLoader(train_dataset_, batch_size=batch_size, sampler=sampler_train)
    return train_loader, test_loader


def get_mutag_dataset_explainer( root = r"mutag\mutagenicity", seed = 0, batch_size = 64):

    seed_everything(seed)
    dataset = Mutagenicity( root =root)
    test_loader = color_nodes_and_mask(dataset)

    dataset = color_nodes_(dataset)
    random.shuffle(dataset)

    dataset_ = InMemoryDataset()
    data, slices = dataset_.collate(dataset)
    dataset_.data = data
    dataset_.slices = slices

    sampler_sub = ImbalancedSampler(dataset_)

    train_loader= DataLoader(dataset_, batch_size=batch_size, sampler=sampler_sub)


    return train_loader, test_loader
