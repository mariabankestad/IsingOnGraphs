import torch
import numpy as np
from torch import nn
from torch_geometric.utils import scatter,to_networkx
import networkx as nx
from torch_geometric.data import Data


def color_nodes(edge_index, num_nodes):
    # Convert edge_index to NetworkX edges directly
    data = Data(edge_index=edge_index, num_nodes = num_nodes)
    G = to_networkx(data)
    # Perform greedy coloring
    graph_coloring = nx.coloring.greedy_color(G, strategy="largest_first")

    # Create a color tensor directly from the dictionary
    colors = torch.tensor(
        [graph_coloring[node] for node in range(num_nodes)], 
        dtype=torch.int64
    )
    return colors

def checkerboard(n):
    a = torch.from_numpy(np.resize([0,1], n))
    return torch.abs(a-a.unsqueeze(0).T)


class Ising_Batch_images():
    def __init__(self, dim, J = -1, temp = 1):
        ''' Simulating the Ising model '''    
        ## monte carlo moves
        self.conv = nn.Conv2d(1,1,kernel_size = 3, stride=1, padding=1, padding_mode = "zeros", bias = False)
        self.conv.weight = torch.nn.Parameter(torch.tensor([[[[0,1,0],[1,0,1], [0.,1.,0.]]]]))
        self.conv.weight.requires_grad = False
        self.chess_board = checkerboard(dim)
        self.J = J
        self.temp = temp

    def mcmove_chess(self, C, beta, h):
        for board in [self.chess_board.to(h.device), torch.abs(self.chess_board.to(h.device)-1)]:
            mess = self.conv(C)
            cost = 2*C*(self.J*mess + h)
            mask = torch.logical_or(cost< 0 , torch.rand(cost.shape, device = h.device) < torch.exp(-cost*beta))
            mask = torch.logical_and(mask,board)
            C[mask] =  C[mask]* -1
        return C

    def simulate(self, h = None, n_iter = 100):   
        ''' This module simulates the Ising model'''
        C  = torch.randint(low=0, high= 2, size = h.shape, device = h.device)*2 -1     
        C = C.float()   
        for i in range(n_iter):

            C = self.mcmove_chess(C, 1.0/self.temp, h)
        return C
    
    def energy(self, x, h):
        beta = 1.0/self.temp
        mess = self.conv(x)
        energy = -x*(self.J*mess + h)*beta    
        return energy

    def h_eff(self, x, h):
        mess = self.conv(x)
        return (self.J*mess + h)    
    


class Ising_Graph():
    ''' Simulating the Ising model '''    
    ## monte carlo moves
    def __init__(
        self,
        J: int = -1,
        temp: float = 1.,

    ) -> None:
        super().__init__()
        self.J = J
        self.temp = temp

    def mcmove_col(self, x, beta, h, edge_index, colors):

        unique_colors = torch.unique(colors)
        row,col = edge_index
        for c_ in unique_colors:
            mess = scatter(x[row], index = col, dim=0, dim_size=x.size(0), reduce='sum')
            mask = colors == c_
            x_u = x[mask]
            cost = 2*x[mask]*(self.J*mess[mask] + h[mask])*beta
            cost_mask = torch.logical_or(cost < 0, torch.rand(cost.size(0), device = x.device) < torch.exp(-cost))
            x_u[cost_mask] = x_u[cost_mask]*(-1)
            x[mask] = x_u
        return x

    
    def simulate(self, h, edge_index, colors, n_iter = 3):   
        ''' This module simulates the Ising model'''
        N = h.size(0)
        #start with a random value
        x = torch.randint(low=0, high= 2, size = (N,), device = h.device)*2 -1
        for i in range(n_iter):
            x = self.mcmove_col(x, 1.0/self.temp, h, edge_index, colors)
        
        return x
    
    def energy(self, x, h, edge_index):
        row, col = edge_index
        mess = scatter(x[row], index = col, dim=0, dim_size=x.size(0), reduce='sum')
        energy = -x*(self.J*mess + h)
        return energy
