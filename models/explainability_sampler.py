import torch
import torch.nn.functional as F
from .ising_models import Ising_Graph
from torch_geometric.utils import scatter, subgraph
from .gnn_models import GIN_3layerNodeEA


 

class SamplerEA(torch.nn.Module):
    def __init__(self,temp= 0.1,J = 1, device = "cuda", input_dim = 10, edge_dim = 3):
            super().__init__()
            self.model = GIN_3layerNodeEA(input_dim,64,1, edge_dim = edge_dim).to(device).to(device)
            self.simulator =   Ising_Graph(temp =temp, J = J)
            self.red = 0.
    def forward(self, x, edge_index, edge_attr,batch, classification_model, colors, train = True, y = None, n_iter = 5, temp = 0):
        
        h = self.model(x = x, edge_index = edge_index,edge_attr = edge_attr)
        mean = scatter(h, batch, dim=0, reduce='mean')
        h = h - mean[batch]  
        h = h -self.red + torch.randn(h.size()).to(h.device)*temp

        reg = torch.mean((h/10)**2) 
        if train:
            energies = []
            losses = []
            unique = []
            masks = []
            for k_ in range(2):
                sim = self.simulator.simulate(edge_index = edge_index, h = h.flatten(), colors = colors, n_iter = n_iter)
                mask = sim>0.
                while sum(mask).item() == 0:
                    sim = self.simulator.simulate(edge_index = edge_index, h = h.flatten(), colors = colors, n_iter = n_iter)
                    mask = sim>0.
    
                masks.append(mask)
                energy = self.simulator.energy(mask.float(), h.flatten(), edge_index)   
                energies.append(scatter(energy, batch, dim=0, reduce='mean'))
                b_mask = batch[mask]
                unique.append(b_mask.unique())
                
            mask_batch = []
            mask_batch.append(torch.zeros(x.size(0)).bool().to(x.device))
            ind_train = []
            batch.max()
            batch_ = batch.clone()
            for i in range(batch.max()):
                if (i in unique[0]) and (i in unique[1]):
                    mask_batch.append(batch == i)
                    ind_train.append(i)
                else:
                      batch_[batch>=i] = batch_[batch>=i] -1
            if len(ind_train) == 0:
                return None, None, None
            
            mask_batch = sum(mask_batch).bool()
            ind_train = torch.tensor(ind_train)
            batch1 = batch_[mask_batch]

            x_ = x[mask_batch]
            edge_index1,edge_attr1 = subgraph(mask_batch, edge_index,edge_attr = edge_attr, relabel_nodes = True )
            with torch.no_grad():
                out1 = classification_model(x, edge_index, edge_attr,batch)
                out1 = out1.argmax(dim=-1).flatten()
                mask_corr = out1 == y
                mask_corr = mask_corr[ind_train]
            for k_ in range(2):
                with torch.no_grad():
                    mask_ = masks[k_][mask_batch]
                    edge_index2,edge_attr2 = subgraph(mask_, edge_index1,edge_attr = edge_attr1,relabel_nodes = True )
                    out2 = classification_model(x_[mask_], edge_index2, edge_attr = edge_attr2,batch = batch1[mask_])

                    loss = F.cross_entropy(out2[mask_corr], (out1[ind_train])[mask_corr],reduction='none')

                    losses.append(loss.flatten())

            f_diff = (losses[0]-losses[1])
            e_diff = (energies[0][ind_train])[[mask_corr]]- (energies[1][ind_train])[[mask_corr]]

            loss_ = -(f_diff*e_diff).mean()
            return loss_, reg, (losses[0].mean().item() + losses[1].mean().item())/2
        else:
            with torch.no_grad():

                #sub_edge_index = subgraph(mask, edge_index)

                sim = self.simulator.simulate(edge_index = edge_index, h = h.flatten(), colors = colors, n_iter = n_iter)
                mask = sim>0.
                return mask, h.flatten()
 