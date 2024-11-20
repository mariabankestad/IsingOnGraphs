import torch
from torch_geometric.nn import GINEConv, global_mean_pool

class GIN_3layerNodeEA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super(GIN_3layerNodeEA, self).__init__()
        self.mlp1 = torch.nn.Linear(in_channels, hidden_channels)
        self.conv1 = GINEConv(self.mlp1,edge_dim = edge_dim)
        self.mlp2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv2 = GINEConv(self.mlp2,edge_dim = edge_dim)
        self.mlp3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv3 = GINEConv(self.mlp3,edge_dim = edge_dim)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index,edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index,edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index,edge_attr)
        x = x.relu()

        x = self.lin(x)

        return x
   
    
class GIN_3layerEA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super(GIN_3layerEA, self).__init__()
        self.mlp1 = torch.nn.Linear(in_channels, hidden_channels)
        self.conv1 = GINEConv(self.mlp1,edge_dim = edge_dim)
        self.mlp2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv2 = GINEConv(self.mlp2,edge_dim = edge_dim)
        self.mlp3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv3 = GINEConv(self.mlp3,edge_dim = edge_dim)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.single_output = False

    def forward(self, x, edge_index, edge_attr,batch):
        x = self.conv1(x, edge_index,edge_attr = edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index,edge_attr = edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index,edge_attr = edge_attr)
        x = x.relu()
        x = global_mean_pool(x, batch.to(x.device))

        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        if self.single_output:
            x = torch.softmax(x,1)
            return x[:,1].flatten()

        return x
    