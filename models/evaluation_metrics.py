import torch
from torch_geometric.utils import subgraph

def derive_fid_harm_score(node_imp, data, model):
    device = data.x.device
    node_imp_inv = (node_imp == 0).long()
    batch=torch.zeros(data.x.size(0)).long().to(device)
    out  = model(data.x, data.edge_index, data.edge_attr, batch)
    edge_index_s, edge_attr_s = subgraph(node_imp_inv.bool().to(device), data.edge_index, edge_attr = data.edge_attr,relabel_nodes=True)
    x_s = data.x[node_imp_inv.bool()]
    batch = torch.zeros(x_s.size(0), dtype = torch.int64).to(device)
    out2 = model(x_s,edge_index_s,edge_attr_s,batch)

    edge_index_s, edge_attr_s = subgraph(node_imp.bool().to(device), data.edge_index, edge_attr = data.edge_attr,relabel_nodes=True)
    x_s = data.x[node_imp.bool()]
    batch = torch.zeros(x_s.size(0), dtype = torch.int64).to(device)
    out3 = model(x_s,edge_index_s,edge_attr_s,batch)

    if len(out2) == 0:
        return None, None, None, None
    if len(out3) == 0:
        return None, None, None, None    
    if out.shape[-1] > 1:
        true= out.softmax(-1)[-1,-1].item()
        pred_pos= out2.softmax(-1)[-1,-1].item()
        pred_neg= out3.softmax(-1)[-1,-1].item()  
    else:
        true= out[-1].float().item()
        pred_pos= out2[-1].float().item()
        pred_neg= out3[-1].float().item()

    fid_plus = abs(true-pred_pos) + 0.001
    fid_min = (1-abs(true-pred_neg))  + 0.001

    fid_harm = fid_plus*fid_min/(fid_plus*0.5+fid_min*0.5)
    return fid_harm, true, pred_pos, pred_neg



def graph_exp_acc(gt_exp, generated_exp, node_thresh_factor = 0.1):
    '''
    Args:
        gt_exp (Explanation): Ground truth explanation from the dataset.
        generated_exp (Explanation): Explanation output by an explainer.
    '''

    EPS = 1e-09
    JAC_node = None
 
    JAC_node = []
    thresh_node = node_thresh_factor*generated_exp.max()
    node_imp = torch.zeros(len(generated_exp))
    for e_ in gt_exp:
        node_imp = node_imp + e_.node_imp

    node_imp = (node_imp > 0).float()

    for exp in [node_imp]:
        TPs = []
        FPs = []
        FNs = []
        true_nodes = exp.nonzero(as_tuple=True)[0]
        for node in range(len(generated_exp)):
            # Restore original node numbering
            positive = generated_exp[node].item() > thresh_node
            if positive:
                if node in true_nodes:
                    TPs.append(node)
                else:
                    FPs.append(node)
            else:
                if node in true_nodes:
                    FNs.append(node)
        TP = len(TPs)
        FP = len(FPs)
        FN = len(FNs)
        JAC_node.append(TP / (TP + FP + FN + EPS))

    JAC_node = sum(JAC_node)/len(JAC_node)

    return JAC_node

   