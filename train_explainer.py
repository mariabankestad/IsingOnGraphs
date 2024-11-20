import argparse
import yaml
import torch
import os
from models.gnn_models import GIN_3layerEA
from dataset.mutag.create_mutag import get_mutag_dataset_explainer,get_mutag_dataset_model
from models.evaluation_metrics import graph_exp_acc, derive_fid_harm_score
from models.explainability_sampler import SamplerEA
import torch.optim.lr_scheduler as lr_scheduler

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def train_explainer(explainer, model, train_loader, optimizer, config, device = "cuda"):
    model = model.to(device)
    explainer = explainer.to(device)
    red = 0.0
    temp = 0.1
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=150)

    for epoch in range(config["num_epochs"]):
        if epoch > 50:
            red = red + 0.005
        if red > 1:
            red = 1
        temp = temp - 0.003
        if temp < 0:
            temp = 0
        explainer.red = red
        losses = []
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss,loss_reg,l_v = explainer(data.x, data.edge_index, data.edge_attr,data.batch, model,data.color, y = data.y, n_iter = 2, temp = temp)
            losses.append(l_v)
            if loss is None:
                continue
            loss_tot = loss + loss_reg/50
            loss_tot.backward()
            optimizer.step() 
        print("Epoch " + str(epoch) + ", Loss: " + str(sum(losses)/len(losses)))
        scheduler.step()
    return explainer 



def evaluate_explainer(explainer, model, test_loader, device = "cuda"):
    model = model.to(device)
    explainer = explainer.to(device)
    fid_scores = []
    gaes =[]
    pred_true =[]
    pred_poss = []
    pred_neg = []
    for d_ in test_loader:
        data = d_[0]
        data = data.to(device)
        if data.y.item() < 0.5:
            continue
        batch = torch.zeros(data.x.size(0)).long().to(data.x.device)
        mask, h = explainer(data.x, data.edge_index,data.edge_attr, batch, model,data.color, y = data.y,train = False, n_iter = 5)
        gaes.append(graph_exp_acc(d_[1], mask))
        fid,true_, pos_, neg_ = derive_fid_harm_score(mask, data, model)
        if fid is not None:
            fid_scores.append(fid)
            pred_true.append(true_)
            pred_poss.append(pos_)
            pred_neg.append(neg_)


    fid_scores= torch.tensor(fid_scores)
    gaes= torch.tensor(gaes)
    pred_trues = torch.tensor(pred_true)
    pred_poss = torch.tensor(pred_poss)
    pred_negs = torch.tensor(pred_neg)

    return gaes, fid_scores, pred_trues, pred_poss,pred_negs

  

def get_dataset(config):
    name = config['dataset_name']
    root_name = "root/" + name
    if name == "mutagenicity":
        train_loader, test_loader = get_mutag_dataset_explainer(root = root_name, 
                                                        seed = 0, 
                                                        batch_size = config['batch_size'])
        return train_loader, test_loader
    else:
        print("Dataset name unknown.")
        return None, None

    
def get_model(config):
    device = "cuda"
    model  = GIN_3layerEA(config["input_dim"],
                          config["hidden_dim"],
                          config["output_dim"], 
                          edge_dim = config["edge_dim"]).to(device)
    path = config["model_path"]
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        return model 
    else:
        print("Model do not exist!")
        return None

def get_ising_explainer(config):
    explainer = SamplerEA(input_dim = 14, edge_dim = 3)
    path = config["explainer_path"]
    exist = False
    if os.path.exists(path):
        explainer.model.load_state_dict(torch.load(path))
        exist = True
    optimizer = torch.optim.Adam(explainer.parameters(), lr=config["lr"])    
    return explainer, optimizer, exist    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a ising explainer model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")    
    args = parser.parse_args()

    
    # Load configuration
    config = load_config(args.config)
    train_loader, test_loader = get_dataset(config)
    model =get_model(config)
    if model:
        
        explainer, optimizer, exist = get_ising_explainer(config)
        if not exist:
            explainer = train_explainer(explainer, model, train_loader, optimizer, config)
            torch.save(explainer.model.state_dict(), config["explainer_path"])
        gaes, fid_scores, _, _,_, = evaluate_explainer(explainer, model, test_loader)
        print("GAE: " + str(gaes.mean().item())+ " " + str(gaes.std().item())+ ", fid: " + str(fid_scores.mean()))

