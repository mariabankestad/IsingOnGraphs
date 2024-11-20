import argparse
import yaml
import torch
import os
from models.gnn_models import GIN_3layerEA
from dataset.mutag.create_mutag import get_mutag_dataset_model


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def train(model, loader, optimizer,criterion, device = "cuda"):

    
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out,data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(loader.dataset)

def train_model(config,train_loader,model, optimizer, device = "cuda"):
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    for epoch in range(1, config["num_epochs"] + 1):
        loss = train(model, train_loader,optimizer, criterion, device)
        print("Epoch: " + str(epoch) + " Train loss: " + str(loss))
    return model


def evaluate_model(test_loader,model, device = "cuda"):
    model.eval()

    total_correct = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = out.argmax(dim=-1).flatten()
        total_correct += int((pred == data.y.flatten()).sum())
    print("Test loss: " + str(total_correct / len(test_loader.dataset)))



def get_dataset(config):
    name = config['dataset_name']
    root_name = "root/" + name
    if name == "mutagenicity":
        train_data, test_data = get_mutag_dataset_model(root = root_name, 
                                                        train_test_split = config["train_test_split"], 
                                                        seed = 0, 
                                                        batch_size = config['batch_size'])
        return train_data, test_data
    else:
        print("Dataset name in unknown")
        return None, None
    
def get_model(config):
    device = "cuda"
    model  = GIN_3layerEA(config["input_dim"],
                          config["hidden_dim"],
                          config["output_dim"], 
                          edge_dim = config["edge_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    path = config["model_path"]
    exist = False
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        exist = True
        print("Model already exist, loading state_dict.")
    return model, optimizer, exist    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a machine learning model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")    
    args = parser.parse_args()

    
    # Load configuration
    config = load_config(args.config)
    model, optimizer, exist =get_model(config)
    train_loader, test_loader = get_dataset(config)
    if not exist:
        model = model = train_model(config, train_loader,model, optimizer)
        torch.save(model.state_dict(), config["model_path"])
    
    evaluate_model(test_loader,model)

