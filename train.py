import json
import torch
import os
from tqdm import tqdm
from models.resnet import ResNet1d_mse
from data.CustomDataset import CustomDataset
import torch.optim as optim
import numpy as np

# loss function mse
def compute_mse(ages, pred_ages, weights):
    diff = ages.flatten() - pred_ages.flatten() #shape:[batch_size]
    loss = torch.sum(weights.flatten() * diff * diff) 
    return loss


def compute_weights(ages, max_weight=np.inf):
    _, inverse, counts = np.unique(ages, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse] #counts[inverse] == counts for each (unique) age
    normalized_weights = weights / sum(weights)
    w = len(ages) * normalized_weights
    # Truncate weights to a maximum
    if max_weight < np.inf:
        w = np.minimum(w, max_weight)
        w = len(ages) * w / sum(w)
    return w



def train(ep, dataload):
    model.train()
    total_loss = 0
    n_entries = 0
    train_desc = "Epoch {:2d}: Model train - Loss: {:.6f}"
    train_bar = tqdm(initial=0, leave=True, total=len(dataload),
                     desc=train_desc.format(ep, 0), position=0)
    for traces, ages, weights in dataload:
        traces = traces.transpose(1, 2).float()
        traces, ages, weights = traces.to(device), ages.to(device), weights.to(device)
        model.zero_grad()
        # Send to device
        # Forward pass
        pred_ages = model(traces) 
        loss = compute_mse(ages, pred_ages, weights) #pred_ages shape: [batch_size, 1]; ages, weights shape: [batch_szie]
        # Backward pass
        loss.backward() 
        # Optimize
        optimizer.step()
        # Update
        bs = len(traces)
        total_loss += float(loss.detach().cpu().numpy())
        n_entries += bs
        # Update train bar
        train_bar.desc = train_desc.format(ep, total_loss / n_entries)
        train_bar.update(1)
    train_bar.close()
    return total_loss / n_entries


def eval(ep, dataload):
    model.eval()
    total_loss = 0
    n_entries = 0
    eval_desc = "Epoch {:2d}: valid - Loss: {:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(dataload),
                    desc=eval_desc.format(ep, 0), position=0)
    for traces, ages, weights in dataload:
        traces = traces.transpose(1, 2).float()
        traces, ages, weights = traces.to(device), ages.to(device), weights.to(device)
        with torch.no_grad():
            # Forward pass
            pred_ages = model(traces)
            loss = compute_mse(ages, pred_ages, weights)
            # Update outputs
            bs = len(traces)
            # Update ids
            total_loss += float(loss.detach().cpu().numpy())
            n_entries += bs
            # Print result
            eval_bar.desc = eval_desc.format(ep, total_loss / n_entries)
            eval_bar.update(1)
    eval_bar.close()
    return total_loss / n_entries


if __name__ == "__main__":
    import pandas as pd
    import argparse
    import yaml
    from torch.utils.data import DataLoader
    from warnings import warn
    import wandb
    import torch.nn as nn

    # Arguments that will be saved in config file
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Train model to predict age from the raw ecg tracing.')
    parser.add_argument('script_yaml',
                        help='script file (yaml) for run')                                          
    cmd_args = parser.parse_args()

    with open(f'scripts/{cmd_args.script_yaml}.yaml') as f:
        args = yaml.safe_load(f) #in dictionary

    data = args["data"]
    setup = args["setup"]
    module_model = args["module"]["model"]
    module_optim = args["module"]["optim"]

    torch.manual_seed(setup['seed'])
    print(args)

    # use wandb to track & log the training process
    wandb.login
    # wandb config
    wandb.init(
        project="PROPHECG-Age", entity="test_sjeom",  # change this part to yours!
        name=module_model['model_name'],
        config={
            "model_name": module_model['model_name'], 
            "epochs": setup['epochs'], 
            "earlystop" : setup['earlystop'],
            "batch_size": setup['batch_size'], 
            "lr": module_optim['lr'], 
            "dropout_rate": module_model['dropout_rate'], 
            "kernel_size": module_model['kernel_size'], 
            "ec2" : setup['ec2'], 
            "num_workers" : setup['num_workers']
        })
    
    config = wandb.config

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = os.path.join(data['folder'], module_model['model_name'])

    # Generate output folder if needed
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save config file
    with open(os.path.join(folder, f"{module_model['model_name']}_args.json"), 'w') as f: #name edit
        json.dump(args, f, indent='\t')
    
    tqdm.write("Building data loaders...")
    # Get csv data
    # train
    ## train dataset splitted into 11 due to huge data size and for faster training; last splitted data with different size
    for i in range(0, 11):
        globals()["train_ages_{}".format(i)] = np.array(pd.read_csv(data["train"]["trains"]["csv"]+f"{i}.csv")[data["age_col"]])
    # valid
    valid_ages = np.array(pd.read_csv(data["train"]['valid']['csv'])[data['age_col']])
    
    # weights; must be done all together (train + valid)
    whole_ages = [globals()["train_ages_{}".format(i)] for i in range(0,11)]
    whole_ages.append(valid_ages)
    whole_ages_concat = np.concatenate(whole_ages)
    print(whole_ages_concat.shape)
    weights = compute_weights(whole_ages_concat)

    weights_idx = [len(whole_ages[0]) * i for i in range(0, 11)] #idx for former 10 train datasets
    weights_idx.append(weights_idx[-1]+len(train_ages_10)) # add idx for the last 11th train dataset
    print(weights_idx)

    # Dataset and Dataloader
    for i in range(0, 11):
        globals()["train_dataset_{}".format(i)] = CustomDataset(whole_ages[i], weights[weights_idx[i]:weights_idx[i+1]], data["train"]['trains']['trace']+f"{i}.npy")
    
    for i in range(0, 11):
        globals()["train_loader_{}".format(i)] = DataLoader(dataset=globals()["train_dataset_{}".format(i)], num_workers=config.num_workers, batch_size=config.batch_size, shuffle=True, drop_last=False)

    valid_dataset = CustomDataset(valid_ages, weights[weights_idx[-1]:], data["train"]['valid']['trace'])
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=config.num_workers, batch_size=config.batch_size, shuffle=False, drop_last=False)
    
    tqdm.write("Done!")

    tqdm.write("Define model...")
    N_LEADS = 8  # the 8 leads
    N_CLASSES = 1  # just the age

    model = ResNet1d_mse(input_dim=(N_LEADS, setup['seq_length']),
                blocks_dim=list(zip(module_model['net_filter_size'], module_model['net_seq_length'])),
                n_classes=N_CLASSES,
                kernel_size=config.kernel_size,
                dropout_rate=config.dropout_rate)

    # Data parallelism
    model = nn.DataParallel(model)
    
    model.to(device)
    tqdm.write("Done!")

    tqdm.write("Define optimizer...")
    optimizer = optim.Adam(model.parameters(), config.lr)
    tqdm.write("Done!")

    tqdm.write("Define scheduler...")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=module_optim['patience'],
                                                     min_lr=module_optim['lr_factor'] * module_optim['min_lr'],
                                                     factor=module_optim['lr_factor'])
    tqdm.write("Done!")

    tqdm.write("Training...")
    history = pd.DataFrame(columns=['model', 'epoch', 'train_loss', 'valid_loss', 'lr'])
    
    start_epoch = 0
    best_loss = np.Inf
    patience = 0
    for ep in range(start_epoch, config.epochs):
        _ = train(ep, train_loader_0)
        _ = train(ep, train_loader_1)
        _ = train(ep, train_loader_2)
        _ = train(ep, train_loader_3)
        _ = train(ep, train_loader_4)
        _ = train(ep, train_loader_5)
        _ = train(ep, train_loader_6)
        _ = train(ep, train_loader_7)
        _ = train(ep, train_loader_8)
        _ = train(ep, train_loader_9)
        train_loss = train(ep, train_loader_10)
        valid_loss = eval(ep, valid_loader)
        # Save best model
        if valid_loss < best_loss:
            # Save model
            torch.save({'epoch': ep,
                        'model': model.module.state_dict(),
                        'valid_loss': valid_loss,
                        'optimizer': optimizer.state_dict()},
                        os.path.join(folder, f'best_model.pth'))
            # Update best validation loss
            best_loss = valid_loss
            patience = 0
            print(f"Model saved!!")
        elif valid_loss > best_loss:
            # if loss doesn't get better
            patience += 1

        # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        # Interrupt for minimum learning rate
        if learning_rate < module_optim['min_lr']:
            break

        # Print message
        tqdm.write('Epoch {:2d}: Train Loss {:.6f} ' \
                '\tValid Loss {:.6f} \tLearning Rate {:.7f}\t'
                .format(ep, train_loss, valid_loss, learning_rate))
    
        # wandb log
        metrics = {
            f"epoch": ep, 
            f"train_loss": train_loss, 
            f"val_loss": valid_loss,
            f"learning_rate": learning_rate
            }
            
        wandb.log(metrics)

        # Save history
        history = history.append({"epoch": ep, "train_loss": train_loss,
                                "valid_loss": valid_loss, "lr": learning_rate}, ignore_index=True)
        history.to_csv(os.path.join(folder, f'{config.model_name}_history.csv'), index=False)
        # Update learning rate
        scheduler.step(valid_loss)
        
        print(f'Patience : {patience}')
        if patience == config.earlystop:
            break
    wandb.finish()
    tqdm.write("Done!")
