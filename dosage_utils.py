import os
import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import sklearn.metrics
import sklearn


def train_model(model, optimizer, trainloader, valloader, criterion, device, save_path, num_epochs, verbose, dim=1):
    avg_train_loss, avg_val_loss = [], []
    top_1_acc, lowest_val_loss = 0, np.inf
    
    for e in range(num_epochs):
        t_loss, v_loss = [], []
        for x, y in trainloader:
            if dim == 1:
                x, y = x.unsqueeze(-1).float().to(device), y.long().to(device)
            elif dim == 2:
                x, y = x.float().to(device), y.long().to(device)
                
            pred = model(x)
            loss = criterion(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            t_loss.append(loss.detach().cpu().numpy())
            
        acc_l = []
        for x, y in valloader:
            if dim == 1:
                x, y = x.unsqueeze(-1).float().to(device), y.long().to(device)
            elif dim == 2:
                x, y = x.float().to(device), y.long().to(device)
            
            pred = model(x)
            loss = criterion(pred, y)
            
            acc = sum(torch.argmax(F.softmax(pred, dim=1), dim=1) == y) / len(y)
            acc_l.append(acc.detach().cpu().numpy())
            v_loss.append(loss.detach().cpu().numpy())
            
        avg_train_loss.append(np.mean(t_loss))
        avg_val_loss.append(np.mean(v_loss))
        
        if np.mean(acc_l) > top_1_acc:
            top_1_acc = np.mean(acc_l)
        
        if np.mean(v_loss) < lowest_val_loss:
            lowest_val_loss = np.mean(v_loss)
            
            torch.save({
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, save_path)
        
        if e % verbose == 0:
            print(f'Epoch: {e}, Train Loss: {np.mean(avg_train_loss)}, Val Loss: {np.mean(avg_val_loss)}, Top 1 Acc: {top_1_acc}, Lowest Val Loss: {lowest_val_loss}')
        
    res = {
        'model': model,
        'optimizer': optimizer,
        'avg_train_loss': avg_train_loss,
        'avg_val_loss': avg_val_loss,
        'top_1_acc': top_1_acc,
        'lowest_val_loss': lowest_val_loss
    }
    
    return res


def prep_dosage_analysis(folder_path, train_csv_path, test_csv_path, dim=1):
    '''
    Returns a dictionary of items for training a classification model and performing analysis
    Trains on R1, test on R2
    '''
    if dim == 1:
        train_set = DosageDataset(folder_path, train_csv_path, remove_nans=True)
        test_set = DosageDataset(folder_path, test_csv_path, remove_nans=True)
    elif dim == 2:
        train_set = DosageDataset2D(folder_path, train_csv_path, remove_nans=True)
        test_set = DosageDataset2D(folder_path, train_csv_path, remove_nans=True)
    
    # create dataloaders
    t, v = int(len(train_set) * 0.9), int(len(train_set) * 0.1)
    while t + v != len(train_set):
        t += 1
    
    train, val = torch.utils.data.random_split(train_set, [t, v])
    trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=64)
    valloader = torch.utils.data.DataLoader(val, shuffle=True, batch_size=64)
    
    return trainloader, valloader, test_set


def visualize_trajectories(dataset, dim=1, title=None):
    
    sep = []
    for i in range(len(dataset) - 1):
        if dataset.labels[i] != dataset.labels[i + 1]:
            sep.append(i + 1)
                
    if dim == 1:     
        trajectories = []
        trajectories.append(np.mean(dataset.data[0:sep[0]], axis=0))
        for i in range(len(sep) - 1):
            trajectory = np.mean(dataset.data[sep[i]:sep[i+1]], axis=0)
            trajectories.append(trajectory)
        trajectories.append(np.mean(dataset.data[sep[-1]:], axis=0))
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set(title=title)
        for dose, i in enumerate(trajectories):
            ax.plot(i, label='Dose ' + str(1 + dose))
            ax.legend()
            
        print(len(trajectories))
       
    
    elif dim == 2:
        trajectories_y, trajectories_z = [], []
        trajectories_y.append(np.mean(dataset.data[0:sep[0], :, 0], axis=0))
        trajectories_z.append(np.mean(dataset.data[0:sep[0], :, 1], axis=0))
        
        for i in range(len(sep) - 1):
            trajectory_y = np.mean(dataset.data[sep[i]:sep[i+1], :, 0], axis=0)
            trajectory_z = np.mean(dataset.data[sep[i]:sep[i+1], :, 1], axis=0)
            
            trajectories_y.append(trajectory_y)
            trajectories_z.append(trajectory_z)
            
        trajectories_y.append(np.mean(dataset.data[sep[-1]:, :, 0], axis=0))
        trajectories_z.append(np.mean(dataset.data[sep[-1]:, :, 1], axis=0))
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.set(title=title)
        
        for dose, (i, j) in enumerate(zip(trajectories_y, trajectories_z)):
            ax.plot(
                xs = np.linspace(1, 95, 95),
                ys = i,
                zs = j,
                label = 'Dose ' + str(dose + 1)
            )
            ax.legend()