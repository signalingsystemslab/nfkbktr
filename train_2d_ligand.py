import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections 
import copy

from sklearn.preprocessing import StandardScaler
import sklearn.metrics
import sklearn
from sklearn.utils import shuffle
from sklearn.utils import class_weight

from data import *
from dosage_utils import *
from utils import *
from network import *

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--nfkb-path', type=str, default=None)
    parser.add_argument('--ktr-path', type=str, default=None)
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--input-size', type=int, default=2)
    parser.add_argument('--hidden-size', type=int, default=95)
    parser.add_argument('--output-size', type=int, default=4)
    parser.add_argument('--num-layers' type=int, default=2)
    parser.add_argument('--linear-hidden-dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--save-path', type=str, default=None)
    
    args = parser.parse_args()
    return args



def main(args):
    device = torch.device('cuda:0')
    
    nfkbktr_dataset = NFkBKTR_Dataset(args.nfkb_path, args.ktr_path, data_path, remove_nans=True)
    
    scaler = StandardScaler()
    
    nfkbktr_dataset.data[:, :, 0] = scaler.fit_transform(nfkbktr_dataset.data[:, :, 0])
    nfkbktr_dataset.data[:, :, 1] = scaler.fit_transform(nfkbktr_dataset.data[:, :, 1])
    
    nfkbktr_splits = []
    for i in range(len(nfkbktr_dataset.labels) - 1):
        if nfkbktr_dataset.labels[i] != nfkbktr_dataset.labels[i + 1]:
            nfkbktr_splits.append(i + 1)
            
    if args.nfkb_permute:
        nfkbktr_dataset.data[:, :, 0] = np.random.permutation(nfkbktr_dataset.data[:, :, 0])
    if args.ktr_permute:
        nfkbktr_dataset.data[:, :, 1] = np.random.permutation(nfkbktr_dataset.data[:, :, 1])
        
    cv_data, cv_labels = shuffle_split(nfkbktr_dataset.data, nfkbktr_dataset.labels, nfkbktr_splits, 5)
    class_counter = collections.Counter(nfkbktr_dataset.labels)
    class_weights = class_weight.compute_class_weight('balanced', 
                                                      classes=np.unique(nfkbktr_dataset.labels), 
                                                      y=nfkbktr_dataset.labels)
    
    class_weights = torch.Tensor(class_weights).to(device)
    loss = nn.CrossEntropyLoss(weight=class_weights)
    
    best_evaluations, validation_loaders = [], []

    for idx in range(len(cv_data)):
        print(f'----------------------------------------{idx}-------------------------------------------')

        model = LSTMClassifier(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            output_size=args.output_size,
            num_layers=args.num_layers,
            linear_hidden_dim=args.linear_hidden_dim
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        train_data, val_data = np.concatenate([cv_data[i] for i in range(len(cv_data)) if i != idx]), cv_data[idx]
        train_labels, val_labels = np.concatenate([cv_labels[i] for i in range(len(cv_labels)) if i != idx]), cv_labels[idx]
        train, val = FoldDataset(train_data, train_labels), FoldDataset(val_data, val_labels)

        trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=args.batch_size)
        valloader = torch.utils.data.DataLoader(val, shuffle=True, batch_size=args.batch_size)

        validation_loaders.append(val)

        avg_train_loss, avg_val_loss = [], []
        top_acc = 0
        lowest_val_loss = np.inf
        best_preds = 0

        for e in range(args.num_epochs):
            train_loss, val_loss = [], []
            for x, y in trainloader:
                x, y = x.float().to(device), y.long().to(device)
                pred = model(x)
                l = loss(pred, y)

                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                train_loss.append(l.detach().cpu().numpy())

            acc_list = []
            for x, y in valloader:
                x, y = x.float().to(device), y.long().to(device)
                pred = model(x)
                l = loss(pred, y)

                val_loss.append(l.detach().cpu().numpy())

                # calculate accuracy
                acc = sum(torch.argmax(F.softmax(pred, dim=1), dim=1) == y) / len(y)
                acc = acc.detach().cpu().numpy()
                acc_list.append(acc)

            if np.mean(acc_list) > top_acc:
                top_acc = np.mean(acc_list)

            if np.mean(val_loss) < lowest_val_loss:
                lowest_val_loss = np.mean(val_loss)
                torch.save({
                    'weights': model.state_dict(),
                    'optim': optimizer.state_dict()
                }, args.save_path + str(idx) + '.pth')

            avg_train_loss.append(np.mean(train_loss))
            avg_val_loss.append(np.mean(val_loss))

        best_evaluations.append(best_preds)