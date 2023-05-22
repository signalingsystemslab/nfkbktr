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
    
    parser.add_argument('--ktr-nfkb-p3c4-path', type=str, default=None)
    parser.add_argument('--ktr-nfkb-cpg-path', type=str, default=None)
    parser.add_argument('--ktr-nfkb-tnf-path', type=str, default=None)
    parser.add_argument('--ktr-nfkb-lps-path', type=str, default=None)
    
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device('cuda:0')
    scaler = StandardScaler()
    
    if args.ktr_nfkb_cpg_path and args.ktr_nfkb_lps_path and args.ktr_nfkb_p3c4_path and args.ktr_nfkb_tnf_path:
        cpg_dosage_dataset = DosageDataset2D(args.data_path, args.ktr_nfkb_cpg_path)
        lps_dosage_dataset = DosageDataset2D(args.data_path, args.ktr_nfkb_lps_path)
        p3c4_dosage_dataset = DosageDataset2D(args.data_path, args.ktr_nfkb_p3c4_path)
        tnf_dosage_dataset = DosageDataset2D(args.data_path, args.ktr_nfkb_tnf_path)
    
    names = ['cpg', 'lps', 'p3c4', 'tnf']
    datasets = [cpg_dosage_dataset, ktr_lps_dosage_dataset, p3c4_dosage_dataset, tnf_dosage_dataset]
    for name, dataset in zip(names, datasets):
        dataset.data[:, :, 0] = scaler.fit_transform(dataset.data[:, :, 0])

        splits = []
        for i in range(len(dataset.labels) - 1):
            if dataset.labels[i] != dataset.labels[i + 1]:
                splits.append(i + 1)

        cv_data, cv_labels = shuffle_split(dataset.data, dataset.labels, splits, 5)
        class_counter = collections.Counter(dataset.labels)
        class_weights = class_weight.compute_class_weight('balanced', 
                                                          classes=np.unique(dataset.labels), 
                                                          y=dataset.labels)

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
                    }, name + '_' + args.save_path + str(idx) + '.pth')

                avg_train_loss.append(np.mean(train_loss))
                avg_val_loss.append(np.mean(val_loss))

            best_evaluations.append(best_preds)
        
        
        
if __name__ == '__main__':
    args = parse_args()
    main(args)