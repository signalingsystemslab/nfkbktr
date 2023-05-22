import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def shuffle_split(data, labels, split_indices, num_chunks):
    '''
    Split the data/labels into n chunks for crossfold validation where each chunk has balanced representation of classes
    Data and labels should match index-wise
    '''
    # split by class
    class_data_splits = np.split(data, split_indices) 
    class_label_splits = np.split(labels, split_indices) 
    
    # split by number of folds
    data_folds, label_folds = [], []
    for data, label in zip(class_data_splits, class_label_splits):
        try:
            data_fold = np.split(data, num_chunks)
            label_fold = np.split(label, num_chunks)
        except: # np.split cannot find equal division
            assert len(data) == len(label)
            n = len(data) / num_chunks
            split_indices = [int(n * i) for i in range(1, num_chunks)]
            data_fold = np.split(data, split_indices)
            label_fold = np.split(label, split_indices)
            
        data_folds.append(data_fold)
        label_folds.append(label_fold)
        
    data_folds = np.array(data_folds) # shape: [num_classes, num_chunks], classes are separated by rows
    label_folds = np.array(label_folds)
    
    data_list, labels_list = [], []
    for i in range(num_chunks):
        # import pdb; pdb.set_trace()
        data = np.concatenate([x for x in data_folds[:, i]])
        labels = np.concatenate([x for x in label_folds[:, i]])
        data_list.append(data)
        labels_list.append(labels)
        
    return data_list, labels_list


def get_cv_evaluation_metrics(val_subset, weight_path, device):
    model = LSTMClassifier(
        input_size=1,
        hidden_size=95,
        output_size=3,
        num_layers=2,
        linear_hidden_dim=512
    ).to(device)
    
    weights = torch.load(weight_path)
    model.load_state_dict(weights['weights'])
    
    val_data = val_subset.__getitem__([i for i in range(0, val_subset.__len__())])[0]
    val_labels = val_subset.__getitem__([i for i in range(0, val_subset.__len__())])[1]
    
    val_pred = model(torch.Tensor(val_data).to(device))
    val_pred = torch.argmax(F.softmax(val_pred, dim=1), dim=1)
    val_pred = val_pred.detach().cpu().numpy()
    
    cr = sklearn.metrics.classification_report(val_labels, val_pred, target_names=['CpG', 'LPS', 'P3C4'], output_dict=True)
    cm = sklearn.metrics.confusion_matrix(val_labels, val_pred)
    cmd = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CpG', 'LPS', 'P3C4'])
    
    return cr, cm, cmd


class FoldDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        assert len(self.data) == len(self.labels)
        
    def __getitem__(self, x):
        return self.data[x], self.labels[x]
    
    def __len__(self):
        return len(self.data)