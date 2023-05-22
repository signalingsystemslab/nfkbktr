import torch
import pandas as pd
import numpy as np


class DosageDataset2D(torch.utils.data.Dataset):
    '''
    Dataset for 2D (KTR + NFkB) LSTM Training
    '''
    def __init__(self, filepath, path, remove_nans=True):
        self.data = []
        self.labels = []
        
        for i, j in path:
            a = np.array(pd.read_csv(filepath + i))
            b = np.array(pd.read_csv(filepath + j))
            assert len(a) == len(b)
            
            if remove_nans:
                b = b[~np.isnan(a).any(axis=1)]
                a = a[~np.isnan(a).any(axis=1)]
                a = a[~np.isnan(b).any(axis=1)]
                b = b[~np.isnan(b).any(axis=1)]
                
            arr = np.dstack([a, b])
            
            label, _ = int(i.split('_')[-1][0]) - 1, int(j.split('_')[-1][0]) - 1
            assert label == _ # make sure they are from the same dosage aka path list was formatted correctly
            
            labels = np.repeat(label, len(arr))
            
            self.data.append(arr)
            self.labels.append(labels)
            
        self.data = np.concatenate(self.data)
        self.labels = np.concatenate(self.labels)
        
        assert len(self.data) == len(self.labels)
        
    def __getitem__(self, i):
        return self.data[i], self.labels[i]
    
    def __len__(self):
        return len(self.data)
    
    
class DosageDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, path, two_dim=False, remove_nans=True):
        
        self.data, self.labels = [], []
        for file in path:
            df = pd.read_csv(filepath + file)
            
            if remove_nans:
                df = df.dropna()
            
            data = np.array(df)
            label = int(file.split('_')[-1][0]) - 1
            labels = np.repeat(label, len(data))
            
            self.data.append(data)
            self.labels.append(labels)
            
        self.data = np.concatenate(self.data)
        self.labels = np.concatenate(self.labels)
        
    def __getitem__(self, i):
        return self.data[i], self.labels[i]
    
    def __len__(self):
        return len(self.data)
    

class NFkBKTR_Dataset(torch.utils.data.Dataset):
    '''
    Create a torch.data.Dataset class for 2 dimension time series sequences of NFkB and KTR 
    Inputs:
        nfkb_path: [...CpG_R1, LPS_R1, ...]
        ktr_path: [...CpG_R1, LPS_R1, ...]
        i.e indices of nfkb and ktr path should have corresponding ligands
    '''
    def __init__(self, nfkb_path, ktr_path, data_path, remove_nans=False):
        assert len(nfkb_path) == len(ktr_path)
        self.data, self.labels = [], []
        
        for label, (i, j) in enumerate(zip(nfkb_path, ktr_path)):
            
            if len(i) == 2: # replicas
                assert len(i) == len(j)
                r1_nfkb, r2_nfkb = np.array(pd.read_csv(data_path + i[0])), np.array(pd.read_csv(data_path + i[1]))
                r1_ktr, r2_ktr = np.array(pd.read_csv(data_path + j[0])), np.array(pd.read_csv(data_path + j[1]))
                nfkb_array = np.concatenate([r1_nfkb, r2_nfkb], axis=0)
                ktr_array = np.concatenate([r1_ktr, r2_ktr], axis=0)
            else:
                nfkb_array = np.array(pd.read_csv(data_path + i))
                ktr_array = np.array(pd.read_csv(data_path + j))
            
            assert nfkb_array.shape == ktr_array.shape # array shapes should be the same
            if remove_nans:
                # remove the union of nan rows for both arrays
                nfkb_array, ktr_array = nfkb_array[~np.isnan(nfkb_array).any(axis=1)], ktr_array[~np.isnan(nfkb_array).any(axis=1)]
                nfkb_array, ktr_array = nfkb_array[~np.isnan(ktr_array).any(axis=1)], ktr_array[~np.isnan(ktr_array).any(axis=1)]
                
            nfkb_array = nfkb_array.reshape(nfkb_array.shape[0], nfkb_array.shape[1], 1)
            ktr_array = ktr_array.reshape(ktr_array.shape[0], ktr_array.shape[1], 1)
            
            data = np.concatenate([nfkb_array, ktr_array], axis=2) # [num_rows, num_timefeatures, 2]
            labels = np.repeat(label, len(nfkb_array))
            
            self.data.append(data)
            self.labels.append(labels)
            
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        assert len(self.data) == len(self.labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    
class OneDData(torch.utils.data.Dataset): 
    def __init__(self, nfkb_path, data_path, remove_nans=False):
            self.data, self.labels = [], []

            for label, i in enumerate(nfkb_path):
                if len(i) == 2: # replicas
                    r1_nfkb, r2_nfkb = np.array(pd.read_csv(data_path + i[0])), np.array(pd.read_csv(data_path + i[1]))
                    nfkb_array = np.concatenate([r1_nfkb, r2_nfkb], axis=0)
                else:
                    nfkb_array = np.array(pd.read_csv(data_path + i))

                if remove_nans:
                    # remove the union of nan rows for both arrays
                    nfkb_array = nfkb_array[~np.isnan(nfkb_array).any(axis=1)]
                    
                nfkb_array = nfkb_array.reshape(nfkb_array.shape[0], nfkb_array.shape[1], 1)
                labels = np.repeat(label, len(nfkb_array))

                self.data.append(nfkb_array)
                self.labels.append(labels)

            self.data = np.concatenate(self.data, axis=0)
            self.labels = np.concatenate(self.labels, axis=0)
            assert len(self.data) == len(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]