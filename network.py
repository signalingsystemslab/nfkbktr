import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,num_layers, linear_hidden_dim):
        super().__init__()
        
        self.lstm = nn.LSTM(batch_first=True, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.linear1 = nn.Linear(in_features=hidden_size * 95, out_features=linear_hidden_dim)
        self.linear2 = nn.Linear(in_features=linear_hidden_dim, out_features=output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.flatten(start_dim=1)
        x = self.linear2(self.relu(self.linear1(x)))
        return x