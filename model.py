"""
A simple FC Pareto Set model.
"""

import torch
import torch.nn as nn

# 67328

class ParetoSetModel(torch.nn.Module):
    def __init__(self, n_dim, n_obj, bound):
        super(ParetoSetModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj
       
        self.fc1 = nn.Linear(self.n_obj, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.n_dim)
        self.bound = bound

        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        torch.nn.init.xavier_uniform(self.fc3.weight)
        
    def forward(self, pref):
        x = torch.relu(self.fc1(pref))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        if self.bound == 0:
            x = torch.sigmoid(x)
        else:
            x = -self.bound + torch.sigmoid(x)*2*self.bound 

        return x.to(torch.float64)