import torch
import torch.nn as nn
import torch.nn.functional as F

class ParetoSetModel_Transformer(torch.nn.Module):
    def __init__(self, n_dim, n_obj, bound, last_activation='relu'):
        super(ParetoSetModel_Transformer, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.last_activation = last_activation

        self.hidden_dim = 256
        if self.n_obj == 2:
                self.embedding_layer1 =  nn.Sequential(nn.Linear(1, self.hidden_dim),nn.ReLU(inplace=True))
                self.embedding_layer2 =  nn.Sequential(nn.Linear(1, self.hidden_dim),nn.ReLU(inplace=True))
        else:
                self.embedding_layer1 =  nn.Sequential(nn.Linear(1, self.hidden_dim),nn.ReLU(inplace=True))
                self.embedding_layer2 =  nn.Sequential(nn.Linear(1, self.hidden_dim),nn.ReLU(inplace=True))
                self.embedding_layer3 =  nn.Sequential(nn.Linear(1, self.hidden_dim),nn.ReLU(inplace=True))

        self.output_layer =  nn.Linear(self.hidden_dim, self.n_dim)
        self.bound = bound
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=1)
        self.ffn1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ffn2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        
    def forward(self, ray):
        if ray.shape[1] == 2:
            x = torch.stack((self.embedding_layer1(ray[:,0].unsqueeze(1)),self.embedding_layer2(ray[:,1].unsqueeze(1))))
        else:
            x = torch.stack((self.embedding_layer1(ray[:,0].unsqueeze(1)),self.embedding_layer2(ray[:,1].unsqueeze(1)),self.embedding_layer3(ray[:,2].unsqueeze(1))))
        x_ = x
        
        x,_ = self.attention(x,x,x)
        x = x + x_
        x_ = x
        x = self.ffn1(x)
        x = F.relu(x)
        x = self.ffn2(x)
        x = x + x_
        x = self.output_layer(x)
        # if self.last_activation == 'relu':
        #     x = F.relu(x)
        # elif self.last_activation == 'sigmoid':
        #     x = F.sigmoid(x)
        # elif self.last_activation == 'softmax':
        #     x = F.softmax(x)    
        # else:
        #     x = x
                
        x = torch.mean(x, dim=0)
        
        if self.bound == 0:
            x = torch.sigmoid(x)
        else:
            x = -self.bound + torch.sigmoid(x) * 2 * self.bound 
            
        return x.to(torch.float64)
    
    

class ParetoSetModel_MLP(torch.nn.Module):
    def __init__(self, n_dim, n_obj, bound):
        super(ParetoSetModel_MLP, self).__init__()
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