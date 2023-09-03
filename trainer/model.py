import torch
import torch.nn as nn

class MalwareModel(nn.Module):
    def __init__(self, max_byte_size, embedding_dim, types=2):
        super(MalwareModel, self).__init__()
        
        self.embedding = nn.Embedding(max_byte_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, embedding_dim)
        self.fc4 = nn.Linear(embedding_dim, types)
        
    def forward(self, x):
        x = self.embedding(x)
        #x = x[:, -1, :]
        #x = torch.mean(x, dim=1)  # You might need to apply some pooling here
        x = torch.mean(x, dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x
