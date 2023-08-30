import torch
import torch.nn as nn

class MalwareModel(nn.Module):
    def __init__(self, max_byte_size, embedding_dim):
        super(MalwareModel, self).__init__()
        
        self.embedding = nn.Embedding(max_byte_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
        
    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)  # You might need to apply some pooling here
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x
