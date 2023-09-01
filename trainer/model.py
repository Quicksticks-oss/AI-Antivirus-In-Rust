import torch
import torch.nn as nn

class MalwareModel(nn.Module):
    def __init__(self, max_byte_size, embedding_dim):
        super(MalwareModel, self).__init__()
        
        self.embedding = nn.Embedding(max_byte_size, embedding_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Adaptive pooling to handle varying sequence lengths
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # Adjust the dimensions for AdaptiveAvgPool1d
        x = self.pooling(x).squeeze()
        
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = nn.functional.relu(self.fc3(x))
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x