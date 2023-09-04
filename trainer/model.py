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

class MalwareModelV2(nn.Module):
    def __init__(self, max_byte_size, embedding_dim, types=2):
        super(MalwareModel, self).__init__()

        self.embedding = nn.Embedding(max_byte_size, embedding_dim)
        
        # Convolutional layers for feature detection
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, types)
        
    def forward(self, x):
        x = self.embedding(x)
        
        # Permute to make the sequence dimension the last dimension
        x = x.permute(0, 2, 1)
        
        # Apply convolutional layers
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        
        # Global max pooling to reduce sequence length
        x = torch.max(x, dim=2).values
        
        # Fully connected layers for classification
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
