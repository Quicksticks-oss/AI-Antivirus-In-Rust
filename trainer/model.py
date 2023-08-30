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

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 16 * 16, 128)  # Assuming input image size is 32x32
        self.fc2 = nn.Linear(128, 10)  # 10 classes for classification
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
