import torch
import torch.nn as nn
import json

class MalwareModel(nn.Module):
    def __init__(self, max_byte_size, embedding_dim, output_size):
        super(MalwareModel, self).__init__()
        
        self.embedding = nn.Embedding(max_byte_size, embedding_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Adaptive pooling to handle varying sequence lengths

        self.fc1 = nn.Linear(embedding_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # Adjust the dimensions for AdaptiveAvgPool1d
        x = self.pooling(x).squeeze()
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Instantiate the model
# Instantiate the model
input_size = 1  # Change this based on the size of your input (e.g., 1 for raw bytes)
num_classes = 10  # Change this to the number of classes in your classification task
output_size = 2
model = MalwareModel(256, 32, output_size)

# Print the model architecture
print(model)

with open('db.json', 'r') as f:
    data = json.load(f)

d = torch.tensor([data['malware'][1][-1]], dtype=torch.int32)
print(d.shape)

out = model(d)
print(out.shape)