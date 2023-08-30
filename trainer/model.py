import torch
import torch.nn as nn
import random
import json

class CustomModel(nn.Module):
    def __init__(self, max_byte_size, embedding_dim):
        super(CustomModel, self).__init__()
        
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

def load_dataset():
    with open('db.json', 'rb') as f:
        return json.load(f)

def get_batch(dataset, batch_size):
    ix = 0#random.randint(0, len(dataset['malware'])-batch_size)
    print(ix)
    batch_data = dataset['malware'][ix:ix+batch_size]
    data_at_index_2 = [item[2] for item in batch_data]
    print(len(data_at_index_2))
    #return mw

dataset = load_dataset()

batch_x = get_batch(dataset, 6)
print(batch_x.shape)

#bytes_ = dataset['malware'][0][2]
#
## Instantiate the model
#vocab_size = 256  # Replace with the actual vocabulary size
#embedding_dim = 256  # Replace with the desired embedding dimension
#model = CustomModel(vocab_size, embedding_dim)
#
## Generate a random input
#input_data = torch.tensor([bytes_])
#
### Get the model's prediction
#output = model(input_data)
#print(output)
#print(output.shape)  # This should print torch.Size([1, 2])
#