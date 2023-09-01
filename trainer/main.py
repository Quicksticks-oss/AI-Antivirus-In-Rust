import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.onnx
import json, random
from model import MalwareModel
import time

def load_dataset():
    with open('mdb.json', 'rb') as f:
        return json.load(f)

def get_batch(dataset):
    index = random.choice(['malware', 'safe'])
    ix = random.randint(0, len(dataset[index])-1)
    batch_data = dataset[index][ix][-1]
    y_output = torch.tensor([0.0, 1.0]) if index == 'malware' else torch.tensor([1.0, 0.0])
    return torch.tensor([batch_data]).to(torch.int32), y_output.unsqueeze(0)

def split_tensor(input_tensor, chunk_size):
    input_tensor = input_tensor.squeeze(0)
    if input_tensor.shape[0] > chunk_size:
        return torch.chunk(input_tensor, chunks=input_tensor.size(0) // chunk_size, dim=0)
    else:
        return [input_tensor]


dataset = load_dataset()

device = torch.device('cuda')
print(f'Device: {device}')

## Instantiate the model
max_byte_size = 256  # Replace with the actual vocabulary size
embedding_dim = 64  # Replace with the desired embedding dimension
model = MalwareModel(max_byte_size, embedding_dim)
model = model.to(device)

# Define hyperparameters
learning_rate = 0.0001
num_epochs = 200

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i in range(16):
        x, y = get_batch(dataset)
        y = y.to(device)
        tensor_x = split_tensor(x, 1000000)
        for _ in range(len(tensor_x)-1):
            tx = tensor_x[_].unsqueeze(0).to(device)
            if tx.shape[0] > 0 and tx.shape[1] > 0:
                optimizer.zero_grad()
                outputs = model(tx)
                loss = criterion(outputs, y.view(-1))
                loss.backward()
                optimizer.step()
    
    if loss!= None:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    else:
        print(loss)

print("Training finished!")

model.eval()  # Set the model to evaluation mode
model.to(torch.device('cpu'))

torch.save(model.state_dict(), 'MalwareModelTiny.pt')

input_data = torch.LongTensor(1, 128).random_(0, max_byte_size)

# Export the embedding layer to ONNX format
torch.onnx.export(
    model,          # Model to export
    input_data,               # Sample input data
    "MalwareModelTiny.onnx",   # File name to save the ONNX model
    verbose=False,
    input_names=["input"],    # Names for the input tensors
    output_names=["output"],  # Names for the output tensors
    dynamic_axes={
        "input": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"}
    }
)

print('Running inference.')

start_t = time.time()

data = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
input_data = torch.tensor(data)
input_data = input_data.unsqueeze(0)
print(input_data.shape)

out = model(input_data)
print(out)

end_t = time.time()
print(end_t-start_t)

# Python: 0.0008323
# Rust:   0.00008449