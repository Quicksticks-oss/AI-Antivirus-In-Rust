import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.onnx
import json, random
from model import MalwareModel
import time

def load_dataset():
    with open('db.json', 'rb') as f:
        return json.load(f)

def get_batch(dataset):
    index = random.choice(['malware', 'safe'])
    ix = random.randint(0, len(dataset[index])-1)
    batch_data = dataset[index][ix][-1]
    y_output = torch.tensor([1.0, 0.0]) if index == 'malware' else torch.tensor([0.0, 1.0])
    return torch.tensor([batch_data]), y_output.unsqueeze(0)

dataset = load_dataset()

## Instantiate the model
max_byte_size = 256  # Replace with the actual vocabulary size
embedding_dim = 256  # Replace with the desired embedding dimension
model = MalwareModel(max_byte_size, embedding_dim)

# Define hyperparameters
learning_rate = 0.001
num_epochs = 30

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    
    for i in range(10):
        x, y = get_batch(dataset)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training finished!")

# Save the trained model
torch.save(model.state_dict(), 'custom_model.pth')

print("Training finished!")
model.eval()  # Set the model to evaluation mode

torch.save(model.state_dict(), 'model.pt')

input_data = torch.LongTensor(1, 128).random_(0, max_byte_size)

# Export the embedding layer to ONNX format
torch.onnx.export(
    model,          # Model to export
    input_data,               # Sample input data
    "embedding_layer.onnx",   # File name to save the ONNX model
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