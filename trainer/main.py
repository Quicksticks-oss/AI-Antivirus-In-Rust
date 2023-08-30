import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.onnx
import json, random
from model import SimpleCNN

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
model = SimpleCNN()

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
        batch_size = x.size(0)
        channels = 3  # Assuming RGB images
        height = 32
        width = 32
        reshaped_input = x.view(batch_size, channels, height, width)
        
        optimizer.zero_grad()
        outputs = model(reshaped_input.float())
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
sample_input = torch.randint(0, max_byte_size, (1, 100))

traced_model = torch.jit.trace(model, sample_input)
torch.onnx.export(traced_model, sample_input, "MalwareModel.onnx", verbose=False, dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'}})
print("Model exported to ONNX successfully.")
