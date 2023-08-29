import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.onnx

class Model(nn.Module):
    def __init__(self, input_size:int, hidden_size:int=32, output_size:int=2):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x


# Create a toy dataset
X = torch.randn(1000, 10)  # 1000 samples with 10 features each
y = torch.randint(0, 2, (1000,))  # Binary labels (0 or 1)


# Create DataLoader for batching
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate the model
model = Model(input_size=10)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

print("Training finished!")
model.eval()  # Set the model to evaluation mode

torch.save(model.state_dict(), 'model.pt')
dummy_input = torch.randn(1, 10)  # Example input with the same shape as your actual input

torch.onnx.export(model, dummy_input, "model.onnx", verbose=False)
print("Model exported to ONNX successfully.")
