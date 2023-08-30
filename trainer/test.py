import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np

# Define a neural network with an LSTM layer
class FileClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(FileClassifier, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)  # Replaced the embedding layer with a linear layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden=None):
        x = self.linear(x)
        
        batch_size, seq_length, _ = x.size()
        
        if hidden is None:
            # Initialize the hidden state
            hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device),
                      torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device))
        
        # Initialize the output tensor
        outputs = torch.zeros(batch_size, seq_length, self.num_classes).to(x.device)
        
        for t in range(seq_length):
            x_t, hidden = self.lstm(x[:, t, :].unsqueeze(1), hidden)  # Process one timestep at a time
            x_t = self.fc(x_t)
            outputs[:, t, :] = self.softmax(x_t)
        
        return outputs, hidden

# Define a basic dataset class to load and preprocess the data
class FileDataset(data.Dataset):
    def __init__(self, file_bytes_list, labels):
        self.file_bytes_list = file_bytes_list
        self.labels = labels

    def __len__(self):
        return len(self.file_bytes_list)

    def __getitem__(self, index):
        file_bytes = self.file_bytes_list[index]
        label = self.labels[index]
        return file_bytes, label

def split_tensor_into_chunks(tensor, chunk_size):
    batch_size, seq_length = tensor.shape
    num_chunks = (seq_length + chunk_size - 1) // chunk_size

    # Calculate the padding size for the last chunk
    padding_size = chunk_size - (seq_length % chunk_size)
    
    # Pad the tensor if needed
    if padding_size > 0:
        padding = torch.zeros(batch_size, padding_size, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=1)
    
    # Reshape the tensor into chunks
    tensor = tensor.view(batch_size, num_chunks, chunk_size)
    
    return tensor

# Dummy data
file_bytes_list = [np.random.randint(0, 256, size=(466)) for _ in range(10)]  # Replace with actual file bytes
labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Replace with corresponding labels

# Hyperparameters
input_size = 256  # Assuming bytes are in the range 0-255
hidden_size = 128
num_layers = 4
num_classes = len(set(labels))
batch_size = 5
learning_rate = 0.01
num_epochs = 1500

# Create instances of the dataset and data loader
dataset = FileDataset(file_bytes_list, labels)
data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model and loss function
model = FileClassifier(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for file_bytes, label in data_loader:
        optimizer.zero_grad()
        file_bytes = torch.tensor(file_bytes, dtype=torch.long).float()
        bytes_ = split_tensor_into_chunks(file_bytes, input_size)
        
        batch_size = file_bytes.size(0)
        
        # Initialize hidden state for each batch and each layer
        hidden = (torch.zeros(num_layers, batch_size, hidden_size),
                  torch.zeros(num_layers, batch_size, hidden_size))
        
        # Process each chunk
        for byte_chunk in bytes_:
            # Move the hidden state to the same device as the input data
            hidden = (hidden[0].to(file_bytes.device), hidden[1].to(file_bytes.device))
            
            outputs, hidden = model(byte_chunk, hidden)
            
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "file_classifier_model.pth")
model.eval()

sample_input = torch.randn(batch_size, 1, input_size)
hidden_state = torch.randn(num_layers, batch_size, hidden_size)
cell_state = torch.randn(num_layers, batch_size, hidden_size)
hidden = (hidden_state, cell_state)

# Export the model to ONNX format
onnx_path = "LSTM.onnx"
torch.onnx.export(
    model,                   # Model instance
    (sample_input, hidden),  # Sample input, adjust if needed
    onnx_path,               # Path to save the ONNX model
    verbose=False,            # Print details while exporting
    input_names=["input", "hidden"],          # Input names in ONNX graph
    output_names=["output", "output_hidden"], # Output names in ONNX graph
    dynamic_axes={
        "input": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"},
        "hidden": {0: "num_layers", 1: "batch_size", 2: "hidden_size"},
        "output_hidden": {0: "num_layers", 1: "batch_size", 2: "hidden_size"}
    }  # Dynamic axes for variable length sequences
)

print(f"Model exported to {onnx_path}")