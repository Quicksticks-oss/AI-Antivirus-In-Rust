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
        x = self.linear(x)  # Using linear layer instead of embedding
        x, (hn, cn) = self.lstm(x, hidden) if hidden is not None else self.lstm(x)
        x = self.fc(x)  # Using the last hidden state for classification
        x = self.softmax(x)
        return x, (hn, cn)

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

def split_tensor_into_chunks(tensor, chunk_size, dim):
    chunks = tensor.chunk(tensor.size(dim) // chunk_size, dim=dim)
    return chunks

# Dummy data
file_bytes_list = [np.random.randint(0, 256, size=(1024*4,)) for _ in range(10)]  # Replace with actual file bytes
labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Replace with corresponding labels

# Hyperparameters
input_size = 256  # Assuming bytes are in the range 0-255
hidden_size = 128
num_layers = 4
num_classes = len(set(labels))
batch_size = 128
learning_rate = 0.001
num_epochs = 150

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
        file_bytes = torch.tensor(file_bytes, dtype=torch.long).float()  # Convert to LongTensor for embedding
        bytes_ = split_tensor_into_chunks(file_bytes, input_size, 1)
        outputs, hid = model(bytes_[0])
        for b in range(1, len(bytes_)):
            outputs, hid = model(bytes_[b], hid)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "file_classifier_model.pth")
model.eval()
onnx_path = "file_classifier_model.onnx"

# Example input data (replace this with your actual data)
dummy_input = torch.tensor(np.random.randint(0, 256, size=(1, input_size)), dtype=torch.long)

# Initialize dummy hidden and cell states
dummy_hidden = torch.zeros(num_layers, 1, hidden_size)
dummy_cell = torch.zeros(num_layers, 1, hidden_size)

torch.onnx.export(
    model,
    (dummy_input, (dummy_hidden, dummy_cell)),  # Pass input and initial states as tuple
    onnx_path,
    verbose=False,
    input_names=["input", "hidden", "cell"],   # Names for input tensors
    output_names=["output", "final_hidden", "final_cell"],  # Names for output tensors and states
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
        "hidden": {1: "num_layers", 2: "batch_size"},  # Dynamic axes for states
        "cell": {1: "num_layers", 2: "batch_size"},
        "final_hidden": {1: "num_layers", 2: "batch_size"},  # Dynamic axes for final states
        "final_cell": {1: "num_layers", 2: "batch_size"}
    },
    opset_version=11  # Use the appropriate ONNX opset version
)

print(f"Model exported to {onnx_path}")