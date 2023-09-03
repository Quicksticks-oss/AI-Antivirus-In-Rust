import torch
import torch.nn as nn
import torch.optim as optim
import json, random
from model import MalwareModel
from tqdm import tqdm
import numpy as np

def load_dataset():
    with open('mdbsl.json', 'rb') as f:
        return json.load(f)

def get_batch(dataset, types, mIDX, sIDX):
    index = random.choice(['malware', 'safe'])
    if index == 'malware':
        mIDX += 1
        if mIDX >= len(dataset['malware']):
            mIDX = 0
        ix = mIDX
    else:
        sIDX += 1
        if sIDX >= len(dataset['safe']):
            sIDX = 0
        ix = sIDX
    batch_data = dataset[index][ix][-1]
    y_output = torch.tensor([0.0, 1.0]) if index == 'malware' else torch.tensor([1.0, 0.0])
    return batch_data, y_output.unsqueeze(0), mIDX, sIDX

def split_tensor(input_tensor, chunk_size):
    if input_tensor.size(0) > chunk_size:
        return torch.chunk(input_tensor, chunks=input_tensor.size(0) // chunk_size, dim=0)
    else:
        return [input_tensor.unsqueeze(0)]

def preprocess(dataset):
    new_m_db = {'malware': [], 'safe': []}

    for category in ['malware', 'safe']:
        for item, type_, tensor_x in dataset[category]:
            tensor_chunks = split_tensor(tensor_x.to(torch.int32), 1048576)
            new_m_db[category].extend([[item, type_, chunk] for chunk in tensor_chunks])

    return new_m_db

print('Loading...')
dataset = load_dataset()
types = list(set(m[1] for m in dataset['malware']))
types.append('Safe.File')
print(types)
print(f'Malware: {len(dataset["malware"])}, Safe: {len(dataset["safe"])}, Types: {len(types)}')

print('Preprocessing....')
dataset = preprocess(dataset)
print(len(dataset['malware']) + len(dataset['safe']))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Instantiate the model
max_byte_size = 256  # Replace with the actual vocabulary size
embedding_dim = 128  # Replace with the desired embedding dimension
model = MalwareModel(max_byte_size, embedding_dim, 2)  # len(types))
model = model.to(device)

# Define hyperparameters
learning_rate = 0.01
num_epochs = 64000

losses = []

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

td = tqdm(range(0, num_epochs + (len(dataset['malware']) + len(dataset['safe']))), dynamic_ncols=True)
# Training loop
mIDX, sIDX = 0, 0  # Initialize indices
for epoch in td:
    x, y, mIDX, sIDX = get_batch(dataset, types, mIDX, sIDX)
    y = y.to(device)
    x = x.unsqueeze(0).to(device)
    if x.shape[0] > 0 and x.shape[1] > 0:
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Print average loss every 100 iterations
    if epoch % 100 == 0:
        avg_loss = np.mean(losses[-100:])
        td.set_description(f'loss: {avg_loss:.4f}, last loss: {losses[-1]:.4f}')

    # Add any other monitoring or logging as needed

# Add code for saving the trained model if needed
print("Training finished!")

model.eval()  # Set the model to evaluation mode
model.to(torch.device('cpu'))

#torch.save(model.state_dict(), f"UnS.pt")

input_data = torch.LongTensor(1, 128).random_(0, max_byte_size).to(torch.int32)

# Export the embedding layer to ONNX format
torch.onnx.export(
    model,          # Model to export
    input_data,               # Sample input data
    f"UnS.onnx",   # File name to save the ONNX model
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
input_data = torch.tensor(data, dtype=torch.int32)
input_data = input_data.unsqueeze(0)
print(input_data.shape)

out = model(input_data)
print(out)

end_t = time.time()
print(end_t-start_t)

# Python: 0.0008323
# Rust:   0.00008449
