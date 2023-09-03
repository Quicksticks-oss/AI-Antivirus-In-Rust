import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.onnx
import json, random
from model import MalwareModel
from tqdm import tqdm
import numpy as np
import time
import sys

def load_dataset():
    with open('mdbsl.json', 'rb') as f:
        return json.load(f)

mIDX = 0
sIDX = 0

def get_batch(dataset, types):
    global mIDX, sIDX
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
    #ix = random.randint(0, len(dataset[index])-1)
    batch_data = dataset[index][ix][-1]
    my_list = [0] * len(types)
    if index == 'malware':
        type_ = dataset[index][ix][1]
    else:
        type_ = 'Safe.File'
    my_list[types.index(type_)] = 1.0
    #y_output = torch.tensor([0.0, 1.0]) if index == 'malware' else torch.tensor([1.0, 0.0])
    y_output = torch.tensor(my_list)
    #print(y_output, index)
    return batch_data, y_output.unsqueeze(0)

def split_tensor(input_tensor, chunk_size):
    input_tensor = input_tensor.squeeze(0)
    if input_tensor.shape[0] > chunk_size:
        return torch.chunk(input_tensor, chunks=input_tensor.size(0) // chunk_size, dim=0)
    else:
        return [input_tensor]

def preprocess(dataset):
    new_m_db = {'malware': [], 'safe': []}

    for f in dataset['malware']:
        item = f[0]
        type_ = f[1]
        tensor_x = split_tensor(torch.tensor(f[2]), 1048576)
        del(f[2])
        for t in tensor_x:
            new_m_db['malware'].append([item, type_, t.to(torch.int32)])
    for f in dataset['safe']:
        item = f[0]
        type_ = f[1]
        tensor_x = split_tensor(torch.tensor(f[2]), 1048576)
        del(f[2])
        for t in tensor_x:
            new_m_db['safe'].append([item, type_, t.to(torch.int32)])
    return new_m_db

print('Loading...')
dataset = load_dataset()
types = []
for m in dataset['malware']:
    types.append(m[1])
types = list(set(types))
types.append('Safe.File')
print(types)
print(f'Malware: {len(dataset["malware"])}, Safe: {len(dataset["safe"])}, Types: {len(types)}')

print('Preprocessing....')
dataset = preprocess(dataset)
print((len(dataset['malware'])+len(dataset['safe'])))

device = torch.device('cuda')
print(f'Device: {device}')

## Instantiate the model
max_byte_size = 256  # Replace with the actual vocabulary size
embedding_dim = 96  # Replace with the desired embedding dimension
model = MalwareModel(max_byte_size, embedding_dim, len(types))
model = model.to(device)

# Define hyperparameters
learning_rate = 0.001
num_epochs = 512000

losses = []

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

td = tqdm(range(0, num_epochs+(len(dataset['malware'])+len(dataset['safe']))), dynamic_ncols=True)
# Training loop
for epoch in td:
    x, y = get_batch(dataset, types)
    y = y.to(device)
    x = x.unsqueeze(0).to(device)
    if x.shape[0] > 0 and x.shape[1] > 0:
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    if loss != None and epoch % 512 == 0:
        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        td.set_description(f'loss: {np.mean(losses):.4f}, {losses[-1]}')
        print(outputs.argmax())
        print(y.argmax())
        print(mIDX, sIDX)

print("Training finished!")

model.eval()  # Set the model to evaluation mode
model.to(torch.device('cpu'))

torch.save(model.state_dict(), f"MalwareModelNew.pt")

input_data = torch.LongTensor(1, 128).random_(0, max_byte_size).to(torch.int32)

# Export the embedding layer to ONNX format
torch.onnx.export(
    model,          # Model to export
    input_data,               # Sample input data
    f"MalwareModelNew.onnx",   # File name to save the ONNX model
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
