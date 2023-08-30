import torch
import torch.onnx
from model import MalwareModel

# Define your embedding layer
vocab_size = 1000
embedding_dim = 128
embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)

# Define sample input
batch_size = 1
sequence_length = 10
input_data = torch.LongTensor(batch_size, sequence_length).random_(0, vocab_size)

# Export the embedding layer to ONNX format
torch.onnx.export(
    embedding_layer,          # Model to export
    input_data,               # Sample input data
    "embedding_layer.onnx",   # File name to save the ONNX model
    verbose=True,
    input_names=["input"],    # Names for the input tensors
    output_names=["output"],  # Names for the output tensors
    dynamic_axes={
        "input": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"}
    }
)
