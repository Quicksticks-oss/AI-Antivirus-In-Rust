import torch

dummy_input = torch.randint(0, 255, size=(1,12))  # Example input with the same shape as your actual input
print(dummy_input.shape)