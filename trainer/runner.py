import torch, time

start_t = time.time()

data = [1, 2, 3, 4, 5, 6]
input_data = torch.tensor(data)
input_data = input_data.unsqueeze(0)
print(input_data.shape)

end_t = time.time()
