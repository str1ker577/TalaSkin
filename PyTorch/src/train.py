import torch
print(torch.__version__)             # should show 2.x.x+cu121
print(torch.cuda.is_available())     # should show True
print(torch.cuda.get_device_name(0)) # should show NVIDIA GeForce RTX 3060
