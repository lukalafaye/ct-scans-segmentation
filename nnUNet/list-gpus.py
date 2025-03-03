import torch

print("CUDA Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

if torch.cuda.device_count() > 1:
    print("GPU 0:", torch.cuda.get_device_name(0))
    print("GPU 1:", torch.cuda.get_device_name(1))
else:
    print("Only one GPU is detected by PyTorch!")
