import torch
if torch.backends.mps.is_available():
    print("PyTorch can use the MPS (Apple Silicon GPU) backend!")
else:
    print("MPS not available. Running on CPU.")