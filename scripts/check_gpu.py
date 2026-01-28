import torch
print(f"CUDA available: {torch.cuda.is_available()}")
try:
    import torch_directml
    print(f"DirectML available: {torch_directml.is_available()}")
except ImportError:
    print("DirectML not installed")
