import torch

def resolve_device(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else torch.device(cfg.device)
    
    if cfg.precision == "auto":
        dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    else:
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[cfg.precision]
    
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU has no native BF16. Use auto / fp32, or explicitly try fp16.")
    
    if dtype != torch.float32 and device.type != "cuda":
        raise ValueError("Portable mixed precision is supported on CUDA only")
    
    return device, dtype

def amp_context(device, dtype):
    return torch.autocast("cuda", dtype=dtype) if device.type == "cuda" and dtype != torch.float32 else nullcontext()

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)