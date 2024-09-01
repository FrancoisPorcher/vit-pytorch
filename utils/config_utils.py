# utils/config_utils.py

import json
import torch

def save_config(config, filename):
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {filename}")


def get_device():
    """
    Returns the appropriate torch device object based on the available hardware.
    
    The function checks for the availability of devices in the following order:
    1. GPU (CUDA)
    2. Metal Performance Shaders (MPS) for Apple devices
    3. Intel GPU (XPU) for Intel-based systems
    4. CPU (default if no other device is available)
    
    Returns:
        torch.device: The torch device object corresponding to the available hardware.
    """
    
    # Check for CUDA (GPU) availability
    if torch.cuda.is_available():
        return torch.device('cuda')

    # Check for MPS (Metal Performance Shaders) availability for Apple devices
    if torch.backends.mps.is_available():
        return torch.device('mps')

    # Check for XPU (Intel GPU) availability
    if torch.xpu.is_available():
        return torch.device('xpu')

    # Default to CPU if no other device is available
    return torch.device('cpu')