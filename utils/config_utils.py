# utils/config_utils.py

import json

def save_config(config, filename):
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {filename}")