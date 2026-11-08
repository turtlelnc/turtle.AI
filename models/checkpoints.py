"""Load the model and exact memory policy saved by association experiments."""
import torch
from .resonant import Config, ResonantLM
from .streaming_memory import MemoryConfig


def load_model_and_memory(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = ResonantLM(Config(**checkpoint['config']))
    model.load_state_dict(checkpoint['model'])
    policy = dict(checkpoint['policy'])
    dtypes = {'torch.float32': torch.float32, 'torch.float16': torch.float16,
              'torch.bfloat16': torch.bfloat16}
    policy['working_dtype'] = dtypes[policy['working_dtype']]
    model.eval()
    return model, MemoryConfig(**policy)
