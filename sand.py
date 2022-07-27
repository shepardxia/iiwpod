import torch
from create_model import iiwpod
import numpy as np
import torch

model = iiwpod()
dummy = torch.zeros((1, 3, 128, 128))
print(model(dummy).shape)