import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, device='cpu'):
        super(BaseModel, self).__init__()
        self.device = device

    def clear_gradients(self):
        for _, layer in self.named_children():
            for _, param in layer.named_parameters():
                del param.grad
        torch.cuda.empty_cache()