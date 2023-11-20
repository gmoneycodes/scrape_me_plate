import torch.nn as nn
import torch.nn.functional as F


def get_activation(name="silu", inplace=True):
    """
    This is insanely next level ugly but what can you do :D
    :param name: activations to use
    :param inplace: option to do inplace activations for SiLU ReLU and LeakyReLU
    :return: activation :D
    """
    supported_activations = ['silu', 'relu', 'leakyrelu', 'hswish', 'gelu']
    assert name in supported_activations, f'WRONG ACTIVATION BUDDY, pick {supported_activations}'
    if name is None:
        return None
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "leakyrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == 'hswish':
        module = HSwish()
    elif name == "gelu":
        module = nn.GELU()
    return module


class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


def get_norm(name, out_channels):
    supported_norms = ['batchnorm', 'layernorm']
    assert name in supported_norms, f"WRONG NORM BUDDY, pick {supported_norms}"
    if name is None:
        return None
    if name == "batchnorm":
        module = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
    elif name == "layernorm":
        module = nn.LayerNorm(out_channels)
    return module
