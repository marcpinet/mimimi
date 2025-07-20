import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ComplexConv2d, self).__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        real, imag = x
        
        real_out = self.conv_real(real) - self.conv_imag(imag)
        imag_out = self.conv_real(imag) + self.conv_imag(real)
        
        return real_out, imag_out


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(ComplexBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = nn.Parameter(torch.ones(num_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(num_features, dtype=torch.complex64))
        
        self.register_buffer('running_mean', torch.zeros(num_features, dtype=torch.complex64))
        self.register_buffer('running_var', torch.ones(num_features, dtype=torch.float32))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
    def forward(self, x):
        real, imag = x
        input_complex = torch.complex(real, imag)
        
        if self.training:
            batch_mean = torch.mean(input_complex, dim=[0, 2, 3], keepdim=False)
            centered = input_complex - batch_mean.view(1, -1, 1, 1)
            batch_var = torch.mean(torch.abs(centered)**2, dim=[0, 2, 3], keepdim=False)
            
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var
        
        normalized = (input_complex - batch_mean.view(1, -1, 1, 1)) / torch.sqrt(batch_var.view(1, -1, 1, 1) + self.eps)
        
        weight_complex = self.weight.view(1, -1, 1, 1) * torch.exp(1j * torch.zeros_like(self.weight.view(1, -1, 1, 1)))
        output = weight_complex * normalized + self.bias.view(1, -1, 1, 1)
        
        return torch.real(output), torch.imag(output)


class ComplexPReLU(nn.Module):
    def __init__(self, num_parameters=1):
        super(ComplexPReLU, self).__init__()
        self.prelu_real = nn.PReLU(num_parameters)
        self.prelu_imag = nn.PReLU(num_parameters)
        
    def forward(self, x):
        real, imag = x
        real_out = self.prelu_real(real)
        imag_out = self.prelu_imag(imag)
        return real_out, imag_out


class ComplexConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ComplexConvBlock, self).__init__()
        self.conv = ComplexConv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = ComplexBatchNorm2d(out_channels)
        self.activation = ComplexPReLU(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


def complex_magnitude(x):
    real, imag = x
    return real**2 + imag**2


def complex_to_magnitude_phase(x):
    real, imag = x
    magnitude = real**2 + imag**2
    phase = torch.atan2(imag, real)
    return magnitude, phase