import torch
import torch.nn as nn
from torch.nn import functional as F

class RL(nn.Module):#input = 16009
    def __init__(self, n_inputs, kernel_size, stride, conv_type, pool_size):
        super(RL, self).__init__()

        self.conv_1 = ConvLayer_RL(n_inputs, 32, kernel_size, stride, conv_type,pool_size) # after 4001
        self.conv_2 = ConvLayer_RL(32,4, kernel_size, stride, conv_type,pool_size) # after 999
        self.linear_1= linear_RL(4*999,1024)
        self.linear_2= linear_RL(1024,128)
        self.linear_3= linear_RL(128,32)
        self.linear_4= (nn.Linear(32,1))
        self.output_layer= nn.Sigmoid()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = x.view(-1,4*999)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        x = self.linear_4(x)

        RL_alpha = self.output_layer(x)

        return RL_alpha

class ConvLayer_RL(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, conv_type,pool_size):
        super(ConvLayer_RL, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv_type = conv_type

        # How many channels should be normalised as one group if GroupNorm is activated
        # WARNING: Number of channels has to be divisible by this number!
        NORM_CHANNELS = 8

        self.filter = nn.Conv1d(n_inputs, n_outputs, self.kernel_size, stride)

        if conv_type == "gn":
            assert(n_outputs % NORM_CHANNELS == 0)
            self.norm = nn.GroupNorm(n_outputs // NORM_CHANNELS, n_outputs)
        elif conv_type == "bn":
            self.norm = nn.BatchNorm1d(n_outputs, momentum=0.01)

        self.pool=nn.MaxPool1d(pool_size)
        # Add you own types of variations here!

    def forward(self, x):
        # Apply the convolution
        if self.conv_type == "gn" or self.conv_type == "bn":
            out = F.relu(self.norm((self.filter(x))))
        else: # Add your own variations here with elifs conditioned on "conv_type" parameter!
            assert(self.conv_type == "normal")
            out = F.leaky_relu(self.filter(x))
        
        out = self.pool(out)
        return out

class linear_RL(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(linear_RL, self).__init__()
        self.fc1 = nn.Linear(n_inputs,n_outputs)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        return out