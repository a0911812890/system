import torch
import torch.nn as nn


class customLoss(nn.Module):
    def __init__(self):
        # --------------------------------------------
        # Initialization
        # --------------------------------------------
        super(customLoss, self).__init__()

    def forward(self, outputs, targets,alpha):
        # --------------------------------------------
        # Define forward pass
        # --------------------------------------------
        # Transform targets to one-hot vector
        criterion = nn.MSELoss()
        loss = torch.mean(torch.pow((outputs-targets),2),2)
        # print("origin:",criterion(outputs,targets))
        # print("MY:",loss)
        loss = loss*alpha
        loss=torch.mean(loss)
        return loss


class RL_customLoss(nn.Module):
    def __init__(self):
        # --------------------------------------------
        # Initialization
        # --------------------------------------------
        super(RL_customLoss, self).__init__()

    def forward(self, RL_alpha, label):
        # loss = torch.abs((RL_alpha-label))
        loss = torch.pow((RL_alpha-label),2)
        loss=torch.mean(loss)
        return loss