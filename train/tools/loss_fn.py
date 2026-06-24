import torch
import torch.nn as nn
from train.tools.transforms3d import axis_angle_to_quaternion, quaternion_multiply

class LogMSELoss(nn.Module):
    def __init__(self, epsilon, contraction=1):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.contraction = contraction
        self.epsilon = epsilon

    def forward(self, pred, target):
        element_wise_mse = self.mse(pred, target)
        sample_wise_mse = torch.mean(element_wise_mse.view(pred.size(0), -1), dim=1)
        return torch.log(self.contraction * sample_wise_mse + self.epsilon).mean()
    

def geodesic_loss(rv1, rv2):
    """
    Compute the geodesic loss between two rotation vectors (axis-angle representation).
    rv: (batch, 3)
    """
    q1 = axis_angle_to_quaternion(rv1)  # (batch, 4)
    q2 = axis_angle_to_quaternion(rv2)  # (batch, 4)
    dot_product = torch.sum(q1 * q2, dim=-1)  # (batch,)
    angle = 2 * torch.acos(torch.abs(dot_product).clamp(-1.0 + 1e-7, 1.0 - 1e-7))  # (batch,)
    return angle # rad