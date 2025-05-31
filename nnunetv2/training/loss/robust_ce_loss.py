import torch
from torch import nn, Tensor
import numpy as np


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()
        
class FocalLoss(nn.Module):
    #https://paperswithcode.com/method/focal-loss#:~:text=A%20Focal%20Loss%20function%20addresses,learning%20on%20hard%20misclassified%20examples.
    def __init__(self, gamma=2.0): 
        self.gamma = gamma
        super(FocalLoss, self).__init__()

    def forward(self, inp, target):
        logpt = torch.nn.functional.log_softmax(inp, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)

        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple, torch.Tensor)):
                alpha = torch.tensor(self.alpha, device=input.device)
                at = alpha.gather(0, target)
            else:
                at = torch.full_like(pt, self.alpha)
            logpt = logpt * at

        loss = -((1 - pt) ** self.gamma) * logpt

        return loss