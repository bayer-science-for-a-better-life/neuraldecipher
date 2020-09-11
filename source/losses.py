import torch
import torch.nn as nn

# Symmetric and smooth loss functions wrt. the difference #


class LogCoshLoss(nn.Module):
    """
    Implements the Log-Cosh loss function with mean reduction as default:
    l = 1/N sum_{i=1}^N ln(cosh(d_i)), where d_i = y_{i, true} - y_{i, pred}
    """
    def __init__(self, epsilon:float = 1e-10, reduction: str ='mean'):
        super(LogCoshLoss, self).__init__()
        self.epsilon = epsilon
        assert reduction in ['mean', 'sum'], 'Please define <reduction> as "mean" or "sum".'
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        difference = y_pred - y_true
        if self.reduction == 'mean':
            return torch.mean(torch.log(torch.cosh(difference + self.epsilon)))
        elif self.reduction == 'sum':
            return torch.sum(torch.log(torch.cosh(difference + self.epsilon)))


class XSigmoidLoss(nn.Module):
    """
    Implements the XSigmoid loss function with mean reduction as default:
    l = 1/N sum_{i=1}^N d_i(2*sigmoid(d_i) -1), where d_i = y_{i, true} - y_{i, pred}
    """

    def __init__(self, reduction: str = 'mean'):
        super(XSigmoidLoss, self).__init__()
        assert reduction in ['mean', 'sum'], 'Please define <reduction> as "mean" or "sum".'
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        difference = y_pred - y_true
        if self.reduction == 'mean':
            return torch.mean(difference * (2*torch.sigmoid(difference) - 1))
        elif self.reduction == 'sum':
            return torch.sum(difference * (2*torch.sigmoid(difference) - 1))


class XTanhLoss(nn.Module):
    """
    Implements the XTanh loss function with mean reduction as default:
    l = 1/N sum_{i=1}^N d_i*tanh(d_i) , where d_i = y_{i, true} - y_{i, pred}
    """

    def __init__(self, reduction: str = 'mean'):
        super(XTanhLoss, self).__init__()
        assert reduction in ['mean', 'sum'], 'Please define <reduction> as "mean" or "sum".'
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        difference = y_pred - y_true
        if self.reduction == 'mean':
            return torch.mean(difference * torch.tanh(difference))
        elif self.reduction == 'sum':
            return torch.sum(difference * torch.tanh(difference))

# Similarity Loss #


class CosineSimLoss(nn.Module):
    """
    Implements the batch cosine similarity of two vectors u,v as:
    sim(u,v) = <u,v> / ||u|| * ||v|| , where <.> is the scalar (dot) product and ||.|| is the 2-norm.
    Reduction is to take the mean of the cosine similarities.
    """

    def __init__(self, reduction: str = 'mean'):
        super(CosineSimLoss, self).__init__()
        assert reduction in ['mean', 'sum'], 'Please define <reduction> as "mean" or "sum".'
        self.reduction = reduction

    @staticmethod
    def lp_norm(x: torch.Tensor, p: int = 2):
        return torch.pow(torch.sum(torch.abs(x) ** p, dim=1), exponent=1 / p)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 10e-5):
        dot_product_numerator = torch.sum(y_pred * y_true, dim=1)
        dot_norm = self.lp_norm(y_pred, p=2) * self.lp_norm(y_true, p=2)
        eps = torch.Tensor([eps]).to(device=y_pred.device)
        l2norm_denominator = torch.max(dot_norm, torch.repeat_interleave(eps, repeats=dot_norm.size(0)))
        cosine_sim_batch = dot_product_numerator / l2norm_denominator
        # note that the cosine similarity ranges between -1 and 1, where the first means complete dissimilar
        # 1 means similar and 0 means orthogonal.
        if self.reduction == 'mean':
            return torch.mean(cosine_sim_batch, dim=0)
        else:
            return torch.sum(cosine_sim_batch, dim=0)