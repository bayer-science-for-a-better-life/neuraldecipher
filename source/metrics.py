import torch


def l2_distance(y_pred: torch.Tensor, y_true: torch.Tensor,
                reduction: str = 'mean') -> torch.Tensor:
    """
    Batch-Computation:
    Computes the euclidean distance between two p-dimensional vectors y_pred and y_true defined as
    distance(y_pred, y_true) = sqrt(sum_j=1^k (y_true,j - y_pred,j)^2) = sqrt( (y_true - y_pred)'(y_true-y_pred) )
    --> note that formula above is for two samples y_true and y_pred
    If reduction is 'mean', the mean euclidean distance of a batch will be returned.
    If reduction is 'sum', the sum euclidean distance of a batch will be returned.
    :param y_pred: [torch.Tensor] shape: (batch_size, k) is the predicted batch
    :param y_true: [torch.Tensor] shape: (batch_size, k) is the true batch
    :param reduction: `sum` or `mean` to obtain the the sum or mean value along the batch dimension
    :return: [torch.Tensor] shape: (1,)
    """
    with torch.no_grad():
        distances = y_true - y_pred
        pairwise_distances = torch.sqrt(torch.sum(distances**2, dim=1))
        if reduction == 'mean':
            return torch.mean(pairwise_distances, dim=0)
        elif reduction == 'sum':
            return torch.sum(pairwise_distances, dim=0)



def lp_norm(x: torch.Tensor, p: int) -> torch.Tensor:
    """
    Given a batch of shape nxd this function computes the lp norm for each samples along the row-dimesion.
    :param x: batch of shape nxk where k is the feature dimension.
    :param p: which norm to compute
    :return: norm for each sample along the batch dimension.
    """
    assert len(x.size()) == 2, print("Please insert matrix")
    with torch.no_grad():
        return torch.pow(torch.sum(torch.abs(x) ** p, dim=1), exponent=1 / p)


def cosine_similarity(y_pred: torch.Tensor, y_true: torch.Tensor,
                      reduction: str = 'mean', eps: float = 1e-6) -> torch.Tensor:
    """
    Batch-Computation
    Computes the cosine similarity between vector y_pred and y_true as
    per vector:
        cosine_sim(y_pred, y_true) = scalar(y_pred,y_true) / max( l_2(y_pred)*l_2(y_true), eps )
        eps is used to avoid zero division in case the product of the two l2 norms are close to 0.
    :param y_pred: [torch.Tensor] shape: (batch_size, k) is the predicted batch
    :param y_true: [torch.Tensor] shape: (batch_size, k) is the true batch
    :param reduction: `sum` or `mean` to obtain the the sum or mean value along the batch dimension
    :param eps: small value to avoid zero division
    :return:
    """
    with torch.no_grad():
        dot_product_numerator = torch.sum(y_pred*y_true, dim=1)
        dot_norm = lp_norm(y_pred, p=2)*lp_norm(y_true, p=2)
        l2norm_denominator = torch.max(dot_norm, torch.repeat_interleave(torch.Tensor([eps]), repeats=dot_norm.size(0)))
        cosine_sim_batch = dot_product_numerator/l2norm_denominator
        # note that the cosine similarity ranges between -1 and 1, where the first means complete dissimilar
        # 1 means similar and 0 means orthogonal.
        if reduction == 'mean':
            return torch.mean(cosine_sim_batch, dim=0)
        else:
            return torch.sum(cosine_sim_batch, dim=0)