import torch
import torch.nn as nn


class ClusteringAffinity(nn.Module):
    def __init__(self, n_classes, n_centers, sigma):
        super(ClusteringAffinity, self).__init__()
        self.n_classes = n_classes
        self.n_centers = n_centers
        self.sigma = sigma

        self.W = nn.Parameter(torch.Tensor(n_classes, n_centers, 1))
        nn.init.kaiming_normal_(self.W)

    def upper_triangle(self, matrix):
        mask = torch.triu(torch.ones_like(matrix), diagonal=1)
        upper = matrix * mask
        return upper

    def forward(self, f):
        N, h = f.shape
        C, m, _ = self.W.shape

        f_expand = f.view(N, 1, 1, h)
        w_expand = self.W.view(1, C, m, h)
        fw_norm = ((f_expand - w_expand) ** 2).sum(dim=-1)
        distance = torch.exp(-fw_norm / self.sigma)
        distance = distance.max(dim=-1).values

        mc = self.n_classes * self.n_centers
        w_reshape = self.W.view(mc, h)
        w_norm_mat = ((w_reshape.unsqueeze(0) - w_reshape.unsqueeze(1)) ** 2).sum(dim=-1)
        w_norm_upper = self.upper_triangle(w_norm_mat)
        mu = 2.0 / (mc ** 2 - mc) * w_norm_upper.sum()
        residuals = self.upper_triangle((w_norm_upper - mu) ** 2)
        rw = 2.0 / (mc ** 2 - mc) * residuals.sum()

        rw_broadcast = torch.ones(N, 1) * rw
        output = torch.cat([distance, rw_broadcast], dim=-1)
        return output

    def output_shape(self, input_shape):
        return (input_shape[0], self.n_classes + 1)


def affinity_loss(lambd):
    def loss(y_true_plusone, y_pred_plusone):
        onehot = y_true_plusone[:, :-1]
        distance = y_pred_plusone[:, :-1]
        rw = y_pred_plusone[:, -1].mean()

        d_fi_wyi = (onehot * distance).sum(dim=-1, keepdim=True)
        losses = torch.max(lambd + distance - d_fi_wyi, torch.zeros_like(distance))
        L_mm = (losses * (1.0 - onehot)).sum(dim=-1)
        return L_mm + rw

    return loss
