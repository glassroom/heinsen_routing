import torch
import torch.nn as nn
from heinsen_routing import Routing


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * x.sigmoid()


class SSTClassifier(nn.Module):
    """
    Args:
        d_depth: int, number of embeddings per token.
        d_emb: int, dimension of token embeddings.
        d_inp: int, number of features computed per embedding.
        d_cap: int, dimension 2 of output capsules.
        n_parts: int, number of parts detected.
        n_classes: int, number of classes.

    Input:
        mask: [..., n] tensor with 1.0 for tokens, 0.0 for padding.
        embs: [..., n, d_depth, d_emb] embeddings for n tokens.

    Output:
        a_out: [..., n_classes] class scores.
        mu_out: [..., n_classes, 1, d_cap] class capsules.
        sig2_out: [..., n_classes, 1, d_cap] class capsule variances.
    """
    def __init__(self, d_depth, d_emb, d_inp, d_cap, n_parts, n_classes):
        super().__init__()
        self.depth_emb = nn.Parameter(torch.zeros(d_depth, d_emb))
        self.detect_parts = nn.Sequential(nn.Linear(d_emb, d_inp), Swish(), nn.LayerNorm(d_inp))
        self.routings = nn.Sequential(
            Routing(d_cov=1, d_inp=d_inp, d_out=d_cap, n_out=n_parts),
            Routing(d_cov=1, d_inp=d_cap, d_out=d_cap, n_inp=n_parts, n_out=n_classes),
        )
        nn.init.kaiming_normal_(self.detect_parts[0].weight)
        nn.init.zeros_(self.detect_parts[0].bias)

    def forward(self, mask, embs):
        a = torch.log(mask / (1.0 - mask))                     # -inf to inf (logit)
        a = a.unsqueeze(-1).expand(-1, -1, embs.shape[-2])     # [bs, n, d_depth]
        a = a.contiguous().view(a.shape[0], -1)                # [bs, (n * d_depth)]

        mu = self.detect_parts(embs + self.depth_emb)          # [bs, n, d_depth, d_inp]
        mu = mu.view(mu.shape[0], -1, 1, mu.shape[-1])         # [bs, (n * d_depth), 1, d_inp]

        for routing in self.routings:
            a, mu, sig2 = routing(a, mu)

        return a, mu, sig2


class SmallNORBClassifier(nn.Module):
    """
    Args:
        n_objs: int, number of objects to detect.
        n_parts: int, number of parts to detect.
        d_chns: int, number of channels in initial convolutions.

    Input:
        images: [..., 2, m, n] stacked smallNORB L and R m x n images.

    Output:
        a_out: [..., n_objs] object scores.
        mu_out: [..., n_objs, 4, 4] object poses.
        sig2_out: [..., n_objs, 4, 4] object pose variances.
    """
    def __init__(self, n_objs, n_parts, d_chns):
        super().__init__()
        self.convolve = nn.Sequential(
            *[m for (inp_ch, out_ch, stride) in zip([4] + [d_chns] * 5,  [d_chns] * 6,  [1, 2] * 3)
              for m in [nn.BatchNorm2d(inp_ch), nn.Conv2d(inp_ch, out_ch, 3, stride), Swish()]]
        )
        self.compute_a = nn.Sequential(nn.BatchNorm2d(d_chns), nn.Conv2d(d_chns, n_parts, 1))
        self.compute_mu = nn.Sequential(nn.BatchNorm2d(d_chns), nn.Conv2d(d_chns, n_parts * 4 * 4, 1))
        self.routings = nn.Sequential(
            Routing(d_cov=4, d_inp=4, d_out=4, n_out=n_parts),
            Routing(d_cov=4, d_inp=4, d_out=4, n_inp=n_parts, n_out=n_objs),
        )
        for conv in [m for m in self.convolve if type(m) == nn.Conv2d]:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.zeros_(conv.bias)

    def add_coord_grid(self, x):
        h, w = x.shape[-2:]
        coord_grid = torch.stack((
            torch.linspace(-1.0, 1.0, steps=h, device=x.device)[:, None].expand(-1, w),
            torch.linspace(-1.0, 1.0, steps=w, device=x.device)[None, :].expand(h, -1),
        )).expand([*x.shape[:-3], -1, -1, -1])
        return torch.cat((x, coord_grid), dim=-3)
        
    def forward(self, images):
        x = self.add_coord_grid(images)                        # [bs, (2 + 2), m, n]
        x = self.convolve(x)                                   # [bs, d_chns, m', n']

        a = self.compute_a(x)                                  # [bs, n_parts, m', n']
        a = a.view(a.shape[0], -1)                             # [bs, (n_parts * m' * n')]

        mu = self.compute_mu(x)                                # [bs, (n_parts * 4 * 4), m', n']
        mu = mu.view([mu.shape[0], -1, 4, 4, *mu.shape[-2:]])  # [bs, n_parts, 4, 4, m', n']
        mu = mu.permute(0, 1, 4, 5, 2, 3).contiguous()         # [bs, n_parts, m', n', 4, 4]
        mu = mu.view(mu.shape[0], -1, 4, 4)                    # [bs, (n_parts * m' * n'), 4, 4]

        for routing in self.routings:
            a, mu, sig2 = routing(a, mu)

        return a, mu, sig2
