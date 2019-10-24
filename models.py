import torch
import torch.nn as nn
from heinsen_routing import Routing


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * x.sigmoid()


class SSTClassifier(nn.Module):

    def __init__(self, d_depth, d_emb, d_cap, n_parts, n_classes):
        super().__init__()
        self.depth_emb = nn.Parameter(torch.zeros(d_depth, d_emb))
        self.detect_parts = nn.Sequential(nn.LayerNorm(d_emb), nn.Linear(d_emb, n_parts), Swish())
        self.routings = nn.Sequential(
            Routing(d_cov=1, d_inp=n_parts, d_out=d_cap, n_out=n_parts),
            Routing(d_cov=1, d_inp=d_cap, d_out=d_cap, n_out=n_classes, n_inp=n_parts),
        )
        nn.init.kaiming_normal_(self.detect_parts[1].weight)
        nn.init.zeros_(self.detect_parts[1].bias)

    def forward(self, mask, embs):
        a = torch.log(mask / (1.0 - mask))                     # -inf to inf (logit)
        a = a.unsqueeze(-1).expand(-1, -1, embs.shape[-2])     # [bs, n, d_depth]
        a = a.contiguous().view(a.shape[0], -1)                # [bs, (n * d_depth)]

        mu = self.detect_parts(embs + self.depth_emb)          # [bs, n, d_depth, n_parts]
        mu = mu.view(mu.shape[0], -1, 1, mu.shape[-1])         # [bs, (n * d_depth), 1, n_parts]

        for routing in self.routings:
            a, mu, sig2 = routing(a, mu)

        return a, mu, sig2

    
class SmallNORBClassifier(nn.Module):

    def __init__(self, n_objs, n_parts, d_chns):
        super().__init__()
        self.convolve = nn.Sequential(
            nn.BatchNorm2d(2 + 2), nn.Conv2d(2 + 2, d_chns, kernel_size=3), Swish(),
            nn.BatchNorm2d(d_chns), nn.Conv2d(d_chns, d_chns, kernel_size=3, stride=2), Swish(),
            nn.BatchNorm2d(d_chns), nn.Conv2d(d_chns, d_chns, kernel_size=3), Swish(),
            nn.BatchNorm2d(d_chns), nn.Conv2d(d_chns, d_chns, kernel_size=3, stride=2), Swish(),
            nn.BatchNorm2d(d_chns), nn.Conv2d(d_chns, d_chns, kernel_size=3), Swish(),
            nn.BatchNorm2d(d_chns), nn.Conv2d(d_chns, d_chns, kernel_size=3, stride=2), Swish(),
        )
        self.compute_a = nn.Sequential(nn.BatchNorm2d(d_chns), nn.Conv2d(d_chns, n_parts, 1))
        self.compute_mu = nn.Sequential(nn.BatchNorm2d(d_chns), nn.Conv2d(d_chns, n_parts * 4 * 4, 1))
        self.routings = nn.Sequential(
            Routing(d_cov=4, d_out=4, n_out=n_parts, d_inp=4, n_iters=3),
            Routing(d_cov=4, d_out=4, n_out=n_objs, d_inp=4, n_inp=n_parts, n_iters=3),
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

        for route in self.routings:
            a, mu, sig2 = route(a, mu)

        return a, mu, sig2                                     # [bs, n_objs], [bs, n_objs, 4, 4], [bs, n_objs, 4, 4]