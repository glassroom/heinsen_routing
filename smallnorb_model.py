import torch
import torch.nn as nn
from heinsen_routing import Routing


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * x.sigmoid()


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
            Routing(d_spc=4, d_out=4, n_out=n_parts, d_inp=4, n_iters=3),
            Routing(d_spc=4, d_out=4, n_out=n_objs, d_inp=4, n_inp=n_parts, n_iters=3),
        )

    def add_coord_grid(self, x):
        h, w = x.shape[-2:]
        coord_grid = torch.stack((
            torch.linspace(-1.0, 1.0, steps=h, device=x.device)[:, None].expand(-1, w),
            torch.linspace(-1.0, 1.0, steps=w, device=x.device)[None, :].expand(h, -1),
        )).expand([*x.shape[:-3], -1, -1, -1])
        return torch.cat((x, coord_grid), dim=-3)
        
    def forward(self, images):
        x = self.add_coord_grid(images)                        # [bs, (2 + 2), h, w]
        x = self.convolve(x)                                   # [bs, d_chns, h', w']

        a = self.compute_a(x)                                  # [bs, n_parts, h', w']
        a = a.view(a.shape[0], -1)                             # [bs, (n_parts * h' * w')]

        mu = self.compute_mu(x)                                # [bs, (n_parts * 4 * 4), h', w']
        mu = mu.view([mu.shape[0], -1, 4, 4, *mu.shape[-2:]])  # [bs, n_parts, 4, 4, h', w']
        mu = mu.permute(0, 1, 4, 5, 2, 3).contiguous()         # [bs, n_parts, h', w', 4, 4]
        mu = mu.view(mu.shape[0], -1, 4, 4)                    # [bs, (n_parts * h' * w'), 4, 4]

        for route in self.routings:
            a, mu, sig2 = route(a, mu)

        return a, mu, sig2                                     # [bs, n_objs], [bs, n_objs, 4, 4], [bs, n_objs, 4, 4]