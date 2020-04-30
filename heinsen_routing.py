import math
import torch
import torch.nn as nn


class Routing(nn.Module):
    """
    Official implementation of the routing algorithm proposed by "An
    Algorithm for Routing Capsules in All Domains" (Heinsen, 2019),
    https://arxiv.org/abs/1911.00792.

    Args:
        d_cov: int, dimension 1 of input and output capsules.
        d_inp: int, dimension 2 of input capsules.
        d_out: int, dimension 2 of output capsules.
        n_inp: (optional) int, number of input capsules. If not provided, any
            number of input capsules will be accepted, limited by memory.
        n_out: (optional) int, number of output capsules. If not provided, it
            can be passed to the forward method; otherwise it will be equal
            to the number of input capsules, limited by memory.
        n_iters: (optional) int, number of routing iterations. Default is 3.
        single_beta: (optional) bool; if True, beta_use and beta_ign are the
            same parameter, otherwise they are distinct. Default: False.
        p_model: (optional) str, specifies how to compute probability of input
            votes at each output capsule. Choices are 'gaussian' for Gaussian
            mixtures and 'skm' for soft k-means. Default: 'gaussian'.
        eps: (optional) small positive float << 1.0 for numerical stability.

    Input:
        a_inp: [..., n_inp] input scores.
        mu_inp: [..., n_inp, d_cov, d_inp] capsules of shape d_cov x d_inp.
        return_R: (optional) bool, if True, return routing probabilities R
            in addition to other outputs. Default: False.
        n_out: (optional) int, number of output capsules. Valid as an input
            only if not already specified as an argument at initialization.

    Output:
        a_out: [..., n_out] output scores.
        mu_out: [..., n_out, d_cov, d_out] capsules of shape d_cov x d_out.
        sig2_out: [..., n_out, d_cov, d_out] variances of shape d_cov x d_out.

    Sample usage:
        >>> a_inp = torch.randn(100)  # 100 input scores
        >>> mu_inp = torch.randn(100, 4, 4)  # 100 capsules of shape 4 x 4
        >>> m = Routing(d_cov=4, d_inp=4, d_out=4, n_inp=100, n_out=10)
        >>> a_out, mu_out, sig2_out = m(a_inp, mu_inp)
        >>> print(mu_out)  # 10 capsules of shape 4 x 4
    """
    def __init__(self, d_cov, d_inp, d_out, n_inp=-1, n_out=-1, n_iters=3, single_beta=False, p_model='gaussian', eps=1e-5):
        super().__init__()
        assert p_model in ['gaussian', 'skm'], 'Unrecognized value for p_model.'
        self.n_iters, self.p_model, self.eps = (n_iters, p_model, eps)
        self.n_inp_is_fixed, self.n_out_is_fixed = (n_inp > 0, n_out > 0)
        one_or_n_inp, one_or_n_out = (max(1, n_inp), max(1, n_out))
        self.register_buffer('CONST_one', torch.tensor(1.0))
        self.W = nn.Parameter(torch.empty(one_or_n_inp, one_or_n_out, d_inp, d_out).normal_() / d_inp)
        self.B = nn.Parameter(torch.zeros(one_or_n_inp, one_or_n_out, d_cov, d_out))
        if not self.n_out_is_fixed: self.B_brk = nn.Parameter(torch.zeros(1, d_cov, d_out))
        self.beta_use = nn.Parameter(torch.zeros(one_or_n_inp, one_or_n_out))
        self.beta_ign = self.beta_use if single_beta else nn.Parameter(torch.zeros(one_or_n_inp, one_or_n_out))
        self.f, self.log_f = (nn.Sigmoid(), nn.LogSigmoid())
        self.softmax, self.log_softmax = (nn.Softmax(dim=-1), nn.LogSoftmax(dim=-1))

    def forward(self, a_inp, mu_inp, return_R=False, **kwargs):
        n_inp = a_inp.shape[-1]
        W = self.W if self.n_inp_is_fixed else self.W.expand(n_inp, -1, -1, -1)
        B = self.B
        if self.n_out_is_fixed:
            if ('n_out' in kwargs): raise ValueError('n_out is fixed!')
            n_out = W.shape[1]
        else:
            n_out = kwargs['n_out'] if ('n_out' in kwargs) else n_inp
            W = W.expand(-1, n_out, -1, -1)
            B = B + self.B_brk * torch.linspace(-1, 1, n_out, device=B.device)[:, None, None]  # break symmetry
        V = torch.einsum('ijdh,...icd->...ijch', W, mu_inp) + B
        f_a_inp = self.f(a_inp).unsqueeze(-1)  # [...i1]
        for iter_num in range(self.n_iters):

            # E-step.
            if iter_num == 0:
                R = (self.CONST_one / n_out).expand(V.shape[:-2])  # [...ij]
            else:
                log_p_simplified = \
                    - torch.einsum('...ijch,...jch->...ij', V_less_mu_out_2, 1.0 / (2.0 * sig2_out)) \
                    - sig2_out.sqrt().log().sum((-2, -1)).unsqueeze(-2) if (self.p_model == 'gaussian') \
                    else self.log_softmax(-V_less_mu_out_2.sum((-2, -1)))  # soft k-means otherwise
                R = self.softmax(self.log_f(a_out).unsqueeze(-2) + log_p_simplified)  # [...ij]

            # D-step.
            D_use = f_a_inp * R
            D_ign = f_a_inp - D_use

            # M-step.
            a_out = (self.beta_use * D_use).sum(dim=-2) - (self.beta_ign * D_ign).sum(dim=-2)  # [...j]
            over_D_use_sum = 1.0 / (D_use.sum(dim=-2) + self.eps)  # [...j]
            mu_out = torch.einsum('...ij,...ijch,...j->...jch', D_use, V, over_D_use_sum)
            V_less_mu_out_2 = (V - mu_out.unsqueeze(-4)) ** 2  # [...ijch]
            sig2_out = torch.einsum('...ij,...ijch,...j->...jch', D_use, V_less_mu_out_2, over_D_use_sum) + self.eps

        return (a_out, mu_out, sig2_out, R) if return_R else (a_out, mu_out, sig2_out)
