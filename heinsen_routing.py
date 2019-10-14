import math
import torch
import torch.nn as nn


class Routing(nn.Module):
    """
    Official implementation of the routing algorithm proposed by "Routing
    Capsules by Their Net Cost to Use or Ignore, with Sample Applications
    in Vision and Language" (Heinsen, 2019).

    Args:
        d_spc: int, dimension 1 of input and output capsules.
        d_out: int, dimension 2 of output capsules.
        n_out: int, number of output capsules.
        d_inp: int, dimension 2 of input capsules.
        n_inp: (optional) int, number of input capsules. If not provided, any
            number of input capsules will be accepted, limited by memory.
        n_iters: (optional) int, number of routing iterations. Default is 3.
        eps: (optional) small positive float << 1.0 for numerical stability.

    Input:
        a_inp: [..., n_inp] input scores
        mu_inp: [..., n_inp, d_spc, d_inp] capsules of shape d_spc x d_inp

    Output:
        a_out: [..., n_out] output scores
        mu_out: [..., n_out, d_spc, d_out] capsules of shape d_spc x d_out
        sig2_out: [..., n_out, d_spc, d_out] variances of shape d_spc x d_out
    """
    def __init__(self, d_spc, d_out, n_out, d_inp, n_inp=-1, n_iters=3, eps=1e-5):
        super().__init__()
        (self.n_iters, self.eps) = n_iters, eps
        self.n_inp_is_fixed = (n_inp > 0)
        one_or_n_inp = max(1, n_inp)
        self.register_buffer('CONST_R_init', torch.tensor(1.0 / n_out))
        self.W = nn.Parameter(torch.empty(one_or_n_inp, n_out, d_inp, d_out).normal_() / d_inp)
        self.B = nn.Parameter(torch.zeros(one_or_n_inp, n_out, d_spc, d_out))
        self.beta_use = nn.Parameter(torch.zeros(one_or_n_inp, n_out))
        self.beta_ign = nn.Parameter(torch.zeros(one_or_n_inp, n_out))
        self.f = nn.Sigmoid()
        self.log_f = nn.LogSigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, a_inp, mu_inp):
        n_inp = a_inp.shape[-1]
        W = self.W if self.n_inp_is_fixed else self.W.expand(n_inp, -1, -1, -1)
        V = torch.einsum('ijdh,...icd->...ijch', W, mu_inp) + self.B
        for iter_num in range(self.n_iters):

            # E-step.
            if iter_num == 0:
                R = self.CONST_R_init.expand(V.shape[:-2])  # [...ij]
            else:
                log_p_simplified = \
                    - torch.einsum('...ijch,...jch->...ij', V_less_mu_out_2, 1.0 / (2.0 * sig2_out)) \
                    - sig2_out.sqrt().log().sum((-2, -1)).unsqueeze(-2)
                R = self.softmax(self.log_f(a_out).unsqueeze(-2) + log_p_simplified)  # [...ij]

            # D-step.
            f_a_inp = self.f(a_inp).unsqueeze(-1)  # [...i1]
            D_use = f_a_inp * R
            D_ign = f_a_inp - D_use

            # M-step.
            a_out = (self.beta_use * D_use).sum(dim=-2) - (self.beta_ign * D_ign).sum(dim=-2)  # [...j]
            over_D_use_sum = 1.0 / (D_use.sum(dim=-2) + self.eps)  # [...j]
            mu_out = torch.einsum('...ij,...ijch,...j->...jch', D_use, V, over_D_use_sum)
            V_less_mu_out_2 = (V - mu_out.unsqueeze(-4)) ** 2  # [...ijch]
            sig2_out = torch.einsum('...ij,...ijch,...j->...jch', D_use, V_less_mu_out_2, over_D_use_sum) + self.eps

        return a_out, mu_out, sig2_out
