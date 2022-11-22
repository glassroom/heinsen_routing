# coding: utf-8

from __future__ import annotations
from typing import Union

import torch
import torch.nn as nn
from torch import einsum


class EfficientVectorRouting(nn.Module):
    """
    Routes input vectors to the output vectors that maximize "bang per bit"
    by best predicting them, with optimizations that reduce parameter count,
    memory use, and computation by orders of magnitude. Each vector is a
    capsule representing an entity in a context (e.g., a word in a paragraph,
    an object in an image). See "An Algorithm for Routing Vectors in
    Sequences" (Heinsen, 2022).

    Args:
        n_inp: int, number of input vectors. If -1, the number is variable.
        n_out: int, number of output vectors.
        d_inp: int, size of input vectors.
        d_out: int, size of output vectors.
        n_iters: (optional) int, number of iterations. Default: 2.
        normalize: (optional) bool, if True and d_out > 1, normalize each
            output vector's elements to mean 0 and variance 1. Default: True.
        memory_efficient: (optional) bool, if True, compute votes lazily to
            reduce memory use by O(n_inp * n_out * d_out), while increasing
            computation by only O(n_iters). Default: True.
        return_dict: (optional) bool, if True, return a dictionary with the
            final state of all internal and output tensors. Default: False.

    Input:
        x_inp: float tensor of input vectors [..., n_inp, d_inp].

    Output:
        x_out: float tensor of output vectors [..., n_out, d_out] by default,
            or a dict with output vectors as 'x_out' if return_dict is True.

    Sample usage:
        >>> # Route 100 vectors of size 1024 to 10 vectors of size 4096.
        >>> m = EfficientVectorRouting(n_inp=100, n_out=10, d_inp=1024, d_out=4096)
        >>> x_inp = torch.randn(100, 1024)  # 100 vectors of size 1024
        >>> x_out = m(x_inp)  # 10 vectors of size 4096
    """
    def __init__(self, n_inp: int, n_out: int, d_inp: int, d_out: int, n_iters: int = 2,
                 normalize: bool = True, memory_efficient: bool = True, return_dict: bool = False) -> None:
        super().__init__()
        assert n_inp > 0 or n_inp == -1, "Number of input vectors must be > 0 or -1 (variable)."
        assert n_out >= 2, "Number of output vectors must be at least 2."
        one_or_n_inp = max(1, n_inp)
        self.n_inp, self.n_out, self.d_inp, self.d_out, self.n_iters = (n_inp, n_out, d_inp, d_out, n_iters)
        self.normalize, self.memory_efficient, self.return_dict = (normalize, memory_efficient, return_dict)
        self.register_buffer('CONST_ones_over_n_out', torch.ones(n_out) / n_out)
        self.W_A = nn.Parameter(torch.empty(one_or_n_inp, d_inp).normal_(std=2.0 * d_inp**-0.5))
        self.B_A = nn.Parameter(torch.zeros(one_or_n_inp))
        self.W_F1 = nn.Parameter(torch.empty(n_out, d_inp).normal_())
        self.W_F2 = nn.Parameter(torch.empty(d_inp, d_out).normal_(std=2.0 * d_inp**-0.5))
        self.B_F2 = nn.Parameter(torch.zeros(n_out, d_out))
        self.W_G1 = nn.Parameter(torch.empty(d_out, d_inp).normal_(std=d_out**-0.5))
        self.W_G2 = nn.Parameter(torch.empty(n_out, d_inp).normal_())
        self.B_G2 = nn.Parameter(torch.zeros(n_out, d_inp))
        self.W_S = nn.Parameter(torch.empty(one_or_n_inp, n_out).normal_(std=d_inp**-0.5))
        self.B_S = nn.Parameter(torch.zeros(one_or_n_inp, n_out))
        if n_inp > 0:
            self.beta_use = nn.Parameter(torch.empty(n_inp, n_out).normal_())
            self.beta_ign = nn.Parameter(torch.empty(n_inp, n_out).normal_())
        else:
            self.compute_beta_use = nn.Linear(d_inp, n_out)
            self.compute_beta_ign = nn.Linear(d_inp, n_out)
        self.N = nn.LayerNorm(d_out, elementwise_affine=False) if d_out > 1 else nn.Identity()
        self.f, self.log_f, self.softmax = (nn.Sigmoid(), nn.LogSigmoid(), nn.Softmax(dim=-1))

    def __repr__(self) -> str:
        cfg_str = ', '.join(f'{s}={getattr(self, s)}' for s in 'n_inp n_out d_inp d_out n_iters normalize memory_efficient return_dict'.split())
        return '{}({})'.format(self._get_name(), cfg_str)

    def forward(self, x_inp: torch.Tensor) -> Union[torch.Tensor, dict]:
        beta_use, beta_ign = (self.beta_use, self.beta_ign) if hasattr(self, 'beta_use') else (self.compute_beta_use(x_inp), self.compute_beta_ign(x_inp))
        scaled_x_inp = x_inp * x_inp.shape[-2]**-0.5  # [...id]
        a_inp = (scaled_x_inp * self.W_A).sum(dim=-1) + self.B_A  # [...i]
        V = None if self.memory_efficient else einsum('...id,jd,dh->...ijh', scaled_x_inp, self.W_F1, self.W_F2) + self.B_F2
        f_a_inp = self.f(a_inp).unsqueeze(-1)  # [...i1]
        for iter_num in range(self.n_iters):

            # E-step.
            if iter_num == 0:
                R = self.CONST_ones_over_n_out  # [j]
            else:
                pred_x_inp = einsum('...jh,hd,jd->...jd', self.N(x_out), self.W_G1, self.W_G2) + self.B_G2
                S = self.log_f(einsum('...id,...jd->...ij', x_inp, pred_x_inp) * self.W_S + self.B_S)
                R = self.softmax(S)  # [...ij]

            # D-step.
            D_use = f_a_inp * R  # [...ij]
            D_ign = f_a_inp - D_use  # [...ij]

            # M-step.
            phi = beta_use * D_use - beta_ign * D_ign  # [...ij] "bang per bit" coefficients
            x_out = einsum('...ij,...id,jd,dh->...jh', phi, scaled_x_inp, self.W_F1, self.W_F2) + einsum('...ij,jh->...jh', phi, self.B_F2) if V is None \
                else einsum('...ij,...ijh->...jh', phi, V)  # use precomputed V if available

        if self.normalize:
            x_out = self.N(x_out)

        if self.return_dict:
            return { 'a_inp': a_inp, 'V': V, 'pred_x_inp': pred_x_inp, 'S': S, 'R': R, 'D_use': D_use, 'D_ign': D_ign, 'phi': phi, 'x_out': x_out }
        else:
            return x_out


class DefinableVectorRouting(nn.Module):
    """
    Routes input vectors to the output vectors that maximize "bang per bit"
    by best predicting them, as specified by four neural networks provided
    as PyTorch module instances. Each vector is a capsule representing an
    entity in a context (e.g., a word in a sentence, an object in an image).
    See "An Algorithm for Routing Vectors in Sequencces" (Heinsen, 2022).

    Args:
        A: nn.Module instance that accepts input vectors and computes input
            vector activation scores: [..., n_inp, d_inp] -> [..., n_inp, 1].
        F: nn.Module instance that accepts input vectors and proposes output
            sequences: [..., n_inp, d_inp] -> [..., n_inp, n_out, d_out].
        G: nn.Module instance that accepts output vectors and predicts input
            vectors: [..., n_out, d_out] -> [..., n_out, d_inp]. G can be a
            generative model that samples from a parametrized distribution.
        S: nn.Module instance that accepts actual and predicted input vectors
            (the latter computed by G) and computes their similary scores:
            [..., n_inp, d_inp], [...,n_out, d_inp] -> [..., n_inp , n_out].
        n_inp: int, number of input vectors. If -1, the number is variable (in
            which case A, F, and S must be able to handle a variable number).
        n_out: int, number of output vectors.
        n_iters: (optional) int, number of iterations. Default: 2.
        return_dict: (optional) bool, if True, return a dictionary with the
            final state of all internal and output tensors. Default: False.

    Input:
        x_inp: float tensor of input vectors [..., n_inp, d_inp].

    Output:
        x_out: float tensor of output vectors [..., n_out, d_out] by default,
            or a dict with output vectors as 'x_out' if return_dict is True.

    Sample usage:
        >>> class LearnedMemories(nn.Module):
        >>>     def __init__(self, n_inp, n_out, d_out):
        >>>         super().__init__()
        >>>         self.W_mem = nn.Parameter(torch.randn(n_inp, n_out, d_out))
        >>>     def forward(self, x_inp):
        >>>         return self.W_mem.expand(*x_inp.shape[:-2], -1, -1, -1)
        >>> 
        >>> class DotProductSimilarities(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>     def forward(self, true_x_inp, pred_x_inp):
        >>>         scaling = true_x_inp.shape[-1]**-0.5
        >>>         return true_x_inp @ pred_x_inp.transpose(-2, -1) * scaling
        >>> 
        >>> # Route 100 vectors of size 1024 to 10 vectors of size 4096.
        >>> m = DefinableVectorRouting(
        >>>     A=nn.Linear(1024, 1),
        >>>     F=LearnedMemories(100, 10, 4096),
        >>>     G=nn.Linear(4096, 1024),
        >>>     S=DotProductSimilarities(),
        >>>     n_inp=100, n_out=10)
        >>> 
        >>> x_inp = torch.randn(100, 1024)  # 100 vectors of size 1024
        >>> x_out = m(x_inp)  # 10 vectors of size 4096
    """
    def __init__(self, A: nn.Module, F: nn.Module, G: nn.Module, S: nn.Module, n_inp: int, n_out: int, n_iters: int = 2, return_dict: bool = False) -> None:
        super().__init__()
        assert n_inp > 0 or n_inp == -1, "Number of input vectors must be > 0 or -1 (variable)."
        assert n_out >= 2, "Number of output vectors must be at least 2."
        self.n_inp, self.n_out = (n_inp, n_out)
        self.A, self.F, self.G, self.S, self.n_iters, self.return_dict = (A, F, G, S, n_iters, return_dict)
        self.register_buffer('CONST_ones_over_n_out', torch.ones(n_out) / n_out)
        if n_inp > 0:
            self.beta_use = nn.Parameter(torch.empty(n_inp, n_out).normal_())
            self.beta_ign = nn.Parameter(torch.empty(n_inp, n_out).normal_())
        else:
            self.compute_beta_use = nn.Linear(d_inp, n_out)
            self.compute_beta_ign = nn.Linear(d_inp, n_out)
        self.f, self.softmax = (nn.Sigmoid(), nn.Softmax(dim=-1))

    def __repr__(self) -> None:
        cfg_str = ',\n '.join(f'{s}={getattr(self, s)}' for s in 'A F G S n_inp n_out n_iters return_dict'.split())
        return '{}({})'.format(self._get_name(), cfg_str)

    def forward(self, x_inp: torch.Tensor) -> Union[torch.Tensor, dict]:
        beta_use, beta_ign = (self.beta_use, self.beta_ign) if hasattr(self, 'beta_use') else (self.compute_beta_use(x_inp), self.compute_beta_ign(x_inp))
        a_inp = self.A(x_inp).view(*x_inp.shape[:-1])  # [...i]
        V = self.F(x_inp).view(*a_inp.shape, self.n_out, -1)  # [...ijh]
        f_a_inp = self.f(a_inp).unsqueeze(-1)  # [...i1]
        for iter_num in range(self.n_iters):

            # E-step.
            if iter_num == 0:
                R = self.CONST_ones_over_n_out  # [j]
            else:
                pred_x_inp = self.G(x_out)  # [...jd]
                S = self.S(x_inp, pred_x_inp)  # [...ij]
                R = self.softmax(S)  # [...ij]

            # D-step.
            D_use = f_a_inp * R  # [...ij]
            D_ign = f_a_inp - D_use  # [...ij]

            # M-step.
            phi = beta_use * D_use - beta_ign * D_ign  # [...ij] "bang per bit" coefficients
            x_out = einsum('...ij,...ijh->...jh', phi, V)

        if self.return_dict:
            return { 'a_inp': a_inp, 'V': V, 'pred_x_inp': pred_x_inp, 'S': S, 'R': R, 'D_use': D_use, 'D_ign': D_ign, 'phi': phi, 'x_out': x_out }
        else:
            return x_out


class GenerativeMatrixRouting(nn.Module):
    """
    Routes input matrices to output matrices generated by probabilistic models
    that best explain certain transformations of the given input matrices. Each
    matrix is a capsule representing an entity in a context (e.g., a word in a
    sentence, an object in an image). Both input and output matrices are paired
    with an activation score quantifying detection of the corresponding entity.
    See "An Algorithm for Routing Capsules in All Domains" (Heinsen, 2019),
    https://arxiv.org/abs/1911.00792.
    Args:
        n_inp: int, number of input matrices. If -1, the number is variable.
        n_out: int, number of output matrices.
        d_cov: int, dimension 1 of input and output matrices.
        d_inp: int, dimension 2 of input matrices.
        d_out: int, dimension 2 of output matrices.
        n_iters: (optional) int, number of routing iterations. Default is 3.
        single_beta: (optional) bool, if True, beta_use (net benefits per unit
            of data) and beta_ign (net costs) are the same. Default: False.
        p_model: (optional) str, specifies how to compute probability of input
            votes at each output matrix. Choices are 'gaussian' for Gaussian
            mixtures and 'skm' for soft k-means. Default: 'gaussian'.
        eps: (optional) small positive float << 1.0 for numerical stability.
        return_dict: (optional) bool, if True, return a dictionary with the
            final state of all internal and output tensors. Default: False.
    Input:
        a_inp: [..., n_inp] input scores.
        mu_inp: [..., n_inp, d_cov, d_inp] matrices of shape d_cov x d_inp.
    Output:
        If return_dict is False (default):
            a_out: [..., n_out] output scores.
            mu_out: [..., n_out, d_cov, d_out] matrices, each d_cov x d_out.
            sig2_out: [..., n_out, d_cov, d_out] matrices, each d_cov x d_out.
        Otherwise:
            Python dict with multiple tensors, including the default output
            tensors as keys 'a_out', 'mu_out', and 'sig2_out'.
    Sample usage:
        >>> a_inp = torch.randn(100)  # 100 input scores
        >>> mu_inp = torch.randn(100, 4, 4)  # 100 capsules of shape 4 x 4
        >>> m = GenerativeMatrixRouting(
        >>>     d_cov=4, d_inp=4, d_out=4, n_inp=100, n_out=10)
        >>> a_out, mu_out, sig2_out = m(a_inp, mu_inp)
        >>> print(a_out)  # 10 activation scores
        >>> print(mu_out)  # 10 matrices of shape 4 x 4 (means)
        >>> print(sig2_out)  # 10 matrices of shape 4 x 4 (variances)
    """
    def __init__(self, n_inp: int, n_out: int, d_cov: int, d_inp: int, d_out: int, n_iters: int = 3,
                 single_beta: bool = False, p_model: str ='gaussian', eps: float = 1e-5, return_dict: bool = False) -> None:
        super().__init__()
        assert n_inp > 0 or n_inp == -1, "Number of input matrices must be > 0 or -1 (variable)."
        assert n_out >= 2, "Number of output matrices must be at least 2."
        assert p_model in ['gaussian', 'skm'], 'Unrecognized value for p_model.'
        one_or_n_inp = max(1, n_inp)
        self.n_inp, self.n_out, self.d_cov, self.d_inp, self.d_out, self.n_iters = (n_inp, n_out, d_cov, d_inp, d_out, n_iters)
        self.single_beta, self.p_model, self.eps, self.return_dict = (single_beta, p_model, eps, return_dict)
        self.n_inp_is_fixed = n_inp > 0
        self.register_buffer('CONST_one', torch.tensor(1.0))
        self.W = nn.Parameter(torch.empty(one_or_n_inp, n_out, d_inp, d_out).normal_() / d_inp)
        self.B = nn.Parameter(torch.zeros(one_or_n_inp, n_out, d_cov, d_out))
        self.beta_use = nn.Parameter(torch.zeros(one_or_n_inp, n_out))
        self.beta_ign = nn.Parameter(torch.zeros(one_or_n_inp, n_out)) if not single_beta else self.beta_use
        self.f, self.log_f, self.softmax, self.log_softmax = (nn.Sigmoid(), nn.LogSigmoid(), nn.Softmax(dim=-1), nn.LogSoftmax(dim=-1))

    def __repr__(self) -> str:
        cfg_str = ', '.join(f'{s}={getattr(self, s)}' for s in 'n_inp n_out d_cov d_inp d_out n_iters single_beta p_model eps return_dict'.split())
        return '{}({})'.format(self._get_name(), cfg_str)

    def forward(self, a_inp: torch.Tensor, mu_inp: torch.Tensor) -> Union[torch.Tensor, dict]:
        W = self.W if self.n_inp_is_fixed else self.W.expand(a_inp.shape[-1], -1, -1, -1)
        V = einsum('...icd,ijdh->...ijch', mu_inp, W) + self.B
        f_a_inp = self.f(a_inp).unsqueeze(-1)  # [...i1]
        for iter_num in range(self.n_iters):

            # E-step.
            if iter_num == 0:
                R = (self.CONST_one / self.n_out).expand(V.shape[:-2])  # [...ij]
            else:
                log_p = \
                    - einsum('...ijch,...jch->...ij', V_less_mu_out_2, 1.0 / (2.0 * sig2_out)) \
                    - sig2_out.sqrt().log().sum((-2, -1)).unsqueeze(-2) if self.p_model == 'gaussian' \
                    else self.log_softmax(-V_less_mu_out_2.sum((-2, -1)))  # soft k-means otherwise
                R = self.softmax(self.log_f(a_out).unsqueeze(-2) + log_p)  # [...ij]

            # D-step.
            D_use = f_a_inp * R  # [...ij]
            D_ign = f_a_inp - D_use  # [...ij]

            # M-step.
            a_out = (self.beta_use * D_use).sum(dim=-2) - (self.beta_ign * D_ign).sum(dim=-2)  # [...j] "bang per bit" activations
            normalized_D_use = D_use / (D_use.sum(dim=-2, keepdims=True) + self.eps)  # [...ij]
            mu_out = einsum('...ij,...ijch->...jch', normalized_D_use, V)
            V_less_mu_out_2 = (V - mu_out.unsqueeze(-4)) ** 2  # [...ijch]
            sig2_out = einsum('...ij,...ijch->...jch', normalized_D_use, V_less_mu_out_2) + self.eps

        if self.return_dict:
            return { 'V': V, 'log_p': log_p, 'R': R, 'D_use': D_use, 'D_ign': D_ign, 'a_out': a_out, 'mu_out': mu_out, 'sig2_out': sig2_out }
        else:
            return a_out, mu_out, sig2_out
