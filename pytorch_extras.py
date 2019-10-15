import numpy as np
import math
import torch
from torch.optim.optimizer import Optimizer, required


class SingleCycleScheduler(object):

    def __init__(self, optimizer, n_iters, keys=['lr', 'betas'], frac=0.1, min_lr=0.0):
        self.optimizer, self.n_iters, self.keys, self.min_lr = (optimizer, n_iters, keys, min_lr)
        # Save the optimizer's initial hyperparam values.
        for group in optimizer.param_groups:
            for key in keys:
                group.setdefault('initial_{}'.format(key), group[key])
        self.turn_pt = n_iters * frac
        self.iter_num = -1
        self.step() # run one step to initialize hyperparams of optimizer

    def _get_hyperparam(self, key, init_value):
        """
        Given a param key (e.g., 'lr') and an init_value (e.g., 0.001),
        returns that key's scheduled value for the current iteration.
        """
        factor = self.iter_num / self.turn_pt if self.iter_num <= self.turn_pt \
            else 0.5 * (1.0 + np.cos(np.pi * (self.iter_num - self.turn_pt) / (self.n_iters - self.turn_pt)))
        if key == 'lr':
            return self.min_lr + (init_value - self.min_lr) * factor
        elif key == 'betas':
            min_momentum = 0.9 * init_value[0]
            return (min_momentum + (init_value[0] - min_momentum) * (1.0 - factor), init_value[1])
        else:
            raise ValueError('\"{}\" is not a valid hyperparam key.'.format(key))

    def step(self):
        self.iter_num += 1
        for group in self.optimizer.param_groups:
            for key in group:
                if key in self.keys:
                    init_value = group['initial_{}'.format(key)]
                    group[key] = self._get_hyperparam(key, init_value)


class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:            
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss