from __future__ import annotations
from typing import Callable

from functools import partial

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer

from einops import pack

# functions

def exists(val):
    return val is not None

def append_dims(t, ndim):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndim))

# class

class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        min_shrink_thres = 0.3, # the minimum shrinkage for the params, compared to the cosine similarity of the -gradients with the parameters
        weight_decay = 0.,
        eps = 1e-8,
        regen_reg_rate = 0.,
        grad_ema = True,
        grad_ema_decay = 0.98,
    ):
        assert lr >= 0.
        assert all([0. <= beta <= 1. for beta in betas])
        assert weight_decay >= 0.
        assert regen_reg_rate >= 0.
        assert eps > 0.
        assert not (weight_decay > 0. and regen_reg_rate > 0.), 'weight decay and regenerative regularization cannot be used together'

        self._init_lr = lr

        defaults = dict(
            lr = lr,
            betas = betas,
            eps = eps,
            weight_decay = weight_decay,
            regen_reg_rate = regen_reg_rate,
            grad_ema = grad_ema,
            grad_ema_decay = grad_ema_decay,
            min_shrink_thres = min_shrink_thres
        )

        super().__init__(params, defaults)

    def turn_on_grad_ema(self):
        for group in self.param_groups:
            group['grad_ema'] = True

    def turn_off_grad_ema(self):
        for group in self.param_groups:
            group['grad_ema'] = False

    def clear_grad_ema(self):
        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):
                state = self.state[p]
                state.pop('grad_ema', None)

    @torch.no_grad()
    def shrink_params(self):
        for group in self.param_groups:
            min_shrink_thres = group['min_shrink_thres']

            for p in group['params']:
                state = self.state[p]
                grad_ema = state.get('grad_ema', None)

                if not exists(grad_ema):
                    continue

                # algorithm 1 - direction-aware shrinking (dash)

                neg_gradient = -grad_ema

                packed_neg_gradient, _ = pack([neg_gradient], 'o *')
                packed_params, _ = pack([p.data], 'o *')

                shrink_factor = F.cosine_similarity(packed_neg_gradient, packed_params, dim = -1)
                shrink_factor.clamp_(min = min_shrink_thres)

                shrink_factor_to_broadcast = append_dims(shrink_factor, p.ndim - 1)

                p.mul_(shrink_factor_to_broadcast)

    @torch.no_grad()
    def step(
        self,
        closure: Callable | None = None,
        only_update_grad_ema = False
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, regen_rate, beta1, beta2, eps, grad_ema, grad_ema_decay, state, init_lr = p.grad, group['lr'], group['weight_decay'], group['regen_reg_rate'], *group['betas'], group['eps'], group['grad_ema'], group['grad_ema_decay'], self.state[p], self._init_lr

                # decoupled weight decay

                if wd > 0.:
                    p.mul_(1. - lr / init_lr * wd)

                # regenerative regularization - ICLR 2024
                # https://openreview.net/forum?id=lyoOWX0e0O

                if regen_rate > 0. and 'param_init' in state:
                    param_init = state['param_init']

                    p.lerp_(param_init, lr / init_lr * regen_rate)

                # init state if needed

                if len(state) == 0:
                    state['steps'] = 0
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                    if regen_rate > 0.:
                        state['param_init'] = p.data.clone()

                # get some of the states

                exp_avg, exp_avg_sq, steps = state['exp_avg'], state['exp_avg_sq'], state['steps']

                steps += 1

                # should grad ema

                should_grad_ema = grad_ema

                if should_grad_ema:

                    if 'grad_ema' not in state:
                        # maintain an ema of the grad

                        state['grad_ema'] = grad.clone()

                    grad_ema = state['grad_ema']

                    # update grad ema

                    grad_ema.lerp_(grad, 1. - grad_ema_decay)

                # only updating grad ema for shrinking

                if only_update_grad_ema:
                    continue

                # bias corrections

                bias_correct1 = 1. - beta1 ** steps
                bias_correct2 = 1. - beta2 ** steps

                # decay running averages

                exp_avg.lerp_(grad, 1. - beta1)
                exp_avg_sq.lerp_(grad * grad, 1. - beta2)

                # adam

                update = -lr * (exp_avg / bias_correct1) / (exp_avg_sq / bias_correct2).sqrt().clamp(min = eps)

                p.add_(update)

                # increment steps

                state['steps'] = steps

        return loss

# shrink from dataset

def shrink_params_with_dataset_(
    network: Module,
    dataset: Dataset,
    optim: AdamW | None = None,
    batch_size = 16,
    network_forward_kwargs: dict = dict()
):
    dl = DataLoader(dataset, batch_size = batch_size)

    if not exists(optim):
        optim = AdamW(network.parameters(), lr = 0.)

    optim.clear_grad_ema()

    for inputs in dl:

        if isinstance(inputs, dict):
            loss = network(**inputs, **network_forward_kwargs)
        else:
            loss = network(inputs, **network_forward_kwargs)

        loss.backward()

        optim.step(only_update_grad_ema = True)
        optim.zero_grad()

    optim.shrink_params()
