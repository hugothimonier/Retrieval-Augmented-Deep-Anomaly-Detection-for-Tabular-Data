"""Learning rate scheduler.
Forked from https://github.com/OATML/non-parametric-transformers."""

import numpy as np
import torch
from dotmap import DotMap
from fairseq.optim.fairseq_optimizer import FairseqOptimizer
from fairseq.optim.lr_scheduler import cosine_lr_scheduler
from torch import nn
from torch.optim.lr_scheduler import (
    LambdaLR, CosineAnnealingLR)
from transformers import (
    get_constant_schedule,
    get_linear_schedule_with_warmup, 
    get_polynomial_decay_schedule_with_warmup)


def clip_gradient(model, clip: float):
    nn.utils.clip_grad_norm_(model.parameters(), clip)


class ConcatLR(torch.optim.lr_scheduler._LRScheduler):
    """
    From Over9000
    https://github.com/mgrankin/over9000/blob/master/train.py
    """
    def __init__(self, optimizer, scheduler1, scheduler2, total_steps,
                 pct_start=0.5, last_epoch=-1):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.step_start = float(pct_start * total_steps) - 1
        self.curr_epoch = 0
        super(ConcatLR, self).__init__(optimizer, last_epoch)

    def step(self):
        if self.curr_epoch <= self.step_start:
            self.scheduler1.step()
        else:
            self.scheduler2.step()
        self.curr_epoch += 1
        super().step()

    def get_lr(self):
        if self.curr_epoch <= self.step_start:
            return self.scheduler1.get_last_lr()
        else:
            return self.scheduler2.get_last_lr()
        
    def state_dict(self):
        return {'scheduler1':self.scheduler1.state_dict(),
                'scheduler2':self.scheduler2.state_dict()}
    
    def load_state_dict(self, state_dict):
        self.scheduler1.load_state_dict(state_dict['scheduler1'])
        self.scheduler2.load_state_dict(state_dict['scheduler2'])


class LRScheduler:
    def __init__(self, c, name, optimizer):
        self.c = c
        self.name = name
        self.optimizer = optimizer
        self.num_steps = 0

        self.construct_auto_scheduler()

        print(f'Initialized "{name}" learning rate scheduler.')

    def construct_auto_scheduler(self):
        total_steps = self.c.exp_train_total_epochs

        if self.c.exp_optimizer_warmup_proportion >= 0:
            num_warmup_steps = (
                    total_steps * self.c.exp_optimizer_warmup_proportion)
        else:
            num_warmup_steps = self.c.exp_optimizer_warmup_fixed_n_steps

        print(f'Warming up for {num_warmup_steps}/{total_steps} steps.')

        if self.name == 'constant':
            self.scheduler = get_constant_schedule(optimizer=self.optimizer)
        elif self.name == 'linear_warmup':
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps)
        elif self.name == 'cosine_cyclic':
            args = dict(
                warmup_updates=num_warmup_steps,
                warmup_init_lr=1e-7,
                max_lr=self.c.exp_lr,
                lr=[1e-7],
                t_mult=2.,
                lr_period_updates=num_warmup_steps * 2,
                lr_shrink=0.5)
            optim = FairseqOptimizer(None)
            optim._optimizer = optim.optimizer = self.optimizer
            self.scheduler = cosine_lr_scheduler.CosineSchedule(
                optimizer=optim, args=DotMap(args))
        elif self.name == 'polynomial_decay_warmup':
            # Based on the fairseq implementation, which is based on BERT
            self.scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                lr_end=1e-7,
                power=1.0)
        elif self.name == 'flat_and_anneal':
            def d(x):
                return 1

            assert self.c.exp_optimizer_warmup_proportion >= 0

            # We use exp_optimizer_warmup_proportion to denote the
            # flat LR regime, prior to annealing
            dummy = LambdaLR(self.optimizer, d)
            cosine = CosineAnnealingLR(
                self.optimizer, int(total_steps * (
                    1 - self.c.exp_optimizer_warmup_proportion)))
            self.scheduler = ConcatLR(
                self.optimizer, dummy, cosine, total_steps,
                self.c.exp_optimizer_warmup_proportion)
        else:
            raise NotImplementedError

    def step(self):
        self.num_steps += 1
        c_lr = self.c.exp_lr
        num = self.num_steps
        tot = self.c.exp_train_total_epochs

        if self.name == 'cosine_cyclic':
            self.scheduler.step_update(num_updates=num)
        else:
            self.scheduler.step()

    def state_dict(self):
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)
    
    def __repr__(self,):
        return 'Custom Scheduler'

