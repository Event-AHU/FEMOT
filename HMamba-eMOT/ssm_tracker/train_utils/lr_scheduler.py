import torch
import math

class TransformerLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    lr = (d_model ** -0.5) * min(step_num ** -0.5, step_num * warmup_steps ** -1.5)
    """
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1):

        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0  
        super(TransformerLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        self.current_step += 1
        scale = self.d_model ** -0.5
        step = self.current_step
        warmup_steps = self.warmup_steps
        lr = scale * min(step ** -0.5, step * warmup_steps ** -1.5)
        return [lr for _ in self.base_lrs]


class NoneLRScheduler:
    """None lr scheduler"""
    def __init__(self, lr) -> None:
        self.lr = lr 

    def step(self, ):
        pass
    
    def get_lr(self):

        return self.lr