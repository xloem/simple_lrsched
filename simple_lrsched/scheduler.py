import torch, torch.optim

from .loss import Loss

class SimpleLR(torch.optim._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        super(SimpleLR, self).__init__(optimizer, last_epoch, verbose)
        self.loss = Loss(self.optimizer)
    def __del__(self):
        super(SimpleLR, self).__del__()
        del self.loss
    def get_lr(self):
        # called from step, which is called after optimizer.step
        return [
            lr # an lr is returned for each param group
            # self.optimizer.param_groups[idx]
            # self.loss.accumulated_loss[idx]
            for idx in range(len(self.optimizer.param_groups))
        ]
