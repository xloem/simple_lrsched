import torch, torch.optim

from .loss import Loss

class SimpleLR(torch.optim._LRScheduler):
    def __init__(self, optimizer, use_diff=True, past_size=1024, degree=3, last_epoch=-1, verbose=False):
        super(SimpleLR, self).__init__(optimizer, last_epoch, verbose)
        self.loss = Loss(self.optimizer)
        self.degree = degree
        self.use_diff = use_diff
        self._N_VARS = 3
        assert past_size >= self._N_VARS * degree
        # this creates a tensor containing the variable exponent combinations in a polynomial.
        # like: [
        #  [0, 0, 0] x^0y^0z^0 = 1
        #  [0, 0, 1] x^0y^0z^1 = z
        #  [0, 0, 2] x^0y^0z^2 = z^2
        #  [0, 1, 0] x^0y^1z^0 = y
        #  ...
        #  [2, 2, 2] = (x^2)(y^2)(z^2)
        # ]
        exp_idcs = torch.arange(degree ** self._N_VARS)
        self._var_exps = torch.stack(
                [
                    exp_idcs.div(degree**var_idx, rounding_mode='floor') % degree
                    for var_idx in range(self._N_VARS)
                ]
                dim=-1,
            )
        self._past = torch.empty((past_size, self._N_VARS))
        self._past_idx = len(self._past)
        self._past_needed = self._past_idx - len(self._var_exps)
        if self.use_diff:
            self.last_loss = [None] * len(self.optimizer.param_groups)
            self.last_test = [None] * len(self.optimizer.param_groups)
    def __del__(self):
        super(SimpleLR, self).__del__()
        del self.loss
    def get_lr(self):
        # called from step, which is called after optimizer.step
        lrs = []
        loss, test = self.loss.accumulated_loss, self.loss.accumulated_test
        # an lr is returned for each param group
        for idx in range(len(self.optimizer.param_groups)):
            loss_value = None
            test_value = 0
            assert self.last_loss[idx] != loss[idx] # if this doesn't pass due to logic error we could just reuse last lr. if it doesn't pass due to coincidence and quantization then the assert would be removed. it might make sense to add some subrandom value for quantization issues.
            if use_diff:
                if self.last_loss is not None:
                    loss_value = loss[idx] - self.last_loss[idx]
                self.last_loss[idx] = loss[idx]
                if test[idx] is not None:
                    if self.last_test is not None:
                        test_value = test[idx] - self.last_test[idx]
                    self.last_test[idx] = test[idx]
            else
                loss_value = loss[idx]
                test_value = test[idx]
            loss_value *= loss_value
            if test_value is not None:
                loss_value += test_value * test_value

            #loss_value is now the value to minimize by setting the learning rate.
            #1 - store past based on the last learning rate and loss_value
            if self._past_idx > 0:
                self._past_idx -= 1
            else:
                self._past[1:] = self._past[:-1]
            new_past_row = self._past[self._past_idx]
            new_past_row[0] = loss_value
            new_past_row[1] = # get last lr
            new_past_row[2] = # get step

            if self._past_idx <= self._past_needed:
                2 - if there is enough past, do a least squares fit of a polynomial to it

                3 - then minimize the polynomial given the current loss or step number to calc the lr
            else:
                2b - otherwise, try a random and small lr to gather data

            lrs.append(lr)

        return lrs
