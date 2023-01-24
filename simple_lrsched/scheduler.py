import torch, torch.optim

from .loss import Loss

class SimpleLR(torch.optim._LRScheduler):
    def __init__(self, optimizer, past_size=1024, degree=3, last_epoch=-1, verbose=False):
        super(SimpleLR, self).__init__(optimizer, last_epoch, verbose)
        self.loss = Loss(self.optimizer)
        self.degree = degree
        self._N_VARS = 3
        assert past_size >= self._N_VARS * degree

        int_dtype = torch.Long
        float_dtype = torch.float64
        device = self.optimizer.param_groups[0][0].device
        int_kwparams = dict(dtype=int_dtype, device=device)
        float_kwparams = dict(dtype=float_dtype, device=device)

        # storage for the variables raised to the different degrees
        self._store_past_row_exponents = torch.ones(self._N_VARS, self.degree, **float_kwparams)
        # degrees to which raising needs to be calculated
        self._const_multiple_degrees = torch.arange(2, self.degree+1, **int_kwparams)
        # this creates a tensor containing the variable exponent combinations in coefficients of
        # a polynomial (a product is formed by indexing past_row_exponents)
        # like: [
        #  [0, 0, 0] x^0y^0z^0 = 1
        #  [0, 0, 1] x^0y^0z^1 = z
        #  [0, 0, 2] x^0y^0z^2 = z^2
        #  [0, 1, 0] x^0y^1z^0 = y
        #  ...
        #  [2, 2, 2] = (x^2)(y^2)(z^2)
        # ]
        exp_idcs = torch.arange(degree ** self._N_VARS, **int_kwparams)
        self._const_past_row_exponent_var_idx = torch.arange(self._N_VARS, **int_kwparams)
        self._const_past_row_exponent_deg_idx = torch.stack(
                [
                    exp_idcs.div(degree**var_idx, rounding_mode='floor') % degree
                    for var_idx in range(self._N_VARS)
                ]
                dim=-1,
                **int_kwparams,
            )
        # then coeffs can be gotten with:
        #  (self._store_past_row_exponents[
        #       self._const_past_row_exponent_var_idx,
        #       self._const_past_row_exponent_deg_idx
        #   ]).prod(dim=-1)
        # storage for the polynomial solution
        self._store_solution = torch.empty((len(exp_idcs),), **float_kwparams)
        # storage for the measured data
        self._past = torch.empty((past_size, 1 + len(exp_idcs)), **float_kwparams)
        self._past_idx = len(self._past)
        self._past_needed = self._past_idx - len(self._var_exps)
        self.last_loss = [None] * len(self.optimizer.param_groups)
        self.last_test = [None] * len(self.optimizer.param_groups)
        self.last_lr = super().get_lr()
    def __del__(self):
        super(SimpleLR, self).__del__()
        del self.loss
    def get_lr(self):
        # called from step, which is called after optimizer.step
        lrs = []
        loss, test = self.loss.accumulated_loss, self.loss.accumulated_test
        # an lr is returned for each param group
        for idx in range(len(self.optimizer.param_groups)):
            loss_diff = None
            test_diff = 0
            assert self.last_loss[idx] != loss[idx] # if this doesn't pass due to logic error we could just reuse last lr. if it doesn't pass due to coincidence and quantization then the assert would be removed. it might make sense to add some subrandom value for quantization issues.
            if self.last_loss is not None:
                loss_diff = loss[idx] - self.last_loss[idx]
            if test[idx] is not None:
                if self.last_test is not None:
                    test_diff = test[idx] - self.last_test[idx]
            #loss_diff *= loss_diff
            if test_diff is not None:
                loss_diff += test_diff #* test_diff

            #loss_value is now the value to minimize by setting the learning rate.
            #1 - store past based on the last learning rate and loss_value
            if self._past_idx > 0:
                self._past_idx -= 1
            else:
                self._past[1:] = self._past[:-1]
            new_past_row = self._past[self._past_idx]
            # 'B' value to minimize
            new_past_row[0] = loss_diff
            # 'A' coefficients raised to powers
            self._store_past_row_exponents[:,1:] = self.last_lr[idx]
            self._store_past_row_exponents[1,1:] = self.last_loss[idx]
            self._store_past_row_exponents[2,1:] = self._step_count - 1
            # perform raising to powers
            self._store_past_row_exponents[:,2:] **= self._const_multiple_degrees
            # calculate coefficients for past entry 'A' from powers
            new_past_row[1:] = (self._store_past_row_exponents[
                    self._const_past_row_exponent_var_idx,
                    self._const_past_row_exponent_deg_idx
                ]).prod(dim=-1)

            if self._past_idx <= self._past_needed:
                #2 - if there is enough past, do a least squares fit of a polynomial to it
                # func calcs X in AX = B, where A and B are the ordered function parameters
                measurements = self._past[self._past_idx:]
                B = measurements[:,:1]
                A = measurements[:,1:]
                self._store_solution[None,:], residuals, rank, singular_values = torch.linalg.lstsq(A, B)


                3 - then minimize the polynomial given the more recent values to calc the lr
                3a - construct the coefficients of the polynomial in 1 variable (lr)
                # we plug in the current values, and find the minimum loss_diff
                # all variables except the lr are provided, and we have a polynomial in 1 variable
                # 
            else:
                2b - otherwise, try a random and small lr to gather data

            lrs.append(lr)
            self.last_loss[idx] = loss[idx]
            self.last_test[idx] = test[idx]

        return lrs
