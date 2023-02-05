## see __main__ at bottom: this file omstly contains a learning rate scheduler and training function
##         it would be helpful to output how the actual loss reduction compares to the expected one from solving the polynomial.
  ## misplaced text:          for key in ["linear3.



# encode:
# - the weights of a model
# - new input data
# - correct new output data
# - the weights of a model trained to include the correct new output data
# and train a model to generate that final item
# needs: training model on generated data
# needs: source of data showing models before and after finetuning on data

# given dataset
# train model on just one item in the dataset
# in a way that also provides for models already trained on large datasets
# needs: training model on generated data
# needs: data to train against
        

def some_model():
    from mega_pytorch import Mega
    model = Mega(
        num_tokens = 256,
        dim = 128,
        depth = 6,
        ema_heads = 16,
        attn_dim_qk = 64,
        attn_dim_value = 256,
        laplacian_attn_fn = True,
    )
    state_dict = model.state_dict()
    for name, val in [*state_dict.items()]:
        state_dict[name] = val.to(torch.float64)
    model.load_state_dict(state_dict)
    return model.to('cuda')

import tqdm
import time

def avg_loss(model, batches, loss_f, train = False, desc = None):
    device = next(model.parameters()).device
    total_loss = torch.tensor(0.0)
    ct = 0
    batch = []
    old_training = model.training
    model.train(train)
    for batch, labels in batches: #tqdm.tqdm(batches, desc=desc, leave=False):
        ct += 1
        batch = batch.to(device, copy=False)
        labels = labels.to(device, copy=False)
        output = model(batch)
        #assert not torch.any(output.isnan())
        loss = loss_f(output, labels)
        if train:
            loss.backward()
        with torch.no_grad():
            total_loss = loss + total_loss
    if ct > 0:
        total_loss /= ct
    return total_loss

def train_test_loss(model, train_batches, test_batches, loss_f, optim, desc = '', train = True):
    if train:
        optim.zero_grad()
    train_loss = avg_loss(model, train_batches, loss_f, train = train, desc = 'Train ' + desc)
    if train:
        optim.step()
    test_loss = avg_loss(model, test_batches, loss_f, train = False, desc = 'Test ' + desc)
    return train_loss, test_loss
    
class polynomial_lr_schedule:
    def __init__(self, device):
        self.ct = 0
        poly_coeffs = 27 
        max_history = poly_coeffs * poly_coeffs
        #max_history = poly_coeffs
        self.recent = torch.empty((max_history, 4), device=device, dtype=torch.float64)
        #self.mat = torch.empty(self.recent.shape[0], self.recent.shape[0], device=device, dtype=torch.float64)
        self.mat = torch.empty((poly_coeffs, max_history), device=device, dtype=torch.float64)
        self.coeffs = torch.empty((poly_coeffs,), dtype=torch.float64) #device=device, dtype=torch.float64)
        self.mat[-1] = 1

        self.residuals = torch.tensor([], dtype=torch.float64)
        self.rank  = torch.tensor(poly_coeffs, dtype=torch.float64)
        self.singular_values = torch.empty(poly_coeffs, dtype=torch.float64)
    def values2change(self, test_loss, train_loss, new_test_loss, new_train_loss):
        #return new_test_loss / test_loss
        #return new_test_loss * new_train_loss / test_loss / train_loss
        #return (new_test_loss + new_train_loss) / (test_loss + train_loss)
        #return (new_test_loss * new_train_loss) / (test_loss * train_loss)
        #return new_test_loss / test_loss + new_train_loss / train_loss
        #return new_test_loss - test_loss + new_train_loss - train_loss
        return (new_test_loss - test_loss) * (new_test_loss + test_loss) + (new_train_loss - train_loss) * (new_train_loss + train_loss)
        #return new_test_loss + new_train_loss
    def change_identity(self):
        #return 2
        return 0
        #return torch.inf
    def lr2calc(self, lr):
        #return torch.log(lr) #+ 1#0.125
        return lr
    def calc2lr(self, calc):
        #return torch.exp(calc) #- 1#0.125
        return calc
    def add(self, lr, test_loss, train_loss, new_test_loss, new_train_loss):
        loss_change = self.values2change(test_loss, train_loss, new_test_loss, new_train_loss)
        if loss_change.isnan() or test_loss.isnan() or train_loss.isnan():
            return
        #assert not loss_change.isnan()
    #matching_idcs = ((self.recent[:,1] == loss) + (self.recent[:,2] == loss_change)).nonzero()
        #if len(matching_idcs):

        if type(lr) is float:
            lr = torch.tensor(lr, device=self.mat.device, dtype=torch.float64)
        calclr = self.lr2calc(lr)
        if calclr.isinf() or calclr.isnan():
            return
        if (loss_change >= self.change_identity() or (self.recent[-1,1] == test_loss and self.recent[-1,1] == train_loss)) and self.ct > self.mat.shape[0]: # fudge: ensure zero-valued and bad changes don't fill the history
        #if loss_change == self.change_identity():
            #if loss_change == self.change_identity():# or lr < 0:
            #    cutoff = self.mat.shape[0] #len(self.recent)//2
            #else:
            if True:
                cutoff = self.ct // 2
            #    cutoff = self.ct*3//4
            #    cutoff = len(self.recent) - 2
            cutoff += -self.ct
            self.recent[0:cutoff - 1] = self.recent[1:cutoff].clone()
            last = self.recent[cutoff - 1]
            last[0] = calclr
            last[1] = test_loss
            last[2] = train_loss
            last[3] = loss_change
            return
            

        if loss_change == self.change_identity(): # wait for better data
        #    matching_idcs = ((self.recent[-self.ct:,2] == loss_change)).nonzero()
        #    if len(matching_idcs):
        #        # fudge: retain just one zero entry
        #        matching_idx = matching_idcs.item()
        #        matching_entries = self.recent[-self.ct + matching_idx]
        #        matching_entries[...,0] = lr
        #        matching_entries[...,1] = loss
        #        matching_entries[...,2] = loss_change
        #        return
                
        #    # fudge: average them
        #    matching_entries = self.recent[matching_idcs]
        #    matching_entries[...,0] += lr
        #    matching_entries[...,1] += loss
        #    matching_entries[...,2] += loss_change
        #    matching_entries /= 2
        #    # fudge: add a little random noise
        #    #lr += lr * (torch.rand(())-.5) * 0.01
        #    #loss += loss * (torch.rand(())-.5) * 0.01
        #    #loss_change += loss_change * (torch.rand(())-.5) * 0.01
            # fudge: do nothing
            return
        ##else:
        if True:
            self.recent[0:-1] = self.recent[1:].clone()
            last = self.recent[-1]
            last[0] = calclr
            last[1] = test_loss
            last[2] = train_loss
            last[3] = loss_change
            if self.ct < self.recent.shape[0]:
                self.ct += 1
    def calc(self, test_loss, train_loss):
        if self.ct < self.mat.shape[0]:
            return 1e-5 * torch.rand(())
        else:
                # solved_coeffs = loss_changes @ torch.linalg.inv(self.coeffs)
                # solved_coeffs @ self.coeffs_or_new_coeffs = loss_changes
            calclrs, test_losses, train_losses, loss_changes = self.recent[-self.ct:].T
            calclrs2 = calclrs * calclrs
            test_losses2 = test_losses * test_losses
            train_losses2 = train_losses * train_losses
            # this approach has each column as a history item
            # TODO: this could be calc'd upon add
            self.mat[ 0,-self.ct:] = calclrs2 * test_losses2 * train_losses2
            self.mat[ 1,-self.ct:] = calclrs2 * test_losses2 * train_losses
            self.mat[ 2,-self.ct:] = calclrs2 * test_losses2
            self.mat[ 3,-self.ct:] = calclrs2 * test_losses * train_losses2
            self.mat[ 4,-self.ct:] = calclrs2 * test_losses * train_losses
            self.mat[ 5,-self.ct:] = calclrs2 * test_losses
            self.mat[ 6,-self.ct:] = calclrs2 * train_losses2
            self.mat[ 7,-self.ct:] = calclrs2 * train_losses
            self.mat[ 8,-self.ct:] = calclrs2
            self.mat[ 9,-self.ct:] = calclrs * test_losses2 * train_losses2
            self.mat[10,-self.ct:] = calclrs * test_losses2 * train_losses
            self.mat[11,-self.ct:] = calclrs * test_losses2
            self.mat[12,-self.ct:] = calclrs * test_losses * train_losses2
            self.mat[13,-self.ct:] = calclrs * test_losses * train_losses
            self.mat[14,-self.ct:] = calclrs * test_losses
            self.mat[15,-self.ct:] = calclrs * train_losses2
            self.mat[16,-self.ct:] = calclrs * train_losses
            self.mat[17,-self.ct:] = calclrs
            self.mat[18,-self.ct:] = test_losses2 * train_losses2
            self.mat[19,-self.ct:] = test_losses2 * train_losses
            self.mat[20,-self.ct:] = test_losses2
            self.mat[21,-self.ct:] = test_losses * train_losses2
            self.mat[22,-self.ct:] = test_losses * train_losses
            self.mat[23,-self.ct:] = test_losses
            self.mat[24,-self.ct:] = train_losses2
            self.mat[25,-self.ct:] = train_losses
            self.mat[26,-self.ct:] = 1

            #self.mat[2,-self.ct:] = lrs2 * test_losses
            #self.mat[0,-self.ct:] = lrs2 * test_losses2
            #self.mat[1,-self.ct:] = lrs2 * test_losses
            #self.mat[2,-self.ct:] = lrs2
            #self.mat[3,-self.ct:] = lrs * test_losses2
            #self.mat[4,-self.ct:] = lrs * test_losses
            #self.mat[5,-self.ct:] = lrs
            #self.mat[6,-self.ct:] = test_losses2
            #self.mat[7,-self.ct:] = test_losses
            #self.mat[8,-self.ct:] = 1

            #torch.linalg.solve(self.mat.T, loss_changes, out=self.coeffs)

            #residuals = torch.empty(0)
            #rank  = torch.empty(0)
            #singular_values = torch.empty(9)
            #self.coeffs = self.coeffs.to('cpu')
            self.residuals.resize_(0)
            torch.linalg.lstsq(self.mat.T[-self.ct:].to('cpu'), loss_changes.to('cpu'), driver='gelsd', out=(self.coeffs,self.residuals,self.rank,self.singular_values))
            #self.coeffs = self.coeffs.to(self.mat.device)

            #coeffs = loss_changes @ torch.linalg.inv(self.coeffs)#, out=self.mat)
            
            #assert torch.all(torch.isclose(loss_changes, self.coeffs @ self.mat))
            #assert torch.all(torch.isclose(loss_changes, self.mat.T @ self.coeffs))

            # now we solve for lr that minimizes loss_change at loss, given mat only
            # we have:
            # lr * mat0 + loss * mat1 + lr * loss * mat2 + lr2 * loss * mat3 + lr2 * mar4 + loss2 * mat5 + lr * loss2 * mat6 + lr2 * loss * mat7 + lr2 * loss2 * mat8 = change
            # can actually collect the lr terms in the original calc
            # ok

            #losscoeffs = torch.tensor((loss * loss, loss, 1), device=self.mat.device)
            test_loss2 = test_loss * test_loss
            train_loss2 = train_loss * train_loss
            losscoeffs = torch.tensor((
                test_loss2 * train_loss2,
                test_loss2 * train_loss,
                test_loss2,
                test_loss * train_loss2,
                test_loss * train_loss,
                test_loss,
                train_loss2,
                train_loss,
                1
            ), device=self.mat.device)

            # so ...
            # (mat0 * loss2 + mat1 * loss + mat2) * lr2 + (mat3 * loss2 + mat4 * loss + mat5) * lr + (mat6 * loss2 + mat7 * loss + mat8) = loss_change
            # and we take the derivative to get the change in loss change
            # 2 * (mat0 * loss2 + mat1 * loss + mat2) * lr + (mat3 * loss2 + mat4 * loss + mat5) = dloss_change/dlr
            # and solve for dloss_change/dlr = 0
            # lr = -(mat3 * loss2 + mat4 * loss + mat5) / 2(mat0 * loss2 + mat1 * loss + mat2)
            # check the second derivative to see if it is a peak or a valley
            # 2 * (mat0 * loss2 + mat1 * loss + mat2) = d2loss_change/dlr2
            coeffs = self.coeffs.to(self.mat.device)
            calclr2_coeff = (coeffs[0:9] * losscoeffs).sum()
            calclr_coeff = (coeffs[9:18] * losscoeffs).sum()
            const_coeff = (coeffs[18:27] * losscoeffs).sum()
            d2loss_change_dlr2 = 2 * calclr2_coeff

            # solve for the maximum
            calclr = -calclr_coeff / d2loss_change_dlr2
            lr = self.calc2lr(calclr)
            #if torch.abs(d2loss_change_dlr2) <= torch.abs(lr_coeff) / 16 or lr <= 0 or lr.isnan(): # not a clear minimum     ### shrinking the denominator on this line (to max_lr) could prevent nan issues
            if d2loss_change_dlr2 == 0 or lr == 0 or (lr < 0 and self.ct < self.mat.shape[1]) or lr > 8 or lr.isnan(): # not a clear minimum     ### shrinking the denominator on this line (to max_lr) could prevent nan issues
            ####if lr_coeff > -d2loss_change_dlr2:
            ####if d2loss_change_dlr2 <= 0 or lr >= 4:
                # pick a good target loss change, and evaluate the curve at that point to update it
                #target_loss_change = min(torch.log(torch.tensor(-1e-8)), loss_changes.min()/2) #2 * loss_changes.max() - loss_changes.min()
                target_loss_change = min(self.change_identity() - 1e-8, loss_changes.min() ** 0.5) #/ 2
                # (mat0 * loss2 + mat1 * loss + mat2) * lr2 + (mat3 * loss2 + mat4 * loss + mat5) * lr + (mat6 * loss2 + mat7 * loss + mat8) = loss_change
                A = calclr2_coeff
                B = calclr_coeff
                C = const_coeff
                #lr = (-B - torch.sqrt(B*B-4*A*C))/(2*A)
                B_2 = B / 2
                calclr = (-B_2 - torch.sqrt(B_2*B_2-A*C))/A
                lr = self.calc2lr(calclr)
                #print('found a min rather than a max')

            expected_change = calclr * calclr * calclr2_coeff + calclr * calclr_coeff + const_coeff
            #if dbg_expected_change > 0:
                #    lr = torch.rand(()) * lrs.max()

            #if d2loss_change_dlr2 <= torch.abs(calclr_coeff) / 16 or lr == 0 or lr.isnan() or expected_change > 0: # not a clear minimum     ### shrinking the denominator on this line (to max_lr) could prevent nan issues
            #while d2loss_change_dlr2 == 0 or lr == 0 or lr < -1e-2 or lr > 8 or lr.isnan():# or expected_change > 1:# or expected_change > 0: # not a clear minimum     ### shrinking the denominator on this line (to max_lr) could prevent nan issues
            if d2loss_change_dlr2 == 0 or lr == 0 or (lr < 0 and self.ct < self.mat.shape[1]) or lr > 8 or lr.isnan() or expected_change > self.change_identity():# or expected_change > 0: # not a clear minimum     ### shrinking the denominator on this line (to max_lr) could prevent nan issues
            #elif lr <= 0 or lr.isnan():
                # just return a random lr
                #lr = torch.rand(()) * torch.rand(()) * lrs[-1]#loss_changes < 0].max()
                #lr = torch.rand(()) * lrs.max()
                good_losses = self.calc2lr(calclrs[loss_changes < 0])
                if len(good_losses) == 0:
                    good_losses = self.calc2lr(calclrs)
                good_losses = good_losses[good_losses != 0]
                which_approach = torch.randint(3, ())#4, ())
                if which_approach == 0:
                    randmax = max(1e-8, self.calc2lr(calclrs.min())) #torch.rand(()) * lrs.min()
                    lr = torch.rand(()) * randmax
                elif which_approach == 1:
                    lr = good_losses[torch.randint(len(good_losses), ())]
                elif which_approach == 2:
                    randmax = good_losses.max() # if max is increased without checking, it can trigger the > 16 assert below
                    if self.ct == self.mat.shape[1]:
                        lr = torch.rand(()) * randmax * 2 - randmax
                    else:
                        lr = torch.rand(()) * randmax
                #elif which_approach == 1:
                #    lr = torch.rand(()) * good_losses[torch.randint(len(good_losses), ())]
                #elif which_approach == 2:
                #    lr = torch.rand(()) * torch.rand(()) * good_losses[torch.randint(len(good_losses), ())]
                #elif which_approach == 3:
                #    lr = good_losses[-1] # i found slower (3min) + more reliable convergence removing this [ seed=14945593739730613184 2022-11-25 20:08]
                #elif which_approach == 4:
                #    lr = torch.rand(())
                #elif which_approach == 5:
                #    lr = torch.rand(()) * 1e-6
            else:
                pass
                #def dbg_calc_histcol(lr):
                #    #return torch.tensor([lr*lr*loss*loss, lr*lr*loss, lr*lr, lr*loss*loss, lr*loss, lr, loss*loss, loss, 1], device=self.coeffs.device)
                #    return torch.tensor([lr * lr * lr2_coeff, lr * lr_coeff, 
                #dbg_histcol = dbg_calc_histcol(lr)
                #dbg_expected_change = self.coeffs @ dbg_histcol
                #dbg_expected_change = lr * lr * lr2_coeff + lr * lr_coeff + const_coeff
                #if dbg_expected_change > 0:
                #    lr = torch.rand(()) * lrs.max()
                #if lr <= 0:
                #    #lr = torch.rand(()) * lrs.min()# / 2
                #    lr = lrs.min() / 2
                #    if lr == 0:
                #        lr = torch.rand(()) * lrs.max()
            assert not lr.isnan()
            assert lr < 16
            assert lr != 0 # > 0
            return lr




def train(model, train_data, test_data, loss_f, optim = None, timeout = None, accuracy = 2):
    if optim is None:
        import torch.optim
        optim = torch.optim.SGD(model.parameters(), lr=1e-6)

    device = next(model.parameters()).device
    pls = polynomial_lr_schedule(device)

    last_good_state = model.state_dict()#, optim.state_dict())
    start_test_loss = avg_loss(model, test_data if test_data is not None else train_data, loss_f, train = False, desc = 'Starting Test')
    last_test_loss = start_test_loss

    max_lr = 1
    lr = optim.param_groups[0]['lr']

    target = 10**-accuracy
    train_loss, test_loss = train_test_loss(model, train_data, test_data if test_data is not None else train_data, loss_f, optim, 'first')
    start_train_loss = train_loss
    last_train_loss = train_loss
    progress = tqdm.tqdm(total=round(torch.log10(start_test_loss).item()+accuracy, 5), unit='ll', leave=False)
    if timeout is not None:
        deadline = progress.start_t + timeout
    '''can use a polynomial to represent learning rate for positive changes'''
    '''for negative changes, we can use a max learning rate to start'''
    #while test_loss < last_test_loss:
    #while test_loss >= target train_loss >= target:
    while timeout is None or time.time() < deadline:
        #if test_loss != last_test_loss:
            
        pls.add(lr, last_test_loss, last_train_loss, test_loss, train_loss)# / last_test_loss) + (train_loss / last_train_loss))# if test_loss < train_loss else (train_loss / last_train_loss))#(test_loss - last_test_loss))
        if lr > 1e-10 and not test_loss.isnan() and not train_loss.isnan() and (test_loss <= last_test_loss) or (train_loss > test_loss and train_loss < last_train_loss):
            assert test_loss <= start_test_loss or train_loss <= start_train_loss
            #assert train_loss <= start_test_loss
            # success #, try faster lr
            #lr *= 1.0625
            lr = pls.calc(test_loss, train_loss)
            if test_loss > start_test_loss:
                start_test_loss = test_loss
                progress.total = round(torch.log10(start_test_loss).item() + accuracy, 5)
                progress.n = 0
            else:
                progress.n = min(progress.total, round(torch.log10(start_test_loss / test_loss).item(), 5)) # start_test_loss - test_loss
            #progress.update(-(test_loss - last_test_loss).item())
            if test_data is not None:
                progress.set_description(f'test = {test_loss:<6.5g} train = {train_loss:<6.5g} lr = {lr:<6.5g}', refresh=True)
            else:
                progress.set_description(f'loss = {train_loss:<6.5g} lr = {lr:<6.5g}', refresh=True)
            if test_loss <= last_test_loss:# or torch.rand(()) < 0.01:
                last_good_state = {name:val.detach().clone() for name, val in model.state_dict().items()}
                last_test_loss = test_loss
                last_train_loss = train_loss
        else:
            # failure, revert model #, set max test loss
            if torch.rand(()) < 0.1 or test_loss.isnan() or train_loss.isnan():
                model.load_state_dict(last_good_state)
                assert torch.isclose(last_test_loss, avg_loss(model, test_data if test_data is not None else train_data, loss_f, train = False, desc = 'dbg')) or torch.isclose(test_loss, last_test_loss * 2)
            #if lr * 0.0625 > 0:
            #    lr *= 0.0625
            #elif lr * 0.875 > 0:
            #    lr *= 0.875
                lr = pls.calc(last_test_loss, last_train_loss)
            else:
                lr = pls.calc(test_loss, train_loss)
            if test_data is not None:
                progress.set_description(f'searching from {last_test_loss:<6.5g},{last_train_loss:<6.5g} test = {test_loss:<6.5g} train = {train_loss:<6.5g} lr = {lr:<6.5g}', refresh=True)
            else:
                progress.set_description(f'searching from {last_train_loss:<6.5g} loss = {train_loss:<6.5g} lr = {lr:<6.5g}', refresh=True)
            #if lr < 0 or train_loss / test_loss > 16 or torch.rand(()) < 0.001:
            ##if train_loss / test_loss > 16 or torch.rand(()) < 0.01:
            #    with torch.no_grad():
            #        v = max([p.max() for p in model.parameters()])
            #        for parameter in model.parameters():
            #            parameter *= (torch.rand(parameter.shape, device=parameter.device) - 0.5)**2 * 0.1 + 1
            #            #parameter *= (torch.rand(parameter.shape, device=parameter.device) - 0.5) *  + 1
            #            #v = parameter.max()
            #            #parameter *= 1/torch.rand(()) #* v
            #    train_loss, test_loss = train_test_loss(model, train_data, test_data, loss_f, optim, f'move', train=False)
            #    #if test_loss <= start_test_loss and train_loss <= start_train_loss:
            #    #    last_good_state = {name:val.detach().clone() for name, val in model.state_dict().items()}
            #    #    last_train_loss = train_loss
            #    #    last_test_loss = test_loss
            #    lr = pls.calc(test_loss, train_loss)
        if test_loss < target and train_loss < target:
            progress.n = progress.total
            progress.close()
            return True
        optim.param_groups[0]['lr'] = lr
        train_loss, test_loss = train_test_loss(model, train_data, test_data if test_data is not None else train_data, loss_f, optim, f'{lr}')
    progress.close()
    return False

import importlib
import json
import os
import time
def clsobj(cls):
    if type(cls) is not str:
        return cls
    if not cls:
        return None
    modname, clsname = cls.rsplit('.', 1)
    mod = importlib.import_module(modname)
    return getattr(mod, clsname)
def clsname(cls):
    if type(cls) is str:
        return cls
    if not cls:
        return ''
    return cls.__module__ + '.' + cls.__name__
def file2tensor(fn, metadata):
    dtype, shape = metadata
    import torch, numpy as np
    return torch.from_numpy(
        np.fromfile(
            fn,
            dtype
        ).reshape(shape)
    )
def tensor2file(tensor, fn):
    v = tensor.numpy()
    metadata = (str(v.dtype), v.shape)
    with open(fn, 'wb') as f:
        v.tofile(f)
    return metadata
class ModelState:
    '''
        save model parameters and data to disk
    '''
    CONFIGS_FN = 'configuration.json'
    DATA_FN = 'data.jsonl'
    WEIGHTS_FN = 'weights'
    # "weights" distinguishes between function and model parameters
    def __init__(self, path):
        self.path = path
    @property
    def configspath(self):
        return os.path.join(self.path, self.CONFIGS_FN)
    @property
    def datapath(self):
        return os.path.join(self.path, self.DATA_FN)
    @property
    def weightspath(self):
        return os.path.join(self.path, self.WEIGHTS_FN)
    @property
    def nextsubpaths(self):
        return [
            name
            for subdir, name in ((os.path.join(self.path, subdir), subdir) for subdir in os.listdir(self.path))
            if os.path.isdir(subdir) and os.path.exists(os.path.join(subdir, self.CONFIGS_FN))
        ]       
    @property
    def prevpath(self):
        return self.configs.get('prev')
    @property
    def prev(self):
        prevpath = self.prevpath
        if prevpath is None:
            return None
        else:
            return type(self)(prevpath)
    @property
    def root(self):
        prev = self
        while prev.prevpath:
            prev = prev.prev
        return prev
    @property
    def configs(self):
        with open(self.configspath) as f:
            return json.load(f)
    @configs.setter
    def configs(self, configs):
        self.weights = None
        with open(self.configspath, 'w') as f:
            json.dump(configs, f)
    @property
    def additional_data(self):
        with open(self.datapath) as f:
            line = f.readline()
            if len(line):
                yield json.loads(line)
    @additional_data.setter
    def additional_data(self, data):
        if data is None:
            data = ()
        self.weights = None
        with open(self.datapath, 'w') as f:
            try:
                for inputs, labels in data:
                    if hasattr(inputs, 'tolist'):
                        inputs = inputs.tolist()
                    if hasattr(labels, 'tolist'):
                        labels = labels.tolist()
                    json.dump((inputs, labels), f)
                    f.write('\n')
            except:
                os.unlink(self.datapath)
                raise
    @property
    def data(self):
        prev = self.prev
        for inputs, labels in self.additional_data:
            yield torch.tensor(inputs), torch.tensor(labels)
        if prev:
            yield from prev.data
    @property
    def weights(self):
        with open(self.weightspath + '.json') as f:
            return {
                key: file2tensor(os.path.join(self.weightspath, key), metadata)
                for key, metadata in json.load(f).items()
            }
    @weights.setter
    def weights(self, state_dict):
        if state_dict is None:
            state_dict = {}
        os.makedirs(self.weightspath, exist_ok=True)
        with open(self.weightspath + '.json', 'w') as f:
            json.dump({
                key: tensor2file(value, os.path.join(self.weightspath, key))
                for key, value in state_dict.items()
            }, f)
    def create(self, timeout = None, device = None):
        with open(self.configspath) as f:
            configs = json.load(f)
        config = configs.get('config')
        model = configs['model']
        params = configs['params']
        kwparams = configs['kwparams']
        if configs['config']:
            config = clsobj(configs['config'])(*params, **kwparams)
            params = ()
            kwparams = dict(config = config)
        instance = clsobj(model)(*params, **kwparams)
        weights = self.weights
        if not weights:
            if self.prevpath is not None:
                # not handled atm: change of model parameter configuration. would raise here.
                instance.load_state_dict(self.root.create(device=device).state_dict())
            data = list(self.data)
            loss_f = clsobj(configs['loss_f'])
            success = train(instance, data, None, lambda logits, label: loss_f(logits.view(-1, logits.shape[-1]), label.view(-1)), timeout = timeout)
            if not success:
                return None
            weights = instance.state_dict()
            self.weights = instance.state_dict()
        else:
            instance.load_state_dict(weights)
        if device is not None:
            instance = instance.to(device)
        return instance
    def destroy(self):
        for dirpath, dirnames, filenames in os.walk(self.path, topdown=False):
            for filename in filenames:
                os.remove(os.path.join(dirpath, filename))
            for dirname in dirnames:
                assert not os.path.exists(dirname)
            os.rmdir(dirpath)

    def add(self, name, data, **new_kwparams):
        configs = self.configs
        return type(self).new(
            os.path.join(self.path, name),
            configs['model'],
            *configs.get('params', ()),
            data = data,
            prev = self,
            **{
                'config': configs.get('config'),
                'loss_f': configs.get('loss_f'),
                **configs.get('kwparams', {}),
                **new_kwparams
            }
        )
    def __iter__(self):
        return self.nextsubpaths
    def __getitem__(self, itemname):
        return type(self)(os.path.join(self.path, itemname))
    def __delitem__(self, itemname):
        self[itemname].destroy()
    @classmethod
    def new(self, path, cls, *params, config = None, data = None, loss_f = None, prev = None, exist_ok = False, **kwparams):
        if exist_ok and os.path.exists(path):
            new = self(path)
            assert all(
                (passed_inputs == loaded_inputs).all() and (passed_labels == loaded_labels).all()
                for (passed_inputs, passed_labels), (loaded_inputs, loaded_labels) in zip(data or (), new.additional_data, strict=True)
            )
            return new
        os.makedirs(path) # asserts doesn't exist
        modelstate = self(path)
        modelstate.configs = dict(
            prev = prev and (prev if type(prev) is str else prev.path),
            config = clsname(config) or None,
            model = clsname(cls),
            loss_f = clsname(loss_f),
            params = params,
            kwparams = kwparams,
        )
        modelstate.additional_data = data
        modelstate.weights = None
        return modelstate
        
    
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("error")
    import array_adhoc.torch as xp
    import torch
    for SEED in [
        2829709153992461857,
        3721455721308910501,
        3207268124065530100,
        14945593739730613184,
        None,
    ]:
        if SEED is None:
            torch.seed()
            print('SEED:', torch.initial_seed())
        else:
            torch.manual_seed(SEED)
        model = some_model()
        train_data = xp.asarray([
        #    [[[0,1,2]], [[1,2,3]]],
        #    [[[1,2,3]], [[2,3,4]]],
        #    [[[3,4,5]], [[4,5,6]]],
        #    [[[4,5,6]], [[5,6,7]]],
            [[[0,1,2],
              [1,2,3],
              [3,4,5],
              [4,5,6]],
                        [[1,2,3],
                         [2,3,4],
                         [4,5,6],
                         [5,6,7]]]
        ])
        test_data = xp.asarray([
            [[[2,3,4]], [[3,4,5]]],
        ])
        from torch.nn.functional import cross_entropy
        def loss_f(logits, label):
            return cross_entropy(logits.view(-1, logits.shape[-1]), label.view(-1))
        train(model, train_data, test_data, loss_f)

    import numpy as np
    import concurrent.futures
    def process_initial(idx):
        random = np.random.Generator(np.random.PCG64(idx))
        initial = ModelState.new(
            f'test/{idx:03d}',
            'mega_pytorch.Mega',
            exist_ok = True,
            loss_f = 'torch.nn.functional.cross_entropy',
            num_tokens = 256,
            dim = 128,
            depth = 6,
            ema_heads = 16,
            attn_dim_qk = 64,
            attn_dim_value = 256,
            laplacian_attn_fn = True,
        )
        # set type to float64
        if not initial.weights:
            initial.weights = {
                key: val.to(torch.float64)
                for key, val in initial.create().state_dict().items()
            }
        nextidx = 0
        data = random.integers(256, size=17)
        data[0] = nextidx
        next = initial.add(f'{nextidx:03d}', [(data[None,:-1], data[None,1:])], exist_ok = True)

        success = False
        try:
            success = next.create(timeout = 60*15, device='cuda')
        finally:
            if not success:
                next.destroy()
                return
        import floatenc_draft1 as floatenc
        logenc = floatenc.LogEncoder(floatenc.arrays_torch, max=1024, base=128)

        # given each one is a shape, the shape could be placed with separators between rows
        NAME_START = torch.tensor([0])
        AXIS_START = torch.tensor([1])
        AXIS_END = torch.tensor([2])

        # this is a little too confusing for keeping track of
        # i think separator tokens would help
        # then we could use them to discern sizes.
        def is_encfloat(token):
            return token > 127
        #def is_decimal(token):
        #    return (token < 9) + (token == 127)
        def encfloat_to_general(tokens):
            assert (tokens < logenc.base).all()
            return tokens + 128
        #def decimal_to_general(tokens):
        #    assert(tokens < 10).all()
        #    output = tokens.clone().detach()
        #    output[tokens == 9] = 127
        #    return output
        def general_to_encfloat(tokens):
            return tokens - 128
        #def general_to_decimal(tokens):
        #    output = tokens.clone().detach()
        #    output[tokens == 127] = 9
        #    return output
        #def shapeenc(tensor):
        #    enc = logenc.encode(torch.tensor(tensor.shape, dtype=tensor.dtype, device=tensor.device))
        #    return torch.cat((
        #        decimal_to_general(torch.tensor(enc.shape, dtype=tensor.dtype, device=tensor.device)))
        #        encfloat_to_general(enc),
        #    ))
        #def shapedec(tensor):

        #    general_to_decimal(
        #    logenc.decode(

        
        def weightsenc(weights):
            def encode_them(weights):
                name_tensors = list(weights.items())
                cattd_tensors = torch.cat([tensor.view(-1) for name, tensor in name_tensors])
                encoded_tensors = logenc.encode(cattd_tensors) # small (2nd) dimension is tensor, large (1st) dimension is tokens
                # since the small dimension is tensor, we don't need a termination token for data content
                # just for anything delineating the weights etc
                # it would be nice to split them based on weight

                # number of tensors
                # then each tensor's data
                device = encoded_tensors.device
                #yield encfloat_to_general(torch.tensor([len(name_tensors)], device=device))
                #for name, tensor in name_tensors:
                #    yield torch.tensor(list(name.encode()), device=device)
                #    yield decimal_to_general(torch.tensor(tensor.shape))
                name_start = NAME_START.to(device)
                axis_start = AXIS_START.to(device)
                axis_end = AXIS_END.to(device)
                axis_queue = []
                for idx, encoded_slice in enumerate(encoded_tensors):
                    offset = 0
                    for name, tensor in name_tensors:
                        if idx == 0:
                            yield name_start.to(device)
                            yield torch.tensor(list(name.encode()), device=device)
                        axis_queue.append(-1)
                        yield axis_start.to(device)
                        while len(axis_queue):
                            axis_queue[-1] += 1
                            if axis_queue[-1] == tensor.shape[len(axis_queue) - 1]:
                                axis_queue.pop()
                                yield axis_end.to(device)
                            elif len(axis_queue) < len(tensor.shape) - 1:
                                axis_queue.append(-1)
                                yield axis_start.to(device)
                            else:
                                yield axis_start.to(device)
                                next_offset = offset + tensor.shape[-1]
                                yield encfloat_to_general(encoded_slice[offset:next_offset])
                                offset = next_offset
                                yield axis_end.to(device)
                        #size = len(tensor.view(-1))
                        #next_offset = offset + size
                        #yield encfloat_to_general(encoded_slice[offset:next_offset])
                        #offset = next_offset
                        # it could make sense to output the shape, then 
            return torch.cat(tuple(encode_them(weights)))
        def weightsdec(tokens):
            is_name_start = (tokens == NAME_START)
            is_axis_start = (tokens == AXIS_START)
            is_axis_end = (tokens == AXIS_END)
            name_starts = is_name_start.nonzero()
            axis_starts = is_axis_start.nonzero()
            axis_ends = is_axis_end.nonzero()
            tensor_ct = (tokens == NAME_START).sum()
            tensor_names = [None] * tensor_ct
            assert name_starts[0] == 0
            idx = 0
            shape = None
            encoded = []
            rows = []
            prev_start = axis_starts[0]
            prev_end = -1
            next_start_idx = 1
            next_start = axis_starts[next_start_idx]
            next_end_idx = 0
            next_end = axis_ends[next_end_idx]
            tensor_names = []
            tensor_shapes = []
            tensors_size = 0
            #import pdb; pdb.set_trace
            while next_start < len(axis_starts):
                for tensor_idx in range(tensor_ct):
                    if idx == 0:
                        # here we start past the name, and look behind to pick it out
                        assert prev_start > name_starts[tensor_idx]
                        name = bytes(tokens[name_starts[tensor_idx]+1:prev_start]).decode()
                        tensor_names.append(name)
                    axis_queue = [0]

                    # came up with logic to do this ...
                    while len(axis_queue):
                        axis_queue[-1] += 1
                        assert next_start != next_end
                        assert prev_start != prev_end
                        if next_start < next_end:
                            if prev_start > prev_end:
                                # moving deeper into shape
                                axis_queue.append(0)
                            elif prev_start < prev_end:
                                # moving shallower out of shape
                                axis_queue.pop()
                            prev_start = next_start
                            next_start_idx += 1
                            if next_start_idx < len(axis_starts):
                                next_start = axis_starts[next_start_idx]
                            else:
                                next_start = len(tokens)
                        elif next_start > next_end:
                            if prev_start > prev_end:
                                # tensor region
                                row = tokens[prev_start+1:next_end]
                                rows.append(row)
                                shape = torch.tensor(axis_queue)
                                tensor_shapes.append(shape)
                                if idx == 0:
                                    tensors_size += len(row)
                            elif prev_start < prev_end:
                                # between axes
                                pass
                            prev_end = next_end
                            next_end_idx += 1
                            if next_end_idx < len(axis_ends):
                                next_end = axis_ends[next_end_idx]
                            else:
                                next_end = len(tokens)
                    print(f'{tensor_names[tensor_idx]}: {shape}')
                encoded.append(torch.cat(rows))
                rows = []
                idx += 1

            flattened_weights = general_to_encfloat(torch.stack(encoded))

            weights = {}
            offset = 0
            for idx, (name, shape) in enumerate(zip(tensor_names, tensor_shapes)):
                size = shape.prod()
                next_offset = offset + size
                weights[name] = flattened_weights[offset:next_offset].view(shape)
                offset = next_offset
            return weights

            #tensor_ct = encfloat_to_decimal(tokens[0])

            # names and shapes, sequences of is_num = False then is_num = True
            named_shapes = {}
            scalar_ct = 0
            is_dec = is_decimal(tokens)
            dec_change = (is_dec[:-1] ^ is_dec[1:]).nonzero() + 1
            assert dec_change[0] == 1
            for idx in range(len(dec_change)//2):#tensor_ct):
                idx2 = idx * 2
                name_start, value_start, end = dec_change[idx2:idx2+3]
                name = bytes(tokens[name_start:value_start]).decode()
                shape = general_to_decimal(tokens[value_start:end])
                scalar_ct += shape.prod()
                named_shapes[name] = shape

            # decode tensors
            encoded_tensors = general_to_encfloat(tokens[end:].view(-1, scalar_ct))
            return {
                name : decoded.view(shape)
                for (name, shape), decoded in zip(
                    named_shapes,
                    logenc.decode(encoded_tensors)
                )
            }
        encoded_next_weights = weightsenc(next.weights)
        decoded_weights = weightsdec(encoded_next_weights)
        assert decoded_weights == next.weights
        import pdb; pdb.set_trace()
        ''
    for idx in range(16):
        process_initial(idx)
    #for _ in concurrent.futures.ThreadPoolExecutor(max_workers=7).map(process_initial, range(16)):
    #    pass

    #second = initial.add('second', [([[0,1,2]], [[1,2,3]])], exist_ok = True) # this includes batching, it would ideally make batches itself
    #if not second.create(timeout = 30):
    #    second.destroy()
