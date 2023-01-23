import collections

class Loss:
    '''
    Tracks a .accumulated_loss member by summing the associated loss from
    .backward() calls for each parameter group of the passed optimizer.

    Optimizers and learning rate schedulers can use an object of this class to
    access training loss.
    '''
    by_optimizer = dict()
    _marked_for_accumulation = set()
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.zero_grad()
        self.by_optimizer[self.optimizer] = self
        def make_mark_accum_hook(index):
            def mark_accum_hook(grad):
                self._marked_for_accumulation.add((self, index))
                # this can also return a tensor to use instead of the passed grad
            return hook
        self.mark_accum_hook_groups = [
            [
                parameter.register_hook(make_mark_accum_hook(index))
                for parameter in group['params']
            ]
            for index, group in enumerate(self.optimizer.param_groups)
        ]
    def zero_grad(self):
        '''Reset the accumulated loss. [should this happen automatically?]'''
        self.accumulated_loss = [0] * len(self.optimizer.param_groups)
        self.accumulated_test = [None] * len(self.optimizer.param_groups)
    @classmethod
    def _accum_marked(cls, loss):
        for (marked_for_accum, group_idx) in cls._marked_for_accumulation:
            marked_for_accum.accumulated_loss[group_idx] += loss
        cls._marked_for_accumulation.clear()
    def __del__(self):
        for group in self.mark_accum_hook_groups:
            for hook in group:
                hook.remove()
        del self.mark_accum_hook_groups
        del self.by_optimizer[self.optimizer]

# hook torch.Tensor.backward to stash loss
import torch
_torch_backward = torch.Tensor.backward
def _wrapped_backward(loss, *params, **kwparams):
    assert not len(Loss._marked_for_accumulation)
    # call backward(), which calls hooks to mark for accumulation
    result = _torch_backward(loss, *params, **kwparams)
    # update any marked objects
    Loss._accum_marked(loss)
    # return to caller
    return result
torch.Tensor.backward = _wrapped_backward

# hook transformers.Trainer.log to maybe get access to evaluation loss
import transformers
_transformers_log = transformers.Trainer.log
def _wrapped_log(trainer, logs):
    result = _transformers_log(trainer, logs)
    import pdb; pdb.set_trace()
    return result
transformers.Trainer.log = _wrapped_log

