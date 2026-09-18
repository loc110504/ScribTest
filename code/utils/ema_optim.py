import torch

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * 0.01
        # self.wd = 0.02 * args.base_lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self, alpha=None):
        # ``alpha`` overrides ``self.alpha`` for this call only (e.g.
        # VoxTrust-3D's Trust-Advantage EMA, which recomputes a per-iteration
        # blend rate); omitting it reproduces the original fixed-alpha EMA.
        alpha = self.alpha if alpha is None else alpha
        one_minus_alpha = 1.0 - alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:

                ema_param.mul_(alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)