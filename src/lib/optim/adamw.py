import torch

class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data

                state = self.state[param]

                if len(state) == 0:
                    state['history:ratio'] = []
                    state['history:denom'] = []
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(param.data)
                    state['exp_avg_sq'] = torch.zeros_like(param.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute the effective learning rate and update the parameters
                denom = (exp_avg_sq.sqrt() / bias_correction2**0.5).add_(eps)
                step_size = lr / bias_correction1

                state['history:ratio'].append(torch.mean(torch.abs(exp_avg / denom)).detach().item())
                state['history:denom'].append(torch.mean(denom).detach().item())

                # Apply weight decay to the parameters directly
                if weight_decay > 0:
                    param.data.mul_(1 - lr * weight_decay)
                param.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def _readoff_history(self, ratio=True, param_list=None):
        if param_list is None:
            for group in self.param_groups:
                for param in group['params']:
                    state = self.state[param]
                    yield state['history:ratio' if ratio else 'history:denom']
        else:
            for param in param_list:
                state = self.state[param]
                yield state['history:ratio' if ratio else 'history:denom']

