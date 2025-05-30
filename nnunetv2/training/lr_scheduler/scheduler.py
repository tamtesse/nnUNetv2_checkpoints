from torch.optim.lr_scheduler import _LRScheduler

class PolyWarmRestartLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9,
                 drop: float = 0.65, fraction: float = 0.25, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps # should be equal to epochs 
        self.exponent = exponent
        self.drop = drop
        self.fraction = fraction
        self.cycle_start = 0
        self.cycle_length = max_steps * fraction # (1000 * 0.25 = 250)
        self.restart_count = 0
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
            self.restart_count +=1

        # Check if it's time for a restart
        if current_step - self.cycle_start >= self.cycle_length:
            self.cycle_start = current_step
            self.cycle_length = int(self.max_steps * (1 - self.fraction))   # (1000 * 0.75 = 750)

        # Compute position in cycle and update LR
        cycle_progress = current_step - self.cycle_start
        effective_lr = self.initial_lr * (self.drop) ** (self.restart_count)
        new_lr = effective_lr * (1 - cycle_progress / self.cycle_length) ** self.exponent

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
