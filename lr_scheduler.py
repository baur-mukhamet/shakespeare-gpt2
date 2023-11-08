from torch.optim.lr_scheduler import LRScheduler
import math

class CosineScheduleLR(LRScheduler):

    def __init__(self, optimizer, final_lr, T_max, last_epoch = -1, verbose = False):
        self.final_lr = final_lr
        self.T_max = T_max
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]

        #self.base_lrs is a list of lrs for each group defined in LRScheduler
        return [self.final_lr + 
                (base_lr - self.final_lr) * 
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]
    


#-------------------------------------------------------------------------------------


class CosineScheduleWarmRestarts(LRScheduler):

    def __init__(self, optimizer, final_lr, T_max, T_mult, lr_restart_decay, last_epoch = -1, verbose = False):
        
        self.final_lr = final_lr
        self.T_max = T_max
        self.T_mult = T_mult
        self.T_cur = last_epoch
        
        self.restart_i = 0
        self.lr_restart_decay = lr_restart_decay

        super().__init__(optimizer, last_epoch, verbose) # self.step() is called at init, last_epoch = 0

    def get_lr(self):
        # if self.last_epoch == 0:
        #     return [group['lr'] for group in self.optimizer.param_groups]

        return [self.final_lr + 
                (base_lr - self.final_lr) * 
                (1 + math.cos(math.pi * self.T_cur / self.T_max)) / 2
                for base_lr in self.base_lrs]
    
    def step(self):

        self.last_epoch += 1
        self.T_cur += 1

        if self.T_cur >= self.T_max:
            self.T_cur = 0
            self.T_max *= self.T_mult
            self.restart_i += 1
        

        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group['lr'] = (self.lr_restart_decay**self.restart_i) * lr



