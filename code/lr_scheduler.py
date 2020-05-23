import math

class NoneDecay:
    '''
    None Decay basic scheduler, constant learning rate.
    '''
    def __init__(self, lr):
        self.init(lr)
    def init(self, lr):
        self.lr = lr
    def update(self, epoch_id, optimizer=None):
        return
    def get_lr(self):
        return self.lr

class StepDecay:
    '''
    Step Decay: decreasing learning rate every 150 epochs
    in general, how well it works depends;
    for our models, normally better than exponential decay in performance, but worse in stability of the performance
    '''
    def __init__(self, lr, interval=100, ratio=0.1, min_lr=1e-5):
        self.interval = interval
        self.ratio = ratio
        self.min_lr = min_lr
        self.init(lr)
    def init(self, lr):
        self.lr = lr
    def update(self, epoch_id, optimizer=None):
        if self.lr > self.min_lr and epoch_id and epoch_id % self.interval == 0:
            self.lr *= self.ratio
            if optimizer:
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = self.lr
    def get_lr(self):
        return self.lr

class ESLearningDecay:
    '''
    from paper: Exponential Decay Sine Wave Learning Rate for Fast Deep Neural Network Training
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8305126&tag=1
    claimed to be optimal by them: alpha: [0.4, 0.6], beta: [0.5, 0.7]
    '''
    def __init__(self, lr, T=600, b=1, alpha=0.5, beta=0.6, gamma=0.5, min_lr=1e-5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.T = T
        self.b = b
        self.min_lr = min_lr
        self.init(lr)
    def init(self, lr):
        self.lr = lr
        self.updated_lr = self.lr
    def update(self, epoch_id, optimizer=None):
        t = epoch_id + 1
        param1 = math.e ** -(self.alpha * t / self.T)
        self.updated_lr = max(self.lr * param1 * (self.gamma * math.sin(self.beta * t / (2 * self.b * math.pi)) + param1 + 0.5), self.min_lr)
        if optimizer:
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = self.updated_lr 
    def get_lr(self):
        return self.updated_lr


def check_lr_curve(Scheduler, lr=0.01, epochs=600, **kwargs):
    '''
    The debugger function used to check whether or not the learning rate schedulers work as designed.
    It conducts checking by drawing a plot.
    Args:
        Scheduler: {object}:scheduler, e.g. NoneDecay, ESLearningDecay
        lr: {float} initial learning rate
        epochs: {int} plot how many epochs
        **kwargs: other key word arguments
    Returns:
        nothing
    Sample usage:
        check_lr_curve(ESLearningDecay, lr = 0.005, alpha=2)
        check_lr_curve(ESLearningDecay, lr = 0.005, alpha=2, beta=0.3)
        check_lr_curve(ESLearningDecay, alpha=2)
    '''
    import matplotlib.pyplot as plt
    lr_scheduler = Scheduler(lr, **kwargs)
    epoch = list(range(epochs))
    lr = list()
    for i in epoch:
        lr_scheduler.update(i)
        lr.append(lr_scheduler.get_lr())
    plt.plot(epoch, lr)
    plt.show()


