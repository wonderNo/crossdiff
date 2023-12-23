import torch
import copy

class OneEuroFilter:

    def __init__(self,te=1, min_cutoff=0.05, beta=0.004, d_cutoff=1.0):
        # The parameters.
        self.te = te
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        # Previous values.
        self.x_prev = None
        self.dx_prev = None
        self.a_d = self.smoothing_factor(self.d_cutoff)

    def smoothing_factor(self, cutoff):
        r = 2 * torch.pi * cutoff * self.te
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter_signal(self, x):
        if self.x_prev is None:
            self.x_prev = copy.deepcopy(x)
            self.dx_prev = torch.zeros_like(x)
            return self.x_prev

        dx = torch.zeros_like(x)
        dx = (x - self.x_prev) / self.te
        self.dx_prev = self.exponential_smoothing(self.a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(self.dx_prev)
        a = self.smoothing_factor(cutoff)
        self.x_prev = self.exponential_smoothing(a, x, self.x_prev)

        return self.x_prev