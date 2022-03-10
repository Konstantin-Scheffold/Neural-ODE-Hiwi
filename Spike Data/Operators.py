
import torch
import numpy as np
from numba import cuda
import math
from scipy.signal import correlate

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

#for calculating LFR

@cuda.jit
def gaussian_kernel_stride(d_x, input, out, width):
    idx = cuda.grid(1)
    stride = cuda.blockDim.x
    if idx >= stride:
        return
    i=idx
    for i in range(idx, out.shape[0], stride):
        start = d_x[i] - 3 * width
        end = d_x[i] + 3 * width
        for time_point in input:
            if start < time_point and time_point < end and time_point != 0:
                out[i] = out[i] + 1/(2*np.pi*width**2)**0.5 * math.exp(-(d_x[i]-time_point)**2/(2*width**2))

def mul(A, B):
    return (A*B)

def e_plus(A, B):
    return np.exp((A+B)/np.max([np.max(A), np.max(B)]))


def r_to_z(r):
    # r to z fisher-transformation
    z = 1/2 * np.log((1+r)/(1-r))
    return z

def Test(z, n):
    # Z test
    T = z * np.sqrt(n) / (1 - z**2)
    return T

def n_auto_corr(x, y):
    # calculate the effective sample size
    N = len(x)
    auto_corr_x = correlate(x, x) / N
    auto_corr_y = correlate(y, y) / N
    n_eff = N / np.sum((1 - np.abs(np.arange(-(N-1), N))/N) * auto_corr_x * auto_corr_y)
    return n_eff