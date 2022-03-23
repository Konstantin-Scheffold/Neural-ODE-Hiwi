import numpy as np
import math
import matplotlib.pyplot as plt
from numba import cuda

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


def get_LFR(data, time_points, resolution_LFR=5000, sigma_cut_off=5, width_small=5,
            width_big=80, trend_removal=True):

    # select the number of time points for which the LFR is calculated over the complete trial
    # if True the trend of the data is removed
    # if True the correlations for the units are calculated
    # if True the correlations are added and a set of these added signals is returned
    # if True the autocorrelation of several units is displayed
    # if True results get plotted
    # since the trend removal distorts the bounds of the time series the cut off slices a sigma_cut_off sigma surrounding from the start and end
    # choose local kernel width and trend kernel in seconds


    number_units = np.min(np.shape(data))
    time_length = np.max(data)

    # calculate local LFR
    LFR_local = np.zeros((number_units, resolution_LFR))
    x = np.linspace(-0.1 * time_length, time_length + 0.1 * time_length, resolution_LFR, dtype=np.float32)
    output = np.zeros_like(x, dtype=np.float32)

    for i in range(number_units):
        print('#' * 30)
        print('calculate local LFR:', int(i/number_units * 100), '%')
        d_x = cuda.to_device(x)
        d_in = cuda.to_device(np.array(data[i, :], dtype=np.float32))
        d_out = cuda.to_device(output)
        spike_number = len(data[i, :])
        gaussian_kernel_stride[int(spike_number % 128) + 1, 128](d_x, d_in, d_out, width_small)
        LFR_local[i] = d_out.copy_to_host()
    LFR = LFR_local

    # Calculate LFR trend
    if trend_removal:
        LFR_Trend = np.zeros((number_units, resolution_LFR))
        output = np.zeros_like(x, dtype=np.float32)

        for i in range(number_units):
            print('#' * 30)
            print('calculate LFR trend:', int(i / number_units * 100), '%')
            d_x = cuda.to_device(x)
            d_in = cuda.to_device(np.array(data[i, :], dtype=np.float32))
            d_out = cuda.to_device(output)
            spike_number = len(data[i, :])
            gaussian_kernel_stride[int(spike_number % 128) + 1, 128](d_x, d_in, d_out, width_big)
            LFR_Trend[i] = d_out.copy_to_host()
        LFR = (LFR_Trend - LFR_local)

    if sigma_cut_off>0:
        # here the middle part of the trial data points is sliced out to avoid correlation from the boundings
        LFR = LFR[:, (x > sigma_cut_off * width_big) * (x < np.max(data) - sigma_cut_off * width_big)]
        x = x[(x > sigma_cut_off * width_big) * (x < np.max(data) - sigma_cut_off * width_big)]
        time_points = time_points[(time_points > sigma_cut_off * width_big) * (time_points < np.max(data) - sigma_cut_off * width_big)]

    return LFR, x, time_points



def visualise(data_reduced, data, n_components, title):

    fig = plt.figure()

    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(data_reduced[:, 0], data_reduced[:, 1])
        plt.title(title, fontsize=18)
        plt.savefig('results/2_d/{}_2D'.format(title))

    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_reduced[:, 0], data_reduced[:, 1], data_reduced[:, 2])
        plt.title(title, fontsize=18)
        plt.savefig('results/3_d/{}_3D'.format(title))
    return

