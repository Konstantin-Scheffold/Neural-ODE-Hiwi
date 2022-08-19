
import numpy as np
from numba import cuda
import scipy.io as sio
import os
import math

@cuda.jit
def gaussian_kernel_stride(d_x, input, weight, out, width):
    idx = cuda.grid(1)
    stride = cuda.blockDim.x
    if idx >= stride:
        return
    i=idx
    for i in range(idx, out.shape[0], stride):
        start = d_x[i] - 3 * width
        end = d_x[i] + 3 * width
        point = 0
        for time_point in input:
            if start < time_point and time_point < end and time_point != 0:
                j = point
                out[i] = out[i] + weight[j] * 1/(2*np.pi*width**2)**0.5 * math.exp(-(d_x[i]-time_point)**2/(2*width**2))
            point +=1

def get_LFR(data, weight, resolution_LFR=5000, sigma_cut_off=5, width_small=5,
            width_big=80, trend_removal=False):

    # select the number of time points for which the LFR is calculated over the complete trial
    # if True the trend of the data is removed
    # if True the correlations for the units are calculated
    # if True the correlations are added and a set of these added signals is returned
    # if True the autocorrelation of several units is displayed
    # if True results get plotted
    # since the trend removal distorts the bounds of the time series the cut off slices a sigma_cut_off sigma surrounding from the start and end
    # choose local kernel width and trend kernel in seconds


    number_units = data.shape[0] #np.min(np.shape(data))
    time_length = data.shape[1] #np.max(data)

    # calculate local LFR
    LFR_local = np.zeros((number_units, resolution_LFR))
    x = np.linspace(0, time_length, resolution_LFR, dtype=np.float32)
    output = np.zeros_like(x, dtype=np.float32)

    for i in range(number_units):
        #print('#' * 30)
        #print('calculate local LFR:', int(i/number_units * 100), '%')
        d_x = cuda.to_device(x)
        d_weigth = cuda.to_device(np.array(weight[i, :], dtype=np.float32))
        d_in = cuda.to_device(np.array(data[i, :], dtype=np.float32))
        d_out = cuda.to_device(output)
        spike_number = len(data[i, :])
        gaussian_kernel_stride[int(spike_number % 128) + 1, 128](d_x, d_in, d_weigth, d_out, width_small)
        LFR_local[i] = d_out.copy_to_host()
    LFR = LFR_local

    if sigma_cut_off>0:
        # here the middle part of the trial data points is sliced out to avoid correlation from the boundings
        LFR = LFR[:, (x > sigma_cut_off * width_big) * (x < np.max(data) - sigma_cut_off * width_big)]
        x = x[(x > sigma_cut_off * width_big) * (x < np.max(data) - sigma_cut_off * width_big)]

    return LFR, x

def poisson_Loss(x_true, x_gen):
    x_gen = x_gen.flatten()
    x_true = x_true.flatten()
    x_true = x_true[x_gen != 0]
    x_gen = x_gen[x_gen != 0]
    return np.mean(x_gen - x_true * np.log(x_gen))

def n_poisson_Loss(x_true, x_gen, n):
    loss_n_ahead = np.zeros(n)
    for i in range(1, n):
        loss_n_ahead[i] = poisson_Loss(x_true[:, :i], x_gen[:, :i])
    return loss_n_ahead

def MSE_Loss(x_true, x_gen):
    return np.mean((x_gen - x_true) ** 2)

def n_MSE_Loss(x_true, x_gen, n):
    loss_n_ahead = np.zeros(n)
    for i in range(1, n):
        loss_n_ahead[i] = MSE_Loss(x_true[:, :i], x_gen[:, :i])
    return loss_n_ahead

def find_number(dirs):
    if len(dirs) == 45:
        numb_latent_dim = int(dirs[17])
        numb_layer_size_1 = int(dirs[41])
        numb_layer_size_2 = int(dirs[43:45])
    elif len(dirs) == 47:
        numb_latent_dim = int(dirs[17:19])
        numb_layer_size_1 = int(dirs[42:44])
        numb_layer_size_2 = int(dirs[45:47])
    elif len(dirs) == 46:
        if int(dirs[17]) == 1:
            numb_latent_dim = int(dirs[17:19])
            hint = int(dirs[42])
        else:
            numb_latent_dim = int(dirs[17])
            hint = int(dirs[41])
        if hint == 8:
            numb_layer_size_1, numb_layer_size_2 = 8, 22
        if hint == 1:
            numb_layer_size_1, numb_layer_size_2 = 16, 36
        if hint == 2:
            numb_layer_size_1, numb_layer_size_2 = 24, 50
        if hint == 3:
            numb_layer_size_1, numb_layer_size_2 = 32, 64

    return numb_latent_dim, numb_layer_size_1, numb_layer_size_2

def find_number_statistical(dirs):

    numb_layer = int(dirs[6])
    numb_latent_dim = int(dirs[12])
    if numb_latent_dim == 1:
        numb_latent_dim = 16

    return numb_latent_dim, numb_layer

def find_idx(numb_laten_dim, numb_layer):
    Layer_num = np.array([1, 2, 3, 4])
    Latent_dim = np.array([2, 4, 8, 16])

    idx_x = np.where(Layer_num == numb_layer)[0][0]
    idx_y = np.where(Latent_dim == numb_laten_dim)[0][0]

    return idx_x, idx_y

def get_params_loss(PARAMS, LOSS, LAY, LAT):

    loss_masked = np.ma.masked_invalid(LOSS)
    params_masked = np.ma.masked_invalid(PARAMS)
    lay_masked = np.ma.masked_invalid(LAY)
    lat_masked = np.ma.masked_invalid(LAT)

    lay_masked = np.ma.masked_array(lay_masked, mask=loss_masked > 10 ** 5)
    lat_masked = np.ma.masked_array(lat_masked, mask=loss_masked > 10 ** 5)
    params_masked = np.ma.masked_array(params_masked, mask=loss_masked > 10 ** 5)
    loss_masked = np.ma.masked_array(loss_masked, mask=loss_masked > 10 ** 5)

    bins_lay = np.unique(LAY[LAY != 0])
    bins_lat = np.unique(LAT[LAT != 0])

    mean_loss, mean_params_lay, mean_params_lat, error_loss = np.zeros((4, 4, 4))
    for i in range(4):
        mean_params_lat[:, i] = bins_lat[i]
        bin_losses = np.ma.masked_array(loss_masked, mask=(lat_masked != bins_lat[i]))
        for j in range(4):
            mean_params_lay[i, j] = bins_lay[j]
            bin_losses_2 = np.ma.masked_array(bin_losses, mask=(lay_masked != bins_lay[j]))
            mean_loss[i, j] = bin_losses_2.mean()
            error_loss[i, j] = bin_losses_2.std()

    mean_params_lat = np.ma.masked_invalid(mean_params_lat)
    mean_params_lay = np.ma.masked_invalid(mean_params_lay)
    mean_loss = np.ma.masked_invalid(mean_loss)
    error_loss = np.ma.masked_invalid(error_loss)

    Lat_Lay_split = (mean_params_lay, mean_params_lat, mean_loss, error_loss)

    loss_masked = np.ma.masked_array(loss_masked, mask=params_masked == 0)
    params_masked = np.ma.masked_array(params_masked, mask=params_masked == 0)
    bins = params_masked.mean(1)
    num_bins = len(bins)

    mean_loss, mean_params, error_loss = np.zeros((3, num_bins))

    for i in range(num_bins):
        mean_params[i] = bins[i]
        bin_losses = loss_masked[i]
        mean_loss[i] = bin_losses.mean()
        error_loss[i] = bin_losses.std()
    # sort the loss values according to the number of parameters
    arg_sort = mean_params.argsort()
    mean_params = mean_params[arg_sort]
    mean_loss = mean_loss[arg_sort]
    error_loss = error_loss[arg_sort]
    # remove nan values from the arrays
    mean_loss = mean_loss[np.invert(np.isnan(mean_params))]
    error_loss = error_loss[np.invert(np.isnan(mean_params))]
    mean_params = mean_params[np.invert(np.isnan(mean_params))]

    only_params = mean_params, mean_loss, error_loss

    return Lat_Lay_split, only_params

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    noise = True

    Params_loss_lay, Params_loss_lat = np.random.randint(1, 5, (2, 16,10))
    Params_loss_loss = Params_loss_lay * Params_loss_lat * 20
    if noise:
        Params_loss_loss = Params_loss_loss + np.random.random((Params_loss_loss.shape))*10
    mean_params_lay, mean_params_lat, mean_loss, error_loss = get_params_loss(np.zeros((16,10)), Params_loss_loss, Params_loss_lay, Params_loss_lat)

    fig_loss_params, ax_loss_params = plt.subplots(1, 2, sharey=True,  figsize=(10, 10), dpi=100)
    ax_loss_params[0].set_ylabel("Loss converged too")

    for i in range(4):
        ax_loss_params[0].fill_between(mean_params_lay[i], mean_loss[i] - error_loss[i], mean_loss[i] + error_loss[i], color='lightskyblue', alpha=0.35)
        ax_loss_params[0].errorbar(mean_params_lay[i], mean_loss[i], yerr=error_loss[i], ls='none')
        ax_loss_params[0].scatter(mean_params_lay[i], mean_loss[i], label='Latent dimensions = {}'.format(mean_params_lat[i, 0]))
        ax_loss_params[0].set_xlabel("Number of Layers")

        ax_loss_params[1].fill_between(mean_params_lat.T[i], mean_loss.T[i] - error_loss.T[i], mean_loss.T[i] + error_loss.T[i], color='lightskyblue', alpha=0.35)
        ax_loss_params[1].errorbar(mean_params_lat.T[i], mean_loss.T[i], yerr=error_loss.T[i], ls='none')
        ax_loss_params[1].scatter(mean_params_lat.T[i], mean_loss.T[i], label='Layers = {}'.format(mean_params_lay.T[i, 0]))
        ax_loss_params[1].set_xlabel("Number of Latent States")
    ax_loss_params[0].legend()
    ax_loss_params[1].legend()
    plt.show()
