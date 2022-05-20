
import numpy as np
from numba import cuda
import scipy.io as sio
import os
import math

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

def get_LFR(data, time_points, event_type, resolution_LFR=5000, sigma_cut_off=5, width_small=5,
            width_big=80, trend_removal=True, Find_Add_Correlated_Signals=False):

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
        event_type = event_type[(time_points > sigma_cut_off * width_big) * (time_points < np.max(data) - sigma_cut_off * width_big)]
        time_points = time_points[(time_points > sigma_cut_off * width_big) * (time_points < np.max(data) - sigma_cut_off * width_big)]

    return LFR, x, time_points, event_type





def get_data(name, load, resolution_LFR, mode, obs_dim):

    if mode == 'Test_data':
        orig_ts = np.linspace(0, 1, num=resolution_LFR)
        orig_trajs = np.sin(orig_ts * np.pi * 8)
        orig_trajs = orig_trajs.reshape(1, np.size(orig_trajs), 1)

        return orig_trajs, orig_ts

    # search for existing data
    found_files = 0
    Question = ''

    for root, dir, files in os.walk('Storage/Train_data/'):
        if 'Trajectories_' + name + '_{}'.format(resolution_LFR) + '.npy' in files:
            found_files += 1
        if 'time_series_' + name + '_{}'.format(resolution_LFR) + '.npy' in files:
            found_files += 1

    if found_files == 2:
        found_files = True
    else:
        found_files = False

    print(os.path.abspath("."))
    print('Test')
    if load:
        if found_files:
            orig_trajs = np.load('Storage/Train_data/Trajectories_' + name + '_{}'.format(resolution_LFR) + '.npy')
            orig_ts = np.load('Storage/Train_data/time_series_' + name + '_{}'.format(resolution_LFR) + '.npy')
        else:
            print('Could not find data with the trial name:' + name)
            Question = input('Do you want to generate new data? y/n')
            if Question == ("y"):
                print('generating new data')
            if Question == ("n"):
                exit()

    # generate data
    if not load or Question=='y':

        data = sio.loadmat(r'C:\Users\Konra\PycharmProjects\Neural_ODE\Data\M533_200219_1_37units.mat')
        Spike_trains = np.asarray(np.nan_to_num(data['STMtx']))
        Event_times = data['EvtT']
        Event_names = data['EvtL']

        # time points

        # get data
        LFR, x, _, _, _ = get_LFR(data=Spike_trains,
                                  time_points=Event_times,
                                  event_type=Event_names,
                                  resolution_LFR=resolution_LFR,
                                  sigma_cut_off=5,
                                  trend_removal=True,
                                  Find_Add_Correlated_Signals=False)

        # due to the sigma_cut_off we have now fewer points than resolution_LFR
        new_resolution = np.max(np.shape(LFR))

        # set time points in the intervall [0, 1] to avoid few number resolution for big floats
        orig_ts = np.linspace(0, 1, num=new_resolution)
        orig_trajs = LFR[:obs_dim, :]
        orig_trajs = orig_trajs.reshape(1, np.size(orig_trajs), 1)

        np.save('Storage/Train_data/Trajectories_' + name + '_{}'.format(resolution_LFR), orig_trajs)
        np.save('Storage/Train_data/time_series_' + name + '_{}'.format(resolution_LFR), orig_ts)

    return orig_trajs, orig_ts