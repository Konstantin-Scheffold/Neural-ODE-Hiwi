import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import scipy.io as sio
import os
from Operators import *
sn.set()


def find_add_correlated_signals(LFR, mode='multiplication'):
    # mode can be 'multiplication' for A*B or 'e_plus' for e^((A+B)/np.max(A, B))

    corrMatrix = pd.DataFrame(np.transpose(LFR)).corr()
    roh = np.asarray(corrMatrix)
    # fisher-transfromation
    Z = r_to_z(roh)
    Z_stern = np.zeros((number_units, number_units))

    for i in range(number_units):
        print('#' * 30)
        print('calculate correlated Signals:', int(i / number_units * 100), '%')
        for j in range(0, i + 1):
            # z-test
            control = Test(Z[j, i], n_auto_corr(LFR[i], LFR[j]))
            if control > 1.960755055:
                Z_stern[i, j] = control

    # addable_signals = np.where(LFR < 2000, LFR, 0)
    significant_corr = np.argwhere(Z_stern)
    shifted_signals = np.zeros((len(significant_corr), 2, resolution_LFR))
    added_signals = np.zeros((len(significant_corr), resolution_LFR))

    for i in range(1, len(significant_corr) + 1):

        print(significant_corr[i - 1])
        A = LFR[significant_corr[i - 1, 0]]
        B = LFR[significant_corr[i - 1, 1]]
        limit = int(len(A) / 2)
        cross_corr = correlate(A, B) / len(A)
        cross_corr = cross_corr[int(len(A) / 2):-int(len(A) / 2)]

        print('found lag = ', np.argmax(cross_corr) - limit)

        if (np.argmax(cross_corr) - limit) > 0:
            A = A[(np.argmax(cross_corr) - limit):]
            B = B[:-(np.argmax(cross_corr) - limit)]
        if (np.argmax(cross_corr) - limit) < 0:
            A = A[:(np.argmax(cross_corr) - limit)]
            B = B[-(np.argmax(cross_corr) - limit):]

        shifted_signals[i - 1, :, :len(A)] = np.array([A, B])
        if mode == 'multiplication':
            added_signals[i - 1, :len(A)] = mul(A, B)
        if mode == 'e_plus':
            added_signals[i - 1, :len(A)] = e_plus(A, B)

    return added_signals[:, :len(LFR[0])]

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

    if Find_Add_Correlated_Signals:
        correlated_signals = find_add_correlated_signals(LFR)
    else:
        correlated_signals = None

    return LFR, x, time_points, event_type, correlated_signals





def get_data(name, load, resolution_LFR, mode, number_units, sample_resolution):

    if mode == 'Test_data':
        orig_ts = np.linspace(0, 1, num=resolution_LFR)
        samp_ts = orig_ts[::int(resolution_LFR/sample_resolution)]

        orig_trajs = np.sin(orig_ts * np.pi * 8)
        noise_trajs = orig_trajs[::int(resolution_LFR/sample_resolution)]

        orig_trajs = orig_trajs.reshape(1, np.size(orig_trajs), 1)
        noise_trajs = noise_trajs.reshape(1, np.size(noise_trajs), 1)

        return orig_trajs, noise_trajs, orig_ts, samp_ts



    if resolution_LFR < sample_resolution:
        raise ValueError('The resolution must be smaller than the sample size')

    # search for existing data
    found_files = 0
    Question = ''

    for root, dir, files in os.walk('Spike Data/Storage/Train_data'):
        if 'orig_trajs_' + name + '.npy' in files:
            found_files += 1
        if 'noise_trajs_' + name + '.npy' in files:
                found_files += 1
        if 'orig_ts_' + name + '.npy' in files:
                found_files += 1
        if 'samp_ts_' + name + '.npy' in files:
                found_files += 1
    if found_files == 4:
        found_files = True
    else:
        found_files = False


    if load:
        if found_files:
            orig_trajs = np.load('Spike Data/Storage/Train_data/orig_trajs_' + name + '.npy')
            noise_trajs = np.load('Spike Data/Storage/Train_data/noise_trajs_' + name + '.npy')
            orig_ts = np.load('Spike Data/Storage/Train_data/orig_ts_' + name + '.npy')
            samp_ts = np.load('Spike Data/Storage/Train_data/samp_ts_' + name + '.npy')
        else:
            print('Could not find data with the trial name:' + name)
            Question = input('Do you want to generate new data? y/n')
            if Question == ("y"):
                print('generating new data')
            if Question == ("n"):
                exit()

    # generate data
    if not load or Question=='y':
        data = sio.loadmat('Data/M533_200219_1_37units.mat')
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
        samp_ts = orig_ts[::int(new_resolution/sample_resolution)]
        samp_ts = samp_ts

        orig_trajs = LFR[:number_units, :]
        noise_trajs = LFR[:number_units, ::int(new_resolution/sample_resolution)]
        noise_trajs = noise_trajs + np.random.random(len(noise_trajs)) * 0.3

        orig_trajs = orig_trajs.reshape(1, np.size(orig_trajs), 1)
        noise_trajs = noise_trajs.reshape(1, np.size(noise_trajs), 1)

        np.save('Spike Data/Storage/Train_data/orig_trajs_' + name, orig_trajs)
        np.save('Spike Data/Storage/Train_data/noise_trajs_' + name, noise_trajs)
        np.save('Spike Data/Storage/Train_data/orig_ts_' + name, orig_ts)
        np.save('Spike Data/Storage/Train_data/samp_ts_' + name, samp_ts)

    return orig_trajs, noise_trajs, orig_ts, samp_ts








if __name__ == '__main__':

    resolution_LFR = 10000 # 20000
    sigma_cut_off = 3
    width_small = 5
    width_big = 90

    # get data
    data = sio.loadmat('Data/M533_200219_1_37units.mat')
    Spike_trains = np.asarray(np.nan_to_num(data['STMtx']))
    Event_times = data['EvtT']
    Event_names = data['EvtL']
    number_units = np.min(np.shape(Spike_trains))
    time_length = np.max(Spike_trains)

    LFR, x, Event_times, Event_names, added_signals = get_LFR(Spike_trains, Event_times, Event_names, resolution_LFR, sigma_cut_off,
                                                           width_small, width_big, trend_removal=True, Find_Add_Correlated_Signals=True)

    # plot several units
    plt.figure(figsize=(15, 15))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.plot(x, LFR[i], label = 'Unit {}'.format(i))
        plt.legend()
    plt.show()

    # plot several correlated signals
    plt.figure(figsize=(15, 15))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.plot(x, added_signals[i], label = 'added_signals {}'.format(i))
        plt.legend()
    plt.show()