
import numpy as np
import pickle
import time
import os

name_Trial = 'Bins_15ms'
count_proc = True
dt = 15
max_trial_length = 21000

if not os.path.exists('Storage/Precompiled/{}'.format(name_Trial)):
    os.mkdir('Storage/Precompiled/{}'.format(name_Trial))

list_of_data = os.listdir(r'Storage\Data\Data_Collection')
for idx, name_input in enumerate(list_of_data):
    name_input = name_input
    name_output = 'Spikes_{}_precomp'.format(idx+1)

    with open('Storage/Data/Data_Collection/{}'.format(name_input), "rb") as f:
        data = pickle.load(f)
    STMtx = data["STMtx"]
    trial_lims = data["trial_lims"]
    num_units = len(STMtx)
    num_trials = len(trial_lims)

    bins = np.arange(0, max_trial_length, dt)
    limits = np.array([bins[:-1], bins[1:]]).T
    output = np.zeros((num_units, num_trials, len(bins)-1))

    t_fast_0 = time.time()
    for i in range(len(trial_lims)):
        start = trial_lims[i, 0]
        end = start + max_trial_length

        # only trial times with offset zero
        for k in range(num_units):
            Trial = STMtx[k, (STMtx[k, :] > start) * (STMtx[k, :] < end)] - start
            Is_In = np.zeros((len(limits), len(Trial)))
            Is_In[:, :] = Trial

            Count = np.sum((Is_In.T < limits[:, 1]) * (Is_In.T > limits[:, 0]), 0)
            if count_proc:
                output[k, i, :] = Count # make it a count process
            else:
                output[k, i, :] = (Count>1)*1 # make it a count process

    t_fast_1 = time.time()

    pickle.dump(output, open("Storage/Precompiled/{}/{}.pkl".format(name_Trial, name_output), "wb"))
    print((idx+1)/len(list_of_data) * 100, '%')


