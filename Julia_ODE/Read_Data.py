
import numpy as np
import pickle
import time
import os

name_Trial = 'First_try'

if not os.path.exists('Storage/Precompiled/{}'.format(name_Trial)):
    os.mkdir('Storage/Precompiled/{}'.format(name_Trial))

for idx, name_input in enumerate(os.listdir(r'Julia_ODE\Storage\Data_Collection')):
    name_input = name_input
    name_output = 'Spikes_{}_precomp'.format(idx+1)

    with open('Storage/Data/{}'.format(name_input), "rb") as f:
        data = pickle.load(f)

    STMtx = data["STMtx"]
    trial_lims = data["trial_lims"]
    num_units = len(STMtx)
    num_trials = len(trial_lims)

    max_trial_length = np.max(trial_lims[:, 1]-trial_lims[:, 0])
    bins = np.arange(0, max_trial_length, 200)
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
            output[k, i, :] = Count
    t_fast_1 = time.time()

    print('Fast geschafft')

    pickle.dump(output, open("{}.pkl".format(name_output), "wb"))


