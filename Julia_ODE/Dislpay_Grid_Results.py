import matplotlib.pyplot as plt
from Methods import *
import pandas as pd
import os
from Julia_ODE.bptt_master_evaluation.pse import *
import seaborn as sns
sns.set_theme()


NAME_Trail = 'Grid_1'

if not os.path.exists('Storage/Result/{}'.format(NAME_Trail)):
    os.mkdir('Storage/Result/{}'.format(NAME_Trail))

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


for root, dirs, files in os.walk("Storage/Generated/.", topdown=False):
    fig_loss, ax_loss = plt.subplots(4, 4)
    fig_pse, ax_pse = plt.subplots(4, 4)
    fig_n_mse, ax_n_mse = plt.subplots(4, 4)
    fig_n_pois, ax_n_pois = plt.subplots(4, 4)

    for idx, root, dirs, files in enumerate(os.walk("Storage/Generated/./{}".format(dirs), topdown=False)):

        idx_x = int(idx/4)
        idx_y = idx%4

        name_ground_truth = '{}/Spike_test.pkl'.format(dirs)
        name_generated = '{}/output_9.pkl'.format(dirs)
        name_loss = '{}/Loss_9.pkl'.format(dirs)
        name_iters = '{}/iters_9.pkl'.format(dirs)

        data_ground_truth = pd.read_pickle('Storage/Generated/{}'.format(name_ground_truth))  # units x trials x time bins
        data_generated = pd.read_pickle('Storage/Generated/{}'.format(name_generated)) * 1  # units x trials x time bins
        data_ground_truth = data_ground_truth[:, :, :np.size(data_generated, 2)]
        data_loss = pd.read_pickle('Storage/Generated/{}'.format(name_loss))
        data_loss = np.nan_to_num(data_loss, 0)
        data_iters = pd.read_pickle('Storage/Generated/{}'.format(name_iters))

        units = np.size(data_ground_truth, 0)
        trials = np.size(data_ground_truth, 1)
        bins = np.size(data_ground_truth, 2)
        dt = 200

        # plot loss
        y_min = np.min(data_loss)
        y_max = np.max(data_loss[15:])
        ax_loss[idx_x, idx_y].plot(data_loss)
        ax_loss[idx_x, idx_y].set_yscale('log')
        ax_loss[idx_x, idx_y].set_ylim(0, y_max)
        ax_loss[idx_x, idx_y].vlines(data_iters, ymin=y_min, ymax=y_max, colors='r', linestyles='dotted')

        # pse analysis
        resolution_LFR = 1000
        sigma_cut_off = 0
        width = dt * 5

        time_points = np.arange(0, 21000, dt)[:bins]
        STMtx_ground_Truth = np.zeros((units, trials, bins))
        STMtx_generated = np.zeros((units, trials, bins))
        STMtx_ground_Truth[:, :] = time_points
        STMtx_generated[:, :] = time_points

        STMtx_ground_Truth *= (data_ground_truth > 0) * 1
        STMtx_generated *= (data_generated > 0) * 1

        LFR_ground_truth = np.zeros((units, trials, resolution_LFR))
        LFR_generated = np.zeros((units, trials, resolution_LFR))

        for i in range(trials):
            LFR_ground_truth[:, i, :], time = get_LFR(STMtx_ground_Truth[:, i, :], data_ground_truth[:, i, :],
                                                      sigma_cut_off=sigma_cut_off,
                                                      width_small=width, resolution_LFR=resolution_LFR)

            LFR_generated[:, i, :], time = get_LFR(STMtx_generated[:, i, :], data_generated[:, i, :],
                                                   sigma_cut_off=sigma_cut_off,
                                                   width_small=width, resolution_LFR=resolution_LFR)

        LFR_generated_conc = np.concatenate((LFR_generated[:, 0, :], LFR_generated[:, 1, :]), axis=1)
        for i in range(1, np.size(LFR_generated, 1) - 1):
            LFR_generated_conc = np.concatenate((LFR_generated_conc, LFR_generated[:, i, :]), axis=1)

        LFR_ground_truth_conc = np.concatenate((LFR_ground_truth[:, 0, :], LFR_ground_truth[:, 1, :]), axis=1)
        for i in range(1, np.size(LFR_ground_truth, 1) - 1):
            LFR_ground_truth_conc = np.concatenate((LFR_ground_truth_conc, LFR_ground_truth[:, i, :]), axis=1)

        Spectrum_gound_truth = get_average_spectrum(LFR_ground_truth_conc)
        Spectrum_generated = get_average_spectrum(LFR_generated_conc)
    
        ax_pse[idx_x, idx_y].plot(Spectrum_gound_truth, label='ground truth')
        ax_pse[idx_x, idx_y].plot(Spectrum_generated, label='generated')
        ax_pse[idx_x, idx_y].legend()

        # mse
        ax_n_mse[idx_x, idx_y].figure(figsize=(12, 12))
        ax_n_mse[idx_x, idx_y].plot(time[:100], n_MSE_Loss(LFR_ground_truth, LFR_generated, 100))

        # poisson loss n ahead
        plt.figure(figsize=(12, 12))
        ax_n_mse[idx_x, idx_y].plot(time[:100], n_poisson_Loss(LFR_ground_truth, LFR_generated, 100))

fig_loss.savefig("Julia_ODE/Storage/Result/{}".format(NAME_Trail))
fig_pse.savefig("Julia_ODE/Storage/Result/{}".format(NAME_Trail))
fig_n_mse.savefig("Julia_ODE/Storage/Result/{}".format(NAME_Trail))
fig_n_pois.savefig("Julia_ODE/Storage/Result/{}".format(NAME_Trail))