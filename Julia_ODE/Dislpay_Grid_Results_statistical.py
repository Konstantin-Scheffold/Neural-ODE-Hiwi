import numpy as np

from Methods import *
import pandas as pd
from Julia_ODE.bptt_master_evaluation.pse import *
import os
from Dimension_reduction.Methods import PCA_method
from Julia_ODE.bptt_master_evaluation.klx import *
import matplotlib.pyplot as plt


import seaborn as sns
sns.set_theme()

NAME_Trail = 'Fast_statistical_search'
dt = 50
mse_n = 11
pois_n = 15
pse_length = 500000
fuck = 0

if not os.path.exists('Storage/Result/{}'.format(NAME_Trail)):
    os.mkdir('Storage/Result/{}'.format(NAME_Trail))

fig_loss_params, ax_loss_params = plt.subplots(1, 2, sharey=True, figsize=(10, 10), dpi=100)
fig_loss_params_1, ax_loss_params_1 = plt.subplots(figsize=(10, 10), dpi=100)
fig_loss, ax_loss = plt.subplots(4, 4, sharex=False, sharey=False, figsize=(15,15), dpi=200)
fig_pse, ax_pse = plt.subplots(4, 4, sharex=False, sharey=False, figsize=(15,15), dpi=200)
fig_n_mse, ax_n_mse = plt.subplots(4, 4, sharex=False, sharey=False, figsize=(15,15), dpi=200)
fig_n_pois, ax_n_pois = plt.subplots(4, 4, sharex=False, sharey=False, figsize=(15,15), dpi=200)

ax_loss_params[0].set_ylabel("Loss converged too")
ax_loss_params_1.set_ylabel("Loss converged too")
ax_loss_params_1.set_xlabel("Number of parameters")

ax_loss[0, 0].set_ylabel("Layer Number 1")
ax_loss[1, 0].set_ylabel("Layer Number 2")
ax_loss[2, 0].set_ylabel("Layer Number 3")
ax_loss[3, 0].set_ylabel("Layer Number 4")
ax_loss[3, 0].set_xlabel("Latent dimension 2")
ax_loss[3, 1].set_xlabel("Latent dimension 4")
ax_loss[3, 2].set_xlabel("Latent dimension 8")
ax_loss[3, 3].set_xlabel("Latent dimension 16")

ax_pse[0, 0].set_ylabel("Layer Number 1")
ax_pse[1, 0].set_ylabel("Layer Number 2")
ax_pse[2, 0].set_ylabel("Layer Number 3")
ax_pse[3, 0].set_ylabel("Layer Number 4")
ax_pse[3, 0].set_xlabel("Latent dimension 2")
ax_pse[3, 1].set_xlabel("Latent dimension 4")
ax_pse[3, 2].set_xlabel("Latent dimension 8")
ax_pse[3, 3].set_xlabel("Latent dimension 16")

ax_n_mse[0, 0].set_ylabel("Layer Number 1")
ax_n_mse[1, 0].set_ylabel("Layer Number 2")
ax_n_mse[2, 0].set_ylabel("Layer Number 3")
ax_n_mse[3, 0].set_ylabel("Layer Number 4")
ax_n_mse[3, 0].set_xlabel("Latent dimension 2")
ax_n_mse[3, 1].set_xlabel("Latent dimension 4")
ax_n_mse[3, 2].set_xlabel("Latent dimension 8")
ax_n_mse[3, 3].set_xlabel("Latent dimension 16")

ax_n_pois[0, 0].set_ylabel("Layer Number 1")
ax_n_pois[1, 0].set_ylabel("Layer Number 2")
ax_n_pois[2, 0].set_ylabel("Layer Number 3")
ax_n_pois[3, 0].set_ylabel("Layer Number 4")
ax_n_pois[3, 0].set_xlabel("Latent dimension 2")
ax_n_pois[3, 1].set_xlabel("Latent dimension 4")
ax_n_pois[3, 2].set_xlabel("Latent dimension 8")
ax_n_pois[3, 3].set_xlabel("Latent dimension 16")

Params_loss_params = np.zeros((16, 10))
Params_loss_lay = np.zeros((16, 10))
Params_loss_lat = np.zeros((16, 10))
Params_loss_loss = np.zeros((16, 10))

for dirs_Main in os.listdir("Storage/Generated/{}".format(NAME_Trail)):
    numb_latent_dim, numb_layer_size_1 = find_number_statistical(dirs_Main)
    idx_x, idx_y = find_idx(numb_latent_dim, numb_layer_size_1)

    y_min = 1000
    y_max = 2000
    LOSS = np.zeros((10, 6012))
    PSE_GROUND = np.zeros((10, pse_length))
    PSE_GEN = np.zeros((10, pse_length))
    MSE_AHEAD = np.zeros((10, mse_n))
    POIS_AHEAD = np.zeros((10, pois_n))

    for idx, dirs in enumerate(sorted(os.listdir("Storage/Generated/{}/{}".format(NAME_Trail, dirs_Main)))):

        name_ground_truth = '{}/Spike_test.pkl'.format(dirs)
        name_generated = '{}/output_9.pkl'.format(dirs)
        name_loss = '{}/Loss.pkl'.format(dirs)
        name_iters = '{}/iters.pkl'.format(dirs)
        name_params = '{}/Opt_Param.pkl'.format(dirs)

        data_ground_truth = pd.read_pickle('Storage/Generated/{}/{}/{}'.format(NAME_Trail, dirs_Main, name_ground_truth))  # units x trials x time bins
        data_generated = pd.read_pickle('Storage/Generated/{}/{}/{}'.format(NAME_Trail, dirs_Main, name_generated)) * 1  # units x trials x time bins
        data_iters = pd.read_pickle('Storage/Generated/{}/{}/{}'.format(NAME_Trail, dirs_Main, name_iters))

        if data_generated.shape == data_ground_truth.shape and int(data_iters.max()) == 4708:

            print(idx_x + idx_y * 4, idx)

            data_ground_truth = data_ground_truth[:, :, :np.size(data_generated, 2)]
            data_loss = pd.read_pickle('Storage/Generated/{}/{}/{}'.format(NAME_Trail, dirs_Main, name_loss))
            data_loss = np.nan_to_num(data_loss, 0)
            num_params = len(pd.read_pickle('Storage/Generated/{}/{}/{}'.format(NAME_Trail, dirs_Main, name_params)))

            # for plot of number of parameter vs Loss plot
            Layer_num = np.array([1, 2, 3, 4])
            Latent_dim = np.array([2, 4, 8, 16])
            last_loss = data_loss[-1]
            Params_loss_lay[idx_x + idx_y * 4, idx] = Layer_num[idx_x]
            Params_loss_lat[idx_x + idx_y * 4, idx] = Latent_dim[idx_y]
            Params_loss_params[idx_x + idx_y * 4, idx] = num_params
            Params_loss_loss[idx_x + idx_y * 4, idx] = last_loss

            units = np.size(data_ground_truth, 0)
            trials = np.size(data_ground_truth, 1)
            bins = np.size(data_ground_truth, 2)

            # loss
            LOSS[idx] = data_loss

            # Count process to LFR
            resolution_LFR = int(21000/dt)*7
            width = dt * 4

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
                                                          sigma_cut_off=0,
                                                          width_small=width, resolution_LFR=resolution_LFR)

                LFR_generated[:, i, :], time = get_LFR(STMtx_generated[:, i, :], data_generated[:, i, :],
                                                       sigma_cut_off=0,
                                                       width_small=width, resolution_LFR=resolution_LFR)

            LFR_generated_conc = np.concatenate((LFR_generated[:, 0, :], LFR_generated[:, 1, :]), axis=1)
            for i in range(1, np.size(LFR_generated, 1) - 1):
                LFR_generated_conc = np.concatenate((LFR_generated_conc, LFR_generated[:, i, :]), axis=1)

            LFR_ground_truth_conc = np.concatenate((LFR_ground_truth[:, 0, :], LFR_ground_truth[:, 1, :]), axis=1)
            for i in range(1, np.size(LFR_ground_truth, 1) - 1):
                LFR_ground_truth_conc = np.concatenate((LFR_ground_truth_conc, LFR_ground_truth[:, i, :]), axis=1)

            # pse analysis
            pse_ground = get_average_spectrum(LFR_ground_truth_conc)
            PSE_GROUND[idx, :pse_ground.shape[0]] = pse_ground
            pse_gen = get_average_spectrum(LFR_generated_conc)
            PSE_GEN[idx, :pse_gen.shape[0]] = pse_gen

            # mse
            MSE_AHEAD[idx] = n_MSE_Loss(LFR_ground_truth, LFR_generated, mse_n)

            # poisson loss n ahead
            POIS_AHEAD[idx] = n_poisson_Loss(LFR_ground_truth, LFR_generated, pois_n)

        else:
            fuck +=1
            print('Fuck', fuck)

    #Loss

    LOSS = np.ma.masked_array(LOSS)
    for i in range(10):
        if np.percentile(LOSS[i, :], 30) == 0:
            LOSS[i] = np.ma.masked_array(LOSS[i], mask=np.ones_like(LOSS[i]))

    error = np.std(LOSS, 0)
    mean = np.mean(LOSS, 0)
    x = np.arange(LOSS.shape[1])
    ax_loss[idx_x, idx_y].fill_between(x, mean - error, mean + error, color='lightskyblue', alpha=0.35)
    ax_loss[idx_x, idx_y].plot(x, mean, color='black')
    ax_loss[idx_x, idx_y].set_yscale('log')
    y_min = np.percentile(mean, 0.5)
    y_max = np.percentile(mean, 99.6)
    ax_loss[idx_x, idx_y].set_ylim(y_min, y_max)
    data_iters = np.append(data_iters, 5109)
    ax_loss[idx_x, idx_y].vlines(data_iters, ymin=y_min, ymax=y_max, colors='r', linestyles='dotted')

    # PSE
    mean_ground = np.mean(PSE_GROUND, 0)
    mean_gen = np.mean(PSE_GEN, 0)
    error_ground = np.std(PSE_GROUND, 0)
    error_gen = np.std(PSE_GEN, 0)
    mean_ground = mean_ground[error_ground != 0]
    mean_gen = mean_gen[error_gen != 0]
    error_ground = error_ground[error_ground != 0]
    error_gen = error_gen[error_gen != 0]

    x = np.arange(mean_ground.shape[0])
    ax_pse[idx_x, idx_y].fill_between(x, mean_ground - error_ground, mean_ground + error_ground, color='lightskyblue', alpha=0.35)
    ax_pse[idx_x, idx_y].fill_between(x, mean_gen - error_gen, mean_gen + error_gen, color='lightcoral', alpha=0.35)
    ax_pse[idx_x, idx_y].plot(x, mean_ground, label='Ground truth', color='skyblue')
    ax_pse[idx_x, idx_y].plot(x, mean_gen, label='Generated', color='orange', linestyle='-.')
    ax_pse[idx_x, idx_y].legend()

    # MSE ahead

    mean = np.mean(MSE_AHEAD, 0)
    error = np.std(MSE_AHEAD, 0)
    x = np.arange(MSE_AHEAD.shape[1])
    ax_n_mse[idx_x, idx_y].fill_between(x, mean - error, mean + error, color='lightskyblue', alpha=0.35)
    ax_n_mse[idx_x, idx_y].plot(x, mean, color='black')

    # POIS ahead

    mean = np.mean(POIS_AHEAD, 0)
    error = np.std(POIS_AHEAD, 0)
    x = np.arange(POIS_AHEAD.shape[1])
    ax_n_pois[idx_x, idx_y].fill_between(x, mean - error, mean + error, color='lightskyblue', alpha=0.35)
    ax_n_pois[idx_x, idx_y].plot(x, mean, color='black')


Lat_Lay_split, only_params = get_params_loss(Params_loss_params, Params_loss_loss, Params_loss_lay, Params_loss_lat)

mean_params, mean_loss, error_loss = only_params

ax_loss_params_1.fill_between(mean_params, mean_loss - error_loss, mean_loss + error_loss, color='lightskyblue', alpha=0.35)
ax_loss_params_1.errorbar(mean_params, mean_loss, yerr=error_loss, ls='none')
ax_loss_params_1.scatter(mean_params, mean_loss, s=40)
ax_loss_params_1.set_xlabel("Number of Paramters")

mean_params_lay, mean_params_lat, mean_loss, error_loss = Lat_Lay_split

ax_loss_params[0].fill_between(mean_params_lay[0], mean_loss.mean(0) - error_loss.mean(0), mean_loss.mean(0) + error_loss.mean(0), color='lightskyblue', alpha=0.35)
ax_loss_params[0].errorbar(mean_params_lay[0], mean_loss.mean(0), yerr=error_loss.mean(0), ls='none')
ax_loss_params[0].scatter(mean_params_lay[0], mean_loss.mean(0), s=20)
ax_loss_params[0].set_xlabel("Number of Layers")
ax_loss_params[1].fill_between(mean_params_lat[0], mean_loss.T.mean(0) - error_loss.T.mean(0), mean_loss.T.mean(0) + error_loss.T.mean(0), color='lightskyblue', alpha=0.35)
ax_loss_params[1].errorbar(mean_params_lat[0], mean_loss.T.mean(0), yerr=error_loss.T.mean(0), ls='none')
ax_loss_params[1].scatter(mean_params_lat[0], mean_loss.T.mean(0), s=40)
ax_loss_params[1].set_xlabel("Number of Latent States")


fig_loss_params.suptitle('Number Parameters vs Loss')
fig_loss_params_1.suptitle('Number Parameters vs Loss')
fig_loss.suptitle('Loss')
fig_pse.suptitle('PSE')
fig_n_mse.suptitle('N_MSE')
fig_n_pois.suptitle('N_POIS')

fig_loss_params.savefig("Storage/Result/{}/Loss_Params_split".format(NAME_Trail))
fig_loss_params_1.savefig("Storage/Result/{}/Loss_Params".format(NAME_Trail))
fig_loss.savefig("Storage/Result/{}/Loss".format(NAME_Trail))
fig_pse.savefig("Storage/Result/{}/PSE".format(NAME_Trail))
fig_n_mse.savefig("Storage/Result/{}/N_MSE".format(NAME_Trail))
fig_n_pois.savefig("Storage/Result/{}/N_POIS".format(NAME_Trail))