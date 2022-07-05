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
y_min = 1000
y_max = 2000
mse_n = 100
pois_n = 75
pse_length = 67621

if not os.path.exists('Storage/Result/{}'.format(NAME_Trail)):
    os.mkdir('Storage/Result/{}'.format(NAME_Trail))

fig_loss_params, ax_loss_params = plt.subplots(figsize=(10, 10), dpi=100)
fig_loss, ax_loss = plt.subplots(4, 4, sharex=False, sharey=False, figsize=(15,15), dpi=200)
fig_pse, ax_pse = plt.subplots(4, 4, sharex=False, sharey=False, figsize=(15,15), dpi=200)
fig_n_mse, ax_n_mse = plt.subplots(4, 4, sharex=False, sharey=False, figsize=(15,15), dpi=200)
fig_n_pois, ax_n_pois = plt.subplots(4, 4, sharex=False, sharey=False, figsize=(15,15), dpi=200)
fig_klx, ax_klx = plt.subplots(4, 4, sharex=False, sharey=False, figsize=(15,15), dpi=200)

ax_loss[0, 0].set_ylabel("Layer Number 1")
ax_loss[1, 0].set_ylabel("Layer Number 2")
ax_loss[2, 0].set_ylabel("Layer Number 3")
ax_loss[3, 0].set_ylabel("Layer Number 4")
ax_loss[3, 0].set_xlabel("Latent dimension 2")
ax_loss[3, 1].set_xlabel("Latent dimension 4")
ax_loss[3, 2].set_xlabel("Latent dimension 8")
ax_loss[3, 3].set_xlabel("Latent dimension 16")

ax_loss_params.set_ylabel("Loss converged too")
ax_loss_params.set_xlabel("Number of parameters")

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

ax_klx[0, 0].set_ylabel("Layer Number 1")
ax_klx[1, 0].set_ylabel("Layer Number 2")
ax_klx[2, 0].set_ylabel("Layer Number 3")
ax_klx[3, 0].set_ylabel("Layer Number 4")
ax_klx[3, 0].set_xlabel("Latent dimension 2")
ax_klx[3, 1].set_xlabel("Latent dimension 4")
ax_klx[3, 2].set_xlabel("Latent dimension 8")
ax_klx[3, 3].set_xlabel("Latent dimension 16")

Params_loss = np.zeros((2, 16, 10))

for dirs_Main in os.listdir("Storage/Generated/{}".format(NAME_Trail)):
    numb_latent_dim, numb_layer_size_1 = find_number_statistical(dirs_Main)
    idx_x, idx_y = find_idx(numb_latent_dim, numb_layer_size_1)

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

        if data_generated.shape == data_ground_truth.shape:
            print(idx_x + idx_y * 4, idx)

            data_ground_truth = data_ground_truth[:, :, :np.size(data_generated, 2)]
            data_loss = pd.read_pickle('Storage/Generated/{}/{}/{}'.format(NAME_Trail, dirs_Main, name_loss))
            data_loss = np.nan_to_num(data_loss, 0)
            data_iters = pd.read_pickle('Storage/Generated/{}/{}/{}'.format(NAME_Trail, dirs_Main, name_iters))
            num_params = len(pd.read_pickle('Storage/Generated/{}/{}/{}'.format(NAME_Trail, dirs_Main, name_params)))

            # for plot of number of parameter vs Loss plot
            last_loss = data_loss[-1]
            Params_loss[0, idx_x + idx_y * 4, idx] = num_params
            Params_loss[1, idx_x + idx_y * 4, idx] = last_loss
            print(num_params, last_loss)

            units = np.size(data_ground_truth, 0)
            trials = np.size(data_ground_truth, 1)
            bins = np.size(data_ground_truth, 2)

            # loss
            y_min = np.min(np.concatenate((data_loss[data_loss!=0], np.array([y_min]))))
            if y_min < 1:
                y_min = 1
            y_max = np.max(np.concatenate((data_loss[10:], np.array([y_max]))))
            if y_max > 10**8:
                y_max = 10**8
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
            PSE_GROUND[idx, :pse_ground.shape[1], :pse_ground.shape[2]] = pse_ground
            pse_gen = get_average_spectrum(LFR_generated_conc)
            PSE_GEN[idx, :pse_gen.shape[1], :pse_gen.shape[2]] = pse_gen

            # mse
            MSE_AHEAD[idx] = n_MSE_Loss(LFR_ground_truth, LFR_generated, mse_n)

            # poisson loss n ahead
            POIS_AHEAD[idx] = n_poisson_Loss(LFR_ground_truth, LFR_generated, pois_n)

            # klx_gmm
            #Reduced_LFR_ground_truth = PCA_method(LFR_ground_truth_conc, 3)
            #Reduced_LFR_generated = PCA_method(LFR_generated_conc, 3)
            #print(idx_x, idx_y)
            #
            #x_gen = tc.from_numpy(Reduced_LFR_generated.T)
            #x_true = tc.from_numpy(Reduced_LFR_ground_truth.T)
            #p_gen, p_true = get_pdf_from_timeseries(x_gen, x_true, 30)
            #kl_value = kullback_leibler_divergence(p_true, p_gen)
            #if kl_value is None:
            #    kl_string = 'None'
            #else:
            #    kl_string = '{:.2f}'.format(kl_value)
            #ax_klx[idx_x, idx_y].set_title('KLx: {} | Difference Heatmap'.format(kl_string))
        else:
            print('Fuck')

    #Loss
    error = np.std(LOSS, 0)
    mean = np.mean(LOSS, 0)
    x = np.arange(LOSS.shape[1])
    ax_loss[idx_x, idx_y].fill_between(x, mean - error, mean + error, color='lightskyblue', alpha=0.35)
    ax_loss[idx_x, idx_y].plot(x, mean, color='black')
    ax_loss[idx_x, idx_y].set_yscale('log')
    ax_loss[idx_x, idx_y].set_ylim(y_min, y_max)
    ax_loss[idx_x, idx_y].vlines(data_iters, ymin=y_min, ymax=y_max, colors='r', linestyles='dotted')

    # PSE

    mean_ground = np.mean(PSE_GROUND, 0)
    mean_gen = np.mean(PSE_GEN, 0)
    error_ground = np.std(PSE_GROUND, 0)
    error_gen = np.std(PSE_GEN, 0)
    error_ground = error_ground[error_ground!=0]
    error_gen = error_gen[error_gen!=0]
    mean_ground = mean_ground[error_ground!=0]
    mean_gen = mean_gen[error_gen!=0]

    x = np.arange(PSE_GEN.shape[1])
    ax_pse[idx_x, idx_y].fill_between(x, mean_ground - error_ground, mean_ground + error_ground, color='lightskyblue', alpha=0.35)
    ax_pse[idx_x, idx_y].fill_between(x, mean_gen - error_gen, mean_gen + error_gen, color='lightcoral', alpha=0.35)
    ax_pse[idx_x, idx_y].plot(x, mean_ground, label='Ground truth', color='skyblue')
    ax_pse[idx_x, idx_y].plot(x, mean_gen, label='Generated', color='orange')
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


mean = np.mean(Params_loss[1, :, :], axis=1)
error = np.std(Params_loss[1, :, :], axis=1)
x = np.max(Params_loss[0, :, :], axis=1)
arg_sort = x.argsort()
x = x[arg_sort]
mean = mean[arg_sort]
error = error[arg_sort]

print(Params_loss[1, :, :])
print(Params_loss[0, :, :])
ax_loss_params.fill_between(x, mean - error, mean + error, color='lightskyblue', alpha=0.35)
ax_loss_params.plot(x[x!=0], mean[mean !=0], color='black')

fig_loss_params.suptitle('Number_Parameters_vs_Loss')
fig_loss.suptitle('Loss')
fig_pse.suptitle('PSE')
fig_n_mse.suptitle('N_MSE')
fig_n_pois.suptitle('N_POIS')
#fig_klx.suptitle('KLX')

fig_loss_params.savefig("Storage/Result/{}/Loss_Params".format(NAME_Trail))
fig_loss.savefig("Storage/Result/{}/Loss".format(NAME_Trail))
fig_pse.savefig("Storage/Result/{}/PSE".format(NAME_Trail))
fig_n_mse.savefig("Storage/Result/{}/N_MSE".format(NAME_Trail))
fig_n_pois.savefig("Storage/Result/{}/N_POIS".format(NAME_Trail))
#fig_klx.savefig("Storage/Result/{}/KLX".format(NAME_Trail))