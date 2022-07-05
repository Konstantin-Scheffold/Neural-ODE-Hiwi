from Methods import *
import pandas as pd
from Julia_ODE.bptt_master_evaluation.pse import *
import os
from Dimension_reduction.Methods import PCA_method
from Julia_ODE.bptt_master_evaluation.klx import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

NAME_Trail = 'Grid_Results_actual_1_Long_train'
dt = 200
y_min = 1000
y_max = 2000

if not os.path.exists('Storage/Result/{}'.format(NAME_Trail)):
    os.mkdir('Storage/Result/{}'.format(NAME_Trail))

for dirs_Main in os.listdir("Storage/Generated/{}".format(NAME_Trail)):

    fig_loss, ax_loss = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(15,15), dpi=200)
    fig_pse, ax_pse = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(15,15), dpi=200)
    fig_n_mse, ax_n_mse = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(15,15), dpi=200)
    fig_n_pois, ax_n_pois = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(15,15), dpi=200)
    fig_klx, ax_klx = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(15,15), dpi=200)


    ax_loss[0, 0].set_ylabel("Layer Size [8, 22]")
    ax_loss[1, 0].set_ylabel("Layer Size [16, 36]")
    ax_loss[2, 0].set_ylabel("Layer Size [24, 50]")
    ax_loss[3, 0].set_ylabel("Layer Size [32, 64]")
    ax_loss[3, 0].set_xlabel("Latent dimension 2")
    ax_loss[3, 1].set_xlabel("Latent dimension 4")
    ax_loss[3, 2].set_xlabel("Latent dimension 8")
    ax_loss[3, 3].set_xlabel("Latent dimension 16")

    ax_pse[0, 0].set_ylabel("Layer Size [8, 22]")
    ax_pse[1, 0].set_ylabel("Layer Size [16, 36]")
    ax_pse[2, 0].set_ylabel("Layer Size [24, 50]")
    ax_pse[3, 0].set_ylabel("Layer Size [32, 64]")
    ax_pse[3, 0].set_xlabel("Latent dimension 2")
    ax_pse[3, 1].set_xlabel("Latent dimension 4")
    ax_pse[3, 2].set_xlabel("Latent dimension 8")
    ax_pse[3, 3].set_xlabel("Latent dimension 16")

    ax_n_mse[0, 0].set_ylabel("Layer Size [8, 22]")
    ax_n_mse[1, 0].set_ylabel("Layer Size [16, 36]")
    ax_n_mse[2, 0].set_ylabel("Layer Size [24, 50]")
    ax_n_mse[3, 0].set_ylabel("Layer Size [32, 64]")
    ax_n_mse[3, 0].set_xlabel("Latent dimension 2")
    ax_n_mse[3, 1].set_xlabel("Latent dimension 4")
    ax_n_mse[3, 2].set_xlabel("Latent dimension 8")
    ax_n_mse[3, 3].set_xlabel("Latent dimension 16")

    ax_n_pois[0, 0].set_ylabel("Layer Size [8, 22]")
    ax_n_pois[1, 0].set_ylabel("Layer Size [16, 36]")
    ax_n_pois[2, 0].set_ylabel("Layer Size [24, 50]")
    ax_n_pois[3, 0].set_ylabel("Layer Size [32, 64]")
    ax_n_pois[3, 0].set_xlabel("Latent dimension 2")
    ax_n_pois[3, 1].set_xlabel("Latent dimension 4")
    ax_n_pois[3, 2].set_xlabel("Latent dimension 8")
    ax_n_pois[3, 3].set_xlabel("Latent dimension 16")

    ax_klx[0, 0].set_ylabel("Layer Size [8, 22]")
    ax_klx[1, 0].set_ylabel("Layer Size [16, 36]")
    ax_klx[2, 0].set_ylabel("Layer Size [24, 50]")
    ax_klx[3, 0].set_ylabel("Layer Size [32, 64]")
    ax_klx[3, 0].set_xlabel("Latent dimension 2")
    ax_klx[3, 1].set_xlabel("Latent dimension 4")
    ax_klx[3, 2].set_xlabel("Latent dimension 8")
    ax_klx[3, 3].set_xlabel("Latent dimension 16")

    for idx, dirs in enumerate(sorted(os.listdir("Storage/Generated/{}/{}".format(NAME_Trail, dirs_Main)))):

        numb_latent_dim, numb_layer_size_1, numb_layer_size_2 = find_number(dirs)
        idx_x, idx_y = find_idx(numb_latent_dim, numb_layer_size_1)

        name_ground_truth = '{}/Spike_test.pkl'.format(dirs)
        name_generated = '{}/output_9.pkl'.format(dirs)
        name_loss = '{}/Loss.pkl'.format(dirs)
        name_iters = '{}/iters.pkl'.format(dirs)

        data_ground_truth = pd.read_pickle('Storage/Generated/{}/{}/{}'.format(NAME_Trail, dirs_Main, name_ground_truth))  # units x trials x time bins
        data_generated = pd.read_pickle('Storage/Generated/{}/{}/{}'.format(NAME_Trail, dirs_Main, name_generated)) * 1  # units x trials x time bins

        if data_generated.shape != data_ground_truth.shape:
            print('Fuck')
            ax_loss[idx_x, idx_y].set_facecolor("lightcoral")
            ax_pse[idx_x, idx_y].set_facecolor("lightcoral")
            ax_n_mse[idx_x, idx_y].set_facecolor("lightcoral")
            ax_n_pois[idx_x, idx_y].set_facecolor("lightcoral")

        data_ground_truth = data_ground_truth[:, :, :np.size(data_generated, 2)]
        data_loss = pd.read_pickle('Storage/Generated/{}/{}/{}'.format(NAME_Trail, dirs_Main, name_loss))
        data_loss = np.nan_to_num(data_loss, 0)
        data_iters = pd.read_pickle('Storage/Generated/{}/{}/{}'.format(NAME_Trail, dirs_Main, name_iters))

        units = np.size(data_ground_truth, 0)
        trials = np.size(data_ground_truth, 1)
        bins = np.size(data_ground_truth, 2)

        # plot loss
        y_min = np.min(np.concatenate((data_loss[data_loss!=0], np.array([y_min]))))
        if y_min < 1:
            y_min = 1
        y_max = np.max(np.concatenate((data_loss[10:], np.array([y_max]))))
        if y_max > 10**6:
            y_max = 10**6

        ax_loss[idx_x, idx_y].plot(data_loss)
        ax_loss[idx_x, idx_y].set_yscale('log')
        ax_loss[idx_x, idx_y].set_ylim(y_min, y_max)
        ax_loss[idx_x, idx_y].vlines(data_iters, ymin=y_min, ymax=y_max, colors='r', linestyles='dotted')

        # pse analysis
        resolution_LFR = 700
        sigma_cut_off = 0
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
        ax_n_mse[idx_x, idx_y].plot(time[:20], n_MSE_Loss(LFR_ground_truth, LFR_generated, 20))

        # poisson loss n ahead
        ax_n_pois[idx_x, idx_y].plot(time[:15], n_poisson_Loss(LFR_ground_truth, LFR_generated, 15))

        # klx_gmm
        Reduced_LFR_ground_truth = PCA_method(LFR_ground_truth_conc, 3)
        Reduced_LFR_generated = PCA_method(LFR_generated_conc, 3)
        print(idx_x, idx_y)
        x_gen = tc.from_numpy(Reduced_LFR_generated.T)
        x_true = tc.from_numpy(Reduced_LFR_ground_truth.T)
        p_gen, p_true = get_pdf_from_timeseries(x_gen, x_true, 30)
        kl_value = kullback_leibler_divergence(p_true, p_gen)
        if p_true is not None and p_gen is not None:
            p_true = marginalize_pdf(p_true, except_dims=(0, 2))
            p_gen = marginalize_pdf(p_gen, except_dims=(0, 2))
            if kl_value is None:
                kl_string = 'None'
            else:
                kl_string = '{:.2f}'.format(kl_value)

            ax_klx[idx_x, idx_y].set_title('KLx: {} | Difference Heatmap'.format(kl_string))
            sns.heatmap(p_gen.numpy().T[::-1]-p_true.numpy().T[::-1], ax=ax_klx[idx_x, idx_y])

    fig_loss.suptitle('Loss_{}'.format(dirs_Main))
    fig_pse.suptitle('PSE_{}'.format(dirs_Main))
    fig_n_mse.suptitle('N_MSE_{}'.format(dirs_Main))
    fig_n_pois.suptitle('N_POIS_{}'.format(dirs_Main))
    fig_klx.suptitle('KLX_{}'.format(dirs_Main))

    fig_loss.savefig("Storage/Result/{}/Loss_{}".format(NAME_Trail, dirs_Main))
    fig_pse.savefig("Storage/Result/{}/PSE_{}".format(NAME_Trail, dirs_Main))
    fig_n_mse.savefig("Storage/Result/{}/N_MSE_{}".format(NAME_Trail, dirs_Main))
    fig_n_pois.savefig("Storage/Result/{}/N_POIS_{}".format(NAME_Trail, dirs_Main))
    fig_klx.savefig("Storage/Result/{}/KLX_{}".format(NAME_Trail, dirs_Main))