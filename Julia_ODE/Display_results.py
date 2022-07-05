from Methods import *
import pandas as pd
import os
from Julia_ODE.bptt_master_evaluation.pse import *
from Julia_ODE.bptt_master_evaluation.klx_gmm import *
from Julia_ODE.bptt_master_evaluation.klx import *
import seaborn as sns
from Dimension_reduction.Methods import PCA_method
sns.set_theme()

'''
Hyperparameter Tuning:
-     loss = ( sum(dt.*λ_hat .- Nlogλ) .+ kld ) ./ N vorzeichen falsch, siehe paper
- <.  führt zu point process anstatt count process, soll ich das ändern
- point vs count process, soll ich ground truth in point process umwandeln oder generated in count process
- falls zweiteres, soll ich den output mit ner activation funciton in den richtigen bereich [0, max(ground_truth)] zwingen?
haben die nicht gemacht aber ist üblich

- 1. normal
- 2. more iterations
- 3. Bins_50 -> Bins_200 + increases latent dimension in optim.jl
'''

name_Trial = 'Test'

name_ground_truth = 'Good_Results/Test/Results_Custom_plnde_1__3/Spike_test.pkl'
name_generated = 'Good_Results/Test/Results_Custom_plnde_1__3/output_9.pkl'
name_loss = 'Good_Results/Test/Results_Custom_plnde_1__3/Loss_8.pkl'
name_iters = 'Good_Results/Test/Results_Custom_plnde_1__3/iters_8.pkl'

data_ground_truth = pd.read_pickle('Storage/Generated/{}'.format(name_ground_truth)) # units x trials x time bins
data_generated = pd.read_pickle('Storage/Generated/{}'.format(name_generated))*1 # units x trials x time bins
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
plt.figure(figsize=(12, 12))
plt.plot(data_loss)
plt.yscale('log')
plt.ylim(0, y_max)
plt.vlines(data_iters, ymin=y_min, ymax=y_max, colors='r', linestyles='dotted')
plt.show()

# plot example trials
plt.figure(figsize=(10,20))

BINS = np.ones((units, bins))
BINS[:, :] = np.linspace(0,21000, bins)


for i in range(3):
    rand_trial = np.random.randint(0, trials, 1)[0]

    data_gen = data_generated[:, rand_trial, :]
    data_ground = data_ground_truth[:, rand_trial, :]
    bins_num_gen = np.size(data_generated, 2)

    bins_ground_truth = BINS * (data_ground!=0)
    bins_generated = BINS[:, :bins_num_gen] * (data_gen[:, :bins_num_gen]!=0)

    plt.subplot(3, 2, 2*i+1)
    plt.title('Ground Truth')
    plt.eventplot(bins_ground_truth, linelengths = 0.8)
    plt.subplot(3, 2, 2*i+2)
    plt.title('Generated')
    plt.eventplot(bins_generated, linelengths=0.8)

plt.show()

# pse analysis

resolution_LFR = 1000
sigma_cut_off = 0
width = dt * 5

time_points = np.arange(0, 21000, dt)[:bins]
STMtx_ground_Truth = np.zeros((units, trials, bins))
STMtx_generated = np.zeros((units, trials, bins))
STMtx_ground_Truth[:, :] = time_points
STMtx_generated[:, :] = time_points

STMtx_ground_Truth *= (data_ground_truth>0)*1
STMtx_generated *= (data_generated>0)*1

LFR_ground_truth = np.zeros((units, trials, resolution_LFR))
LFR_generated = np.zeros((units, trials, resolution_LFR))

for i in range(trials):
    LFR_ground_truth[:, i, :], time = get_LFR(STMtx_ground_Truth[:, i, :], data_ground_truth[:, i, :], sigma_cut_off=sigma_cut_off,
                                           width_small=width, resolution_LFR=resolution_LFR)

    LFR_generated[:, i, :], time = get_LFR(STMtx_generated[:, i, :], data_generated[:, i, :], sigma_cut_off=sigma_cut_off,
                                        width_small=width, resolution_LFR=resolution_LFR)

LFR_generated_conc = np.concatenate((LFR_generated[:, 0, :], LFR_generated[:, 1, :]), axis=1)
for i in range(1, np.size(LFR_generated, 1)-1):
    LFR_generated_conc = np.concatenate((LFR_generated_conc, LFR_generated[:, i, :]), axis=1)


LFR_ground_truth_conc = np.concatenate((LFR_ground_truth[:, 0, :], LFR_ground_truth[:, 1, :]), axis=1)
for i in range(1, np.size(LFR_ground_truth, 1)-1):
    LFR_ground_truth_conc = np.concatenate((LFR_ground_truth_conc, LFR_ground_truth[:, i, :]), axis=1)

Spectrum_gound_truth = get_average_spectrum(LFR_ground_truth_conc)
Spectrum_generated = get_average_spectrum(LFR_generated_conc)

plt.plot(Spectrum_gound_truth, label='ground truth')
plt.plot(Spectrum_generated, label='generated')
plt.legend()
plt.show()

# mse

def MSE_Loss(x_true, x_gen):
    return np.mean((x_gen - x_true)**2)

def n_MSE_Loss(x_true, x_gen, n):
    loss_n_ahead = np.zeros(n)
    for i in range(1, n):
        loss_n_ahead[i] = MSE_Loss(x_true[:, :i], x_gen[:, :i])
    return loss_n_ahead

plt.figure(figsize=(12,12))
plt.plot(n_MSE_Loss(LFR_ground_truth, LFR_generated, 100))
plt.show()

# poisson loss n ahead

def poisson_Loss(x_true, x_gen):
    x_gen = x_gen.flatten()
    x_true = x_true.flatten()
    x_true = x_true[x_gen!=0]
    x_gen = x_gen[x_gen!=0]
    return np.mean(x_gen - x_true * np.log(x_gen))

def n_poisson_Loss(x_true, x_gen, n):
    loss_n_ahead = np.zeros(n)
    for i in range(1, n):
        loss_n_ahead[i] = poisson_Loss(x_true[:, :i], x_gen[:, :i])
    return loss_n_ahead

plt.figure(figsize=(12,12))
plt.plot(n_poisson_Loss(LFR_ground_truth, LFR_generated, 100))
plt.show()

Reduced_LFR_ground_truth = PCA_method(LFR_ground_truth_conc, 3)
Reduced_LFR_generated = PCA_method(LFR_generated_conc, 3)
# klx_gmm
plot_kl(Reduced_LFR_generated, Reduced_LFR_ground_truth, 50)

# klx

#klx_gmm = calc_kl_from_data(Reduced_LFR_generated[:, :35000], Reduced_LFR_ground_truth[:, :35000])
