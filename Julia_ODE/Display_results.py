
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.fft import rfft
sns.set_theme()

'''
Hyperparameter Tuning:
- grid search combined with Hyperparameter fitting?
- what quality measure
- implement model in python or measure in julia?
- influence of time length of integrator of Neural ODE

- different trials get handeled just as a different time bin
- why input of Encoder equal to size of Latent space and input forced to this form?
    u0_m = reshape(p[end-L*N-L*N+1:end-L*N], L, :)
    u0_s = clamp.(reshape(p[end-L*N+1:end], L, :), -1e8, 0)
    u0s = u0_m .+ exp.(u0_s) .* randn(size(u0_s))
Da fuq?

'''

name_Trial = 'First_try'
name_generated = 'opt_output_short.pkl'
name_ground_truth = 'Pickle_LFR_data_1.pkl'

if not os.path.exists('Storage/Result/{}'.format(name_Trial)):
    os.mkdir('Storage/Result/{}'.format(name_Trial))

data_ground_truth = pd.read_pickle('Storage/Data/{}'.format(name_ground_truth)) # units x trials x time bins
data_generated = pd.read_pickle('Storage/Generated/{}'.format(name_generated)) # units x trials x time bins

units = np.size(data_generated, 0)
trials = np.size(data_generated, 1)
bins = np.size(data_generated, 2)

# power Spectrum

fourier_transform_ground_truth = np.zeros((units, trials, int(bins/2) + 1 if bins % 2 == 0 else int((bins+1)/2)))
fourier_transform_generated = np.zeros((units, trials, int(bins/2) + 1 if bins % 2 == 0 else int((bins+1)/2)))

fourier_transform_generated = rfft(data_generated)
fourier_transform_ground_truth = rfft(data_ground_truth)

# calculated absolut square of complex numbers and calc mean across units and trials
power_spectrum_generated = np.mean(np.mean(np.abs(fourier_transform_generated)**2, 1), 0)
power_spectrum_ground_truth = np.mean(np.mean(np.abs(fourier_transform_ground_truth)**2, 1), 0)

# find scale

'''
variance = np.zeros(201)
for idx, x in enumerate(range(0, bins, int(bins/200))):
    var_truth = np.std(power_spectrum_ground_truth[x:x+int(bins/200)])
    var_gen = np.std(power_spectrum_generated[x:x+int(bins/200)])
    variance[idx] = np.max([var_gen, var_truth])
'''

cutoff = 500 #np.min(np.argwhere((variance<10**3)*1))

power_spectrum_generated = power_spectrum_generated[:cutoff]
power_spectrum_ground_truth = power_spectrum_ground_truth[:cutoff]

cut = 2
ymin= np.min([np.min(power_spectrum_generated), np.min(power_spectrum_ground_truth)])
ymax= np.max([np.max(power_spectrum_generated)/cut, np.max(power_spectrum_ground_truth)/cut])


plt.figure(figsize=(15,15))
#plt.plot(power_spectrum_ground_truth)
plt.plot(power_spectrum_generated)
plt.ylim(ymin, ymax)
plt.xlim(0, cutoff)
plt.yscale('log')
plt.show()
#plt.savefig('Storage/Result/{}/Power_Spectrum'.format(name_Trial))



# example trials
plt.figure(figsize=(15,15))
rand_unit = np.random.randint(0, units, 6)
rand_trial = np.random.randint(0, trials, 6)
rand_bins = np.random.randint(0, bins-3000, 6)


for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.plot(data_generated[rand_unit[i], rand_trial[i], rand_bins[i]:rand_bins[i]+3000])
    plt.plot(data_ground_truth[rand_unit[i], rand_trial[i], rand_bins[i]:rand_bins[i]+3000])

plt.savefig('Storage/Result/{}/example_Trials'.format(name_Trial))
plt.show()
