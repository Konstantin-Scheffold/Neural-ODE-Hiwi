
import numpy as np
import matplotlib.pyplot as plt
from Methods import PCA_method
from Methods import LLE_method
from Methods import MDS_method
from Methods import ISO_method
from Methods import TSNE_method
from Methods import UMAP_method
from Methods import SPEC_EMB_method
from Methods import dPCA_method
import pandas as pd
from Data_Handling import get_LFR, visualise
import seaborn as sns

sns.set(style='white', context='poster', rc={'figure.figsize':(14,10)})


data_1 = pd.read_pickle(r'C:\Users\Konra\PycharmProjects\Neural_ODE\Dimension reduction\Data\ot_collection_07.mat')
data_2 = pd.read_pickle(r'C:\Users\Konra\PycharmProjects\Neural_ODE\Dimension reduction\Data\ot_collection_10.mat')
data_3 = pd.read_pickle(r'C:\Users\Konra\PycharmProjects\Neural_ODE\Dimension reduction\Data\ot_collection_20.mat')
data_4 = pd.read_pickle(r'C:\Users\Konra\PycharmProjects\Neural_ODE\Dimension reduction\Data\ot_JG18_190828.mat')

new_data_1 = np.concatenate((data_1[0], data_1[1]), axis=1)
new_data_2 = np.concatenate((data_2[0], data_2[1]), axis=1)
new_data_3 = np.concatenate((data_3[0], data_3[1]), axis=1)
new_data_4 = np.concatenate((data_4[0], data_4[1]), axis=1)

for i in range(len(data_1)-1):
    new_data_1 = np.concatenate((new_data_1, data_1[i+1]), axis=1)
for i in range(len(data_2)-1):
    new_data_2 = np.concatenate((new_data_2, data_2[i+1]), axis=1)
for i in range(len(data_3)-1):
    new_data_3 = np.concatenate((new_data_3, data_3[i+1]), axis=1)
for i in range(len(data_4)-1):
    new_data_4 = np.concatenate((new_data_4, data_4[i+1]), axis=1)

reduced_data = dPCA_method(new_data_1[:,:100], 3)
visualise(reduced_data, new_data_1[:,:100], 3, 'dPCA')

'''
data_1 = np.sort(new_data_1, axis=1)
data_2 = np.sort(new_data_2, axis=1)
data_3 = np.sort(new_data_3, axis=1)
data_4 = np.sort(new_data_4, axis=1)

data_1 = get_LFR(data_1, resolution_LFR = 500)
data_2 = get_LFR(data_2, resolution_LFR = 500)
#data_3 = get_LFR(data_3, resolution_LFR = 500)
#data_4 = get_LFR(data_4, resolution_LFR = 500)
'''

'''collection_methods = [ISO_method, TSNE_method, SPEC_EMB_method, UMAP_method]
collection_methods_names = ['PCA_method', 'LLE_method', 'MDS_method', 'ISO_method', 'TSNE_method', 'SPEC_EMB_method', 'UMAP_method']

for index, method in enumerate(collection_methods):
    for data_index, Spike_trains in enumerate([new_data_1, new_data_2, new_data_3, new_data_4]):
        print(collection_methods_names[index], ' data set:', data_index)
        reduced_data = method(Spike_trains, 3)
        visualise(reduced_data, Spike_trains, 3, '{}'.format(collection_methods_names[index]))
        np.save(r'results/reduced_data/{}_data{}'
                r''.format(collection_methods_names[index], data_index), reduced_data)'''