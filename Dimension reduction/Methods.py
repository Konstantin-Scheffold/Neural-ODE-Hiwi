
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE
from dPCA import dPCA
import umap


def PCA_method(data, component_number):
    if data.shape[0] < data.shape[1]:
        data = np.transpose(data)

    pca_method = PCA(n_components=component_number)
    data_reduced = pca_method.fit_transform(data)

    return data_reduced

def LLE_method(data, component_number):
    if data.shape[0] < data.shape[1]:
        data = np.transpose(data)

    lle_method = LocallyLinearEmbedding(n_components=component_number)
    data_reduced = lle_method.fit_transform(data)

    return data_reduced

def MDS_method(data, component_number):
    if data.shape[0] < data.shape[1]:
        data = np.transpose(data)

    mds_method = MDS(n_components=component_number)
    data_reduced = mds_method.fit_transform(data)

    return data_reduced

def ISO_method(data, component_number):
    if data.shape[0] < data.shape[1]:
        data = np.transpose(data)

    iso_method = Isomap(n_components=component_number)
    data_reduced = iso_method.fit_transform(data)

    return data_reduced

def ISO_method(data, component_number):
    if data.shape[0] < data.shape[1]:
        data = np.transpose(data)

    iso_method = Isomap(n_components=component_number)
    data_reduced = iso_method.fit_transform(data)

    return data_reduced

def SPEC_EMB_method(data, component_number):
    if data.shape[0] < data.shape[1]:
        data = np.transpose(data)

    spec_emb_method = SpectralEmbedding(n_components=component_number)
    data_reduced = spec_emb_method.fit_transform(data)

    return data_reduced

def TSNE_method(data, component_number):
    if data.shape[0] < data.shape[1]:
        data = np.transpose(data)

    tsne_method = TSNE(n_components=component_number)
    data_reduced = tsne_method.fit_transform(data)

    return data_reduced

def UMAP_method(data, component_number):
    num_units = np.min(np.shape(data))
    num_time_points = np.max(np.shape(data))
    if data.shape[0] < data.shape[1]:
        data = np.transpose(data)

    umap_method = umap.UMAP(n_neighbors=np.max([int(num_units/10), 2]),
                            min_dist=0.1,
                            n_components=component_number,
                            metric='euclidean')
    data_reduced = umap_method.fit_transform(data)

    return data_reduced

def dPCA_method(data, component_number):

    if data.shape[0] > data.shape[1]:
        data = np.transpose(data)

    data = data[None, :, :]

    mean_data = np.mean(data, 0)
    mean_data -= np.mean(mean_data)

    dpca_method = dPCA.dPCA('t', component_number, regularizer='auto')
    dpca_method.protect = ['t']
    data_reduced = dpca_method.fit_transform(mean_data, data)

    return data_reduced