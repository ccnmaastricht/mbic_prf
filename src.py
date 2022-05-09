import numpy as np
import nibabel as nib
from scipy.fft import fft, ifft
from scipy.signal import detrend, filtfilt, butter
from cni_tlbx.gadgets import two_gamma, gaussian

def simulate_data(ground_truth, stimulus, stimulus_duration, sampling_frequency):
    mask = ground_truth['masks']['all']
    num_voxels = sum(mask)
    resolution, _, timepoints = stimulus.shape
    num_pixels = resolution**2

    len_hrf = int(34 * sampling_frequency)
    time_vector = np.linspace(0, 34 + stimulus_duration,
                                  timepoints + len_hrf)
    hrf_fft = fft(two_gamma(time_vector))

    r = np.linspace(-8.4, 8.4, resolution)
    x_coordinates, y_coordinates = np.meshgrid(r,r)
    x_coordinates = x_coordinates.flatten()
    y_coordinates = -y_coordinates.flatten()

    W = np.zeros((num_voxels, num_pixels))
    mu_x = ground_truth['mu_x'][mask]
    mu_y = ground_truth['mu_y'][mask]
    sigma = ground_truth['sigma'][mask]
    for v in range(num_voxels):
        W[v, :] = gaussian(mu_x[v], mu_y[v], sigma[v], x_coordinates, y_coordinates)

    tc_fft = np.matmul(W, stimulus.reshape(num_pixels, timepoints)).transpose()
    tc_fft = fft(np.append(tc_fft, np.zeros((len_hrf, num_voxels)), axis=0), axis=0)

    data =  np.real(ifft(tc_fft * np.expand_dims(hrf_fft, axis=1), axis=0))

    variance = np.var(data, axis=0)
    peak = np.sqrt(2 * variance)
    breathing = 0.5 * peak * \
                np.cos(np.pi * 2 * np.expand_dims(time_vector, axis=1) *
                0.3 + np.random.random(num_voxels) * np.pi)
    heart_beat = 0.5 * peak * \
                 np.cos(np.pi * 2 * np.expand_dims(time_vector, axis=1) *
                 1 + np.random.random(num_voxels) * np.pi)
    drift = peak * \
            np.cos(np.pi * 2 * np.expand_dims(time_vector, axis=1) *
            np.random.random(num_voxels) * 1e-3 + \
            np.random.random(num_voxels) * np.pi)

    data += breathing + heart_beat + drift

    return data[:timepoints], W


def preprocess(data, sampling_frequency):
    b, a = butter(2, 0.005,'hp', fs = sampling_frequency)
    return filtfilt(b, a, data, axis=0)
