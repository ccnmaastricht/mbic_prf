import numpy as np
from scipy.fft import fft, ifft
from cni_tlbx.gadgets import two_gamma, gaussian

def simulate_data(ground_truth, stimulus, stimulus_duration, sampling_frequency):
    good_voxels = sum(ground_truth['mask'])
    total_voxels = len(ground_truth['mask'])
    resolution, _, timepoints = stimulus.shape
    num_pixels = resolution**2

    len_hrf = int(34 * sampling_frequency)
    time_vector = np.linspace(0, 34 + stimulus_duration,
                              timepoints + len_hrf)
    hrf_fft = fft(two_gamma(time_vector))

    r = np.linspace(-4.2, 4.2, resolution)
    x_coordinates, y_coordinates = np.meshgrid(r,r)
    x_coordinates = x_coordinates.flatten()
    y_coordinates = -y_coordinates.flatten()

    W = np.zeros((good_voxels, num_pixels))
    mu_x = ground_truth['mu_x'][ground_truth['mask']]
    mu_y = ground_truth['mu_y'][ground_truth['mask']]
    sigma = ground_truth['sigma'][ground_truth['mask']]
    for v in range(good_voxels):
        W[v, :] = gaussian(mu_x[v], mu_y[v], sigma[v], x_coordinates, y_coordinates)

    tc_fft = np.matmul(W, stimulus.reshape(num_pixels, timepoints)).transpose()
    tc_fft = fft(np.append(tc_fft, np.zeros((len_hrf, good_voxels)), axis=0), axis=0)

    data = np.zeros((timepoints, total_voxels))
    data[:, ground_truth['mask']] = np.real(ifft(tc_fft *
                                            np.expand_dims(hrf_fft,
                                            axis=1), axis=0))[:timepoints]

    return data, W


def load_GT()
    ground_truth = dict()

    img = nib.load('sample_data/lh.prfangle.mgz')
    pa = img.get_fdata().flatten()
    pa[pa>180] -= 360
    pa = pa / 180 * np.pi
    ground_truth['polar_angle'] = pa

    img = nib.load('sample_data/lh.prfeccentricity.mgz')
    ground_truth['eccentricity'] = img.get_fdata().flatten()

    img = nib.load('sample_data/lh.prfsize.mgz')
    ground_truth['sigma'] = img.get_fdata().flatten()

    ground_truth['mu_x'] = np.cos(ground_truth['polar_angle']) * ground_truth['eccentricity']
    ground_truth['mu_y'] = np.sin(ground_truth['polar_angle']) * ground_truth['eccentricity']

    img = nib.load('sample_data/lh.prf-visualrois.mgz')
    rois = img.get_fdata().flatten()
    mask = (rois>1) & (rois<=6)

    img = nib.load('sample_data/lh.prfR2.mgz')
    r2 = img.get_fdata().flatten() / 100
    ground_truth['mask'] = (r2 > 0.15) & (ground_truth['eccentricity'] <= 8.4) & mask


    # prepare also example "voxels"
    example_voxels = {'mu_x': np.array([0.5, -3]),
                  'mu_y': np.array([0.5, -3]),
                  'sigma': np.array([0.5, 2.5]),
                  'mask': np.array([1, 1]).astype(bool)}

    example_voxels['polar_angle'] = np.angle(example_voxels['mu_x'] + example_voxels['mu_x'] * 1j)
    example_voxels['eccentricity'] = np.abs(example_voxels['mu_x'] + example_voxels['mu_x'] * 1j)

    return ground_truth, example_voxels
