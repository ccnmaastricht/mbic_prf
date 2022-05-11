import numpy as np
import nibabel as nib
from PIL import Image
from PIL import ImageDraw
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


# simple code for creating wedge aperture (just for testing stuff. @Alex, I assume you will provide something more fancy)
def create_wedge(direction, duty_cycle, temporal_frequency,
                 stimulus_duration, time_steps, resolution=100):

  if direction == 'clockwise':
    sign = 1
  elif direction == 'counterclockwise':
    sign = -1

  r = np.linspace(-1, 1, resolution)
  X, Y = np.meshgrid(r, r)

  angle = np.angle(X - Y * 1j)
  radius = np.abs(X - Y * 1j)

  stimulus = np.zeros((resolution, resolution, time_steps))
  time_vec = np.linspace(0, stimulus_duration, time_steps)

  phase_offset = np.angle(np.exp(2j * np.pi * time_vec * temporal_frequency))
  phase_offset[phase_offset < 0] += 2 * np.pi

  for t in range(time_steps):
    new_angle = np.angle(np.exp((angle + sign* phase_offset[t]) * 1j))
    stimulus[:, :, t] = (radius <= 1) & (np.abs(new_angle) <= (duty_cycle * np.pi))

  return stimulus


def create_ring(direction, duty_cycle, temporal_frequency,
                stimulus_duration, time_steps, resolution=100):

  r = np.linspace(-1, 1, resolution)
  X, Y = np.meshgrid(r, r)

  radius = np.abs(X - Y * 1j)

  stimulus = np.zeros((resolution, resolution, time_steps))
  time_vec = np.linspace(0, stimulus_duration, time_steps)

  steps = (1 - duty_cycle) * time_vec * temporal_frequency

  lower_radius = steps % (1 - duty_cycle)
  upper_radius = lower_radius + duty_cycle

  if direction == 'contracting':
    lower_radius = lower_radius[::-1]
    upper_radius = upper_radius[::-1]

  for t in range(time_steps):
    stimulus[:, :, t] = ((lower_radius[t] <= radius) &
                         (radius <= upper_radius[t]))
  return stimulus


def create_bar(duty_cycle, time_steps, mean_luminance, positions, resolution=100):
  def define_bar(x, y, width, height, angle):
    img = Image.fromarray(np.zeros((resolution,
                                    resolution)))
    draw = ImageDraw.Draw(img)

    rect = np.array([(-width / 2, -height / 2),
                     (width / 2, -height / 2),
                     (width / 2, height / 2),
                     (-width / 2, height / 2),
                     (-width / 2, -height / 2)])

    theta = (np.pi / 180.0) * angle

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    bar = np.dot(rect, R) + np.array([x, y])

    draw.polygon([tuple(p) for p in bar], fill=1)

    return np.asarray(img)

  def rotate_coords(point, angle, origin=(50, 50)):
    angle = np.radians(angle)

    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)

    return qx, resolution - qy

  r = np.linspace(-1, 1, resolution)
  X, Y = np.meshgrid(r, r)

  radius = np.abs(X - Y * 1j)

  steps = int(duty_cycle**-1)
  width = duty_cycle * 100

  reps = int(time_steps / 8 / steps)

  oris = [0, 45, 90, 135, 180, 225, 270, 315]

  vals = np.linspace(width / 2, resolution - width / 2, steps, dtype=int)

  coords = [(x, 50) for x in vals]

  bars = []

  for rep in range(reps):
    for angle in oris:
      new_coords = [rotate_coords(coord, angle)
                    for coord in coords]

      if positions == "random":
        np.random.shuffle(new_coords)
    
      rand_pos = np.random.randint(len(new_coords))
    
      for idx, (x, y) in enumerate(new_coords):
        if (idx == rand_pos) & mean_luminance:
          bar = np.zeros((resolution, resolution))
        else:
          bar = define_bar(x, y, width, 150, angle)
            
        bars.append(bar * (radius <= 1))

  stimulus = np.stack(bars, -1)

  return stimulus
