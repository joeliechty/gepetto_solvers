import os
from collections import deque

import numpy as np
import matplotlib.pyplot as plt


def tensions_function(t):
    max_tensions = np.array([6.0, 2.0, 2.0, 2.0])
    rate_hz = 0.1

    tensions_rate_hz = np.array([1.0 * rate_hz, 1.1 * rate_hz, 1.2 * rate_hz, 1.3 * rate_hz])
    tensions = 0.5 * (1.0 - np.cos(2 * np.pi * tensions_rate_hz * t))

    return tensions * max_tensions


def pulse_function(t, rate_hz):
    return (0.5 * (1.0 - np.cos(2 * np.pi * rate_hz * t))) ** 3


class TipForceFunction:
    def __init__(self, max_magnitude=0.1, force_rate_hz=0.3, framerate=30):
        self.max_magnitude = max_magnitude
        self.force_rate_hz = force_rate_hz
        self.steps_per_cycle = framerate / force_rate_hz

        self.magnitude = 0
        self.direction = np.ones(3)

        self.step = 0

    def sample_parameters(self):
        d = np.random.randn(3)
        self.direction = d / np.linalg.norm(d)

    def __call__(self, t):
        if self.step % self.steps_per_cycle == 0:
            self.sample_parameters()
        
        pulse_scale = pulse_function(t, self.force_rate_hz)

        self.step = self.step + 1

        return pulse_scale * self.max_magnitude * self.direction 
    

class TipForceFunction:
    def __init__(self, max_magnitude=0.1, force_rate_hz=0.3, framerate=30, seed=None):
        self.max_magnitude = max_magnitude
        self.force_rate_hz = force_rate_hz
        self.steps_per_cycle = int(framerate / force_rate_hz)

        self.magnitude = 0
        self.direction = np.ones(3)

        self.step = 0

        self.rng = np.random.default_rng(seed)

    def sample_parameters(self):
        d = self.rng.normal(size=3)
        self.direction = d / np.linalg.norm(d)

    def __call__(self, t):
        if self.step % self.steps_per_cycle == 0:
            self.sample_parameters()

        pulse_scale = pulse_function(t, self.force_rate_hz)

        self.step += 1

        return pulse_scale * self.max_magnitude * self.direction
    

class BumpFunction:
    def __init__(self, num_forces):
        self.framerate = 30
        self.force_rate_hz = 0.1
        self.s = np.linspace(0, 1, num_forces)
        self.steps_per_cycle = self.framerate / self.force_rate_hz

        self.mu = 0
        self.sigma = 1
        self.mag = 0
        self.angle = 0

        self.sigma_range = [0.05, 0.1]
        self.mag_range = [0.02, 0.1]
        self.mu_rage = [0.2, 0.9]
        self.angle_range = [0.0, 2 * np.pi]
        self.step = 0

    def sample_parameters(self):
        self.mu = np.random.uniform(self.mu_rage[0], self.mu_rage[1])
        self.sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        self.mag = np.random.uniform(self.mag_range[0], self.mag_range[1])
        self.angle = np.random.uniform(self.angle_range[0], self.angle_range[1])

    def __call__(self, t):
        if self.step % self.steps_per_cycle == 0:
            self.sample_parameters()
        
        mag_t = (0.5 * (1.0 - np.cos(2 * np.pi * self.force_rate_hz * t))) ** 4

        self.step = self.step + 1

        return mag_t * self.get_max_forces()
    
    def get_max_forces(self):
        bump = 1 / (np.sqrt(2 * np.pi * self.sigma**2)) * np.exp(- (self.s - self.mu)**2 / (2 * self.sigma**2))
        f_1d = self.mag * bump / np.sum(bump)

        f_x = f_1d * np.cos(self.angle)
        f_y = f_1d * np.sin(self.angle)
        f_z = np.zeros_like(f_y)

        return np.column_stack((f_x, f_y, f_z))


class DistLoadFunction:
    def __init__(self, num_forces, num_bumps=3):
        self.bump_functions = [BumpFunction(num_forces) for i in range(num_bumps)]
    
    def __call__(self, t):
        bumps_eval = [bump(t) for bump in self.bump_functions]
        return np.array(bumps_eval).sum(axis=0)

    def get_max_forces(self):
        max_bumps = [bump.get_max_forces() for bump in self.bump_functions]
        return np.array(max_bumps).sum(axis=0)
    
    def sample_parameters(self):
        [bump.sample_parameters() for bump in self.bump_functions]
    

class moving_savgol:
    def __init__(self, window_size=15, poly_order=2):
        self.buffer = deque(maxlen=window_size)
        self.poly_order = poly_order
        self.window_size = window_size

    def update(self, new_value):
        new_array = np.asarray(new_value)
        self.buffer.append(new_array)

        if len(self.buffer) < self.window_size:
            return new_array

        data = np.stack(self.buffer, axis=0)
        t = np.arange(self.window_size)

        original_shape = data.shape[1:]
        data_flat = data.reshape(self.window_size, -1)

        coeffs = np.polyfit(t, data_flat, self.poly_order)
        last_val = np.polyval(coeffs, t[-1])

        # Reshape back to original shape
        return last_val.reshape(original_shape)


def setup_plt(width=3.5, height=5.0, grid=False):

    os.makedirs("figures", exist_ok=True)

    plt.rcParams.update({
        "figure.figsize": (width, height),
        "font.family": "STIXGeneral",
        "font.size": 8,         
        # "axes.labelsize": 7,         
        # "axes.titlesize": 7,        
        "xtick.labelsize": 7,        
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": grid,
        "grid.alpha": 0.3,
        "pdf.fonttype": 42,  # embed fonts in PDF
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",  # math text compatible with Times
        "mathtext.rm": "stix"
    })


if __name__ == "__main__":
    config = get_simulation_config()

    num_poses = config.num_discs + (config.num_discs - 1) * config.poses_between_discs
    dist_load = dist_load_function(num_poses - 1)

    all_jerks = []
    for i in range(1000000):
        dist_load.sample_parameters()
        forces = dist_load.get_max_forces()
        jerk = np.diff(forces, n=3, axis=0)
        all_jerks.append(jerk[:, :2].reshape(-1))

    all_jerks = np.concatenate(all_jerks)

    # Keep only the top 5% biggest jerks by magnitude
    cutoff = np.percentile(np.abs(all_jerks), 95)
    big_jerks = all_jerks[np.abs(all_jerks) >= cutoff]
    
    print(big_jerks.std())        
        