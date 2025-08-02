import numpy as np

def tensions_function(t):
    max_tensions = np.array([6.0, 2.0, 2.0, 2.0])
    rate_hz = 0.1

    tensions_rate_hz = np.array([1.0 * rate_hz, 1.1 * rate_hz, 1.2 * rate_hz, 1.3 * rate_hz])
    tensions = 0.5 * (1.0 - np.cos(2 * np.pi * tensions_rate_hz * t))

    return tensions * max_tensions


def tip_force_function(t):
    max_magnitude = 0.1
    magnitude_rate_hz = 0.3
    direction_rate_hz = 0.07

    direction = np.array([
        np.sin(2 * np.pi * direction_rate_hz * 1.0 * t),
        np.sin(2 * np.pi * direction_rate_hz * 1.1 * t),
        np.sin(2 * np.pi * direction_rate_hz * 1.2 * t),
    ])

    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        direction = np.array([1.0, 0.0, 0.0])  # default direction if zero
    else:
        direction /= norm

    # Pulses from 0 to 1 
    magnitude = (0.5 * (1.0 - np.cos(2 * np.pi * magnitude_rate_hz * t))) ** 4

    return max_magnitude * magnitude * direction


class dist_load_function:
    def __init__(self, num_poses):
        self.framerate = 30
        self.force_rate_hz = 0.1
        self.s = np.linspace(0, 1, num_poses)
        self.steps_per_cycle = self.framerate / self.force_rate_hz

        self.mu = 0
        self.sigma = 1
        self.mag = 0
        self.angle = 0

        self.sigma_range = [0.05, 0.1]
        self.mag_range = [0.09, 0.1]
        self.mu_rage = [0.2, 1.0]
        self.angle_range = [0.0, 2 * np.pi]
        self.step = 0

    def sample_parameters(self):
        self.mu = np.random.uniform(self.mu_rage[0], self.mu_rage[1])
        self.sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        self.mag = np.random.uniform(self.mag_range[0], self.mag_range[1])
        self.angle = np.random.uniform(self.angle_range[0], self.angle_range[1])

    def update(self, t):
        if self.step % self.steps_per_cycle == 0:
            self.sample_parameters()
        
        mag = self.mag * (0.5 * (1.0 - np.cos(2 * np.pi * self.force_rate_hz * t))) ** 4
        gaussian = 1 / (np.sqrt(2 * np.pi * self.sigma**2)) * np.exp(- (self.s - self.mu)**2 / (2 * self.sigma**2))
        gaussian = gaussian / np.sum(gaussian)
        f_1d = mag * gaussian

        f_x = f_1d * np.cos(self.angle)
        f_y = f_1d * np.sin(self.angle)
        f_z = np.zeros_like(f_y)

        self.step = self.step + 1

        return np.column_stack((f_x, f_y, f_z))