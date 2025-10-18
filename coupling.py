import numpy as np
from scipy.constants import c, epsilon_0, pi
from scipy.interpolate import griddata

# === Input Data ===
data = {
    'Efield1': 'D8_07_E.txt',
    'Efield2': 'D8_10_E.txt',
    'Efield3': 'D8_14_E.txt',
    'Efield4': 'D8_20_E.txt',
    'core_radius1': 4.035e-6,
    'core_radius2': 4.05e-6,
    'core_radius3': 4.07e-6,
    'core_radius4': 4.1e-6,
    'ncore1': 1.45050330,
    'ncore2': 1.45071305,
    'ncore3': 1.45076118,
    'ncore4': 1.45077083,
    'nclad': 1.444,
    'n_eff1': 1.4464676,
    'n_eff2': 1.4466484,
    'n_eff3': 1.4466709,
    'n_eff4': 1.4467458,
    'comsol_clad_rad_um': 62.5,
    'avg_of': 100,
    'core_pitch': 40e-6,
    'Bending_radius': 85e-3,
    'ds': 1,
    'twist_rate': 0.1 * pi,
    'fiberLength': 10000,
    'wavelength': 1550e-9
}

# === Derived Constants ===
comsol_clad_rad_um = data['comsol_clad_rad_um']
clad_diam = (2 * comsol_clad_rad_um + 20) * 1e-6
pixel_size = 1e-8
curr_rad = 40e-6
size_n = int(np.floor(2 * curr_rad / pixel_size) + 1)

# === Core radii and RI ===
core_radii = [data['core_radius1'], data['core_radius2'], data['core_radius3'], data['core_radius4']]
ncores = [data['ncore1'], data['ncore2'], data['ncore3'], data['ncore4']]
nclad = data['nclad']
core_pitch = data['core_pitch']
wavelength = data['wavelength']

# === Mesh for core ===
x = np.linspace(-40e-6, 40e-6, size_n)
y = np.linspace(-40e-6, 40e-6, size_n)
xx, yy = np.meshgrid(x, y)

# === Function to make n_core ===
def make_core(ncore, radius):
    n = nclad * np.ones((size_n, size_n))
    mask = np.sqrt(xx**2 + yy**2) < radius
    n[mask] = ncore
    s = int(np.floor(len(n)/2 - radius/pixel_size) - 1)
    e = int(np.floor(len(n)/2 + radius/pixel_size))
    return n[s:e, s:e]

n_cores = [make_core(ncores[i], core_radii[i]) for i in range(4)]

# === Constants ===
omega = 2 * pi * c / wavelength

# === Grid for imported Efield ===
lim = 62.5
nx = int(np.floor(2 * lim * 1e-6 / pixel_size + 1))
xnew = np.linspace(-lim, lim, nx)
ynew = np.linspace(-lim, lim, nx)
grid_x, grid_y = np.meshgrid(xnew, ynew)

# === Function to load and interpolate E-field ===
def load_efield(filename, rotations=0):
    data_arr = np.loadtxt(filename)
    col1, col2, col3 = data_arr[:, 0], data_arr[:, 1], data_arr[:, 2]
    E = griddata((col1, col2), col3, (grid_x, grid_y), method='linear', fill_value=0)
    # handle rotations
    for _ in range(rotations):
        E = np.rot90(E)
    E[np.isnan(E)] = 0
    return E

# Adjust rotations per your original code
E_fields_loaded = [
    load_efield(data['Efield1'], rotations=2),
    load_efield(data['Efield2'], rotations=3),
    load_efield(data['Efield3'], rotations=0),
    load_efield(data['Efield4'], rotations=1)
]

# === Fiber grid ===
fiber_dim = int(np.floor(clad_diam / pixel_size))
core_pitch_dim = int(np.floor(core_pitch / pixel_size))

fiber_RI = nclad * np.ones((1, fiber_dim, fiber_dim))
fiber_isolated_core = nclad * np.ones((4, fiber_dim, fiber_dim))
E_fields = np.zeros((4, fiber_dim, fiber_dim), dtype=np.complex128)
ly = fiber_RI.shape[1]

# === Helper to add cores and fields ===
def place_core_and_field(idx, n_core, E_core):
    core_dim = int(2 * core_radii[idx] / pixel_size + 1)
    s = int(np.floor(fiber_dim/2 - core_pitch_dim/2 - core_dim/2))
    e = s + n_core.shape[0] - 1
    if s < 0 or e >= fiber_dim:
        raise ValueError("Increase core pitch")
    # core placement
    if idx == 0:
        fiber_RI[0, s:e+1, s:e+1] = n_core
        fiber_isolated_core[idx, s:e+1, s:e+1] = n_core
    elif idx == 1:
        fiber_RI[0, ly - e - 1:ly - s, s:e+1] = n_core
        fiber_isolated_core[idx, ly - e - 1:ly - s, s:e+1] = n_core
    elif idx == 2:
        fiber_RI[0, ly - e - 1:ly - s, ly - e - 1:ly - s] = n_core
        fiber_isolated_core[idx, ly - e - 1:ly - s, ly - e - 1:ly - s] = n_core
    elif idx == 3:
        fiber_RI[0, s:e+1, ly - e - 1:ly - s] = n_core
        fiber_isolated_core[idx, s:e+1, ly - e - 1:ly - s] = n_core

    # E field placement
    s_e = int(np.floor(fiber_dim/2 - E_core.shape[0]/2) - 1)
    e_e = s_e + E_core.shape[0] - 1
    if e_e >= fiber_dim or s_e <= 0:
        raise ValueError("E field outside fiber, reduce pitch")
    E_fields[idx, s_e:e_e+1, s_e:e_e+1] = E_core

# Place cores and fields
for i in range(4):
    place_core_and_field(i, n_cores[i], E_fields_loaded[i])

# === Epsilon difference ===
epsilons = np.zeros((4, fiber_dim, fiber_dim))
p = np.zeros(4)
for i in range(4):
    epsilons[i] = fiber_RI[0]**2 - fiber_isolated_core[i]**2
    p[i] = np.sum(np.abs(E_fields[i])**2) * pixel_size**2 * epsilon_0 * c / 2

# === Coupling coefficients ===
k_cc = np.zeros((4, 4), dtype=np.complex128)
for i in range(4):
    for j in range(4):
        if i == j:
            continue
        integrand = epsilons[j] * np.conjugate(E_fields[i]) * E_fields[j]
        k_cc[i, j] = omega * epsilon_0 * np.sum(integrand) * pixel_size**2 / (4 * p[i])

print("Coupling Coefficient Matrix (4x4):")
print(np.real(k_cc))
