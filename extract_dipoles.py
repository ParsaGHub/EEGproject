import numpy as np

# All numbers in cm
dipole_loc = 7.8
brain_rad = 7.8
csftop_rad = 8.1
skull_rad = 8.6
scalp_rad = 9.0

sigma_brain = 1. / 300.  # S / cm
sigma_scalp = sigma_brain
sigma_csf = 5 * sigma_brain
sigma_skull20 = sigma_brain / 20.
sigma_skull40 = sigma_brain / 40.
sigma_skull80 = sigma_brain / 80.

# from gmsh sphere_4.geo
whitemattervol = 32
graymattervol = 64
csfvol = 96
skullvol = 128
scalpvol = 160

# measument points
# theta = np.arange(0, 180)
# phi_angle = 0 # -90 to 90

theta, phi_angle = np.mgrid[0:180:1, -90:90:1]
theta = theta.flatten()
phi_angle = phi_angle.flatten()

theta_r = np.deg2rad(theta)
phi_angle_r = np.deg2rad(phi_angle)

rad_tol = 1e-2
x_points = (scalp_rad - rad_tol) * np.sin(theta_r) * np.cos(phi_angle_r)
y_points = (scalp_rad - rad_tol) * np.sin(theta_r) * np.sin(phi_angle_r)
z_points = (scalp_rad - rad_tol) * np.cos(theta_r)

ele_coords = np.vstack((x_points, y_points, z_points)).T

# 80 locations sampled uniformly inside the brain, 19 orientations each = 1520 dipoles
# Orientations sweep y-z plane from radial (0°) to tangential (90°) in 5° steps
_rng = np.random.default_rng(42)
_n_locations = 80
_half_len = 0.05
_max_r = brain_rad - 0.1  # keep dipoles away from the brain boundary

_u = _rng.uniform(0, 1, _n_locations)
_r = _max_r * _u ** (1 / 3)          # uniform-in-sphere radial distribution
_cos_t = _rng.uniform(-1, 1, _n_locations)
_sin_t = np.sqrt(1 - _cos_t ** 2)
_phi = _rng.uniform(0, 2 * np.pi, _n_locations)

_locations = np.column_stack([
    _r * _sin_t * np.cos(_phi),
    _r * _sin_t * np.sin(_phi),
    _r * _cos_t,
])

dipole_list = []
for _i, _loc in enumerate(_locations):
    for _deg in range(0, 91, 5):
        _alpha = np.deg2rad(_deg)
        _direction = np.array([0., np.sin(_alpha), np.cos(_alpha)])
        _src = _loc + _half_len * _direction
        _snk = _loc - _half_len * _direction
        dipole_list.append({
            'src_pos': _src.tolist(),
            'snk_pos': _snk.tolist(),
            'name': f'loc{_i:03d}_deg{_deg:03d}',
        })

if __name__ == "__main__":
    import csv

    output_file = "dipole_list.csv"

    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "name",
            "r0_x", "r0_y", "r0_z",
            "p_x", "p_y", "p_z",
        ])

        for d in dipole_list:
            src = np.array(d["src_pos"])
            snk = np.array(d["snk_pos"])

            r0 = 0.5 * (src + snk)
            p = src - snk

            writer.writerow([
                d["name"],
                r0[0], r0[1], r0[2],
                p[0], p[1], p[2],
            ])

    print(f"Saved {len(dipole_list)} dipoles to {output_file}")
