'''
Runs the forward model wth fixed nomimal parameters so then we can feed into the model and see if it works at all
'''

import os
import numpy as np
import matplotlib.pyplot as plt

import parameters as params

# Change this if your forward model file has a different name.
# Example: from forward_model import forward
from forwardmodel import forward


# ------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------

def default_radii():
    return np.array([
        params.brain_rad,
        params.csftop_rad,
        params.skull_rad,
        params.scalp_rad,
    ], dtype=float)


def default_conductivities():
    return np.array([
        params.sigma_brain,
        params.sigma_csf,
        params.sigma_skull20,
        params.sigma_scalp,
    ], dtype=float)


def sample_unit_vector():
    v = np.random.normal(size=3)
    return v / np.linalg.norm(v)


def sample_dipole_location(inner_radius, outer_radius):
    direction = sample_unit_vector()

    # Uniform in 3D volume, not uniform in radius
    u = np.random.uniform(0.0, 1.0)
    radius = (inner_radius**3 + u * (outer_radius**3 - inner_radius**3)) ** (1.0 / 3.0)

    return radius * direction


def sample_dipole_orientation(strength=1.0):
    return strength * sample_unit_vector()


# ------------------------------------------------------------
# Generate E1 analytical validation data
# ------------------------------------------------------------

def generate_e1_data(
    n_samples=2000,
    output_dir="E1_data",
    training_stats_path="training_data/normalization_stats.npy",
    dipole_strength=1.0,
    inner_radius_fraction=0.05,
    outer_radius_fraction=0.95,
    n_terms=100,
):
    os.makedirs(output_dir, exist_ok=True)

    radii_nominal = default_radii()
    conductivities_nominal = default_conductivities()

    brain_radius = radii_nominal[0]

    inner_radius = inner_radius_fraction * brain_radius
    outer_radius = outer_radius_fraction * brain_radius

    # Test one forward solve to determine number of electrodes
    test_location = np.array([0.5 * brain_radius, 0.0, 0.0])
    test_orientation = np.array([1.0, 0.0, 0.0])

    test_voltage = forward(
        p=test_orientation,
        r0=test_location,
        radii=radii_nominal,
        sigmas=conductivities_nominal,
        n_terms=n_terms,
    )

    n_electrodes = test_voltage.shape[0]

    V_e1 = np.zeros((n_samples, n_electrodes), dtype=float)
    r_e1 = np.zeros((n_samples, 4), dtype=float)
    sigma_e1 = np.zeros((n_samples, 4), dtype=float)
    p_e1 = np.zeros((n_samples, 3), dtype=float)
    r0_e1 = np.zeros((n_samples, 3), dtype=float)

    for i in range(n_samples):
        dipole_location = sample_dipole_location(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
        )

        dipole_orientation = sample_dipole_orientation(
            strength=dipole_strength
        )

        voltages = forward(
            p=dipole_orientation,
            r0=dipole_location,
            radii=radii_nominal,
            sigmas=conductivities_nominal,
            n_terms=n_terms,
        )

        V_e1[i] = voltages
        r_e1[i] = radii_nominal
        sigma_e1[i] = conductivities_nominal
        p_e1[i] = dipole_orientation
        r0_e1[i] = dipole_location

        if (i + 1) % 250 == 0:
            print(f"Generated {i + 1}/{n_samples} E1 samples")

    # Save raw E1 arrays
    np.save(os.path.join(output_dir, "E1_V.npy"), V_e1)
    np.save(os.path.join(output_dir, "E1_r.npy"), r_e1)
    np.save(os.path.join(output_dir, "E1_sigma.npy"), sigma_e1)
    np.save(os.path.join(output_dir, "E1_p.npy"), p_e1)
    np.save(os.path.join(output_dir, "E1_r0.npy"), r0_e1)

    print("Saved raw E1 arrays.")

    # Apply Day 2 training normalization
    stats = np.load(training_stats_path, allow_pickle=True).item()

    V_e1_norm = (V_e1 - stats["V_mean"]) / stats["V_std"]
    r_e1_norm = (r_e1 - stats["r_mean"]) / stats["r_std"]
    sigma_e1_norm = (sigma_e1 - stats["sigma_mean"]) / stats["sigma_std"]
    p_e1_norm = (p_e1 - stats["p_mean"]) / stats["p_std"]
    r0_e1_norm = (r0_e1 - stats["r0_mean"]) / stats["r0_std"]

    np.save(os.path.join(output_dir, "E1_V_norm.npy"), V_e1_norm)
    np.save(os.path.join(output_dir, "E1_r_norm.npy"), r_e1_norm)
    np.save(os.path.join(output_dir, "E1_sigma_norm.npy"), sigma_e1_norm)
    np.save(os.path.join(output_dir, "E1_p_norm.npy"), p_e1_norm)
    np.save(os.path.join(output_dir, "E1_r0_norm.npy"), r0_e1_norm)

    print("Saved normalized E1 arrays.")

    make_e1_plots(
        V_e1=V_e1,
        V_e1_norm=V_e1_norm,
        p_e1=p_e1,
        r0_e1=r0_e1,
        output_dir=output_dir,
    )

    print("Saved E1 plots.")

    return V_e1, r_e1, sigma_e1, p_e1, r0_e1


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------

def make_e1_plots(V_e1, V_e1_norm, p_e1, r0_e1, output_dir):
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure()
    plt.hist(V_e1.flatten(), bins=100)
    plt.xlabel("E1 electrode voltage")
    plt.ylabel("count")
    plt.title("E1 voltage distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "E1_voltage_distribution.png"))
    plt.close()

    plt.figure()
    plt.hist(V_e1_norm.flatten(), bins=100)
    plt.xlabel("normalized E1 electrode voltage")
    plt.ylabel("count")
    plt.title("Normalized E1 voltage distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "E1_voltage_normalized_distribution.png"))
    plt.close()

    dipole_distances = np.linalg.norm(r0_e1, axis=1)

    plt.figure()
    plt.hist(dipole_distances, bins=50)
    plt.xlabel("distance from origin")
    plt.ylabel("count")
    plt.title("E1 dipole distance from origin")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "E1_dipole_distance_from_origin.png"))
    plt.close()

    for j, coord in enumerate(["x", "y", "z"]):
        plt.figure()
        plt.hist(r0_e1[:, j], bins=50)
        plt.xlabel(f"dipole location {coord}")
        plt.ylabel("count")
        plt.title(f"E1 dipole location {coord}-coordinate")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"E1_dipole_location_{coord}.png"))
        plt.close()

    for j, coord in enumerate(["x", "y", "z"]):
        plt.figure()
        plt.hist(p_e1[:, j], bins=50)
        plt.xlabel(f"dipole orientation {coord}")
        plt.ylabel("count")
        plt.title(f"E1 dipole orientation {coord}-component")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"E1_dipole_orientation_{coord}.png"))
        plt.close()


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

if __name__ == "__main__":
    generate_e1_data(
        n_samples=2000,
        output_dir="E1_data",
        training_stats_path="training_data/normalization_stats.npy",
        dipole_strength=1.0,
        inner_radius_fraction=0.05,
        outer_radius_fraction=0.95,
        n_terms=100,
    )