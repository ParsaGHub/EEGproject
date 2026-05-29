'''
Sample 50,000 random radii
Sample 50,000 random conductivities
Sample 50,000 random dipole locations r0
Sample 50,000 random dipole orientations p
Run forward model 50,000 times
Save V_train.npy, sigma_train.npy, r_train.npy, p_train.npy, r0_train.npy
Compute normalization stats
Make plots
'''



from forwardmodel import forward
import os
import numpy as np
import matplotlib.pyplot as plt
import parameters as params



# ------------------------------------------------------------
# Random samplers
# ------------------------------------------------------------

def sample_radii():
    """
    Sample the four shell radii and enforce correct ordering:
    brain < csf < skull < scalp
    """
    while True:
        brain = np.random.uniform(0.95 * params.brain_rad, 1.05 * params.brain_rad)
        csf = np.random.uniform(0.95 * params.csftop_rad, 1.05 * params.csftop_rad)
        skull = np.random.uniform(0.95 * params.skull_rad, 1.05 * params.skull_rad)
        scalp = np.random.uniform(0.95 * params.scalp_rad, 1.05 * params.scalp_rad)

        if brain < csf < skull < scalp:
            return np.array([brain, csf, skull, scalp], dtype=float)


def sample_conductivities():
    """
    Sample conductivities log-uniformly within +/- 30% of nominal values.
    """
    brain = np.exp(np.random.uniform(
        np.log(0.7 * params.sigma_brain),
        np.log(1.3 * params.sigma_brain),
    ))

    csf = np.exp(np.random.uniform(
        np.log(0.7 * params.sigma_csf),
        np.log(1.3 * params.sigma_csf),
    ))

    skull = np.exp(np.random.uniform(
        np.log(0.7 * params.sigma_skull20),
        np.log(1.3 * params.sigma_skull20),
    ))

    scalp = np.exp(np.random.uniform(
        np.log(0.7 * params.sigma_scalp),
        np.log(1.3 * params.sigma_scalp),
    ))

    return np.array([brain, csf, skull, scalp], dtype=float)


def sample_unit_vector():
    """
    Sample a random direction uniformly on the unit sphere.
    """
    v = np.random.normal(size=3)
    return v / np.linalg.norm(v)


def sample_dipole_location(inner_radius, outer_radius):
    """
    Sample a point uniformly inside a spherical shell.

    inner_radius avoids points too close to zero, where the forward model
    can become numerically unstable.
    """
    direction = sample_unit_vector()

    u = np.random.uniform(0.0, 1.0)
    radius = (inner_radius**3 + u * (outer_radius**3 - inner_radius**3)) ** (1.0 / 3.0)

    return radius * direction


def sample_dipole_orientation(strength=1.0):
    """
    Sample dipole orientation uniformly on the sphere.
    """
    return strength * sample_unit_vector()


# ------------------------------------------------------------
# Training data generation
# ------------------------------------------------------------

def generate_training_data(
    n_samples=50_000,
    output_dir="training_data",
    dipole_strength=1.0,
    inner_radius_fraction=0.05,
    n_terms=100,
):
    os.makedirs(output_dir, exist_ok=True)

    # Number of electrodes is determined by one test forward call
    test_location = np.array([0.5 * params.brain_rad, 0.0, 0.0])
    test_orientation = np.array([1.0, 0.0, 0.0])
    test_voltage = forward(
        p=test_orientation,
        r0=test_location,
        radii=np.array([
            params.brain_rad,
            params.csftop_rad,
            params.skull_rad,
            params.scalp_rad,
        ]),
        sigmas=np.array([
            params.sigma_brain,
            params.sigma_csf,
            params.sigma_skull20,
            params.sigma_scalp,
        ]),
        n_terms=n_terms,
)
    n_electrodes = test_voltage.shape[0]

    V_train = np.zeros((n_samples, n_electrodes), dtype=float)
    sigma_train = np.zeros((n_samples, 4), dtype=float)
    r_train = np.zeros((n_samples, 4), dtype=float)
    p_train = np.zeros((n_samples, 3), dtype=float)
    r0_train = np.zeros((n_samples, 3), dtype=float)

    inner_radius = inner_radius_fraction * params.brain_rad
    outer_radius = 0.95 * params.brain_rad

    for i in range(n_samples):
        radii = sample_radii()
        conductivities = sample_conductivities()

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
            radii=radii,
            sigmas=conductivities,
            n_terms=n_terms,
        )
        

        r_train[i] = radii
        sigma_train[i] = conductivities
        r0_train[i] = dipole_location
        p_train[i] = dipole_orientation
        V_train[i] = voltages

        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{n_samples} samples")

    # Save training arrays
    np.save(os.path.join(output_dir, "V_train.npy"), V_train)
    np.save(os.path.join(output_dir, "sigma_train.npy"), sigma_train)
    np.save(os.path.join(output_dir, "r_train.npy"), r_train)
    np.save(os.path.join(output_dir, "p_train.npy"), p_train)
    np.save(os.path.join(output_dir, "r0_train.npy"), r0_train)

    print("Saved training arrays.")

    # Compute normalization statistics from training set only
    normalization_stats = {
        "V_mean": V_train.mean(axis=0),
        "V_std": V_train.std(axis=0) + 1e-12,

        "sigma_mean": sigma_train.mean(axis=0),
        "sigma_std": sigma_train.std(axis=0) + 1e-12,

        "r_mean": r_train.mean(axis=0),
        "r_std": r_train.std(axis=0) + 1e-12,

        "p_mean": p_train.mean(axis=0),
        "p_std": p_train.std(axis=0) + 1e-12,

        "r0_mean": r0_train.mean(axis=0),
        "r0_std": r0_train.std(axis=0) + 1e-12,
    }

    np.save(
        os.path.join(output_dir, "normalization_stats.npy"),
        normalization_stats,
        allow_pickle=True,
    )

    print("Saved normalization statistics.")

    make_plots(
        V_train=V_train,
        sigma_train=sigma_train,
        r_train=r_train,
        p_train=p_train,
        r0_train=r0_train,
        output_dir=output_dir,
    )

    print("Saved plots.")

    return V_train, sigma_train, r_train, p_train, r0_train


# Sanity-check plots

def make_plots(V_train, sigma_train, r_train, p_train, r0_train, output_dir):
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    radius_names = ["brain", "csf", "skull", "scalp"]
    conductivity_names = ["brain", "csf", "skull", "scalp"]

    for j, name in enumerate(radius_names):
        plt.figure()
        plt.hist(r_train[:, j], bins=50)
        plt.xlabel(f"{name} radius")
        plt.ylabel("count")
        plt.title(f"Sampled {name} radius")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"radius_{name}.png"))
        plt.close()

    for j, name in enumerate(conductivity_names):
        plt.figure()
        plt.hist(sigma_train[:, j], bins=50)
        plt.xlabel(f"{name} conductivity")
        plt.ylabel("count")
        plt.title(f"Sampled {name} conductivity")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"conductivity_{name}.png"))
        plt.close()

    for j, coord in enumerate(["x", "y", "z"]):
        plt.figure()
        plt.hist(r0_train[:, j], bins=50)
        plt.xlabel(f"dipole location {coord}")
        plt.ylabel("count")
        plt.title(f"Dipole location {coord}-coordinate")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"dipole_location_{coord}.png"))
        plt.close()

    for j, coord in enumerate(["x", "y", "z"]):
        plt.figure()
        plt.hist(p_train[:, j], bins=50)
        plt.xlabel(f"dipole orientation {coord}")
        plt.ylabel("count")
        plt.title(f"Dipole orientation {coord}-component")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"dipole_orientation_{coord}.png"))
        plt.close()

    dipole_distances = np.linalg.norm(r0_train, axis=1)

    plt.figure()
    plt.hist(dipole_distances, bins=50)
    plt.xlabel("distance from origin")
    plt.ylabel("count")
    plt.title("Dipole distance from origin")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "dipole_distance_from_origin.png"))
    plt.close()

    plt.figure()
    plt.hist(V_train.flatten(), bins=100)
    plt.xlabel("electrode voltage")
    plt.ylabel("count")
    plt.title("All electrode voltages")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "voltage_distribution.png"))
    plt.close()


# ------------------------------------------------------------
# Run script
# ------------------------------------------------------------

if __name__ == "__main__":
    generate_training_data(
        n_samples=50_000,
        output_dir="training_data",
        dipole_strength=1.0,
        inner_radius_fraction=0.05,
        n_terms=100,
    )
