'''we stochastically sample the shell radii r_s (s=1,2,3,4) where r_s ~ Unif(0.95r_s, 1.05r_s)
and the conductivities sigma_s (s=1,2,3,4) where sigma_s ~ Unif(log(0.7sigma_s), log(1.3sigma_s))
additionally we resample if the following assumption is broken by the stochasticity : r1 < r2 < r3 < r4
saves output as a csv file: "random_head_params.csv".
'''

import csv
import numpy as np
import parameters as params


def random_sample_params():
    while True:
        r1 = np.random.uniform(0.95 * params.brain_rad, 1.05 * params.brain_rad)
        r2 = np.random.uniform(0.95 * params.csftop_rad, 1.05 * params.csftop_rad)
        r3 = np.random.uniform(0.95 * params.skull_rad, 1.05 * params.skull_rad)
        r4 = np.random.uniform(0.95 * params.scalp_rad, 1.05 * params.scalp_rad)

        if r1 < r2 < r3 < r4:
            break

    sigma1 = np.exp(np.random.uniform(
        np.log(0.7 * params.sigma_brain),
        np.log(1.3 * params.sigma_brain),
    ))

    sigma2 = np.exp(np.random.uniform(
        np.log(0.7 * params.sigma_csf),
        np.log(1.3 * params.sigma_csf),
    ))

    sigma3 = np.exp(np.random.uniform(
        np.log(0.7 * params.sigma_skull20),
        np.log(1.3 * params.sigma_skull20),
    ))

    sigma4 = np.exp(np.random.uniform(
        np.log(0.7 * params.sigma_scalp),
        np.log(1.3 * params.sigma_scalp),
    ))

    return (r1, r2, r3, r4), (sigma1, sigma2, sigma3, sigma4)


def save_random_params_csv(output_file="random_head_params.csv", n_samples=1000):
    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "sample_id",
            "brain_rad", "csf_rad", "skull_rad", "scalp_rad",
            "sigma_brain", "sigma_csf", "sigma_skull", "sigma_scalp",
        ])

        for i in range(n_samples):
            radii, sigmas = random_sample_params()

            writer.writerow([
                i,
                radii[0], radii[1], radii[2], radii[3],
                sigmas[0], sigmas[1], sigmas[2], sigmas[3],
            ])

    print(f"Saved {n_samples} samples to {output_file}")


if __name__ == "__main__":
    save_random_params_csv(
        output_file="random_head_params.csv",
        n_samples=1000,
    )