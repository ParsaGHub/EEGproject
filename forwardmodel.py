# This file is adapted from `analytical_correct.py` in:
# https://github.com/Neuroinflab/fourspheremodel
#
# Original work: Corrected Four-Sphere Head Model for EEG Signals
# Authors: Solveig Næss, Chaitanya Chintaluri, Torbjørn V. Ness,
# Anders M. Dale, Gaute T. Einevoll, and Daniel K. Wójcik
# License: GNU General Public License v3.
#
# Modified by: Parsa Gorji
# Modification date: 2026
# Changes: Adapted for inverse WPINN EEG dipole-localization experiments.

import os
import numpy as np
from scipy.special import lpmv
import parameters as params
import argparse

# ---------------------------------------------------------------------------
# Electrode-layout generator
# ---------------------------------------------------------------------------

def make_scalp_electrodes(
    n_electrodes=64,
    radius=None,
    hemisphere="upper",
    min_polar_deg=0.0,
    max_polar_deg=90.0,
):
    if radius is None:
        radius = params.scalp_rad

    i = np.arange(n_electrodes, dtype=float)

    if hemisphere == "upper":
        cos_max = np.cos(np.deg2rad(min_polar_deg))
        cos_min = np.cos(np.deg2rad(max_polar_deg))
        cos_theta = cos_max - (cos_max - cos_min) * (i + 0.5) / n_electrodes

    elif hemisphere == "full":
        cos_theta = 1.0 - 2.0 * (i + 0.5) / n_electrodes

    else:
        raise ValueError(f"hemisphere must be 'upper' or 'full', got {hemisphere!r}")

    sin_theta = np.sqrt(np.clip(1.0 - cos_theta**2, 0.0, 1.0))

    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    phi = np.mod(golden_angle * i, 2.0 * np.pi)

    x = radius * sin_theta * np.cos(phi)
    y = radius * sin_theta * np.sin(phi)
    z = radius * cos_theta

    electrodes_xyz = np.stack([x, y, z], axis=1)

    theta = np.arccos(cos_theta)
    electrodes_sph = np.stack([theta, phi], axis=1)

    return electrodes_xyz, electrodes_sph


SCALP_ELECTRODES_64, SCALP_ELECTRODES_64_SPH = make_scalp_electrodes(
    n_electrodes=64,
    radius=params.scalp_rad,
    hemisphere="upper",
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def default_radii():
    return np.array([
        params.brain_rad,
        params.csftop_rad,
        params.skull_rad,
        params.scalp_rad,
    ], dtype=float)


def default_sigmas():
    return np.array([
        params.sigma_brain,
        params.sigma_csf,
        params.sigma_skull20,
        params.sigma_scalp,
    ], dtype=float)


# ---------------------------------------------------------------------------
# Conductivity and geometry helpers
# ---------------------------------------------------------------------------

def conductivity_ratios(sigmas):
    sigma_brain, sigma_csf, sigma_skull, sigma_scalp = sigmas

    s12 = sigma_brain / sigma_csf
    s23 = sigma_csf / sigma_skull
    s34 = sigma_skull / sigma_scalp

    return s12, s23, s34


def geometry_ratios(radii):
    brain_rad, csf_rad, skull_rad, scalp_rad = radii

    r12 = brain_rad / csf_rad
    r23 = csf_rad / skull_rad
    r34 = skull_rad / scalp_rad

    r21 = 1.0 / r12
    r32 = 1.0 / r23
    r43 = 1.0 / r34

    return r12, r23, r34, r21, r32, r43


def coefficient_functions(r0_norm, radii, sigmas):
    brain_rad, csf_rad, skull_rad, scalp_rad = radii

    s12, s23, s34 = conductivity_ratios(sigmas)
    r12, r23, r34, r21, r32, r43 = geometry_ratios(radii)

    rz = r0_norm
    rz1 = rz / brain_rad

    def Vcoef(n):
        n = np.asarray(n, dtype=float)
        k = (n + 1.0) / n

        factor = (r34**n - r43 ** (n + 1.0)) / (
            k * r34**n + r43 ** (n + 1.0)
        )

        num = (s34 / k) - factor
        den = s34 + factor

        return num / den

    def Ycoef(n):
        n = np.asarray(n, dtype=float)
        k = n / (n + 1.0)

        factor = ((r23**n) * k - Vcoef(n) * r32 ** (n + 1.0)) / (
            r23**n + Vcoef(n) * r32 ** (n + 1.0)
        )

        num = s23 * k - factor
        den = s23 + factor

        return num / den

    def Zcoef(n):
        n = np.asarray(n, dtype=float)
        k = (n + 1.0) / n

        num = r12**n - k * Ycoef(n) * r21 ** (n + 1.0)
        den = r12**n + Ycoef(n) * r21 ** (n + 1.0)

        return num / den

    def A1(n):
        n = np.asarray(n, dtype=float)

        num = rz1 ** (n + 1.0) * (Zcoef(n) + s12 * ((n + 1.0) / n))
        den = s12 - Zcoef(n)

        return num / den

    def A2(n):
        n = np.asarray(n, dtype=float)

        num = A1(n) + rz1 ** (n + 1.0)
        den = Ycoef(n) * r21 ** (n + 1.0) + r12**n

        return num / den

    def B2(n):
        return A2(n) * Ycoef(n)

    def A3(n):
        n = np.asarray(n, dtype=float)

        num = A2(n) + B2(n)
        den = r23**n + Vcoef(n) * r32 ** (n + 1.0)

        return num / den

    def B3(n):
        return A3(n) * Vcoef(n)

    def A4(n):
        n = np.asarray(n, dtype=float)

        num = A3(n) + B3(n)
        k = (n + 1.0) / n
        den = k * r34**n + r43 ** (n + 1.0)

        return k * (num / den)

    def B4(n):
        n = np.asarray(n, dtype=float)
        return A4(n) * (n / (n + 1.0))

    def H(n, r_ele=None):
        n = np.asarray(n, dtype=float)

        if r_ele is None:
            r_ele = scalp_rad

        if r_ele < brain_rad:
            T1 = (r_ele / brain_rad) ** n * A1(n)
            T2 = (rz / r_ele) ** (n + 1.0)

        elif r_ele < csf_rad:
            T1 = (r_ele / csf_rad) ** n * A2(n)
            T2 = (csf_rad / r_ele) ** (n + 1.0) * B2(n)

        elif r_ele < skull_rad:
            T1 = (r_ele / skull_rad) ** n * A3(n)
            T2 = (skull_rad / r_ele) ** (n + 1.0) * B3(n)

        elif r_ele <= scalp_rad:
            T1 = (r_ele / scalp_rad) ** n * A4(n)
            T2 = (scalp_rad / r_ele) ** (n + 1.0) * B4(n)

        else:
            raise ValueError("Electrode radius is outside scalp radius.")

        return T1 + T2

    return {
        "V": Vcoef,
        "Y": Ycoef,
        "Z": Zcoef,
        "A1": A1,
        "A2": A2,
        "B2": B2,
        "A3": A3,
        "B3": B3,
        "A4": A4,
        "B4": B4,
        "H": H,
    }


# ---------------------------------------------------------------------------
# Dipole helpers
# ---------------------------------------------------------------------------

def decompose_dipole(p, r0):
    p = np.asarray(p, dtype=float)
    r0 = np.asarray(r0, dtype=float)

    r0_norm = np.linalg.norm(r0)

    if r0_norm == 0:
        raise ValueError("Dipole location cannot be the zero vector.")

    rhat = r0 / r0_norm

    p_rad = np.dot(p, rhat) * rhat
    p_tan = p - p_rad

    return p_rad, p_tan


def electrode_angles(electrodes, r0):
    electrodes = np.asarray(electrodes, dtype=float)
    r0 = np.asarray(r0, dtype=float)

    ele_norms = np.linalg.norm(electrodes, axis=1)
    r0_norm = np.linalg.norm(r0)

    cos_theta = electrodes @ r0 / (ele_norms * r0_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)

    return theta, cos_theta


def tangential_phi_angle(electrodes, r0, p_tan):
    electrodes = np.asarray(electrodes, dtype=float)
    r0 = np.asarray(r0, dtype=float)
    p_tan = np.asarray(p_tan, dtype=float)

    p_tan_norm = np.linalg.norm(p_tan)

    if p_tan_norm < 1e-14:
        return np.zeros(electrodes.shape[0])

    proj = ((electrodes @ r0) / np.sum(r0**2)).reshape(-1, 1) * r0.reshape(1, 3)
    rxy = electrodes - proj

    x_axis = np.cross(p_tan, r0)

    rxy_norm = np.linalg.norm(rxy, axis=1)
    x_norm = np.linalg.norm(x_axis)

    denom = rxy_norm * x_norm
    denom = np.where(denom == 0, np.inf, denom)

    cos_phi = (rxy @ x_axis) / denom
    cos_phi = np.clip(cos_phi, -1.0, 1.0)

    phi_temp = np.arccos(cos_phi)

    range_test = rxy @ p_tan
    phi = np.where(range_test < 0, 2.0 * np.pi - phi_temp, phi_temp)

    return phi


# ---------------------------------------------------------------------------
# Potential pieces
# ---------------------------------------------------------------------------

def radial_potential(p_rad, r0, electrodes, radii, sigmas, n_terms=100):
    r0 = np.asarray(r0, dtype=float)
    p_rad = np.asarray(p_rad, dtype=float)
    electrodes = np.asarray(electrodes, dtype=float)

    sigma_brain = sigmas[0]

    r0_norm = np.linalg.norm(r0)
    rhat = r0 / r0_norm

    p_rad_mag = np.dot(p_rad, rhat)

    coeffs = coefficient_functions(r0_norm, radii, sigmas)

    n = np.arange(1, n_terms + 1, dtype=float)
    Hn = coeffs["H"](n)

    _, cos_theta = electrode_angles(electrodes, r0)

    legendre_coeffs = np.insert(n * Hn, 0, 0.0)
    legendre_series = np.polynomial.legendre.Legendre(legendre_coeffs)

    V_rad = p_rad_mag * legendre_series(cos_theta)

    prefactor = 1.0 / (4.0 * np.pi * sigma_brain * r0_norm**2)

    return prefactor * V_rad


def tangential_potential(p_tan, r0, electrodes, radii, sigmas, n_terms=100):
    r0 = np.asarray(r0, dtype=float)
    p_tan = np.asarray(p_tan, dtype=float)
    electrodes = np.asarray(electrodes, dtype=float)

    sigma_brain = sigmas[0]

    r0_norm = np.linalg.norm(r0)
    p_tan_mag = np.linalg.norm(p_tan)

    if p_tan_mag < 1e-14:
        return np.zeros(electrodes.shape[0])

    coeffs = coefficient_functions(r0_norm, radii, sigmas)

    n = np.arange(1, n_terms + 1, dtype=int)
    Hn = coeffs["H"](n)

    _, cos_theta = electrode_angles(electrodes, r0)
    phi = tangential_phi_angle(electrodes, r0, p_tan)

    Lsum = np.zeros(electrodes.shape[0])

    for Hk, nk in zip(Hn, n):
        Lsum += Hk * lpmv(1, nk, cos_theta)

    V_tan = -p_tan_mag * np.sin(phi) * Lsum

    prefactor = 1.0 / (4.0 * np.pi * sigma_brain * r0_norm**2)

    return prefactor * V_tan


# ---------------------------------------------------------------------------
# Main forward model
# ---------------------------------------------------------------------------

def forward(
    p,
    r0,
    radii=None,
    sigmas=None,
    electrodes=None,
    n_terms=100,
):
    p = np.asarray(p, dtype=float)
    r0 = np.asarray(r0, dtype=float)

    if radii is None:
        radii = default_radii()
    else:
        radii = np.asarray(radii, dtype=float)

    if sigmas is None:
        sigmas = default_sigmas()
    else:
        sigmas = np.asarray(sigmas, dtype=float)

    brain_rad, csf_rad, skull_rad, scalp_rad = radii

    if not (brain_rad < csf_rad < skull_rad < scalp_rad):
        raise ValueError("Head radii must satisfy brain < CSF < skull < scalp.")

    if np.any(sigmas <= 0):
        raise ValueError("All conductivities must be positive.")

    if electrodes is None:
        electrodes, _ = make_scalp_electrodes(
            n_electrodes=64,
            radius=scalp_rad,
            hemisphere="upper",
        )

    electrodes = np.asarray(electrodes, dtype=float)

    r0_norm = np.linalg.norm(r0)

    if r0_norm <= 0:
        raise ValueError("Dipole location cannot be zero.")

    if r0_norm >= brain_rad:
        raise ValueError("Dipole must be strictly inside the brain sphere.")

    p_rad, p_tan = decompose_dipole(p, r0)

    V_rad = radial_potential(
        p_rad=p_rad,
        r0=r0,
        electrodes=electrodes,
        radii=radii,
        sigmas=sigmas,
        n_terms=n_terms,
    )

    V_tan = tangential_potential(
        p_tan=p_tan,
        r0=r0,
        electrodes=electrodes,
        radii=radii,
        sigmas=sigmas,
        n_terms=n_terms,
    )

    return V_rad + V_tan


def forward_from_src_snk(
    src_pos,
    snk_pos,
    I=1.0,
    radii=None,
    sigmas=None,
    electrodes=None,
    n_terms=100,
):
    src_pos = np.asarray(src_pos, dtype=float)
    snk_pos = np.asarray(snk_pos, dtype=float)

    r0 = 0.5 * (src_pos + snk_pos)
    p = I * (src_pos - snk_pos)

    return forward(
        p=p,
        r0=r0,
        radii=radii,
        sigmas=sigmas,
        electrodes=electrodes,
        n_terms=n_terms,
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing corrected stochastic forward model.")

    dipole = params.dipole_list[0]

    base_radii = default_radii()
    base_sigmas = default_sigmas()

    V_base = forward_from_src_snk(
        src_pos=dipole["src_pos"],
        snk_pos=dipole["snk_pos"],
        I=1.0,
        radii=base_radii,
        sigmas=base_sigmas,
        n_terms=100,
    )

    changed_radii = np.array([
        1.01 * params.brain_rad,
        1.01 * params.csftop_rad,
        1.01 * params.skull_rad,
        1.01 * params.scalp_rad,
    ])

    changed_sigmas = np.array([
        params.sigma_brain,
        params.sigma_csf,
        1.3 * params.sigma_skull20,
        params.sigma_scalp,
    ])

    V_changed = forward_from_src_snk(
        src_pos=dipole["src_pos"],
        snk_pos=dipole["snk_pos"],
        I=1.0,
        radii=changed_radii,
        sigmas=changed_sigmas,
        n_terms=100,
    )

    print("Base voltage shape:", V_base.shape)
    print("Base min/max:", V_base.min(), V_base.max())
    print("Changed min/max:", V_changed.min(), V_changed.max())
    print("Difference norm:", np.linalg.norm(V_base - V_changed))