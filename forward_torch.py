"""
forward_torch.py  —  Batched PyTorch four-sphere EEG forward model
-------------------------------------------------------------------
Differentiable w.r.t. p and r0 (and optionally sigmas/radii).
All operations are pure PyTorch so autograd flows through everything.

API
---
forward_torch(p, r0, sigmas, radii, electrodes, n_terms=100)
    p         : (B, 3)   dipole moment vectors
    r0        : (B, 3)   dipole locations
    sigmas    : (B, 4)   conductivities [brain, csf, skull, scalp]
    radii     : (B, 4)   shell radii    [brain, csf, skull, scalp]
    electrodes: (E, 3)   fixed electrode positions
    returns   : (B, E)   scalp potentials
"""

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Legendre polynomials  (no scipy — pure recurrence)
# ---------------------------------------------------------------------------

def legendre_Pn(n_max, cos_theta):
    """
    Compute P_1 ... P_n_max via recurrence.
    cos_theta : (...) any shape
    returns   : (n_max, ...) stacked Legendre values
    """
    shape = cos_theta.shape
    P_prev = torch.ones(shape, dtype=cos_theta.dtype, device=cos_theta.device)   # P0
    P_curr = cos_theta.clone()                                                     # P1
    polys  = [P_curr]
    for n in range(2, n_max + 1):
        P_next = ((2*n - 1) * cos_theta * P_curr - (n - 1) * P_prev) / n
        polys.append(P_next)
        P_prev, P_curr = P_curr, P_next
    return torch.stack(polys, dim=0)   # (n_max, ...)


def legendre_P1n(n_max, cos_theta):
    """
    Associated Legendre P^1_n via recurrence.
    cos_theta : (...) any shape
    returns   : (n_max, ...)
    """
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta**2, min=1e-30))
    P1_prev = -sin_theta                        # P^1_1
    polys   = [P1_prev]
    if n_max == 1:
        return torch.stack(polys, dim=0)
    P1_curr = -3.0 * sin_theta * cos_theta      # P^1_2
    polys.append(P1_curr)
    for n in range(3, n_max + 1):
        P1_next = ((2*n - 1) * cos_theta * P1_curr - n * P1_prev) / (n - 1)
        polys.append(P1_next)
        P1_prev, P1_curr = P1_curr, P1_next
    return torch.stack(polys, dim=0)   # (n_max, ...)


# ---------------------------------------------------------------------------
# Coefficient recursion  (batched over B)
# ---------------------------------------------------------------------------

def compute_coefficients(n_vec, sigmas, radii):
    """
    Compute H_n for each n in n_vec, batched over B head geometries.

    n_vec  : (N,)  1-D integer tensor  [1, 2, ..., n_max]
    sigmas : (B, 4)
    radii  : (B, 4)
    returns: (B, N)  H_n values
    """
    brain_rad = radii[:, 0:1]   # (B,1)
    csf_rad   = radii[:, 1:2]
    skull_rad = radii[:, 2:3]
    scalp_rad = radii[:, 3:4]

    s_brain = sigmas[:, 0:1]
    s_csf   = sigmas[:, 1:2]
    s_skull = sigmas[:, 2:3]
    s_scalp = sigmas[:, 3:4]

    s12 = s_brain / s_csf
    s23 = s_csf   / s_skull
    s34 = s_skull / s_scalp

    r12 = brain_rad / csf_rad
    r23 = csf_rad   / skull_rad
    r34 = skull_rad / scalp_rad
    r21 = 1.0 / r12
    r32 = 1.0 / r23
    r43 = 1.0 / r34

    n = n_vec.float().unsqueeze(0)   # (1, N)

    # V, Y, Z coefficients  (B, N)
    k34 = (n + 1.0) / n
    V_num = (r34**n - r43**(n+1)) / (k34 * r34**n + r43**(n+1))
    Vn = ((s34 / k34) - V_num) / (s34 + V_num)

    k23 = n / (n + 1.0)
    Y_fac = (r23**n * k23 - Vn * r32**(n+1)) / (r23**n + Vn * r32**(n+1))
    Yn = (s23 * k23 - Y_fac) / (s23 + Y_fac)

    k12 = (n + 1.0) / n
    Zn = (r12**n - k12 * Yn * r21**(n+1)) / (r12**n + Yn * r21**(n+1))

    return Vn, Yn, Zn, r12, r23, r34, r21, r32, r43, s12, s23, s34


def H_scalp(n_vec, rz_norm, sigmas, radii):
    """
    H_n evaluated at the scalp surface.
    rz_norm : (B,)   dipole distance from origin
    returns : (B, N)
    """
    brain_rad = radii[:, 0]   # (B,)
    scalp_rad = radii[:, 3]

    Vn, Yn, Zn, r12, r23, r34, r21, r32, r43, s12, s23, s34 = \
        compute_coefficients(n_vec, sigmas, radii)

    n    = n_vec.float().unsqueeze(0)   # (1, N)
    rz1  = (rz_norm / brain_rad).unsqueeze(1)  # (B,1)

    # A1
    A1 = rz1**(n+1) * (Zn + s12 * (n+1)/n) / (s12 - Zn)
    # A2, B2
    A2 = (A1 + rz1**(n+1)) / (Yn * r21**(n+1) + r12**n)
    B2 = A2 * Yn
    # A3, B3
    A3 = (A2 + B2) / (r23**n + Vn * r32**(n+1))
    B3 = A3 * Vn
    # A4, B4  at scalp
    k  = (n + 1.0) / n
    A4 = k * (A3 + B3) / (k * r34**n + r43**(n+1))
    B4 = A4 * (n / (n+1))

    # H at r_ele = scalp_rad
    r_ratio = (scalp_rad.unsqueeze(1) / scalp_rad.unsqueeze(1))  # =1, just for shape
    T1 = A4   # (r_ele/scalp_rad)^n = 1
    T2 = B4   # (scalp_rad/r_ele)^(n+1) = 1
    return T1 + T2   # (B, N)


# ---------------------------------------------------------------------------
# Radial potential  (B, E)
# ---------------------------------------------------------------------------

def radial_potential_torch(p_rad_mag, cos_theta_BE, Hn, rz_norm, sigmas):
    """
    p_rad_mag  : (B,)
    cos_theta_BE: (B, E)
    Hn         : (B, N)
    rz_norm    : (B,)
    sigmas     : (B, 4)
    """
    n_vec = torch.arange(1, Hn.shape[1]+1, dtype=Hn.dtype, device=Hn.device)
    # sum_n  n * H_n * P_n(cos_theta)
    # cos_theta_BE : (B, E)
    # need P_n for each electrode: shape (N, B, E)
    Pn_BEN = legendre_Pn(Hn.shape[1], cos_theta_BE)   # (N, B, E)
    # n * Hn : (B, N) -> broadcast -> (N, B, 1)
    nHn = (n_vec.unsqueeze(0) * Hn).permute(1, 0).unsqueeze(2)  # (N, B, 1) -- wrong, fix:
    nHn = (n_vec * Hn).unsqueeze(2)   # (B, N, 1)  -- still wrong
    # let's be explicit
    nHn = n_vec.unsqueeze(0) * Hn          # (B, N)
    # sum over n: einsum bn, nbe -> be
    V_rad = torch.einsum('bn,nbe->be', nHn, Pn_BEN)  # (B, E)

    sigma_brain = sigmas[:, 0]   # (B,)
    prefactor   = 1.0 / (4.0 * torch.pi * sigma_brain * rz_norm**2)  # (B,)
    V_rad = prefactor.unsqueeze(1) * p_rad_mag.unsqueeze(1) * V_rad
    return V_rad


# ---------------------------------------------------------------------------
# Tangential potential  (B, E)
# ---------------------------------------------------------------------------

def tangential_potential_torch(p_tan, r0, electrodes, Hn, rz_norm, sigmas):
    """
    p_tan      : (B, 3)
    r0         : (B, 3)
    electrodes : (E, 3)
    Hn         : (B, N)
    rz_norm    : (B,)
    sigmas     : (B, 4)
    """
    B, N = Hn.shape
    E    = electrodes.shape[0]

    p_tan_mag = torch.linalg.norm(p_tan, dim=1)   # (B,)

    # cos_theta between electrodes and r0
    r0_norm    = rz_norm.unsqueeze(1)              # (B,1)
    ele_norm   = torch.linalg.norm(electrodes, dim=1)  # (E,)
    cos_theta  = (electrodes.unsqueeze(0) @ r0.unsqueeze(2)).squeeze(2) / \
                 (ele_norm.unsqueeze(0) * r0_norm)  # (B, E)
    cos_theta  = torch.clamp(cos_theta, -1.0, 1.0)

    # phi angle
    proj = (torch.bmm(electrodes.unsqueeze(0).expand(B,-1,-1),
                      r0.unsqueeze(2)) / (r0_norm**2).unsqueeze(2)) \
           * r0.unsqueeze(1)                       # (B, E, 3)  -- projection of ele onto r0
    # actually:
    # proj_i = (ele_i . r0 / |r0|^2) * r0
    dots   = torch.einsum('ed,bd->be', electrodes, r0)  # (B,E)
    r0_sq  = (rz_norm**2).unsqueeze(1)                  # (B,1)
    proj   = (dots / r0_sq).unsqueeze(2) * r0.unsqueeze(1)  # (B,E,3)
    rxy    = electrodes.unsqueeze(0) - proj              # (B,E,3)

    x_axis = torch.linalg.cross(
        p_tan.unsqueeze(1).expand(-1, E, -1),
        r0.unsqueeze(1).expand(-1, E, -1)
    )  # (B,E,3)

    rxy_norm  = torch.linalg.norm(rxy,   dim=2, keepdim=True).clamp(min=1e-30)
    xax_norm  = torch.linalg.norm(x_axis,dim=2, keepdim=True).clamp(min=1e-30)
    cos_phi   = (rxy * x_axis).sum(dim=2) / (rxy_norm * xax_norm).squeeze(2)
    cos_phi   = torch.clamp(cos_phi, -1.0, 1.0)
    phi_temp  = torch.acos(cos_phi)
    range_t   = (rxy * p_tan.unsqueeze(1)).sum(dim=2)
    phi       = torch.where(range_t < 0, 2*torch.pi - phi_temp, phi_temp)  # (B,E)

    # sum_n H_n * P^1_n(cos_theta)
    P1n_BEN = legendre_P1n(N, cos_theta)     # (N, B, E)
    Lsum    = torch.einsum('bn,nbe->be', Hn, P1n_BEN)  # (B, E)

    sigma_brain = sigmas[:, 0]
    prefactor   = 1.0 / (4.0 * torch.pi * sigma_brain * rz_norm**2)  # (B,)
    V_tan = -prefactor.unsqueeze(1) * p_tan_mag.unsqueeze(1) * torch.sin(phi) * Lsum
    return V_tan


# ---------------------------------------------------------------------------
# Main batched forward
# ---------------------------------------------------------------------------

def forward_torch(p, r0, sigmas, radii, electrodes, n_terms=100):
    """
    p          : (B, 3)
    r0         : (B, 3)
    sigmas     : (B, 4)
    radii      : (B, 4)
    electrodes : (E, 3)  numpy array or tensor
    returns    : (B, E)  scalp potentials
    """
    if not isinstance(electrodes, torch.Tensor):
        electrodes = torch.tensor(electrodes, dtype=p.dtype, device=p.device)

    B = p.shape[0]
    n_vec   = torch.arange(1, n_terms+1, dtype=p.dtype, device=p.device)
    rz_norm = torch.linalg.norm(r0, dim=1)   # (B,)

    # H_n at scalp
    Hn = H_scalp(n_vec, rz_norm, sigmas, radii)  # (B, N)

    # Decompose dipole into radial and tangential
    rhat      = r0 / rz_norm.unsqueeze(1).clamp(min=1e-30)
    p_rad_mag = (p * rhat).sum(dim=1)            # (B,)
    p_rad     = p_rad_mag.unsqueeze(1) * rhat    # (B, 3)
    p_tan     = p - p_rad                        # (B, 3)

    # cos_theta for radial part
    ele_norm    = torch.linalg.norm(electrodes, dim=1)  # (E,)
    cos_theta   = torch.einsum('ed,bd->be', electrodes, r0) / \
                  (ele_norm.unsqueeze(0) * rz_norm.unsqueeze(1))  # (B,E)
    cos_theta   = torch.clamp(cos_theta, -1.0, 1.0)

    V_rad = radial_potential_torch(p_rad_mag, cos_theta, Hn, rz_norm, sigmas)
    V_tan = tangential_potential_torch(p_tan, r0, electrodes, Hn, rz_norm, sigmas)

    return V_rad + V_tan


# ---------------------------------------------------------------------------
# Quick test against numpy forward
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from forwardmodel import forward as forward_np, default_radii, default_sigmas, SCALP_ELECTRODES_64
    import parameters as params

    rng = np.random.default_rng(0)
    B   = 4

    r0_np  = np.array([rng.uniform(-1,1,3) * 0.5 * params.brain_rad for _ in range(B)])
    p_np   = np.array([rng.standard_normal(3) for _ in range(B)])
    for i in range(B):
        p_np[i] /= np.linalg.norm(p_np[i])

    radii_np  = np.tile(default_radii(),  (B,1))
    sigmas_np = np.tile(default_sigmas(), (B,1))

    # numpy reference
    V_ref = np.stack([
        forward_np(p_np[i], r0_np[i], radii_np[i], sigmas_np[i], n_terms=100)
        for i in range(B)
    ])

    # torch
    p_t      = torch.tensor(p_np,      dtype=torch.float64)
    r0_t     = torch.tensor(r0_np,     dtype=torch.float64)
    sigmas_t = torch.tensor(sigmas_np, dtype=torch.float64)
    radii_t  = torch.tensor(radii_np,  dtype=torch.float64)
    eles_t   = torch.tensor(SCALP_ELECTRODES_64, dtype=torch.float64)

    V_torch = forward_torch(p_t, r0_t, sigmas_t, radii_t, eles_t, n_terms=100)

    err = (V_torch.detach().numpy() - V_ref)
    print(f"Max abs error : {np.abs(err).max():.2e}")
    print(f"Mean abs error: {np.abs(err).mean():.2e}")
    print("PASSED" if np.abs(err).max() < 1e-6 else "FAILED — check implementation")